from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def build_model_signature(
    *,
    backend: str,
    model_path: str,
    tokenizer_path: str,
    id2label: dict[int | str, str],
    num_labels: int,
) -> str:
    payload = {
        "backend": str(backend),
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "id2label": {str(key): str(value) for key, value in sorted(id2label.items(), key=lambda item: str(item[0]))},
        "num_labels": int(num_labels),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(slots=True, frozen=True)
class CachedPair:
    premise: str
    hypothesis: str


@dataclass(slots=True)
class CacheStats:
    memory_hits: int = 0
    persistent_hits: int = 0
    misses: int = 0
    writes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "memory_hits": int(self.memory_hits),
            "persistent_hits": int(self.persistent_hits),
            "misses": int(self.misses),
            "writes": int(self.writes),
        }


class PersistentNliCache:
    def __init__(
        self,
        *,
        path: str | Path,
        model_signature: str,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_signature = str(model_signature)
        self.model_signature_hash = sha256_text(self.model_signature)
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=60.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._init_schema()

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nli_model_signatures (
                signature_hash TEXT PRIMARY KEY,
                signature_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS nli_texts (
                text_hash TEXT PRIMARY KEY,
                text_kind TEXT NOT NULL,
                text_value TEXT NOT NULL,
                char_len INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS nli_cache_entries (
                pair_hash TEXT PRIMARY KEY,
                signature_hash TEXT NOT NULL,
                premise_hash TEXT NOT NULL,
                hypothesis_hash TEXT NOT NULL,
                logits_json TEXT NOT NULL,
                num_labels INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT,
                hit_count INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(signature_hash) REFERENCES nli_model_signatures(signature_hash),
                FOREIGN KEY(premise_hash) REFERENCES nli_texts(text_hash),
                FOREIGN KEY(hypothesis_hash) REFERENCES nli_texts(text_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_nli_cache_signature
            ON nli_cache_entries(signature_hash);

            CREATE INDEX IF NOT EXISTS idx_nli_cache_premise
            ON nli_cache_entries(premise_hash);

            CREATE INDEX IF NOT EXISTS idx_nli_cache_hypothesis
            ON nli_cache_entries(hypothesis_hash);
            """
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO nli_model_signatures(signature_hash, signature_json, created_at)
            VALUES (?, ?, ?)
            """,
            (self.model_signature_hash, self.model_signature, _utc_now()),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _pair_hash(self, premise_hash: str, hypothesis_hash: str) -> str:
        return sha256_text(f"{self.model_signature_hash}|{premise_hash}|{hypothesis_hash}")

    def lookup_many(self, pairs: list[CachedPair]) -> dict[int, np.ndarray]:
        if not self.enabled or not pairs or self._conn is None:
            return {}
        pair_meta = []
        pair_hashes = []
        for index, pair in enumerate(pairs):
            premise_hash = sha256_text(pair.premise)
            hypothesis_hash = sha256_text(pair.hypothesis)
            pair_hash = self._pair_hash(premise_hash, hypothesis_hash)
            pair_meta.append((index, pair_hash))
            pair_hashes.append(pair_hash)

        found: dict[str, np.ndarray] = {}
        chunk_size = 200
        for start in range(0, len(pair_hashes), chunk_size):
            chunk = pair_hashes[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"""
                SELECT pair_hash, logits_json
                FROM nli_cache_entries
                WHERE pair_hash IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for pair_hash, logits_json in rows:
                found[str(pair_hash)] = np.asarray(json.loads(str(logits_json)), dtype=np.float32)

        now = _utc_now()
        hit_hashes = [pair_hash for _, pair_hash in pair_meta if pair_hash in found]
        for start in range(0, len(hit_hashes), chunk_size):
            chunk = hit_hashes[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            self._conn.execute(
                f"""
                UPDATE nli_cache_entries
                SET hit_count = hit_count + 1,
                    last_accessed_at = ?
                WHERE pair_hash IN ({placeholders})
                """,
                [now, *chunk],
            )
        if hit_hashes:
            self._conn.commit()

        out: dict[int, np.ndarray] = {}
        for index, pair_hash in pair_meta:
            logits = found.get(pair_hash)
            if logits is not None:
                out[index] = logits
        return out

    def store_many(
        self,
        pairs: list[CachedPair],
        logits_matrix: np.ndarray,
    ) -> int:
        if not self.enabled or not pairs or self._conn is None:
            return 0
        if len(pairs) != int(logits_matrix.shape[0]):
            raise ValueError("pairs/logits length mismatch")

        now = _utc_now()
        text_rows: list[tuple[str, str, str, int, str]] = []
        entry_rows: list[tuple[str, str, str, str, str, int, str]] = []
        for pair, logits in zip(pairs, logits_matrix, strict=True):
            premise_hash = sha256_text(pair.premise)
            hypothesis_hash = sha256_text(pair.hypothesis)
            pair_hash = self._pair_hash(premise_hash, hypothesis_hash)
            text_rows.append((premise_hash, "premise", str(pair.premise), len(str(pair.premise)), now))
            text_rows.append((hypothesis_hash, "hypothesis", str(pair.hypothesis), len(str(pair.hypothesis)), now))
            entry_rows.append(
                (
                    pair_hash,
                    self.model_signature_hash,
                    premise_hash,
                    hypothesis_hash,
                    json.dumps(np.asarray(logits, dtype=np.float32).tolist(), ensure_ascii=False, separators=(",", ":")),
                    int(len(logits)),
                    now,
                )
            )

        self._conn.executemany(
            """
            INSERT OR IGNORE INTO nli_texts(text_hash, text_kind, text_value, char_len, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            text_rows,
        )
        before = self._conn.total_changes
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO nli_cache_entries(
                pair_hash,
                signature_hash,
                premise_hash,
                hypothesis_hash,
                logits_json,
                num_labels,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            entry_rows,
        )
        written = int(self._conn.total_changes - before)
        self._conn.commit()
        return written

    def count_rows(self) -> int:
        if not self.enabled or self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM nli_cache_entries").fetchone()
        return int(row[0]) if row else 0

    def count_text_rows(self) -> int:
        if not self.enabled or self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM nli_texts").fetchone()
        return int(row[0]) if row else 0


def cached_pairs_from_strings(
    premises: Iterable[str],
    hypotheses: Iterable[str],
) -> list[CachedPair]:
    return [
        CachedPair(premise=str(premise), hypothesis=str(hypothesis))
        for premise, hypothesis in zip(premises, hypotheses, strict=True)
    ]
