from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS review_status (
            review_id TEXT PRIMARY KEY,
            batch_id TEXT,
            status TEXT NOT NULL DEFAULT 'not_started',
            committed INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS system_decisions (
            prediction_id TEXT PRIMARY KEY,
            review_id TEXT NOT NULL,
            system_aspect TEXT NOT NULL,
            system_rating REAL,
            aspect_source TEXT,
            manual_decision TEXT,
            mapped_gold_aspect TEXT,
            manual_sentiment_decision TEXT,
            comment TEXT,
            source TEXT NOT NULL DEFAULT 'human',
            confirmed_by_human INTEGER NOT NULL DEFAULT 0,
            committed INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS gold_decisions (
            review_id TEXT NOT NULL,
            gold_aspect TEXT NOT NULL,
            gold_rating REAL,
            status TEXT,
            matched_system_prediction_id TEXT,
            comment TEXT,
            committed INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (review_id, gold_aspect)
        );

        CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()


def load_review_state(conn: sqlite3.Connection, review_id: str) -> dict[str, Any]:
    system_rows = conn.execute(
        "SELECT * FROM system_decisions WHERE review_id = ?",
        (review_id,),
    ).fetchall()
    gold_rows = conn.execute(
        "SELECT * FROM gold_decisions WHERE review_id = ?",
        (review_id,),
    ).fetchall()
    status_row = conn.execute(
        "SELECT * FROM review_status WHERE review_id = ?",
        (review_id,),
    ).fetchone()
    return {
        "system": {row["prediction_id"]: dict(row) for row in system_rows},
        "gold": {row["gold_aspect"]: dict(row) for row in gold_rows},
        "status": dict(status_row) if status_row else None,
    }


def upsert_system_decision(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO system_decisions (
            prediction_id, review_id, system_aspect, system_rating, aspect_source,
            manual_decision, mapped_gold_aspect, manual_sentiment_decision, comment,
            source, confirmed_by_human, committed, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(prediction_id) DO UPDATE SET
            review_id = excluded.review_id,
            system_aspect = excluded.system_aspect,
            system_rating = excluded.system_rating,
            aspect_source = excluded.aspect_source,
            manual_decision = excluded.manual_decision,
            mapped_gold_aspect = excluded.mapped_gold_aspect,
            manual_sentiment_decision = excluded.manual_sentiment_decision,
            comment = excluded.comment,
            source = excluded.source,
            confirmed_by_human = excluded.confirmed_by_human,
            committed = excluded.committed,
            updated_at = excluded.updated_at
        """,
        (
            payload["prediction_id"],
            payload["review_id"],
            payload["system_aspect"],
            payload.get("system_rating"),
            payload.get("aspect_source"),
            payload.get("manual_decision"),
            payload.get("mapped_gold_aspect"),
            payload.get("manual_sentiment_decision"),
            payload.get("comment"),
            payload.get("source", "human"),
            int(payload.get("confirmed_by_human", False)),
            int(payload.get("committed", False)),
            payload.get("updated_at", now_iso()),
        ),
    )


def upsert_gold_decision(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO gold_decisions (
            review_id, gold_aspect, gold_rating, status, matched_system_prediction_id,
            comment, committed, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(review_id, gold_aspect) DO UPDATE SET
            gold_rating = excluded.gold_rating,
            status = excluded.status,
            matched_system_prediction_id = excluded.matched_system_prediction_id,
            comment = excluded.comment,
            committed = excluded.committed,
            updated_at = excluded.updated_at
        """,
        (
            payload["review_id"],
            payload["gold_aspect"],
            payload.get("gold_rating"),
            payload.get("status"),
            payload.get("matched_system_prediction_id"),
            payload.get("comment"),
            int(payload.get("committed", False)),
            payload.get("updated_at", now_iso()),
        ),
    )


def upsert_review_status(conn: sqlite3.Connection, review_id: str, batch_id: str, status: str, committed: bool) -> None:
    conn.execute(
        """
        INSERT INTO review_status (review_id, batch_id, status, committed, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(review_id) DO UPDATE SET
            batch_id = excluded.batch_id,
            status = excluded.status,
            committed = excluded.committed,
            updated_at = excluded.updated_at
        """,
        (review_id, batch_id, status, int(committed), now_iso()),
    )


def clear_review(conn: sqlite3.Connection, review_id: str) -> None:
    conn.execute("DELETE FROM system_decisions WHERE review_id = ?", (review_id,))
    conn.execute("DELETE FROM gold_decisions WHERE review_id = ?", (review_id,))
    conn.execute("DELETE FROM review_status WHERE review_id = ?", (review_id,))
    conn.commit()


def commit_batch(conn: sqlite3.Connection, review_ids: list[str], batch_id: str) -> None:
    updated = now_iso()
    for review_id in review_ids:
        conn.execute(
            "UPDATE system_decisions SET committed = 1, confirmed_by_human = 1, updated_at = ? WHERE review_id = ?",
            (updated, review_id),
        )
        conn.execute(
            "UPDATE gold_decisions SET committed = 1, updated_at = ? WHERE review_id = ?",
            (updated, review_id),
        )
        upsert_review_status(conn, review_id, batch_id, "done", committed=True)
    conn.commit()


def load_overview_frames(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    system = pd.read_sql_query("SELECT * FROM system_decisions", conn)
    gold = pd.read_sql_query("SELECT * FROM gold_decisions", conn)
    status = pd.read_sql_query("SELECT * FROM review_status", conn)
    return system, gold, status


def load_app_meta(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    raw_value = str(row["value"])
    try:
        return json.loads(raw_value)
    except Exception:
        return raw_value


def save_app_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    conn.execute(
        """
        INSERT INTO app_meta (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value
        """,
        (key, serialized),
    )
    conn.commit()


def dirty_review_ids(conn: sqlite3.Connection, review_ids: list[str]) -> set[str]:
    if not review_ids:
        return set()
    placeholders = ",".join("?" for _ in review_ids)
    rows = conn.execute(
        f"SELECT review_id FROM review_status WHERE review_id IN ({placeholders}) AND committed = 0",
        tuple(review_ids),
    ).fetchall()
    return {str(row["review_id"]) for row in rows}


def build_batch_progress(
    batches: list[Any],
    status_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    normalized_batches: list[dict[str, Any]] = []
    for idx, batch in enumerate(batches):
        if isinstance(batch, dict):
            batch_id = str(batch.get("batch_id") or f"batch_{idx + 1:03d}")
            review_ids = [str(review_id) for review_id in batch.get("review_ids", [])]
        else:
            batch_id = f"batch_{idx + 1:03d}"
            review_ids = [str(review_id) for review_id in batch]
        normalized_batches.append({"batch_id": batch_id, "review_ids": review_ids})

    if status_df.empty:
        return [
            {
                "batch_id": batch["batch_id"],
                "total": len(batch["review_ids"]),
                "done": 0,
                "dirty": 0,
                "status": "new",
                "label": f"{batch['batch_id']} [new] ({len(batch['review_ids'])} reviews)",
            }
            for batch in normalized_batches
        ]

    normalized = status_df.copy()
    normalized["review_id"] = normalized["review_id"].astype(str)
    normalized["status"] = normalized["status"].fillna("not_started").astype(str)
    normalized["committed"] = normalized["committed"].fillna(0).astype(int)

    status_map = normalized.set_index("review_id")["status"].to_dict()
    committed_map = normalized.set_index("review_id")["committed"].to_dict()

    summaries: list[dict[str, Any]] = []
    for batch in normalized_batches:
        batch_id = batch["batch_id"]
        review_ids = batch["review_ids"]
        counts = Counter(status_map.get(review_id, "not_started") for review_id in review_ids)
        done = int(counts.get("done", 0))
        dirty = sum(1 for review_id in review_ids if review_id in committed_map and not bool(committed_map[review_id]))
        started = len(review_ids) - int(counts.get("not_started", 0))

        if done == len(review_ids) and review_ids:
            batch_status = "done"
            label = f"{batch_id} [done {done}/{len(review_ids)}] ({len(review_ids)} reviews)"
        elif started > 0 or dirty > 0:
            batch_status = "partial"
            label = f"{batch_id} [partial {done}/{len(review_ids)} | draft {dirty}] ({len(review_ids)} reviews)"
        else:
            batch_status = "new"
            label = f"{batch_id} [new] ({len(review_ids)} reviews)"

        summaries.append(
            {
                "batch_id": batch_id,
                "total": len(review_ids),
                "done": done,
                "dirty": dirty,
                "status": batch_status,
                "label": label,
            }
        )
    return summaries


def _build_prediction_resolver(review_lookup: dict[str, Any] | None) -> dict[tuple[str, str, str], str]:
    resolver: dict[tuple[str, str, str], str] = {}
    if not review_lookup:
        return resolver
    for review_id, review in review_lookup.items():
        for system in getattr(review, "system_aspects", []):
            key = (
                str(review_id),
                str(getattr(system, "aspect_name", "")).strip().casefold(),
                str(getattr(system, "aspect_source", "")).strip().casefold(),
            )
            resolver.setdefault(key, str(getattr(system, "prediction_id", "")))
            fallback_key = (
                str(review_id),
                str(getattr(system, "aspect_name", "")).strip().casefold(),
                "",
            )
            resolver.setdefault(fallback_key, str(getattr(system, "prediction_id", "")))
    return resolver


def import_ai_draft(conn: sqlite3.Connection, payload: dict[str, Any], review_lookup: dict[str, Any] | None = None) -> dict[str, Any]:
    resolver = _build_prediction_resolver(review_lookup)
    items = payload.get("items", [])
    imported_system = 0
    imported_gold = 0
    unresolved_system: list[dict[str, str]] = []
    for item in items:
        review_id = str(item.get("review_id", ""))
        if not review_id:
            continue
        for system_decision in item.get("system_aspect_decisions", []):
            prediction_id = str(system_decision.get("prediction_id", ""))
            if not prediction_id:
                prediction_id = resolver.get(
                    (
                        review_id,
                        str(system_decision.get("system_aspect", "")).strip().casefold(),
                        str(system_decision.get("aspect_source", "")).strip().casefold(),
                    ),
                    "",
                )
            if not prediction_id:
                unresolved_system.append(
                    {
                        "review_id": review_id,
                        "system_aspect": str(system_decision.get("system_aspect", "")),
                    }
                )
                continue
            upsert_system_decision(
                conn,
                {
                    "prediction_id": prediction_id,
                    "review_id": review_id,
                    "system_aspect": system_decision.get("system_aspect", ""),
                    "system_rating": system_decision.get("system_rating"),
                    "aspect_source": system_decision.get("aspect_source", ""),
                    "manual_decision": system_decision.get("manual_decision", ""),
                    "mapped_gold_aspect": system_decision.get("mapped_gold_aspect") or "NONE",
                    "manual_sentiment_decision": system_decision.get("manual_sentiment_decision", ""),
                    "comment": system_decision.get("comment", ""),
                    "source": "ai_prefill_draft",
                    "confirmed_by_human": False,
                    "committed": False,
                },
            )
            imported_system += 1
        for gold_decision in item.get("gold_aspect_decisions", []):
            gold_aspect = str(gold_decision.get("gold_aspect", ""))
            if not gold_aspect:
                continue
            matched_prediction_id = gold_decision.get("matched_system_prediction_id")
            if not matched_prediction_id:
                matched_prediction_id = resolver.get(
                    (
                        review_id,
                        str(gold_decision.get("matched_system_aspect", "")).strip().casefold(),
                        "",
                    ),
                    None,
                )
            upsert_gold_decision(
                conn,
                {
                    "review_id": review_id,
                    "gold_aspect": gold_aspect,
                    "gold_rating": gold_decision.get("gold_rating"),
                    "status": gold_decision.get("status", ""),
                    "matched_system_prediction_id": matched_prediction_id,
                    "comment": gold_decision.get("comment", ""),
                    "committed": False,
                },
            )
            imported_gold += 1
        upsert_review_status(
            conn,
            review_id=review_id,
            batch_id=str(payload.get("batch_id", "")),
            status="in_progress",
            committed=False,
        )
    conn.commit()
    return {
        "imported_system": imported_system,
        "imported_gold": imported_gold,
        "unresolved_system": unresolved_system,
        "review_count": len(items),
    }


def export_all(conn: sqlite3.Connection, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    system, gold, status = load_overview_frames(conn)
    outputs = {
        "system": out_dir / "manual_system_decisions.csv",
        "gold": out_dir / "manual_gold_decisions.csv",
        "status": out_dir / "manual_review_status.csv",
    }
    system.to_csv(outputs["system"], index=False, encoding="utf-8-sig")
    gold.to_csv(outputs["gold"], index=False, encoding="utf-8-sig")
    status.to_csv(outputs["status"], index=False, encoding="utf-8-sig")
    return outputs


def save_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        """
        INSERT INTO app_meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, json.dumps(value, ensure_ascii=False)),
    )
    conn.commit()
