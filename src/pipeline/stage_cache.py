from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.tracing import sanitize_for_json


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_jsonable(payload: Any) -> str:
    normalized = sanitize_for_json(payload)
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hash_bytes(raw)


def hash_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    for file_path in files:
        rel = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(hash_file(file_path).encode("utf-8"))
    return digest.hexdigest()


class StageCacheManager:
    def __init__(self, *, root_dir: str | Path, enabled: bool) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def fingerprint(self, payload: Any) -> str:
        return hash_jsonable(payload)

    def entry_dir(self, stage_name: str, fingerprint: str) -> Path:
        return self.root_dir / stage_name / fingerprint

    def metadata_path(self, stage_name: str, fingerprint: str) -> Path:
        return self.entry_dir(stage_name, fingerprint) / "stage_meta.json"

    def load_metadata(self, stage_name: str, fingerprint: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        meta_path = self.metadata_path(stage_name, fingerprint)
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def has(self, stage_name: str, fingerprint: str) -> bool:
        meta = self.load_metadata(stage_name, fingerprint)
        if not meta:
            return False
        entry_dir = self.entry_dir(stage_name, fingerprint)
        for rel_path in meta.get("artifact_files", []):
            if not (entry_dir / rel_path).exists():
                return False
        return True

    def restore_to_run_dir(self, stage_name: str, fingerprint: str, run_dir: str | Path) -> dict[str, Any] | None:
        meta = self.load_metadata(stage_name, fingerprint)
        if not meta:
            return None
        entry_dir = self.entry_dir(stage_name, fingerprint)
        run_path = Path(run_dir)
        for rel_path in meta.get("artifact_files", []):
            src = entry_dir / rel_path
            dst = run_path / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return meta

    def store_from_run_dir(
        self,
        stage_name: str,
        fingerprint: str,
        run_dir: str | Path,
        artifact_files: list[str],
        *,
        inputs: Any | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        entry_dir = self.entry_dir(stage_name, fingerprint)
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        entry_dir.mkdir(parents=True, exist_ok=True)
        run_path = Path(run_dir)
        for rel_path in artifact_files:
            src = run_path / rel_path
            dst = entry_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        meta = {
            "stage_name": stage_name,
            "fingerprint": fingerprint,
            "artifact_files": list(artifact_files),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": sanitize_for_json(inputs),
        }
        self.metadata_path(stage_name, fingerprint).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return meta
