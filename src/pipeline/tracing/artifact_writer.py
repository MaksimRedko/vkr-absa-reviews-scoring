from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sanitize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return sanitize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class ArtifactWriter:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def write_dataframe(
        self,
        relative_path: str | Path,
        frame: pd.DataFrame,
        *,
        sort_by: list[str] | None = None,
    ) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        if sort_by:
            existing = [col for col in sort_by if col in out.columns]
            if existing:
                out = out.sort_values(existing, kind="mergesort").reset_index(drop=True)
        out.to_parquet(path, index=False)
        return path

    def write_csv(
        self,
        relative_path: str | Path,
        frame: pd.DataFrame,
        *,
        sort_by: list[str] | None = None,
    ) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        if sort_by:
            existing = [col for col in sort_by if col in out.columns]
            if existing:
                out = out.sort_values(existing, kind="mergesort").reset_index(drop=True)
        out.to_csv(path, index=False, encoding="utf-8")
        return path

    def write_npy(self, relative_path: str | Path, array: np.ndarray) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        return path
