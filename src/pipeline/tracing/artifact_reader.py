from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ArtifactReader:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def path(self, relative_path: str | Path) -> Path:
        return self.run_dir / relative_path

    def exists(self, relative_path: str | Path) -> bool:
        return self.path(relative_path).exists()

    def read_json(self, relative_path: str | Path) -> dict[str, Any]:
        return json.loads(self.path(relative_path).read_text(encoding="utf-8"))

    def read_dataframe(
        self,
        relative_path: str | Path,
        *,
        sort_by: list[str] | None = None,
    ) -> pd.DataFrame:
        path = self.path(relative_path)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_parquet(path)
        if sort_by:
            existing = [col for col in sort_by if col in frame.columns]
            if existing:
                frame = frame.sort_values(existing, kind="mergesort").reset_index(drop=True)
        return frame

    def read_npy(self, relative_path: str | Path) -> np.ndarray:
        return np.load(self.path(relative_path))

    def is_traced_run(self) -> bool:
        return self.exists("MANIFEST.json")

    def is_old_e2e_run(self) -> bool:
        return self.exists("run_summary.json")
