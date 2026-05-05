from __future__ import annotations

from pathlib import Path


def figures_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
