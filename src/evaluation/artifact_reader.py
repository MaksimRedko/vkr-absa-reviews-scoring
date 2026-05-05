from __future__ import annotations

from pathlib import Path

from src.pipeline.tracing import ArtifactReader


def open_run(run_dir: str | Path) -> ArtifactReader:
    return ArtifactReader(run_dir)
