from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.plotting.common import figures_dir


def plot(run_dir: str | Path) -> Path:
    df = pd.read_csv(Path(run_dir) / "metrics_track_a_vocab_only.csv")
    out = figures_dir(run_dir) / "coverage_breakdown.png"
    ax = df.groupby("category_id")["detection_recall"].mean().plot(kind="bar", title="Detection Recall By Category")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=160)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    print(plot(parser.parse_args().run_dir))
