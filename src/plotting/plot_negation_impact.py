from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline.tracing.artifact_reader import ArtifactReader
from src.plotting.common import figures_dir


def plot(run_dir: str | Path) -> Path:
    summary = ArtifactReader(run_dir).read_json("run_summary.json")
    neg = summary.get("negation_correction", {})
    data = pd.DataFrame(
        [
            {"metric": "before", "mae": neg.get("avg_mae_before_correction")},
            {"metric": "after", "mae": neg.get("avg_mae_after_correction")},
        ]
    )
    out = figures_dir(run_dir) / "negation_impact.png"
    ax = data.plot(kind="bar", x="metric", y="mae", legend=False, title="Negation Correction Impact")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=160)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    print(plot(parser.parse_args().run_dir))
