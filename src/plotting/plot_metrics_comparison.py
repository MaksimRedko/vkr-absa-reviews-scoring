from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.metrics_overall import compute_all_metrics
from src.plotting.common import figures_dir


def plot(run_dir: str | Path) -> Path:
    metrics = compute_all_metrics(run_dir)
    data = pd.DataFrame(
        [
            {"metric": "Track A F1", "value": metrics["detection_f1_track_a"]},
            {"metric": "Review MAE", "value": metrics["sentiment_mae_review"]},
            {"metric": "Product MAE n3", "value": metrics["product_mae_n3"]},
        ]
    )
    out = figures_dir(run_dir) / "metrics_comparison.png"
    ax = data.plot(kind="bar", x="metric", y="value", legend=False, title="Metrics Comparison")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=160)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    print(plot(parser.parse_args().run_dir))
