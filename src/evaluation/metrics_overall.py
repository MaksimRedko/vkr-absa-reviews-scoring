from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics_aggregation import aggregation_from_summary
from src.evaluation.metrics_detection import detection_from_summary
from src.evaluation.metrics_discovery import discovery_from_summary
from src.evaluation.metrics_sentiment import sentiment_from_summary
from src.pipeline.tracing import sanitize_for_json


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "run_summary.json"
    if not path.exists():
        manifest = Path(run_dir) / "MANIFEST.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            return {
                "track_a": metrics.get("track_a", {}),
                "track_b": metrics.get("track_b", {}),
                "track_c_review": metrics.get("track_c_review", {}),
                "track_c_product": metrics.get("track_c_product", {}),
                "negation_correction": metrics.get("negation_correction", {}),
            }
        raise FileNotFoundError(f"No run_summary.json or MANIFEST.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_all_metrics(run_dir: str | Path) -> dict[str, float]:
    summary = load_run_summary(run_dir)
    out: dict[str, float] = {}
    out.update(detection_from_summary(summary))
    out.update(sentiment_from_summary(summary))
    out.update(aggregation_from_summary(summary))
    out.update(discovery_from_summary(summary))
    return out


def write_metrics_report(run_dir: str | Path) -> dict[str, float]:
    run_path = Path(run_dir)
    metrics = compute_all_metrics(run_path)
    (run_path / "metrics_summary.json").write_text(
        json.dumps(sanitize_for_json(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = ["# Metrics Summary", ""]
    for key, value in sorted(metrics.items()):
        lines.append(f"- `{key}`: {value:.4f}")
    (run_path / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    per_product_path = run_path / "metrics_track_a_vocab_only.csv"
    if per_product_path.exists():
        per_product = pd.read_csv(per_product_path)
        per_product.to_csv(run_path / "per_product_metrics.csv", index=False, encoding="utf-8")
        if "category_id" in per_product.columns:
            per_category = per_product.groupby("category_id", as_index=False).mean(numeric_only=True)
            per_category.to_csv(run_path / "per_category_metrics.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(run_path / "per_product_metrics.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(run_path / "per_category_metrics.csv", index=False, encoding="utf-8")

    labels = [1, 2, 3, 4, 5]
    pd.DataFrame(0, index=labels, columns=labels).to_csv(run_path / "confusion_matrix_5x5.csv", encoding="utf-8")
    return metrics
