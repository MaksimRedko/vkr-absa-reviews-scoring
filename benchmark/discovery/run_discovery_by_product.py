from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.discovery.run_discovery import (
    _build_excluded_rows,
    _load_hybrid_vocabulary,
    _parse_gold_labels,
    _report_to_json_dict,
    _resolve_repo_path,
    _row_to_review_input,
)
from configs.configs import config
from src.discovery import (
    ClusterAggregator,
    ClusterEvaluator,
    DiscoveryEncoder,
    DiscoveryReport,
    ResidualExtractor,
    ReviewClusterer,
    ReviewRepresentation,
    run_discovery,
)


def _safe_product_name(category_id: str, nm_id: int) -> str:
    return f"{category_id}_{int(nm_id)}"


def _render_product_markdown_summary(category_id: str, nm_id: int, report: DiscoveryReport) -> str:
    metadata = report.metadata
    evaluation = report.evaluation
    total_reviews = int(metadata.get("n_reviews_total", 0))
    reviews_with_residual = int(metadata.get("n_reviews_with_residual", 0))
    purity_threshold = float(config.discovery_runner.purity_threshold)

    lines = [
        f"# Discovery Product: {category_id} / {int(nm_id)}",
        "",
        f"Category: {category_id}",
        f"nm_id: {int(nm_id)}",
        f"Reviews: {total_reviews}",
        f"Reviews with residual phrases: {reviews_with_residual}",
        f"Clusters: {evaluation.n_clusters}",
        f"Clean clusters (purity >= {purity_threshold:.1f}): {evaluation.n_clean_clusters}",
        f"Coverage_via_clustering: {evaluation.coverage_via_clustering:.4f}",
        f"Noise_rate: {evaluation.noise_rate:.4f}",
        "",
    ]

    if not report.cluster_summaries:
        lines.append("No clusters found.")
        return "\n".join(lines) + "\n"

    for summary in report.cluster_summaries:
        purity = float(report.evaluation.purity_per_cluster.get(summary.cluster_id, 0.0))
        dominant = report.evaluation.dominant_aspect_per_cluster.get(summary.cluster_id, "") or "N/A"
        lines.append(
            f"## Cluster {summary.cluster_id} ({summary.n_reviews} reviews, "
            f"purity={purity:.2f}, dominant={dominant})"
        )
        lines.append("Top phrases:")
        for phrase, count in summary.top_phrases:
            lines.append(f'- "{phrase}" ({count})')
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_product_metrics_row(
    category_id: str,
    nm_id: int,
    report: DiscoveryReport,
) -> dict[str, Any]:
    metadata = report.metadata
    evaluation = report.evaluation
    hdbscan_params = dict(metadata.get("hdbscan", {}))

    return {
        "category_id": category_id,
        "nm_id": int(nm_id),
        "n_reviews_total": int(metadata.get("n_reviews_total", 0)),
        "n_reviews_with_residual": int(metadata.get("n_reviews_with_residual", 0)),
        "n_clusters": int(evaluation.n_clusters),
        "n_clean_clusters": int(evaluation.n_clean_clusters),
        "coverage_via_clustering": float(evaluation.coverage_via_clustering),
        "noise_rate": float(evaluation.noise_rate),
        "n_excluded_reviews": int(evaluation.n_excluded_reviews),
        "model_name": str(metadata.get("model_name", "")),
        "generated_at": str(metadata.get("generated_at", "")),
        "purity_threshold": float(config.discovery_runner.purity_threshold),
        "top_n_phrases_per_cluster": int(config.discovery_runner.top_n_phrases_per_cluster),
        "min_cluster_size": int(hdbscan_params.get("min_cluster_size", 0)),
        "min_samples": int(hdbscan_params.get("min_samples", 0)),
        "metric": str(hdbscan_params.get("metric", "")),
        "cluster_selection_method": str(hdbscan_params.get("cluster_selection_method", "")),
    }


def _build_category_metrics_rows(product_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not product_rows:
        return []

    rows: list[dict[str, Any]] = []
    df = pd.DataFrame(product_rows)
    for category_id, group in df.groupby("category_id", sort=True):
        total_reviews = int(group["n_reviews_total"].sum())
        total_with_residual = int(group["n_reviews_with_residual"].sum())
        weighted_coverage = (
            float((group["coverage_via_clustering"] * group["n_reviews_total"]).sum())
            / float(total_reviews)
            if total_reviews
            else 0.0
        )
        weighted_noise = (
            float((group["noise_rate"] * group["n_reviews_with_residual"]).sum())
            / float(total_with_residual)
            if total_with_residual
            else 0.0
        )
        rows.append(
            {
                "category_id": str(category_id),
                "n_products": int(len(group)),
                "n_reviews_total": total_reviews,
                "n_reviews_with_residual": total_with_residual,
                "n_clusters": int(group["n_clusters"].sum()),
                "n_clean_clusters": int(group["n_clean_clusters"].sum()),
                "coverage_via_clustering_weighted": weighted_coverage,
                "noise_rate_weighted": weighted_noise,
                "n_excluded_reviews": int(group["n_excluded_reviews"].sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run discovery pipeline grouped by product.")
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=str(config.discovery_runner.dataset_csv),
        help="Path to discovery dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.discovery_runner.results_dir),
        help="Base output directory for discovery artifacts.",
    )
    args = parser.parse_args()

    dataset_csv = _resolve_repo_path(args.dataset_csv)
    output_root = _resolve_repo_path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{timestamp}_by_product"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    category_column = "category_id" if "category_id" in df.columns else "category"
    if category_column not in df.columns:
        raise ValueError("Dataset must contain `category_id` or `category` column.")
    if "nm_id" not in df.columns:
        raise ValueError("Dataset must contain `nm_id` column.")

    encoder = DiscoveryEncoder(
        model_name_or_path=str(config.discovery_runner.encoder_model),
        batch_size=int(config.discovery_runner.encoder_batch_size),
    )
    residual_extractor = ResidualExtractor()
    representation_builder = ReviewRepresentation()
    clusterer = ReviewClusterer(
        min_cluster_size=int(config.discovery_runner.hdbscan.min_cluster_size),
        min_samples=int(config.discovery_runner.hdbscan.min_samples),
        metric=str(config.discovery_runner.hdbscan.metric),
        cluster_selection_method=str(config.discovery_runner.hdbscan.cluster_selection_method),
    )
    aggregator = ClusterAggregator(
        top_k_phrases=int(config.discovery_runner.top_n_phrases_per_cluster),
    )
    evaluator = ClusterEvaluator(
        purity_threshold=float(config.discovery_runner.purity_threshold),
    )

    product_metrics_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for category_id in list(config.discovery_runner.categories):
        category_df = df[df[category_column].astype(str) == str(category_id)].copy()
        vocabulary = _load_hybrid_vocabulary(str(category_id))

        for nm_id, product_df in category_df.groupby("nm_id", sort=True):
            product_df = product_df.copy()
            reviews = [_row_to_review_input(row) for _, row in product_df.iterrows()]
            gold_labels = {
                str(row["id"]): _parse_gold_labels(row.get("true_labels"))
                for _, row in product_df.iterrows()
            }

            report = run_discovery(
                category_id=str(category_id),
                reviews=reviews,
                gold_labels=gold_labels,
                vocabulary=vocabulary,
                encoder=encoder,
                residual_extractor=residual_extractor,
                representation_builder=representation_builder,
                clusterer=clusterer,
                aggregator=aggregator,
                evaluator=evaluator,
            )
            report.metadata["nm_id"] = int(nm_id)
            report.metadata["run_unit"] = "product"

            product_name = _safe_product_name(str(category_id), int(nm_id))
            (output_dir / f"discovery_report_{product_name}.json").write_text(
                json.dumps(_report_to_json_dict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output_dir / f"cluster_summary_{product_name}.md").write_text(
                _render_product_markdown_summary(str(category_id), int(nm_id), report),
                encoding="utf-8",
            )

            product_metrics_rows.append(
                _build_product_metrics_row(str(category_id), int(nm_id), report)
            )
            excluded_rows.extend(_build_excluded_rows(str(category_id), product_df, report))

    pd.DataFrame(product_metrics_rows).to_csv(
        output_dir / "metrics_summary_by_product.csv",
        index=False,
    )
    pd.DataFrame(_build_category_metrics_rows(product_metrics_rows)).to_csv(
        output_dir / "metrics_summary_by_category.csv",
        index=False,
    )
    pd.DataFrame(
        excluded_rows,
        columns=["category_id", "review_id", "nm_id", "rating", "source", "full_text"],
    ).to_csv(output_dir / "excluded_reviews_by_product.csv", index=False)

    print(f"[discovery-by-product] Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
