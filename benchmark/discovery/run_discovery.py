from __future__ import annotations

import argparse
import ast
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
from src.schemas.models import ReviewInput
from src.vocabulary.loader import AspectDefinition, Vocabulary


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (_ROOT / path).resolve()


def _parse_gold_labels(raw_value: object) -> dict[str, float]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return {}
    if isinstance(raw_value, dict):
        return {str(key): float(value) for key, value in raw_value.items()}

    text = str(raw_value).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return {str(key): float(value) for key, value in parsed.items()}

    return {}


def _load_hybrid_vocabulary(category_id: str) -> Vocabulary:
    core_path = _ROOT / "src" / "vocabulary" / "universal_aspects_v1.yaml"
    domain_path = _ROOT / "src" / "vocabulary" / "domain" / f"{category_id}.yaml"

    core_vocab = Vocabulary.load_from_yaml(core_path)
    domain_vocab = Vocabulary.load_from_yaml(domain_path)

    merged_aspects: list[AspectDefinition] = []
    seen_ids: set[str] = set()
    for aspect in core_vocab.aspects + domain_vocab.aspects:
        if aspect.id in seen_ids:
            continue
        merged_aspects.append(aspect)
        seen_ids.add(aspect.id)

    return Vocabulary(
        merged_aspects,
        _by_id={aspect.id: aspect for aspect in merged_aspects},
        _by_canonical={aspect.canonical_name: aspect for aspect in merged_aspects},
    )


def _row_to_review_input(row: pd.Series) -> ReviewInput:
    return ReviewInput(
        id=str(row["id"]),
        nm_id=int(row["nm_id"]),
        rating=int(row["rating"]),
        full_text=str(row.get("full_text") or ""),
    )


def _report_to_json_dict(report: DiscoveryReport) -> dict[str, Any]:
    return asdict(report)


def _render_markdown_summary(category_id: str, report: DiscoveryReport) -> str:
    metadata = report.metadata
    evaluation = report.evaluation
    total_reviews = int(metadata.get("n_reviews_total", 0))
    reviews_with_residual = int(metadata.get("n_reviews_with_residual", 0))
    purity_threshold = float(config.discovery_runner.purity_threshold)

    lines = [
        f"# Discovery: {category_id}",
        "",
        f"Категория: {category_id}",
        f"Отзывов: {total_reviews}",
        f"Из них с непокрытыми фразами: {reviews_with_residual}",
        f"Найдено кластеров: {evaluation.n_clusters}",
        f"Чистых (purity ≥ {purity_threshold:.1f}): {evaluation.n_clean_clusters}",
        f"Coverage_via_clustering: {evaluation.coverage_via_clustering:.2f}",
        "",
    ]

    if not report.cluster_summaries:
        lines.append("Кластеры не найдены.")
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


def _build_metrics_row(category_id: str, report: DiscoveryReport) -> dict[str, Any]:
    metadata = report.metadata
    evaluation = report.evaluation
    hdbscan_params = dict(metadata.get("hdbscan", {}))

    return {
        "category_id": category_id,
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


def _build_excluded_rows(
    category_id: str,
    category_df: pd.DataFrame,
    report: DiscoveryReport,
) -> list[dict[str, Any]]:
    excluded_review_ids = {
        str(review_id)
        for review_id in report.metadata.get("excluded_review_ids", [])
    }
    if not excluded_review_ids:
        return []

    rows: list[dict[str, Any]] = []
    for _, row in category_df.iterrows():
        review_id = str(row["id"])
        if review_id not in excluded_review_ids:
            continue
        rows.append(
            {
                "category_id": category_id,
                "review_id": review_id,
                "nm_id": int(row["nm_id"]),
                "rating": int(row["rating"]),
                "source": str(row.get("source") or ""),
                "full_text": str(row.get("full_text") or ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run discovery pipeline for all benchmark categories.")
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
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    category_column = "category_id" if "category_id" in df.columns else "category"
    if category_column not in df.columns:
        raise ValueError("Dataset must contain `category_id` or `category` column.")

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

    metrics_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for category_id in list(config.discovery_runner.categories):
        category_df = df[df[category_column].astype(str) == str(category_id)].copy()
        reviews = [_row_to_review_input(row) for _, row in category_df.iterrows()]
        gold_labels = {
            str(row["id"]): _parse_gold_labels(row.get("true_labels"))
            for _, row in category_df.iterrows()
        }
        vocabulary = _load_hybrid_vocabulary(str(category_id))

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

        (output_dir / f"discovery_report_{category_id}.json").write_text(
            json.dumps(_report_to_json_dict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"cluster_summary_{category_id}.md").write_text(
            _render_markdown_summary(str(category_id), report),
            encoding="utf-8",
        )

        metrics_rows.append(_build_metrics_row(str(category_id), report))
        excluded_rows.extend(_build_excluded_rows(str(category_id), category_df, report))

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(
        excluded_rows,
        columns=["category_id", "review_id", "nm_id", "rating", "source", "full_text"],
    ).to_csv(output_dir / "excluded_reviews.csv", index=False)

    print(f"[discovery] Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
