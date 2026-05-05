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

from benchmark.discovery.run_discovery import (  # noqa: E402
    _load_hybrid_vocabulary,
    _parse_gold_labels,
    _resolve_repo_path,
    _row_to_review_input,
)
from src.discovery.config_v3 import (  # noqa: E402
    ENCODER_BATCH_SIZE,
    ENCODER_MODEL,
    HDBSCAN_PARAMS,
    MIN_UNIQUE_PHRASES_TO_CLUSTER,
    TOP_N_PHRASES_PER_CLUSTER,
)
from src.discovery.encoder import DiscoveryEncoder  # noqa: E402
from src.discovery.manual_eval import prepare_manual_labels_template  # noqa: E402
from src.discovery.per_product_pipeline_v3 import (  # noqa: E402
    PerProductDiscoveryV3,
    ProductDiscoveryReport,
)
from src.discovery.residual_extractor import ResidualExtractor  # noqa: E402
from src.discovery.snapshot_cache import (  # noqa: E402
    DiscoverySnapshotCache,
    compute_product_snapshot_key,
)


def _json_default(obj: object) -> object:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _build_metrics_row(report: ProductDiscoveryReport, config_name: str) -> dict[str, Any]:
    sensitivity = report.semantic_metrics.sensitivity
    return {
        "nm_id": report.nm_id,
        "category_id": report.category_id,
        "config": config_name,
        "skipped": bool(report.metadata.get("skipped", False)),
        "n_reviews": report.n_reviews,
        "n_unique_phrases": report.n_unique_residuals_after_filter,
        "n_unique_phrases_before_filter": report.n_unique_residuals_before_filter,
        "n_clusters": report.n_clusters,
        "noise_rate": report.noise_rate,
        "cohesion": report.intrinsic_metrics.cohesion,
        "separation": report.intrinsic_metrics.separation,
        "silhouette": report.intrinsic_metrics.silhouette,
        "avg_concentration": report.intrinsic_metrics.avg_concentration,
        "coverage_at_0.60": sensitivity[0.60].coverage,
        "coverage_at_0.65": sensitivity[0.65].coverage,
        "coverage_at_0.70": sensitivity[0.70].coverage,
        "avg_soft_purity": report.semantic_metrics.avg_soft_purity,
        "n_novel_clusters": report.semantic_metrics.n_novel_clusters,
        "filter_rate": report.filter_report.filter_rate if report.filter_report else 0.0,
    }


def _render_product_markdown(
    *,
    no_filter: ProductDiscoveryReport,
    filtered: ProductDiscoveryReport,
) -> str:
    lines = [
        f"# Discovery v3 comparison: nm_id {filtered.nm_id}",
        "",
        f"Категория: {filtered.category_id}",
        f"Отзывов: {filtered.n_reviews}",
        "",
        "| Metric | No filter | Filtered |",
        "|---|---:|---:|",
    ]
    pairs = [
        ("unique phrases", no_filter.n_unique_residuals_after_filter, filtered.n_unique_residuals_after_filter),
        ("clusters", no_filter.n_clusters, filtered.n_clusters),
        ("noise_rate", no_filter.noise_rate, filtered.noise_rate),
        ("cohesion", no_filter.intrinsic_metrics.cohesion, filtered.intrinsic_metrics.cohesion),
        ("separation", no_filter.intrinsic_metrics.separation, filtered.intrinsic_metrics.separation),
        ("silhouette", no_filter.intrinsic_metrics.silhouette, filtered.intrinsic_metrics.silhouette),
        ("coverage@0.65", no_filter.semantic_metrics.coverage_primary.coverage, filtered.semantic_metrics.coverage_primary.coverage),
        ("soft_purity", no_filter.semantic_metrics.avg_soft_purity, filtered.semantic_metrics.avg_soft_purity),
        ("novel clusters", no_filter.semantic_metrics.n_novel_clusters, filtered.semantic_metrics.n_novel_clusters),
    ]
    for name, left, right in pairs:
        lines.append(f"| {name} | {_fmt(left)} | {_fmt(right)} |")
    lines.extend(["", "## Filtered clusters", ""])
    if bool(filtered.metadata.get("skipped", False)):
        lines.append("Skipped: меньше 30 уникальных фраз.")
    elif not filtered.cluster_summaries:
        lines.append("Кластеры не найдены.")
    for summary in filtered.cluster_summaries:
        lines.append(f"### Cluster {summary.cluster_id} ({summary.n_phrases} phrases, weight {summary.total_weight})")
        for phrase, weight in summary.top_phrases:
            lines.append(f'- "{phrase}" ({weight})')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_ab_summary(rows: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(rows)
    no_filter = df[df["config"] == "no_filter"]
    filtered = df[df["config"] == "filtered"]
    metrics = [
        ("Avg unique phrases", "n_unique_phrases", "lower"),
        ("Avg n_clusters", "n_clusters", "higher"),
        ("Avg noise_rate", "noise_rate", "lower"),
        ("Avg cohesion", "cohesion", "higher"),
        ("Avg separation", "separation", "lower"),
        ("Avg silhouette", "silhouette", "higher"),
        ("Coverage@0.65", "coverage_at_0.65", "higher"),
        ("Avg soft_purity", "avg_soft_purity", "higher"),
        ("Novel aspects total", "n_novel_clusters", "higher_sum"),
    ]
    verdict_metrics = {
        "noise_rate": "lower",
        "cohesion": "higher",
        "separation": "lower",
        "silhouette": "higher",
        "avg_concentration": "higher",
        "coverage_at_0.65": "higher",
        "avg_soft_purity": "higher",
    }

    lines = [
        "# A/B: v2 (no filter) vs v3 (filtered)",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | No Filter | Filtered | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, column, direction in metrics:
        left = float(no_filter[column].sum()) if direction == "higher_sum" else float(no_filter[column].mean())
        right = float(filtered[column].sum()) if direction == "higher_sum" else float(filtered[column].mean())
        lines.append(f"| {label} | {left:.4f} | {right:.4f} | {right - left:+.4f} |")

    better = 0
    worse = 0
    for column, direction in verdict_metrics.items():
        left = float(no_filter[column].mean())
        right = float(filtered[column].mean())
        if direction == "lower":
            better += int(right < left)
            worse += int(right > left)
        else:
            better += int(right > left)
            worse += int(right < left)

    if better >= 4:
        verdict = "ФИЛЬТР ПОЛЕЗЕН, использовать"
    elif worse >= 4:
        verdict = "ФИЛЬТР ВРЕДИТ, НЕ использовать"
    else:
        verdict = "ФИЛЬТР НЕЙТРАЛЕН, решение за человеком"

    lines.extend(["", "## Per-product winners", ""])
    lines.append("| nm_id | winner | wins_filtered | wins_no_filter |")
    lines.append("|---:|---|---:|---:|")
    for nm_id in sorted(df["nm_id"].unique()):
        product_no = no_filter[no_filter["nm_id"] == nm_id].iloc[0].to_dict()
        product_filtered = filtered[filtered["nm_id"] == nm_id].iloc[0].to_dict()
        wins_filtered = 0
        wins_no = 0
        for column, direction in verdict_metrics.items():
            left = _metric_value(product_no, column)
            right = _metric_value(product_filtered, column)
            if left is None or right is None or left == right:
                continue
            if (direction == "lower" and right < left) or (direction == "higher" and right > left):
                wins_filtered += 1
            else:
                wins_no += 1
        winner = "filtered" if wins_filtered > wins_no else "no_filter" if wins_no > wins_filtered else "tie"
        lines.append(f"| {int(nm_id)} | {winner} | {wins_filtered} | {wins_no} |")

    lines.extend(["", "## Verdict", "", verdict])
    return "\n".join(lines).rstrip() + "\n"


def _render_filter_impact(filtered_reports: list[ProductDiscoveryReport]) -> str:
    total_input = sum(report.filter_report.total_input for report in filtered_reports if report.filter_report)
    total_filtered = sum(report.filter_report.total_filtered for report in filtered_reports if report.filter_report)
    by_rule: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for report in filtered_reports:
        if not report.filter_report:
            continue
        for rule, count in report.filter_report.filtered_by_rule.items():
            by_rule[rule] = by_rule.get(rule, 0) + int(count)
        for rule, items in report.filter_report.sample_filtered.items():
            samples.setdefault(rule, [])
            for item in items:
                if len(samples[rule]) < 5:
                    samples[rule].append(item)

    rate = float(total_filtered) / float(total_input) if total_input else 0.0
    lines = [
        "# Filter impact report",
        "",
        f"Total phrases input: {total_input}",
        f"Total phrases filtered: {total_filtered} ({rate:.2%})",
        "",
        "## Top filter reasons",
    ]
    for rule, count in sorted(by_rule.items(), key=lambda item: (-item[1], item[0])):
        examples = ", ".join(f'"{item}"' for item in samples.get(rule, []))
        lines.append(f"- {rule}: {count}; examples: {examples}")
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _snapshot_config() -> dict[str, object]:
    return {
        "encoder_model": ENCODER_MODEL,
        "encoder_batch_size": ENCODER_BATCH_SIZE,
        "min_unique_phrases_to_cluster": MIN_UNIQUE_PHRASES_TO_CLUSTER,
        "top_n_phrases_per_cluster": TOP_N_PHRASES_PER_CLUSTER,
        "hdbscan": dict(HDBSCAN_PARAMS),
    }


def _run_or_load_product(
    *,
    cache: DiscoverySnapshotCache | None,
    pipeline: PerProductDiscoveryV3,
    nm_id: int,
    category_id: str,
    reviews: list,
    gold_labels: dict,
    vocabulary: object,
    apply_filter: bool,
) -> tuple[ProductDiscoveryReport, bool]:
    if cache is None:
        return (
            pipeline.run(
                nm_id=nm_id,
                category_id=category_id,
                reviews=reviews,
                gold_labels=gold_labels,
                vocabulary=vocabulary,
                apply_filter=apply_filter,
            ),
            False,
        )

    key = compute_product_snapshot_key(
        nm_id=nm_id,
        category_id=category_id,
        reviews=reviews,
        gold_labels=gold_labels,
        vocabulary=vocabulary,
        apply_filter=apply_filter,
        config=_snapshot_config(),
    )
    cached = cache.load(key, nm_id=nm_id, apply_filter=apply_filter)
    if cached is not None:
        return cached, True

    report = pipeline.run(
        nm_id=nm_id,
        category_id=category_id,
        reviews=reviews,
        gold_labels=gold_labels,
        vocabulary=vocabulary,
        apply_filter=apply_filter,
    )
    cache.save(report, key, apply_filter=apply_filter)
    return report, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run discovery v3 A/B per product.")
    parser.add_argument("--dataset-csv", default="./data/dataset_final.csv")
    parser.add_argument("--output-dir", default="./benchmark/discovery/results")
    parser.add_argument(
        "--snapshot-cache-dir",
        default="./benchmark/discovery/snapshots/v3",
        help="Directory for deterministic per-product discovery snapshots.",
    )
    parser.add_argument(
        "--no-snapshot-cache",
        action="store_true",
        help="Disable snapshot reuse and recompute all discovery artifacts.",
    )
    args = parser.parse_args()

    start = datetime.now()
    dataset_csv = _resolve_repo_path(args.dataset_csv)
    output_root = _resolve_repo_path(args.output_dir)
    output_dir = output_root / f"{start.strftime('%Y%m%d_%H%M%S')}_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_cache = (
        None
        if args.no_snapshot_cache
        else DiscoverySnapshotCache(_resolve_repo_path(args.snapshot_cache_dir))
    )

    df = pd.read_csv(dataset_csv)
    category_column = "category_id" if "category_id" in df.columns else "category"
    if category_column not in df.columns:
        raise ValueError("Dataset must contain `category_id` or `category` column.")
    if "nm_id" not in df.columns:
        raise ValueError("Dataset must contain `nm_id` column.")

    encoder = DiscoveryEncoder(
        model_name_or_path=ENCODER_MODEL,
        batch_size=ENCODER_BATCH_SIZE,
    )
    pipeline = PerProductDiscoveryV3(
        encoder=encoder,
        residual_extractor=ResidualExtractor(),
        min_unique_phrases_to_cluster=MIN_UNIQUE_PHRASES_TO_CLUSTER,
        top_n_phrases_per_cluster=TOP_N_PHRASES_PER_CLUSTER,
        hdbscan_params=HDBSCAN_PARAMS,
    )

    rows: list[dict[str, Any]] = []
    filtered_reports: list[ProductDiscoveryReport] = []
    template_reports: dict[int, ProductDiscoveryReport] = {}
    cache_hits = 0
    cache_misses = 0
    for (category_id, nm_id), product_df in df.groupby([category_column, "nm_id"], sort=True):
        category_id = str(category_id)
        nm_id = int(nm_id)
        vocabulary = _load_hybrid_vocabulary(category_id)
        reviews = [_row_to_review_input(row) for _, row in product_df.iterrows()]
        gold_labels = {
            str(row["id"]): _parse_gold_labels(row.get("true_labels"))
            for _, row in product_df.iterrows()
        }
        no_filter_report, no_filter_hit = _run_or_load_product(
            cache=snapshot_cache,
            pipeline=pipeline,
            nm_id=nm_id,
            category_id=category_id,
            reviews=reviews,
            gold_labels=gold_labels,
            vocabulary=vocabulary,
            apply_filter=False,
        )
        filtered_report, filtered_hit = _run_or_load_product(
            cache=snapshot_cache,
            pipeline=pipeline,
            nm_id=nm_id,
            category_id=category_id,
            reviews=reviews,
            gold_labels=gold_labels,
            vocabulary=vocabulary,
            apply_filter=True,
        )
        cache_hits += int(no_filter_hit) + int(filtered_hit)
        cache_misses += int(not no_filter_hit) + int(not filtered_hit)
        print(
            "[discovery-v3] "
            f"{category_id}/{nm_id}: "
            f"no_filter={'cache' if no_filter_hit else 'computed'}, "
            f"filtered={'cache' if filtered_hit else 'computed'}"
        )
        filtered_reports.append(filtered_report)
        template_reports[nm_id] = filtered_report
        rows.append(_build_metrics_row(no_filter_report, "no_filter"))
        rows.append(_build_metrics_row(filtered_report, "filtered"))

        (output_dir / f"product_{nm_id}_no_filter.json").write_text(
            json.dumps(asdict(no_filter_report), ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        (output_dir / f"product_{nm_id}_filtered.json").write_text(
            json.dumps(asdict(filtered_report), ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        (output_dir / f"product_{nm_id}_comparison.md").write_text(
            _render_product_markdown(no_filter=no_filter_report, filtered=filtered_report),
            encoding="utf-8",
        )

    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary_v3.csv", index=False)
    (output_dir / "ab_comparison_summary.md").write_text(
        _render_ab_summary(rows),
        encoding="utf-8",
    )
    (output_dir / "filter_impact_report.md").write_text(
        _render_filter_impact(filtered_reports),
        encoding="utf-8",
    )
    prepare_manual_labels_template(template_reports, output_dir / "manual_cluster_labels.csv")

    elapsed = datetime.now() - start
    print(f"[discovery-v3] Saved artifacts to {output_dir}")
    print(f"[discovery-v3] Products: {len(filtered_reports)}")
    print(f"[discovery-v3] Rows: {len(rows)}")
    print(f"[discovery-v3] Snapshot cache hits: {cache_hits}")
    print(f"[discovery-v3] Snapshot cache misses: {cache_misses}")
    print(f"[discovery-v3] Runtime: {elapsed}")


if __name__ == "__main__":
    main()
