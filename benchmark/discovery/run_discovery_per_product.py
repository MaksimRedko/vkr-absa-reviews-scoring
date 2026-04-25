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
from configs.configs import config  # noqa: E402
from src.discovery import DiscoveryEncoder, ResidualExtractor  # noqa: E402
from src.discovery.per_product_pipeline import (  # noqa: E402
    PerProductDiscoveryReport,
    PerProductPhraseDiscovery,
)


def _render_product_markdown(report: PerProductDiscoveryReport, product_name: str = "") -> str:
    meta = report.metadata
    evaluation = report.evaluation
    title_suffix = f" ({product_name})" if product_name else ""
    noise_pct = 100.0 * evaluation.noise_rate
    lines = [
        f"# Discovery: nm_id {report.nm_id}{title_suffix}",
        "",
        f"Категория: {report.category_id}",
        f"Отзывов о товаре: {int(meta.get('n_reviews', 0))}",
        f"Уникальных непокрытых фраз: {int(meta.get('n_unique_residual_phrases', 0))}",
        f"Найдено кластеров: {evaluation.n_clusters}",
        f"Шум: {int(meta.get('n_noise_phrases', 0))} фраз ({noise_pct:.0f}%)",
        f"Coverage_via_clustering: {evaluation.coverage_via_clustering:.4f}",
        "",
    ]

    if bool(meta.get("skipped_low_unique_phrases", False)):
        lines.append("Пропущено: меньше MIN_UNIQUE_PHRASES_TO_CLUSTER.")
        return "\n".join(lines).rstrip() + "\n"
    if not report.cluster_summaries:
        lines.append("Кластеры не найдены.")
        return "\n".join(lines).rstrip() + "\n"

    for summary in report.cluster_summaries:
        lines.append(
            f"## Кластер {summary.cluster_id} "
            f"({summary.n_phrases} фраз, общий вес {summary.total_weight})"
        )
        for phrase, weight in summary.top_phrases:
            lines.append(f'  - "{phrase}" (вес {weight})')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_metrics_row(report: PerProductDiscoveryReport) -> dict[str, Any]:
    meta = report.metadata
    evaluation = report.evaluation
    return {
        "nm_id": report.nm_id,
        "category_id": report.category_id,
        "n_reviews": int(meta.get("n_reviews", 0)),
        "n_unique_residual_phrases": int(meta.get("n_unique_residual_phrases", 0)),
        "n_clusters": evaluation.n_clusters,
        "n_noise_phrases": int(meta.get("n_noise_phrases", 0)),
        "noise_rate": evaluation.noise_rate,
        "coverage_via_clustering": evaluation.coverage_via_clustering,
        "n_clean_clusters": evaluation.n_clean_clusters,
        "n_uncovered_gold_aspects": evaluation.n_uncovered_gold_aspects,
        "skipped_low_unique_phrases": bool(meta.get("skipped_low_unique_phrases", False)),
    }


def _render_summary(metrics_rows: list[dict[str, Any]], reports: list[PerProductDiscoveryReport]) -> str:
    df = pd.DataFrame(metrics_rows)
    total_products = len(metrics_rows)
    clustered = int((df["n_clusters"] > 0).sum()) if not df.empty else 0
    mean_coverage = float(df["coverage_via_clustering"].mean()) if not df.empty else 0.0
    mean_noise = float(df["noise_rate"].mean()) if not df.empty else 0.0

    lines = [
        "# Discovery v2 per product",
        "",
        f"Товаров: {total_products}",
        f"С кластерами: {clustered}",
        f"Средний coverage_via_clustering: {mean_coverage:.4f}",
        f"Средний noise_rate: {mean_noise:.4f}",
        "",
        "## Лучшие товары",
    ]
    if df.empty:
        lines.append("Нет данных.")
        return "\n".join(lines) + "\n"

    best = df.sort_values(
        ["coverage_via_clustering", "n_clusters", "n_unique_residual_phrases"],
        ascending=[False, False, False],
    ).head(10)
    for _, row in best.iterrows():
        lines.append(
            f"- {row['category_id']} / {int(row['nm_id'])}: "
            f"clusters={int(row['n_clusters'])}, "
            f"coverage={float(row['coverage_via_clustering']):.4f}, "
            f"noise={float(row['noise_rate']):.4f}, "
            f"unique={int(row['n_unique_residual_phrases'])}"
        )

    lines.extend(["", "## Субъективная оценка кластеров"])
    sample_reports = [
        report for report in reports if report.cluster_summaries
    ][:10]
    if not sample_reports:
        lines.append("Осмысленные кластеры визуально не подтверждены: кластеры почти не найдены.")
    for report in sample_reports:
        first_cluster = report.cluster_summaries[0]
        examples = ", ".join(f'"{phrase}"' for phrase, _ in first_cluster.top_phrases[:5])
        lines.append(
            f"- {report.category_id} / {report.nm_id}: "
            f"{report.evaluation.n_clusters} clusters; первый кластер: {examples}"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phrase-level discovery per product.")
    parser.add_argument("--dataset-csv", default=str(config.discovery_per_product.dataset_csv))
    parser.add_argument("--output-dir", default=str(config.discovery_per_product.results_dir))
    args = parser.parse_args()

    start = datetime.now()
    dataset_csv = _resolve_repo_path(args.dataset_csv)
    output_root = _resolve_repo_path(args.output_dir)
    timestamp = start.strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{timestamp}_per_product"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    category_column = "category_id" if "category_id" in df.columns else "category"
    if category_column not in df.columns:
        raise ValueError("Dataset must contain `category_id` or `category` column.")
    if "nm_id" not in df.columns:
        raise ValueError("Dataset must contain `nm_id` column.")

    encoder = DiscoveryEncoder(
        model_name_or_path=str(config.discovery_per_product.encoder_model),
        batch_size=int(config.discovery_per_product.encoder_batch_size),
    )
    pipeline = PerProductPhraseDiscovery(
        encoder=encoder,
        residual_extractor=ResidualExtractor(),
        min_unique_phrases_to_cluster=int(
            config.discovery_per_product.min_unique_phrases_to_cluster
        ),
        top_n_phrases_per_cluster=int(config.discovery_per_product.top_n_phrases_per_cluster),
        purity_threshold=float(config.discovery_per_product.purity_threshold),
        hdbscan_params=dict(config.discovery_per_product.hdbscan),
    )

    reports: list[PerProductDiscoveryReport] = []
    metrics_rows: list[dict[str, Any]] = []
    for (category_id, nm_id), product_df in df.groupby([category_column, "nm_id"], sort=True):
        category_id = str(category_id)
        nm_id = int(nm_id)
        vocabulary = _load_hybrid_vocabulary(category_id)
        reviews = [_row_to_review_input(row) for _, row in product_df.iterrows()]
        gold_labels = {
            str(row["id"]): _parse_gold_labels(row.get("true_labels"))
            for _, row in product_df.iterrows()
        }
        report = pipeline.run(
            nm_id=nm_id,
            category_id=category_id,
            reviews=reviews,
            gold_labels=gold_labels,
            vocabulary=vocabulary,
        )
        reports.append(report)
        metrics_rows.append(_build_metrics_row(report))

        (output_dir / f"discovery_{nm_id}.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"discovery_{nm_id}.md").write_text(
            _render_product_markdown(report),
            encoding="utf-8",
        )

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics_per_product.csv", index=False)
    (output_dir / "summary.md").write_text(
        _render_summary(metrics_rows, reports),
        encoding="utf-8",
    )

    elapsed = datetime.now() - start
    print(f"[discovery-v2] Saved artifacts to {output_dir}")
    print(f"[discovery-v2] Products: {len(metrics_rows)}")
    print(f"[discovery-v2] Runtime: {elapsed}")


if __name__ == "__main__":
    main()
