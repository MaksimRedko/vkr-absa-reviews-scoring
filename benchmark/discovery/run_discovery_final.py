from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.discovery.manual_eval import (  # noqa: E402
    VALID_LABELS,
    ManualLabel,
    compute_manual_metrics,
    load_manual_labels,
)


FINAL_EXTRA_COLUMNS = [
    "manual_n_total_clusters",
    "manual_n_labeled_clusters",
    "manual_valid_known",
    "manual_valid_novel",
    "manual_mixed",
    "manual_noise",
    "manual_valid_rate",
    "manual_novel_rate",
    "manual_noise_rate",
    "manual_novel_aspect_count",
]


def main() -> None:
    args = parse_args()
    v3_dir = args.v3_dir or find_latest_v3_dir()
    metrics_path = v3_dir / "metrics_summary_v3.csv"
    manual_path = args.manual_labels or Path("benchmark/discovery/manual_labels/manual_cluster_labels_draft.csv")
    output_dir = args.output_dir or default_output_dir(v3_dir)

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics_summary_v3.csv not found: {metrics_path}")
    if not manual_path.exists():
        raise FileNotFoundError(
            "manual labels file not found: "
            f"{manual_path}. Put manual_cluster_labels_draft.csv into benchmark/discovery/manual_labels/"
        )

    rows = read_csv_dicts(metrics_path)
    blank_labels = count_blank_labels(manual_path)
    if blank_labels:
        print(f"[discovery-final] WARNING: {blank_labels} строк в manual разметке без label, пропущены")

    labels = load_manual_labels(manual_path)
    filtered_rows = [row for row in rows if row.get("config") == "filtered"]
    total_clusters = {
        int(row["nm_id"]): parse_int(row.get("n_clusters"))
        for row in filtered_rows
    }
    expected_clusters = sum(total_clusters.values())
    if len(labels) != expected_clusters:
        print(
            "[discovery-final] WARNING: labeled clusters count mismatch: "
            f"manual={len(labels)}, filtered_csv={expected_clusters}"
        )

    labels_by_product = group_labels_by_product(labels.values())
    product_manual = build_product_manual_metrics(labels_by_product, total_clusters)
    final_rows = attach_manual_metrics(rows, product_manual)
    per_category = build_per_category_metrics(filtered_rows, labels_by_product)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_dicts(output_dir / "metrics_summary_final.csv", final_rows)
    write_csv_dicts(output_dir / "per_category_metrics.csv", per_category)
    write_report(output_dir / "final_report.md", filtered_rows, product_manual, per_category)

    print(f"[discovery-final] Source v3: {v3_dir}")
    print(f"[discovery-final] Manual labels: {manual_path}")
    print(f"[discovery-final] Output: {output_dir}")
    print(f"[discovery-final] Rows: {len(final_rows)}")
    print(f"[discovery-final] Manual labeled clusters: {len(labels)}")
    print(f"[discovery-final] Generated: metrics_summary_final.csv")
    print(f"[discovery-final] Generated: final_report.md")
    print(f"[discovery-final] Generated: per_category_metrics.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add Level 3 manual metrics to Discovery v3 results.")
    parser.add_argument("--v3-dir", type=Path, default=None, help="Path to *_v3 results directory.")
    parser.add_argument(
        "--manual-labels",
        type=Path,
        default=None,
        help="Path to manual_cluster_labels_draft.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Path to *_final output directory.")
    return parser.parse_args()


def find_latest_v3_dir() -> Path:
    results_root = Path("benchmark/discovery/results")
    candidates = sorted(
        [path for path in results_root.glob("*_v3") if path.is_dir()],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError("No *_v3 directory found in benchmark/discovery/results")
    return candidates[-1]


def default_output_dir(v3_dir: Path) -> Path:
    if v3_dir.name.endswith("_v3"):
        return v3_dir.with_name(v3_dir.name[:-3] + "_final")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("benchmark/discovery/results") / f"{stamp}_final"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_blank_labels(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for row in reader if not str(row.get("label", "")).strip())


def group_labels_by_product(labels: Iterable[ManualLabel]) -> dict[int, dict[tuple[int, int], ManualLabel]]:
    grouped: dict[int, dict[tuple[int, int], ManualLabel]] = defaultdict(dict)
    for label in labels:
        grouped[label.nm_id][(label.nm_id, label.cluster_id)] = label
    return dict(grouped)


def build_product_manual_metrics(
    labels_by_product: dict[int, dict[tuple[int, int], ManualLabel]],
    total_clusters: dict[int, int],
) -> dict[int, dict[str, object]]:
    product_metrics: dict[int, dict[str, object]] = {}
    for nm_id, n_total in total_clusters.items():
        labels = labels_by_product.get(nm_id, {})
        metrics = compute_manual_metrics(labels, {nm_id: n_total})
        distribution = metrics.label_distribution
        n_labeled = metrics.n_labeled_clusters
        valid_novel = distribution["valid_novel"]
        product_metrics[nm_id] = {
            "manual_n_total_clusters": metrics.n_total_clusters,
            "manual_n_labeled_clusters": n_labeled,
            "manual_valid_known": distribution["valid_known"],
            "manual_valid_novel": valid_novel,
            "manual_mixed": distribution["mixed"],
            "manual_noise": distribution["noise"],
            "manual_valid_rate": metrics.valid_rate if n_labeled else "",
            "manual_novel_rate": float(valid_novel) / float(n_labeled) if n_labeled else "",
            "manual_noise_rate": metrics.noise_rate if n_labeled else "",
            "manual_novel_aspect_count": metrics.novel_aspect_count,
        }
    return product_metrics


def attach_manual_metrics(
    rows: list[dict[str, str]],
    product_manual: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    final_rows: list[dict[str, object]] = []
    for row in rows:
        out: dict[str, object] = dict(row)
        if row.get("config") == "filtered":
            out.update(product_manual.get(int(row["nm_id"]), blank_manual_fields()))
        else:
            out.update(blank_manual_fields())
        final_rows.append(out)
    return final_rows


def blank_manual_fields() -> dict[str, object]:
    return {column: "" for column in FINAL_EXTRA_COLUMNS}


def build_per_category_metrics(
    filtered_rows: list[dict[str, str]],
    labels_by_product: dict[int, dict[tuple[int, int], ManualLabel]],
) -> list[dict[str, object]]:
    category_rows: dict[str, dict[str, object]] = {}
    for row in filtered_rows:
        category = row["category_id"]
        item = category_rows.setdefault(
            category,
            {
                "category": category,
                "n_products": 0,
                "n_clusters": 0,
                "n_labeled_clusters": 0,
                "valid_known": 0,
                "valid_novel": 0,
                "mixed": 0,
                "noise": 0,
            },
        )
        item["n_products"] = int(item["n_products"]) + 1
        item["n_clusters"] = int(item["n_clusters"]) + parse_int(row.get("n_clusters"))
        for label in labels_by_product.get(int(row["nm_id"]), {}).values():
            item["n_labeled_clusters"] = int(item["n_labeled_clusters"]) + 1
            item[label.label] = int(item[label.label]) + 1

    result: list[dict[str, object]] = []
    for item in sorted(category_rows.values(), key=lambda value: str(value["category"])):
        labeled = int(item["n_labeled_clusters"])
        valid = int(item["valid_known"]) + int(item["valid_novel"])
        result.append(
            {
                **item,
                "valid_rate": float(valid) / float(labeled) if labeled else "",
                "novel_rate": float(int(item["valid_novel"])) / float(labeled) if labeled else "",
                "noise_rate": float(int(item["noise"])) / float(labeled) if labeled else "",
            }
        )
    return result


def write_report(
    path: Path,
    filtered_rows: list[dict[str, str]],
    product_manual: dict[int, dict[str, object]],
    per_category: list[dict[str, object]],
) -> None:
    l3_totals = sum_manual(product_manual.values())
    labeled = int(l3_totals["manual_n_labeled_clusters"])
    valid_known = int(l3_totals["manual_valid_known"])
    valid_novel = int(l3_totals["manual_valid_novel"])
    mixed = int(l3_totals["manual_mixed"])
    noise = int(l3_totals["manual_noise"])
    valid_total = valid_known + valid_novel

    lines = [
        "# Discovery v3 Final Report - All Three Metric Levels",
        "",
        "## Aggregate statistics (filtered configuration only)",
        "",
        "### Level 1 - Intrinsic metrics (без gold)",
        f"- Avg cohesion: {fmt(avg(filtered_rows, 'cohesion'))}",
        f"- Avg separation: {fmt(avg(filtered_rows, 'separation'))}",
        f"- Avg silhouette: {fmt(avg(filtered_rows, 'silhouette'))} (considering only k>=2 products)",
        f"- Avg concentration: {fmt(avg(filtered_rows, 'avg_concentration'))}",
        "",
        "### Level 2 - Semantic vs gold (threshold=0.65)",
        f"- Avg coverage: {pct(avg(filtered_rows, 'coverage_at_0.65'))}",
        "- Sensitivity (0.60 / 0.65 / 0.70): "
        f"{pct(avg(filtered_rows, 'coverage_at_0.60'))} / "
        f"{pct(avg(filtered_rows, 'coverage_at_0.65'))} / "
        f"{pct(avg(filtered_rows, 'coverage_at_0.70'))}",
        f"- Avg soft purity: {fmt(avg(filtered_rows, 'avg_soft_purity'))}",
        f"- Novel aspects (automatic detection): {sum_int(filtered_rows, 'n_novel_clusters')}",
        "",
        f"### Level 3 - Manual evaluation ({labeled} clusters labeled)",
        f"- Total valid: {valid_total} ({pct_ratio(valid_total, labeled)})",
        f"  - valid_known: {valid_known} ({pct_ratio(valid_known, labeled)})",
        f"  - valid_novel: {valid_novel} ({pct_ratio(valid_novel, labeled)})",
        f"- mixed: {mixed} ({pct_ratio(mixed, labeled)})",
        f"- noise: {noise} ({pct_ratio(noise, labeled)})",
        "",
        "## Per-category breakdown",
        "",
        "| category | n_clusters | valid_rate | novel_rate | noise_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in per_category:
        lines.append(
            f"| {row['category']} | {row['n_clusters']} | "
            f"{pct_value(row['valid_rate'])} | {pct_value(row['novel_rate'])} | {pct_value(row['noise_rate'])} |"
        )

    lines.extend(["", "## Per-product breakdown (top-5 best, bottom-3 worst)", ""])
    ranked = ranked_products(product_manual)
    lines.extend(product_table("Top-5 best", ranked[:5]))
    lines.extend([""])
    lines.extend(product_table("Bottom-3 worst", ranked[-3:]))

    best_categories = sorted(
        [row for row in per_category if row["valid_rate"] != ""],
        key=lambda row: float(row["valid_rate"]),
        reverse=True,
    )
    weak_categories = list(reversed(best_categories[-2:])) if best_categories else []
    lines.extend(
        [
            "",
            "## Key findings",
            f"- Метод даёт valid clusters в {pct_ratio(valid_total, labeled)} случаев (ручная оценка, n={labeled})",
            f"- Обнаружено {valid_novel} новых аспектов, не учтённых в разметке",
            "- Лучшие категории: " + ", ".join(str(row["category"]) for row in best_categories[:2]),
            "- Слабые категории: " + ", ".join(str(row["category"]) for row in weak_categories),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def sum_manual(items: Iterable[dict[str, object]]) -> dict[str, int]:
    totals = {column: 0 for column in FINAL_EXTRA_COLUMNS if column.startswith("manual_")}
    for item in items:
        for key in [
            "manual_n_total_clusters",
            "manual_n_labeled_clusters",
            "manual_valid_known",
            "manual_valid_novel",
            "manual_mixed",
            "manual_noise",
            "manual_novel_aspect_count",
        ]:
            value = item.get(key, 0)
            totals[key] = totals.get(key, 0) + int(value or 0)
    return totals


def ranked_products(product_manual: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for nm_id, metrics in product_manual.items():
        if metrics.get("manual_valid_rate") == "":
            continue
        rows.append({"nm_id": nm_id, **metrics})
    return sorted(rows, key=lambda row: float(row["manual_valid_rate"]), reverse=True)


def product_table(title: str, rows: list[dict[str, object]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| nm_id | labeled | valid_rate | valid_novel | noise |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['nm_id']} | {row['manual_n_labeled_clusters']} | "
            f"{pct_value(row['manual_valid_rate'])} | {row['manual_valid_novel']} | {row['manual_noise']} |"
        )
    return lines


def avg(rows: list[dict[str, str]], key: str) -> float | None:
    values = [parse_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def sum_int(rows: list[dict[str, str]], key: str) -> int:
    return sum(parse_int(row.get(key)) for row in rows)


def parse_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def parse_int(value: object) -> int:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else 0


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.1f}%"


def pct_ratio(numerator: int, denominator: int) -> str:
    return "NA" if denominator == 0 else f"{float(numerator) / float(denominator) * 100:.1f}%"


def pct_value(value: object) -> str:
    if value == "":
        return "NA"
    return pct(float(value))


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        valid = ", ".join(sorted(VALID_LABELS))
        raise SystemExit(f"[discovery-final] ERROR: {exc}. Valid labels: {valid}") from exc
