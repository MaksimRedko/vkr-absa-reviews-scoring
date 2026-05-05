from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

VALID_LABELS = {"valid_known", "valid_novel", "mixed", "noise"}


@dataclass(slots=True)
class ManualLabel:
    nm_id: int
    cluster_id: int
    label: Literal["valid_known", "valid_novel", "mixed", "noise"]
    comment: str
    dominant_gold_aspect: Optional[str]


@dataclass(slots=True)
class ManualMetrics:
    n_total_clusters: int
    n_labeled_clusters: int
    label_distribution: dict[str, int]
    valid_rate: float
    novel_aspect_count: int
    noise_rate: float
    per_product: dict[int, dict[str, int]]
    coverage_by_human: dict[int, float]


def prepare_manual_labels_template(discovery_results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for report in discovery_results.values():
        if hasattr(report, "cluster_summaries"):
            nm_id = int(getattr(report, "nm_id"))
            category_id = str(getattr(report, "category_id"))
            summaries = getattr(report, "cluster_summaries")
        else:
            nm_id = int(report["nm_id"])
            category_id = str(report["category_id"])
            summaries = report.get("cluster_summaries", [])
        for summary in summaries:
            cluster_id = int(_get(summary, "cluster_id"))
            top_phrases = _get(summary, "top_phrases") or []
            rows.append(
                {
                    "nm_id": nm_id,
                    "category_id": category_id,
                    "cluster_id": cluster_id,
                    "n_phrases": int(_get(summary, "n_phrases")),
                    "top_5_phrases": " | ".join(str(item[0]) for item in top_phrases[:5]),
                    "label": "",
                    "comment": "",
                    "dominant_gold_aspect": "",
                }
            )

    fieldnames = [
        "nm_id",
        "category_id",
        "cluster_id",
        "n_phrases",
        "top_5_phrases",
        "label",
        "comment",
        "dominant_gold_aspect",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manual_labels(path: Path) -> dict[tuple[int, int], ManualLabel]:
    labels: dict[tuple[int, int], ManualLabel] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            raw_label = str(row.get("label", "")).strip()
            if not raw_label:
                print(f"[manual-eval] skip unlabeled row {line_number}")
                continue
            if raw_label not in VALID_LABELS:
                raise ValueError(f"Invalid label at row {line_number}: {raw_label}")
            nm_id = int(str(row.get("nm_id", "")).strip())
            cluster_id = int(str(row.get("cluster_id", "")).strip())
            labels[(nm_id, cluster_id)] = ManualLabel(
                nm_id=nm_id,
                cluster_id=cluster_id,
                label=raw_label,  # type: ignore[arg-type]
                comment=str(row.get("comment", "") or ""),
                dominant_gold_aspect=str(row.get("dominant_gold_aspect", "") or "").strip() or None,
            )
    return labels


def compute_manual_metrics(
    labels: dict[tuple[int, int], ManualLabel],
    total_clusters_per_product: dict[int, int],
) -> ManualMetrics:
    distribution = {label: 0 for label in sorted(VALID_LABELS)}
    per_product: dict[int, dict[str, int]] = {
        int(nm_id): {label: 0 for label in sorted(VALID_LABELS)}
        for nm_id in total_clusters_per_product
    }
    for manual_label in labels.values():
        distribution[manual_label.label] += 1
        per_product.setdefault(
            manual_label.nm_id,
            {label: 0 for label in sorted(VALID_LABELS)},
        )[manual_label.label] += 1

    n_labeled = len(labels)
    valid_count = distribution["valid_known"] + distribution["valid_novel"]
    noise_count = distribution["noise"]
    coverage_by_human: dict[int, float] = {}
    for nm_id, product_counts in per_product.items():
        labeled = sum(product_counts.values())
        valid = product_counts["valid_known"] + product_counts["valid_novel"]
        coverage_by_human[nm_id] = float(valid) / float(labeled) if labeled else 0.0

    return ManualMetrics(
        n_total_clusters=sum(int(value) for value in total_clusters_per_product.values()),
        n_labeled_clusters=n_labeled,
        label_distribution=distribution,
        valid_rate=float(valid_count) / float(n_labeled) if n_labeled else 0.0,
        novel_aspect_count=distribution["valid_novel"],
        noise_rate=float(noise_count) / float(n_labeled) if n_labeled else 0.0,
        per_product=per_product,
        coverage_by_human=coverage_by_human,
    )


def _get(obj: object, name: str) -> object:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name)
