from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from src.discovery.metrics_l1_intrinsic import IntrinsicMetrics
from src.discovery.metrics_l2_semantic import CoverageReport, SemanticMetrics
from src.discovery.per_product_pipeline_v3 import ClusterSummary, ProductDiscoveryReport
from src.discovery.phrase_filter import FilterReport
from src.schemas.models import ReviewInput
from src.vocabulary.loader import Vocabulary

SNAPSHOT_SCHEMA_VERSION = "discovery_v3_product_snapshot_v1"


def json_default(obj: object) -> object:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def compute_product_snapshot_key(
    *,
    nm_id: int,
    category_id: str,
    reviews: list[ReviewInput],
    gold_labels: Mapping[str, object],
    vocabulary: Vocabulary,
    apply_filter: bool,
    config: Mapping[str, object],
) -> str:
    payload = {
        "schema": SNAPSHOT_SCHEMA_VERSION,
        "nm_id": int(nm_id),
        "category_id": str(category_id),
        "apply_filter": bool(apply_filter),
        "reviews": [_review_payload(review) for review in reviews],
        "gold_labels": _normalize_jsonable(gold_labels),
        "vocabulary": _vocabulary_payload(vocabulary),
        "config": _normalize_jsonable(config),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


class DiscoverySnapshotCache:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def path_for(self, key: str, *, nm_id: int, apply_filter: bool) -> Path:
        suffix = "filtered" if apply_filter else "no_filter"
        return self.root_dir / key[:2] / key / f"product_{int(nm_id)}_{suffix}.json"

    def load(self, key: str, *, nm_id: int, apply_filter: bool) -> ProductDiscoveryReport | None:
        path = self.path_for(key, nm_id=nm_id, apply_filter=apply_filter)
        if not path.exists():
            return None
        report = load_product_report(path)
        report.metadata["snapshot_cache"] = {
            "hit": True,
            "schema": SNAPSHOT_SCHEMA_VERSION,
            "key": key,
            "path": str(path),
        }
        return report

    def save(
        self,
        report: ProductDiscoveryReport,
        key: str,
        *,
        apply_filter: bool,
    ) -> Path:
        path = self.path_for(key, nm_id=report.nm_id, apply_filter=apply_filter)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.metadata["snapshot_cache"] = {
            "hit": False,
            "schema": SNAPSHOT_SCHEMA_VERSION,
            "key": key,
            "path": str(path),
        }
        path.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )
        return path


def load_product_report(path: str | Path) -> ProductDiscoveryReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    filter_report = data.get("filter_report")
    return ProductDiscoveryReport(
        nm_id=int(data["nm_id"]),
        category_id=str(data["category_id"]),
        n_reviews=int(data["n_reviews"]),
        n_unique_residuals_before_filter=int(data["n_unique_residuals_before_filter"]),
        n_unique_residuals_after_filter=int(data["n_unique_residuals_after_filter"]),
        filter_report=_filter_report(filter_report) if filter_report else None,
        cluster_summaries=[
            ClusterSummary(
                cluster_id=int(item["cluster_id"]),
                n_phrases=int(item["n_phrases"]),
                total_weight=int(item["total_weight"]),
                top_phrases=[(str(phrase), int(weight)) for phrase, weight in item["top_phrases"]],
            )
            for item in data.get("cluster_summaries", [])
        ],
        n_clusters=int(data["n_clusters"]),
        noise_rate=float(data["noise_rate"]),
        intrinsic_metrics=_intrinsic_metrics(data["intrinsic_metrics"]),
        semantic_metrics=_semantic_metrics(data["semantic_metrics"]),
        manual_metrics=None,
        metadata=dict(data.get("metadata", {})),
    )


def _review_payload(review: ReviewInput) -> dict[str, object]:
    return {
        "id": str(review.id),
        "nm_id": int(review.nm_id),
        "rating": int(review.rating),
        "clean_text": review.clean_text,
    }


def _vocabulary_payload(vocabulary: Vocabulary) -> list[dict[str, object]]:
    return [
        {
            "id": aspect.id,
            "canonical_name": aspect.canonical_name,
            "synonyms": list(aspect.synonyms),
            "level": aspect.level,
            "domains": list(aspect.domains),
            "hypothesis_template": aspect.hypothesis_template,
        }
        for aspect in sorted(vocabulary.aspects, key=lambda item: item.id)
    ]


def _normalize_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_jsonable(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _filter_report(data: Mapping[str, object]) -> FilterReport:
    return FilterReport(
        total_input=int(data["total_input"]),
        total_kept=int(data["total_kept"]),
        total_filtered=int(data["total_filtered"]),
        filtered_by_rule={str(k): int(v) for k, v in dict(data["filtered_by_rule"]).items()},
        filter_rate=float(data["filter_rate"]),
        sample_filtered={
            str(k): [str(item) for item in v]
            for k, v in dict(data.get("sample_filtered", {})).items()
        },
    )


def _intrinsic_metrics(data: Mapping[str, object]) -> IntrinsicMetrics:
    silhouette = data.get("silhouette")
    return IntrinsicMetrics(
        cohesion=float(data["cohesion"]),
        separation=float(data["separation"]),
        silhouette=float(silhouette) if silhouette is not None else None,
        avg_concentration=float(data["avg_concentration"]),
        n_clusters_evaluated=int(data["n_clusters_evaluated"]),
    )


def _semantic_metrics(data: Mapping[str, object]) -> SemanticMetrics:
    sensitivity = {
        float(threshold): _coverage_report(report)
        for threshold, report in dict(data["sensitivity"]).items()
    }
    primary_threshold = float(data["primary_threshold"])
    coverage_primary = sensitivity.get(primary_threshold) or _coverage_report(
        data["coverage_primary"]
    )
    return SemanticMetrics(
        primary_threshold=primary_threshold,
        coverage_primary=coverage_primary,
        sensitivity=sensitivity,
        avg_soft_purity=float(data["avg_soft_purity"]),
        n_novel_clusters=int(data["n_novel_clusters"]),
        novel_cluster_ids=[int(item) for item in data.get("novel_cluster_ids", [])],
    )


def _coverage_report(data: Mapping[str, object]) -> CoverageReport:
    return CoverageReport(
        threshold=float(data["threshold"]),
        coverage=float(data["coverage"]),
        matches={
            str(aspect): (int(match[0]), float(match[1]))
            for aspect, match in dict(data.get("matches", {})).items()
        },
        unmatched_aspects=[str(item) for item in data.get("unmatched_aspects", [])],
    )
