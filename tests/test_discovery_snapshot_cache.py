from __future__ import annotations

from src.discovery.metrics_l1_intrinsic import IntrinsicMetrics
from src.discovery.metrics_l2_semantic import CoverageReport, SemanticMetrics
from src.discovery.per_product_pipeline_v3 import ClusterSummary, ProductDiscoveryReport
from src.discovery.phrase_filter import FilterReport
from src.discovery.snapshot_cache import (
    DiscoverySnapshotCache,
    compute_product_snapshot_key,
)
from src.schemas.models import ReviewInput
from src.vocabulary.loader import AspectDefinition, Vocabulary


def _vocabulary() -> Vocabulary:
    aspect = AspectDefinition(
        id="quality",
        canonical_name="Quality",
        synonyms=["build", "material"],
        level="core",
        domains=["physical_goods"],
        hypothesis_template="Quality is mentioned",
    )
    return Vocabulary([aspect], _by_id={aspect.id: aspect}, _by_canonical={aspect.canonical_name: aspect})


def _reviews(text: str = "good build") -> list[ReviewInput]:
    return [ReviewInput(id="r1", nm_id=123, rating=5, full_text=text)]


def _report() -> ProductDiscoveryReport:
    coverage = CoverageReport(
        threshold=0.65,
        coverage=1.0,
        matches={"Quality": (0, 0.91)},
        unmatched_aspects=[],
    )
    return ProductDiscoveryReport(
        nm_id=123,
        category_id="physical_goods",
        n_reviews=1,
        n_unique_residuals_before_filter=3,
        n_unique_residuals_after_filter=2,
        filter_report=FilterReport(
            total_input=3,
            total_kept=2,
            total_filtered=1,
            filtered_by_rule={"too_short": 1},
            filter_rate=1 / 3,
            sample_filtered={"too_short": ["x"]},
        ),
        cluster_summaries=[
            ClusterSummary(
                cluster_id=0,
                n_phrases=2,
                total_weight=4,
                top_phrases=[("build", 3), ("material", 1)],
            )
        ],
        n_clusters=1,
        noise_rate=0.0,
        intrinsic_metrics=IntrinsicMetrics(
            cohesion=0.8,
            separation=0.0,
            silhouette=None,
            avg_concentration=1.0,
            n_clusters_evaluated=1,
        ),
        semantic_metrics=SemanticMetrics(
            primary_threshold=0.65,
            coverage_primary=coverage,
            sensitivity={0.65: coverage},
            avg_soft_purity=1.0,
            n_novel_clusters=0,
            novel_cluster_ids=[],
        ),
        manual_metrics=None,
        metadata={"generated_at": "test"},
    )


def test_snapshot_key_is_stable_and_sensitive_to_inputs() -> None:
    common = {
        "nm_id": 123,
        "category_id": "physical_goods",
        "gold_labels": {"r1": ["Quality"]},
        "vocabulary": _vocabulary(),
        "apply_filter": True,
        "config": {"hdbscan": {"min_cluster_size": 5}},
    }

    first = compute_product_snapshot_key(reviews=_reviews(), **common)
    second = compute_product_snapshot_key(reviews=_reviews(), **common)
    changed = compute_product_snapshot_key(reviews=_reviews("bad build"), **common)

    assert first == second
    assert first != changed


def test_snapshot_cache_roundtrip(tmp_path) -> None:
    cache = DiscoverySnapshotCache(tmp_path)
    report = _report()
    key = "a" * 64

    saved_path = cache.save(report, key, apply_filter=True)
    loaded = cache.load(key, nm_id=123, apply_filter=True)

    assert saved_path.exists()
    assert loaded is not None
    assert loaded.nm_id == 123
    assert loaded.cluster_summaries[0].top_phrases == [("build", 3), ("material", 1)]
    assert loaded.filter_report is not None
    assert loaded.filter_report.total_filtered == 1
    assert loaded.semantic_metrics.coverage_primary.matches["Quality"] == (0, 0.91)
    assert loaded.metadata["snapshot_cache"]["hit"] is True
