from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.discovery.aggregator import ClusterAggregator, ClusterSummary
from src.discovery.clusterer import ReviewClusterer
from src.discovery.encoder import DEFAULT_DISCOVERY_MODEL_NAME, DiscoveryEncoder
from src.discovery.evaluator import ClusterEvaluator, EvaluationReport
from src.discovery.representation import ReviewRepresentation
from src.discovery.residual_extractor import ResidualExtractor
from src.schemas.models import ReviewInput
from src.vocabulary.loader import Vocabulary


@dataclass(slots=True)
class DiscoveryReport:
    cluster_summaries: list[ClusterSummary]
    evaluation: EvaluationReport
    metadata: dict[str, object]


def run_discovery(
    category_id: str,
    reviews: list[ReviewInput],
    gold_labels: dict,
    vocabulary: Vocabulary,
    *,
    encoder: DiscoveryEncoder | None = None,
    residual_extractor: ResidualExtractor | None = None,
    representation_builder: ReviewRepresentation | None = None,
    clusterer: ReviewClusterer | None = None,
    aggregator: ClusterAggregator | None = None,
    evaluator: ClusterEvaluator | None = None,
) -> DiscoveryReport:
    resolved_encoder = encoder or DiscoveryEncoder()
    resolved_residual_extractor = residual_extractor or ResidualExtractor()
    resolved_representation_builder = representation_builder or ReviewRepresentation()
    resolved_clusterer = clusterer or ReviewClusterer()
    resolved_aggregator = aggregator or ClusterAggregator()
    resolved_evaluator = evaluator or ClusterEvaluator()

    residuals = [
        resolved_residual_extractor.extract(
            review=review,
            category_id=category_id,
            vocabulary=vocabulary,
        )
        for review in reviews
    ]
    batch = resolved_representation_builder.build(residuals, resolved_encoder)
    clustering = resolved_clusterer.cluster(batch)
    cluster_summaries = resolved_aggregator.aggregate(residuals, clustering)
    evaluation = resolved_evaluator.evaluate(
        clustering=clustering,
        residuals=residuals,
        gold_labels=gold_labels,
        vocabulary=vocabulary,
    )

    metadata = {
        "category_id": category_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": str(
            getattr(resolved_encoder, "model_name_or_path", DEFAULT_DISCOVERY_MODEL_NAME)
        ),
        "n_reviews_total": len(reviews),
        "n_reviews_with_residual": len(batch.review_ids),
        "n_excluded_reviews": len(batch.excluded_review_ids),
        "excluded_review_ids": list(batch.excluded_review_ids),
        "hdbscan": {
            "min_cluster_size": int(getattr(resolved_clusterer, "min_cluster_size", 15)),
            "min_samples": int(getattr(resolved_clusterer, "min_samples", 5)),
            "metric": str(getattr(resolved_clusterer, "metric", "euclidean")),
            "cluster_selection_method": str(
                getattr(resolved_clusterer, "cluster_selection_method", "eom")
            ),
        },
    }

    return DiscoveryReport(
        cluster_summaries=cluster_summaries,
        evaluation=evaluation,
        metadata=metadata,
    )
