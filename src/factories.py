from __future__ import annotations

from typing import Callable, Dict, Optional

from sentence_transformers import SentenceTransformer

from configs.configs import config
from src.stages.aggregation import RatingMathEngine
from src.stages.clustering import (
    AspectClusterer,
    DivisiveClusterer,
    MDLDivisiveClusterer,
)
from src.stages.contracts import (
    AggregationStage,
    ClusteringStage,
    ExtractionStage,
    FraudStage,
    PairingStage,
    ScoringStage,
    SentimentStage,
)
from src.stages.extraction import build_extraction_stage as build_configured_extraction_stage
from src.stages.fraud import AntiFraudEngine
from src.stages.naming import MedoidNamer
from src.stages.pairing import build_pairing_stage as build_configured_pairing_stage
from src.stages.scoring import KeyBERTScorer
from src.stages.sentiment import SentimentEngine


ClustererBuilder = Callable[[SentenceTransformer], ClusteringStage]


def _build_aspect_clusterer(encoder: SentenceTransformer) -> ClusteringStage:
    return AspectClusterer(model=encoder)


def _build_divisive_clusterer(encoder: SentenceTransformer) -> ClusteringStage:
    return DivisiveClusterer(model=encoder, namer=MedoidNamer())


def _build_mdl_divisive_clusterer(encoder: SentenceTransformer) -> ClusteringStage:
    return MDLDivisiveClusterer(
        model=encoder,
        namer=MedoidNamer(),
        use_aicc_correction=bool(
            getattr(config.discovery, "mdl_use_aicc_correction", True)
        ),
        model_penalty_alpha=float(
            getattr(config.discovery, "mdl_model_penalty_alpha", 1.0)
        ),
    )


CLUSTERING_STAGE_REGISTRY: Dict[str, ClustererBuilder] = {
    "aspect": _build_aspect_clusterer,
    "divisive": _build_divisive_clusterer,
    "mdl_divisive": _build_mdl_divisive_clusterer,
}


def build_extraction_stage() -> ExtractionStage:
    return build_configured_extraction_stage()


def build_scoring_stage(encoder: SentenceTransformer) -> ScoringStage:
    return KeyBERTScorer(model=encoder)


def build_clustering_stage(
    encoder: SentenceTransformer,
    name: Optional[str] = None,
) -> ClusteringStage:
    clusterer_name = str(name or "aspect")
    builder = CLUSTERING_STAGE_REGISTRY.get(clusterer_name)
    if builder is None:
        raise ValueError(
            f"Unsupported clusterer={clusterer_name!r}; "
            f"expected one of {sorted(CLUSTERING_STAGE_REGISTRY)}."
        )
    return builder(encoder)


def build_pairing_stage() -> PairingStage:
    return build_configured_pairing_stage()


def build_fraud_stage(encoder: SentenceTransformer) -> FraudStage:
    return AntiFraudEngine(model=encoder)


def build_sentiment_stage() -> SentimentStage:
    return SentimentEngine()


def build_aggregation_stage() -> AggregationStage:
    return RatingMathEngine()
