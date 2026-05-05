from .aggregator import ClusterAggregator, ClusterSummary
from .clusterer import ClusteringResult, ReviewClusterer
from .encoder import (
    DEFAULT_DISCOVERY_CACHE_DIR,
    DEFAULT_DISCOVERY_LOCAL_DIR,
    DEFAULT_DISCOVERY_MODEL_NAME,
    DiscoveryEncoder,
)
from .evaluator import ClusterEvaluator, EvaluationReport
from .manual_eval import ManualLabel, ManualMetrics
from .metrics_l1_intrinsic import IntrinsicMetrics
from .metrics_l2_semantic import CoverageReport, SemanticMetrics
from .pipeline import DiscoveryReport, run_discovery
from .per_product_pipeline import (
    PerProductDiscoveryReport,
    PerProductEvaluationReport,
    PerProductPhraseDiscovery,
    PhraseClusterSummary,
)
from .per_product_pipeline_v3 import ProductDiscoveryReport, run_discovery_v3_per_product
from .phrase_filter import FilterReport, PhraseFilter
from .residual_extractor import ResidualExtractor, ResidualResult
from .snapshot_cache import (
    DiscoverySnapshotCache,
    compute_product_snapshot_key,
    load_product_report,
)
from .representation import ReviewRepresentation, ReviewRepresentationBatch

__all__ = [
    "ClusterAggregator",
    "ClusterSummary",
    "ClusterEvaluator",
    "ClusteringResult",
    "DEFAULT_DISCOVERY_CACHE_DIR",
    "DEFAULT_DISCOVERY_LOCAL_DIR",
    "DEFAULT_DISCOVERY_MODEL_NAME",
    "DiscoveryReport",
    "DiscoveryEncoder",
    "EvaluationReport",
    "FilterReport",
    "PhraseFilter",
    "IntrinsicMetrics",
    "CoverageReport",
    "SemanticMetrics",
    "ManualLabel",
    "ManualMetrics",
    "ProductDiscoveryReport",
    "run_discovery_v3_per_product",
    "ResidualExtractor",
    "ResidualResult",
    "DiscoverySnapshotCache",
    "compute_product_snapshot_key",
    "load_product_report",
    "ReviewClusterer",
    "ReviewRepresentation",
    "ReviewRepresentationBatch",
    "PerProductDiscoveryReport",
    "PerProductEvaluationReport",
    "PerProductPhraseDiscovery",
    "PhraseClusterSummary",
    "run_discovery",
]
