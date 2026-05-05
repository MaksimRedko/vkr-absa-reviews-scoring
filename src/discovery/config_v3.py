from __future__ import annotations

PHRASE_FILTER_ENABLED = True
COSINE_THRESHOLD_PRIMARY = 0.65
COSINE_THRESHOLDS_SENSITIVITY = [0.60, 0.65, 0.70]
NOVEL_COHESION_MIN = 0.60
MIN_UNIQUE_PHRASES_TO_CLUSTER = 30
HDBSCAN_PARAMS = {
    "min_cluster_size": 5,
    "min_samples": 3,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}
TOP_N_PHRASES_PER_CLUSTER = 10
ENCODER_MODEL = "ai-forever/sbert_large_nlu_ru"
ENCODER_BATCH_SIZE = 8
