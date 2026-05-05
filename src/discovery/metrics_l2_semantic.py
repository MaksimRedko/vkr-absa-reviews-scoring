from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.discovery.config_v3 import (
    COSINE_THRESHOLD_PRIMARY,
    COSINE_THRESHOLDS_SENSITIVITY,
    NOVEL_COHESION_MIN,
)
from src.discovery.metrics_l1_intrinsic import compute_cohesion_from_embeddings


@dataclass(slots=True)
class CoverageReport:
    threshold: float
    coverage: float
    matches: dict[str, tuple[int, float]]
    unmatched_aspects: list[str]


@dataclass(slots=True)
class SemanticMetrics:
    primary_threshold: float
    coverage_primary: CoverageReport
    sensitivity: dict[float, CoverageReport]
    avg_soft_purity: float
    n_novel_clusters: int
    novel_cluster_ids: list[int]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (matrix / norms).astype(np.float32, copy=False)


def _aspect_name(aspect: object) -> str:
    if hasattr(aspect, "canonical_name"):
        return str(getattr(aspect, "canonical_name")).strip()
    return str(aspect).strip()


def _aspect_terms(aspect: object) -> list[str]:
    if hasattr(aspect, "canonical_name"):
        terms = [str(getattr(aspect, "canonical_name")).strip()]
        terms.extend(str(item).strip() for item in getattr(aspect, "synonyms", []) or [])
        return [term for term in terms if term]
    value = str(aspect).strip()
    return [value] if value else []


def encode_gold_centroids(gold_aspects: Iterable[object], encoder) -> dict[str, np.ndarray]:
    aspects = [aspect for aspect in gold_aspects if _aspect_name(aspect)]
    if not aspects:
        return {}

    terms: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    for aspect in aspects:
        name = _aspect_name(aspect)
        aspect_terms = _aspect_terms(aspect)
        start = len(terms)
        terms.extend(aspect_terms)
        offsets[name] = (start, len(terms))

    embeddings = _l2_normalize(encoder.encode(terms))
    centroids: dict[str, np.ndarray] = {}
    for name, (start, end) in offsets.items():
        centroid = embeddings[start:end].mean(axis=0, keepdims=True)
        centroids[name] = _l2_normalize(centroid)[0]
    return centroids


def compute_aspect_match_coverage(
    clusters: dict[int, list[str]],
    uncovered_gold_aspects: Iterable[object],
    encoder,
    threshold: float,
) -> CoverageReport:
    gold_centroids = encode_gold_centroids(uncovered_gold_aspects, encoder)
    phrase_embeddings = _encode_cluster_phrases(clusters, encoder)
    return compute_aspect_match_coverage_from_embeddings(
        phrase_embeddings_by_cluster=phrase_embeddings,
        gold_centroids=gold_centroids,
        threshold=threshold,
    )


def compute_aspect_match_coverage_from_embeddings(
    *,
    phrase_embeddings_by_cluster: dict[int, np.ndarray],
    gold_centroids: dict[str, np.ndarray],
    threshold: float,
) -> CoverageReport:
    if not gold_centroids:
        return CoverageReport(
            threshold=threshold,
            coverage=1.0,
            matches={},
            unmatched_aspects=[],
        )

    matches: dict[str, tuple[int, float]] = {}
    unmatched: list[str] = []
    for aspect_name, centroid in gold_centroids.items():
        best_cluster = -1
        best_score = -1.0
        for cluster_id, embeddings in phrase_embeddings_by_cluster.items():
            if len(embeddings) == 0:
                continue
            score = float(np.max(_l2_normalize(embeddings) @ centroid))
            if score > best_score:
                best_score = score
                best_cluster = cluster_id
        if best_score >= threshold:
            matches[aspect_name] = (best_cluster, best_score)
        else:
            unmatched.append(aspect_name)

    coverage = float(len(matches)) / float(len(gold_centroids))
    return CoverageReport(
        threshold=threshold,
        coverage=coverage,
        matches=matches,
        unmatched_aspects=unmatched,
    )


def compute_soft_purity(
    cluster_phrases: list[str],
    gold_aspects: Iterable[object],
    encoder,
    threshold: float,
) -> float:
    gold_centroids = encode_gold_centroids(gold_aspects, encoder)
    if not cluster_phrases:
        return 0.0
    embeddings = _l2_normalize(encoder.encode(cluster_phrases))
    return compute_soft_purity_from_embeddings(
        phrase_embeddings=embeddings,
        gold_centroids=gold_centroids,
        threshold=threshold,
    )


def compute_soft_purity_from_embeddings(
    *,
    phrase_embeddings: np.ndarray,
    gold_centroids: dict[str, np.ndarray],
    threshold: float,
) -> float:
    if len(phrase_embeddings) == 0 or not gold_centroids:
        return 0.0

    aspect_names = list(gold_centroids)
    aspect_matrix = np.vstack([gold_centroids[name] for name in aspect_names])
    scores = _l2_normalize(phrase_embeddings) @ aspect_matrix.T
    best_indices = np.argmax(scores, axis=1)
    best_scores = np.max(scores, axis=1)

    counts: dict[str, int] = {}
    for index, score in zip(best_indices, best_scores, strict=True):
        if float(score) >= threshold:
            name = aspect_names[int(index)]
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return 0.0
    return float(max(counts.values())) / float(len(phrase_embeddings))


def compute_novel_aspect_indicator(
    cluster_phrases: list[str],
    gold_centroids: dict[str, np.ndarray],
    encoder,
    threshold: float,
) -> bool:
    if not cluster_phrases:
        return False
    embeddings = _l2_normalize(encoder.encode(cluster_phrases))
    return compute_novel_aspect_indicator_from_embeddings(
        phrase_embeddings=embeddings,
        gold_centroids=gold_centroids,
        threshold=threshold,
    )


def compute_novel_aspect_indicator_from_embeddings(
    *,
    phrase_embeddings: np.ndarray,
    gold_centroids: dict[str, np.ndarray],
    threshold: float,
) -> bool:
    if len(phrase_embeddings) == 0:
        return False
    cohesion = compute_cohesion_from_embeddings(phrase_embeddings)
    if cohesion < NOVEL_COHESION_MIN:
        return False
    if not gold_centroids:
        return True
    aspect_matrix = np.vstack(list(gold_centroids.values()))
    max_score = float(np.max(_l2_normalize(phrase_embeddings) @ aspect_matrix.T))
    return max_score < threshold


def compute_semantic_metrics(
    *,
    phrase_embeddings_by_cluster: dict[int, np.ndarray],
    gold_centroids: dict[str, np.ndarray],
    thresholds: list[float] | None = None,
    primary_threshold: float = COSINE_THRESHOLD_PRIMARY,
) -> SemanticMetrics:
    thresholds = thresholds or list(COSINE_THRESHOLDS_SENSITIVITY)
    sensitivity = {
        threshold: compute_aspect_match_coverage_from_embeddings(
            phrase_embeddings_by_cluster=phrase_embeddings_by_cluster,
            gold_centroids=gold_centroids,
            threshold=threshold,
        )
        for threshold in thresholds
    }
    coverage_primary = sensitivity.get(primary_threshold) or compute_aspect_match_coverage_from_embeddings(
        phrase_embeddings_by_cluster=phrase_embeddings_by_cluster,
        gold_centroids=gold_centroids,
        threshold=primary_threshold,
    )
    purities = [
        compute_soft_purity_from_embeddings(
            phrase_embeddings=embeddings,
            gold_centroids=gold_centroids,
            threshold=primary_threshold,
        )
        for embeddings in phrase_embeddings_by_cluster.values()
    ]
    novel_cluster_ids = [
        cluster_id
        for cluster_id, embeddings in phrase_embeddings_by_cluster.items()
        if compute_novel_aspect_indicator_from_embeddings(
            phrase_embeddings=embeddings,
            gold_centroids=gold_centroids,
            threshold=primary_threshold,
        )
    ]
    return SemanticMetrics(
        primary_threshold=primary_threshold,
        coverage_primary=coverage_primary,
        sensitivity=sensitivity,
        avg_soft_purity=float(np.mean(purities)) if purities else 0.0,
        n_novel_clusters=len(novel_cluster_ids),
        novel_cluster_ids=sorted(novel_cluster_ids),
    )


def _encode_cluster_phrases(clusters: dict[int, list[str]], encoder) -> dict[int, np.ndarray]:
    phrases: list[str] = []
    offsets: dict[int, tuple[int, int]] = {}
    for cluster_id, cluster_phrases in clusters.items():
        start = len(phrases)
        phrases.extend(cluster_phrases)
        offsets[cluster_id] = (start, len(phrases))
    if not phrases:
        return {cluster_id: np.empty((0, 0), dtype=np.float32) for cluster_id in clusters}
    embeddings = _l2_normalize(encoder.encode(phrases))
    return {
        cluster_id: embeddings[start:end]
        for cluster_id, (start, end) in offsets.items()
    }
