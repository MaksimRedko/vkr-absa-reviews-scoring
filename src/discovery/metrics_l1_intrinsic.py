from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import silhouette_score


@dataclass(slots=True)
class IntrinsicMetrics:
    cohesion: float
    separation: float
    silhouette: Optional[float]
    avg_concentration: float
    n_clusters_evaluated: int


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (matrix / norms).astype(np.float32, copy=False)


def compute_cohesion(cluster_phrases: list[str], encoder) -> float:
    if len(cluster_phrases) <= 1:
        return 1.0 if cluster_phrases else 0.0
    embeddings = _l2_normalize(encoder.encode(cluster_phrases))
    return compute_cohesion_from_embeddings(embeddings)


def compute_cohesion_from_embeddings(embeddings: np.ndarray) -> float:
    if len(embeddings) <= 1:
        return 1.0 if len(embeddings) else 0.0
    embeddings = _l2_normalize(embeddings)
    sims = embeddings @ embeddings.T
    mask = ~np.eye(len(embeddings), dtype=bool)
    return float(sims[mask].mean())


def compute_separation(clusters: dict[int, list[str]], encoder) -> float:
    if len(clusters) < 2:
        return 0.0
    phrases: list[str] = []
    offsets: dict[int, tuple[int, int]] = {}
    for cluster_id, cluster_phrases in clusters.items():
        start = len(phrases)
        phrases.extend(cluster_phrases)
        offsets[cluster_id] = (start, len(phrases))
    embeddings = _l2_normalize(encoder.encode(phrases))
    embeddings_by_cluster = {
        cluster_id: embeddings[start:end]
        for cluster_id, (start, end) in offsets.items()
    }
    return compute_separation_from_embeddings(embeddings_by_cluster)


def compute_separation_from_embeddings(clusters: dict[int, np.ndarray]) -> float:
    if len(clusters) < 2:
        return 0.0
    centroids: list[np.ndarray] = []
    for embeddings in clusters.values():
        if len(embeddings) == 0:
            continue
        centroid = embeddings.mean(axis=0, keepdims=True)
        centroids.append(_l2_normalize(centroid)[0])
    if len(centroids) < 2:
        return 0.0
    matrix = np.vstack(centroids)
    sims = matrix @ matrix.T
    mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
    return float(sims[mask].mean())


def compute_silhouette(clusters: dict[int, list[str]], encoder) -> float | None:
    if len(clusters) < 2:
        return None
    phrases: list[str] = []
    labels: list[int] = []
    for cluster_id, cluster_phrases in clusters.items():
        phrases.extend(cluster_phrases)
        labels.extend([cluster_id] * len(cluster_phrases))
    embeddings = _l2_normalize(encoder.encode(phrases))
    return compute_silhouette_from_embeddings(embeddings, labels)


def compute_silhouette_from_embeddings(embeddings: np.ndarray, labels: list[int]) -> float | None:
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(labels) <= len(unique_labels):
        return None
    return float(silhouette_score(_l2_normalize(embeddings), labels, metric="cosine"))


def compute_phrase_concentration(cluster_top_phrases: list[tuple[str, int]]) -> float:
    if not cluster_top_phrases:
        return 0.0
    weights = [max(0, int(weight)) for _phrase, weight in cluster_top_phrases]
    total = sum(weights)
    if total <= 0:
        return 0.0
    return float(sum(weights[:3])) / float(total)


def compute_intrinsic_metrics(
    *,
    embeddings_by_cluster: dict[int, np.ndarray],
    top_phrases_by_cluster: dict[int, list[tuple[str, int]]],
) -> IntrinsicMetrics:
    cluster_ids = sorted(embeddings_by_cluster)
    if not cluster_ids:
        return IntrinsicMetrics(
            cohesion=0.0,
            separation=0.0,
            silhouette=None,
            avg_concentration=0.0,
            n_clusters_evaluated=0,
        )
    cohesions = [
        compute_cohesion_from_embeddings(embeddings_by_cluster[cluster_id])
        for cluster_id in cluster_ids
    ]
    concentrations = [
        compute_phrase_concentration(top_phrases_by_cluster.get(cluster_id, []))
        for cluster_id in cluster_ids
    ]
    all_embeddings = np.vstack([embeddings_by_cluster[cluster_id] for cluster_id in cluster_ids])
    labels = [
        cluster_id
        for cluster_id in cluster_ids
        for _ in range(len(embeddings_by_cluster[cluster_id]))
    ]
    return IntrinsicMetrics(
        cohesion=float(np.mean(cohesions)),
        separation=compute_separation_from_embeddings(embeddings_by_cluster),
        silhouette=compute_silhouette_from_embeddings(all_embeddings, labels),
        avg_concentration=float(np.mean(concentrations)),
        n_clusters_evaluated=len(cluster_ids),
    )
