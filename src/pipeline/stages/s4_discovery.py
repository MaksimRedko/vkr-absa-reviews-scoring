from __future__ import annotations

from typing import Any

import numpy as np

from src.discovery.encoder import DiscoveryEncoder
from src.pipeline.reference import e2e
from src.pipeline.stages.s2_encoding import _encode_cached


def load_frozen_discovery(
    discovery_dir,
    encoder: DiscoveryEncoder,
    cache: dict[str, np.ndarray],
) -> dict[int, Any]:
    return e2e()._load_discovery(discovery_dir, encoder, cache)


def bind_unmatched_to_clusters(
    reviews: list[Any],
    discovery_by_product: dict[int, Any],
    encoder: DiscoveryEncoder,
    cache: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
) -> None:
    for review in reviews:
        discovery = discovery_by_product.get(review.nm_id)
        if not discovery:
            continue
        cluster_ids = sorted(discovery.clusters)
        if not cluster_ids:
            continue
        cluster_matrix = np.vstack([discovery.clusters[cid].centroid for cid in cluster_ids]).astype(np.float32)
        if not review.unmatched_phrases or cluster_matrix.size == 0:
            continue
        vectors = _encode_cached(encoder, review.unmatched_phrases, cache)
        for phrase in review.unmatched_phrases:
            vector = vectors.get(phrase)
            if vector is None:
                continue
            scores = cluster_matrix @ vector
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score >= threshold:
                review.discovery_cluster_ids.add(int(cluster_ids[best_idx]))


def cluster_payloads(discovery_by_product: dict[int, Any]) -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    for nm_id, discovery in sorted(discovery_by_product.items()):
        clusters = []
        for cluster in sorted(discovery.clusters.values(), key=lambda item: item.cluster_id):
            clusters.append(
                {
                    "cluster_id": int(cluster.cluster_id),
                    "candidate_ids": [],
                    "top_phrases": [
                        [phrase, int(cluster.top_phrase_weights.get(phrase, 1))]
                        for phrase in cluster.top_phrases
                    ],
                    "medoid_phrase": cluster.medoid,
                    "cohesion": None,
                    "matched_to_gold_aspect": next(iter(cluster.gold_matches), None),
                    "is_novel": not bool(cluster.gold_matches),
                    "gold_matches": cluster.gold_matches,
                }
            )
        payloads[nm_id] = {
            "nm_id": int(nm_id),
            "category_id": discovery.category_id,
            "n_unmatched_candidates": None,
            "n_clusters": len(clusters),
            "noise_rate": None,
            "filter_applied": True,
            "filter_report": {},
            "clusters": clusters,
        }
    return payloads


def centroid_arrays(discovery_by_product: dict[int, Any]) -> dict[int, np.ndarray]:
    arrays: dict[int, np.ndarray] = {}
    for nm_id, discovery in sorted(discovery_by_product.items()):
        cluster_ids = sorted(discovery.clusters)
        if cluster_ids:
            arrays[nm_id] = np.vstack([discovery.clusters[cid].centroid for cid in cluster_ids]).astype(np.float32)
        else:
            arrays[nm_id] = np.empty((0, 0), dtype=np.float32)
    return arrays


def run_stage(
    reviews: list[Any],
    discovery_dir,
    encoder: DiscoveryEncoder,
    cache: dict[str, np.ndarray],
    *,
    phrase_to_cluster_threshold: float = 0.5,
) -> dict[str, Any]:
    discovery_by_product = load_frozen_discovery(discovery_dir, encoder, cache)
    bind_unmatched_to_clusters(
        reviews,
        discovery_by_product,
        encoder,
        cache,
        threshold=phrase_to_cluster_threshold,
    )
    e2e()._compute_discovery_gold_matches(reviews, discovery_by_product, encoder, cache)
    return {
        "discovery_by_product": discovery_by_product,
        "cluster_payloads": cluster_payloads(discovery_by_product),
        "centroid_arrays": centroid_arrays(discovery_by_product),
    }
