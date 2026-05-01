from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.discovery.encoder import DiscoveryEncoder
from src.pipeline.reference import e2e
from src.pipeline.stages.common import stable_id
from src.pipeline.stages.s2_encoding import _encode_cached


def load_frozen_discovery(
    discovery_dir,
    encoder: DiscoveryEncoder,
    cache: dict[str, np.ndarray],
) -> dict[int, Any]:
    return e2e()._load_discovery(discovery_dir, encoder, cache)


def bind_unmatched_to_clusters(
    reviews: list[Any],
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    discovery_by_product: dict[int, Any],
    encoder: DiscoveryEncoder,
    cache: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    unmatched = matches[matches["is_unmatched"] == True][["candidate_id"]].drop_duplicates().copy()
    if unmatched.empty:
        return pd.DataFrame(
            columns=[
                "binding_id",
                "review_id",
                "candidate_id",
                "cluster_id",
                "cluster_score",
            ]
        )
    unmatched = unmatched.merge(
        candidates[
            [
                "candidate_id",
                "review_id",
                "text",
                "start_offset",
                "end_offset",
            ]
        ],
        on="candidate_id",
        how="inner",
    )
    rows_by_review = {
        str(review_id): group.copy()
        for review_id, group in unmatched.groupby("review_id", sort=False)
    }
    binding_rows: list[dict[str, Any]] = []
    for review in reviews:
        discovery = discovery_by_product.get(review.nm_id)
        if not discovery:
            continue
        cluster_ids = sorted(discovery.clusters)
        if not cluster_ids:
            continue
        cluster_matrix = np.vstack([discovery.clusters[cid].centroid for cid in cluster_ids]).astype(np.float32)
        rows = rows_by_review.get(str(review.review_id))
        if rows is None or rows.empty or cluster_matrix.size == 0:
            continue
        phrase_texts = rows["text"].astype(str).tolist()
        vectors = _encode_cached(encoder, phrase_texts, cache)
        for row in rows.itertuples(index=False):
            phrase = str(row.text)
            vector = vectors.get(phrase)
            if vector is None:
                continue
            scores = cluster_matrix @ vector
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score >= threshold:
                cluster_id = int(cluster_ids[best_idx])
                review.discovery_cluster_ids.add(cluster_id)
                binding_rows.append(
                    {
                        "binding_id": stable_id(
                            review.review_id,
                            "discovery",
                            row.candidate_id,
                            int(row.start_offset),
                            int(row.end_offset),
                            cluster_id,
                        ),
                        "review_id": str(review.review_id),
                        "candidate_id": str(row.candidate_id),
                        "start_offset": int(row.start_offset),
                        "end_offset": int(row.end_offset),
                        "cluster_id": cluster_id,
                        "cluster_score": best_score,
                    }
                )
    return pd.DataFrame(
        binding_rows,
        columns=[
            "binding_id",
            "review_id",
            "candidate_id",
            "start_offset",
            "end_offset",
            "cluster_id",
            "cluster_score",
        ],
    )


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
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    discovery_dir,
    encoder: DiscoveryEncoder,
    cache: dict[str, np.ndarray],
    *,
    phrase_to_cluster_threshold: float = 0.5,
) -> dict[str, Any]:
    discovery_by_product = load_frozen_discovery(discovery_dir, encoder, cache)
    discovery_candidate_bindings = bind_unmatched_to_clusters(
        reviews,
        candidates,
        matches,
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
        "discovery_candidate_bindings": discovery_candidate_bindings,
    }
