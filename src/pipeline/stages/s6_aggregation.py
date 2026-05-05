from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.reference import e2e


def run_stage(
    reviews: list[Any],
    sentiment_by_pair: dict[tuple[str, str], dict[str, float]],
    aspect_by_id_by_category: dict[str, dict[str, Any]],
    discovery_by_product: dict[int, Any],
) -> dict[str, Any]:
    aggregated = e2e()._aggregate_product_scores(
        reviews,
        sentiment_by_pair,
        aspect_by_id_by_category,
        discovery_by_product,
    )
    by_product: dict[int, list[Any]] = defaultdict(list)
    for review in reviews:
        by_product[int(review.nm_id)].append(review)

    rows: list[dict[str, Any]] = []
    for nm_id, payload in sorted(aggregated.items()):
        result = payload["raw"]
        product_reviews = by_product[int(nm_id)]
        contributing_by_aspect: dict[str, list[str]] = defaultdict(list)
        for review in product_reviews:
            for aspect_id in sorted(review.vocab_aspect_ids):
                aspect = aspect_by_id_by_category[review.category_id].get(aspect_id)
                name = aspect.canonical_name if aspect is not None else aspect_id
                key = (review.review_id, f"vocab::{aspect_id}")
                if key in sentiment_by_pair:
                    contributing_by_aspect[f"vocab::{aspect_id}::{name}"].append(review.review_id)
            discovery = discovery_by_product.get(int(nm_id))
            if discovery:
                for cluster_id in sorted(review.discovery_cluster_ids):
                    cluster = discovery.clusters.get(cluster_id)
                    key = (review.review_id, f"discovery::{nm_id}::{cluster_id}")
                    if cluster is not None and key in sentiment_by_pair:
                        contributing_by_aspect[f"discovery::{cluster_id}::{cluster.medoid}"].append(review.review_id)

        for key, score in sorted(result.aspects.items()):
            parts = key.split("::", 2)
            source = parts[0] if len(parts) > 1 else "vocab"
            aspect_name = parts[2] if len(parts) == 3 else key
            review_ids = sorted(set(contributing_by_aspect.get(key, [])))
            rows.append(
                {
                    "nm_id": int(nm_id),
                    "aspect_name": aspect_name,
                    "aspect_source": source,
                    "n_reviews_contributing": int(len(review_ids)),
                    "contributing_review_ids": review_ids,
                    "raw_mean_rating": float(score.raw_mean),
                    "shrunken_rating": float(score.score),
                    "variance": float(score.controversy) ** 2,
                    "shrinkage_strength": float(score.score) - float(score.raw_mean),
                    "gold_rating": np.nan,
                    "abs_error": np.nan,
                }
            )
    return {
        "aggregated": aggregated,
        "product_aggregates": pd.DataFrame(rows),
    }
