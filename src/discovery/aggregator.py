from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random

from src.discovery.clusterer import ClusteringResult
from src.discovery.residual_extractor import ResidualResult


@dataclass(slots=True)
class ClusterSummary:
    cluster_id: int
    n_reviews: int
    top_phrases: list[tuple[str, int]]
    sample_review_ids: list[str]


class ClusterAggregator:
    def __init__(
        self,
        *,
        top_k_phrases: int = 20,
        sample_size: int = 5,
        random_seed: int = 42,
    ) -> None:
        self.top_k_phrases = int(top_k_phrases)
        self.sample_size = int(sample_size)
        self.random_seed = int(random_seed)

    def aggregate(
        self,
        residuals: list[ResidualResult],
        clustering: ClusteringResult,
    ) -> list[ClusterSummary]:
        residual_by_review_id = {residual.review_id: residual for residual in residuals}
        reviews_by_cluster: dict[int, list[str]] = defaultdict(list)

        for review_id, cluster_id in clustering.review_to_cluster.items():
            if cluster_id == -1:
                continue
            if review_id not in residual_by_review_id:
                continue
            reviews_by_cluster[int(cluster_id)].append(review_id)

        summaries: list[ClusterSummary] = []
        for cluster_id in sorted(reviews_by_cluster):
            review_ids = reviews_by_cluster[cluster_id]
            phrase_counter: Counter[str] = Counter()
            for review_id in review_ids:
                phrase_counter.update(residual_by_review_id[review_id].residual_phrases)

            top_phrases = sorted(
                phrase_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )[: self.top_k_phrases]

            summaries.append(
                ClusterSummary(
                    cluster_id=cluster_id,
                    n_reviews=len(review_ids),
                    top_phrases=top_phrases,
                    sample_review_ids=self._sample_review_ids(cluster_id, review_ids),
                )
            )

        return summaries

    def _sample_review_ids(self, cluster_id: int, review_ids: list[str]) -> list[str]:
        if len(review_ids) <= self.sample_size:
            return list(review_ids)

        rng = random.Random(self.random_seed + int(cluster_id))
        return rng.sample(review_ids, k=self.sample_size)
