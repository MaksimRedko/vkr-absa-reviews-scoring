from __future__ import annotations

from dataclasses import dataclass

import hdbscan

from src.discovery.representation import ReviewRepresentationBatch


@dataclass(slots=True)
class ClusteringResult:
    review_to_cluster: dict[str, int]
    cluster_sizes: dict[int, int]
    n_clusters: int
    n_noise: int
    noise_rate: float


class ReviewClusterer:
    def __init__(
        self,
        *,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ) -> None:
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples)
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

    def cluster(self, batch: ReviewRepresentationBatch) -> ClusteringResult:
        n_reviews = len(batch.review_ids)
        if n_reviews == 0:
            return ClusteringResult(
                review_to_cluster={},
                cluster_sizes={},
                n_clusters=0,
                n_noise=0,
                noise_rate=0.0,
            )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
        )
        labels = clusterer.fit_predict(batch.embeddings)

        review_to_cluster = {
            review_id: int(label)
            for review_id, label in zip(batch.review_ids, labels, strict=True)
        }

        cluster_sizes: dict[int, int] = {}
        n_noise = 0
        for label in labels:
            cluster_id = int(label)
            if cluster_id == -1:
                n_noise += 1
                continue
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        return ClusteringResult(
            review_to_cluster=review_to_cluster,
            cluster_sizes=cluster_sizes,
            n_clusters=len(cluster_sizes),
            n_noise=n_noise,
            noise_rate=float(n_noise) / float(n_reviews),
        )
