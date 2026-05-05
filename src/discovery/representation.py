from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.discovery.encoder import DEFAULT_DISCOVERY_EMBEDDING_DIM, DiscoveryEncoder
from src.discovery.residual_extractor import ResidualResult


@dataclass(slots=True)
class ReviewRepresentationBatch:
    review_ids: list[str]
    embeddings: np.ndarray
    excluded_review_ids: list[str]


class ReviewRepresentation:
    def build(
        self,
        residuals: list[ResidualResult],
        encoder: DiscoveryEncoder,
    ) -> ReviewRepresentationBatch:
        review_ids: list[str] = []
        excluded_review_ids: list[str] = []
        flattened_phrases: list[str] = []
        phrase_counts: list[int] = []

        for residual in residuals:
            phrases = [phrase.strip() for phrase in residual.residual_phrases if phrase.strip()]
            if not phrases:
                excluded_review_ids.append(residual.review_id)
                continue

            review_ids.append(residual.review_id)
            phrase_counts.append(len(phrases))
            flattened_phrases.extend(phrases)

        if not review_ids:
            embedding_dim = int(getattr(encoder, "embedding_dim", DEFAULT_DISCOVERY_EMBEDDING_DIM))
            return ReviewRepresentationBatch(
                review_ids=[],
                embeddings=np.empty((0, embedding_dim), dtype=np.float32),
                excluded_review_ids=excluded_review_ids,
            )

        phrase_embeddings = encoder.encode(flattened_phrases)
        if phrase_embeddings.shape[0] != len(flattened_phrases):
            raise RuntimeError(
                f"Unexpected phrase embedding count {phrase_embeddings.shape[0]}; "
                f"expected {len(flattened_phrases)}."
            )

        review_embeddings: list[np.ndarray] = []
        offset = 0
        for phrase_count in phrase_counts:
            review_matrix = phrase_embeddings[offset : offset + phrase_count]
            offset += phrase_count
            review_vector = review_matrix.mean(axis=0)
            norm = float(np.linalg.norm(review_vector))
            if norm > 0.0:
                review_vector = review_vector / norm
            review_embeddings.append(review_vector.astype(np.float32, copy=False))

        embeddings = np.vstack(review_embeddings).astype(np.float32, copy=False)
        return ReviewRepresentationBatch(
            review_ids=review_ids,
            embeddings=embeddings,
            excluded_review_ids=excluded_review_ids,
        )
