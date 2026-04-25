from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re

import pymorphy3

from src.discovery.clusterer import ClusteringResult
from src.discovery.residual_extractor import ResidualResult
from src.vocabulary.loader import Vocabulary

_TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)


@dataclass(slots=True)
class EvaluationReport:
    n_clusters: int
    n_clean_clusters: int
    coverage_via_clustering: float
    purity_per_cluster: dict[int, float]
    dominant_aspect_per_cluster: dict[int, str]
    noise_rate: float
    n_excluded_reviews: int


class ClusterEvaluator:
    def __init__(
        self,
        *,
        purity_threshold: float = 0.7,
        morph: pymorphy3.MorphAnalyzer | None = None,
    ) -> None:
        self.purity_threshold = float(purity_threshold)
        self._morph = morph or pymorphy3.MorphAnalyzer()

    def evaluate(
        self,
        clustering: ClusteringResult,
        residuals: list[ResidualResult],
        gold_labels: Mapping[str, object],
        vocabulary: Vocabulary,
    ) -> EvaluationReport:
        residual_by_review_id = {residual.review_id: residual for residual in residuals}
        n_excluded_reviews = sum(
            1
            for residual in residuals
            if not any(phrase.strip() for phrase in residual.residual_phrases)
        )

        vocabulary_lemma_sets = [
            self._build_vocabulary_aspect_lemmas(aspect.canonical_name, aspect.synonyms)
            for aspect in vocabulary.aspects
        ]
        uncovered_gold_by_review = {
            str(review_id): self._extract_uncovered_gold_aspects(raw_labels, vocabulary_lemma_sets)
            for review_id, raw_labels in gold_labels.items()
        }

        reviews_by_cluster: dict[int, list[str]] = {}
        for review_id, cluster_id in clustering.review_to_cluster.items():
            if cluster_id == -1:
                continue
            reviews_by_cluster.setdefault(int(cluster_id), []).append(str(review_id))

        purity_per_cluster: dict[int, float] = {}
        dominant_aspect_per_cluster: dict[int, str] = {}
        clean_cluster_ids: set[int] = set()

        for cluster_id in sorted(reviews_by_cluster):
            review_ids = reviews_by_cluster[cluster_id]
            aspect_counts: dict[str, int] = {}
            for review_id in review_ids:
                for aspect_name in uncovered_gold_by_review.get(review_id, set()):
                    aspect_counts[aspect_name] = aspect_counts.get(aspect_name, 0) + 1

            if not aspect_counts:
                purity_per_cluster[cluster_id] = 0.0
                dominant_aspect_per_cluster[cluster_id] = ""
                continue

            dominant_aspect, dominant_count = min(
                aspect_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            purity = float(dominant_count) / float(len(review_ids))

            purity_per_cluster[cluster_id] = purity
            dominant_aspect_per_cluster[cluster_id] = dominant_aspect
            if purity >= self.purity_threshold:
                clean_cluster_ids.add(cluster_id)

        total_uncovered_gold = sum(len(aspects) for aspects in uncovered_gold_by_review.values())
        matched_uncovered_gold = 0
        for review_id, cluster_id in clustering.review_to_cluster.items():
            if cluster_id not in clean_cluster_ids:
                continue
            dominant_aspect = dominant_aspect_per_cluster.get(int(cluster_id), "")
            if not dominant_aspect:
                continue
            if dominant_aspect in uncovered_gold_by_review.get(str(review_id), set()):
                matched_uncovered_gold += 1

        coverage_via_clustering = (
            float(matched_uncovered_gold) / float(total_uncovered_gold)
            if total_uncovered_gold
            else 0.0
        )

        return EvaluationReport(
            n_clusters=clustering.n_clusters,
            n_clean_clusters=len(clean_cluster_ids),
            coverage_via_clustering=coverage_via_clustering,
            purity_per_cluster=purity_per_cluster,
            dominant_aspect_per_cluster=dominant_aspect_per_cluster,
            noise_rate=clustering.noise_rate,
            n_excluded_reviews=n_excluded_reviews,
        )

    def _extract_uncovered_gold_aspects(
        self,
        raw_labels: object,
        vocabulary_lemma_sets: list[set[str]],
    ) -> set[str]:
        aspect_names = self._normalize_gold_aspect_names(raw_labels)
        return {
            aspect_name
            for aspect_name in aspect_names
            if not self._is_covered_by_vocabulary(aspect_name, vocabulary_lemma_sets)
        }

    def _normalize_gold_aspect_names(self, raw_labels: object) -> set[str]:
        if raw_labels is None:
            return set()
        if isinstance(raw_labels, Mapping):
            return {
                str(aspect_name).strip()
                for aspect_name in raw_labels.keys()
                if str(aspect_name).strip()
            }
        if isinstance(raw_labels, str):
            value = raw_labels.strip()
            return {value} if value else set()
        if isinstance(raw_labels, Iterable):
            return {
                str(aspect_name).strip()
                for aspect_name in raw_labels
                if str(aspect_name).strip()
            }
        value = str(raw_labels).strip()
        return {value} if value else set()

    def _is_covered_by_vocabulary(
        self,
        aspect_name: str,
        vocabulary_lemma_sets: list[set[str]],
    ) -> bool:
        aspect_lemmas = self._lemmatize_text(aspect_name)
        if not aspect_lemmas:
            return False
        return any(aspect_lemmas & vocab_lemmas for vocab_lemmas in vocabulary_lemma_sets)

    def _build_vocabulary_aspect_lemmas(
        self,
        canonical_name: str,
        synonyms: list[str],
    ) -> set[str]:
        lemmas = self._lemmatize_text(canonical_name)
        for synonym in synonyms:
            lemmas.update(self._lemmatize_text(synonym))
        return lemmas

    def _lemmatize_text(self, text: str) -> set[str]:
        tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
        return {self._lemmatize_token(token) for token in tokens if token}

    def _lemmatize_token(self, token: str) -> str:
        parses = self._morph.parse(token)
        if not parses:
            return token.lower()
        return str(parses[0].normal_form or token).lower()
