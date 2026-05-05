from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Optional

import hdbscan
import numpy as np
import pymorphy3

from src.discovery.config_v3 import (
    COSINE_THRESHOLD_PRIMARY,
    COSINE_THRESHOLDS_SENSITIVITY,
    HDBSCAN_PARAMS,
    MIN_UNIQUE_PHRASES_TO_CLUSTER,
    TOP_N_PHRASES_PER_CLUSTER,
)
from src.discovery.encoder import DiscoveryEncoder
from src.discovery.metrics_l1_intrinsic import IntrinsicMetrics, compute_intrinsic_metrics
from src.discovery.metrics_l2_semantic import (
    CoverageReport,
    SemanticMetrics,
    compute_semantic_metrics,
    encode_gold_centroids,
)
from src.discovery.manual_eval import ManualMetrics
from src.discovery.phrase_filter import FilterReport, PhraseFilter
from src.discovery.residual_extractor import ResidualExtractor, ResidualResult
from src.schemas.models import ReviewInput
from src.vocabulary.loader import Vocabulary

_TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)


@dataclass(slots=True)
class ClusterSummary:
    cluster_id: int
    n_phrases: int
    total_weight: int
    top_phrases: list[tuple[str, int]]


@dataclass(slots=True)
class ProductDiscoveryReport:
    nm_id: int
    category_id: str
    n_reviews: int
    n_unique_residuals_before_filter: int
    n_unique_residuals_after_filter: int
    filter_report: Optional[FilterReport]
    cluster_summaries: list[ClusterSummary]
    n_clusters: int
    noise_rate: float
    intrinsic_metrics: IntrinsicMetrics
    semantic_metrics: SemanticMetrics
    manual_metrics: Optional[ManualMetrics]
    metadata: dict[str, object]


@dataclass(slots=True)
class _PhraseRecord:
    key: str
    phrase: str
    weight: int
    lemmas: set[str]


class PerProductDiscoveryV3:
    def __init__(
        self,
        *,
        encoder: DiscoveryEncoder,
        residual_extractor: ResidualExtractor | None = None,
        phrase_filter: PhraseFilter | None = None,
        min_unique_phrases_to_cluster: int = MIN_UNIQUE_PHRASES_TO_CLUSTER,
        top_n_phrases_per_cluster: int = TOP_N_PHRASES_PER_CLUSTER,
        hdbscan_params: Mapping[str, object] | None = None,
        morph: pymorphy3.MorphAnalyzer | None = None,
    ) -> None:
        self.encoder = encoder
        self.residual_extractor = residual_extractor or ResidualExtractor()
        self.phrase_filter = phrase_filter or PhraseFilter()
        self.min_unique_phrases_to_cluster = int(min_unique_phrases_to_cluster)
        self.top_n_phrases_per_cluster = int(top_n_phrases_per_cluster)
        self.hdbscan_params = dict(HDBSCAN_PARAMS)
        if hdbscan_params:
            self.hdbscan_params.update(dict(hdbscan_params))
        self._morph = morph or pymorphy3.MorphAnalyzer()

    def run(
        self,
        *,
        nm_id: int,
        category_id: str,
        reviews: list[ReviewInput],
        gold_labels: Mapping[str, object],
        vocabulary: Vocabulary,
        apply_filter: bool = True,
    ) -> ProductDiscoveryReport:
        residuals = [
            self.residual_extractor.extract(
                review=review,
                category_id=category_id,
                vocabulary=vocabulary,
            )
            for review in reviews
        ]
        raw_phrases = [
            phrase.strip()
            for residual in residuals
            for phrase in residual.residual_phrases
            if phrase.strip()
        ]
        records_before_filter = self._build_phrase_records(raw_phrases)
        filter_report: FilterReport | None = None
        phrases_for_clustering = raw_phrases
        if apply_filter:
            phrases_for_clustering, filter_report = self.phrase_filter.filter(raw_phrases)

        phrase_records = self._build_phrase_records(phrases_for_clustering)
        uncovered_gold_aspects = self._extract_product_uncovered_gold(gold_labels, vocabulary)

        if len(phrase_records) < self.min_unique_phrases_to_cluster:
            return self._build_report(
                nm_id=nm_id,
                category_id=category_id,
                reviews=reviews,
                residuals=residuals,
                records_before_filter=records_before_filter,
                phrase_records=phrase_records,
                labels=[],
                embeddings=np.empty((0, self.encoder.embedding_dim), dtype=np.float32),
                cluster_summaries=[],
                filter_report=filter_report,
                uncovered_gold_aspects=uncovered_gold_aspects,
                skipped=True,
            )

        embeddings = self.encoder.encode([record.phrase for record in phrase_records])
        labels = self._cluster_embeddings(embeddings)
        cluster_summaries = self._summarize_clusters(phrase_records, labels)
        return self._build_report(
            nm_id=nm_id,
            category_id=category_id,
            reviews=reviews,
            residuals=residuals,
            records_before_filter=records_before_filter,
            phrase_records=phrase_records,
            labels=labels,
            embeddings=embeddings,
            cluster_summaries=cluster_summaries,
            filter_report=filter_report,
            uncovered_gold_aspects=uncovered_gold_aspects,
            skipped=False,
        )

    def _cluster_embeddings(self, embeddings: np.ndarray) -> list[int]:
        clusterer = hdbscan.HDBSCAN(**self.hdbscan_params)
        return [int(label) for label in clusterer.fit_predict(embeddings)]

    def _build_phrase_records(self, phrases: list[str]) -> list[_PhraseRecord]:
        counts: Counter[str] = Counter()
        surfaces_by_key: dict[str, Counter[str]] = {}
        lemmas_by_key: dict[str, set[str]] = {}

        for phrase in phrases:
            clean_phrase = phrase.strip()
            if not clean_phrase:
                continue
            lemmas = self._lemmatize_sequence(clean_phrase)
            if not lemmas:
                continue
            key = " ".join(lemmas)
            counts[key] += 1
            surfaces_by_key.setdefault(key, Counter())[clean_phrase] += 1
            lemmas_by_key[key] = set(lemmas)

        records: list[_PhraseRecord] = []
        for key, weight in counts.most_common():
            records.append(
                _PhraseRecord(
                    key=key,
                    phrase=surfaces_by_key[key].most_common(1)[0][0],
                    weight=int(weight),
                    lemmas=lemmas_by_key[key],
                )
            )
        return records

    def _summarize_clusters(
        self,
        phrase_records: list[_PhraseRecord],
        labels: list[int],
    ) -> list[ClusterSummary]:
        records_by_cluster: dict[int, list[_PhraseRecord]] = {}
        for record, label in zip(phrase_records, labels, strict=True):
            if label == -1:
                continue
            records_by_cluster.setdefault(label, []).append(record)

        summaries: list[ClusterSummary] = []
        for cluster_id in sorted(records_by_cluster):
            cluster_records = sorted(
                records_by_cluster[cluster_id],
                key=lambda record: (-record.weight, record.phrase),
            )
            summaries.append(
                ClusterSummary(
                    cluster_id=cluster_id,
                    n_phrases=len(cluster_records),
                    total_weight=sum(record.weight for record in cluster_records),
                    top_phrases=[
                        (record.phrase, record.weight)
                        for record in cluster_records[: self.top_n_phrases_per_cluster]
                    ],
                )
            )
        return summaries

    def _build_report(
        self,
        *,
        nm_id: int,
        category_id: str,
        reviews: list[ReviewInput],
        residuals: list[ResidualResult],
        records_before_filter: list[_PhraseRecord],
        phrase_records: list[_PhraseRecord],
        labels: list[int],
        embeddings: np.ndarray,
        cluster_summaries: list[ClusterSummary],
        filter_report: FilterReport | None,
        uncovered_gold_aspects: set[str],
        skipped: bool,
    ) -> ProductDiscoveryReport:
        n_noise = sum(1 for label in labels if label == -1)
        noise_rate = float(n_noise) / float(len(labels)) if labels else 0.0
        embeddings_by_cluster = self._embeddings_by_cluster(embeddings, labels)
        top_embeddings_by_cluster = self._top_embeddings_by_cluster(
            phrase_records=phrase_records,
            labels=labels,
            embeddings=embeddings,
            cluster_summaries=cluster_summaries,
        )
        intrinsic_metrics = compute_intrinsic_metrics(
            embeddings_by_cluster=embeddings_by_cluster,
            top_phrases_by_cluster={
                summary.cluster_id: summary.top_phrases for summary in cluster_summaries
            },
        )
        if top_embeddings_by_cluster:
            gold_centroids = encode_gold_centroids(sorted(uncovered_gold_aspects), self.encoder)
            semantic_metrics = compute_semantic_metrics(
                phrase_embeddings_by_cluster=top_embeddings_by_cluster,
                gold_centroids=gold_centroids,
                thresholds=list(COSINE_THRESHOLDS_SENSITIVITY),
                primary_threshold=COSINE_THRESHOLD_PRIMARY,
            )
        else:
            semantic_metrics = self._empty_semantic_metrics(uncovered_gold_aspects)
        metadata: dict[str, object] = {
            "nm_id": int(nm_id),
            "category_id": category_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": self.encoder.model_name_or_path,
            "n_reviews_with_residual": sum(
                1 for residual in residuals if any(p.strip() for p in residual.residual_phrases)
            ),
            "n_raw_residual_phrases": sum(len(residual.residual_phrases) for residual in residuals),
            "n_noise_phrases": n_noise,
            "skipped": skipped,
            "skipped_low_unique_phrases": skipped,
            "apply_filter": filter_report is not None,
            "min_unique_phrases_to_cluster": self.min_unique_phrases_to_cluster,
            "top_n_phrases_per_cluster": self.top_n_phrases_per_cluster,
            "hdbscan": dict(self.hdbscan_params),
            "cosine_threshold_primary": COSINE_THRESHOLD_PRIMARY,
            "cosine_thresholds_sensitivity": list(COSINE_THRESHOLDS_SENSITIVITY),
            "n_uncovered_gold_aspects": len(uncovered_gold_aspects),
            "uncovered_gold_aspects": sorted(uncovered_gold_aspects),
        }
        return ProductDiscoveryReport(
            nm_id=int(nm_id),
            category_id=category_id,
            n_reviews=len(reviews),
            n_unique_residuals_before_filter=len(records_before_filter),
            n_unique_residuals_after_filter=len(phrase_records),
            filter_report=filter_report,
            cluster_summaries=cluster_summaries,
            n_clusters=len(cluster_summaries),
            noise_rate=noise_rate,
            intrinsic_metrics=intrinsic_metrics,
            semantic_metrics=semantic_metrics,
            manual_metrics=None,
            metadata=metadata,
        )

    def _embeddings_by_cluster(
        self,
        embeddings: np.ndarray,
        labels: list[int],
    ) -> dict[int, np.ndarray]:
        return {
            cluster_id: embeddings[[index for index, label in enumerate(labels) if label == cluster_id]]
            for cluster_id in sorted(set(labels))
            if cluster_id != -1
        }

    def _top_embeddings_by_cluster(
        self,
        *,
        phrase_records: list[_PhraseRecord],
        labels: list[int],
        embeddings: np.ndarray,
        cluster_summaries: list[ClusterSummary],
    ) -> dict[int, np.ndarray]:
        if not labels or not cluster_summaries:
            return {}
        phrase_index_by_cluster: dict[int, dict[str, int]] = {}
        for index, (record, label) in enumerate(zip(phrase_records, labels, strict=True)):
            if label == -1:
                continue
            phrase_index_by_cluster.setdefault(label, {})[record.phrase] = index

        top_embeddings: dict[int, np.ndarray] = {}
        for summary in cluster_summaries:
            indices = [
                phrase_index_by_cluster.get(summary.cluster_id, {})[phrase]
                for phrase, _weight in summary.top_phrases
                if phrase in phrase_index_by_cluster.get(summary.cluster_id, {})
            ]
            top_embeddings[summary.cluster_id] = embeddings[indices] if indices else embeddings[:0]
        return top_embeddings

    def _empty_semantic_metrics(self, uncovered_gold_aspects: set[str]) -> SemanticMetrics:
        coverage = 1.0 if not uncovered_gold_aspects else 0.0
        sensitivity = {
            threshold: CoverageReport(
                threshold=threshold,
                coverage=coverage,
                matches={},
                unmatched_aspects=sorted(uncovered_gold_aspects),
            )
            for threshold in COSINE_THRESHOLDS_SENSITIVITY
        }
        return SemanticMetrics(
            primary_threshold=COSINE_THRESHOLD_PRIMARY,
            coverage_primary=sensitivity[COSINE_THRESHOLD_PRIMARY],
            sensitivity=sensitivity,
            avg_soft_purity=0.0,
            n_novel_clusters=0,
            novel_cluster_ids=[],
        )

    def _extract_product_uncovered_gold(
        self,
        gold_labels: Mapping[str, object],
        vocabulary: Vocabulary,
    ) -> set[str]:
        vocab_lemma_sets = [
            set(self._lemmatize_sequence(aspect.canonical_name)).union(
                *[set(self._lemmatize_sequence(synonym)) for synonym in aspect.synonyms]
            )
            for aspect in vocabulary.aspects
        ]
        uncovered: set[str] = set()
        for raw_labels in gold_labels.values():
            for aspect_name in self._normalize_gold_aspect_names(raw_labels):
                aspect_lemmas = set(self._lemmatize_sequence(aspect_name))
                if aspect_lemmas and not any(aspect_lemmas & vocab for vocab in vocab_lemma_sets):
                    uncovered.add(aspect_name)
        return uncovered

    def _normalize_gold_aspect_names(self, raw_labels: object) -> set[str]:
        if raw_labels is None:
            return set()
        if isinstance(raw_labels, Mapping):
            return {str(key).strip() for key in raw_labels.keys() if str(key).strip()}
        if isinstance(raw_labels, str):
            value = raw_labels.strip()
            return {value} if value else set()
        if isinstance(raw_labels, Iterable):
            return {str(item).strip() for item in raw_labels if str(item).strip()}
        value = str(raw_labels).strip()
        return {value} if value else set()

    def _lemmatize_sequence(self, text: str) -> list[str]:
        lemmas: list[str] = []
        for token in _TOKEN_RE.findall(text.lower()):
            parses = self._morph.parse(token)
            lemma = str(parses[0].normal_form or token).lower() if parses else token.lower()
            if lemma:
                lemmas.append(lemma)
        return lemmas


def run_discovery_v3_per_product(
    nm_id: int,
    category_id: str,
    reviews: list[ReviewInput],
    gold_labels: dict,
    vocabulary: Vocabulary,
    encoder: DiscoveryEncoder,
    apply_filter: bool = True,
) -> ProductDiscoveryReport:
    pipeline = PerProductDiscoveryV3(encoder=encoder)
    return pipeline.run(
        nm_id=nm_id,
        category_id=category_id,
        reviews=reviews,
        gold_labels=gold_labels,
        vocabulary=vocabulary,
        apply_filter=apply_filter,
    )
