from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import re

import hdbscan
import numpy as np
import pymorphy3

from src.discovery.encoder import DiscoveryEncoder
from src.discovery.residual_extractor import ResidualExtractor, ResidualResult
from src.schemas.models import ReviewInput
from src.vocabulary.loader import Vocabulary

_TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)


@dataclass(slots=True)
class PhraseClusterSummary:
    cluster_id: int
    n_phrases: int
    total_weight: int
    top_phrases: list[tuple[str, int]]


@dataclass(slots=True)
class PerProductEvaluationReport:
    n_clusters: int
    n_clean_clusters: int
    coverage_via_clustering: float
    purity_per_cluster: dict[int, float]
    dominant_aspect_per_cluster: dict[int, str]
    noise_rate: float
    n_uncovered_gold_aspects: int


@dataclass(slots=True)
class PerProductDiscoveryReport:
    nm_id: int
    category_id: str
    cluster_summaries: list[PhraseClusterSummary]
    evaluation: PerProductEvaluationReport
    metadata: dict[str, object]


@dataclass(slots=True)
class _PhraseRecord:
    key: str
    phrase: str
    weight: int
    lemmas: set[str]


class PerProductPhraseDiscovery:
    def __init__(
        self,
        *,
        encoder: DiscoveryEncoder,
        residual_extractor: ResidualExtractor | None = None,
        min_unique_phrases_to_cluster: int = 30,
        top_n_phrases_per_cluster: int = 10,
        purity_threshold: float = 0.7,
        hdbscan_params: Mapping[str, object] | None = None,
        morph: pymorphy3.MorphAnalyzer | None = None,
    ) -> None:
        self.encoder = encoder
        self.residual_extractor = residual_extractor or ResidualExtractor()
        self.min_unique_phrases_to_cluster = int(min_unique_phrases_to_cluster)
        self.top_n_phrases_per_cluster = int(top_n_phrases_per_cluster)
        self.purity_threshold = float(purity_threshold)
        self.hdbscan_params = {
            "min_cluster_size": 5,
            "min_samples": 3,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        }
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
    ) -> PerProductDiscoveryReport:
        residuals = [
            self.residual_extractor.extract(
                review=review,
                category_id=category_id,
                vocabulary=vocabulary,
            )
            for review in reviews
        ]
        phrase_records = self._build_phrase_records(residuals)
        uncovered_gold_aspects = self._extract_product_uncovered_gold(gold_labels, vocabulary)

        if len(phrase_records) < self.min_unique_phrases_to_cluster:
            evaluation = PerProductEvaluationReport(
                n_clusters=0,
                n_clean_clusters=0,
                coverage_via_clustering=0.0,
                purity_per_cluster={},
                dominant_aspect_per_cluster={},
                noise_rate=0.0,
                n_uncovered_gold_aspects=len(uncovered_gold_aspects),
            )
            return self._build_report(
                nm_id=nm_id,
                category_id=category_id,
                reviews=reviews,
                residuals=residuals,
                phrase_records=phrase_records,
                labels=[],
                cluster_summaries=[],
                evaluation=evaluation,
                skipped=True,
            )

        embeddings = self.encoder.encode([record.phrase for record in phrase_records])
        labels = self._cluster_embeddings(embeddings)
        cluster_summaries = self._summarize_clusters(phrase_records, labels)
        evaluation = self._evaluate_clusters(
            phrase_records=phrase_records,
            labels=labels,
            cluster_summaries=cluster_summaries,
            uncovered_gold_aspects=uncovered_gold_aspects,
        )
        return self._build_report(
            nm_id=nm_id,
            category_id=category_id,
            reviews=reviews,
            residuals=residuals,
            phrase_records=phrase_records,
            labels=labels,
            cluster_summaries=cluster_summaries,
            evaluation=evaluation,
            skipped=False,
        )

    def _cluster_embeddings(self, embeddings: np.ndarray) -> list[int]:
        clusterer = hdbscan.HDBSCAN(**self.hdbscan_params)
        return [int(label) for label in clusterer.fit_predict(embeddings)]

    def _build_phrase_records(self, residuals: list[ResidualResult]) -> list[_PhraseRecord]:
        counts: Counter[str] = Counter()
        surfaces_by_key: dict[str, Counter[str]] = {}
        lemmas_by_key: dict[str, set[str]] = {}

        for residual in residuals:
            for phrase in residual.residual_phrases:
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
            phrase = surfaces_by_key[key].most_common(1)[0][0]
            records.append(
                _PhraseRecord(
                    key=key,
                    phrase=phrase,
                    weight=int(weight),
                    lemmas=lemmas_by_key[key],
                )
            )
        return records

    def _summarize_clusters(
        self,
        phrase_records: list[_PhraseRecord],
        labels: list[int],
    ) -> list[PhraseClusterSummary]:
        records_by_cluster: dict[int, list[_PhraseRecord]] = {}
        for record, label in zip(phrase_records, labels, strict=True):
            if label == -1:
                continue
            records_by_cluster.setdefault(label, []).append(record)

        summaries: list[PhraseClusterSummary] = []
        for cluster_id in sorted(records_by_cluster):
            cluster_records = sorted(
                records_by_cluster[cluster_id],
                key=lambda record: (-record.weight, record.phrase),
            )
            summaries.append(
                PhraseClusterSummary(
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

    def _evaluate_clusters(
        self,
        *,
        phrase_records: list[_PhraseRecord],
        labels: list[int],
        cluster_summaries: list[PhraseClusterSummary],
        uncovered_gold_aspects: set[str],
    ) -> PerProductEvaluationReport:
        n_phrases = len(labels)
        n_noise = sum(1 for label in labels if label == -1)
        noise_rate = float(n_noise) / float(n_phrases) if n_phrases else 0.0
        if not uncovered_gold_aspects:
            return PerProductEvaluationReport(
                n_clusters=len(cluster_summaries),
                n_clean_clusters=0,
                coverage_via_clustering=0.0,
                purity_per_cluster={summary.cluster_id: 0.0 for summary in cluster_summaries},
                dominant_aspect_per_cluster={summary.cluster_id: "" for summary in cluster_summaries},
                noise_rate=noise_rate,
                n_uncovered_gold_aspects=0,
            )

        aspect_lemmas = {
            aspect: set(self._lemmatize_sequence(aspect))
            for aspect in uncovered_gold_aspects
        }
        records_by_cluster: dict[int, list[_PhraseRecord]] = {}
        for record, label in zip(phrase_records, labels, strict=True):
            if label == -1:
                continue
            records_by_cluster.setdefault(label, []).append(record)

        purity_per_cluster: dict[int, float] = {}
        dominant_aspect_per_cluster: dict[int, str] = {}
        matched_aspects: set[str] = set()

        for summary in cluster_summaries:
            top_phrase_set = {phrase for phrase, _ in summary.top_phrases}
            top_records = [
                record
                for record in records_by_cluster.get(summary.cluster_id, [])
                if record.phrase in top_phrase_set
            ]
            if not top_records:
                purity_per_cluster[summary.cluster_id] = 0.0
                dominant_aspect_per_cluster[summary.cluster_id] = ""
                continue

            best_aspect = ""
            best_purity = 0.0
            for aspect, lemmas in aspect_lemmas.items():
                if not lemmas:
                    continue
                matched = sum(1 for record in top_records if record.lemmas & lemmas)
                purity = float(matched) / float(len(top_records))
                if purity > best_purity or (purity == best_purity and aspect < best_aspect):
                    best_aspect = aspect
                    best_purity = purity

            purity_per_cluster[summary.cluster_id] = best_purity
            dominant_aspect_per_cluster[summary.cluster_id] = best_aspect
            if best_aspect and best_purity >= self.purity_threshold:
                matched_aspects.add(best_aspect)

        coverage = float(len(matched_aspects)) / float(len(uncovered_gold_aspects))
        return PerProductEvaluationReport(
            n_clusters=len(cluster_summaries),
            n_clean_clusters=sum(
                1 for purity in purity_per_cluster.values() if purity >= self.purity_threshold
            ),
            coverage_via_clustering=coverage,
            purity_per_cluster=purity_per_cluster,
            dominant_aspect_per_cluster=dominant_aspect_per_cluster,
            noise_rate=noise_rate,
            n_uncovered_gold_aspects=len(uncovered_gold_aspects),
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

    def _build_report(
        self,
        *,
        nm_id: int,
        category_id: str,
        reviews: list[ReviewInput],
        residuals: list[ResidualResult],
        phrase_records: list[_PhraseRecord],
        labels: list[int],
        cluster_summaries: list[PhraseClusterSummary],
        evaluation: PerProductEvaluationReport,
        skipped: bool,
    ) -> PerProductDiscoveryReport:
        n_noise = sum(1 for label in labels if label == -1)
        metadata: dict[str, object] = {
            "nm_id": int(nm_id),
            "category_id": category_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": self.encoder.model_name_or_path,
            "n_reviews": len(reviews),
            "n_reviews_with_residual": sum(
                1 for residual in residuals if any(p.strip() for p in residual.residual_phrases)
            ),
            "n_unique_residual_phrases": len(phrase_records),
            "n_noise_phrases": n_noise,
            "skipped_low_unique_phrases": skipped,
            "min_unique_phrases_to_cluster": self.min_unique_phrases_to_cluster,
            "top_n_phrases_per_cluster": self.top_n_phrases_per_cluster,
            "purity_threshold": self.purity_threshold,
            "hdbscan": dict(self.hdbscan_params),
        }
        return PerProductDiscoveryReport(
            nm_id=int(nm_id),
            category_id=category_id,
            cluster_summaries=cluster_summaries,
            evaluation=evaluation,
            metadata=metadata,
        )

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
