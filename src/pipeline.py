"""
Оркестратор ABSA Pipeline v2.

Связывает модули 1-4 в единый пайплайн:
  1. DataLoader        → загрузка отзывов из БД
  2. AntiFraudEngine   → веса доверия
  3. CandidateExtractor → морфо-кандидаты
  4. KeyBERTScorer     → семантический отбор + MMR
  5. AspectClusterer   → кластеризация → аспекты
  6. SentimentEngine   → NLI-скоры
  7. RatingMathEngine  → байесовская агрегация
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from configs.configs import config
from src.data.loader import DataLoader
from src.discovery.candidates import CandidateExtractor
from src.discovery.clusterer import AspectClusterer, AspectInfo
from src.discovery.scorer import KeyBERTScorer, ScoredCandidate
from src.fraud.engine import AntiFraudEngine
from src.math.engine import AggregationResult, RatingMathEngine
from src.schemas.models import ReviewInput
from src.sentiment.engine import SentimentEngine, SentimentResult


@dataclass
class PipelineResult:
    """Итог работы пайплайна для одного товара"""
    product_id: int
    reviews_processed: int
    processing_time: float
    aspects: Dict[str, dict]              # {name: {score, raw_mean, controversy, mentions}}
    aggregation: Optional[AggregationResult] = None
    sentiment_details: List[SentimentResult] = field(default_factory=list)
    aspect_keywords: Dict[str, List[str]] = field(default_factory=dict)


class ABSAPipeline:
    """
    Оркестратор v2.

    Использование:
        pipeline = ABSAPipeline()
        result = pipeline.analyze_product(nm_id=154532597, limit=100)
    """

    def __init__(self, db_path: str = "data/dataset.db"):
        print("[Pipeline] Инициализация модулей...")
        t0 = time.time()

        self.loader = DataLoader(db_path)

        # Общая модель-энкодер (rubert-tiny2) — одна на всех
        self._encoder = SentenceTransformer(config.models.encoder_path)

        self.candidate_extractor = CandidateExtractor()
        self.scorer = KeyBERTScorer(model=self._encoder)
        self.clusterer = AspectClusterer(model=self._encoder)
        self.fraud_engine = AntiFraudEngine()
        self.sentiment_engine = SentimentEngine()
        self.math_engine = RatingMathEngine()

        print(f"[Pipeline] Готов за {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def analyze_product(
        self, nm_id: int, limit: int = 200
    ) -> PipelineResult:
        """End-to-end анализ одного товара (отзывы из SQLite через DataLoader)."""
        print(f"[1/7] Загрузка отзывов для {nm_id}...")
        reviews = self.loader.load_reviews(nm_id, limit)
        if not reviews:
            return PipelineResult(
                product_id=nm_id, reviews_processed=0,
                processing_time=0.0, aspects={},
            )
        print(f"       Загружено {len(reviews)} отзывов")
        return self.analyze_reviews_list(reviews, nm_id)

    def analyze_reviews_list(
        self, reviews: List[ReviewInput], product_id: int
    ) -> PipelineResult:
        """
        Полный прогон (как analyze_product), но отзывы задаются снаружи
        (например из merged_checked_reviews.csv / JSON).
        """
        t_start = time.time()
        if not reviews:
            return PipelineResult(
                product_id=product_id,
                reviews_processed=0,
                processing_time=0.0,
                aspects={},
            )

        texts = [r.clean_text for r in reviews]
        step_t = time.time()

        def _tick(step_label: str, step_idx: int) -> None:
            nonlocal step_t
            now = time.time()
            seg = now - step_t
            tot = now - t_start
            print(
                f"       ⏱ шаг {step_idx}/7 «{step_label}» {seg:.1f}s "
                f"| с начала {tot:.1f}s"
            )
            step_t = now

        # 2. AntiFraud
        print("[2/7] AntiFraud...")
        trust_weights = self.fraud_engine.calculate_trust_weights(texts)
        _tick("AntiFraud", 2)

        # 3. Candidate Extraction
        print("[3/7] Извлечение кандидатов...")
        all_candidates = []
        review_candidates_map: Dict[str, list] = {}
        sentence_to_review: Dict[str, str] = {}

        for i, text in enumerate(
            tqdm(texts, desc="      отзывы", leave=False, unit="шт")
        ):
            candidates = self.candidate_extractor.extract(text)
            all_candidates.extend(candidates)
            review_candidates_map[reviews[i].id] = candidates
            for c in candidates:
                sentence_to_review[c.sentence.strip()] = reviews[i].id
                sentence_to_review[c.sentence.lower().strip()] = reviews[i].id

        print(f"       Всего кандидатов: {len(all_candidates)}")
        _tick("кандидаты", 3)

        # 4. KeyBERT + MMR
        print("[4/7] KeyBERT-скоринг + MMR...")
        scored_candidates = self.scorer.score_and_select(all_candidates)
        print(f"       Отобрано после MMR: {len(scored_candidates)}")
        _tick("KeyBERT+MMR", 4)

        # 5. Кластеризация
        print("[5/7] Кластеризация аспектов...")
        aspects = self.clusterer.cluster(scored_candidates)
        aspect_names = list(aspects.keys())
        print(f"       Найдено аспектов: {len(aspect_names)} — {aspect_names}")
        _tick("кластеризация", 5)

        if not aspects:
            return PipelineResult(
                product_id=product_id, reviews_processed=len(reviews),
                processing_time=time.time() - t_start, aspects={},
            )

        # 6. NLI Sentiment
        print(f"[6/7] NLI Sentiment ({len(aspect_names)} аспектов)...")
        sentiment_pairs = self._build_sentiment_pairs(
            scored_candidates, aspects, sentence_to_review,
        )
        print(f"       Пар для NLI: {len(sentiment_pairs)}")
        sentiment_scores = self.sentiment_engine.batch_analyze(sentiment_pairs)
        print(f"       Получено оценок: {len(sentiment_scores)}")
        _tick("NLI", 6)

        # 7. Математическая агрегация
        print("[7/7] Агрегация...")
        aggregation_input = self._build_aggregation_input(
            reviews, sentiment_scores, trust_weights,
        )
        agg_result = self.math_engine.aggregate(aggregation_input)
        _tick("агрегация", 7)

        elapsed = time.time() - t_start
        print(f"[Pipeline] Готово за {elapsed:.1f}s")

        return PipelineResult(
            product_id=product_id,
            reviews_processed=len(reviews),
            processing_time=round(elapsed, 2),
            aspects={
                name: {
                    "score": a.score,
                    "raw_mean": a.raw_mean,
                    "controversy": a.controversy,
                    "mentions": a.mentions,
                    "effective_mentions": a.effective_mentions,
                }
                for name, a in agg_result.aspects.items()
            },
            aggregation=agg_result,
            sentiment_details=sentiment_scores,
            aspect_keywords={
                name: info.keywords for name, info in aspects.items()
            },
        )

    # ------------------------------------------------------------------
    # Связующие функции
    # ------------------------------------------------------------------

    def _build_sentiment_pairs(
        self,
        scored_candidates: List[ScoredCandidate],
        aspects: Dict[str, AspectInfo],
        sentence_to_review: Dict[str, str],
    ) -> List[Tuple[str, str, str, str, float]]:
        """
        Multi-label: cos(span, anchor) >= threshold → NLI-пара (до max_aspects якорей).
        product_anchors — из результата кластеризации (имена якорей / nli_label).
        (review_id, sentence, aspect_name, nli_label, weight); здесь aspect_name = nli_label = якорь.
        """
        if not aspects or not scored_candidates:
            return []

        threshold = float(config.discovery.multi_label_threshold)
        max_aspects = int(config.discovery.multi_label_max_aspects)

        anchor_names = list(self.clusterer._anchor_embeddings.keys())
        anchor_matrix = np.stack(
            [self.clusterer._anchor_embeddings[n] for n in anchor_names]
        )

        product_anchors: set[str] = set()
        for asp_name, info in aspects.items():
            nli = (info.nli_label or asp_name).strip() or asp_name
            if nli in self.clusterer._anchor_embeddings:
                product_anchors.add(nli)
            if asp_name in self.clusterer._anchor_embeddings:
                product_anchors.add(asp_name)

        seen_pairs: set = set()
        pairs: List[Tuple[str, str, str, str, float]] = []

        for cand in scored_candidates:
            emb = np.asarray(cand.embedding, dtype=np.float64).reshape(1, -1)
            sims = cosine_similarity(emb, anchor_matrix)[0]

            candidates_anchors: List[Tuple[str, float]] = []
            for idx, sim in enumerate(sims):
                aname = anchor_names[idx]
                if sim >= threshold and aname in product_anchors:
                    candidates_anchors.append((aname, float(sim)))

            candidates_anchors.sort(key=lambda x: x[1], reverse=True)
            candidates_anchors = candidates_anchors[:max_aspects]

            if not candidates_anchors:
                continue

            review_id = sentence_to_review.get(
                cand.sentence.strip(),
                sentence_to_review.get(cand.sentence.lower().strip(), "unknown"),
            )

            for aname, sim in candidates_anchors:
                pair_key = (review_id, cand.sentence, aname)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pairs.append(
                    (review_id, cand.sentence, aname, aname, float(sim)),
                )

        return pairs

    def _build_aggregation_input(
        self,
        reviews: List[ReviewInput],
        sentiment_scores: List[SentimentResult],
        trust_weights: List[float],
    ) -> List[Dict]:
        """
        Формирует вход для math_engine.aggregate():
        [{"review_id": ..., "aspects": {...}, "fraud_weight": ..., "date": ...}, ...]
        """
        review_id_to_idx = {r.id: i for i, r in enumerate(reviews)}

        # Собираем оценки по review_id: (score, confidence-вес)
        review_aspects: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        for sr in sentiment_scores:
            if sr.review_id not in review_aspects:
                review_aspects[sr.review_id] = {}
            aspects = review_aspects[sr.review_id]
            if sr.aspect not in aspects:
                aspects[sr.aspect] = []
            aspects[sr.aspect].append((sr.score, sr.confidence))

        # Взвешенное среднее по аспектам внутри одного отзыва
        result = []
        for review_id, aspects in review_aspects.items():
            idx = review_id_to_idx.get(review_id)
            if idx is None:
                continue

            aspect_means: Dict[str, float] = {}
            for name, pairs_sw in aspects.items():
                scores = [p[0] for p in pairs_sw]
                weights = [p[1] for p in pairs_sw]
                wsum = sum(weights)
                if wsum > 0:
                    aspect_means[name] = float(
                        sum(s * w for s, w in zip(scores, weights)) / wsum
                    )
                else:
                    aspect_means[name] = float(sum(scores) / len(scores))

            result.append({
                "review_id": review_id,
                "aspects": aspect_means,
                "fraud_weight": trust_weights[idx],
                "date": reviews[idx].created_date,
            })

        return result

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Разбивка текста на предложения (та же логика, что в CandidateExtractor)."""
        import re
        parts = re.split(r'[.!?]+', text)
        return [p.strip() for p in parts if p.strip()]


# ------------------------------------------------------------------
# End-to-end тест
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    pipeline = ABSAPipeline()

    # Тест на реальном товаре (гитара)
    nm_id = 154532597
    print(f"\n{'='*60}")
    print(f"End-to-end тест: товар {nm_id}")
    print(f"{'='*60}\n")

    result = pipeline.analyze_product(nm_id, limit=100)

    print(f"\nТовар: {result.product_id}")
    print(f"Отзывов: {result.reviews_processed}")
    print(f"Время: {result.processing_time}s")
    print(f"\nАспекты ({len(result.aspects)}):")
    for name, metrics in sorted(result.aspects.items(), key=lambda x: x[1]["score"], reverse=True):
        kw = result.aspect_keywords.get(name, [])[:5]
        print(f"  {name:25s} score={metrics['score']:.2f}  "
              f"raw={metrics['raw_mean']:.2f}  "
              f"controversy={metrics['controversy']:.2f}  "
              f"mentions={metrics['mentions']}  "
              f"keywords={kw}")
