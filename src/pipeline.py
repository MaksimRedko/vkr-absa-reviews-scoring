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

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x
from typing import Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

from configs.configs import config
from src.data.loader import DataLoader
from src.discovery.candidates import CandidateExtractor
from src.discovery.clusterer import AspectClusterer
from src.discovery.scorer import KeyBERTScorer
from src.fraud.engine import AntiFraudEngine
from src.math.engine import AggregationResult, RatingMathEngine
from src.pairing import build_sentiment_pairs, extract_all_with_mapping
from src.schemas.models import ReviewInput
from src.sentiment.engine import SentimentEngine, SentimentResult
from src.stages import (
    AggregationStage, ClusteringStage, ExtractionStage,
    FraudStage, ScoringStage, SentimentStage,
)
from src.snapshots import SnapshotWriter


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

    def __init__(
        self,
        db_path: str = "data/dataset.db",
        # Опциональные переопределения стадий (Dependency Injection).
        # None → используется стандартная реализация.
        # Передай любой объект, реализующий соответствующий ABC из src/stages.py,
        # и пайплайн подхватит его без других изменений.
        fraud_stage: Optional[FraudStage] = None,
        extraction_stage: Optional[ExtractionStage] = None,
        scoring_stage: Optional[ScoringStage] = None,
        clustering_stage: Optional[ClusteringStage] = None,
        sentiment_stage: Optional[SentimentStage] = None,
        aggregation_stage: Optional[AggregationStage] = None,
    ):
        print("[Pipeline] Инициализация модулей...")
        t0 = time.time()

        self.loader = DataLoader(db_path)

        # Общая модель-энкодер (rubert-tiny2) — одна на всех
        self._encoder = SentenceTransformer(config.models.encoder_path)

        # Стандартные реализации как дефолт; заменяются через DI-параметры выше
        self.candidate_extractor: ExtractionStage  = extraction_stage  or CandidateExtractor()
        self.scorer:              ScoringStage      = scoring_stage     or KeyBERTScorer(model=self._encoder)
        self.clusterer:           ClusteringStage   = clustering_stage  or AspectClusterer(model=self._encoder)
        self.fraud_engine:        FraudStage        = fraud_stage       or AntiFraudEngine()
        self.sentiment_engine:    SentimentStage    = sentiment_stage   or SentimentEngine()
        self.math_engine:         AggregationStage  = aggregation_stage or RatingMathEngine()

        print(f"[Pipeline] Готов за {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def analyze_product(
        self,
        nm_id: int,
        limit: int = 200,
        save_input_snapshot: bool = False,
        input_snapshot_path: Optional[str] = None,
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
        return self.analyze_reviews_list(
            reviews=reviews,
            product_id=nm_id,
            save_input_snapshot=save_input_snapshot,
            input_snapshot_path=input_snapshot_path,
        )

    def analyze_reviews_list(
        self,
        reviews: List[ReviewInput],
        product_id: int,
        save_input_snapshot: bool = False,
        input_snapshot_path: Optional[str] = None,
        snapshot_writer: Optional[SnapshotWriter] = None,
    ) -> PipelineResult:
        """
        Полный прогон (как analyze_product), но отзывы задаются снаружи
        (например из merged_checked_reviews.csv / JSON).

        snapshot_writer: если передан — сохраняет снепшот после каждой стадии.
            Создаётся через ExperimentManager.snapshot_writer(nm_id) или напрямую:
              SnapshotWriter(base_dir=Path("experiments/runs/.../snapshots"), product_id=nm_id)
        """
        t_start = time.time()
        if not reviews:
            return PipelineResult(
                product_id=product_id,
                reviews_processed=0,
                processing_time=0.0,
                aspects={},
            )

        if save_input_snapshot:
            self._save_input_snapshot(
                reviews=reviews,
                product_id=product_id,
                snapshot_path=input_snapshot_path,
            )

        if snapshot_writer:
            snapshot_writer.save_reviews(reviews)

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
        if snapshot_writer:
            snapshot_writer.save_fraud([r.id for r in reviews], trust_weights)

        # 3. Candidate Extraction
        print("[3/7] Извлечение кандидатов...")
        all_candidates, sentence_to_review = extract_all_with_mapping(
            self.candidate_extractor,
            texts,
            [r.id for r in reviews],
        )

        print(f"       Всего кандидатов: {len(all_candidates)}")
        _tick("кандидаты", 3)
        if snapshot_writer:
            review_candidates_map: Dict[str, list] = {}
            for i, text in enumerate(
                tqdm(texts, desc="      отзывы", leave=False, unit="шт")
            ):
                review_candidates_map[reviews[i].id] = self.candidate_extractor.extract(text)
            snapshot_writer.save_candidates(review_candidates_map)

        # 4. KeyBERT + MMR
        print("[4/7] KeyBERT-скоринг + MMR...")
        scored_candidates = self.scorer.score_and_select(all_candidates)
        print(f"       Отобрано после MMR: {len(scored_candidates)}")
        _tick("KeyBERT+MMR", 4)
        if snapshot_writer:
            snapshot_writer.save_scored(scored_candidates)

        # 5. Кластеризация
        print("[5/7] Кластеризация аспектов...")
        aspects = self.clusterer.cluster(scored_candidates)
        aspect_names = list(aspects.keys())
        print(f"       Найдено аспектов: {len(aspect_names)} — {aspect_names}")
        _tick("кластеризация", 5)
        if snapshot_writer:
            snapshot_writer.save_clusters(aspects)

        if not aspects:
            return PipelineResult(
                product_id=product_id, reviews_processed=len(reviews),
                processing_time=time.time() - t_start, aspects={},
            )

        # 6. NLI Sentiment
        print(f"[6/7] NLI Sentiment ({len(aspect_names)} аспектов)...")
        sentiment_pairs = build_sentiment_pairs(
            scored_candidates=scored_candidates,
            aspects=aspects,
            sentence_to_review=sentence_to_review,
            anchor_embeddings=self.clusterer._anchor_embeddings,
            threshold=float(config.discovery.multi_label_threshold),
            max_aspects=int(config.discovery.multi_label_max_aspects),
        )
        print(f"       Пар для NLI: {len(sentiment_pairs)}")
        if snapshot_writer:
            snapshot_writer.save_sentiment_pairs(sentiment_pairs)
        sentiment_scores = self.sentiment_engine.batch_analyze(sentiment_pairs)
        print(f"       Получено оценок: {len(sentiment_scores)}")
        _tick("NLI", 6)
        if snapshot_writer:
            snapshot_writer.save_sentiment_results(sentiment_scores)

        # 7. Математическая агрегация
        print("[7/7] Агрегация...")
        aggregation_input = self._build_aggregation_input(
            reviews, sentiment_scores, trust_weights,
        )
        if snapshot_writer:
            snapshot_writer.save_aggregation_input(aggregation_input)
        agg_result = self.math_engine.aggregate(aggregation_input)
        _tick("агрегация", 7)

        elapsed = time.time() - t_start
        print(f"[Pipeline] Готово за {elapsed:.1f}s")

        result = PipelineResult(
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
        if snapshot_writer:
            snapshot_writer.save_pipeline_result(result)
        return result

    # ------------------------------------------------------------------
    # Связующие функции
    # ------------------------------------------------------------------

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

    @staticmethod
    def _save_input_snapshot(
        reviews: List[ReviewInput],
        product_id: int,
        snapshot_path: Optional[str] = None,
    ) -> str:
        """
        Сохраняет исходный массив отзывов ДО обработки.
        Формат: JSONL (1 отзыв = 1 строка).
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        default_path = Path("data/snapshots") / f"input_reviews_nm{product_id}_{ts}.jsonl"
        out_path = Path(snapshot_path) if snapshot_path else default_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            for r in reviews:
                row = {
                    "id": r.id,
                    "nm_id": r.nm_id,
                    "rating": r.rating,
                    "created_date": r.created_date.isoformat(),
                    "full_text": r.full_text,
                    "pros": r.pros,
                    "cons": r.cons,
                    "clean_text": r.clean_text,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[Pipeline] Snapshot входных отзывов сохранён: {out_path}")
        return str(out_path)


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
