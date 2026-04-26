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
from collections import Counter
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
from src.factories import (
    build_aggregation_stage,
    build_clustering_stage,
    build_extraction_stage,
    build_fraud_stage,
    build_pairing_stage,
    build_scoring_stage,
    build_sentiment_stage,
)
from src.schemas.models import (
    AggregationInput,
    AggregationResult,
    EvalData,
    PairingContext,
    ReviewInput,
    SentimentResult,
)
from src.stages import (
    AggregationStage, ClusteringStage, ExtractionStage,
    FraudStage, PairingStage, ScoringStage, SentimentStage,
)
from src.stages.pairing import extract_all_with_mapping
from src.snapshots import SnapshotWriter


def _extract_pair_head_label(label: str) -> Optional[str]:
    normalized = str(label or "").strip().lower()
    if not normalized or normalized.startswith("event_") or "_" not in normalized:
        return None
    head = normalized.split("_", 1)[0].strip()
    return head or None


def build_aspect_eval_labels(
    aspects: Dict[str, object],
    scored_candidates: List[object],
) -> Dict[str, str]:
    keyword_source_counts: Dict[str, Counter[str]] = {}
    for candidate in scored_candidates:
        span = str(getattr(candidate, "span", "") or "").strip()
        source_span = str(getattr(candidate, "source_span", "") or "").strip().lower()
        if not span or not source_span:
            continue
        keyword_source_counts.setdefault(span, Counter())[source_span] += 1

    aspect_eval_labels: Dict[str, str] = {}
    for aspect_name, info in aspects.items():
        keywords = list(getattr(info, "keywords", []) or [])
        head_counts: Counter[str] = Counter()
        for keyword in keywords:
            head = _extract_pair_head_label(str(keyword))
            if head:
                head_counts[head] += 1

        if head_counts:
            best_count = max(head_counts.values())
            best_heads = sorted(
                head for head, count in head_counts.items()
                if count == best_count
            )
            aspect_eval_labels[str(aspect_name)] = best_heads[0]
            continue

        source_counts: Counter[str] = Counter()
        for keyword in keywords:
            source_counts.update(keyword_source_counts.get(str(keyword), Counter()))

        if source_counts:
            best_count = max(source_counts.values())
            best_labels = sorted(
                label for label, count in source_counts.items()
                if count == best_count
            )
            aspect_eval_labels[str(aspect_name)] = best_labels[0]
        elif keywords:
            aspect_eval_labels[str(aspect_name)] = str(keywords[0])
        else:
            aspect_eval_labels[str(aspect_name)] = str(aspect_name)

    return aspect_eval_labels


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
    diagnostics: Dict[str, object] = field(default_factory=dict)


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
        encoder: Optional[SentenceTransformer] = None,
        fraud_stage: Optional[FraudStage] = None,
        extraction_stage: Optional[ExtractionStage] = None,
        scoring_stage: Optional[ScoringStage] = None,
        clustering_stage: Optional[ClusteringStage] = None,
        pairing_stage: Optional[PairingStage] = None,
        sentiment_stage: Optional[SentimentStage] = None,
        aggregation_stage: Optional[AggregationStage] = None,
    ):
        print("[Pipeline] Инициализация модулей...")
        t0 = time.time()

        self.loader = DataLoader(db_path)

        # Общая модель-энкодер (rubert-tiny2) — одна на всех
        self._encoder = encoder or SentenceTransformer(config.models.encoder_path)

        # Стандартные реализации как дефолт; заменяются через DI-параметры выше
        self.candidate_extractor: ExtractionStage = extraction_stage or build_extraction_stage()
        self.scorer: ScoringStage = scoring_stage or build_scoring_stage(self._encoder)
        self.clusterer: ClusteringStage = clustering_stage or build_clustering_stage(
            encoder=self._encoder,
            name="aspect",
        )
        self.fraud_engine: FraudStage = fraud_stage or build_fraud_stage(self._encoder)
        self.pairing_stage: PairingStage = pairing_stage or build_pairing_stage()
        self.sentiment_engine: SentimentStage = sentiment_stage or build_sentiment_stage()
        self.math_engine: AggregationStage = aggregation_stage or build_aggregation_stage()

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

        stages_result = self._run_stages(
            reviews=reviews,
            product_id=product_id,
            snapshot_writer=snapshot_writer,
            t_start=t_start,
        )
        if not stages_result.aspects:
            return PipelineResult(
                product_id=product_id,
                reviews_processed=len(reviews),
                processing_time=time.time() - t_start,
                aspects={},
                diagnostics=stages_result.diagnostics,
            )

        # 7. Математическая агрегация
        print("[7/7] Агрегация...")
        aggregation_input = self._build_aggregation_input(
            reviews=reviews,
            per_review=stages_result.per_review,
            trust_weights=stages_result.trust_weights,
        )
        if snapshot_writer:
            snapshot_writer.save_aggregation_input(aggregation_input)
        agg_result = self.math_engine.aggregate(aggregation_input)

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
            sentiment_details=stages_result.sentiment_results,
            aspect_keywords=stages_result.aspect_keywords,
            diagnostics=stages_result.diagnostics,
        )
        if snapshot_writer:
            snapshot_writer.save_pipeline_result(result)
        return result

    def analyze_for_eval(
        self,
        reviews: List[ReviewInput],
        product_id: int,
        snapshot_writer: Optional[SnapshotWriter] = None,
    ) -> EvalData:
        return self._run_stages(
            reviews=reviews,
            product_id=product_id,
            snapshot_writer=snapshot_writer,
        )

    # ------------------------------------------------------------------
    # Связующие функции
    # ------------------------------------------------------------------

    def _collect_clusterer_diagnostics(self) -> Dict[str, object]:
        diagnostics = self.clusterer.get_diagnostics()
        return dict(diagnostics) if isinstance(diagnostics, dict) else {}

    def _build_aggregation_input(
        self,
        reviews: List[ReviewInput],
        per_review: Dict[str, Dict[str, float]],
        trust_weights: List[float],
    ) -> List[AggregationInput]:
        """
        Формирует вход для math_engine.aggregate():
        [{"review_id": ..., "aspects": {...}, "fraud_weight": ..., "date": ...}, ...]
        """
        review_id_to_idx = {r.id: i for i, r in enumerate(reviews)}

        result: List[AggregationInput] = []
        for review_id, aspects in per_review.items():
            idx = review_id_to_idx.get(review_id)
            if idx is None:
                continue
            result.append(
                AggregationInput(
                    review_id=review_id,
                    aspects=aspects,
                    fraud_weight=float(trust_weights[idx]),
                    date=reviews[idx].created_date,
                )
            )

        return result

    def _run_stages(
        self,
        reviews: List[ReviewInput],
        product_id: int,
        snapshot_writer: Optional[SnapshotWriter] = None,
        t_start: Optional[float] = None,
    ) -> EvalData:
        _ = product_id
        texts = [r.clean_text for r in reviews]
        review_ids = [r.id for r in reviews]
        start = t_start if t_start is not None else time.time()
        step_t = time.time()

        def _tick(step_label: str, step_idx: int) -> None:
            nonlocal step_t
            now = time.time()
            seg = now - step_t
            tot = now - start
            print(
                f"       ⏱ шаг {step_idx}/7 «{step_label}» {seg:.1f}s "
                f"| с начала {tot:.1f}s"
            )
            step_t = now

        print("[2/7] AntiFraud...")
        trust_weights = self.fraud_engine.calculate_trust_weights(texts)
        _tick("AntiFraud", 2)
        if snapshot_writer:
            snapshot_writer.save_fraud(review_ids, trust_weights)

        print("[3/7] Извлечение кандидатов...")
        all_candidates, sentence_to_review = extract_all_with_mapping(
            self.candidate_extractor,
            texts,
            review_ids,
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

        print("[4/7] KeyBERT-скоринг + MMR...")
        scored_candidates = self.scorer.score_and_select(all_candidates)
        print(f"       Отобрано после MMR: {len(scored_candidates)}")
        _tick("KeyBERT+MMR", 4)
        if snapshot_writer:
            snapshot_writer.save_scored(scored_candidates)

        print("[5/7] Кластеризация аспектов...")
        aspects = self.clusterer.cluster(scored_candidates)
        diagnostics = self._collect_clusterer_diagnostics()
        diagnostics["aspect_eval_labels"] = build_aspect_eval_labels(
            aspects,
            scored_candidates,
        )
        aspect_names = list(aspects.keys())
        print(f"       Найдено аспектов: {len(aspect_names)} — {aspect_names}")
        _tick("кластеризация", 5)
        if snapshot_writer:
            snapshot_writer.save_clusters(aspects)

        if not aspects:
            return EvalData(
                aspects={},
                sentiment_results=[],
                sentence_to_review=sentence_to_review,
                trust_weights=trust_weights,
                per_review={},
                aspect_keywords={},
                diagnostics=diagnostics,
            )

        print(f"[6/7] NLI Sentiment ({len(aspect_names)} аспектов)...")
        pairing_context = PairingContext(
            review_text_by_id={
                review_id: text
                for review_id, text in zip(review_ids, texts)
            },
            sentence_to_review=sentence_to_review,
            scored_candidates=scored_candidates,
            aspects=aspects,
            metadata=self.clusterer.get_pairing_metadata(),
            multi_label_threshold=float(config.discovery.multi_label_threshold),
            multi_label_max_aspects=int(config.discovery.multi_label_max_aspects),
        )
        sentiment_pairs = self.pairing_stage.build_pairs(pairing_context)
        print(f"       Пар для NLI: {len(sentiment_pairs)}")
        diagnostics["nli_pairs_count"] = int(len(sentiment_pairs))
        if snapshot_writer:
            snapshot_writer.save_sentiment_pairs(sentiment_pairs)
        sentiment_scores = self.sentiment_engine.batch_analyze(sentiment_pairs)
        relevance_threshold = float(getattr(config.sentiment, "relevance_threshold", 0.0))
        if relevance_threshold > 0:
            unfiltered_scores = sentiment_scores
            sentiment_scores = [
                s
                for s in unfiltered_scores
                # v4: P(ent)+P(contra) = 1 - P(neutral)
                if (float(s.p_ent_pos) + float(s.p_ent_neg)) >= relevance_threshold
            ]
            if unfiltered_scores and not sentiment_scores:
                print(
                    "       [WARN] relevance_threshold отфильтровал все NLI-пары; "
                    "используется нефильтрованный набор."
                )
                sentiment_scores = unfiltered_scores
        print(f"       Получено оценок: {len(sentiment_scores)}")
        _tick("NLI", 6)
        if snapshot_writer:
            snapshot_writer.save_sentiment_results(sentiment_scores)

        per_review_pairs: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        for sr in sentiment_scores:
            review_bucket = per_review_pairs.setdefault(sr.review_id, {})
            review_bucket.setdefault(sr.aspect, []).append((sr.score, sr.confidence))

        per_review: Dict[str, Dict[str, float]] = {}
        for review_id, aspect_pairs in per_review_pairs.items():
            per_review[review_id] = {}
            for aspect_name, pairs_sw in aspect_pairs.items():
                scores = [score for score, _ in pairs_sw]
                weights = [weight for _, weight in pairs_sw]
                wsum = float(sum(weights))
                if wsum > 0:
                    per_review[review_id][aspect_name] = float(
                        sum(s * w for s, w in zip(scores, weights)) / wsum
                    )
                else:
                    per_review[review_id][aspect_name] = float(sum(scores) / len(scores))

        return EvalData(
            aspects=aspects,
            sentiment_results=sentiment_scores,
            sentence_to_review=sentence_to_review,
            trust_weights=trust_weights,
            per_review=per_review,
            aspect_keywords={name: info.keywords for name, info in aspects.items()},
            diagnostics=diagnostics,
        )

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
