"""
Evaluation pipeline: разметка vs пайплайн.

Шаги:
  1. Статистика по разметке
  2. Прогон Discovery + Sentiment
  3. Маппинг аспектов (ручная таблица)
  4. Метрики: Precision, Recall, Sentiment MAE

Фиксы v2:
  - FIX1: _collect_score_pairs — reverse_map → Dict[str, List[str]] (many-to-one)
  - FIX2: _validate_mapping — проверка полноты MAPPING перед расчётом метрик
  - FIX3: evaluate_with_mapping — честный LOO для калибровки
  - FIX4: diagnose_unknown_reviews — диагностика unknown review_id
"""

from __future__ import annotations

import ast
import argparse
import json
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ── Шаг 1: Статистика по разметке ──────────────────────────────────────────

def load_markup(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["true_labels_parsed"] = df["true_labels"].apply(_parse_labels)
    return df


def _parse_labels(val) -> Optional[Dict[str, float]]:
    if pd.isna(val) or str(val).strip() in ("", "nan", "{}"):
        return None
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, dict) and parsed:
            return {k: float(v) for k, v in parsed.items()}
    except (ValueError, SyntaxError):
        pass
    return None


def markup_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nm_id, grp in df.groupby("nm_id"):
        total = len(grp)
        labeled = grp["true_labels_parsed"].notna().sum()

        aspect_scores: Dict[str, List[float]] = defaultdict(list)
        for labels in grp["true_labels_parsed"].dropna():
            for aspect, score in labels.items():
                aspect_scores[aspect].append(score)

        aspect_summary = {
            asp: f"{np.mean(scores):.2f} (n={len(scores)})"
            for asp, scores in sorted(aspect_scores.items())
        }

        rows.append({
            "nm_id": nm_id,
            "total_reviews": total,
            "labeled_reviews": labeled,
            "unique_aspects": list(sorted(aspect_scores.keys())),
            "aspect_avg_scores": aspect_summary,
        })

    return pd.DataFrame(rows)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def apply_config_overrides(overrides: Dict[str, object]) -> None:
    from configs.configs import config
    for section, values in overrides.items():
        if not hasattr(config, section):
            continue
        if not isinstance(values, dict):
            continue
        for key, val in values.items():
            setattr(getattr(config, section), key, val)


def load_eval_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg if isinstance(cfg, dict) else {}


# ── Шаг 2: Прогон пайплайна ────────────────────────────────────────────────

def run_pipeline_for_ids(
    nm_ids: List[int],
    csv_path: str,
    json_path: str,
) -> Dict[int, dict]:
    """
    Прогоняет пайплайн на отзывах из json-файла (те же, что в разметке).
    Возвращает {nm_id: {"aspects": [...], "per_review": [{...}]}}
    """
    from src.discovery.candidates import CandidateExtractor
    from src.discovery.clusterer import AspectClusterer
    from src.discovery.scorer import KeyBERTScorer
    from src.fraud.engine import AntiFraudEngine
    from src.sentiment.engine import SentimentEngine

    from sentence_transformers import SentenceTransformer
    from configs.configs import config

    with open(json_path, "r", encoding="utf-8") as f:
        all_reviews_raw = json.load(f)

    reviews_by_nm = defaultdict(list)
    for r in all_reviews_raw:
        reviews_by_nm[r["nm_id"]].append(r)

    encoder = SentenceTransformer(config.models.encoder_path)
    extractor = CandidateExtractor()
    scorer = KeyBERTScorer(model=encoder)
    clusterer = AspectClusterer(model=encoder)
    fraud = AntiFraudEngine()
    sentiment = SentimentEngine()

    results = {}

    for nm_id in nm_ids:
        raw_reviews = reviews_by_nm.get(nm_id, [])
        if not raw_reviews:
            print(f"[SKIP] nm_id={nm_id}: нет отзывов в JSON")
            continue

        print(f"\n{'='*60}")
        print(f"nm_id={nm_id}  ({len(raw_reviews)} отзывов)")
        print(f"{'='*60}")

        from src.schemas.models import ReviewInput
        reviews = []
        for r in raw_reviews:
            try:
                ri = ReviewInput(**r)
                if ri.clean_text:
                    reviews.append(ri)
            except Exception:
                continue

        texts = [r.clean_text for r in reviews]
        review_id_list = [r.id for r in reviews]

        trust_weights = fraud.calculate_trust_weights(texts)

        all_candidates = []
        for text in texts:
            all_candidates.extend(extractor.extract(text))

        scored = scorer.score_and_select(all_candidates)
        aspects = clusterer.cluster(scored)
        aspect_names = list(aspects.keys())
        print(f"  Аспекты: {aspect_names}")

        if not aspects:
            results[nm_id] = {"aspects": [], "per_review": {}}
            continue

        pairs = _build_pairs(scored, aspects, reviews)
        sentiment_scores = sentiment.batch_analyze(pairs)

        per_review = defaultdict(dict)
        for sr in sentiment_scores:
            if sr.aspect not in per_review[sr.review_id]:
                per_review[sr.review_id][sr.aspect] = []
            per_review[sr.review_id][sr.aspect].append(sr.score)

        per_review_avg = {}
        for rid, asp_dict in per_review.items():
            per_review_avg[rid] = {
                asp: round(float(np.mean(scores)), 2)
                for asp, scores in asp_dict.items()
            }

        # Диагностика: какие кандидаты были до кластеризации
        all_cand_texts = [c.span for c in all_candidates]
        scored_texts = [c.span for c in scored]

        results[nm_id] = {
            "aspects": aspect_names,
            "per_review": per_review_avg,
            "aspect_keywords": {
                name: info.keywords for name, info in aspects.items()
            },
            "diagnostics": {
                "raw_candidates_count": len(all_candidates),
                "scored_candidates_count": len(scored),
                "raw_candidate_samples": all_cand_texts[:30],
                "scored_candidate_samples": scored_texts,
            },
        }

    return results


def _build_pairs(scored, aspects, reviews):
    """Формирует (review_id, sentence, aspect_name) — аналог pipeline._build_sentiment_pairs."""
    import re
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    sentence_to_review = {}
    for review in reviews:
        for sent in re.split(r'[.!?\n]+', review.clean_text):
            s = sent.strip()
            if s:
                sentence_to_review[s] = review.id
                sentence_to_review[s.lower()] = review.id

    aspect_names = list(aspects.keys())
    centroids = np.stack([aspects[n].centroid_embedding for n in aspect_names])

    seen = set()
    pairs = []
    for cand in scored:
        sim = cos_sim(cand.embedding.reshape(1, -1), centroids)[0]
        best_idx = int(np.argmax(sim))
        aspect_name = aspect_names[best_idx]
        review_id = sentence_to_review.get(
            cand.sentence.strip(),
            sentence_to_review.get(cand.sentence.lower().strip(), "unknown"),
        )
        key = (review_id, cand.sentence, aspect_name)
        if key not in seen:
            seen.add(key)
            pairs.append((review_id, cand.sentence, aspect_name))

    return pairs


# ── Шаг 3 + 4: Маппинг и метрики ──────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FIX1: _collect_score_pairs — many-to-one через mean-агрегацию          │
# │                                                                         │
# │ БЫЛО:  reverse_map = {ta: pa for pa, ta in ...}  → dict перезатирал    │
# │ СТАЛО: reverse_map = defaultdict(list) → mean по всем pred scores      │
# └─────────────────────────────────────────────────────────────────────────┘

def _collect_score_pairs(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, dict],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> Dict[int, List[Tuple[float, float]]]:
    """Собирает (pred_score, true_score) пары per product.

    FIX1: reverse_map теперь Dict[str, List[str]], агрегация — mean.
    Если два predicted-аспекта маппятся на один true, берём среднее их скоров.
    """
    pairs_by_product: Dict[int, List[Tuple[float, float]]] = {}

    for nm_id, pred_data in pipeline_results.items():
        product_mapping = mapping.get(nm_id, {})
        per_review_pred = pred_data["per_review"]

        # FIX1: reverse_map как Dict[str, List[str]]
        reverse_map: Dict[str, List[str]] = defaultdict(list)
        for pa, ta in product_mapping.items():
            if ta is not None:
                reverse_map[ta].append(pa)

        grp = markup_df[markup_df["nm_id"] == nm_id]
        pairs = []
        for _, row in grp.iterrows():
            true_labels = row["true_labels_parsed"]
            if not true_labels:
                continue
            rid = row["id"]
            pred_scores = per_review_pred.get(rid, {})

            for true_asp, true_score in true_labels.items():
                pred_asp_list = reverse_map.get(true_asp, [])
                if not pred_asp_list:
                    continue

                # Собираем все pred scores для этого true aspect в данном review
                found_scores = []
                for pa in pred_asp_list:
                    if pa in pred_scores:
                        found_scores.append(pred_scores[pa])

                if found_scores:
                    # Mean-агрегация: не подглядываем в true_score → no leakage
                    mean_pred = float(np.mean(found_scores))
                    pairs.append((mean_pred, true_score))

        pairs_by_product[nm_id] = pairs

    return pairs_by_product


def _fit_calibration(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Линейная регрессия S_cal = a*S_raw + b. Возвращает (a, b)."""
    from scipy.stats import linregress
    if len(pairs) < 5:
        return 1.0, 0.0
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    res = linregress(x, y)
    return float(res.slope), float(res.intercept)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FIX2: _validate_mapping — проверка полноты MAPPING                     │
# └─────────────────────────────────────────────────────────────────────────┘

def _validate_mapping(
    pipeline_results: Dict[int, dict],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> None:
    """Печатает WARNING для каждого predicted-аспекта, которого нет в MAPPING.

    Без этого precision занижается из-за неполноты MAPPING, а не из-за модели.
    """
    print("\n" + "=" * 70)
    print("ВАЛИДАЦИЯ ПОЛНОТЫ MAPPING")
    print("=" * 70)

    total_missing = 0
    for nm_id, pred_data in pipeline_results.items():
        pred_aspects = set(pred_data["aspects"])
        mapped_aspects = set(mapping.get(nm_id, {}).keys())
        unmapped = pred_aspects - mapped_aspects

        if unmapped:
            total_missing += len(unmapped)
            print(f"\n⚠ nm_id={nm_id}: {len(unmapped)} predicted аспектов НЕТ в MAPPING:")
            for asp in sorted(unmapped):
                print(f"    → \"{asp}\": None,  # TODO: замапить или пометить как мусор")

    if total_missing == 0:
        print("\n✓ Все predicted-аспекты присутствуют в MAPPING.")
    else:
        print(f"\n✗ ИТОГО: {total_missing} unmapped аспектов.")
        print("  Precision занижается! Добавь их в MAPPING перед замером метрик.")
    print("=" * 70 + "\n")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FIX4: diagnose_unknown_reviews — диагностика unknown review_id         │
# └─────────────────────────────────────────────────────────────────────────┘

def diagnose_unknown_reviews(per_review_path: str) -> Dict[int, dict]:
    """Загружает eval_per_review.json и считает долю unknown review_id.

    Вызывай после step12 для диагностики.
    Возвращает: {nm_id: {"total_entries": N, "unknown_count": K, "unknown_pct": ...}}
    """
    with open(per_review_path, "r", encoding="utf-8") as f:
        per_review = json.load(f)

    print("\n" + "=" * 70)
    print("ДИАГНОСТИКА: unknown review_id")
    print("=" * 70)

    results = {}
    total_all = 0
    unknown_all = 0

    for nm_id_str, reviews_dict in per_review.items():
        total = len(reviews_dict)
        unknown_count = 0
        unknown_aspects = []

        for rid, aspects in reviews_dict.items():
            if rid == "unknown":
                unknown_count += 1
                unknown_aspects = list(aspects.keys())

        total_all += total
        unknown_all += unknown_count

        results[int(nm_id_str)] = {
            "total_entries": total,
            "unknown_count": unknown_count,
            "unknown_pct": round(100 * unknown_count / total, 1) if total else 0,
        }

        status = "✓" if unknown_count == 0 else "⚠"
        print(f"  {status} nm_id={nm_id_str}: {unknown_count}/{total} "
              f"({results[int(nm_id_str)]['unknown_pct']}%) unknown")
        if unknown_aspects:
            print(f"    аспекты в unknown: {unknown_aspects}")

    print(f"\n  ИТОГО: {unknown_all}/{total_all} "
          f"({round(100 * unknown_all / total_all, 1) if total_all else 0}%) unknown")
    print("=" * 70 + "\n")
    return results


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FIX2+FIX3: evaluate_with_mapping — валидация + честный LOO             │
# └─────────────────────────────────────────────────────────────────────────┘

def evaluate_with_mapping(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, dict],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> Dict[str, object]:
    """
    mapping: {nm_id: {predicted_aspect: true_aspect_or_None, ...}}

    FIX2: Валидация полноты MAPPING в начале.
    FIX3: Калибровка — честный LOO по товарам (train на 4, test на 1).
    """
    # FIX2: валидация полноты MAPPING
    _validate_mapping(pipeline_results, mapping)

    score_pairs = _collect_score_pairs(markup_df, pipeline_results, mapping)
    nm_ids = list(pipeline_results.keys())

    all_precision_hits = 0
    all_precision_total = 0
    all_recall_hits = 0
    all_recall_total = 0
    all_mae_raw = []
    all_mae_cal = []

    per_product = {}

    for nm_id in nm_ids:
        pred_data = pipeline_results[nm_id]
        product_mapping = mapping.get(nm_id, {})
        pred_aspects = set(pred_data["aspects"])

        grp = markup_df[markup_df["nm_id"] == nm_id]
        true_aspects_all = set()
        for labels in grp["true_labels_parsed"].dropna():
            true_aspects_all.update(labels.keys())

        mapped_pred = {pa for pa, ta in product_mapping.items() if ta is not None}
        prec_hits = len(mapped_pred)
        prec_total = len(pred_aspects)

        mapped_true = {ta for ta in product_mapping.values() if ta is not None}
        recall_hits = len(mapped_true)
        recall_total = len(true_aspects_all)

        # FIX3: честный LOO — train на ВСЕХ КРОМЕ текущего товара
        loo_train_pairs = []
        for pid in nm_ids:
            if pid != nm_id:
                loo_train_pairs.extend(score_pairs.get(pid, []))
        a, b = _fit_calibration(loo_train_pairs)

        test_pairs = score_pairs.get(nm_id, [])
        mae_raw_errs = [abs(p - t) for p, t in test_pairs]
        mae_cal_errs = [abs(np.clip(a * p + b, 1, 5) - t) for p, t in test_pairs]

        precision = prec_hits / prec_total if prec_total else 0
        recall = recall_hits / recall_total if recall_total else 0
        mae_raw = float(np.mean(mae_raw_errs)) if mae_raw_errs else None
        mae_cal = float(np.mean(mae_cal_errs)) if mae_cal_errs else None

        per_product[nm_id] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "mae_raw": round(mae_raw, 3) if mae_raw is not None else None,
            "mae_calibrated": round(mae_cal, 3) if mae_cal is not None else None,
            "calibration_a": round(a, 4),
            "calibration_b": round(b, 4),
            "mae_n": len(test_pairs),
            "pred_aspects": sorted(pred_aspects),
            "true_aspects": sorted(true_aspects_all),
        }

        all_precision_hits += prec_hits
        all_precision_total += prec_total
        all_recall_hits += recall_hits
        all_recall_total += recall_total
        all_mae_raw.extend(mae_raw_errs)
        all_mae_cal.extend(mae_cal_errs)

    macro_precision = np.mean([p["precision"] for p in per_product.values()])
    macro_recall = np.mean([p["recall"] for p in per_product.values()])
    micro_precision = all_precision_hits / all_precision_total if all_precision_total else 0
    micro_recall = all_recall_hits / all_recall_total if all_recall_total else 0
    global_mae_raw = float(np.mean(all_mae_raw)) if all_mae_raw else None
    global_mae_cal = float(np.mean(all_mae_cal)) if all_mae_cal else None

    # Global calibration coefficients (train on all — только для production/inference)
    all_pairs = []
    for pairs in score_pairs.values():
        all_pairs.extend(pairs)
    a_global, b_global = _fit_calibration(all_pairs)

    return {
        "per_product": per_product,
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "global_mae_raw": round(global_mae_raw, 3) if global_mae_raw is not None else None,
        "global_mae_calibrated": round(global_mae_cal, 3) if global_mae_cal is not None else None,
        "global_mae_n": len(all_mae_raw),
        "calibration_global": {"a": round(a_global, 4), "b": round(b_global, 4)},
    }


# ── MAPPING ────────────────────────────────────────────────────────────────
# Обновлён по реальным predicted aspects из eval_results_step1_2.json.
# Аспекты которых не было в старом MAPPING помечены # NEW.
# Аспекты из старого MAPPING, которых нет в свежем прогоне — удалены.
# Строки с # TODO требуют ручной проверки по keywords из step12.

MAPPING: Dict[int, Dict[str, Optional[str]]] = {

    # ── nm_id=117808756 (книга) ──────────────────────────────────────
    # pred: ["листы", "шрифт", "цену"]  (3)
    # true: Внешний вид, Впечатление, Запах, Качество, Логистика,
    #       Содержание, Соответствие, Текст, Удобство, Упаковка, Цена (12)
    117808756: {
        "листы": "Качество",            # keywords: страницы, обложка, бумага, печать
        "шрифт": "Содержание",          # keywords: текст, шрифт, формат
        "цену": "Цена",                 # keywords: качество, цена
        # MISS: Внешний вид, Впечатление, Запах, Логистика, Соответствие, Упаковка, и др.
    },

    # ── nm_id=254445126 (толстовка) ──────────────────────────────────
    # pred: ["размер", "отличное качество", "нитки неровные",
    #        "цвет насыщенный", "ткань плотная", "стирки",
    #        "толстовка отличная", "денег 100"]  (8)
    # true: Ассортимент, Внешний вид, Запах, Качество, Комплектация,
    #       Комфорт, Общее впечатление, Соответствие, Уход, Цена (10)
    254445126: {
        "отличное качество": "Качество",      # keywords: качество, бренд
        "нитки неровные": None,                # шум (швы — деталь качества)
        "цвет насыщенный": "Внешний вид",      # keywords: цвет
        "ткань плотная": "Комфорт",            # keywords: ткань, материал
        "стирки": None,                        # шум (уход, но keywords: наклейки, продавцам)
        "толстовка отличная": "Общее впечатление",  # keywords: кофта, худак, пошив
        "размер": "Соответствие",              # NEW! keywords: размер, рост, форму
        "денег 100": "Цена",                   # keywords: цена, денег
        # MISS: Ассортимент, Запах, Комплектация, Уход
    },

    # ── nm_id=311233470 (платье) ─────────────────────────────────────
    # pred: ["прожженая ткань", "балды", "Цена", "продавца", "Логистика",
    #        "модель классная", "крой платья", "фигуре", "складок все",
    #        "размер", "рост 165", "классное платье", "качество",
    #        "идеально сборки", "пользу другого", "фасон цвет",
    #        "комплиментов"]  (17)
    # true: Внешний вид, Качество, Комфорт, Логистика, Продавец,
    #       Соответствие, Состояние, Упаковка, Цена (11)
    311233470: {
        "прожженая ткань": "Качество",         # keywords: ткань, качество ткани
        "качество": "Качество",                # NEW — many-to-one с "прожженая ткань"
        "Цена": "Цена",                        # keywords: цена, денег
        "продавца": "Продавец",                # keywords: продавец, покупки, товар
        "Логистика": "Упаковка",               # keywords: пакет, упаковка
        "размер": "Соответствие",              # NEW — keywords: размер
        "фигуре": "Соответствие",              # keywords: фигуре (может отсутствовать в новом прогоне)
        "классное платье": "Внешний вид",      # keywords: платье, красивое
        "фасон цвет": "Внешний вид",           # NEW — many-to-one с "классное платье"
        "идеально сборки": "Состояние",        # NEW — keywords: сборки, идеально
        "модель классная": None,               # шум / дубль Внешнего вида  # TODO: можно → "Внешний вид"
        "балды": None,                         # keywords: нитки, возврат, шов, подплечники — мусорный кластер
        "крой платья": None,                   # шум
        "складок все": None,                   # шум
        "рост 165": None,                      # шум
        "пользу другого": None,                # шум
        "комплиментов": None,                  # шум (кучу комплиментов, отзывов)
        # MISS: Комфорт(10), Логистика (true, отдельно от Упаковки)
    },

    # ── nm_id=441378025 (портсигар) ──────────────────────────────────
    # pred: ["качество", "упаковка хлипкая", "прикуривателя", "пальцем",
    #        "тугая пружина", "сигареты", "портсигар", "размера компактен"]  (8)
    # true: Вместимость, Внешний вид, Запах, Качество, Комплектация,
    #       Логистика, Соответствие, Состояние, Удобство, Упаковка,
    #       Функционал, Цена (12)
    441378025: {
        "качество": "Качество",                # keywords: качество, цена
        "упаковка хлипкая": "Логистика",       # keywords: доставка, упаковка
        "прикуривателя": "Функционал",         # keywords: прикуриватель, клавиша
        "пальцем": "Удобство",                 # keywords: кнопка, индикатор
        "тугая пружина": None,                 # деталь Функционала
        "сигареты": "Вместимость",             # keywords: сигарет, сигареты
        "портсигар": None,                     # общий шум
        "размера компактен": "Внешний вид",    # keywords: дизайн, компактность
        # MISS: Цена(2), Соответствие(4), Состояние(3), Запах, Комплектация, Упаковка
    },

    # ── nm_id=506358703 (кошачий корм) ───────────────────────────────
    # pred: ["кусочки порции", "беру корма", "проблем", "состав отличный",
    #        "упаковка пакет", "лакомство", "цена качество", "Логистика",
    #        "ветеринара", "питомцев", "реакция", "коту корм",
    #        "аллергии"]  (13)
    # true: Ассортимент, Запах, Здоровье, Качество, Логистика,
    #       Поедаемость, Свежесть, Состав, Упаковка, Цена (11)
    506358703: {
        "кусочки порции": "Поедаемость",       # keywords: гранулы, порции
        "беру корма": "Качество",              # NEW — keywords: корм, составу корм, еды
        "состав отличный": "Состав",           # keywords: состав
        "упаковка пакет": "Упаковка",          # keywords: упаковка, клапан
        "лакомство": None,                     # NEW — keywords: замок, застёжка, лакомство (деталь Упаковки / Поедаемости)  # TODO: мб "Упаковка"?
        "цена качество": "Цена",               # NEW — keywords: цена, качество, годности, срок
        "Логистика": "Логистика",              # keywords: доставка
        "реакция": "Здоровье",                 # keywords: реакция, эффект
        "проблем": None,                       # NEW — keywords: никаких проблем, недостатков (слишком общий)
        "ветеринара": None,                    # шум
        "питомцев": None,                      # шум
        "коту корм": None,                     # дубль "беру корма"
        "аллергии": None,                      # деталь Здоровья  # TODO: мб → "Здоровье"?
        # MISS: Свежесть(9), Ассортимент(3), Запах(1)
    },
}


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "step12", "step4", "--step4"],
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--csv-path", type=str, default="parser/razmetka/checked_reviews.csv")
    parser.add_argument("--json-path", type=str, default="parser/razmetka/longest_reviews.json")
    parser.add_argument("--write-prefix", type=str, default="")
    args = parser.parse_args()

    mode = args.mode
    cfg = {}
    if args.config:
        cfg = load_eval_config(args.config)

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_global_seed(seed)

    overrides = cfg.get("overrides", {})
    if isinstance(overrides, dict):
        apply_config_overrides(overrides)

    CSV_PATH = str(cfg.get("csv_path", args.csv_path))
    JSON_PATH = str(cfg.get("json_path", args.json_path))
    write_prefix = str(cfg.get("write_prefix", args.write_prefix or "")).strip()
    if write_prefix and not write_prefix.endswith("_"):
        write_prefix = f"{write_prefix}_"

    print(f"[Eval] seed={seed}")
    if args.config:
        print(f"[Eval] config={args.config}")

    df = load_markup(CSV_PATH)

    if mode in ("all", "step12"):
        print("=" * 70)
        print("ШАГ 1: СТАТИСТИКА ПО РАЗМЕТКЕ")
        print("=" * 70)

        stats = markup_stats(df)
        for _, row in stats.iterrows():
            print(f"\nnm_id: {row['nm_id']}")
            print(f"  Отзывов всего:    {row['total_reviews']}")
            print(f"  С разметкой:      {row['labeled_reviews']}")
            print(f"  Уникальные аспекты: {row['unique_aspects']}")
            print(f"  Средние оценки:")
            for asp, info in row['aspect_avg_scores'].items():
                print(f"    {asp:25s} {info}")

        nm_ids = stats["nm_id"].tolist()

        print(f"\n{'='*70}")
        print("ШАГ 2: ПРОГОН ПАЙПЛАЙНА")
        print("=" * 70)

        pipeline_results = run_pipeline_for_ids(nm_ids, CSV_PATH, JSON_PATH)

        for nm_id, data in pipeline_results.items():
            print(f"\nnm_id={nm_id}")
            print(f"  Аспекты пайплайна: {data['aspects']}")
            if data.get("aspect_keywords"):
                for asp, kw in data["aspect_keywords"].items():
                    print(f"    {asp}: {kw[:5]}")
            print(f"  Отзывов с оценками: {len(data['per_review'])}")

        with open(f"{write_prefix}eval_results_step1_2.json", "w", encoding="utf-8") as f:
            json.dump({
                "pipeline_results": {
                    str(k): {"aspects": v["aspects"], "aspect_keywords": v.get("aspect_keywords", {})}
                    for k, v in pipeline_results.items()
                },
            }, f, ensure_ascii=False, indent=2, default=str)

        per_review_dump = {str(k): v["per_review"] for k, v in pipeline_results.items()}
        with open(f"{write_prefix}eval_per_review.json", "w", encoding="utf-8") as f:
            json.dump(per_review_dump, f, ensure_ascii=False, indent=2)

        print("\nРезультаты шагов 1-2 сохранены.")

        # FIX4: диагностика unknown review_id
        diagnose_unknown_reviews(f"{write_prefix}eval_per_review.json")

    if mode in ("all", "step4", "--step4"):
        print(f"\n{'='*70}")
        print("ШАГ 4: МЕТРИКИ (маппинг из MAPPING)")
        print("=" * 70)

        with open(f"{write_prefix}eval_per_review.json", "r", encoding="utf-8") as f:
            per_review_loaded = json.load(f)
        with open(f"{write_prefix}eval_results_step1_2.json", "r", encoding="utf-8") as f:
            step12 = json.load(f)

        pipeline_results_for_eval = {}
        for nm_id_str, info in step12["pipeline_results"].items():
            nm_id = int(nm_id_str)
            pipeline_results_for_eval[nm_id] = {
                "aspects": info["aspects"],
                "per_review": per_review_loaded.get(nm_id_str, {}),
            }

        metrics = evaluate_with_mapping(df, pipeline_results_for_eval, MAPPING)

        for nm_id, pm in metrics["per_product"].items():
            print(f"\nnm_id={nm_id}:")
            print(f"  Precision:  {pm['precision']}")
            print(f"  Recall:     {pm['recall']}")
            print(f"  MAE raw:    {pm['mae_raw']}  →  calibrated: {pm['mae_calibrated']}  "
                  f"(n={pm['mae_n']}, a={pm['calibration_a']}, b={pm['calibration_b']})")
            print(f"  Predicted:  {pm['pred_aspects']}")
            print(f"  True:       {pm['true_aspects']}")

        cal = metrics["calibration_global"]
        print(f"\n{'='*70}")
        print("ИТОГО:")
        print(f"  Macro Precision:     {metrics['macro_precision']}")
        print(f"  Macro Recall:        {metrics['macro_recall']}")
        print(f"  Micro Precision:     {metrics['micro_precision']}")
        print(f"  Micro Recall:        {metrics['micro_recall']}")
        print(f"  Global MAE raw:      {metrics['global_mae_raw']}  (n={metrics['global_mae_n']})")
        print(f"  Global MAE calibr:   {metrics['global_mae_calibrated']}")
        print(f"  Global calibration:  S_cal = {cal['a']}*S_raw + {cal['b']}")

        with open(f"{write_prefix}eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nМетрики сохранены в {write_prefix}eval_metrics.json")