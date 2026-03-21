"""
Evaluation pipeline: разметка vs пайплайн.

Шаги:
  1. Статистика по разметке
  2. Прогон Discovery + Sentiment
  3. Маппинг аспектов (ручная таблица)
  4. Метрики: Precision, Recall, Sentiment MAE
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


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
            results[nm_id] = {"aspects": [], "per_review": []}
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

        results[nm_id] = {
            "aspects": aspect_names,
            "per_review": per_review_avg,
            "aspect_keywords": {
                name: info.keywords for name, info in aspects.items()
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

def evaluate_with_mapping(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, dict],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> Dict[str, object]:
    """
    mapping: {nm_id: {predicted_aspect: true_aspect_or_None, ...}}
    Аспекты из разметки, которых нет в mapping, считаются пропущенными (recall penalty).
    """
    all_precision_hits = 0
    all_precision_total = 0
    all_recall_hits = 0
    all_recall_total = 0
    all_mae_errors = []

    per_product = {}

    for nm_id, pred_data in pipeline_results.items():
        product_mapping = mapping.get(nm_id, {})
        pred_aspects = set(pred_data["aspects"])
        per_review_pred = pred_data["per_review"]

        grp = markup_df[markup_df["nm_id"] == nm_id]
        true_aspects_all = set()
        for labels in grp["true_labels_parsed"].dropna():
            true_aspects_all.update(labels.keys())

        # --- Discovery Precision ---
        mapped_pred = {pa for pa, ta in product_mapping.items() if ta is not None}
        prec_hits = len(mapped_pred)
        prec_total = len(pred_aspects)

        # --- Discovery Recall ---
        mapped_true = {ta for ta in product_mapping.values() if ta is not None}
        recall_hits = len(mapped_true)
        recall_total = len(true_aspects_all)

        # --- Sentiment MAE ---
        reverse_map = {ta: pa for pa, ta in product_mapping.items() if ta is not None}
        mae_errors = []
        for _, row in grp.iterrows():
            true_labels = row["true_labels_parsed"]
            if not true_labels:
                continue
            rid = row["id"]
            pred_scores = per_review_pred.get(rid, {})
            for true_asp, true_score in true_labels.items():
                pred_asp = reverse_map.get(true_asp)
                if pred_asp and pred_asp in pred_scores:
                    err = abs(pred_scores[pred_asp] - true_score)
                    mae_errors.append(err)

        precision = prec_hits / prec_total if prec_total else 0
        recall = recall_hits / recall_total if recall_total else 0
        mae = float(np.mean(mae_errors)) if mae_errors else None

        per_product[nm_id] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "mae": round(mae, 3) if mae is not None else None,
            "mae_n": len(mae_errors),
            "pred_aspects": sorted(pred_aspects),
            "true_aspects": sorted(true_aspects_all),
        }

        all_precision_hits += prec_hits
        all_precision_total += prec_total
        all_recall_hits += recall_hits
        all_recall_total += recall_total
        all_mae_errors.extend(mae_errors)

    macro_precision = np.mean([p["precision"] for p in per_product.values()])
    macro_recall = np.mean([p["recall"] for p in per_product.values()])
    micro_precision = all_precision_hits / all_precision_total if all_precision_total else 0
    micro_recall = all_recall_hits / all_recall_total if all_recall_total else 0
    global_mae = float(np.mean(all_mae_errors)) if all_mae_errors else None

    return {
        "per_product": per_product,
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "global_mae": round(global_mae, 3) if global_mae is not None else None,
        "global_mae_n": len(all_mae_errors),
    }


# ── CLI ────────────────────────────────────────────────────────────────────

MAPPING: Dict[int, Dict[str, Optional[str]]] = {
        # nm_id=117808756 (книга)
        # predicted → true
        # "листы"              → keywords: страницы, обложка, бумага   → Качество
        # "шрифт"              → keywords: текст, шрифт               → Содержание
        # "цену"               → keywords: качество, цена             → Цена
        # "бумага качественная" → keywords: упаковку, доставка        → Упаковка
        # "листы белые"        → keywords: листы белые, белые         → Внешний вид
        # MISS true: Логистика(10), Соответствие(9), Запах(7)
        117808756: {
            "листы": "Качество",
            "шрифт": "Содержание",
            "цену": "Цена",
        },
        # nm_id=254445126 (толстовка)
        # "размер"             → Соответствие (размер, рост, форму)
        # "отличное качество"  → Качество
        # "продавцу"           → None (нет в разметке)
        # "цвет"               → Внешний вид (цвет, фото)
        # "ткань плотная"      → Комфорт (материал, капюшон тяжёлый)
        # "стирки"             → Качество — дубль, лучше Уход? (n=1, малозначим) → None
        # "толстовка отличная" → Общее впечатление
        # "класс"              → None (шум)
        # MISS true: Цена(9)
        254445126: {
            "размер": "Соответствие",
            "отличное качество": "Качество",
            "нитки неровные": None,
            "цвет насыщенный": "Внешний вид",
            "ткань плотная": "Комфорт",
            "стирки": None,
            "толстовка отличная": "Общее впечатление",
            "денег 100": "Цена",
        },
        # nm_id=311233470 (платье)
        # "ткань"              → Качество (качество ткани)
        # "Цена"               → Цена
        # "продавец"           → Продавец
        # "Логистика"          → Упаковка (фирменная упаковка, пакет)
        # "фигуре"             → Соответствие (фото, фигуре)
        # "шов"                → Состояние (возврат, шов, сборка)
        # "складок все"        → None (шум, пересекается с Качество)
        # "классное платье"    → Внешний вид (красивое платье, крой)
        # "6000"               → None (дубль Цена)
        # "ужасное качество"   → None (дубль Качество)
        # "платья"             → None (общий шум)
        # "идеально сборки"    → None (шум, мало)
        # "пользу другого"     → None (шум)
        # "фасон цвет"         → None (дубль Внешний вид)
        # "комплиментов"       → None (эмоция)
        311233470: {
            "прожженая ткань": "Качество",
            "балды": "Состояние",
            "Цена": "Цена",
            "продавца": "Продавец",
            "Логистика": "Упаковка",
            "модель классная": None,
            "крой платья": None,
            "складок все": None,
            "размер": "Соответствие",
            "рост 165": None,
            "классное платье": "Внешний вид",
            "качество": None,
            "идеально сборки": None,
            "пользу другого": None,
            "фасон цвет": None,
            "комплиментов": None,
        },
        # nm_id=441378025 (портсигар)
        # "качество"           → Качество (+ цена в keywords → частично Цена)
        # "упаковка хлипкая"   → Логистика (доставка, упаковка → нет "Упаковка" в true, но Логистика есть)
        # "прикуривателя"      → Функционал (прикуриватель, клавиша)
        # "пальцем"            → Удобство (руке удобно, кнопка)
        # "пружина"            → None (деталь Функционала, дубль)
        # "формами"            → Внешний вид (обтекаемые формы, компакт)
        # "сигареты"           → Вместимость (сигарет, тонкие сигареты)
        # "нормальный портсигар" → None (общий шум)
        # "дизайн компактность" → None (дубль Внешний вид)
        # MISS true: Соответствие(4), Состояние(3)
        441378025: {
            "качество": "Качество",
            "упаковка хлипкая": "Логистика",
            "прикуривателя": "Функционал",
            "пальцем": "Удобство",
            "тугая пружина": None,
            "сигареты": "Вместимость",
            "портсигар": None,
            "размера компактен": "Внешний вид",
        },
        # nm_id=506358703 (кошачий корм)
        # "кусочки порции"     → Поедаемость (размер гранул, порции)
        # "корм"               → Качество (корм, пищеварение)
        # "проблем"            → Здоровье (никаких проблем)
        # "составу"            → Состав
        # "упаковка пакет"     → Упаковка
        # "удобная застёжка"   → None (дубль Упаковка)
        # "срокам"             → Свежесть (срок годности, свежий)
        # "Логистика"          → Логистика (доставка)
        # "ветиренар"          → None (шум)
        # "питомцев"           → None (шум)
        # "кошки довольны"     → None (эмоция)
        # "аллергии"           → None (дубль Здоровье)
        # MISS true: Цена(12), Ассортимент(3), Запах(1)
        506358703: {
            "кусочки порции": "Поедаемость",
            "беру корма": "Качество",
            "проблем": "Здоровье",
            "состав отличный": "Состав",
            "упаковка пакет": "Упаковка",
            "лакомство": None,
            "цена качество": "Цена",
            "Логистика": "Логистика",
            "ветеринара": None,
            "питомцев": None,
            "реакция": None,
            "коту корм": None,
            "аллергии": None,
        },
    }



if __name__ == "__main__":
    CSV_PATH = "parser/razmetka/checked_reviews.csv"
    JSON_PATH = "parser/razmetka/longest_reviews.json"

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

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

        with open("eval_results_step1_2.json", "w", encoding="utf-8") as f:
            json.dump({
                "pipeline_results": {
                    str(k): {"aspects": v["aspects"], "aspect_keywords": v.get("aspect_keywords", {})}
                    for k, v in pipeline_results.items()
                },
            }, f, ensure_ascii=False, indent=2, default=str)

        per_review_dump = {str(k): v["per_review"] for k, v in pipeline_results.items()}
        with open("eval_per_review.json", "w", encoding="utf-8") as f:
            json.dump(per_review_dump, f, ensure_ascii=False, indent=2)

        print("\nРезультаты шагов 1-2 сохранены.")

    if mode in ("all", "step4", "--step4"):
        print(f"\n{'='*70}")
        print("ШАГ 4: МЕТРИКИ (маппинг из MAPPING)")
        print("=" * 70)

        with open("eval_per_review.json", "r", encoding="utf-8") as f:
            per_review_loaded = json.load(f)
        with open("eval_results_step1_2.json", "r", encoding="utf-8") as f:
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
            print(f"  Precision: {pm['precision']}")
            print(f"  Recall:    {pm['recall']}")
            print(f"  MAE:       {pm['mae']}  (n={pm['mae_n']})")
            print(f"  Predicted: {pm['pred_aspects']}")
            print(f"  True:      {pm['true_aspects']}")

        print(f"\n{'='*70}")
        print("ИТОГО:")
        print(f"  Macro Precision: {metrics['macro_precision']}")
        print(f"  Macro Recall:    {metrics['macro_recall']}")
        print(f"  Micro Precision: {metrics['micro_precision']}")
        print(f"  Micro Recall:    {metrics['micro_recall']}")
        print(f"  Global MAE:      {metrics['global_mae']}  (n={metrics['global_mae_n']})")

        with open("eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("\nМетрики сохранены в eval_metrics.json")
