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
import os
import random
import sys
from collections import Counter, defaultdict
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


def load_pipeline_reviews_from_csv(csv_path: str, nm_ids: List[int]) -> List[dict]:
    """
    Те же поля, что в longest_reviews.json, для ReviewInput.
    Берётся из разметочного CSV (например merged_checked_reviews.csv).
    """
    df = pd.read_csv(csv_path, dtype={"id": str})
    df = df[df["nm_id"].isin(nm_ids)]
    out: List[dict] = []
    for _, row in df.iterrows():
        ft = row.get("full_text")
        pr = row.get("pros")
        cn = row.get("cons")
        out.append(
            {
                "nm_id": int(row["nm_id"]),
                "id": str(row["id"]),
                "rating": int(row["rating"]),
                "created_date": str(row["created_date"]).strip(),
                "full_text": "" if pd.isna(ft) else str(ft),
                "pros": "" if pd.isna(pr) else str(pr),
                "cons": "" if pd.isna(cn) else str(cn),
            }
        )
    return out


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
    json_path: Optional[str] = None,
) -> Dict[int, dict]:
    """
    Прогоняет пайплайн на отзывах: по умолчанию из csv_path (как merged_checked_reviews.csv),
    либо из json_path, если файл задан и существует (старый режим longest_reviews.json).
    Возвращает {nm_id: {"aspects": [...], "per_review": [{...}]}}
    """
    from src.discovery.candidates import CandidateExtractor
    from src.discovery.clusterer import AspectClusterer
    from src.discovery.scorer import KeyBERTScorer
    from src.fraud.engine import AntiFraudEngine
    from src.sentiment.engine import SentimentEngine

    from sentence_transformers import SentenceTransformer
    from configs.configs import config

    if json_path and os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            all_reviews_raw = json.load(f)
    else:
        all_reviews_raw = load_pipeline_reviews_from_csv(csv_path, nm_ids)
        if json_path:
            print(
                f"[Eval] json_path={json_path!r} не найден — отзывы из CSV: {csv_path}"
            )

    reviews_by_nm = defaultdict(list)
    for r in all_reviews_raw:
        reviews_by_nm[r["nm_id"]].append(r)

    encoder = SentenceTransformer(config.models.encoder_path)
    extractor = CandidateExtractor()
    scorer = KeyBERTScorer(model=encoder)
    clusterer = AspectClusterer(model=encoder)
    fraud = AntiFraudEngine()
    sentiment = SentimentEngine()

    print(
        f"[Eval] multi_label: threshold={config.discovery.multi_label_threshold}, "
        f"max_aspects={config.discovery.multi_label_max_aspects}"
    )

    results = {}
    total_nli_pairs = 0

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
        # FIX: строим sentence_to_review ИЗ РЕАЛЬНЫХ candidate.sentence,
        # а не из повторного split. Это устраняет 29% потерь на string mismatch.
        sentence_to_review = {}
        for i, text in enumerate(texts):
            candidates = extractor.extract(text)
            for c in candidates:
                sentence_to_review[c.sentence.strip()] = reviews[i].id
                sentence_to_review[c.sentence.lower().strip()] = reviews[i].id
            all_candidates.extend(candidates)

        scored = scorer.score_and_select(all_candidates)
        aspects = clusterer.cluster(scored)
        aspect_names = list(aspects.keys())
        print(f"  Аспекты: {aspect_names}")

        ac = getattr(clusterer, "last_assignment_counts", {})
        c_assign = Counter(ac)
        medoid_names = list(getattr(clusterer, "last_residual_medoid_names", []))
        print(f"  Counter (confident / residual): {c_assign}")
        print(f"  Residual medoid names (final): {medoid_names}")

        nli_lines = list(getattr(clusterer, "last_nli_medoid_diagnostics", []))

        if not aspects:
            results[nm_id] = {
                "aspects": [],
                "per_review": {},
                "diagnostics": {
                    "anchor_assignment_counts": dict(ac),
                    "residual_medoid_names": medoid_names,
                    "nli_medoid_diagnostics": nli_lines,
                },
            }
            continue

        pairs = _build_pairs(scored, aspects, sentence_to_review, clusterer)
        n_nli = len(pairs)
        total_nli_pairs += n_nli
        print(f"  NLI пар: {n_nli}")
        sentiment_scores = sentiment.batch_analyze(pairs)

        per_review = defaultdict(dict)
        for sr in sentiment_scores:
            if sr.aspect not in per_review[sr.review_id]:
                per_review[sr.review_id][sr.aspect] = []
            per_review[sr.review_id][sr.aspect].append((sr.score, sr.confidence))

        per_review_avg = {}
        for rid, asp_dict in per_review.items():
            avg = {}
            for asp, pairs_sw in asp_dict.items():
                scores = [p[0] for p in pairs_sw]
                weights = [p[1] for p in pairs_sw]
                wsum = float(sum(weights))
                if wsum > 0:
                    avg[asp] = round(
                        float(sum(s * w for s, w in zip(scores, weights)) / wsum),
                        2,
                    )
                else:
                    avg[asp] = round(float(np.mean(scores)), 2)
            # Apply aliases: "Органолептика"→"Запах" etc.
            # Добавляем aliased ключи чтобы identity match работал
            for pred_name, true_name in ASPECT_ALIASES.items():
                if pred_name in avg and true_name not in avg:
                    avg[true_name] = avg[pred_name]
            per_review_avg[rid] = avg

        # Диагностика: какие кандидаты были до кластеризации
        all_cand_texts = [c.span for c in all_candidates]
        scored_texts = [c.span for c in scored]

        results[nm_id] = {
            "aspects": aspect_names + [
                ASPECT_ALIASES[a] for a in aspect_names if a in ASPECT_ALIASES
            ],
            "per_review": per_review_avg,
            "aspect_keywords": {
                name: info.keywords for name, info in aspects.items()
            },
            "diagnostics": {
                "raw_candidates_count": len(all_candidates),
                "scored_candidates_count": len(scored),
                "raw_candidate_samples": all_cand_texts[:30],
                "scored_candidate_samples": scored_texts,
                "anchor_assignment_counts": dict(ac),
                "residual_medoid_names": medoid_names,
                "nli_medoid_diagnostics": nli_lines,
                "nli_pairs_count": n_nli,
            },
        }

    print(f"\n[Eval] Всего NLI пар (по всем nm_id): {total_nli_pairs}")

    return results


def _build_pairs(scored, aspects, sentence_to_review, clusterer):
    """
    Multi-label: cos(span, anchor) >= threshold — до max_aspects якорей на кандидата.
    (review_id, sentence, aspect_name, nli_label, weight) — как в pipeline.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    from configs.configs import config

    if not aspects or not scored:
        return []

    threshold = float(config.discovery.multi_label_threshold)
    max_aspects = int(config.discovery.multi_label_max_aspects)

    anchor_names = list(clusterer._anchor_embeddings.keys())
    anchor_matrix = np.stack(
        [clusterer._anchor_embeddings[n] for n in anchor_names]
    )

    product_anchors: set[str] = set()
    for asp_name, info in aspects.items():
        nli = (getattr(info, "nli_label", None) or asp_name).strip() or asp_name
        if nli in clusterer._anchor_embeddings:
            product_anchors.add(nli)
        if asp_name in clusterer._anchor_embeddings:
            product_anchors.add(asp_name)

    seen: set = set()
    pairs: List[Tuple[str, str, str, str, float]] = []

    for cand in scored:
        emb = np.asarray(cand.embedding, dtype=np.float64).reshape(1, -1)
        sims = cosine_similarity(emb, anchor_matrix)[0]

        cand_anchors: List[Tuple[str, float]] = []
        for idx, sim in enumerate(sims):
            aname = anchor_names[idx]
            if sim >= threshold and aname in product_anchors:
                cand_anchors.append((aname, float(sim)))

        cand_anchors.sort(key=lambda x: x[1], reverse=True)
        cand_anchors = cand_anchors[:max_aspects]

        if not cand_anchors:
            continue

        review_id = sentence_to_review.get(
            cand.sentence.strip(),
            sentence_to_review.get(cand.sentence.lower().strip(), "unknown"),
        )

        for aname, sim in cand_anchors:
            key = (review_id, cand.sentence, aname)
            if key not in seen:
                seen.add(key)
                pairs.append(
                    (review_id, cand.sentence, aname, aname, float(sim)),
                )

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
    all_mention_found = 0
    all_mention_total = 0
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

        # Reverse map for per-review coverage: true_asp → [pred_asp, ...]
        reverse_map: Dict[str, List[str]] = defaultdict(list)
        for pa, ta in product_mapping.items():
            if ta is not None:
                reverse_map[ta].append(pa)

        per_review_pred = pred_data.get("per_review", {})

        # 3-tier mention-level recall:
        #   product_covered: true aspect TYPE exists in mapping (current, inflated)
        #   review_covered:  this review has ANY pred score for mapped pred aspects
        #   score_covered:   this review has pred score for THIS SPECIFIC true aspect's mapped pred
        mention_total = 0
        mention_product = 0   # tier 1: product-level (inflated)
        mention_review = 0    # tier 2: per-review (honest)

        for _, row in grp.iterrows():
            true_labels = row["true_labels_parsed"]
            if not true_labels:
                continue
            rid = row["id"]
            pred_scores = per_review_pred.get(rid, {})

            for true_asp in true_labels:
                mention_total += 1

                # Tier 1: product-level (is this true aspect type covered at all?)
                if true_asp in mapped_true:
                    mention_product += 1

                # Tier 2: per-review (does THIS review have a score for a mapped pred?)
                pred_asps = reverse_map.get(true_asp, [])
                if any(pa in pred_scores for pa in pred_asps):
                    mention_review += 1

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
        mention_recall_product = mention_product / mention_total if mention_total else 0
        mention_recall_review = mention_review / mention_total if mention_total else 0
        mae_raw = float(np.mean(mae_raw_errs)) if mae_raw_errs else None
        mae_cal = float(np.mean(mae_cal_errs)) if mae_cal_errs else None

        per_product[nm_id] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "mention_recall_product": round(mention_recall_product, 3),
            "mention_recall_review": round(mention_recall_review, 3),
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
        all_mention_found += mention_review       # use honest per-review metric
        all_mention_total += mention_total
        all_mae_raw.extend(mae_raw_errs)
        all_mae_cal.extend(mae_cal_errs)

    macro_precision = np.mean([p["precision"] for p in per_product.values()])
    macro_recall = np.mean([p["recall"] for p in per_product.values()])
    macro_mention_recall_product = np.mean([p["mention_recall_product"] for p in per_product.values()])
    macro_mention_recall_review = np.mean([p["mention_recall_review"] for p in per_product.values()])
    micro_precision = all_precision_hits / all_precision_total if all_precision_total else 0
    micro_recall = all_recall_hits / all_recall_total if all_recall_total else 0
    global_mention_recall = all_mention_found / all_mention_total if all_mention_total else 0
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
        "macro_mention_recall_product": round(macro_mention_recall_product, 3),
        "macro_mention_recall_review": round(macro_mention_recall_review, 3),
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "global_mention_recall_review": round(global_mention_recall, 3),
        "global_mae_raw": round(global_mae_raw, 3) if global_mae_raw is not None else None,
        "global_mae_calibrated": round(global_mae_cal, 3) if global_mae_cal is not None else None,
        "global_mae_n": len(all_mae_raw),
        "calibration_global": {"a": round(a_global, 4), "b": round(b_global, 4)},
    }


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ PRODUCT-LEVEL ASPECT RATING COMPARISON                                 │
# │                                                                         │
# │ Сравниваем агрегированные рейтинги: true_avg vs pred_avg per aspect.   │
# │ Это то, что реально видит пользователь в UI (radar chart).            │
# │ Снимает проблему 32% per-review coverage — даже если пайплайн поймал  │
# │ аспект в 20/55 reviews, агрегат может быть близок к правде.           │
# └─────────────────────────────────────────────────────────────────────────┘

def evaluate_product_ratings(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, dict],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> Dict[str, object]:
    """Product-level MAE: средняя оценка аспекта по разметке vs по пайплайну.

    Для каждого товара, для каждого true аспекта с маппингом:
      true_avg  = mean(true_scores[aspect] across all reviews)
      pred_avg  = mean(pred_scores[mapped_pred_aspects] across all reviews)
      error     = |true_avg - pred_avg|

    Returns dict с per-product и global метриками.
    """
    print("\n" + "=" * 70)
    print("PRODUCT-LEVEL ASPECT RATING COMPARISON")
    print("=" * 70)

    all_errors = []
    all_errors_filtered = []  # only n_true >= 3
    per_product = {}

    for nm_id, pred_data in pipeline_results.items():
        product_mapping = mapping.get(nm_id, {})
        per_review_pred = pred_data.get("per_review", {})

        # Reverse map: true_asp → [pred_asp1, pred_asp2, ...]
        reverse_map: Dict[str, List[str]] = defaultdict(list)
        for pa, ta in product_mapping.items():
            if ta is not None:
                reverse_map[ta].append(pa)

        # True avg scores per aspect
        grp = markup_df[markup_df["nm_id"] == nm_id]
        true_scores_by_aspect: Dict[str, List[float]] = defaultdict(list)
        for _, row in grp.iterrows():
            labels = row["true_labels_parsed"]
            if not labels:
                continue
            for asp, score in labels.items():
                true_scores_by_aspect[asp].append(score)

        # Pred avg scores per mapped true aspect
        pred_scores_by_true: Dict[str, List[float]] = defaultdict(list)
        for rid, pred_scores in per_review_pred.items():
            if rid == "unknown":
                continue
            for true_asp, pred_asps in reverse_map.items():
                found = [pred_scores[pa] for pa in pred_asps if pa in pred_scores]
                if found:
                    pred_scores_by_true[true_asp].append(float(np.mean(found)))

        # Compare
        aspect_comparisons = []
        print(f"\n  nm_id={nm_id}:")
        print(f"    {'Aspect':25s} {'True':>6s} {'Pred':>6s} {'Δ':>6s}  {'n_true':>6s} {'n_pred':>6s}")
        print(f"    {'-'*65}")

        for true_asp in sorted(true_scores_by_aspect.keys()):
            true_avg = float(np.mean(true_scores_by_aspect[true_asp]))
            n_true = len(true_scores_by_aspect[true_asp])

            if true_asp not in pred_scores_by_true or not pred_scores_by_true[true_asp]:
                print(f"    {true_asp:25s} {true_avg:6.2f}    —      —    {n_true:6d}      0")
                continue

            pred_avg = float(np.mean(pred_scores_by_true[true_asp]))
            n_pred = len(pred_scores_by_true[true_asp])
            error = abs(true_avg - pred_avg)

            aspect_comparisons.append({
                "aspect": true_asp,
                "true_avg": round(true_avg, 2),
                "pred_avg": round(pred_avg, 2),
                "error": round(error, 2),
                "n_true": n_true,
                "n_pred": n_pred,
            })

            # All errors (unfiltered)
            all_errors.append(error)
            # Filtered: only aspects with n_true >= 3 (statistically meaningful)
            if n_true >= 3:
                all_errors_filtered.append(error)

            flag = " *" if n_true < 3 else ""
            print(f"    {true_asp:25s} {true_avg:6.2f} {pred_avg:6.2f} {error:6.2f}  {n_true:6d} {n_pred:6d}{flag}")

        # Per-product MAE (filtered)
        filtered_errors = [c["error"] for c in aspect_comparisons if c["n_true"] >= 3]
        product_mae = float(np.mean(filtered_errors)) if filtered_errors else None
        product_mae_all = float(np.mean([c["error"] for c in aspect_comparisons])) if aspect_comparisons else None
        per_product[nm_id] = {
            "aspects": aspect_comparisons,
            "product_mae": round(product_mae, 3) if product_mae is not None else None,
            "product_mae_unfiltered": round(product_mae_all, 3) if product_mae_all is not None else None,
            "n_aspects_compared": len(aspect_comparisons),
        }

        if product_mae is not None:
            extra = f"  (unfiltered: {product_mae_all:.3f})" if product_mae_all != product_mae else ""
            print(f"    {'':25s} {'':>6s} {'MAE':>6s} {product_mae:6.3f}  (n_true≥3){extra}")
        print(f"    (* = n_true < 3, excluded from MAE)")

    global_mae = float(np.mean(all_errors)) if all_errors else None
    global_mae_filtered = float(np.mean(all_errors_filtered)) if all_errors_filtered else None
    macro_mae = float(np.mean([
        p["product_mae"] for p in per_product.values() if p["product_mae"] is not None
    ])) if per_product else None

    print(f"\n{'='*70}")
    print(f"  Product-Level MAE (n_true≥3): {round(global_mae_filtered, 3) if global_mae_filtered else 'N/A'}"
          f"  (n={len(all_errors_filtered)} pairs)")
    print(f"  Product-Level MAE (all):      {round(global_mae, 3) if global_mae else 'N/A'}"
          f"  (n={len(all_errors)} pairs)")
    print(f"  Product-Level MAE (macro):    {round(macro_mae, 3) if macro_mae else 'N/A'}")
    print(f"{'='*70}\n")

    return {
        "per_product": per_product,
        "global_product_mae": round(global_mae, 3) if global_mae is not None else None,
        "global_product_mae_filtered": round(global_mae_filtered, 3) if global_mae_filtered is not None else None,
        "macro_product_mae": round(macro_mae, 3) if macro_mae is not None else None,
        "n_pairs": len(all_errors),
        "n_pairs_filtered": len(all_errors_filtered),
    }


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ AUTO MAPPING: семантическое назначение predicted → true aspects        │
# │                                                                         │
# │ Greedy assignment: для каждого pred берём argmax cos(pred, true).      │
# │ Если max cos < τ → None (мусор). Many-to-one поддерживается.          │
# │                                                                         │
# │ Predicted centroid = mean(encode(keywords)).                            │
# │ True embedding = mean(encode(description_phrases)) — NOT bare name.    │
# └─────────────────────────────────────────────────────────────────────────┘

# ── ASPECT ALIASES: pred name → true name для identity matching ──────────
# Решает проблему: якорь "Органолептика" → true "Запах", якорь
# "Функциональность" → true "Функционал". Вместо переименования якорей
# (что ломает кластеризацию) — alias map в eval.
ASPECT_ALIASES: Dict[str, str] = {
    "Органолептика": "Запах",
    "Функциональность": "Функционал",
}


# Расширенные описания true aspects для домена e-commerce.
# Один словарь на все товары — domain knowledge, не per-product хардкод.
# Каждое описание — 3-6 коротких фраз, характерных для этого аспекта
# в реальных отзывах. Это решает проблему "голого имени" (encode("Функционал")
# слишком абстрактен и плохо отделяется от других аспектов в rubert-tiny2).

ASPECT_DESCRIPTIONS: Dict[str, List[str]] = {
    # ── Общие (встречаются в большинстве товаров) ────────────────────
    "Качество": [
        "качество товара", "качество материала", "хорошее качество",
        "плохое качество", "качество изготовления", "качество сборки",
    ],
    "Цена": [
        "цена", "стоимость", "дорого", "дёшево",
        "за такие деньги", "соотношение цена качество",
    ],
    "Внешний вид": [
        "красивый", "внешний вид", "выглядит", "дизайн",
        "цвет", "стильный", "смотрится",
    ],
    "Упаковка": [
        "упаковка", "пакет", "коробка", "упаковано",
        "пришло в пакете", "фирменная упаковка",
    ],
    "Логистика": [
        "доставка", "доставили быстро", "курьер", "пришло быстро",
        "долго шло", "сроки доставки",
    ],
    "Соответствие": [
        "размер подошёл", "соответствует описанию", "как на фото",
        "не соответствует", "размер не подошёл", "соответствие",
    ],
    "Соответсвие": [  # опечатка в разметке — дублируем
        "размер подошёл", "соответствует описанию", "как на фото",
        "не соответствует", "размер не подошёл", "соответствие",
    ],
    "Комфорт": [
        "удобно носить", "комфортно", "приятно к телу",
        "неудобно", "натирает", "мягкий материал",
    ],
    "Кофморт": [  # опечатка в разметке
        "удобно носить", "комфортно", "приятно к телу",
        "неудобно", "натирает", "мягкий материал",
    ],
    "Общее впечатление": [
        "в целом нравится", "общее впечатление", "рекомендую",
        "не рекомендую", "отличная вещь", "разочарование",
    ],
    "Впечатление": [
        "впечатление", "в целом", "нравится", "рекомендую",
    ],
    "Продавец": [
        "продавец", "магазин", "обслуживание", "возврат продавец",
        "ответ продавца", "отношение продавца",
    ],
    "Состояние": [
        "пришло в плохом состоянии", "брак", "дефект",
        "нитки торчат", "швы порваны", "повреждённый товар",
    ],
    "Запах": [
        "запах", "пахнет", "вонь", "аромат",
        "неприятный запах", "без запаха",
    ],

    # ── Одежда ───────────────────────────────────────────────────────
    "Уход": [
        "стирка", "после стирки", "уход за вещью",
        "стирать", "линяет", "садится после стирки",
    ],
    "Ассортимент": [
        "выбор цветов", "ассортимент", "разнообразие",
        "другие расцветки", "мало вариантов",
    ],
    "Комплектация": [
        "комплектация", "в комплекте", "не хватает",
        "полный комплект", "бирка", "пломба",
    ],

    # ── Книга ────────────────────────────────────────────────────────
    "Содержание": [
        "содержание книги", "текст", "интересно читать",
        "сюжет", "полезная информация", "содержание",
    ],
    "Текст": [
        "шрифт", "текст", "читается", "мелкий шрифт",
        "крупный шрифт", "удобно читать",
    ],
    "Удобство": [
        "удобный формат", "удобно держать", "удобство использования",
        "эргономика", "удобно пользоваться",
    ],

    # ── Портсигар / техника ──────────────────────────────────────────
    "Функционал": [
        "функционал", "работает", "механизм", "кнопка",
        "прикуриватель", "зажигание", "функция",
    ],
    "Вместимость": [
        "вместимость", "помещается", "сигареты влезают",
        "мало места", "ёмкость", "количество сигарет",
    ],

    # ── Кошачий корм ─────────────────────────────────────────────────
    "Поедаемость": [
        "кот ест", "кошка ест с удовольствием", "нравится коту",
        "не ест", "привередливый", "вкусный корм",
    ],
    "Состав": [
        "состав корма", "натуральный состав", "хороший состав",
        "ингредиенты", "без добавок", "состав",
    ],
    "Здоровье": [
        "здоровье питомца", "аллергия", "реакция",
        "шерсть блестит", "стул нормальный", "самочувствие кота",
    ],
    "Свежесть": [
        "срок годности", "свежий", "свежесть",
        "не просроченный", "дата изготовления",
    ],
    "Свеежсть": [  # опечатка в разметке
        "срок годности", "свежий", "свежесть",
        "не просроченный", "дата изготовления",
    ],
    # Логисика — опечатка в разметке для книги (nm_id=117808756)
    "Логисика": [
        "доставка", "доставили быстро", "курьер", "пришло быстро",
    ],
}


def _build_auto_mapping(
    pipeline_results: Dict[int, dict],
    markup_df: pd.DataFrame,
    threshold: float = 0.3,
) -> Dict[int, Dict[str, Optional[str]]]:
    """Автоматический маппинг predicted→true через косинусную близость.

    Алгоритм (per product):
      1. Для каждого predicted aspect: encode(keywords) → mean = pseudo-centroid
      2. Для каждого true aspect: encode(ASPECT_DESCRIPTIONS[name]) → mean
         Fallback: если описания нет — encode(name)
      3. Cosine similarity matrix S[i,j]
      4. Greedy: pred_i → argmax_j S[i,j] если S[i,j] ≥ τ, иначе None

    Args:
        pipeline_results: {nm_id: {"aspects": [...], "aspect_keywords": {...}, ...}}
        markup_df: DataFrame с колонкой true_labels_parsed
        threshold: порог косинусной близости для назначения

    Returns:
        {nm_id: {predicted_aspect: true_aspect_or_None, ...}}
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from configs.configs import config

    print("\n" + "=" * 70)
    print(f"AUTO MAPPING (threshold={threshold}, with aspect descriptions)")
    print("=" * 70)

    encoder = SentenceTransformer(config.models.encoder_path)

    # Pre-encode все описания из словаря (один раз)
    desc_cache: Dict[str, np.ndarray] = {}
    for aspect_name, phrases in ASPECT_DESCRIPTIONS.items():
        embs = encoder.encode(phrases, show_progress_bar=False)
        desc_cache[aspect_name] = np.mean(embs, axis=0)

    mapping: Dict[int, Dict[str, Optional[str]]] = {}

    for nm_id, pred_data in pipeline_results.items():
        pred_aspects = pred_data["aspects"]
        aspect_keywords = pred_data.get("aspect_keywords", {})

        # True aspects из разметки
        grp = markup_df[markup_df["nm_id"] == nm_id]
        true_aspects_set = set()
        for labels in grp["true_labels_parsed"].dropna():
            true_aspects_set.update(labels.keys())
        true_aspects = sorted(true_aspects_set)

        if not pred_aspects or not true_aspects:
            mapping[nm_id] = {pa: None for pa in pred_aspects}
            continue

        # Encode predicted: mean of keywords embeddings (pseudo-centroid)
        pred_embeddings = []
        for pa in pred_aspects:
            kws = aspect_keywords.get(pa, [pa])
            if not kws:
                kws = [pa]
            embs = encoder.encode(kws, show_progress_bar=False)
            pred_embeddings.append(np.mean(embs, axis=0))
        pred_matrix = np.stack(pred_embeddings)

        # Encode true: из desc_cache если есть, иначе fallback на encode(name)
        true_embeddings = []
        for ta in true_aspects:
            if ta in desc_cache:
                true_embeddings.append(desc_cache[ta])
            else:
                print(f"    [WARN] No description for '{ta}', using bare name")
                true_embeddings.append(encoder.encode(ta, show_progress_bar=False))
        true_matrix = np.stack(true_embeddings)

        # Cosine similarity matrix: |pred| × |true|
        sim_matrix = cosine_similarity(pred_matrix, true_matrix)

        # Greedy assignment с порогом
        product_mapping: Dict[str, Optional[str]] = {}
        print(f"\n  nm_id={nm_id}:")
        for i, pa in enumerate(pred_aspects):
            best_j = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_j])
            if best_sim >= threshold:
                product_mapping[pa] = true_aspects[best_j]
                print(f"    {pa:30s} → {true_aspects[best_j]:25s} (cos={best_sim:.3f})")
            else:
                product_mapping[pa] = None
                print(f"    {pa:30s} → None                      (max_cos={best_sim:.3f} < {threshold})")

        mapping[nm_id] = product_mapping

    print("=" * 70 + "\n")
    return mapping


# ── MANUAL MAPPING ─────────────────────────────────────────────────────────
# Пересобран для Anchor-First v2. Predicted aspects теперь = имена якорей.
# Маппинг: identity где pred=true, семантический где pred≈true, None для мусора.
# Мусор = якорь нерелевантный для данного товара ("Поедаемость" для книги).

MANUAL_MAPPING: Dict[int, Dict[str, Optional[str]]] = {

    # ── nm_id=117808756 (книга) ──────────────────────────────────────
    # pred (15): Цена, Качество, Внешний вид, Удобство, Логистика, Упаковка,
    #   Соответствие, Органолептика, Содержание, Состав, Свежесть, Комфорт,
    #   Продавец, Состояние, Вместимость
    # true (12): Внешний вид(12), Впечатление(1), Запах(7), Качество(55),
    #   Логисика(1), Логистика(10), Содержание(4), Соответствие(9),
    #   Текст(1), Удобство(1), Упаковка(25), Цена(7)
    117808756: {
        # Identity
        "Цена": "Цена",
        "Качество": "Качество",
        "Внешний вид": "Внешний вид",
        "Удобство": "Удобство",
        "Логистика": "Логистика",
        "Упаковка": "Упаковка",
        "Соответствие": "Соответствие",
        "Содержание": "Содержание",
        # Семантический
        "Органолептика": "Запах",       # kw: запаха, запах, посторонних запахов
        "Комфорт": "Текст",            # kw: шрифт хороший, шрифт приятный, мягкая обложка
        "Состояние": "Качество",        # kw: отвратительная подделка, хлам → many-to-one
        # Мусор (нерелевантный якорь для книги)
        "Состав": None,                 # kw: часть франшизы, все целое
        "Свежесть": None,               # kw: raincoast books
        "Продавец": None,               # kw: продавцу, книжных магазинах (нет в true)
        "Вместимость": None,            # kw: полям, kdp, преобрести
    },

    # ── nm_id=254445126 (толстовка) ──────────────────────────────────
    # pred (15): Цена, Качество, Внешний вид, Удобство, Логистика, Упаковка,
    #   Соответствие, Состав, Здоровье, Поедаемость, Свежесть, Комфорт,
    #   Продавец, Состояние, Вместимость
    # true (10): Ассортимент(2), Внешний вид(27), Запах(1), Качество(63),
    #   Комплектация(2), Комфорт(28), Общее впечатление(27),
    #   Соответствие(30), Уход(1), Цена(9)
    254445126: {
        # Identity
        "Цена": "Цена",
        "Качество": "Качество",
        "Внешний вид": "Внешний вид",
        "Соответствие": "Соответствие",
        "Комфорт": "Комфорт",
        # Семантический
        "Поедаемость": "Общее впечатление",  # kw: прекрасное худи, лучшая худи, отличное худи
        "Свежесть": "Уход",                  # kw: после стирки
        "Состояние": "Качество",              # kw: швы недоработки, нитки неровные → many-to-one
        # Мусор
        "Удобство": None,               # kw: при стирке, понятии рук — mixed мусор
        "Логистика": None,               # kw: худи сразу (1 kw)
        "Упаковка": None,                # kw: карточке товара, наклейки — мусор
        "Состав": None,                  # kw: материала особенно (1 kw)
        "Здоровье": None,                # kw: параметры ог-83, рост, обхват — мусор
        "Продавец": None,                # kw: этот бренд (нет Продавец в true)
        "Вместимость": None,             # kw: вся кофта, нитки — мусор
    },

    # ── nm_id=311233470 (платье) ─────────────────────────────────────
    # pred (18): Цена, Качество, Внешний вид, Удобство, Функциональность,
    #   Логистика, Упаковка, Соответствие, Органолептика, Содержание,
    #   Состав, Здоровье, Поедаемость, Свежесть, Комфорт, Продавец,
    #   Состояние, Вместимость
    # true (11): Внешний вид(67), Качество(71), Комфорт(10), Кофморт(1),
    #   Логистика(1), Продавец(10), Соответсвие(2), Соответствие(39),
    #   Состояние(13), Упаковка(10), Цена(43)
    311233470: {
        # Identity
        "Цена": "Цена",
        "Качество": "Качество",
        "Внешний вид": "Внешний вид",
        "Упаковка": "Упаковка",
        "Соответствие": "Соответствие",
        "Комфорт": "Комфорт",
        "Продавец": "Продавец",
        "Состояние": "Состояние",
        "Логистика": "Логистика",
        # Семантический
        "Удобство": "Комфорт",          # kw: драпировки хорошо, задумка хорошая → many-to-one
        "Состав": "Качество",            # kw: синтетики, чистая синтетика → many-to-one
        # Мусор
        "Функциональность": None,        # kw: пункте работник, сотрудник — мусор
        "Органолептика": None,           # kw: горло, горла — мусор
        "Содержание": None,              # kw: лживых отзывов, материал — мусор
        "Здоровье": None,                # kw: шею, ягодицах, спине — мусор (части тела)
        "Поедаемость": None,             # kw: навариться хотят — мусор
        "Свежесть": None,                # kw: новогодним столом — мусор
        "Вместимость": None,             # kw: сборок, нитки — мусор
    },

    # ── nm_id=441378025 (портсигар) ──────────────────────────────────
    # pred (16): Цена, Качество, Внешний вид, Удобство, Функциональность,
    #   Логистика, Упаковка, Соответствие, Органолептика, Содержание,
    #   Здоровье, Свежесть, Комфорт, Продавец, Состояние, Вместимость
    # true (12): Вместимость(6), Внешний вид(13), Запах(1), Качество(29),
    #   Комплектация(1), Логистика(3), Соответствие(4), Состояние(3),
    #   Удобство(33), Упаковка(1), Функционал(6), Цена(2)
    441378025: {
        # Identity
        "Цена": "Цена",
        "Качество": "Качество",
        "Внешний вид": "Внешний вид",
        "Удобство": "Удобство",
        "Логистика": "Логистика",
        "Упаковка": "Упаковка",
        "Соответствие": "Соответствие",
        "Состояние": "Состояние",
        "Вместимость": "Вместимость",
        # Семантический
        "Функциональность": "Функционал",  # pred=Функциональность, true=Функционал
        "Органолептика": "Вместимость",    # kw: сигарет, сигареты → many-to-one
        # Мусор
        "Содержание": None,              # kw: слова, описании
        "Здоровье": None,                # kw: условиях сибири
        "Свежесть": None,                # kw: нужна доработка, бесполезное
        "Комфорт": None,                 # kw: скользкий материал — скорее Качество, но спорно
        "Продавец": None,                # kw: клавиша прикуривателя — мусор
    },

    # ── nm_id=506358703 (кошачий корм) ───────────────────────────────
    # pred (17): Цена, Качество, Внешний вид, Удобство, Функциональность,
    #   Логистика, Упаковка, Соответствие, Органолептика, Состав,
    #   Здоровье, Поедаемость, Свежесть, Комфорт, Продавец,
    #   Состояние, Вместимость
    # true (11): Ассортимент(3), Запах(1), Здоровье(23), Качество(33),
    #   Логистика(18), Поедаемость(82), Свеежсть(1), Свежесть(9),
    #   Состав(12), Упаковка(25), Цена(12)
    506358703: {
        # Identity
        "Цена": "Цена",
        "Качество": "Качество",
        "Логистика": "Логистика",
        "Упаковка": "Упаковка",
        "Состав": "Состав",
        "Здоровье": "Здоровье",
        "Поедаемость": "Поедаемость",
        "Свежесть": "Свежесть",
        # Семантический
        "Органолептика": "Поедаемость",  # kw: лакомство, вкус, вкусы → many-to-one
        "Удобство": "Упаковка",          # kw: удобный замок → many-to-one
        # Мусор
        "Внешний вид": None,             # kw: бренды, цветом — мусор для корма
        "Функциональность": None,        # kw: эффект, действии
        "Соответствие": None,            # kw: количеству порции — слишком размытый
        "Комфорт": None,                 # kw: гранул, застёжка-липучка — mixed мусор
        "Продавец": None,                # kw: зоо магазинах
        "Состояние": None,               # kw: хвостам, олениной — мусор
        "Вместимость": None,             # kw: ели все — мусор
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
    parser.add_argument(
        "--csv-path",
        type=str,
        default="parser/reviews_batches/merged_checked_reviews.csv",
        help="Разметка и по умолчанию источник отзывов для пайплайна",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Если задан и файл существует — отзывы из JSON; иначе из --csv-path",
    )
    parser.add_argument("--write-prefix", type=str, default="")
    parser.add_argument(
        "--mapping",
        type=str,
        default="manual",
        choices=["manual", "auto"],
        help="Тип маппинга: manual (ручная таблица) или auto (косинусная близость)",
    )
    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=0.3,
        help="Порог косинусной близости для auto mapping (default: 0.3)",
    )
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
    if "json_path" in cfg:
        j = cfg["json_path"]
        JSON_PATH = None if j in (None, "") else str(j)
    else:
        JSON_PATH = args.json_path
    write_prefix = str(cfg.get("write_prefix", args.write_prefix or "")).strip()
    if write_prefix and not write_prefix.endswith("_"):
        write_prefix = f"{write_prefix}_"

    print(f"[Eval] seed={seed}")
    if args.config:
        print(f"[Eval] config={args.config}")
    print(f"[Eval] csv_path={CSV_PATH}")
    print(f"[Eval] json_path={JSON_PATH or '(none — отзывы из CSV)'}")

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
                    str(k): {
                        "aspects": v["aspects"],
                        "aspect_keywords": v.get("aspect_keywords", {}),
                        "diagnostics": v.get("diagnostics", {}),
                    }
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
        mapping_mode = args.mapping
        auto_threshold = args.auto_threshold

        print(f"\n{'='*70}")
        print(f"ШАГ 4: МЕТРИКИ (mapping={mapping_mode})")
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
                "aspect_keywords": info.get("aspect_keywords", {}),
                "per_review": per_review_loaded.get(nm_id_str, {}),
                "diagnostics": info.get("diagnostics", {}),
            }

        # Выбор маппинга
        if mapping_mode == "auto":
            active_mapping = _build_auto_mapping(
                pipeline_results_for_eval, df, threshold=auto_threshold,
            )
        else:
            active_mapping = MANUAL_MAPPING

        metrics = evaluate_with_mapping(df, pipeline_results_for_eval, active_mapping)

        metrics["assignment_by_product"] = {
            str(k): (v.get("diagnostics") or {}).get("anchor_assignment_counts", {})
            for k, v in pipeline_results_for_eval.items()
        }
        metrics["residual_medoid_by_product"] = {
            str(k): (v.get("diagnostics") or {}).get("residual_medoid_names", [])
            for k, v in pipeline_results_for_eval.items()
        }
        metrics["nli_medoid_diagnostics_by_product"] = {
            str(k): (v.get("diagnostics") or {}).get("nli_medoid_diagnostics", [])
            for k, v in pipeline_results_for_eval.items()
        }

        print("\n--- Counter: confident / residual (per product) ---")
        for nm_id in sorted(pipeline_results_for_eval.keys(), key=lambda x: int(x) if isinstance(x, int) else x):
            ac = metrics["assignment_by_product"].get(str(nm_id), {})
            print(f"  nm_id={nm_id}: Counter({dict(ac)})")

        print("\n--- Residual medoid names (final, per product) ---")
        for nm_id in sorted(pipeline_results_for_eval.keys(), key=lambda x: int(x) if isinstance(x, int) else x):
            names = metrics["residual_medoid_by_product"].get(str(nm_id), [])
            print(f"  nm_id={nm_id}: {names}")

        print("\n--- NLI medoid routing (per product) ---")
        for nm_id in sorted(pipeline_results_for_eval.keys(), key=lambda x: int(x) if isinstance(x, int) else x):
            lines = metrics["nli_medoid_diagnostics_by_product"].get(str(nm_id), [])
            print(f"  nm_id={nm_id}:")
            if not lines:
                print("    (нет medoid → nli маршрутизации)")
            for line in lines:
                print(f"    {line}")

        if mapping_mode == "manual":
            print("\n--- Manual mapping: precision, recall, mention recall, MAE (per product) ---")
            print(
                f"  {'nm_id':>8}  {'P':>6}  {'R':>6}  {'mR_pr':>6}  "
                f"{'mR_rev':>7}  {'MAE_r':>7}  {'MAE_cal':>8}  {'n':>4}"
            )
            for nm_id in sorted(metrics["per_product"].keys()):
                pm = metrics["per_product"][nm_id]
                mr = pm["mae_raw"]
                mc = pm["mae_calibrated"]
                mr_s = f"{mr:.3f}" if mr is not None else "   N/A"
                mc_s = f"{mc:.3f}" if mc is not None else "    N/A"
                print(
                    f"  {nm_id:8d}  {pm['precision']:6.3f}  {pm['recall']:6.3f}  "
                    f"{pm['mention_recall_product']:6.3f}  {pm['mention_recall_review']:7.3f}  "
                    f"{mr_s:>7}  {mc_s:>8}  {pm['mae_n']:4d}"
                )

        # Сохраняем использованный маппинг в метрики для воспроизводимости
        metrics["mapping_mode"] = mapping_mode
        if mapping_mode == "auto":
            metrics["auto_threshold"] = auto_threshold
            metrics["auto_mapping"] = {
                str(k): v for k, v in active_mapping.items()
            }

        for nm_id, pm in metrics["per_product"].items():
            print(f"\nnm_id={nm_id}:")
            print(f"  Precision:              {pm['precision']}")
            print(f"  Recall (aspect):        {pm['recall']}")
            print(f"  Recall (mention/prod):  {pm['mention_recall_product']}")
            print(f"  Recall (mention/review):{pm['mention_recall_review']}")
            print(f"  MAE raw:                {pm['mae_raw']}  →  calibrated: {pm['mae_calibrated']}  "
                  f"(n={pm['mae_n']})")
            print(f"  Predicted:              {pm['pred_aspects']}")
            print(f"  True:                   {pm['true_aspects']}")

        cal = metrics["calibration_global"]
        print(f"\n{'='*70}")
        print("ИТОГО:")
        print(f"  Macro Precision:               {metrics['macro_precision']}")
        print(f"  Macro Recall (aspect):         {metrics['macro_recall']}")
        print(f"  Macro Recall (mention/product): {metrics['macro_mention_recall_product']}")
        print(f"  Macro Recall (mention/review):  {metrics['macro_mention_recall_review']}")
        print(f"  Micro Precision:               {metrics['micro_precision']}")
        print(f"  Micro Recall (aspect):         {metrics['micro_recall']}")
        print(f"  Global Mention Recall (review): {metrics['global_mention_recall_review']}")
        print(f"  Global MAE raw:                {metrics['global_mae_raw']}  (n={metrics['global_mae_n']})")
        print(f"  Global MAE calibr:             {metrics['global_mae_calibrated']}")
        print(f"  Global calibration:            S_cal = {cal['a']}*S_raw + {cal['b']}")

        suffix = f"_{mapping_mode}" if mapping_mode != "manual" else ""
        metrics_path = f"{write_prefix}eval_metrics{suffix}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nМетрики сохранены в {metrics_path}")

        # Product-level aspect rating comparison
        product_ratings = evaluate_product_ratings(
            df, pipeline_results_for_eval, active_mapping,
        )
        metrics["product_ratings"] = product_ratings

        from configs.configs import config as _cfg_ml
        total_nli_pairs = sum(
            int((v.get("diagnostics") or {}).get("nli_pairs_count") or 0)
            for v in pipeline_results_for_eval.values()
        )
        metrics["run_summary"] = {
            "multi_label_threshold": float(_cfg_ml.discovery.multi_label_threshold),
            "multi_label_max_aspects": int(_cfg_ml.discovery.multi_label_max_aspects),
            "nli_pairs_total": total_nli_pairs,
            "mention_recall_review": metrics["global_mention_recall_review"],
            "sentence_mae_raw": metrics["global_mae_raw"],
            "product_mae_n_ge_3": product_ratings.get("global_product_mae_filtered"),
        }
        print(f"\n{'='*70}")
        print("СВОДКА ПРОГОНА (multi-label, discovery)")
        print(f"  multi_label_threshold = {_cfg_ml.discovery.multi_label_threshold}")
        print(f"  multi_label_max_aspects = {_cfg_ml.discovery.multi_label_max_aspects}")
        print(f"  NLI пар (всего):              {total_nli_pairs}")
        print(f"  Mention recall (review):     {metrics['global_mention_recall_review']}")
        print(f"  Sentence MAE (global raw):   {metrics['global_mae_raw']}")
        print(f"  Product MAE (n_true≥3):      {product_ratings.get('global_product_mae_filtered')}")
        print(f"{'='*70}\n")

        # Перезаписываем с product_ratings
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)