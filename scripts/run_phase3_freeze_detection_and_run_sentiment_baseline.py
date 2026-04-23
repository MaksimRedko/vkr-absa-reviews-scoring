from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import config
from scripts import run_phase2_baseline_matching as lexical
from src.schemas.models import AggregationInput, SentimentPair
from src.stages.aggregation import RatingMathEngine
from src.stages.fraud import AntiFraudEngine
from src.stages.sentiment import SentimentEngine
from src.vocabulary.loader import AspectDefinition

REFERENCE_DETECTION = {
    "macro_precision": 0.4806,
    "macro_recall": 0.4130,
    "macro_f1": 0.4251,
}


@dataclass(slots=True)
class ReviewRecord:
    review_id: str
    nm_id: int
    category: str
    source: str
    text: str
    rating: int
    true_labels: dict[str, float]
    true_labels_lemma: dict[str, float]
    pred_aspect_ids: set[str]
    true_aspect_ids: set[str]


@dataclass(slots=True)
class DetectionSummary:
    macro_precision: float
    macro_recall: float
    macro_f1: float
    n_reviews: int
    reproduction_ok: bool


def _parse_true_labels(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in parsed.items():
        key_s = str(key).strip()
        if not key_s:
            continue
        try:
            out[key_s] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _load_reviews(dataset_csv: Path) -> list[ReviewRecord]:
    df = pd.read_csv(dataset_csv, dtype={"id": str})
    required = {"nm_id", "id", "full_text", "true_labels", "category", "source", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    rows: list[ReviewRecord] = []
    for _, row in df.iterrows():
        text = str(row["full_text"]).strip()
        if not text:
            continue
        true_labels = _parse_true_labels(row["true_labels"])
        true_labels_lemma = {lexical._normalize(k): float(v) for k, v in true_labels.items() if lexical._normalize(k)}
        rows.append(
            ReviewRecord(
                review_id=str(row["id"]),
                nm_id=int(row["nm_id"]),
                category=str(row["category"]).strip(),
                source=str(row["source"]).strip(),
                text=text,
                rating=int(row["rating"]),
                true_labels=true_labels,
                true_labels_lemma=true_labels_lemma,
                pred_aspect_ids=set(),
                true_aspect_ids=set(),
            )
        )
    return rows


def _build_hybrid_cache(
    core_vocab_path: Path,
    domain_vocab_by_category: dict[str, Path],
    categories: set[str],
) -> tuple[
    dict[str, list[AspectDefinition]],
    dict[str, dict[str, set[str]]],
    dict[str, dict[str, str]],
]:
    aspects_by_category: dict[str, list[AspectDefinition]] = {}
    term_to_aspects_by_category: dict[str, dict[str, set[str]]] = {}
    aspect_name_by_category: dict[str, dict[str, str]] = {}
    for category in sorted(categories):
        domain_path = domain_vocab_by_category.get(category)
        paths = [core_vocab_path] + ([domain_path] if domain_path else [])
        aspects = lexical._build_vocabulary(paths)
        term_to_aspects, _ = lexical._term_indexes(aspects)
        aspects_by_category[category] = aspects
        term_to_aspects_by_category[category] = term_to_aspects
        aspect_name_by_category[category] = {aspect.id: aspect.canonical_name for aspect in aspects}
    return aspects_by_category, term_to_aspects_by_category, aspect_name_by_category


def _run_frozen_detection(
    reviews: list[ReviewRecord],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
) -> tuple[DetectionSummary, pd.DataFrame]:
    extractor = lexical.CandidateExtractor(ngram_range=(1, 2), min_word_length=3)
    extractor.dependency_filter_enabled = False

    rows: list[dict[str, Any]] = []
    for review in reviews:
        term_to_aspects = term_to_aspects_by_category[review.category]
        candidate_lemmas = lexical._extract_candidate_lemmas_by_unit(review.text, extractor, "candidates")
        matched_terms = lexical._match_terms(candidate_lemmas, term_to_aspects, "lexical_only")

        pred_aspect_ids: set[str] = set()
        for term in matched_terms:
            pred_aspect_ids.update(term_to_aspects[term])
        review.pred_aspect_ids = pred_aspect_ids

        true_aspect_ids: set[str] = set()
        for gold_lemma in review.true_labels_lemma.keys():
            true_aspect_ids.update(term_to_aspects.get(gold_lemma, set()))
        review.true_aspect_ids = true_aspect_ids

        p, r, f1 = lexical._eval_prf(pred_aspect_ids, true_aspect_ids)
        rows.append(
            {
                "review_id": review.review_id,
                "product_id": review.nm_id,
                "category": review.category,
                "source": review.source,
                "predicted_aspect_ids_json": json.dumps(sorted(pred_aspect_ids), ensure_ascii=False),
                "true_aspect_ids_json": json.dumps(sorted(true_aspect_ids), ensure_ascii=False),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }
        )

    df = pd.DataFrame(rows)
    macro_precision = float(df["precision"].mean()) if not df.empty else 0.0
    macro_recall = float(df["recall"].mean()) if not df.empty else 0.0
    macro_f1 = float(df["f1"].mean()) if not df.empty else 0.0
    reproduction_ok = (
        round(macro_precision, 4) == REFERENCE_DETECTION["macro_precision"]
        and round(macro_recall, 4) == REFERENCE_DETECTION["macro_recall"]
        and round(macro_f1, 4) == REFERENCE_DETECTION["macro_f1"]
    )
    return (
        DetectionSummary(
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            n_reviews=len(df),
            reproduction_ok=reproduction_ok,
        ),
        df,
    )


def _build_sentiment_pairs(
    reviews: list[ReviewRecord],
    aspect_name_by_category: dict[str, dict[str, str]],
) -> list[SentimentPair]:
    pairs: list[SentimentPair] = []
    for review in reviews:
        aspect_name_map = aspect_name_by_category[review.category]
        for aspect_id in sorted(review.pred_aspect_ids):
            canonical_name = aspect_name_map.get(aspect_id, aspect_id)
            pairs.append(
                SentimentPair(
                    review_id=review.review_id,
                    sentence=review.text,
                    aspect=aspect_id,
                    nli_label=canonical_name,
                    weight=1.0,
                )
            )
    return pairs


def _run_review_level_sentiment(
    pairs: list[SentimentPair],
) -> tuple[list[Any], dict[str, dict[str, float]], int, dict[str, Any]]:
    if not pairs:
        return [], {}, 0, {
            "sentiment_pair_count": 0,
            "nli_hypothesis_calls": 0,
            "relevance_threshold": float(getattr(config.sentiment, "relevance_threshold", 0.0)),
            "filtered_out_pairs": 0,
        }

    engine = SentimentEngine()
    sentiment_results = engine.batch_analyze(pairs)

    relevance_threshold = float(getattr(config.sentiment, "relevance_threshold", 0.0))
    filtered_out = 0
    if relevance_threshold > 0:
        unfiltered = sentiment_results
        sentiment_results = [
            result
            for result in unfiltered
            if (float(result.p_ent_pos) + float(result.p_ent_neg)) >= relevance_threshold
        ]
        filtered_out = len(unfiltered) - len(sentiment_results)
        if unfiltered and not sentiment_results:
            sentiment_results = unfiltered
            filtered_out = 0

    per_review: dict[str, dict[str, float]] = defaultdict(dict)
    for result in sentiment_results:
        per_review[str(result.review_id)][str(result.aspect)] = float(result.score)

    meta = {
        "sentiment_pair_count": len(pairs),
        "nli_hypothesis_calls": int(2 * len(pairs)),
        "relevance_threshold": relevance_threshold,
        "filtered_out_pairs": int(filtered_out),
    }
    return sentiment_results, per_review, engine.batch_size, meta


def _build_review_level_rows(
    reviews: list[ReviewRecord],
    per_review_scores: dict[str, dict[str, float]],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
    aspect_name_by_category: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_gold_pairs = 0
    total_mappable_gold_pairs = 0
    total_matched_gold_pairs = 0

    review_maes: list[float] = []
    review_ids_with_matches: set[str] = set()

    for review in reviews:
        review_errors: list[float] = []
        pred_scores = per_review_scores.get(review.review_id, {})
        term_to_aspects = term_to_aspects_by_category[review.category]
        aspect_name_map = aspect_name_by_category[review.category]

        for gold_label, gold_score in review.true_labels.items():
            total_gold_pairs += 1
            gold_lemma = lexical._normalize(gold_label)
            mapped_ids = sorted(term_to_aspects.get(gold_lemma, set()))
            predicted_scores = [float(pred_scores[aspect_id]) for aspect_id in mapped_ids if aspect_id in pred_scores]
            matched = bool(predicted_scores)
            if mapped_ids:
                total_mappable_gold_pairs += 1
            if matched:
                total_matched_gold_pairs += 1
                pred_score = float(np.mean(predicted_scores))
                abs_error = abs(pred_score - float(gold_score))
                review_errors.append(abs_error)
                review_ids_with_matches.add(review.review_id)
            else:
                pred_score = None
                abs_error = None

            inversion = bool(
                pred_score is not None
                and (
                    (float(gold_score) >= 4.0 and pred_score <= 2.0)
                    or (float(gold_score) <= 2.0 and pred_score >= 4.0)
                )
            )

            rows.append(
                {
                    "review_id": review.review_id,
                    "product_id": review.nm_id,
                    "category": review.category,
                    "source": review.source,
                    "gold_aspect": gold_label,
                    "gold_score": float(gold_score),
                    "gold_aspect_mappable": bool(mapped_ids),
                    "mapped_aspect_ids_json": json.dumps(mapped_ids, ensure_ascii=False),
                    "mapped_aspect_names_json": json.dumps(
                        [aspect_name_map.get(aspect_id, aspect_id) for aspect_id in mapped_ids],
                        ensure_ascii=False,
                    ),
                    "predicted_score": round(pred_score, 4) if pred_score is not None else None,
                    "matched": matched,
                    "abs_error": round(abs_error, 4) if abs_error is not None else None,
                    "inversion": inversion,
                    "review_text_preview": review.text[:240],
                }
            )

        if review_errors:
            review_maes.append(float(np.mean(review_errors)))

    df = pd.DataFrame(rows)
    metrics = {
        "mae_review_level_macro": round(float(np.mean(review_maes)) if review_maes else 0.0, 4),
        "n_matched_pairs": int(total_matched_gold_pairs),
        "n_gold_pairs_total": int(total_gold_pairs),
        "n_gold_pairs_mappable": int(total_mappable_gold_pairs),
        "coverage_all_gold_pairs": round(
            float(total_matched_gold_pairs / total_gold_pairs) if total_gold_pairs else 0.0,
            4,
        ),
        "coverage_mappable_gold_pairs": round(
            float(total_matched_gold_pairs / total_mappable_gold_pairs) if total_mappable_gold_pairs else 0.0,
            4,
        ),
        "n_reviews_with_matched_pairs": int(len(review_ids_with_matches)),
    }
    return df, metrics


def _aggregate_products(
    reviews: list[ReviewRecord],
    per_review_scores: dict[str, dict[str, float]],
) -> tuple[dict[int, Any], dict[str, Any]]:
    by_product: dict[int, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        by_product[review.nm_id].append(review)

    fraud_engine = AntiFraudEngine()
    math_engine = RatingMathEngine()

    agg_by_product: dict[int, Any] = {}
    fraud_weights_by_review: dict[str, float] = {}
    fraud_sec = 0.0
    aggregation_sec = 0.0

    for product_id, product_reviews in sorted(by_product.items()):
        texts = [review.text for review in product_reviews]
        t_fraud = time.perf_counter()
        trust_weights = fraud_engine.calculate_trust_weights(texts)
        fraud_sec += time.perf_counter() - t_fraud
        inputs: list[AggregationInput] = []
        for review, trust_weight in zip(product_reviews, trust_weights):
            fraud_weights_by_review[review.review_id] = float(trust_weight)
            aspects = per_review_scores.get(review.review_id, {})
            if not aspects:
                continue
            inputs.append(
                AggregationInput(
                    review_id=review.review_id,
                    aspects=aspects,
                    fraud_weight=float(trust_weight),
                    date=None,
                )
            )
        t_agg = time.perf_counter()
        agg_by_product[product_id] = math_engine.aggregate(inputs)
        aggregation_sec += time.perf_counter() - t_agg

    return agg_by_product, {
        "fraud_weights_by_review": fraud_weights_by_review,
        "fraud_sec": fraud_sec,
        "aggregation_sec": aggregation_sec,
    }


def _build_product_level_rows(
    reviews: list[ReviewRecord],
    per_review_scores: dict[str, dict[str, float]],
    agg_by_product: dict[int, Any],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
    aspect_name_by_category: dict[str, dict[str, str]],
) -> pd.DataFrame:
    reviews_by_product: dict[int, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        reviews_by_product[review.nm_id].append(review)

    columns = [
        "product_id",
        "category",
        "source",
        "gold_aspect",
        "mapped_aspect_ids_json",
        "mapped_aspect_names_json",
        "predicted_score",
        "true_avg_all",
        "true_avg_matched",
        "abs_error_all",
        "abs_error_matched",
        "n_true_reviews",
        "n_matched_reviews",
        "aggregation_predicted_aspect_ids_json",
    ]
    rows: list[dict[str, Any]] = []
    for product_id, product_reviews in sorted(reviews_by_product.items()):
        category = product_reviews[0].category if product_reviews else ""
        source = product_reviews[0].source if product_reviews else ""
        term_to_aspects = term_to_aspects_by_category[category]
        aspect_name_map = aspect_name_by_category[category]
        aggregation_result = agg_by_product.get(product_id)
        predicted_product_scores = {
            str(name): float(score.score)
            for name, score in (aggregation_result.aspects.items() if aggregation_result is not None else [])
        }

        gold_scores_by_label: dict[str, list[float]] = defaultdict(list)
        matched_review_scores_by_label: dict[str, list[float]] = defaultdict(list)
        matched_true_scores_by_label: dict[str, list[float]] = defaultdict(list)

        for review in product_reviews:
            pred_scores = per_review_scores.get(review.review_id, {})
            for gold_label, gold_score in review.true_labels.items():
                gold_scores_by_label[gold_label].append(float(gold_score))
                mapped_ids = sorted(term_to_aspects.get(lexical._normalize(gold_label), set()))
                review_pred_scores = [float(pred_scores[aspect_id]) for aspect_id in mapped_ids if aspect_id in pred_scores]
                if review_pred_scores:
                    matched_review_scores_by_label[gold_label].append(float(np.mean(review_pred_scores)))
                    matched_true_scores_by_label[gold_label].append(float(gold_score))

        for gold_label, true_scores in sorted(gold_scores_by_label.items()):
            mapped_ids = sorted(term_to_aspects.get(lexical._normalize(gold_label), set()))
            pred_scores = [predicted_product_scores[aspect_id] for aspect_id in mapped_ids if aspect_id in predicted_product_scores]
            if not pred_scores:
                continue

            pred_score = float(np.mean(pred_scores))
            true_avg_all = float(np.mean(true_scores))
            true_avg_matched = (
                float(np.mean(matched_true_scores_by_label[gold_label]))
                if matched_true_scores_by_label.get(gold_label)
                else None
            )
            rows.append(
                {
                    "product_id": int(product_id),
                    "category": category,
                    "source": source,
                    "gold_aspect": gold_label,
                    "mapped_aspect_ids_json": json.dumps(mapped_ids, ensure_ascii=False),
                    "mapped_aspect_names_json": json.dumps(
                        [aspect_name_map.get(aspect_id, aspect_id) for aspect_id in mapped_ids],
                        ensure_ascii=False,
                    ),
                    "predicted_score": round(pred_score, 4),
                    "true_avg_all": round(true_avg_all, 4),
                    "true_avg_matched": round(true_avg_matched, 4) if true_avg_matched is not None else None,
                    "abs_error_all": round(abs(pred_score - true_avg_all), 4),
                    "abs_error_matched": (
                        round(abs(pred_score - true_avg_matched), 4)
                        if true_avg_matched is not None
                        else None
                    ),
                    "n_true_reviews": int(len(true_scores)),
                    "n_matched_reviews": int(len(matched_review_scores_by_label.get(gold_label, []))),
                    "aggregation_predicted_aspect_ids_json": json.dumps(
                        sorted(predicted_product_scores.keys()),
                        ensure_ascii=False,
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _per_group_mae(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for key, group in sorted(df.groupby(group_col), key=lambda item: str(item[0])):
        out.append(
            {
                group_col: str(key),
                "n_rows": int(len(group)),
                "mae_all": round(float(group["abs_error_all"].mean()), 4),
                "mae_n_ge_3": round(float(group[group["n_true_reviews"] >= 3]["abs_error_all"].mean()), 4)
                if not group[group["n_true_reviews"] >= 3].empty
                else None,
            }
        )
    return out


def _top_inversions(review_df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if review_df.empty:
        return review_df.head(0)
    subset = review_df[review_df["inversion"] == True].copy()
    if subset.empty:
        return subset
    return subset.sort_values(["abs_error", "review_id"], ascending=[False, True]).head(limit)


def _top_sentiment_errors(review_df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if review_df.empty:
        return review_df.head(0)
    subset = review_df[review_df["matched"] == True].copy()
    if subset.empty:
        return subset
    return subset.sort_values(["abs_error", "review_id"], ascending=[False, True]).head(limit)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_none_"
    view = df[columns].fillna("").astype(str)
    widths = {col: max(len(col), int(view[col].map(len).max())) for col in columns}
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in columns) + " |"
    body = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep] + body)


def _write_summary(
    out_dir: Path,
    detection: DetectionSummary,
    review_metrics: dict[str, Any],
    product_metrics: dict[str, Any] | None,
    worst_product_rows: pd.DataFrame,
    inversion_rows: pd.DataFrame,
    sentiment_error_rows: pd.DataFrame,
) -> None:
    lines = [
        "# phase3_freeze_detection_and_run_sentiment_baseline",
        "",
        "## A. Frozen detection",
        f"- precision: {detection.macro_precision:.4f}",
        f"- recall: {detection.macro_recall:.4f}",
        f"- F1: {detection.macro_f1:.4f}",
        f"- reproduction_ok: {str(detection.reproduction_ok).lower()}",
        "",
    ]

    if not detection.reproduction_ok:
        lines.extend(
            [
                "## Stop",
                "- Frozen detection was not reproduced. Downstream run was not executed.",
                "",
            ]
        )
        (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        return

    assert product_metrics is not None
    lines.extend(
        [
            "## B. Review-level sentiment",
            f"- MAE: {review_metrics['mae_review_level_macro']:.4f}",
            f"- n matched pairs: {review_metrics['n_matched_pairs']}",
            f"- coverage of gold review-aspect pairs: {review_metrics['coverage_all_gold_pairs']:.4f}",
            f"- coverage of mappable gold review-aspect pairs: {review_metrics['coverage_mappable_gold_pairs']:.4f}",
            "",
            "## C. Product-level aggregation",
            f"- product MAE (all): {product_metrics['product_mae_all']:.4f}" if product_metrics["product_mae_all"] is not None else "- product MAE (all): null",
            f"- product MAE (n>=3): {product_metrics['product_mae_n_ge_3']:.4f}" if product_metrics["product_mae_n_ge_3"] is not None else "- product MAE (n>=3): null",
            "",
            "### per-category",
            _markdown_table(pd.DataFrame(product_metrics["per_category"]), ["category", "n_rows", "mae_all", "mae_n_ge_3"])
            if product_metrics["per_category"]
            else "_none_",
            "",
            "### per-source",
            _markdown_table(pd.DataFrame(product_metrics["per_source"]), ["source", "n_rows", "mae_all", "mae_n_ge_3"])
            if product_metrics["per_source"]
            else "_none_",
            "",
            "## D. Short diagnosis",
            "### 10 worst aspect cases by MAE",
            _markdown_table(
                worst_product_rows,
                ["product_id", "category", "source", "gold_aspect", "predicted_score", "true_avg_all", "abs_error_all", "n_true_reviews"],
            )
            if not worst_product_rows.empty
            else "_none_",
            "",
            "### 5 obvious sentiment inversions",
            _markdown_table(
                inversion_rows,
                ["review_id", "product_id", "category", "gold_aspect", "gold_score", "predicted_score", "abs_error", "review_text_preview"],
            )
            if not inversion_rows.empty
            else "_none_",
            "",
            "### 5 cases where detection is ok and sentiment failed",
            _markdown_table(
                sentiment_error_rows,
                ["review_id", "product_id", "category", "gold_aspect", "gold_score", "predicted_score", "abs_error", "review_text_preview"],
            )
            if not sentiment_error_rows.empty
            else "_none_",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 frozen detection + sentiment baseline")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase3_freeze_detection_and_run_sentiment_baseline")
    args = parser.parse_args()

    total_started = time.perf_counter()
    out_dir = ROOT / args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews = _load_reviews(ROOT / args.dataset_csv)
    domain_vocab_by_category = {
        "physical_goods": ROOT / "src/vocabulary/domain/physical_goods.yaml",
        "consumables": ROOT / "src/vocabulary/domain/consumables.yaml",
        "hospitality": ROOT / "src/vocabulary/domain/hospitality.yaml",
        "services": ROOT / "src/vocabulary/domain/services.yaml",
    }
    _, term_to_aspects_by_category, aspect_name_by_category = _build_hybrid_cache(
        core_vocab_path=ROOT / args.core_vocab,
        domain_vocab_by_category=domain_vocab_by_category,
        categories={review.category for review in reviews},
    )

    t_detection = time.perf_counter()
    detection_summary, detection_df = _run_frozen_detection(reviews, term_to_aspects_by_category)
    detection_latency = time.perf_counter() - t_detection
    detection_payload = {
        "macro_precision": round(detection_summary.macro_precision, 4),
        "macro_recall": round(detection_summary.macro_recall, 4),
        "macro_f1": round(detection_summary.macro_f1, 4),
        "reference_macro_precision": REFERENCE_DETECTION["macro_precision"],
        "reference_macro_recall": REFERENCE_DETECTION["macro_recall"],
        "reference_macro_f1": REFERENCE_DETECTION["macro_f1"],
        "reproduction_ok": detection_summary.reproduction_ok,
        "n_reviews": detection_summary.n_reviews,
    }
    _write_json(out_dir / "frozen_detection_check.json", detection_payload)
    detection_df.to_csv(out_dir / "frozen_detection_review_predictions.csv", index=False, encoding="utf-8")

    if not detection_summary.reproduction_ok:
        latency_payload = {
            "total_latency_sec": round(time.perf_counter() - total_started, 4),
            "detection_sec": round(detection_latency, 4),
            "sentiment_sec": 0.0,
            "fraud_sec": 0.0,
            "aggregation_sec": 0.0,
            "nli_hypothesis_calls": 0,
            "sentiment_pair_count": 0,
        }
        _write_json(out_dir / "latency_breakdown.json", latency_payload)
        _write_summary(
            out_dir=out_dir,
            detection=detection_summary,
            review_metrics={},
            product_metrics=None,
            worst_product_rows=pd.DataFrame(),
            inversion_rows=pd.DataFrame(),
            sentiment_error_rows=pd.DataFrame(),
        )
        print(json.dumps({"status": "STOP", "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
        return

    pairs = _build_sentiment_pairs(reviews, aspect_name_by_category)

    t_sentiment = time.perf_counter()
    sentiment_results, per_review_scores, _batch_size, sentiment_meta = _run_review_level_sentiment(pairs)
    sentiment_latency = time.perf_counter() - t_sentiment

    agg_by_product, fraud_meta = _aggregate_products(reviews, per_review_scores)

    review_level_df, review_metrics = _build_review_level_rows(
        reviews=reviews,
        per_review_scores=per_review_scores,
        term_to_aspects_by_category=term_to_aspects_by_category,
        aspect_name_by_category=aspect_name_by_category,
    )
    review_level_df.to_csv(out_dir / "review_level_sentiment_predictions.csv", index=False, encoding="utf-8")

    product_level_df = _build_product_level_rows(
        reviews=reviews,
        per_review_scores=per_review_scores,
        agg_by_product=agg_by_product,
        term_to_aspects_by_category=term_to_aspects_by_category,
        aspect_name_by_category=aspect_name_by_category,
    )
    product_level_df.to_csv(out_dir / "product_level_scores.csv", index=False, encoding="utf-8")

    product_metrics = {
        "product_mae_all": round(float(product_level_df["abs_error_all"].mean()), 4) if not product_level_df.empty else None,
        "product_mae_n_ge_3": (
            round(float(product_level_df[product_level_df["n_true_reviews"] >= 3]["abs_error_all"].mean()), 4)
            if not product_level_df[product_level_df["n_true_reviews"] >= 3].empty
            else None
        ),
        "n_product_aspect_rows": int(len(product_level_df)),
        "per_category": _per_group_mae(product_level_df, "category"),
        "per_source": _per_group_mae(product_level_df, "source"),
    }

    metrics_review_payload = {
        **review_metrics,
        "sentiment_pair_count": int(sentiment_meta["sentiment_pair_count"]),
        "nli_hypothesis_calls": int(sentiment_meta["nli_hypothesis_calls"]),
        "relevance_threshold": float(sentiment_meta["relevance_threshold"]),
        "filtered_out_pairs": int(sentiment_meta["filtered_out_pairs"]),
    }
    _write_json(out_dir / "metrics_review_level.json", metrics_review_payload)
    _write_json(out_dir / "metrics_product_level.json", product_metrics)

    latency_payload = {
        "total_latency_sec": round(time.perf_counter() - total_started, 4),
        "detection_sec": round(detection_latency, 4),
        "sentiment_sec": round(sentiment_latency, 4),
        "fraud_sec": round(float(fraud_meta["fraud_sec"]), 4),
        "aggregation_sec": round(float(fraud_meta["aggregation_sec"]), 4),
        "sentiment_pair_count": int(sentiment_meta["sentiment_pair_count"]),
        "nli_hypothesis_calls": int(sentiment_meta["nli_hypothesis_calls"]),
        "filtered_out_pairs": int(sentiment_meta["filtered_out_pairs"]),
    }
    _write_json(out_dir / "latency_breakdown.json", latency_payload)

    worst_product_rows = (
        product_level_df.sort_values(["abs_error_all", "product_id"], ascending=[False, True]).head(10)
        if not product_level_df.empty
        else product_level_df
    )
    inversion_rows = _top_inversions(review_level_df, limit=5)
    sentiment_error_rows = _top_sentiment_errors(review_level_df, limit=5)
    _write_summary(
        out_dir=out_dir,
        detection=detection_summary,
        review_metrics=metrics_review_payload,
        product_metrics=product_metrics,
        worst_product_rows=worst_product_rows,
        inversion_rows=inversion_rows,
        sentiment_error_rows=sentiment_error_rows,
    )

    run_summary = {
        "status": "OK",
        "out_dir": str(out_dir),
        "frozen_detection_check": detection_payload,
        "metrics_review_level": metrics_review_payload,
        "metrics_product_level": product_metrics,
        "latency": latency_payload,
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
