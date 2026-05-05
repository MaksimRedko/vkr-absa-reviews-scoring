from __future__ import annotations

import argparse
import ast
import json
import math
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

from configs import configs as _cfg_module
from scripts import run_phase2_baseline_matching as lexical
from src.stages.segmentation import RuleBasedClauseSegmenter
from src.vocabulary.loader import AspectDefinition

REFERENCE_BASELINE = {
    "macro_precision": 0.4806,
    "macro_recall": 0.4130,
    "macro_f1": 0.4251,
}


@dataclass(slots=True)
class ReviewRecord:
    review_id: str
    product_id: int
    category: str
    source: str
    text: str
    true_labels_raw: dict[str, float]
    true_labels_lemma: dict[str, float]


@dataclass(slots=True)
class ReviewEval:
    review_id: str
    product_id: int
    category: str
    source: str
    pred_aspect_ids: set[str]
    true_aspect_ids: set[str]
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class SegmentMatch:
    review_id: str
    product_id: int
    category: str
    source: str
    segment_index: int
    segment_text: str
    best_aspect_id: str
    best_aspect_name: str
    best_score: float
    second_best_aspect_id: str
    second_best_score: float


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
    required = {"nm_id", "id", "full_text", "true_labels", "category", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    rows: list[ReviewRecord] = []
    for _, row in df.iterrows():
        text = str(row["full_text"]).strip()
        if not text:
            continue
        true_labels_raw = _parse_true_labels(row["true_labels"])
        true_labels_lemma = {
            lexical._normalize(key): float(value)
            for key, value in true_labels_raw.items()
            if lexical._normalize(key)
        }
        rows.append(
            ReviewRecord(
                review_id=str(row["id"]),
                product_id=int(row["nm_id"]),
                category=str(row["category"]).strip(),
                source=str(row["source"]).strip(),
                text=text,
                true_labels_raw=true_labels_raw,
                true_labels_lemma=true_labels_lemma,
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
    dict[str, dict[str, set[str]]],
]:
    aspects_by_category: dict[str, list[AspectDefinition]] = {}
    term_to_aspects_by_category: dict[str, dict[str, set[str]]] = {}
    aspect_name_by_category: dict[str, dict[str, str]] = {}
    aspect_terms_by_category: dict[str, dict[str, set[str]]] = {}

    for category in sorted(categories):
        domain_path = domain_vocab_by_category.get(category)
        paths = [core_vocab_path] + ([domain_path] if domain_path else [])
        aspects = lexical._build_vocabulary(paths)
        term_to_aspects, _ = lexical._term_indexes(aspects)
        aspects_by_category[category] = aspects
        term_to_aspects_by_category[category] = term_to_aspects
        aspect_name_by_category[category] = {aspect.id: aspect.canonical_name for aspect in aspects}
        aspect_terms_by_category[category] = {
            aspect.id: {
                lexical._normalize(aspect.canonical_name),
                *(lexical._normalize(term) for term in aspect.synonyms),
            }
            for aspect in aspects
        }
    return (
        aspects_by_category,
        term_to_aspects_by_category,
        aspect_name_by_category,
        aspect_terms_by_category,
    )


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if denom <= 0.0:
        return np.zeros_like(values)
    return exp / denom


def _load_encoder_once() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_cfg_module.config.models.encoder_path)


def _encode_texts_cached(
    model: Any,
    texts: list[str],
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    unique = sorted({text for text in texts if text and text not in cache})
    if unique:
        embeddings = model.encode(unique, show_progress_bar=False, convert_to_numpy=True)
        normalized = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        for idx, text in enumerate(unique):
            cache[text] = normalized[idx]
    return {text: cache[text] for text in texts if text in cache}


def _build_aspect_centroids(
    model: Any,
    aspects: list[AspectDefinition],
    cache: dict[str, np.ndarray],
) -> tuple[list[str], list[str], np.ndarray]:
    aspect_ids: list[str] = []
    aspect_names: list[str] = []
    aspect_vectors: list[np.ndarray] = []
    for aspect in aspects:
        texts = [aspect.canonical_name] + list(aspect.synonyms)
        vectors = _encode_texts_cached(model, texts, cache)
        if not vectors:
            continue
        mat = np.stack([vectors[text] for text in texts if text in vectors], axis=0).astype(np.float32)
        centroid = _l2_normalize(mat.mean(axis=0, keepdims=True))[0]
        aspect_ids.append(aspect.id)
        aspect_names.append(aspect.canonical_name)
        aspect_vectors.append(centroid)
    if not aspect_vectors:
        return [], [], np.zeros((0, 0), dtype=np.float32)
    return aspect_ids, aspect_names, np.stack(aspect_vectors, axis=0).astype(np.float32)


def _true_aspect_ids_for_review(
    review: ReviewRecord,
    term_to_aspects: dict[str, set[str]],
) -> set[str]:
    true_aspect_ids: set[str] = set()
    for gold_lemma in review.true_labels_lemma.keys():
        true_aspect_ids.update(term_to_aspects.get(gold_lemma, set()))
    return true_aspect_ids


def _evaluate_rows(rows: list[ReviewEval]) -> dict[str, Any]:
    if not rows:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "n_reviews": 0,
            "avg_predicted_aspects_per_review": 0.0,
        }
    return {
        "macro_precision": round(float(np.mean([row.precision for row in rows])), 4),
        "macro_recall": round(float(np.mean([row.recall for row in rows])), 4),
        "macro_f1": round(float(np.mean([row.f1 for row in rows])), 4),
        "n_reviews": int(len(rows)),
        "avg_predicted_aspects_per_review": round(float(np.mean([len(row.pred_aspect_ids) for row in rows])), 4),
    }


def _group_metrics(rows: list[ReviewEval], key: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[ReviewEval]] = defaultdict(list)
    for row in rows:
        buckets[str(getattr(row, key))].append(row)
    out: list[dict[str, Any]] = []
    for group_value, group_rows in sorted(buckets.items()):
        metric = _evaluate_rows(group_rows)
        metric[key] = group_value
        out.append(metric)
    return out


def _run_baseline_detection(
    reviews: list[ReviewRecord],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
) -> tuple[dict[str, Any], list[ReviewEval], pd.DataFrame]:
    extractor = lexical.CandidateExtractor(ngram_range=(1, 2), min_word_length=3)
    extractor.dependency_filter_enabled = False

    rows: list[ReviewEval] = []
    csv_rows: list[dict[str, Any]] = []

    for review in reviews:
        term_to_aspects = term_to_aspects_by_category[review.category]
        candidate_lemmas = lexical._extract_candidate_lemmas_by_unit(review.text, extractor, "candidates")
        matched_terms = lexical._match_terms(candidate_lemmas, term_to_aspects, "lexical_only")

        pred_aspect_ids: set[str] = set()
        for term in matched_terms:
            pred_aspect_ids.update(term_to_aspects[term])

        true_aspect_ids = _true_aspect_ids_for_review(review, term_to_aspects)
        p, r, f1 = lexical._eval_prf(pred_aspect_ids, true_aspect_ids)
        eval_row = ReviewEval(
            review_id=review.review_id,
            product_id=review.product_id,
            category=review.category,
            source=review.source,
            pred_aspect_ids=pred_aspect_ids,
            true_aspect_ids=true_aspect_ids,
            precision=p,
            recall=r,
            f1=f1,
        )
        rows.append(eval_row)
        csv_rows.append(
            {
                "review_id": review.review_id,
                "product_id": review.product_id,
                "category": review.category,
                "source": review.source,
                "predicted_aspect_ids_json": json.dumps(sorted(pred_aspect_ids), ensure_ascii=False),
                "true_aspect_ids_json": json.dumps(sorted(true_aspect_ids), ensure_ascii=False),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }
        )

    overall = _evaluate_rows(rows)
    overall["per_category"] = _group_metrics(rows, "category")
    overall["per_source"] = _group_metrics(rows, "source")
    overall["reproduction_ok"] = (
        overall["macro_precision"] == REFERENCE_BASELINE["macro_precision"]
        and overall["macro_recall"] == REFERENCE_BASELINE["macro_recall"]
        and overall["macro_f1"] == REFERENCE_BASELINE["macro_f1"]
    )
    return overall, rows, pd.DataFrame(csv_rows)


def _collect_segment_matches(
    reviews: list[ReviewRecord],
    aspects_by_category: dict[str, list[AspectDefinition]],
    aspect_name_by_category: dict[str, dict[str, str]],
) -> tuple[list[SegmentMatch], dict[str, list[str]], float]:
    segmenter = RuleBasedClauseSegmenter()
    model = _load_encoder_once()
    cache: dict[str, np.ndarray] = {}
    aspect_cache: dict[str, tuple[list[str], list[str], np.ndarray]] = {}
    for category, aspects in aspects_by_category.items():
        aspect_cache[category] = _build_aspect_centroids(model, aspects, cache)

    segment_texts_by_review: dict[str, list[str]] = {}
    matches: list[SegmentMatch] = []
    segment_count = 0

    for review in reviews:
        segments = segmenter.split(review.text, source_review_id=review.review_id)
        segment_texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        if not segment_texts:
            segment_texts = [review.text.strip()]
        segment_texts_by_review[review.review_id] = segment_texts
        segment_count += len(segment_texts)

        aspect_ids, aspect_names, aspect_matrix = aspect_cache[review.category]
        if aspect_matrix.size == 0:
            continue

        segment_vectors = _encode_texts_cached(model, segment_texts, cache)
        for seg_idx, segment_text in enumerate(segment_texts):
            seg_vec = segment_vectors.get(segment_text)
            if seg_vec is None:
                continue
            scores = aspect_matrix @ seg_vec
            probs = _softmax(scores)
            order = np.argsort(scores)[::-1]
            best_idx = int(order[0])
            second_idx = int(order[1]) if len(order) > 1 else best_idx
            matches.append(
                SegmentMatch(
                    review_id=review.review_id,
                    product_id=review.product_id,
                    category=review.category,
                    source=review.source,
                    segment_index=seg_idx,
                    segment_text=segment_text,
                    best_aspect_id=aspect_ids[best_idx],
                    best_aspect_name=aspect_name_by_category[review.category].get(aspect_ids[best_idx], aspect_names[best_idx]),
                    best_score=float(probs[best_idx]),
                    second_best_aspect_id=aspect_ids[second_idx],
                    second_best_score=float(probs[second_idx]),
                )
            )

    avg_segments = float(segment_count / len(reviews)) if reviews else 0.0
    return matches, segment_texts_by_review, avg_segments


def _build_review_aspect_score_map(matches: list[SegmentMatch]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for match in matches:
        prev = out[match.review_id].get(match.best_aspect_id)
        if prev is None or match.best_score > prev:
            out[match.review_id][match.best_aspect_id] = match.best_score
    return out


def _evaluate_semantic_at_threshold(
    reviews: list[ReviewRecord],
    review_aspect_score_map: dict[str, dict[str, float]],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
    threshold: float,
) -> list[ReviewEval]:
    rows: list[ReviewEval] = []
    for review in reviews:
        aspect_score_map = review_aspect_score_map.get(review.review_id, {})
        pred_aspect_ids = {
            aspect_id
            for aspect_id, score in aspect_score_map.items()
            if float(score) > threshold
        }
        true_aspect_ids = _true_aspect_ids_for_review(review, term_to_aspects_by_category[review.category])
        p, r, f1 = lexical._eval_prf(pred_aspect_ids, true_aspect_ids)
        rows.append(
            ReviewEval(
                review_id=review.review_id,
                product_id=review.product_id,
                category=review.category,
                source=review.source,
                pred_aspect_ids=pred_aspect_ids,
                true_aspect_ids=true_aspect_ids,
                precision=p,
                recall=r,
                f1=f1,
            )
        )
    return rows


def _select_threshold_lopo(
    reviews: list[ReviewRecord],
    review_aspect_score_map: dict[str, dict[str, float]],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
) -> tuple[dict[int, float], list[ReviewEval]]:
    all_scores = sorted(
        {
            round(float(score), 6)
            for review_scores in review_aspect_score_map.values()
            for score in review_scores.values()
        }
    )
    if not all_scores:
        threshold_by_product = {review.product_id: 1.0 for review in reviews}
        return threshold_by_product, _evaluate_semantic_at_threshold(reviews, review_aspect_score_map, term_to_aspects_by_category, 1.0)

    thr_min = min(all_scores)
    thr_max = max(all_scores)
    thresholds = sorted(
        {
            round(float(x), 6)
            for x in np.linspace(thr_min, thr_max, num=101)
        }
        | {round(thr_min - 1e-6, 6), round(thr_max + 1e-6, 6)}
    )

    by_product: dict[int, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        by_product[review.product_id].append(review)

    threshold_by_product: dict[int, float] = {}
    oof_rows: list[ReviewEval] = []

    for product_id, holdout_reviews in sorted(by_product.items()):
        train_reviews = [review for review in reviews if review.product_id != product_id]
        best_threshold = thresholds[0]
        best_f1 = -1.0
        best_recall = -1.0
        best_precision = -1.0

        for threshold in thresholds:
            train_eval = _evaluate_semantic_at_threshold(
                train_reviews,
                review_aspect_score_map,
                term_to_aspects_by_category,
                threshold,
            )
            metric = _evaluate_rows(train_eval)
            f1 = float(metric["macro_f1"])
            recall = float(metric["macro_recall"])
            precision = float(metric["macro_precision"])
            if (
                f1 > best_f1
                or (math.isclose(f1, best_f1) and recall > best_recall)
                or (math.isclose(f1, best_f1) and math.isclose(recall, best_recall) and precision > best_precision)
            ):
                best_threshold = threshold
                best_f1 = f1
                best_recall = recall
                best_precision = precision

        threshold_by_product[product_id] = float(best_threshold)
        oof_rows.extend(
            _evaluate_semantic_at_threshold(
                holdout_reviews,
                review_aspect_score_map,
                term_to_aspects_by_category,
                best_threshold,
            )
        )

    return threshold_by_product, oof_rows


def _semantic_predictions_df(rows: list[ReviewEval]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_id": row.review_id,
                "product_id": row.product_id,
                "category": row.category,
                "source": row.source,
                "predicted_aspect_ids_json": json.dumps(sorted(row.pred_aspect_ids), ensure_ascii=False),
                "true_aspect_ids_json": json.dumps(sorted(row.true_aspect_ids), ensure_ascii=False),
                "precision": round(row.precision, 4),
                "recall": round(row.recall, 4),
                "f1": round(row.f1, 4),
            }
            for row in rows
        ]
    )


def _build_metrics_payload(rows: list[ReviewEval], avg_segments_per_review: float | None = None) -> dict[str, Any]:
    overall = _evaluate_rows(rows)
    overall["per_category"] = _group_metrics(rows, "category")
    overall["per_source"] = _group_metrics(rows, "source")
    if avg_segments_per_review is not None:
        overall["avg_segments_per_review"] = round(float(avg_segments_per_review), 4)
    return overall


def _head_to_head_rows(
    baseline: dict[str, Any],
    semantic: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_row(scope: str, group_value: str, base: dict[str, Any], sem: dict[str, Any]) -> None:
        rows.append(
            {
                "scope": scope,
                "group": group_value,
                "baseline_precision": base["macro_precision"],
                "baseline_recall": base["macro_recall"],
                "baseline_f1": base["macro_f1"],
                "semantic_precision": sem["macro_precision"],
                "semantic_recall": sem["macro_recall"],
                "semantic_f1": sem["macro_f1"],
                "delta_precision": round(float(sem["macro_precision"] - base["macro_precision"]), 4),
                "delta_recall": round(float(sem["macro_recall"] - base["macro_recall"]), 4),
                "delta_f1": round(float(sem["macro_f1"] - base["macro_f1"]), 4),
            }
        )

    append_row("overall", "all", baseline, semantic)

    base_cat = {row["category"]: row for row in baseline["per_category"]}
    sem_cat = {row["category"]: row for row in semantic["per_category"]}
    for category in sorted(set(base_cat) | set(sem_cat)):
        append_row("category", category, base_cat.get(category, {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}), sem_cat.get(category, {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}))

    base_src = {row["source"]: row for row in baseline["per_source"]}
    sem_src = {row["source"]: row for row in semantic["per_source"]}
    for source in sorted(set(base_src) | set(sem_src)):
        append_row("source", source, base_src.get(source, {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}), sem_src.get(source, {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}))

    return pd.DataFrame(rows)


def _contains_explicit_vocab_term(segment_text: str, aspect_terms_lemma: set[str]) -> bool:
    segment_lemma = lexical._normalize(segment_text)
    if not segment_lemma:
        return False
    padded = f" {segment_lemma} "
    for term in aspect_terms_lemma:
        if not term:
            continue
        if f" {term} " in padded:
            return True
    return False


def _find_diagnostics(
    reviews: list[ReviewRecord],
    baseline_rows: list[ReviewEval],
    semantic_rows: list[ReviewEval],
    matches: list[SegmentMatch],
    threshold_by_product: dict[int, float],
    aspect_terms_by_category: dict[str, dict[str, set[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    review_by_id = {review.review_id: review for review in reviews}
    baseline_by_id = {row.review_id: row for row in baseline_rows}
    semantic_by_id = {row.review_id: row for row in semantic_rows}

    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for match in matches:
        threshold = float(threshold_by_product.get(match.product_id, 1.0))
        if not (match.best_score > threshold):
            continue
        review = review_by_id[match.review_id]
        baseline_pred = baseline_by_id[match.review_id].pred_aspect_ids
        semantic_pred = semantic_by_id[match.review_id].pred_aspect_ids
        true_aspects = semantic_by_id[match.review_id].true_aspect_ids
        if match.best_aspect_id not in semantic_pred:
            continue

        explicit = _contains_explicit_vocab_term(
            match.segment_text,
            aspect_terms_by_category[match.category].get(match.best_aspect_id, set()),
        )
        base_payload = {
            "review_id": match.review_id,
            "product_id": match.product_id,
            "category": match.category,
            "source": match.source,
            "segment_text": match.segment_text,
            "aspect_id": match.best_aspect_id,
            "best_score": round(match.best_score, 4),
            "second_best_score": round(match.second_best_score, 4),
            "baseline_had_aspect": match.best_aspect_id in baseline_pred,
            "explicit_vocab_term_in_segment": explicit,
            "review_text_preview": review.text[:240],
        }
        if (
            match.best_aspect_id in true_aspects
            and match.best_aspect_id not in baseline_pred
            and not explicit
        ):
            success_rows.append(base_payload)
        if match.best_aspect_id not in true_aspects:
            failure_rows.append(base_payload)

    success_df = pd.DataFrame(success_rows)
    if not success_df.empty:
        success_df = success_df.sort_values(
            ["best_score", "review_id", "segment_text"],
            ascending=[False, True, True],
        ).drop_duplicates(["review_id", "aspect_id"]).head(10)

    failure_df = pd.DataFrame(failure_rows)
    if not failure_df.empty:
        failure_df = failure_df.sort_values(
            ["best_score", "review_id", "segment_text"],
            ascending=[False, True, True],
        ).drop_duplicates(["review_id", "aspect_id"]).head(10)

    return success_df, failure_df


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(
    out_dir: Path,
    baseline: dict[str, Any],
    semantic: dict[str, Any] | None,
    head_to_head_df: pd.DataFrame | None,
    success_df: pd.DataFrame | None,
    failure_df: pd.DataFrame | None,
    threshold_by_product: dict[int, float] | None,
    stop_reason: str | None = None,
) -> None:
    lines = [
        "# phase3_step9_segment_to_aspect_semantic_matching",
        "",
        "## A. Head-to-head",
        f"- baseline precision / recall / F1: {baseline['macro_precision']:.4f} / {baseline['macro_recall']:.4f} / {baseline['macro_f1']:.4f}",
    ]

    if semantic is None:
        lines.extend(
            [
                "- semantic precision / recall / F1: not run",
                "- delta: not run",
                "",
                "## Stop",
                f"- {stop_reason or 'baseline reproduction failed'}",
                "",
            ]
        )
        (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            f"- semantic precision / recall / F1: {semantic['macro_precision']:.4f} / {semantic['macro_recall']:.4f} / {semantic['macro_f1']:.4f}",
            f"- delta precision / recall / F1: {semantic['macro_precision'] - baseline['macro_precision']:+.4f} / {semantic['macro_recall'] - baseline['macro_recall']:+.4f} / {semantic['macro_f1'] - baseline['macro_f1']:+.4f}",
            f"- avg segments per review: {semantic['avg_segments_per_review']:.4f}",
            f"- avg predicted aspects per review: {semantic['avg_predicted_aspects_per_review']:.4f}",
            "",
            "## B. Breakdown",
            "### per-category",
            _markdown_table(
                pd.DataFrame(head_to_head_df[head_to_head_df["scope"] == "category"]),
                ["group", "baseline_precision", "baseline_recall", "baseline_f1", "semantic_precision", "semantic_recall", "semantic_f1", "delta_f1"],
            ) if head_to_head_df is not None else "_none_",
            "",
            "### per-source",
            _markdown_table(
                pd.DataFrame(head_to_head_df[head_to_head_df["scope"] == "source"]),
                ["group", "baseline_precision", "baseline_recall", "baseline_f1", "semantic_precision", "semantic_recall", "semantic_f1", "delta_f1"],
            ) if head_to_head_df is not None else "_none_",
            "",
            "## C. Diagnostics",
            f"- LOPO median threshold: {float(np.median(list(threshold_by_product.values()))):.4f}" if threshold_by_product else "- LOPO median threshold: null",
            "",
            "### 10 successful cases",
            _markdown_table(
                success_df if success_df is not None else pd.DataFrame(),
                ["review_id", "product_id", "category", "source", "aspect_id", "best_score", "segment_text"],
            ),
            "",
            "### 10 bad cases",
            _markdown_table(
                failure_df if failure_df is not None else pd.DataFrame(),
                ["review_id", "product_id", "category", "source", "aspect_id", "best_score", "segment_text"],
            ),
            "",
            "## D. Decision",
        ]
    )

    semantic_f1 = float(semantic["macro_f1"])
    if semantic_f1 >= 0.44:
        decision = "STRONG PASS"
    elif semantic_f1 > REFERENCE_BASELINE["macro_f1"]:
        decision = "PASS"
    else:
        decision = "FAIL"
    lines.append(f"- {decision}")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="phase3_step9 segment-to-aspect semantic matching")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase3_step9_segment_to_aspect_semantic_matching")
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
    (
        aspects_by_category,
        term_to_aspects_by_category,
        aspect_name_by_category,
        aspect_terms_by_category,
    ) = _build_hybrid_cache(
        core_vocab_path=ROOT / args.core_vocab,
        domain_vocab_by_category=domain_vocab_by_category,
        categories={review.category for review in reviews},
    )

    t_baseline = time.perf_counter()
    baseline_metrics, baseline_rows, baseline_df = _run_baseline_detection(reviews, term_to_aspects_by_category)
    baseline_sec = time.perf_counter() - t_baseline
    _write_json(out_dir / "metrics_baseline.json", baseline_metrics)
    baseline_df.to_csv(out_dir / "review_predictions_baseline.csv", index=False, encoding="utf-8")

    if not baseline_metrics["reproduction_ok"]:
        _write_summary(
            out_dir=out_dir,
            baseline=baseline_metrics,
            semantic=None,
            head_to_head_df=None,
            success_df=None,
            failure_df=None,
            threshold_by_product=None,
            stop_reason="baseline branch did not reproduce 0.4806 / 0.4130 / 0.4251",
        )
        run_summary = {
            "status": "STOP",
            "out_dir": str(out_dir),
            "baseline_metrics": baseline_metrics,
            "baseline_sec": round(baseline_sec, 4),
            "total_sec": round(time.perf_counter() - total_started, 4),
        }
        _write_json(out_dir / "run_summary.json", run_summary)
        print(json.dumps(run_summary, ensure_ascii=False, indent=2))
        return

    t_semantic = time.perf_counter()
    matches, _segment_texts_by_review, avg_segments_per_review = _collect_segment_matches(
        reviews=reviews,
        aspects_by_category=aspects_by_category,
        aspect_name_by_category=aspect_name_by_category,
    )
    review_aspect_score_map = _build_review_aspect_score_map(matches)
    threshold_by_product, semantic_rows = _select_threshold_lopo(
        reviews=reviews,
        review_aspect_score_map=review_aspect_score_map,
        term_to_aspects_by_category=term_to_aspects_by_category,
    )
    semantic_sec = time.perf_counter() - t_semantic

    semantic_metrics = _build_metrics_payload(semantic_rows, avg_segments_per_review=avg_segments_per_review)
    semantic_metrics["lopo_threshold_by_product"] = {str(key): round(float(value), 6) for key, value in threshold_by_product.items()}
    semantic_metrics["n_segment_matches"] = int(len(matches))
    semantic_metrics["n_reviews"] = int(len(semantic_rows))
    _write_json(out_dir / "metrics_semantic.json", semantic_metrics)

    segment_level_df = pd.DataFrame(
        [
            {
                "review_id": match.review_id,
                "product_id": match.product_id,
                "category": match.category,
                "source": match.source,
                "segment_index": match.segment_index,
                "segment_text": match.segment_text,
                "best_aspect_id": match.best_aspect_id,
                "best_aspect_name": match.best_aspect_name,
                "best_score": round(match.best_score, 6),
                "second_best_aspect_id": match.second_best_aspect_id,
                "second_best_score": round(match.second_best_score, 6),
                "threshold_used": round(float(threshold_by_product.get(match.product_id, 1.0)), 6),
                "accepted": bool(match.best_score > float(threshold_by_product.get(match.product_id, 1.0))),
            }
            for match in matches
        ]
    )
    segment_level_df.to_csv(out_dir / "segment_level_matches.csv", index=False, encoding="utf-8")

    semantic_df = _semantic_predictions_df(semantic_rows)
    semantic_df.to_csv(out_dir / "review_predictions_semantic.csv", index=False, encoding="utf-8")

    head_to_head_df = _head_to_head_rows(baseline_metrics, semantic_metrics)
    head_to_head_df.to_csv(out_dir / "head_to_head.csv", index=False, encoding="utf-8")

    success_df, failure_df = _find_diagnostics(
        reviews=reviews,
        baseline_rows=baseline_rows,
        semantic_rows=semantic_rows,
        matches=matches,
        threshold_by_product=threshold_by_product,
        aspect_terms_by_category=aspect_terms_by_category,
    )
    _write_summary(
        out_dir=out_dir,
        baseline=baseline_metrics,
        semantic=semantic_metrics,
        head_to_head_df=head_to_head_df,
        success_df=success_df,
        failure_df=failure_df,
        threshold_by_product=threshold_by_product,
    )

    run_summary = {
        "status": "OK",
        "out_dir": str(out_dir),
        "baseline_metrics": baseline_metrics,
        "semantic_metrics": semantic_metrics,
        "latency": {
            "baseline_sec": round(baseline_sec, 4),
            "semantic_sec": round(semantic_sec, 4),
            "total_sec": round(time.perf_counter() - total_started, 4),
        },
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
