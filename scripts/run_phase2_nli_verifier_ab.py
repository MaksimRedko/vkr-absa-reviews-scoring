from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_phase2_baseline_matching as lexical
from src.schemas.models import Candidate
from src.stages.extraction import CandidateExtractor
from src.stages.sentiment import SentimentEngine
from src.vocabulary.loader import AspectDefinition

REFERENCE_BASELINE = {
    "macro_precision": 0.4806,
    "macro_recall": 0.4130,
    "macro_f1": 0.4251,
}
VERIFIER_HYPOTHESIS = "В этом фрагменте обсуждается {canonical_aspect_name}"


@dataclass(slots=True)
class ReviewRecord:
    review_id: str
    nm_id: int
    category: str
    text: str
    true_aspect_ids: set[str]
    baseline_pred_aspect_ids: set[str]


@dataclass(slots=True)
class EvalSummary:
    macro_precision: float
    macro_recall: float
    macro_f1: float
    n_reviews: int


def _aspect_name_map(aspects: list[AspectDefinition]) -> dict[str, str]:
    return {asp.id: asp.canonical_name for asp in aspects}


def _build_hybrid_aspects_by_category(
    core_vocab_path: Path,
    domain_vocab_by_category: dict[str, Path],
    categories: set[str],
) -> dict[str, list[AspectDefinition]]:
    cache: dict[str, list[AspectDefinition]] = {}
    for category in sorted(categories):
        domain_path = domain_vocab_by_category.get(category)
        paths = [core_vocab_path] + ([domain_path] if domain_path else [])
        cache[category] = lexical._build_vocabulary(paths)
    return cache


def _minimal_premise_text(candidate: Candidate, review_text: str) -> str:
    sentence = str(candidate.sentence or "").strip()
    if sentence:
        return sentence

    target = str(candidate.source_span or candidate.span or "").strip()
    if target:
        fragments = [
            part.strip()
            for part in lexical._split_segments_rule_based(review_text)
            if part and target.lower() in part.lower()
        ]
        if fragments:
            return min(fragments, key=len)
    return target or str(candidate.span or "").strip() or str(review_text).strip()


def _collect_lexical_hits(
    reviews: list[lexical.ReviewSample],
    core_vocab_path: Path,
    domain_vocab_by_category: dict[str, Path],
) -> tuple[list[ReviewRecord], pd.DataFrame]:
    extractor = CandidateExtractor(ngram_range=(1, 2), min_word_length=3)
    extractor.dependency_filter_enabled = False

    aspects_by_category = _build_hybrid_aspects_by_category(
        core_vocab_path=core_vocab_path,
        domain_vocab_by_category=domain_vocab_by_category,
        categories={review.category for review in reviews},
    )

    review_records: list[ReviewRecord] = []
    hit_rows: list[dict[str, Any]] = []

    for review in reviews:
        aspects = aspects_by_category[review.category]
        term_to_aspects, _ = lexical._term_indexes(aspects)
        aspect_name_by_id = _aspect_name_map(aspects)
        candidates = extractor.extract(review.text)

        candidate_lemmas: set[str] = set()
        for candidate in candidates:
            lemma = lexical._normalize(candidate.span)
            if not lemma:
                continue
            candidate_lemmas.add(lemma)
            if lemma not in term_to_aspects:
                continue
            premise_text = _minimal_premise_text(candidate, review.text)
            candidate_span = str(candidate.source_span or candidate.span).strip()
            for aspect_id in sorted(term_to_aspects[lemma]):
                hit_rows.append(
                    {
                        "review_id": review.review_id,
                        "product_id": int(review.nm_id),
                        "category": review.category,
                        "candidate_span": candidate_span,
                        "aspect_id": aspect_id,
                        "canonical_aspect_name": aspect_name_by_id[aspect_id],
                        "lexical_hit": True,
                        "premise_text": premise_text,
                    }
                )

        matched_terms = lexical._match_terms(candidate_lemmas, term_to_aspects, "lexical_only")
        baseline_pred_aspect_ids: set[str] = set()
        for term in matched_terms:
            baseline_pred_aspect_ids.update(term_to_aspects[term])

        true_aspect_ids: set[str] = set()
        for gold_label in review.true_labels_lemma:
            true_aspect_ids.update(term_to_aspects.get(gold_label, set()))

        review_records.append(
            ReviewRecord(
                review_id=review.review_id,
                nm_id=int(review.nm_id),
                category=review.category,
                text=review.text,
                true_aspect_ids=true_aspect_ids,
                baseline_pred_aspect_ids=baseline_pred_aspect_ids,
            )
        )

    return review_records, pd.DataFrame(hit_rows)


def _evaluate_predictions(
    review_records: list[ReviewRecord],
    pred_by_review: dict[str, set[str]],
) -> tuple[EvalSummary, pd.DataFrame, pd.DataFrame, Counter[str], Counter[str]]:
    rows: list[dict[str, Any]] = []
    per_category_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    fp_by_aspect: Counter[str] = Counter()
    fn_by_aspect: Counter[str] = Counter()

    for review in review_records:
        pred = set(pred_by_review.get(review.review_id, set()))
        true = set(review.true_aspect_ids)
        p, r, f1 = lexical._eval_prf(pred, true)
        per_category_rows[review.category].append((p, r, f1))
        for aspect_id in (pred - true):
            fp_by_aspect[aspect_id] += 1
        for aspect_id in (true - pred):
            fn_by_aspect[aspect_id] += 1
        rows.append(
            {
                "review_id": review.review_id,
                "product_id": review.nm_id,
                "category": review.category,
                "predicted_aspect_ids_json": json.dumps(sorted(pred), ensure_ascii=False),
                "true_aspect_ids_json": json.dumps(sorted(true), ensure_ascii=False),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }
        )

    review_df = pd.DataFrame(rows)
    if review_df.empty:
        summary = EvalSummary(0.0, 0.0, 0.0, 0)
    else:
        summary = EvalSummary(
            macro_precision=float(review_df["precision"].mean()),
            macro_recall=float(review_df["recall"].mean()),
            macro_f1=float(review_df["f1"].mean()),
            n_reviews=int(len(review_df)),
        )

    per_category_records: list[dict[str, Any]] = []
    for category, vals in sorted(per_category_rows.items()):
        ps = [x[0] for x in vals]
        rs = [x[1] for x in vals]
        fs = [x[2] for x in vals]
        per_category_records.append(
            {
                "category": category,
                "n_reviews": len(vals),
                "macro_precision": round(sum(ps) / len(ps), 4),
                "macro_recall": round(sum(rs) / len(rs), 4),
                "macro_f1": round(sum(fs) / len(fs), 4),
            }
        )

    return summary, review_df, pd.DataFrame(per_category_records), fp_by_aspect, fn_by_aspect


def _baseline_prediction_map(review_records: list[ReviewRecord]) -> dict[str, set[str]]:
    return {review.review_id: set(review.baseline_pred_aspect_ids) for review in review_records}


def _score_verifier_pairs(hit_df: pd.DataFrame) -> tuple[pd.DataFrame, float, int, int]:
    if hit_df.empty:
        scored = hit_df.copy()
        scored["p_ent"] = []
        return scored, 0.0, 0, 0

    engine = SentimentEngine()
    premises = hit_df["premise_text"].astype(str).tolist()
    hypotheses = [
        VERIFIER_HYPOTHESIS.format(canonical_aspect_name=name)
        for name in hit_df["canonical_aspect_name"].astype(str).tolist()
    ]

    started_at = time.perf_counter()
    logits = engine._forward_logits_tensor(premises, hypotheses)
    probs = torch.softmax(logits / engine.temperature, dim=1).cpu().numpy()
    elapsed = time.perf_counter() - started_at

    scored = hit_df.copy()
    scored["p_ent"] = probs[:, engine.ent_idx].astype(np.float32)
    unique_pairs = len(set(zip(premises, hypotheses)))
    return scored, elapsed, int(len(scored)), int(unique_pairs)


def _review_aspect_max_scores(hit_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    if hit_df.empty:
        return scores
    grouped = (
        hit_df.groupby(["review_id", "aspect_id"], as_index=False)["p_ent"]
        .max()
        .sort_values(["review_id", "aspect_id"])
    )
    for row in grouped.itertuples(index=False):
        scores[str(row.review_id)][str(row.aspect_id)] = float(row.p_ent)
    return scores


def _threshold_grid(max_score_by_review_aspect: dict[str, dict[str, float]]) -> list[float]:
    values = {-1.0}
    for aspect_scores in max_score_by_review_aspect.values():
        values.update(float(score) for score in aspect_scores.values())
    return sorted(values)


def _prediction_map_for_threshold(
    review_aspect_scores: dict[str, dict[str, float]],
    threshold: float,
    review_ids: list[str] | None = None,
) -> dict[str, set[str]]:
    pred_map: dict[str, set[str]] = {}
    keys = review_ids if review_ids is not None else list(review_aspect_scores.keys())
    for review_id in keys:
        aspect_scores = review_aspect_scores.get(review_id, {})
        pred_map[review_id] = {
            aspect_id for aspect_id, score in aspect_scores.items() if float(score) > threshold
        }
    return pred_map


def _choose_threshold_on_train(
    threshold_grid: list[float],
    train_review_ids: list[str],
    review_aspect_scores: dict[str, dict[str, float]],
    review_lookup: dict[str, ReviewRecord],
) -> tuple[float, EvalSummary]:
    best_threshold = threshold_grid[0]
    best_summary = EvalSummary(-1.0, -1.0, -1.0, 0)

    for threshold in threshold_grid:
        pred_map = _prediction_map_for_threshold(
            review_aspect_scores=review_aspect_scores,
            threshold=threshold,
            review_ids=train_review_ids,
        )
        rows: list[tuple[float, float, float]] = []
        for review_id in train_review_ids:
            review = review_lookup[review_id]
            pred = set(pred_map.get(review_id, set()))
            true = set(review.true_aspect_ids)
            rows.append(lexical._eval_prf(pred, true))
        if not rows:
            summary = EvalSummary(0.0, 0.0, 0.0, 0)
        else:
            ps = [x[0] for x in rows]
            rs = [x[1] for x in rows]
            fs = [x[2] for x in rows]
            summary = EvalSummary(
                macro_precision=float(sum(ps) / len(ps)),
                macro_recall=float(sum(rs) / len(rs)),
                macro_f1=float(sum(fs) / len(fs)),
                n_reviews=len(rows),
            )
        better = False
        if summary.macro_f1 > best_summary.macro_f1 + 1e-12:
            better = True
        elif abs(summary.macro_f1 - best_summary.macro_f1) <= 1e-12:
            if summary.macro_recall > best_summary.macro_recall + 1e-12:
                better = True
            elif abs(summary.macro_recall - best_summary.macro_recall) <= 1e-12:
                if summary.macro_precision > best_summary.macro_precision + 1e-12:
                    better = True
                elif abs(summary.macro_precision - best_summary.macro_precision) <= 1e-12:
                    if threshold < best_threshold:
                        better = True
        if better:
            best_threshold = threshold
            best_summary = summary

    return best_threshold, best_summary


def _run_lopo_threshold_selection(
    review_records: list[ReviewRecord],
    review_aspect_scores: dict[str, dict[str, float]],
) -> tuple[dict[int, float], dict[str, set[str]], pd.DataFrame]:
    threshold_grid = _threshold_grid(review_aspect_scores)
    review_lookup = {review.review_id: review for review in review_records}
    product_to_review_ids: dict[int, list[str]] = defaultdict(list)
    for review in review_records:
        product_to_review_ids[review.nm_id].append(review.review_id)

    threshold_by_product: dict[int, float] = {}
    pred_by_review: dict[str, set[str]] = {}
    fold_rows: list[dict[str, Any]] = []

    for product_id in sorted(product_to_review_ids.keys()):
        holdout_review_ids = list(product_to_review_ids[product_id])
        train_review_ids = [
            review.review_id
            for review in review_records
            if review.nm_id != product_id
        ]
        selected_threshold, train_summary = _choose_threshold_on_train(
            threshold_grid=threshold_grid,
            train_review_ids=train_review_ids,
            review_aspect_scores=review_aspect_scores,
            review_lookup=review_lookup,
        )
        threshold_by_product[product_id] = selected_threshold

        holdout_pred_map = _prediction_map_for_threshold(
            review_aspect_scores=review_aspect_scores,
            threshold=selected_threshold,
            review_ids=holdout_review_ids,
        )
        pred_by_review.update(holdout_pred_map)

        holdout_rows: list[tuple[float, float, float]] = []
        for review_id in holdout_review_ids:
            review = review_lookup[review_id]
            pred = set(holdout_pred_map.get(review_id, set()))
            true = set(review.true_aspect_ids)
            holdout_rows.append(lexical._eval_prf(pred, true))
        ps = [x[0] for x in holdout_rows]
        rs = [x[1] for x in holdout_rows]
        fs = [x[2] for x in holdout_rows]
        fold_rows.append(
            {
                "product_id": product_id,
                "selected_threshold": float(selected_threshold),
                "train_macro_precision": round(train_summary.macro_precision, 4),
                "train_macro_recall": round(train_summary.macro_recall, 4),
                "train_macro_f1": round(train_summary.macro_f1, 4),
                "holdout_macro_precision": round(sum(ps) / len(ps), 4) if ps else 0.0,
                "holdout_macro_recall": round(sum(rs) / len(rs), 4) if rs else 0.0,
                "holdout_macro_f1": round(sum(fs) / len(fs), 4) if fs else 0.0,
                "n_holdout_reviews": len(holdout_review_ids),
            }
        )

    return threshold_by_product, pred_by_review, pd.DataFrame(fold_rows)


def _apply_candidate_acceptance(
    scored_hits: pd.DataFrame,
    threshold_by_product: dict[int, float],
) -> pd.DataFrame:
    if scored_hits.empty:
        accepted = scored_hits.copy()
        accepted["accepted"] = []
        return accepted

    accepted = scored_hits.copy()
    accepted["accepted"] = accepted.apply(
        lambda row: bool(float(row["p_ent"]) > float(threshold_by_product[int(row["product_id"])])),
        axis=1,
    )
    return accepted


def _max_score_rows(scored_hits: pd.DataFrame) -> pd.DataFrame:
    if scored_hits.empty:
        return scored_hits.copy()
    idx = (
        scored_hits.groupby(["review_id", "aspect_id"])["p_ent"]
        .idxmax()
        .dropna()
        .astype(int)
        .tolist()
    )
    return scored_hits.loc[idx].copy().sort_values(["review_id", "aspect_id"])


def _removed_examples(
    baseline_map: dict[str, set[str]],
    verifier_map: dict[str, set[str]],
    review_records: list[ReviewRecord],
    max_score_rows: pd.DataFrame,
    want_true_positive: bool,
    limit: int = 10,
) -> pd.DataFrame:
    review_lookup = {review.review_id: review for review in review_records}
    row_lookup = {
        (str(row.review_id), str(row.aspect_id)): row
        for row in max_score_rows.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []

    for review in review_records:
        removed = set(baseline_map.get(review.review_id, set())) - set(verifier_map.get(review.review_id, set()))
        for aspect_id in sorted(removed):
            is_tp = aspect_id in review.true_aspect_ids
            if bool(is_tp) != bool(want_true_positive):
                continue
            row = row_lookup.get((review.review_id, aspect_id))
            if row is None:
                continue
            rows.append(
                {
                    "review_id": review.review_id,
                    "product_id": review.nm_id,
                    "category": review.category,
                    "aspect_id": aspect_id,
                    "candidate_span": str(row.candidate_span),
                    "p_ent": round(float(row.p_ent), 4),
                    "premise_text": str(row.premise_text),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["review_id", "product_id", "category", "aspect_id", "candidate_span", "p_ent", "premise_text"])
    out = pd.DataFrame(rows)
    return out.sort_values(["p_ent", "review_id", "aspect_id"]).head(limit)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_none_"
    view = df[columns].copy()
    return view.to_markdown(index=False)


def _write_summary(
    out_dir: Path,
    baseline_summary: EvalSummary,
    verifier_summary: EvalSummary | None,
    baseline_matches_reference: bool,
    baseline_delta_vs_ref: float,
    threshold_by_product: dict[int, float],
    fold_df: pd.DataFrame,
    nli_latency_sec: float,
    nli_pair_count: int,
    nli_unique_pair_count: int,
    removed_fp_df: pd.DataFrame,
    removed_tp_df: pd.DataFrame,
) -> None:
    if not baseline_matches_reference:
        status = "STOP"
    elif verifier_summary is None:
        status = "STOP"
    else:
        delta_f1 = verifier_summary.macro_f1 - baseline_summary.macro_f1
        delta_recall = verifier_summary.macro_recall - baseline_summary.macro_recall
        status = "PASS" if (delta_f1 >= 0.02 and delta_recall >= -0.03) else "FAIL"

    lines: list[str] = [
        f"# phase2_step6_lexical_only_nli_verifier",
        "",
        f"Status: **{status}**",
        "",
        "## Baseline reproduction",
        f"- baseline macro precision: {baseline_summary.macro_precision:.4f}",
        f"- baseline macro recall: {baseline_summary.macro_recall:.4f}",
        f"- baseline macro F1: {baseline_summary.macro_f1:.4f}",
        f"- reference macro F1: {REFERENCE_BASELINE['macro_f1']:.4f}",
        f"- delta vs reference F1: {baseline_delta_vs_ref:+.4f}",
        f"- reproduction_ok: {baseline_matches_reference}",
        "",
    ]

    if verifier_summary is None:
        lines.extend(
            [
                "## Verifier run",
                "- skipped because baseline was not reproduced closely enough.",
                "",
            ]
        )
    else:
        delta_p = verifier_summary.macro_precision - baseline_summary.macro_precision
        delta_r = verifier_summary.macro_recall - baseline_summary.macro_recall
        delta_f1 = verifier_summary.macro_f1 - baseline_summary.macro_f1
        threshold_values = [float(v) for v in threshold_by_product.values()]
        median_threshold = float(np.median(threshold_values)) if threshold_values else float("nan")
        lines.extend(
            [
                "## Head-to-head",
                f"- verifier macro precision: {verifier_summary.macro_precision:.4f}",
                f"- verifier macro recall: {verifier_summary.macro_recall:.4f}",
                f"- verifier macro F1: {verifier_summary.macro_f1:.4f}",
                f"- delta precision vs baseline: {delta_p:+.4f}",
                f"- delta recall vs baseline: {delta_r:+.4f}",
                f"- delta F1 vs baseline: {delta_f1:+.4f}",
                "",
                "## Threshold",
                f"- LOPO-selected threshold (median across folds): {median_threshold:.4f}",
                f"- thresholds by fold are saved in `lopo_thresholds.csv`",
                "",
                "## Runtime",
                f"- latency_sec: {nli_latency_sec:.2f}",
                f"- NLI calls (candidate-aspect rows): {nli_pair_count}",
                f"- unique premise+hypothesis pairs: {nli_unique_pair_count}",
                "",
                "## 10 examples: verifier removed obvious FP",
                _markdown_table(
                    removed_fp_df,
                    ["review_id", "product_id", "category", "aspect_id", "candidate_span", "p_ent", "premise_text"],
                ),
                "",
                "## 10 examples: verifier removed true positive",
                _markdown_table(
                    removed_tp_df,
                    ["review_id", "product_id", "category", "aspect_id", "candidate_span", "p_ent", "premise_text"],
                ),
                "",
            ]
        )
        if status != "PASS":
            lines.extend(
                [
                    "## Decision",
                    "- FAIL",
                    "- lexical_only remains final detection baseline.",
                    "- detection should not be tuned further in this branch.",
                    "",
                ]
            )

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if not fold_df.empty:
        fold_df.to_csv(out_dir / "lopo_thresholds.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: lexical_only vs lexical_only + NLI verifier")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase2_step6_lexical_nli_verifier")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews = lexical._load_reviews(ROOT / args.dataset_csv)
    domain_vocab_by_category = {
        "physical_goods": ROOT / "src/vocabulary/domain/physical_goods.yaml",
        "consumables": ROOT / "src/vocabulary/domain/consumables.yaml",
        "hospitality": ROOT / "src/vocabulary/domain/hospitality.yaml",
        "services": ROOT / "src/vocabulary/domain/services.yaml",
    }

    review_records, hit_df = _collect_lexical_hits(
        reviews=reviews,
        core_vocab_path=ROOT / args.core_vocab,
        domain_vocab_by_category=domain_vocab_by_category,
    )

    baseline_map = _baseline_prediction_map(review_records)
    baseline_summary, baseline_review_df, baseline_per_category_df, baseline_fp, baseline_fn = _evaluate_predictions(
        review_records=review_records,
        pred_by_review=baseline_map,
    )
    baseline_review_df.to_csv(out_dir / "review_predictions_baseline.csv", index=False, encoding="utf-8")
    baseline_per_category_df.to_csv(out_dir / "per_category_breakdown_baseline.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [{"aspect_id": key, "fp_count": int(value)} for key, value in baseline_fp.most_common()]
    ).to_csv(out_dir / "top_false_positives_baseline.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [{"aspect_id": key, "fn_count": int(value)} for key, value in baseline_fn.most_common()]
    ).to_csv(out_dir / "top_false_negatives_baseline.csv", index=False, encoding="utf-8")

    baseline_matches_reference = (
        round(baseline_summary.macro_precision, 4) == REFERENCE_BASELINE["macro_precision"]
        and round(baseline_summary.macro_recall, 4) == REFERENCE_BASELINE["macro_recall"]
        and round(baseline_summary.macro_f1, 4) == REFERENCE_BASELINE["macro_f1"]
    )
    baseline_delta_vs_ref = baseline_summary.macro_f1 - REFERENCE_BASELINE["macro_f1"]

    if not baseline_matches_reference:
        _write_summary(
            out_dir=out_dir,
            baseline_summary=baseline_summary,
            verifier_summary=None,
            baseline_matches_reference=False,
            baseline_delta_vs_ref=baseline_delta_vs_ref,
            threshold_by_product={},
            fold_df=pd.DataFrame(),
            nli_latency_sec=0.0,
            nli_pair_count=0,
            nli_unique_pair_count=0,
            removed_fp_df=pd.DataFrame(),
            removed_tp_df=pd.DataFrame(),
        )
        print(
            json.dumps(
                {
                    "status": "STOP",
                    "reason": "baseline_mismatch",
                    "out_dir": str(out_dir),
                    "baseline_macro_f1": round(baseline_summary.macro_f1, 4),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    scored_hits_df, nli_latency_sec, nli_pair_count, nli_unique_pair_count = _score_verifier_pairs(hit_df)
    review_aspect_scores = _review_aspect_max_scores(scored_hits_df)
    threshold_by_product, verifier_map, fold_df = _run_lopo_threshold_selection(
        review_records=review_records,
        review_aspect_scores=review_aspect_scores,
    )
    accepted_hits_df = _apply_candidate_acceptance(
        scored_hits=scored_hits_df,
        threshold_by_product=threshold_by_product,
    )
    accepted_hits_df[
        ["review_id", "product_id", "candidate_span", "aspect_id", "lexical_hit", "premise_text", "p_ent", "accepted"]
    ].to_csv(out_dir / "candidate_verifier_scores.csv", index=False, encoding="utf-8")

    verifier_summary, verifier_review_df, verifier_per_category_df, verifier_fp, verifier_fn = _evaluate_predictions(
        review_records=review_records,
        pred_by_review=verifier_map,
    )
    threshold_used = {review.review_id: float(threshold_by_product[review.nm_id]) for review in review_records}
    verifier_review_df["threshold_used"] = verifier_review_df["review_id"].map(threshold_used)
    verifier_review_df.to_csv(out_dir / "review_predictions_nli_verifier.csv", index=False, encoding="utf-8")
    verifier_per_category_df.to_csv(out_dir / "per_category_breakdown_nli_verifier.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [{"aspect_id": key, "fp_count": int(value)} for key, value in verifier_fp.most_common()]
    ).to_csv(out_dir / "top_false_positives_nli_verifier.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [{"aspect_id": key, "fn_count": int(value)} for key, value in verifier_fn.most_common()]
    ).to_csv(out_dir / "top_false_negatives_nli_verifier.csv", index=False, encoding="utf-8")

    max_rows_df = _max_score_rows(scored_hits_df)
    removed_fp_df = _removed_examples(
        baseline_map=baseline_map,
        verifier_map=verifier_map,
        review_records=review_records,
        max_score_rows=max_rows_df,
        want_true_positive=False,
        limit=10,
    )
    removed_tp_df = _removed_examples(
        baseline_map=baseline_map,
        verifier_map=verifier_map,
        review_records=review_records,
        max_score_rows=max_rows_df,
        want_true_positive=True,
        limit=10,
    )
    removed_fp_df.to_csv(out_dir / "examples_removed_obvious_fp.csv", index=False, encoding="utf-8")
    removed_tp_df.to_csv(out_dir / "examples_removed_true_positive.csv", index=False, encoding="utf-8")

    _write_summary(
        out_dir=out_dir,
        baseline_summary=baseline_summary,
        verifier_summary=verifier_summary,
        baseline_matches_reference=True,
        baseline_delta_vs_ref=baseline_delta_vs_ref,
        threshold_by_product=threshold_by_product,
        fold_df=fold_df,
        nli_latency_sec=nli_latency_sec,
        nli_pair_count=nli_pair_count,
        nli_unique_pair_count=nli_unique_pair_count,
        removed_fp_df=removed_fp_df,
        removed_tp_df=removed_tp_df,
    )

    payload = {
        "status": "PASS" if (verifier_summary.macro_f1 - baseline_summary.macro_f1 >= 0.02 and verifier_summary.macro_recall - baseline_summary.macro_recall >= -0.03) else "FAIL",
        "out_dir": str(out_dir),
        "baseline": {
            "macro_precision": round(baseline_summary.macro_precision, 4),
            "macro_recall": round(baseline_summary.macro_recall, 4),
            "macro_f1": round(baseline_summary.macro_f1, 4),
        },
        "verifier": {
            "macro_precision": round(verifier_summary.macro_precision, 4),
            "macro_recall": round(verifier_summary.macro_recall, 4),
            "macro_f1": round(verifier_summary.macro_f1, 4),
        },
        "delta": {
            "macro_precision": round(verifier_summary.macro_precision - baseline_summary.macro_precision, 4),
            "macro_recall": round(verifier_summary.macro_recall - baseline_summary.macro_recall, 4),
            "macro_f1": round(verifier_summary.macro_f1 - baseline_summary.macro_f1, 4),
        },
        "threshold_median": round(float(np.median(list(threshold_by_product.values()))), 4) if threshold_by_product else None,
        "nli_latency_sec": round(nli_latency_sec, 2),
        "nli_calls": int(nli_pair_count),
        "nli_unique_pairs": int(nli_unique_pair_count),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
