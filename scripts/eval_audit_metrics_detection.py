"""
Review-level aspect detection metrics (pure functions).

A_true(r): gold aspect keys; A_pred(r): predicted aspects mapped to true-label space
via product mapping (non-None values only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class ReviewDetectionRow:
    nm_id: int
    review_id: str
    precision_r: float
    recall_r: float
    f1_r: float
    n_true: int
    n_pred_mapped: int
    n_intersection: int


@dataclass
class DetectionEvalResult:
    per_review: List[ReviewDetectionRow]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    reviews_with_no_candidates: int
    reviews_with_candidates_but_no_pred_aspects: int
    reviews_with_wrong_aspect_only: int
    reviews_with_at_least_one_correct_aspect: int


def _f1(p: float, r: float) -> float:
    if p + r <= 0:
        return 0.0
    return 2 * p * r / (p + r)


def _pred_aspects_mapped_for_review(
    pred_scores: Dict[str, float],
    product_mapping: Dict[str, Optional[str]],
) -> Set[str]:
    out: Set[str] = set()
    for pa in pred_scores:
        ta = product_mapping.get(pa)
        if ta is not None:
            out.add(ta)
    return out


def compute_review_level_detection(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> DetectionEvalResult:
    """Macro/micro P/R/F1 on review sets in true-label space."""
    per_review_rows: List[ReviewDetectionRow] = []
    tp_micro = fp_micro = fn_micro = 0

    cov_no_cand = 0
    cov_cand_no_pred = 0
    cov_wrong_only = 0
    cov_at_least_one = 0

    for nm_id, pred_data in pipeline_results.items():
        product_mapping = mapping.get(nm_id, {}) or {}
        grp = markup_df[markup_df["nm_id"] == nm_id]
        per_review_pred: Dict[str, Dict[str, float]] = pred_data.get("per_review") or {}
        product_has_aspects = bool(pred_data.get("aspects"))

        for _, row in grp.iterrows():
            true_labels = row["true_labels_parsed"]
            if not true_labels:
                continue
            rid = str(row["id"])
            A_true: Set[str] = set(true_labels.keys())
            pred_scores = per_review_pred.get(rid, {})
            if isinstance(pred_scores, dict) and rid == "unknown":
                continue

            A_pred = _pred_aspects_mapped_for_review(pred_scores, product_mapping)
            inter = A_true & A_pred

            prec = len(inter) / len(A_pred) if A_pred else (1.0 if not A_true else 0.0)
            rec = len(inter) / len(A_true) if A_true else 1.0
            f1 = _f1(prec, rec)

            per_review_rows.append(
                ReviewDetectionRow(
                    nm_id=int(nm_id),
                    review_id=rid,
                    precision_r=float(prec),
                    recall_r=float(rec),
                    f1_r=float(f1),
                    n_true=len(A_true),
                    n_pred_mapped=len(A_pred),
                    n_intersection=len(inter),
                )
            )

            tp_micro += len(inter)
            fp_micro += len(A_pred - A_true)
            fn_micro += len(A_true - A_pred)

            if not product_has_aspects:
                cov_no_cand += 1
            elif not pred_scores:
                cov_cand_no_pred += 1
            elif not inter and A_pred:
                cov_wrong_only += 1
            elif inter:
                cov_at_least_one += 1
            elif not A_pred and A_true:
                cov_cand_no_pred += 1

    if not per_review_rows:
        return DetectionEvalResult(
            per_review=[],
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            micro_precision=0.0,
            micro_recall=0.0,
            micro_f1=0.0,
            reviews_with_no_candidates=cov_no_cand,
            reviews_with_candidates_but_no_pred_aspects=cov_cand_no_pred,
            reviews_with_wrong_aspect_only=cov_wrong_only,
            reviews_with_at_least_one_correct_aspect=cov_at_least_one,
        )

    mp = float(np.mean([r.precision_r for r in per_review_rows]))
    mr = float(np.mean([r.recall_r for r in per_review_rows]))
    mf = float(np.mean([r.f1_r for r in per_review_rows]))

    p_mi = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) else 0.0
    r_mi = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) else 0.0
    f_mi = _f1(p_mi, r_mi)

    return DetectionEvalResult(
        per_review=per_review_rows,
        macro_precision=mp,
        macro_recall=mr,
        macro_f1=mf,
        micro_precision=p_mi,
        micro_recall=r_mi,
        micro_f1=f_mi,
        reviews_with_no_candidates=cov_no_cand,
        reviews_with_candidates_but_no_pred_aspects=cov_cand_no_pred,
        reviews_with_wrong_aspect_only=cov_wrong_only,
        reviews_with_at_least_one_correct_aspect=cov_at_least_one,
    )


def detection_per_review_to_df(rows: List[ReviewDetectionRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "nm_id",
                "review_id",
                "precision_r",
                "recall_r",
                "f1_r",
                "n_true",
                "n_pred_mapped",
                "n_intersection",
            ]
        )
    return pd.DataFrame([r.__dict__ for r in rows])
