"""
Review-level sentiment metrics on intersection(A_true, A_pred) only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def round_clip_rating(x: float) -> float:
    return float(max(1.0, min(5.0, round(float(x)))))


@dataclass
class SentimentEvalResult:
    mae_continuous_review_macro: float
    rmse_continuous_review_macro: float
    mae_rounded_review_macro: float
    mae_continuous_micro: float
    n_matched_pairs: int
    n_reviews_with_pairs: int
    per_review_mae_cont: List[float]


def _reverse_map(
    product_mapping: Dict[str, Optional[str]],
) -> Dict[str, List[str]]:
    rm: Dict[str, List[str]] = defaultdict(list)
    for pa, ta in product_mapping.items():
        if ta is not None:
            rm[ta].append(pa)
    return dict(rm)


def _mean_pred_for_true(
    true_asp: str,
    pred_scores: Dict[str, float],
    reverse_map: Dict[str, List[str]],
) -> Optional[float]:
    pred_asps = reverse_map.get(true_asp, [])
    found = [pred_scores[pa] for pa in pred_asps if pa in pred_scores]
    if not found:
        return None
    return float(np.mean(found))


def compute_review_level_sentiment(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> SentimentEvalResult:
    per_review_mae: List[float] = []
    per_review_mae_round: List[float] = []
    all_abs_err: List[float] = []
    all_abs_err_round: List[float] = []

    for nm_id, pred_data in pipeline_results.items():
        product_mapping = mapping.get(nm_id, {}) or {}
        reverse = _reverse_map(product_mapping)
        per_review_pred = pred_data.get("per_review") or {}
        grp = markup_df[markup_df["nm_id"] == nm_id]

        for _, row in grp.iterrows():
            true_labels = row["true_labels_parsed"]
            if not true_labels:
                continue
            rid = str(row["id"])
            pred_scores = per_review_pred.get(rid, {})
            if not pred_scores:
                continue

            errs: List[float] = []
            errs_r: List[float] = []
            for true_asp, true_score in true_labels.items():
                pred_mean = _mean_pred_for_true(true_asp, pred_scores, reverse)
                if pred_mean is None:
                    continue
                te = float(true_score)
                errs.append(abs(pred_mean - te))
                errs_r.append(abs(round_clip_rating(pred_mean) - te))
                all_abs_err.append(abs(pred_mean - te))
                all_abs_err_round.append(abs(round_clip_rating(pred_mean) - te))

            if errs:
                per_review_mae.append(float(np.mean(errs)))
                per_review_mae_round.append(float(np.mean(errs_r)))

    n_pairs = len(all_abs_err)
    n_rev = len(per_review_mae)
    mae_rmse = (
        float(np.sqrt(np.mean([e**2 for e in all_abs_err]))) if all_abs_err else 0.0
    )

    return SentimentEvalResult(
        mae_continuous_review_macro=float(np.mean(per_review_mae)) if per_review_mae else 0.0,
        rmse_continuous_review_macro=mae_rmse,
        mae_rounded_review_macro=float(np.mean(per_review_mae_round)) if per_review_mae_round else 0.0,
        mae_continuous_micro=float(np.mean(all_abs_err)) if all_abs_err else 0.0,
        n_matched_pairs=n_pairs,
        n_reviews_with_pairs=n_rev,
        per_review_mae_cont=per_review_mae,
    )


def build_synthetic_per_review_baseline(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
    mode: str,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    mode: star | neutral | product_mean
    Copies structure of pipeline per_review; replaces scores.
    """
    from collections import defaultdict

    ratings_by_nm: Dict[int, List[int]] = defaultdict(list)
    for _, row in markup_df.iterrows():
        ratings_by_nm[int(row["nm_id"])].append(int(row["rating"]))

    product_mean: Dict[int, float] = {
        k: float(np.mean(v)) for k, v in ratings_by_nm.items()
    }

    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for nm_id, pred_data in pipeline_results.items():
        nm = int(nm_id)
        per_in = pred_data.get("per_review") or {}
        out[nm] = {}
        pm = product_mean.get(nm, 3.0)
        for rid, scores in per_in.items():
            if rid == "unknown":
                continue
            row_match = markup_df[
                (markup_df["nm_id"] == nm) & (markup_df["id"].astype(str) == str(rid))
            ]
            star = int(row_match.iloc[0]["rating"]) if len(row_match) else 3

            out[nm][rid] = {}
            for pa in scores:
                if mode == "star":
                    out[nm][rid][pa] = float(star)
                elif mode == "neutral":
                    out[nm][rid][pa] = 3.0
                else:
                    out[nm][rid][pa] = pm
    return out


def compute_sentiment_for_per_review_override(
    markup_df: pd.DataFrame,
    synthetic_per_review: Dict[int, Dict[str, Dict[str, float]]],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> SentimentEvalResult:
    wrapped: Dict[int, Dict[str, Any]] = {}
    for nm_id, pr in synthetic_per_review.items():
        wrapped[nm_id] = {"per_review": pr, "aspects": ["_"]}
    return compute_review_level_sentiment(markup_df, wrapped, mapping)
