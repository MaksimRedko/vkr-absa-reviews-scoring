"""
Product-level matched MAE (per aspect, aggregated).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eval_audit_metrics_sentiment import round_clip_rating


@dataclass
class ProductAspectRow:
    nm_id: int
    aspect: str
    error_matched: float
    error_matched_rounded: float
    true_avg_matched: float
    pred_avg: float
    pred_avg_rounded: float
    n_true: int
    n_true_matched: int
    n_pred: int
    true_avg_all: float


@dataclass
class ProductEvalResult:
    per_aspect_rows: List[ProductAspectRow]
    product_mae_matched_all: Optional[float]
    product_mae_matched_n_true_ge_3: Optional[float]
    product_mae_rounded_matched_all: Optional[float]
    product_mae_rounded_matched_n_true_ge_3: Optional[float]
    per_product: Dict[int, Dict[str, Any]]


def _build_rows(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> List[ProductAspectRow]:
    rows: List[ProductAspectRow] = []

    for nm_id, pred_data in pipeline_results.items():
        product_mapping = mapping.get(nm_id, {}) or {}
        per_review_pred = pred_data.get("per_review") or {}

        reverse_map: Dict[str, List[str]] = defaultdict(list)
        for pa, ta in product_mapping.items():
            if ta is not None:
                reverse_map[ta].append(pa)

        grp = markup_df[markup_df["nm_id"] == nm_id]
        true_scores_by_aspect: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for _, row in grp.iterrows():
            labels = row["true_labels_parsed"]
            if not labels:
                continue
            rid = str(row["id"])
            for asp, score in labels.items():
                true_scores_by_aspect[asp].append((rid, float(score)))

        pred_scores_by_true_raw: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for rid, pred_scores in per_review_pred.items():
            if rid == "unknown":
                continue
            for true_asp, pred_asps in reverse_map.items():
                found = [pred_scores[pa] for pa in pred_asps if pa in pred_scores]
                if not found:
                    continue
                raw_m = float(np.mean(found))
                pred_scores_by_true_raw[true_asp].append((rid, raw_m))

        for true_asp in sorted(true_scores_by_aspect.keys()):
            tlist = true_scores_by_aspect[true_asp]
            true_scores_all = [s for _, s in tlist]
            true_avg_all = float(np.mean(true_scores_all))
            n_true = len(true_scores_all)

            if true_asp not in pred_scores_by_true_raw or not pred_scores_by_true_raw[true_asp]:
                continue

            pred_pairs = pred_scores_by_true_raw[true_asp]
            pred_rids = {rid for rid, _ in pred_pairs}
            matched_true_scores = [sc for rid, sc in tlist if rid in pred_rids]
            if not matched_true_scores:
                continue

            true_avg_matched = float(np.mean(matched_true_scores))
            raw_preds = [s for _, s in pred_pairs]
            pred_avg = float(np.mean(raw_preds))
            pred_avg_r = float(np.mean([round_clip_rating(x) for x in raw_preds]))
            err = abs(true_avg_matched - pred_avg)
            err_r = abs(true_avg_matched - pred_avg_r)

            rows.append(
                ProductAspectRow(
                    nm_id=int(nm_id),
                    aspect=true_asp,
                    error_matched=err,
                    error_matched_rounded=err_r,
                    true_avg_matched=true_avg_matched,
                    pred_avg=pred_avg,
                    pred_avg_rounded=pred_avg_r,
                    n_true=n_true,
                    n_true_matched=len(matched_true_scores),
                    n_pred=len(pred_pairs),
                    true_avg_all=true_avg_all,
                )
            )

    return rows


def compute_product_level_metrics(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> ProductEvalResult:
    rows = _build_rows(markup_df, pipeline_results, mapping)
    err_all = [r.error_matched for r in rows]
    err_f3 = [r.error_matched for r in rows if r.n_true >= 3]
    err_all_r = [r.error_matched_rounded for r in rows]
    err_f3_r = [r.error_matched_rounded for r in rows if r.n_true >= 3]

    m_all = float(np.mean(err_all)) if err_all else None
    m_ge3 = float(np.mean(err_f3)) if err_f3 else None
    mr_all = float(np.mean(err_all_r)) if err_all_r else None
    mr_ge3 = float(np.mean(err_f3_r)) if err_f3_r else None

    per_nm: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"aspect_errors": []})
    for r in rows:
        per_nm[r.nm_id]["aspect_errors"].append(r)

    return ProductEvalResult(
        per_aspect_rows=rows,
        product_mae_matched_all=m_all,
        product_mae_matched_n_true_ge_3=m_ge3,
        product_mae_rounded_matched_all=mr_all,
        product_mae_rounded_matched_n_true_ge_3=mr_ge3,
        per_product=dict(per_nm),
    )


def product_rows_to_df(rows: List[ProductAspectRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "nm_id",
                "aspect",
                "error_matched",
                "error_matched_rounded",
                "true_avg_matched",
                "pred_avg",
                "pred_avg_rounded",
                "n_true",
                "n_true_matched",
                "n_pred",
                "true_avg_all",
            ]
        )
    return pd.DataFrame([r.__dict__ for r in rows])
