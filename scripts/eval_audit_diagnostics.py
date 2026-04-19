"""
Funnel labels and confusion-matrix helpers for eval audit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


def compute_funnel_rows(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> List[Dict[str, Any]]:
    """One row per labeled review with funnel category."""
    rows: List[Dict[str, Any]] = []

    def mapped_set(pred_scores: Dict[str, float], pm: Dict[str, Optional[str]]) -> Set[str]:
        s: Set[str] = set()
        for pa in pred_scores:
            ta = pm.get(pa)
            if ta is not None:
                s.add(ta)
        return s

    for nm_id, pred_data in pipeline_results.items():
        pm = mapping.get(nm_id, {}) or {}
        has_aspects = bool(pred_data.get("aspects"))
        per_review_pred = pred_data.get("per_review") or {}
        grp = markup_df[markup_df["nm_id"] == nm_id]

        for _, row in grp.iterrows():
            tl = row["true_labels_parsed"]
            if not tl:
                continue
            rid = str(row["id"])
            A_true = set(tl.keys())
            pred_scores = per_review_pred.get(rid, {})

            if not has_aspects:
                cat = "NO_CANDIDATES"
            elif not pred_scores:
                cat = "CANDIDATES_BUT_NOT_MAPPED"
            else:
                A_pred = mapped_set(pred_scores, pm)
                inter = A_true & A_pred
                if not A_pred:
                    cat = "CANDIDATES_BUT_NOT_MAPPED"
                elif not inter:
                    cat = "MAPPED_TO_WRONG_ASPECT"
                else:
                    cat = "CORRECTLY_MAPPED_AND_SCORED"

            rows.append(
                {
                    "nm_id": int(nm_id),
                    "review_id": rid,
                    "funnel_category": cat,
                }
            )

    return rows


def aggregate_funnel_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(r["funnel_category"] for r in rows)
    return dict(c)


def confusion_pred_to_gold(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> Tuple[pd.DataFrame, List[Tuple[str, str, int]]]:
    """
    For each predicted aspect name pa with mapping(pa)=m, count co-occurrence with each gold aspect g in same review.
    Matrix rows = pred aspect name, cols = gold aspect (from labels).
    """
    pair_counts: Counter[Tuple[str, str]] = Counter()

    for nm_id, pred_data in pipeline_results.items():
        pm = mapping.get(nm_id, {}) or {}
        per_review = pred_data.get("per_review") or {}
        grp = markup_df[markup_df["nm_id"] == nm_id]

        for _, row in grp.iterrows():
            tl = row["true_labels_parsed"]
            if not tl:
                continue
            rid = str(row["id"])
            pred_scores = per_review.get(rid, {})
            gold_set = set(tl.keys())
            for pa in pred_scores:
                m = pm.get(pa)
                if m is None:
                    continue
                for g in gold_set:
                    pair_counts[(pa, g)] += 1

    if not pair_counts:
        return pd.DataFrame(), []

    preds = sorted({p for p, _ in pair_counts.keys()})
    golds = sorted({g for _, g in pair_counts.keys()})
    mat = pd.DataFrame(0, index=preds, columns=golds, dtype=int)
    for (pa, g), k in pair_counts.items():
        mat.loc[pa, g] = k

    top: List[Tuple[str, str, int]] = []
    for (pa, g), k in pair_counts.most_common(50):
        if pa != g:
            top.append((pa, g, k))
    top.sort(key=lambda x: -x[2])
    return mat, top[:20]
