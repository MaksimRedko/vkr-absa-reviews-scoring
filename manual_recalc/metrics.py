from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


STRICT_ERROR_DECISIONS = {"FP", "DUPLICATE", "UNCLEAR", "OUT_OF_SCOPE"}
SOFT_ERROR_DECISIONS = {"FP", "DUPLICATE", "OUT_OF_SCOPE"}
STRICT_ERROR_GOLD = {"FN", "UNCLEAR"}
SOFT_ERROR_GOLD = {"FN"}


@dataclass(frozen=True)
class MetricsSummary:
    mode: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mae: float | None
    matched_pairs: int


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_detection_and_mae(
    system_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    review_gold_lookup: dict[tuple[str, str], float],
) -> list[MetricsSummary]:
    return [
        _compute_mode("strict", system_df, gold_df, review_gold_lookup, STRICT_ERROR_DECISIONS, STRICT_ERROR_GOLD),
        _compute_mode("soft", system_df, gold_df, review_gold_lookup, SOFT_ERROR_DECISIONS, SOFT_ERROR_GOLD),
    ]


def _compute_mode(
    mode: str,
    system_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    review_gold_lookup: dict[tuple[str, str], float],
    system_error_decisions: set[str],
    gold_error_statuses: set[str],
) -> MetricsSummary:
    system = system_df.copy()
    gold = gold_df.copy()

    tp_mask = system["manual_decision"] == "TP"
    fp_mask = system["manual_decision"].isin(system_error_decisions)
    fn_mask = gold["status"].isin(gold_error_statuses)

    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0

    mae_values: list[float] = []
    for _, row in system[tp_mask].iterrows():
        mapped = str(row.get("mapped_gold_aspect", "")).strip()
        if not mapped or mapped.upper() == "NONE":
            continue
        gold_rating = review_gold_lookup.get((str(row["review_id"]), mapped))
        if gold_rating is None:
            continue
        mae_values.append(abs(float(row["system_rating"]) - float(gold_rating)))

    mae = float(sum(mae_values) / len(mae_values)) if mae_values else None
    return MetricsSummary(
        mode=mode,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mae=mae,
        matched_pairs=len(mae_values),
    )


def metrics_to_frame(metrics: Iterable[MetricsSummary]) -> pd.DataFrame:
    rows = []
    for item in metrics:
        rows.append(
            {
                "mode": item.mode,
                "tp": item.tp,
                "fp": item.fp,
                "fn": item.fn,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
                "mae": item.mae,
                "matched_pairs": item.matched_pairs,
            }
        )
    return pd.DataFrame(rows)
