from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

ACCURACY_THRESHOLDS = (0.5, 1.0, 1.5)
NEGATIVE = "negative"
NEUTRAL = "neutral"
POSITIVE = "positive"
VECTOR_CLASS_NAMES = (NEGATIVE, NEUTRAL, POSITIVE)


def clip_rating(value: float) -> float:
    return float(min(5.0, max(1.0, float(value))))


def round_rating(value: float) -> int:
    return int(min(5, max(1, int(np.rint(float(value))))))


def gold_direction(value: float) -> str:
    rating = float(value)
    if rating <= 2.0:
        return NEGATIVE
    if rating >= 4.0:
        return POSITIVE
    return NEUTRAL


def predicted_direction(value: float) -> str:
    rating = float(value)
    if rating < 2.75:
        return NEGATIVE
    if rating <= 3.25:
        return NEUTRAL
    return POSITIVE


def wrong_polarity(gold_rating: float, pred_rating: float) -> bool:
    gold = float(gold_rating)
    pred = float(pred_rating)
    return (gold >= 4.0 and pred < 2.75) or (gold <= 2.0 and pred > 3.25)


def strong_wrong_polarity(gold_rating: float, pred_rating: float) -> bool:
    gold = float(gold_rating)
    pred = float(pred_rating)
    return (gold >= 4.0 and pred <= 2.0) or (gold <= 2.0 and pred >= 4.0)


def rating_to_fuzzy_vector(value: float) -> np.ndarray:
    rating = clip_rating(value)
    if rating <= 3.0:
        negative = (3.0 - rating) / 2.0
        neutral = (rating - 1.0) / 2.0
        positive = 0.0
    else:
        negative = 0.0
        neutral = (5.0 - rating) / 2.0
        positive = (rating - 3.0) / 2.0
    return np.array([negative, neutral, positive], dtype=float)


def dominant_classes(vector: np.ndarray) -> tuple[str, ...]:
    max_value = float(np.max(vector))
    return tuple(
        label
        for label, component in zip(VECTOR_CLASS_NAMES, vector, strict=True)
        if math.isclose(float(component), max_value, abs_tol=1e-9)
    )


def dominant_class_accuracy(gold_rating: float, pred_rating: float) -> bool:
    gold_set = set(dominant_classes(rating_to_fuzzy_vector(gold_rating)))
    pred_set = set(dominant_classes(rating_to_fuzzy_vector(pred_rating)))
    return bool(gold_set & pred_set)


def add_error_columns(df: pd.DataFrame, *, pred_col: str = "pred_rating", gold_col: str = "gold_rating") -> pd.DataFrame:
    out = df.copy()
    out["pred_rating_clipped"] = out[pred_col].astype(float).map(clip_rating)
    out["pred_round"] = out["pred_rating_clipped"].map(round_rating)
    out["abs_error"] = (out["pred_rating_clipped"] - out[gold_col].astype(float)).abs()
    out["signed_error"] = out["pred_rating_clipped"] - out[gold_col].astype(float)
    out["round_abs_error"] = (out["pred_round"] - out[gold_col].astype(float)).abs()
    out["gold_direction"] = out[gold_col].astype(float).map(gold_direction)
    out["pred_direction"] = out["pred_rating_clipped"].map(predicted_direction)
    out["wrong_polarity"] = [
        wrong_polarity(gold, pred)
        for gold, pred in zip(out[gold_col], out["pred_rating_clipped"], strict=True)
    ]
    out["strong_wrong_polarity"] = [
        strong_wrong_polarity(gold, pred)
        for gold, pred in zip(out[gold_col], out["pred_rating_clipped"], strict=True)
    ]
    out["dominant_class_match"] = [
        dominant_class_accuracy(gold, pred)
        for gold, pred in zip(out[gold_col], out["pred_rating_clipped"], strict=True)
    ]
    strong_gold_mask = out[gold_col].astype(float).map(lambda value: bool(value <= 2.0 or value >= 4.0))
    out["strong_gold_polarity"] = strong_gold_mask
    out["neutral_collapse"] = strong_gold_mask & (out["pred_direction"] == NEUTRAL)
    out["polarity_flip"] = strong_gold_mask & out["wrong_polarity"]
    return out


def compute_pair_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_pairs": 0,
            "n_reviews": 0,
            "n_products": 0,
            "mae": math.nan,
            "mae_round": math.nan,
            "rmse": math.nan,
            "mean_signed_error": math.nan,
            "accuracy_at_0_5": math.nan,
            "accuracy_at_1_0": math.nan,
            "accuracy_at_1_5": math.nan,
            "wrong_polarity_rate": math.nan,
            "strong_wrong_polarity_rate": math.nan,
            "neutral_collapse_rate": math.nan,
            "polarity_flip_rate": math.nan,
            "dominant_class_accuracy": math.nan,
        }

    strong_gold = df["strong_gold_polarity"].astype(bool)
    strong_count = int(strong_gold.sum())
    metrics: dict[str, Any] = {
        "n_pairs": int(len(df)),
        "n_reviews": int(df["review_id"].astype(str).nunique()) if "review_id" in df.columns else 0,
        "n_products": int(df["nm_id"].nunique()) if "nm_id" in df.columns else 0,
        "mae": float(df["abs_error"].mean()),
        "mae_round": float(df["round_abs_error"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(df["signed_error"])))),
        "mean_signed_error": float(df["signed_error"].mean()),
        "wrong_polarity_rate": float(df["wrong_polarity"].mean()),
        "strong_wrong_polarity_rate": float(df["strong_wrong_polarity"].mean()),
        "neutral_collapse_rate": float(df.loc[strong_gold, "neutral_collapse"].mean()) if strong_count else math.nan,
        "polarity_flip_rate": float(df.loc[strong_gold, "polarity_flip"].mean()) if strong_count else math.nan,
        "dominant_class_accuracy": float(df["dominant_class_match"].mean()),
    }
    for threshold in ACCURACY_THRESHOLDS:
        metrics[f"accuracy_at_{str(threshold).replace('.', '_')}"] = float((df["abs_error"] <= threshold).mean())
    return metrics


def compute_product_aggregate_details(
    df: pd.DataFrame,
    *,
    aspect_col: str = "mapped_gold_aspect",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "nm_id",
                aspect_col,
                "n_pairs",
                "pred_product_rating",
                "gold_product_rating",
                "product_abs_error",
            ]
        )
    grouped = (
        df.groupby(["nm_id", aspect_col], dropna=False, sort=True)
        .agg(
            n_pairs=("review_id", "size"),
            pred_product_rating=("pred_rating_clipped", "mean"),
            gold_product_rating=("gold_rating", "mean"),
        )
        .reset_index()
    )
    grouped["product_abs_error"] = (grouped["pred_product_rating"] - grouped["gold_product_rating"]).abs()
    return grouped


def compute_product_aggregate_metrics(df: pd.DataFrame, *, aspect_col: str = "mapped_gold_aspect") -> dict[str, Any]:
    details = compute_product_aggregate_details(df, aspect_col=aspect_col)
    if details.empty:
        return {
            "product_mae_simple_mean": math.nan,
            "product_mae_n3_simple_mean": math.nan,
            "product_groups": 0,
            "product_groups_n3": 0,
        }
    n3 = details[details["n_pairs"] >= 3].copy()
    return {
        "product_mae_simple_mean": float(details["product_abs_error"].mean()),
        "product_mae_n3_simple_mean": float(n3["product_abs_error"].mean()) if not n3.empty else math.nan,
        "product_groups": int(len(details)),
        "product_groups_n3": int(len(n3)),
    }


def slice_metric_rows(
    df: pd.DataFrame,
    *,
    slice_type: str,
    group_col: str,
    formula_name: str | None = None,
    model_id: str | None = None,
    feature_set: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for value, group in df.groupby(group_col, dropna=False, sort=True):
        payload: dict[str, Any] = {
            "slice_type": slice_type,
            "slice_value": str(value),
        }
        if formula_name is not None:
            payload["formula_name"] = formula_name
        if model_id is not None:
            payload["model_id"] = model_id
        if feature_set is not None:
            payload["feature_set"] = feature_set
        payload.update(compute_pair_metrics(group))
        rows.append(payload)
    return pd.DataFrame(rows)


__all__ = [
    "add_error_columns",
    "clip_rating",
    "compute_pair_metrics",
    "compute_product_aggregate_details",
    "compute_product_aggregate_metrics",
    "slice_metric_rows",
]
