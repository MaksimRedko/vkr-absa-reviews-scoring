from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_recalc.data_access import parse_true_labels

DEFAULT_DB_PATH = ROOT / "manual_recalc" / "data" / "manual_recalc.sqlite3"
DEFAULT_RUN_DIR = ROOT / "results" / "20260502_171530_traced"
DEFAULT_DATASET_PATH = ROOT / "data" / "dataset_final.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "manual_recalc" / "exports"

SYSTEM_DECISION_VALUES = {"TP", "FP", "UNCLEAR", "DUPLICATE", "OUT_OF_SCOPE"}
GOLD_STATUS_VALUES = {"FOUND", "FN", "UNCLEAR"}
REVIEW_STATUS_VALUES = {"not_started", "in_progress", "done", "needs_review"}
MANUAL_SENTIMENT_VALUES = {"", "OK", "WRONG_POLARITY", "TOO_HIGH", "TOO_LOW", "NOT_EVALUATED"}
NULL_LIKE_VALUES = {"", "NONE", "NULL", "NAN", "NA"}
DIRECTION_NEGATIVE = "negative"
DIRECTION_NEUTRAL = "neutral"
DIRECTION_POSITIVE = "positive"
VECTOR_CLASS_NAMES = (DIRECTION_NEGATIVE, DIRECTION_NEUTRAL, DIRECTION_POSITIVE)
OUT_FILE_NAMES = {
    "validation": "manual_audit_validation_report.md",
    "detection": "manual_detection_metrics.csv",
    "sentiment": "manual_sentiment_metrics.csv",
    "category": "manual_metrics_by_category.csv",
    "product": "manual_metrics_by_product.csv",
    "aspect": "manual_metrics_by_aspect.csv",
    "errors": "manual_error_summary.csv",
    "hard_cases": "manual_hard_cases.csv",
    "summary": "manual_final_summary.md",
    "vector_sentiment": "vector_sentiment_metrics.csv",
    "vector_category": "vector_sentiment_by_category.csv",
    "vector_product": "vector_sentiment_by_product.csv",
    "vector_summary": "vector_sentiment_summary.md",
}


@dataclass(slots=True)
class ValidationCheck:
    rule_id: int
    name: str
    passed: bool
    violation_count: int
    details: str


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.upper() in NULL_LIKE_VALUES:
        return ""
    return text


def _pred_round(value: float) -> int:
    return max(1, min(5, int(np.rint(float(value)))))


def gold_direction(gold_rating: float) -> str:
    value = float(gold_rating)
    if value <= 2.0:
        return DIRECTION_NEGATIVE
    if value >= 4.0:
        return DIRECTION_POSITIVE
    return DIRECTION_NEUTRAL


def predicted_direction(pred_rating: float) -> str:
    value = float(pred_rating)
    if value < 2.75:
        return DIRECTION_NEGATIVE
    if value <= 3.25:
        return DIRECTION_NEUTRAL
    return DIRECTION_POSITIVE


def wrong_polarity(gold_rating: float, pred_rating: float) -> bool:
    gold = float(gold_rating)
    pred = float(pred_rating)
    return (gold >= 4.0 and pred < 2.75) or (gold <= 2.0 and pred > 3.25)


def strong_wrong_polarity(gold_rating: float, pred_rating: float) -> bool:
    gold = float(gold_rating)
    pred = float(pred_rating)
    return (gold >= 4.0 and pred <= 2.0) or (gold <= 2.0 and pred >= 4.0)


def rating_to_fuzzy_sentiment_vector(value: float) -> np.ndarray:
    rating = float(np.clip(float(value), 1.0, 5.0))
    if rating <= 3.0:
        negative = (3.0 - rating) / 2.0
        neutral = (rating - 1.0) / 2.0
        positive = 0.0
    else:
        negative = 0.0
        neutral = (5.0 - rating) / 2.0
        positive = (rating - 3.0) / 2.0
    return np.array([negative, neutral, positive], dtype=float)


def _vector_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _dominant_classes(vector: np.ndarray) -> tuple[str, ...]:
    max_value = float(np.max(vector))
    return tuple(
        name
        for name, component in zip(VECTOR_CLASS_NAMES, vector, strict=True)
        if math.isclose(float(component), max_value, abs_tol=1e-9)
    )


def _dominant_classes_label(vector: np.ndarray) -> str:
    return "|".join(_dominant_classes(vector))


def _strict_polar_class(vector: np.ndarray) -> str:
    dominant = _dominant_classes(vector)
    if dominant == (DIRECTION_NEGATIVE,):
        return DIRECTION_NEGATIVE
    if dominant == (DIRECTION_POSITIVE,):
        return DIRECTION_POSITIVE
    return ""


def _label_f1(df: pd.DataFrame, label: str) -> float:
    tp = int(((df["gold_direction"] == label) & (df["pred_direction"] == label)).sum())
    fp = int(((df["gold_direction"] != label) & (df["pred_direction"] == label)).sum())
    fn = int(((df["gold_direction"] == label) & (df["pred_direction"] != label)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def classify_sentiment_error(row: pd.Series) -> str:
    if bool(row["strong_wrong_polarity"]):
        return "strong_wrong_polarity"
    if bool(row["wrong_polarity"]):
        return "wrong_polarity"
    if bool(row["large_too_low"]):
        return "large_too_low"
    if bool(row["large_too_high"]):
        return "large_too_high"
    if bool(row["too_low"]):
        return "too_low"
    if bool(row["too_high"]):
        return "too_high"
    return "near_miss"


def _load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(dataset_path, dtype={"id": str}).copy()
    dataset["review_id"] = dataset["id"].astype(str)
    dataset["nm_id"] = dataset["nm_id"].astype(int)
    dataset["category_id"] = dataset["category"].astype(str)
    dataset["review_rating"] = dataset["rating"].astype(float)
    dataset["gold_dict"] = dataset["true_labels"].apply(parse_true_labels)

    gold_rows: list[dict[str, Any]] = []
    for row in dataset.itertuples(index=False):
        for gold_aspect, gold_rating in row.gold_dict.items():
            gold_rows.append(
                {
                    "review_id": str(row.review_id),
                    "nm_id": int(row.nm_id),
                    "category_id": str(row.category_id),
                    "full_text": str(row.full_text),
                    "review_rating": float(row.review_rating),
                    "gold_aspect": str(gold_aspect),
                    "gold_rating": float(gold_rating),
                }
            )
    gold_df = pd.DataFrame(gold_rows)
    return dataset, gold_df


def _load_expected_system(run_dir: Path, dataset_reviews: pd.DataFrame) -> pd.DataFrame:
    nli = pd.read_parquet(run_dir / "nli_predictions.parquet").copy()
    nli["review_id"] = nli["review_id"].astype(str)
    merged = nli.merge(
        dataset_reviews[["review_id", "nm_id", "category_id", "review_rating", "full_text"]],
        on=["review_id", "nm_id"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.rename(
        columns={
            "aspect_name": "system_aspect",
            "final_rating": "system_rating",
        }
    )
    return merged[
        [
            "prediction_id",
            "review_id",
            "nm_id",
            "category_id",
            "review_rating",
            "full_text",
            "aspect_source",
            "system_aspect",
            "system_rating",
            "raw_rating",
            "passed_relevance_filter",
            "relevance_filter_value",
            "premise_text",
            "hypothesis_text",
            "p_entailment",
            "p_neutral",
            "p_contradiction",
            "has_negation_match",
            "negation_correction_applied",
        ]
    ].copy()


def _load_manual_tables(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        review_status = pd.read_sql_query("SELECT * FROM review_status", conn)
        system = pd.read_sql_query("SELECT * FROM system_decisions", conn)
        gold = pd.read_sql_query("SELECT * FROM gold_decisions", conn)
    finally:
        conn.close()
    for df in (review_status, system, gold):
        if "review_id" in df.columns:
            df["review_id"] = df["review_id"].astype(str)
    return review_status, system, gold


def build_audit_base(
    *,
    db_path: Path,
    run_dir: Path,
    dataset_path: Path,
) -> dict[str, pd.DataFrame]:
    review_status, manual_system, manual_gold = _load_manual_tables(db_path)
    dataset_reviews, dataset_gold = _load_dataset(dataset_path)
    expected_system = _load_expected_system(run_dir, dataset_reviews)

    reviewed_ids = sorted(review_status["review_id"].astype(str).unique())
    dataset_reviews = dataset_reviews[dataset_reviews["review_id"].isin(reviewed_ids)].copy()
    dataset_gold = dataset_gold[dataset_gold["review_id"].isin(reviewed_ids)].copy()
    expected_system = expected_system[expected_system["review_id"].isin(reviewed_ids)].copy()

    manual_system["prediction_id"] = manual_system["prediction_id"].astype(str)
    manual_system["manual_decision_norm"] = manual_system["manual_decision"].fillna("").astype(str).str.strip()
    manual_system["mapped_gold_aspect_norm"] = manual_system["mapped_gold_aspect"].map(_normalize_text)
    manual_system["manual_sentiment_decision_norm"] = manual_system["manual_sentiment_decision"].fillna("").astype(str).str.strip()
    system = manual_system.merge(
        expected_system,
        on=["prediction_id", "review_id"],
        how="left",
        suffixes=("_manual", ""),
        validate="one_to_one",
    )

    manual_gold["gold_aspect"] = manual_gold["gold_aspect"].astype(str)
    manual_gold["status_norm"] = manual_gold["status"].fillna("").astype(str).str.strip()
    manual_gold["matched_system_prediction_id_norm"] = manual_gold["matched_system_prediction_id"].map(_normalize_text)
    gold = manual_gold.merge(
        dataset_gold,
        on=["review_id", "gold_aspect"],
        how="left",
        suffixes=("_manual", ""),
        validate="one_to_one",
    )

    return {
        "review_status": review_status,
        "dataset_reviews": dataset_reviews,
        "dataset_gold": dataset_gold,
        "expected_system": expected_system,
        "system": system,
        "gold": gold,
    }


def build_validation_checks(base: dict[str, pd.DataFrame]) -> tuple[list[ValidationCheck], dict[str, pd.DataFrame]]:
    review_status = base["review_status"]
    dataset_reviews = base["dataset_reviews"]
    dataset_gold = base["dataset_gold"]
    expected_system = base["expected_system"]
    system = base["system"]
    gold = base["gold"]
    details: dict[str, pd.DataFrame] = {}

    expected_sys_keys = expected_system[["review_id", "prediction_id"]].copy()
    actual_sys_keys = system[["review_id", "prediction_id"]].copy()
    missing_system = expected_sys_keys.merge(actual_sys_keys, on=["review_id", "prediction_id"], how="left", indicator=True)
    missing_system = missing_system[missing_system["_merge"] == "left_only"].drop(columns="_merge")
    details["missing_system_decisions"] = missing_system.copy()

    expected_gold_keys = dataset_gold[["review_id", "gold_aspect"]].copy()
    actual_gold_keys = gold[["review_id", "gold_aspect"]].copy()
    missing_gold = expected_gold_keys.merge(actual_gold_keys, on=["review_id", "gold_aspect"], how="left", indicator=True)
    missing_gold = missing_gold[missing_gold["_merge"] == "left_only"].drop(columns="_merge")
    details["missing_gold_decisions"] = missing_gold.copy()

    tp_missing_map = system[
        (system["manual_decision_norm"] == "TP")
        & (system["mapped_gold_aspect_norm"] == "")
    ][["review_id", "prediction_id", "system_aspect"]].copy()
    details["tp_missing_map"] = tp_missing_map

    fp_with_map = system[
        (system["manual_decision_norm"] == "FP")
        & (system["mapped_gold_aspect_norm"] != "")
    ][["review_id", "prediction_id", "system_aspect", "mapped_gold_aspect"]].copy()
    details["fp_with_map"] = fp_with_map

    found_missing_match = gold[
        (gold["status_norm"] == "FOUND")
        & (gold["matched_system_prediction_id_norm"] == "")
    ][["review_id", "gold_aspect"]].copy()
    details["found_missing_match"] = found_missing_match

    mapped = system[system["mapped_gold_aspect_norm"] != ""].copy()
    multi_map_rows: list[dict[str, Any]] = []
    for (review_id, gold_aspect), group in mapped.groupby(["review_id", "mapped_gold_aspect_norm"], sort=True):
        non_duplicate = group[group["manual_decision_norm"] != "DUPLICATE"]
        if len(non_duplicate) <= 1:
            continue
        multi_map_rows.append(
            {
                "review_id": review_id,
                "gold_aspect": gold_aspect,
                "n_mapped_rows": int(len(group)),
                "n_non_duplicate_rows": int(len(non_duplicate)),
                "prediction_ids": "|".join(non_duplicate["prediction_id"].astype(str).tolist()),
                "decisions": "|".join(non_duplicate["manual_decision_norm"].astype(str).tolist()),
            }
        )
    multi_map = pd.DataFrame(multi_map_rows)
    details["multi_map_violations"] = multi_map

    unknown_system = system[~system["manual_decision_norm"].isin(SYSTEM_DECISION_VALUES)][["review_id", "prediction_id", "manual_decision"]].copy()
    unknown_gold = gold[~gold["status_norm"].isin(GOLD_STATUS_VALUES)][["review_id", "gold_aspect", "status"]].copy()
    unknown_review = review_status[~review_status["status"].fillna("").astype(str).str.strip().isin(REVIEW_STATUS_VALUES)][["review_id", "status"]].copy()
    unknown_sentiment = system[~system["manual_sentiment_decision_norm"].isin(MANUAL_SENTIMENT_VALUES)][["review_id", "prediction_id", "manual_sentiment_decision"]].copy()
    details["unknown_system_status"] = unknown_system
    details["unknown_gold_status"] = unknown_gold
    details["unknown_review_status"] = unknown_review
    details["unknown_manual_sentiment_status"] = unknown_sentiment

    dataset_review_ids = set(dataset_reviews["review_id"].astype(str))
    all_manual_review_ids = set(review_status["review_id"].astype(str)) | set(system["review_id"].astype(str)) | set(gold["review_id"].astype(str))
    absent_review_ids = sorted(all_manual_review_ids - dataset_review_ids)
    details["absent_review_ids"] = pd.DataFrame({"review_id": absent_review_ids})

    checks = [
        ValidationCheck(
            rule_id=1,
            name="Every reviewed review_id has decisions for all system aspects",
            passed=missing_system.empty,
            violation_count=int(len(missing_system)),
            details=f"missing system rows: {len(missing_system)} across {missing_system['review_id'].nunique() if not missing_system.empty else 0} review_ids",
        ),
        ValidationCheck(
            rule_id=2,
            name="Every reviewed review_id has decisions for all gold aspects",
            passed=missing_gold.empty,
            violation_count=int(len(missing_gold)),
            details=f"missing gold rows: {len(missing_gold)} across {missing_gold['review_id'].nunique() if not missing_gold.empty else 0} review_ids",
        ),
        ValidationCheck(
            rule_id=3,
            name="TP rows must have mapped_gold_aspect",
            passed=tp_missing_map.empty,
            violation_count=int(len(tp_missing_map)),
            details=f"TP without mapped gold aspect: {len(tp_missing_map)}",
        ),
        ValidationCheck(
            rule_id=4,
            name="FP rows must not have mapped_gold_aspect",
            passed=fp_with_map.empty,
            violation_count=int(len(fp_with_map)),
            details=f"FP with mapped gold aspect: {len(fp_with_map)}",
        ),
        ValidationCheck(
            rule_id=5,
            name="FOUND gold rows must have matched_system_aspect",
            passed=found_missing_match.empty,
            violation_count=int(len(found_missing_match)),
            details=f"FOUND without matched system id: {len(found_missing_match)}",
        ),
        ValidationCheck(
            rule_id=6,
            name="One gold_aspect must not map to multiple system_aspect except DUPLICATE",
            passed=multi_map.empty,
            violation_count=int(len(multi_map)),
            details=f"multi-map violations: {len(multi_map)}",
        ),
        ValidationCheck(
            rule_id=7,
            name="No unknown statuses",
            passed=unknown_system.empty and unknown_gold.empty and unknown_review.empty and unknown_sentiment.empty,
            violation_count=int(len(unknown_system) + len(unknown_gold) + len(unknown_review) + len(unknown_sentiment)),
            details=(
                f"unknown system={len(unknown_system)}, gold={len(unknown_gold)}, "
                f"review={len(unknown_review)}, manual_sentiment={len(unknown_sentiment)}"
            ),
        ),
        ValidationCheck(
            rule_id=8,
            name="No review_id absent from source dataset",
            passed=not absent_review_ids,
            violation_count=int(len(absent_review_ids)),
            details=f"absent review_ids: {len(absent_review_ids)}",
        ),
    ]
    return checks, details


def compute_detection_row(
    *,
    slice_type: str,
    slice_value: str,
    system_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    allow_recall: bool,
    allow_precision: bool,
    notes: str = "",
) -> dict[str, Any]:
    tp = int((system_df["manual_decision_norm"] == "TP").sum())
    fp = int((system_df["manual_decision_norm"] == "FP").sum())
    unclear = int((system_df["manual_decision_norm"] == "UNCLEAR").sum())
    duplicate = int((system_df["manual_decision_norm"] == "DUPLICATE").sum())
    out_of_scope = int((system_df["manual_decision_norm"] == "OUT_OF_SCOPE").sum())
    fn = int((gold_df["status_norm"] == "FN").sum()) if allow_recall else 0
    gold_unclear = int((gold_df["status_norm"] == "UNCLEAR").sum()) if not gold_df.empty else 0

    precision_strict = _safe_div(tp, tp + fp + unclear) if allow_precision else math.nan
    precision_soft = _safe_div(tp, tp + fp) if allow_precision else math.nan
    recall = _safe_div(tp, tp + fn) if allow_recall else math.nan
    f1_strict = _safe_div(2.0 * precision_strict * recall, precision_strict + recall) if allow_precision and allow_recall and (precision_strict + recall) else math.nan
    f1_soft = _safe_div(2.0 * precision_soft * recall, precision_soft + recall) if allow_precision and allow_recall and (precision_soft + recall) else math.nan

    return {
        "slice_type": slice_type,
        "slice_value": slice_value,
        "n_reviews": int(pd.concat([system_df["review_id"], gold_df["review_id"]], ignore_index=True).astype(str).nunique()) if (not system_df.empty or not gold_df.empty) else 0,
        "system_rows": int(len(system_df)),
        "gold_rows": int(len(gold_df)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "unclear": unclear,
        "duplicate": duplicate,
        "out_of_scope": out_of_scope,
        "gold_unclear": gold_unclear,
        "manual_precision_strict": precision_strict,
        "manual_precision_soft": precision_soft,
        "manual_recall": recall,
        "manual_f1_strict": f1_strict,
        "manual_f1_soft": f1_soft,
        "notes": notes,
    }


def enrich_sentiment_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        out = pairs.copy()
        for column in [
            "pred_round",
            "abs_error",
            "signed_error",
            "round_abs_error",
            "gold_direction",
            "pred_direction",
            "wrong_polarity",
            "strong_wrong_polarity",
            "too_low",
            "too_high",
            "large_too_low",
            "large_too_high",
            "error_type",
        ]:
            out[column] = pd.Series(dtype="object")
        return out

    out = pairs.copy()
    out["pred_round"] = out["predicted_rating"].map(_pred_round)
    out["abs_error"] = (out["predicted_rating"] - out["gold_rating"]).abs()
    out["signed_error"] = out["predicted_rating"] - out["gold_rating"]
    out["round_abs_error"] = (out["pred_round"] - out["gold_rating"]).abs()
    out["gold_direction"] = out["gold_rating"].map(gold_direction)
    out["pred_direction"] = out["predicted_rating"].map(predicted_direction)
    out["wrong_polarity"] = [
        wrong_polarity(gold, pred)
        for gold, pred in zip(out["gold_rating"], out["predicted_rating"], strict=True)
    ]
    out["strong_wrong_polarity"] = [
        strong_wrong_polarity(gold, pred)
        for gold, pred in zip(out["gold_rating"], out["predicted_rating"], strict=True)
    ]
    out["too_low"] = out["predicted_rating"] < (out["gold_rating"] - 1.0)
    out["too_high"] = out["predicted_rating"] > (out["gold_rating"] + 1.0)
    out["large_too_low"] = out["predicted_rating"] < (out["gold_rating"] - 2.0)
    out["large_too_high"] = out["predicted_rating"] > (out["gold_rating"] + 2.0)
    out["error_type"] = out.apply(classify_sentiment_error, axis=1)
    return out


def enrich_vector_sentiment_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        out = pairs.copy()
        for column in [
            "gold_vector_negative",
            "gold_vector_neutral",
            "gold_vector_positive",
            "pred_vector_negative",
            "pred_vector_neutral",
            "pred_vector_positive",
            "vector_l1_distance",
            "vector_l2_distance",
            "cosine_similarity",
            "gold_dominant_class",
            "pred_dominant_class",
            "dominant_class_match",
            "gold_strict_polar_class",
            "pred_strict_polar_class",
            "gold_polar_strength",
            "pred_polar_strength",
            "gold_strong_polar",
            "pred_strict_neutral",
            "neutral_collapse",
            "polarity_flip",
            "same_polar_direction",
            "intensity_underestimate",
        ]:
            out[column] = pd.Series(dtype="object")
        return out

    out = pairs.copy()
    gold_vectors = np.vstack([rating_to_fuzzy_sentiment_vector(value) for value in out["gold_rating"]])
    pred_vectors = np.vstack([rating_to_fuzzy_sentiment_vector(value) for value in out["predicted_rating"]])

    out["gold_vector_negative"] = gold_vectors[:, 0]
    out["gold_vector_neutral"] = gold_vectors[:, 1]
    out["gold_vector_positive"] = gold_vectors[:, 2]
    out["pred_vector_negative"] = pred_vectors[:, 0]
    out["pred_vector_neutral"] = pred_vectors[:, 1]
    out["pred_vector_positive"] = pred_vectors[:, 2]

    out["vector_l1_distance"] = np.abs(gold_vectors - pred_vectors).sum(axis=1)
    out["vector_l2_distance"] = np.linalg.norm(gold_vectors - pred_vectors, axis=1)
    out["cosine_similarity"] = [
        _vector_cosine_similarity(gold_vector, pred_vector)
        for gold_vector, pred_vector in zip(gold_vectors, pred_vectors, strict=True)
    ]
    out["gold_dominant_class"] = [_dominant_classes_label(vector) for vector in gold_vectors]
    out["pred_dominant_class"] = [_dominant_classes_label(vector) for vector in pred_vectors]
    out["dominant_class_match"] = [
        bool(set(_dominant_classes(gold_vector)) & set(_dominant_classes(pred_vector)))
        for gold_vector, pred_vector in zip(gold_vectors, pred_vectors, strict=True)
    ]
    out["gold_strict_polar_class"] = [_strict_polar_class(vector) for vector in gold_vectors]
    out["pred_strict_polar_class"] = [_strict_polar_class(vector) for vector in pred_vectors]
    out["gold_polar_strength"] = np.maximum(gold_vectors[:, 0], gold_vectors[:, 2])
    out["pred_polar_strength"] = np.maximum(pred_vectors[:, 0], pred_vectors[:, 2])
    out["gold_strong_polar"] = (
        out["gold_strict_polar_class"].astype(str).ne("")
        & (out["gold_polar_strength"] >= 0.75)
    )
    out["pred_strict_neutral"] = out["pred_dominant_class"].astype(str).eq(DIRECTION_NEUTRAL)
    out["neutral_collapse"] = out["gold_strong_polar"] & out["pred_strict_neutral"]
    out["polarity_flip"] = (
        out["gold_strict_polar_class"].astype(str).ne("")
        & out["pred_strict_polar_class"].astype(str).ne("")
        & out["gold_strict_polar_class"].astype(str).ne(out["pred_strict_polar_class"].astype(str))
    )
    out["same_polar_direction"] = [
        gold_class != "" and gold_class in set(_dominant_classes(pred_vector))
        for gold_class, pred_vector in zip(out["gold_strict_polar_class"], pred_vectors, strict=True)
    ]
    out["intensity_underestimate"] = (
        out["same_polar_direction"]
        & (out["pred_polar_strength"] <= (out["gold_polar_strength"] - 0.5))
    )
    return out


def compute_sentiment_row(
    *,
    slice_type: str,
    slice_value: str,
    pairs: pd.DataFrame,
) -> dict[str, Any]:
    if pairs.empty:
        return {
            "slice_type": slice_type,
            "slice_value": slice_value,
            "n_reviews": 0,
            "n_pairs": 0,
            "manual_sentiment_mae": math.nan,
            "manual_sentiment_mae_round": math.nan,
            "manual_accuracy_at_0_5": math.nan,
            "manual_accuracy_at_1_0": math.nan,
            "manual_accuracy_at_1_5": math.nan,
            "manual_direction_accuracy": math.nan,
            "manual_macro_direction_f1": math.nan,
            "manual_wrong_polarity_rate": math.nan,
            "manual_strong_wrong_polarity_rate": math.nan,
            "manual_mean_signed_error": math.nan,
            "manual_rmse": math.nan,
        }

    return {
        "slice_type": slice_type,
        "slice_value": slice_value,
        "n_reviews": int(pairs["review_id"].astype(str).nunique()),
        "n_pairs": int(len(pairs)),
        "manual_sentiment_mae": float(pairs["abs_error"].mean()),
        "manual_sentiment_mae_round": float(pairs["round_abs_error"].mean()),
        "manual_accuracy_at_0_5": float((pairs["abs_error"] <= 0.5).mean()),
        "manual_accuracy_at_1_0": float((pairs["abs_error"] <= 1.0).mean()),
        "manual_accuracy_at_1_5": float((pairs["abs_error"] <= 1.5).mean()),
        "manual_direction_accuracy": float((pairs["gold_direction"] == pairs["pred_direction"]).mean()),
        "manual_macro_direction_f1": float(np.mean([_label_f1(pairs, label) for label in (DIRECTION_NEGATIVE, DIRECTION_NEUTRAL, DIRECTION_POSITIVE)])),
        "manual_wrong_polarity_rate": float(pairs["wrong_polarity"].mean()),
        "manual_strong_wrong_polarity_rate": float(pairs["strong_wrong_polarity"].mean()),
        "manual_mean_signed_error": float(pairs["signed_error"].mean()),
        "manual_rmse": float(np.sqrt(np.mean(np.square(pairs["signed_error"])))),
    }


def compute_vector_sentiment_row(
    *,
    slice_type: str,
    slice_value: str,
    pairs: pd.DataFrame,
) -> dict[str, Any]:
    if pairs.empty:
        return {
            "slice_type": slice_type,
            "slice_value": slice_value,
            "n_reviews": 0,
            "n_pairs": 0,
            "n_gold_strict_polar_pairs": 0,
            "n_gold_strong_polar_pairs": 0,
            "n_same_polar_direction_pairs": 0,
            "mean_vector_l1_distance": math.nan,
            "mean_vector_l2_distance": math.nan,
            "mean_cosine_similarity": math.nan,
            "dominant_class_accuracy": math.nan,
            "neutral_collapse_rate": math.nan,
            "polarity_flip_rate": math.nan,
            "intensity_underestimate_rate": math.nan,
        }

    gold_polar_mask = pairs["gold_strict_polar_class"].astype(str).ne("")
    strong_gold_polar_mask = pairs["gold_strong_polar"].astype(bool)
    same_polar_mask = pairs["same_polar_direction"].astype(bool)
    return {
        "slice_type": slice_type,
        "slice_value": slice_value,
        "n_reviews": int(pairs["review_id"].astype(str).nunique()),
        "n_pairs": int(len(pairs)),
        "n_gold_strict_polar_pairs": int(gold_polar_mask.sum()),
        "n_gold_strong_polar_pairs": int(strong_gold_polar_mask.sum()),
        "n_same_polar_direction_pairs": int(same_polar_mask.sum()),
        "mean_vector_l1_distance": float(pairs["vector_l1_distance"].mean()),
        "mean_vector_l2_distance": float(pairs["vector_l2_distance"].mean()),
        "mean_cosine_similarity": float(pairs["cosine_similarity"].mean()),
        "dominant_class_accuracy": float(pairs["dominant_class_match"].mean()),
        "neutral_collapse_rate": _safe_div(float(pairs["neutral_collapse"].sum()), float(strong_gold_polar_mask.sum())),
        "polarity_flip_rate": _safe_div(float(pairs["polarity_flip"].sum()), float(gold_polar_mask.sum())),
        "intensity_underestimate_rate": _safe_div(float(pairs["intensity_underestimate"].sum()), float(same_polar_mask.sum())),
    }


def build_sentiment_pairs(base: dict[str, pd.DataFrame]) -> pd.DataFrame:
    system = base["system"].copy()
    gold_lookup = base["dataset_gold"][
        ["review_id", "gold_aspect", "gold_rating", "nm_id", "category_id", "full_text", "review_rating"]
    ].copy()

    pairs = system[
        (system["manual_decision_norm"] == "TP")
        & (system["mapped_gold_aspect_norm"] != "")
    ].copy()
    pairs = pairs.merge(
        gold_lookup,
        left_on=["review_id", "mapped_gold_aspect_norm"],
        right_on=["review_id", "gold_aspect"],
        how="left",
        validate="many_to_one",
    )
    pairs = pairs.rename(
        columns={
            "system_aspect": "system_aspect_name",
            "mapped_gold_aspect_norm": "gold_aspect_name",
            "system_rating": "predicted_rating",
        }
    )
    pairs = pairs[pd.notna(pairs["gold_rating"]) & pd.notna(pairs["predicted_rating"])].copy()
    pairs["gold_rating"] = pairs["gold_rating"].astype(float)
    pairs["predicted_rating"] = pairs["predicted_rating"].astype(float)
    pairs["nm_id"] = pairs["nm_id_x"].fillna(pairs["nm_id_y"]).astype(int)
    pairs["category_id"] = pairs["category_id_x"].fillna(pairs["category_id_y"]).astype(str)
    pairs["full_text"] = pairs["full_text_x"].fillna(pairs["full_text_y"]).astype(str)
    pairs["review_rating"] = pairs["review_rating_x"].fillna(pairs["review_rating_y"]).astype(float)
    pairs["aspect_source"] = pairs["aspect_source"].fillna("").astype(str)
    pairs = pairs[
        [
            "review_id",
            "nm_id",
            "category_id",
            "full_text",
            "review_rating",
            "prediction_id",
            "aspect_source",
            "system_aspect_name",
            "gold_aspect_name",
            "gold_rating",
            "predicted_rating",
            "manual_sentiment_decision_norm",
            "comment",
        ]
    ].copy()
    return enrich_vector_sentiment_pairs(enrich_sentiment_pairs(pairs))


def _metric_group_rows(
    *,
    system_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    detection_rows = [
        compute_detection_row(
            slice_type="overall",
            slice_value="overall",
            system_df=system_df,
            gold_df=gold_df,
            allow_recall=True,
            allow_precision=True,
        )
    ]
    for aspect_source, group in system_df.groupby("aspect_source", sort=True):
        detection_rows.append(
            compute_detection_row(
                slice_type="aspect_source",
                slice_value=str(aspect_source),
                system_df=group,
                gold_df=gold_df.iloc[0:0].copy(),
                allow_recall=False,
                allow_precision=True,
                notes="FN has no aspect_source label; recall/F1 undefined for this slice.",
            )
        )
    detection_df = pd.DataFrame(detection_rows)

    sentiment_rows = [
        compute_sentiment_row(
            slice_type="overall",
            slice_value="overall",
            pairs=pairs_df,
        )
    ]
    for aspect_source, group in pairs_df.groupby("aspect_source", sort=True):
        sentiment_rows.append(
            compute_sentiment_row(
                slice_type="aspect_source",
                slice_value=str(aspect_source),
                pairs=group,
            )
        )
    sentiment_df = pd.DataFrame(sentiment_rows)

    category_rows: list[dict[str, Any]] = []
    categories = sorted(set(system_df["category_id"].dropna().astype(str)) | set(gold_df["category_id"].dropna().astype(str)))
    for category_id in categories:
        det = compute_detection_row(
            slice_type="category_id",
            slice_value=category_id,
            system_df=system_df[system_df["category_id"].astype(str) == category_id],
            gold_df=gold_df[gold_df["category_id"].astype(str) == category_id],
            allow_recall=True,
            allow_precision=True,
        )
        sent = compute_sentiment_row(
            slice_type="category_id",
            slice_value=category_id,
            pairs=pairs_df[pairs_df["category_id"].astype(str) == category_id],
        )
        category_rows.append({**det, **{k: v for k, v in sent.items() if k not in {"slice_type", "slice_value"}}})
    category_df = pd.DataFrame(category_rows)

    product_rows: list[dict[str, Any]] = []
    products = sorted(set(system_df["nm_id"].dropna().astype(int)) | set(gold_df["nm_id"].dropna().astype(int)))
    for nm_id in products:
        system_slice = system_df[system_df["nm_id"].fillna(-1).astype(int) == int(nm_id)]
        gold_slice = gold_df[gold_df["nm_id"].fillna(-1).astype(int) == int(nm_id)]
        det = compute_detection_row(
            slice_type="nm_id",
            slice_value=str(nm_id),
            system_df=system_slice,
            gold_df=gold_slice,
            allow_recall=True,
            allow_precision=True,
        )
        sent = compute_sentiment_row(
            slice_type="nm_id",
            slice_value=str(nm_id),
            pairs=pairs_df[pairs_df["nm_id"].astype(int) == int(nm_id)],
        )
        category_id = ""
        if not system_slice.empty:
            category_id = str(system_slice["category_id"].iloc[0])
        elif not gold_slice.empty:
            category_id = str(gold_slice["category_id"].iloc[0])
        row = {**det, **{k: v for k, v in sent.items() if k not in {"slice_type", "slice_value"}}}
        row["category_id"] = category_id
        product_rows.append(row)
    product_df = pd.DataFrame(product_rows)

    aspect_rows: list[dict[str, Any]] = []
    for gold_aspect, gold_slice in gold_df.groupby("gold_aspect", sort=True):
        tp_slice = system_df[
            (system_df["manual_decision_norm"] == "TP")
            & (system_df["mapped_gold_aspect_norm"] == str(gold_aspect))
        ].copy()
        det = compute_detection_row(
            slice_type="gold_aspect",
            slice_value=str(gold_aspect),
            system_df=tp_slice,
            gold_df=gold_slice,
            allow_recall=True,
            allow_precision=False,
            notes="FP/UNCLEAR are not attributable to gold_aspect; precision/F1 undefined.",
        )
        sent = compute_sentiment_row(
            slice_type="gold_aspect",
            slice_value=str(gold_aspect),
            pairs=pairs_df[pairs_df["gold_aspect_name"] == str(gold_aspect)],
        )
        aspect_rows.append(
            {
                "aspect_dimension": "gold_aspect",
                **det,
                **{k: v for k, v in sent.items() if k not in {"slice_type", "slice_value"}},
            }
        )

    for system_aspect, system_slice in system_df.groupby("system_aspect", sort=True):
        det = compute_detection_row(
            slice_type="system_aspect",
            slice_value=str(system_aspect),
            system_df=system_slice,
            gold_df=gold_df.iloc[0:0].copy(),
            allow_recall=False,
            allow_precision=True,
            notes="FN has no system_aspect label; recall/F1 undefined for this slice.",
        )
        sent = compute_sentiment_row(
            slice_type="system_aspect",
            slice_value=str(system_aspect),
            pairs=pairs_df[pairs_df["system_aspect_name"] == str(system_aspect)],
        )
        aspect_rows.append(
            {
                "aspect_dimension": "system_aspect",
                **det,
                **{k: v for k, v in sent.items() if k not in {"slice_type", "slice_value"}},
            }
        )
    aspect_df = pd.DataFrame(aspect_rows)

    return {
        "detection": detection_df,
        "sentiment": sentiment_df,
        "category": category_df,
        "product": product_df,
        "aspect": aspect_df,
    }


def _vector_metric_group_rows(pairs_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics_rows = [
        compute_vector_sentiment_row(
            slice_type="overall",
            slice_value="overall",
            pairs=pairs_df,
        )
    ]
    for aspect_source, group in pairs_df.groupby("aspect_source", sort=True):
        metrics_rows.append(
            compute_vector_sentiment_row(
                slice_type="aspect_source",
                slice_value=str(aspect_source),
                pairs=group,
            )
        )
    for gold_aspect, group in pairs_df.groupby("gold_aspect_name", sort=True):
        metrics_rows.append(
            compute_vector_sentiment_row(
                slice_type="gold_aspect",
                slice_value=str(gold_aspect),
                pairs=group,
            )
        )
    for system_aspect, group in pairs_df.groupby("system_aspect_name", sort=True):
        metrics_rows.append(
            compute_vector_sentiment_row(
                slice_type="system_aspect",
                slice_value=str(system_aspect),
                pairs=group,
            )
        )
    metrics_df = pd.DataFrame(metrics_rows)

    category_rows: list[dict[str, Any]] = []
    for category_id, group in pairs_df.groupby("category_id", sort=True):
        category_rows.append(
            compute_vector_sentiment_row(
                slice_type="category_id",
                slice_value=str(category_id),
                pairs=group,
            )
        )
    category_df = pd.DataFrame(category_rows)

    product_rows: list[dict[str, Any]] = []
    for nm_id, group in pairs_df.groupby("nm_id", sort=True):
        row = compute_vector_sentiment_row(
            slice_type="nm_id",
            slice_value=str(int(nm_id)),
            pairs=group,
        )
        row["category_id"] = str(group["category_id"].iloc[0]) if not group.empty else ""
        product_rows.append(row)
    product_df = pd.DataFrame(product_rows)

    return {
        "metrics": metrics_df,
        "category": category_df,
        "product": product_df,
    }


def build_error_summary(system_df: pd.DataFrame, gold_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for decision, count in system_df["manual_decision_norm"].value_counts(dropna=False).to_dict().items():
        rows.append({"section": "system_manual_decision", "dimension": "manual_decision", "value": decision, "count": int(count), "rate": float(count / len(system_df)) if len(system_df) else math.nan})

    for status, count in gold_df["status_norm"].value_counts(dropna=False).to_dict().items():
        rows.append({"section": "gold_status", "dimension": "status", "value": status, "count": int(count), "rate": float(count / len(gold_df)) if len(gold_df) else math.nan})

    if not pairs_df.empty:
        for error_type, count in pairs_df["error_type"].value_counts(dropna=False).to_dict().items():
            rows.append({"section": "sentiment_error_type", "dimension": "error_type", "value": error_type, "count": int(count), "rate": float(count / len(pairs_df))})

    fp_top = system_df[system_df["manual_decision_norm"] == "FP"]["system_aspect"].value_counts().head(20)
    for aspect, count in fp_top.items():
        rows.append({"section": "top_fp_system_aspect", "dimension": "system_aspect", "value": str(aspect), "count": int(count), "rate": math.nan})

    fn_top = gold_df[gold_df["status_norm"] == "FN"]["gold_aspect"].value_counts().head(20)
    for aspect, count in fn_top.items():
        rows.append({"section": "top_fn_gold_aspect", "dimension": "gold_aspect", "value": str(aspect), "count": int(count), "rate": math.nan})

    return pd.DataFrame(rows).sort_values(["section", "count", "value"], ascending=[True, False, True]).reset_index(drop=True)


def extract_auto_metrics(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    track_b = payload.get("track_b", {})
    return {
        "auto_detection_precision": track_b.get("detection_precision"),
        "auto_detection_recall": track_b.get("detection_recall"),
        "auto_detection_f1": track_b.get("detection_f1"),
        "auto_sentiment_mae_review": track_b.get("sentiment_mae_review"),
        "auto_sentiment_mae_review_round": track_b.get("sentiment_mae_review_round"),
        "auto_product_mae_n3": track_b.get("product_mae_n3"),
        "auto_n_sentiment_review_matches": track_b.get("n_sentiment_review_matches"),
        "auto_n_discovery_sentiment_pairs": track_b.get("n_discovery_sentiment_pairs"),
    }


def write_validation_report(
    out_path: Path,
    checks: list[ValidationCheck],
    details: dict[str, pd.DataFrame],
    base: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# Manual Audit Validation Report",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- reviewed review_ids: {base['review_status']['review_id'].astype(str).nunique()}",
        f"- expected system rows: {len(base['expected_system'])}",
        f"- stored system rows: {len(base['system'])}",
        f"- expected gold rows: {len(base['dataset_gold'])}",
        f"- stored gold rows: {len(base['gold'])}",
        "",
        "## Rule Summary",
        "",
        "| Rule | Status | Violations | Details |",
        "|---:|---|---:|---|",
    ]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"| {check.rule_id} | {status} | {check.violation_count} | {check.details} |")

    extra_warning_rows = []
    review_status = base["review_status"]
    non_committed_done = review_status[(review_status["status"].astype(str) == "done") & (review_status["committed"].fillna(0).astype(int) == 0)]
    if not non_committed_done.empty:
        extra_warning_rows.append(f"- done but committed=0: {len(non_committed_done)} review_id(s)")
    if extra_warning_rows:
        lines.extend(["", "## Extra Warnings", ""])
        lines.extend(extra_warning_rows)

    lines.extend(["", "## Samples", ""])
    sample_order = [
        ("missing_system_decisions", "Missing system decisions"),
        ("missing_gold_decisions", "Missing gold decisions"),
        ("tp_missing_map", "TP without mapped gold"),
        ("found_missing_match", "FOUND without matched system id"),
        ("multi_map_violations", "Multi-map violations"),
        ("unknown_system_status", "Unknown system statuses"),
        ("absent_review_ids", "Review IDs absent from dataset"),
    ]
    for key, title in sample_order:
        frame = details.get(key, pd.DataFrame())
        lines.append(f"### {title}")
        if frame.empty:
            lines.append("")
            lines.append("No violations.")
            lines.append("")
            continue
        lines.append("")
        lines.append(frame.head(10).to_markdown(index=False))
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_summary(
    out_path: Path,
    *,
    base: dict[str, pd.DataFrame],
    validation_checks: list[ValidationCheck],
    detection_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    error_summary_df: pd.DataFrame,
    auto_metrics: dict[str, Any],
) -> None:
    overall_detection = detection_df[detection_df["slice_type"] == "overall"].iloc[0]
    overall_sentiment = sentiment_df[sentiment_df["slice_type"] == "overall"].iloc[0]
    top_sections = error_summary_df[error_summary_df["section"].isin(["system_manual_decision", "gold_status", "sentiment_error_type"])].copy()

    lines = [
        "# Manual Audit Final Summary",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- validation_failed_rules: {sum(0 if check.passed else 1 for check in validation_checks)} / {len(validation_checks)}",
        "",
        "## Coverage",
        "",
        f"1. checked reviews: {base['review_status']['review_id'].astype(str).nunique()}",
        f"2. stored system aspects: {len(base['system'])} (expected {len(base['expected_system'])})",
        f"3. stored gold aspects: {len(base['gold'])} (expected {len(base['dataset_gold'])})",
        "",
        "## Manual Detection",
        "",
        f"4. TP={int(overall_detection['tp'])}, FP={int(overall_detection['fp'])}, FN={int(overall_detection['fn'])}, UNCLEAR={int(overall_detection['unclear'])}, DUPLICATE={int(overall_detection['duplicate'])}",
        f"5. manual_precision_strict={overall_detection['manual_precision_strict']:.4f}",
        f"6. manual_precision_soft={overall_detection['manual_precision_soft']:.4f}",
        f"7. manual_recall={overall_detection['manual_recall']:.4f}",
        f"8. manual_f1_strict={overall_detection['manual_f1_strict']:.4f}",
        f"9. manual_f1_soft={overall_detection['manual_f1_soft']:.4f}",
        "",
        "## Manual Sentiment",
        "",
        f"10. matched TP pairs used for sentiment: {int(overall_sentiment['n_pairs'])}",
        f"11. manual_sentiment_mae={overall_sentiment['manual_sentiment_mae']:.4f}",
        f"12. manual_sentiment_mae_round={overall_sentiment['manual_sentiment_mae_round']:.4f}",
        f"13. manual_accuracy_at_1_0={overall_sentiment['manual_accuracy_at_1_0']:.4f}",
        f"14. manual_wrong_polarity_rate={overall_sentiment['manual_wrong_polarity_rate']:.4f}",
        "",
        "## Main Error Types",
        "",
    ]

    for row in top_sections.head(12).itertuples(index=False):
        rate_text = ""
        if isinstance(row.rate, float) and not math.isnan(row.rate):
            rate_text = f" ({row.rate:.4f})"
        lines.append(f"- {row.section}: {row.value} = {row.count}{rate_text}")

    lines.extend(
        [
            "",
            "## Manual vs Auto",
            "",
            f"- auto detection precision (track_b): {auto_metrics['auto_detection_precision']:.4f}",
            f"- auto detection recall (track_b): {auto_metrics['auto_detection_recall']:.4f}",
            f"- auto detection f1 (track_b): {auto_metrics['auto_detection_f1']:.4f}",
            f"- auto sentiment mae review (track_b): {auto_metrics['auto_sentiment_mae_review']:.4f}",
            f"- auto sentiment mae round (track_b): {auto_metrics['auto_sentiment_mae_review_round']:.4f}",
            "- note: auto sentiment in run_summary is review-level Track B; manual sentiment here is TP-pair-level after manual mapping, so MAE values are informative but not strictly the same unit.",
            "",
            "## Vkr Conclusion",
            "",
            "- Простыми словами: ручная проверка показывает, где модель реально попадает в аспект, а где ошибается на ложных аспектах, дублях и тональности.",
            "- Эти manual-метрики можно использовать в ВКР как честную пост-оценку текущего inference без нового прогона модели.",
            "- Если validation report содержит FAIL, это нужно явно оговорить: часть audit-таблиц заполнена неполно, и итоговые manual-метрики считаются по фактически сохранённой ручной разметке.",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_vector_sentiment_figures(
    output_dir: Path,
    *,
    pairs_df: pd.DataFrame,
    vector_metric_frames: dict[str, pd.DataFrame],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    figures_dir = output_dir / "vector_sentiment_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_names: list[str] = []

    if not pairs_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].hist(pairs_df["vector_l1_distance"], bins=20, color="#1f77b4", edgecolor="white")
        axes[0].set_title("Vector L1 Distance")
        axes[0].set_xlabel("L1")
        axes[0].set_ylabel("Pairs")

        axes[1].hist(pairs_df["vector_l2_distance"], bins=20, color="#ff7f0e", edgecolor="white")
        axes[1].set_title("Vector L2 Distance")
        axes[1].set_xlabel("L2")

        axes[2].hist(pairs_df["cosine_similarity"], bins=20, color="#2ca02c", edgecolor="white")
        axes[2].set_title("Cosine Similarity")
        axes[2].set_xlabel("Cosine")

        fig.tight_layout()
        name = "vector_distance_distributions.png"
        fig.savefig(figures_dir / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_names.append(name)

    category_df = vector_metric_frames["category"].copy()
    if not category_df.empty:
        category_df = category_df.sort_values("mean_vector_l1_distance", ascending=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        y = np.arange(len(category_df))

        axes[0, 0].barh(y, category_df["mean_vector_l1_distance"], color="#1f77b4")
        axes[0, 0].set_yticks(y, category_df["slice_value"])
        axes[0, 0].set_title("Mean Vector L1 by Category")

        axes[0, 1].barh(y, category_df["mean_cosine_similarity"], color="#2ca02c")
        axes[0, 1].set_yticks(y, category_df["slice_value"])
        axes[0, 1].set_title("Mean Cosine by Category")

        axes[1, 0].barh(y, category_df["neutral_collapse_rate"].fillna(0.0), color="#d62728")
        axes[1, 0].set_yticks(y, category_df["slice_value"])
        axes[1, 0].set_title("Neutral Collapse Rate")

        axes[1, 1].barh(y, category_df["polarity_flip_rate"].fillna(0.0), color="#9467bd")
        axes[1, 1].set_yticks(y, category_df["slice_value"])
        axes[1, 1].set_title("Polarity Flip Rate")

        fig.tight_layout()
        name = "vector_metrics_by_category.png"
        fig.savefig(figures_dir / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_names.append(name)

    aspect_source_df = vector_metric_frames["metrics"].copy()
    aspect_source_df = aspect_source_df[aspect_source_df["slice_type"] == "aspect_source"].copy()
    if not aspect_source_df.empty:
        labels = aspect_source_df["slice_value"].astype(str).tolist()
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, aspect_source_df["mean_vector_l1_distance"], width=width, label="mean_l1", color="#1f77b4")
        ax.bar(x, aspect_source_df["mean_cosine_similarity"], width=width, label="mean_cosine", color="#2ca02c")
        ax.bar(x + width, aspect_source_df["dominant_class_accuracy"], width=width, label="dominant_acc", color="#ff7f0e")
        ax.set_xticks(x, labels)
        ax.set_title("Vector Metrics by Aspect Source")
        ax.legend()
        fig.tight_layout()
        name = "vector_metrics_by_aspect_source.png"
        fig.savefig(figures_dir / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_names.append(name)

    return figure_names


def write_vector_sentiment_summary(
    out_path: Path,
    *,
    pairs_df: pd.DataFrame,
    vector_metric_frames: dict[str, pd.DataFrame],
    figure_names: list[str],
) -> None:
    overall = vector_metric_frames["metrics"][vector_metric_frames["metrics"]["slice_type"] == "overall"].iloc[0]
    category_df = vector_metric_frames["category"].copy()
    aspect_source_df = vector_metric_frames["metrics"][vector_metric_frames["metrics"]["slice_type"] == "aspect_source"].copy()

    lines = [
        "# Vector Sentiment Summary",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- matched_tp_pairs: {int(overall['n_pairs'])}",
        f"- reviews_covered: {int(overall['n_reviews'])}",
        "",
        "## Overall",
        "",
        f"- mean_vector_l1_distance: {overall['mean_vector_l1_distance']:.4f}",
        f"- mean_vector_l2_distance: {overall['mean_vector_l2_distance']:.4f}",
        f"- mean_cosine_similarity: {overall['mean_cosine_similarity']:.4f}",
        f"- dominant_class_accuracy: {overall['dominant_class_accuracy']:.4f}",
        f"- neutral_collapse_rate: {overall['neutral_collapse_rate']:.4f}",
        f"- polarity_flip_rate: {overall['polarity_flip_rate']:.4f}",
        f"- intensity_underestimate_rate: {overall['intensity_underestimate_rate']:.4f}",
        "",
        "## Rate Denominators",
        "",
        f"- n_gold_strict_polar_pairs: {int(overall['n_gold_strict_polar_pairs'])}",
        f"- n_gold_strong_polar_pairs: {int(overall['n_gold_strong_polar_pairs'])}",
        f"- n_same_polar_direction_pairs: {int(overall['n_same_polar_direction_pairs'])}",
        "- neutral_collapse_rate = collapse / strong gold polarity pairs",
        "- polarity_flip_rate = flips / gold strict polarity pairs",
        "- intensity_underestimate_rate = severe underestimates / same-polarity direction pairs",
    ]

    if not category_df.empty:
        lowest_l1 = category_df.sort_values("mean_vector_l1_distance", ascending=True).iloc[0]
        highest_l1 = category_df.sort_values("mean_vector_l1_distance", ascending=False).iloc[0]
        highest_flip = category_df.sort_values("polarity_flip_rate", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                "## Category Highlights",
                "",
                f"- best_l1_category: {lowest_l1['slice_value']} ({lowest_l1['mean_vector_l1_distance']:.4f})",
                f"- worst_l1_category: {highest_l1['slice_value']} ({highest_l1['mean_vector_l1_distance']:.4f})",
                f"- max_polarity_flip_category: {highest_flip['slice_value']} ({highest_flip['polarity_flip_rate']:.4f})",
            ]
        )

    if not aspect_source_df.empty:
        lines.extend(["", "## Aspect Source", ""])
        for row in aspect_source_df.itertuples(index=False):
            lines.append(
                f"- {row.slice_value}: l1={row.mean_vector_l1_distance:.4f}, "
                f"cosine={row.mean_cosine_similarity:.4f}, dominant_acc={row.dominant_class_accuracy:.4f}"
            )

    if figure_names:
        lines.extend(["", "## Figures", ""])
        for name in figure_names:
            lines.append(f"- vector_sentiment_figures/{name}")

    if not pairs_df.empty:
        hardest = pairs_df.sort_values(["vector_l1_distance", "review_id"], ascending=[False, True]).head(3)
        lines.extend(["", "## Hard Examples", ""])
        for row in hardest.itertuples(index=False):
            lines.append(
                f"- review_id={row.review_id}, gold={row.gold_aspect_name}:{row.gold_rating:.2f}, "
                f"pred={row.system_aspect_name}:{row.predicted_rating:.2f}, l1={row.vector_l1_distance:.4f}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    *,
    output_dir: Path,
    base: dict[str, pd.DataFrame],
    validation_checks: list[ValidationCheck],
    validation_details: dict[str, pd.DataFrame],
    metric_frames: dict[str, pd.DataFrame],
    vector_metric_frames: dict[str, pd.DataFrame],
    error_summary_df: pd.DataFrame,
    hard_cases_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    auto_metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_validation_report(output_dir / OUT_FILE_NAMES["validation"], validation_checks, validation_details, base)
    metric_frames["detection"].to_csv(output_dir / OUT_FILE_NAMES["detection"], index=False, encoding="utf-8")
    metric_frames["sentiment"].to_csv(output_dir / OUT_FILE_NAMES["sentiment"], index=False, encoding="utf-8")
    metric_frames["category"].to_csv(output_dir / OUT_FILE_NAMES["category"], index=False, encoding="utf-8")
    metric_frames["product"].to_csv(output_dir / OUT_FILE_NAMES["product"], index=False, encoding="utf-8")
    metric_frames["aspect"].to_csv(output_dir / OUT_FILE_NAMES["aspect"], index=False, encoding="utf-8")
    error_summary_df.to_csv(output_dir / OUT_FILE_NAMES["errors"], index=False, encoding="utf-8")
    hard_cases_df.to_csv(output_dir / OUT_FILE_NAMES["hard_cases"], index=False, encoding="utf-8")
    vector_metric_frames["metrics"].to_csv(output_dir / OUT_FILE_NAMES["vector_sentiment"], index=False, encoding="utf-8")
    vector_metric_frames["category"].to_csv(output_dir / OUT_FILE_NAMES["vector_category"], index=False, encoding="utf-8")
    vector_metric_frames["product"].to_csv(output_dir / OUT_FILE_NAMES["vector_product"], index=False, encoding="utf-8")
    write_final_summary(
        output_dir / OUT_FILE_NAMES["summary"],
        base=base,
        validation_checks=validation_checks,
        detection_df=metric_frames["detection"],
        sentiment_df=metric_frames["sentiment"],
        error_summary_df=error_summary_df,
        auto_metrics=auto_metrics,
    )
    figure_names = create_vector_sentiment_figures(output_dir, pairs_df=pairs_df, vector_metric_frames=vector_metric_frames)
    write_vector_sentiment_summary(
        output_dir / OUT_FILE_NAMES["vector_summary"],
        pairs_df=pairs_df,
        vector_metric_frames=vector_metric_frames,
        figure_names=figure_names,
    )


def run_manual_recompute(
    *,
    db_path: Path,
    run_dir: Path,
    dataset_path: Path,
    output_root: Path,
) -> Path:
    base = build_audit_base(db_path=db_path, run_dir=run_dir, dataset_path=dataset_path)
    validation_checks, validation_details = build_validation_checks(base)
    pairs_df = build_sentiment_pairs(base)
    metric_frames = _metric_group_rows(system_df=base["system"], gold_df=base["gold"], pairs_df=pairs_df)
    vector_metric_frames = _vector_metric_group_rows(pairs_df)
    error_summary_df = build_error_summary(base["system"], base["gold"], pairs_df)
    hard_cases_df = pairs_df.sort_values(["abs_error", "review_id", "gold_aspect_name"], ascending=[False, True, True]).head(50).reset_index(drop=True)
    auto_metrics = extract_auto_metrics(run_dir)

    output_dir = output_root / datetime.now(timezone.utc).strftime("manual_metrics_%Y%m%d_%H%M%S")
    save_outputs(
        output_dir=output_dir,
        base=base,
        validation_checks=validation_checks,
        validation_details=validation_details,
        metric_frames=metric_frames,
        vector_metric_frames=vector_metric_frames,
        error_summary_df=error_summary_df,
        hard_cases_df=hard_cases_df,
        pairs_df=pairs_df,
        auto_metrics=auto_metrics,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate completed manual audit and recompute manual metrics without rerunning inference.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    out_dir = run_manual_recompute(
        db_path=Path(args.db_path),
        run_dir=Path(args.run_dir),
        dataset_path=Path(args.dataset_path),
        output_root=Path(args.output_root),
    )
    print(out_dir)


if __name__ == "__main__":
    main()
