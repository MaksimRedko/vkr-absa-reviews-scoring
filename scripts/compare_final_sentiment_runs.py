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

from benchmark.sentiment import common as sentiment_benchmark
from scripts import freeze_final_results
from scripts import run_phase2_baseline_matching as lexical

RUN_V1_DEFAULT = ROOT / "results" / "final_res_v1"
RUN_V2_DEFAULT = ROOT / "results" / "final_res_v2"
OUTPUT_ROOT_DEFAULT = ROOT / "results" / "final_res_v1_vs_v2_diagnostics"
TARGET_NM_ID = 619500952
SCOPE_OWN = "own_pairs"
SCOPE_COMMON = "common_pairs"
RUN_ID_V1 = "v1"
RUN_ID_V2 = "v2"
RUN_ID_DELTA = "v2_minus_v1"
GOLD_DIRECTION_NEGATIVE = "negative"
GOLD_DIRECTION_NEUTRAL = "neutral"
GOLD_DIRECTION_POSITIVE = "positive"
TOTAL_GOLD_PAIR_LABELS = ("negative", "neutral", "positive")
ACCURACY_THRESHOLDS = (0.5, 1.0, 1.5)
CONFUSION_LABELS = [1, 2, 3, 4, 5]
PAIR_KEY_COLUMNS = ["review_id", "nm_id", "category_id", "gold_label", "gold_rating"]
RUN_METRIC_COLUMNS = [
    "mae",
    "mae_round",
    "median_ae",
    "rmse",
    "mean_signed_error",
    "accuracy_at_0_5",
    "accuracy_at_1_0",
    "accuracy_at_1_5",
    "direction_accuracy",
    "positive_recall",
    "negative_recall",
    "neutral_recall",
    "macro_direction_f1",
    "wrong_polarity_rate",
    "strong_wrong_polarity_rate",
    "too_low_rate",
    "too_high_rate",
    "large_too_low_rate",
    "large_too_high_rate",
    "vocab_mae",
    "discovery_mae",
    "vocab_accuracy_at_1",
    "discovery_accuracy_at_1",
    "vocab_wrong_polarity",
    "discovery_wrong_polarity",
    "coverage",
    "n_pairs",
    "n_products",
]
BY_PRODUCT_COLUMNS = [
    "scope",
    "run_id",
    "nm_id",
    "category_id",
    "n_pairs",
    "mae",
    "accuracy_at_1",
    "direction_accuracy",
    "wrong_polarity_rate",
    "mean_signed_error",
]


@dataclass(slots=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    context: sentiment_benchmark.BenchmarkContext
    discovery_by_product: dict[int, Any]
    pair_predictions: pd.DataFrame
    evaluated_pairs: pd.DataFrame
    total_gold_pairs: int


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _round_and_clamp_rating(value: float) -> int:
    rounded = int(np.rint(float(value)))
    return max(1, min(5, rounded))


def gold_direction(gold_rating: float) -> str:
    value = float(gold_rating)
    if value <= 2.0:
        return GOLD_DIRECTION_NEGATIVE
    if value >= 4.0:
        return GOLD_DIRECTION_POSITIVE
    return GOLD_DIRECTION_NEUTRAL


def predicted_direction(pred_rating: float) -> str:
    value = float(pred_rating)
    if value < 2.75:
        return GOLD_DIRECTION_NEGATIVE
    if value <= 3.25:
        return GOLD_DIRECTION_NEUTRAL
    return GOLD_DIRECTION_POSITIVE


def wrong_polarity(gold_rating: float, pred_rating: float) -> bool:
    gold = float(gold_rating)
    pred = float(pred_rating)
    return (gold >= 4.0 and pred < 2.75) or (gold <= 2.0 and pred > 3.25)


def strong_wrong_polarity(gold_rating: float, pred_rating: float) -> bool:
    gold = float(gold_rating)
    pred = float(pred_rating)
    return (gold >= 4.0 and pred <= 2.0) or (gold <= 2.0 and pred >= 4.0)


def _label_recall(df: pd.DataFrame, label: str) -> float:
    gold_rows = df[df["gold_direction"] == label]
    if gold_rows.empty:
        return math.nan
    return float((gold_rows["pred_direction"] == label).mean())


def _label_f1(df: pd.DataFrame, label: str) -> float:
    tp = int(((df["gold_direction"] == label) & (df["pred_direction"] == label)).sum())
    fp = int(((df["gold_direction"] != label) & (df["pred_direction"] == label)).sum())
    fn = int(((df["gold_direction"] == label) & (df["pred_direction"] != label)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _classify_error_type(row: pd.Series) -> str:
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


def enrich_pair_rows(df: pd.DataFrame, *, total_gold_pairs: int) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["pred_round"] = pd.Series(dtype="int64")
        out["abs_error"] = pd.Series(dtype="float64")
        out["signed_error"] = pd.Series(dtype="float64")
        out["round_abs_error"] = pd.Series(dtype="float64")
        out["gold_direction"] = pd.Series(dtype="object")
        out["pred_direction"] = pd.Series(dtype="object")
        out["wrong_polarity"] = pd.Series(dtype="bool")
        out["strong_wrong_polarity"] = pd.Series(dtype="bool")
        out["too_low"] = pd.Series(dtype="bool")
        out["too_high"] = pd.Series(dtype="bool")
        out["large_too_low"] = pd.Series(dtype="bool")
        out["large_too_high"] = pd.Series(dtype="bool")
        out["error_type"] = pd.Series(dtype="object")
        out["coverage"] = 0.0
        return out

    out = df.copy()
    out["pred_round"] = out["predicted_rating"].map(_round_and_clamp_rating)
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
    out["error_type"] = out.apply(_classify_error_type, axis=1)
    out["coverage"] = float(len(out) / total_gold_pairs) if total_gold_pairs else 0.0
    return out


def compute_run_metrics(df: pd.DataFrame, *, total_gold_pairs: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if df.empty:
        for column in RUN_METRIC_COLUMNS:
            if column in {"n_pairs", "n_products"}:
                metrics[column] = 0.0
            else:
                metrics[column] = math.nan if column != "coverage" else 0.0
        return metrics

    metrics["mae"] = float(df["abs_error"].mean())
    metrics["mae_round"] = float(df["round_abs_error"].mean())
    metrics["median_ae"] = float(df["abs_error"].median())
    metrics["rmse"] = float(np.sqrt(np.mean(np.square(df["signed_error"]))))
    metrics["mean_signed_error"] = float(df["signed_error"].mean())
    for threshold in ACCURACY_THRESHOLDS:
        key = f"accuracy_at_{str(threshold).replace('.', '_')}"
        metrics[key] = float((df["abs_error"] <= threshold).mean())
    metrics["direction_accuracy"] = float((df["gold_direction"] == df["pred_direction"]).mean())
    metrics["positive_recall"] = _label_recall(df, GOLD_DIRECTION_POSITIVE)
    metrics["negative_recall"] = _label_recall(df, GOLD_DIRECTION_NEGATIVE)
    metrics["neutral_recall"] = _label_recall(df, GOLD_DIRECTION_NEUTRAL)
    metrics["macro_direction_f1"] = float(np.mean([_label_f1(df, label) for label in TOTAL_GOLD_PAIR_LABELS]))
    metrics["wrong_polarity_rate"] = float(df["wrong_polarity"].mean())
    metrics["strong_wrong_polarity_rate"] = float(df["strong_wrong_polarity"].mean())
    metrics["too_low_rate"] = float(df["too_low"].mean())
    metrics["too_high_rate"] = float(df["too_high"].mean())
    metrics["large_too_low_rate"] = float(df["large_too_low"].mean())
    metrics["large_too_high_rate"] = float(df["large_too_high"].mean())

    vocab_df = df[df["aspect_source"] == "vocab"]
    discovery_df = df[df["aspect_source"] == "discovery"]
    metrics["vocab_mae"] = float(vocab_df["abs_error"].mean()) if not vocab_df.empty else math.nan
    metrics["discovery_mae"] = float(discovery_df["abs_error"].mean()) if not discovery_df.empty else math.nan
    metrics["vocab_accuracy_at_1"] = float((vocab_df["abs_error"] <= 1.0).mean()) if not vocab_df.empty else math.nan
    metrics["discovery_accuracy_at_1"] = float((discovery_df["abs_error"] <= 1.0).mean()) if not discovery_df.empty else math.nan
    metrics["vocab_wrong_polarity"] = float(vocab_df["wrong_polarity"].mean()) if not vocab_df.empty else math.nan
    metrics["discovery_wrong_polarity"] = float(discovery_df["wrong_polarity"].mean()) if not discovery_df.empty else math.nan

    metrics["coverage"] = float(len(df) / total_gold_pairs) if total_gold_pairs else 0.0
    metrics["n_pairs"] = float(len(df))
    metrics["n_products"] = float(df["nm_id"].nunique())
    return metrics


def _delta_dict(v1_metrics: dict[str, float], v2_metrics: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in RUN_METRIC_COLUMNS:
        left = _safe_float(v1_metrics.get(key))
        right = _safe_float(v2_metrics.get(key))
        out[key] = right - left if not (math.isnan(left) or math.isnan(right)) else math.nan
    return out


def _prediction_lookup(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if df.empty:
        return {}
    return {
        (str(row.review_id), str(row.aspect_key)): row._asdict()
        for row in df.itertuples(index=False)
    }


def _load_pair_predictions(run_dir: Path) -> pd.DataFrame:
    prediction_rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("predictions_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        nm_id = int(payload["nm_id"])
        category_id = str(payload["category"])
        for review_payload in payload.get("reviews", []):
            review_id = str(review_payload["review_id"])
            review_rating = _safe_float(review_payload.get("rating"))
            for item in review_payload.get("vocabulary_aspects", []):
                aspect_id = str(item["aspect_id"])
                prediction_rows.append(
                    {
                        "review_id": review_id,
                        "nm_id": nm_id,
                        "category_id": category_id,
                        "aspect_key": f"vocab::{aspect_id}",
                        "aspect_name": str(item["aspect"]),
                        "aspect_source": "vocab",
                        "rating": _safe_float(item["rating"]),
                        "raw_rating": _safe_float(item.get("raw_rating", item["rating"])),
                        "negation_corrected": bool(item.get("negation_corrected", False)),
                        "negation_pattern": str(item.get("negation_pattern", "")),
                        "negation_hit_lemma": str(item.get("negation_hit_lemma", "")),
                        "gold_matches_json": "{}",
                        "review_rating": review_rating,
                    }
                )
            for item in review_payload.get("discovery_aspects", []):
                cluster_id = int(item["cluster_id"])
                prediction_rows.append(
                    {
                        "review_id": review_id,
                        "nm_id": nm_id,
                        "category_id": category_id,
                        "aspect_key": f"discovery::{nm_id}::{cluster_id}",
                        "aspect_name": str(item["medoid"]),
                        "aspect_source": "discovery",
                        "rating": _safe_float(item["rating"]),
                        "raw_rating": _safe_float(item.get("raw_rating", item["rating"])),
                        "negation_corrected": bool(item.get("negation_corrected", False)),
                        "negation_pattern": str(item.get("negation_pattern", "")),
                        "negation_hit_lemma": str(item.get("negation_hit_lemma", "")),
                        "gold_matches_json": _json_dumps(item.get("gold_matches", {})),
                        "review_rating": review_rating,
                    }
                )

    base = pd.DataFrame(prediction_rows)
    if base.empty:
        return base

    nli = pd.read_parquet(run_dir / "nli_predictions.parquet").copy()
    nli = nli.rename(columns={"final_rating": "nli_final_rating", "raw_rating": "nli_raw_rating"})
    meta_columns = [
        "review_id",
        "aspect_name",
        "aspect_source",
        "premise_text",
        "p_entailment",
        "p_neutral",
        "p_contradiction",
        "relevance_filter_value",
        "passed_relevance_filter",
        "nli_final_rating",
        "nli_raw_rating",
    ]
    merged = base.merge(
        nli[meta_columns],
        on=["review_id", "aspect_name", "aspect_source"],
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values(["review_id", "aspect_source", "aspect_name"]).reset_index(drop=True)


def _hydrate_reviews_from_pair_predictions(
    context: sentiment_benchmark.BenchmarkContext,
    pair_predictions: pd.DataFrame,
) -> None:
    for review in context.reviews:
        review.vocab_aspect_ids = set()
        review.discovery_cluster_ids = set()

    if pair_predictions.empty:
        return

    for row in pair_predictions.itertuples(index=False):
        review = context.reviews_by_id.get(str(row.review_id))
        if review is None:
            continue
        if str(row.aspect_source) == "vocab":
            review.vocab_aspect_ids.add(str(row.aspect_key).split("::", 1)[1])
            continue
        parts = str(row.aspect_key).split("::")
        if len(parts) >= 3:
            review.discovery_cluster_ids.add(int(parts[2]))


def _build_pair_evaluations(
    context: sentiment_benchmark.BenchmarkContext,
    pair_predictions: pd.DataFrame,
    discovery_by_product: dict[int, Any],
) -> pd.DataFrame:
    by_key = _prediction_lookup(pair_predictions)
    rows: list[dict[str, Any]] = []
    for review in context.reviews:
        term_to_aspects = context.term_to_aspects_by_category[review.category_id]
        for gold_label, gold_rating in review.true_labels.items():
            mapped_ids = sorted(term_to_aspects.get(lexical._normalize(gold_label), set()))
            predicted_rows: list[dict[str, Any]] = []
            for aspect_id in mapped_ids:
                row = by_key.get((str(review.review_id), f"vocab::{aspect_id}"))
                if row is not None:
                    predicted_rows.append(row)
            if not mapped_ids:
                discovery = discovery_by_product.get(int(review.nm_id))
                if discovery is not None:
                    for cluster_id in sorted(review.discovery_cluster_ids):
                        cluster = discovery.clusters.get(int(cluster_id))
                        if cluster is None or gold_label not in cluster.gold_matches:
                            continue
                        row = by_key.get((str(review.review_id), f"discovery::{review.nm_id}::{cluster_id}"))
                        if row is not None:
                            predicted_rows.append(row)
            if not predicted_rows:
                continue

            predicted_rating = float(np.mean([float(item["rating"]) for item in predicted_rows]))
            raw_predicted_rating = float(np.mean([float(item["raw_rating"]) for item in predicted_rows]))
            aspect_source = "discovery" if any(str(item["aspect_source"]) == "discovery" for item in predicted_rows) else "vocab"
            rows.append(
                {
                    "review_id": str(review.review_id),
                    "nm_id": int(review.nm_id),
                    "category_id": str(review.category_id),
                    "review_rating": _safe_float(review.rating),
                    "review_text": str(review.text),
                    "gold_label": str(gold_label),
                    "gold_rating": float(gold_rating),
                    "predicted_rating": predicted_rating,
                    "raw_predicted_rating": raw_predicted_rating,
                    "aspect_source": aspect_source,
                    "n_predicted_items": int(len(predicted_rows)),
                    "predicted_keys_json": _json_dumps([str(item["aspect_key"]) for item in predicted_rows]),
                    "predicted_aspect_names_json": _json_dumps([str(item["aspect_name"]) for item in predicted_rows]),
                    "premise_texts_json": _json_dumps([str(item.get("premise_text", "")) for item in predicted_rows]),
                    "negation_patterns_json": _json_dumps(
                        [str(item.get("negation_pattern", "")) for item in predicted_rows if str(item.get("negation_pattern", ""))]
                    ),
                    "negation_corrected": bool(any(bool(item.get("negation_corrected", False)) for item in predicted_rows)),
                }
            )
    return pd.DataFrame(rows)


def _load_run_artifacts(run_id: str, run_dir: Path) -> RunArtifacts:
    context = sentiment_benchmark.load_benchmark_context(run_dir)
    pair_predictions = _load_pair_predictions(run_dir)
    _hydrate_reviews_from_pair_predictions(context, pair_predictions)
    discovery_by_product = freeze_final_results._load_discovery_from_traced(run_dir)
    evaluated_pairs = _build_pair_evaluations(context, pair_predictions, discovery_by_product)
    total_gold_pairs = int(sum(len(review.true_labels) for review in context.reviews))
    evaluated_pairs = enrich_pair_rows(evaluated_pairs, total_gold_pairs=total_gold_pairs)
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        context=context,
        discovery_by_product=discovery_by_product,
        pair_predictions=pair_predictions,
        evaluated_pairs=evaluated_pairs,
        total_gold_pairs=total_gold_pairs,
    )


def _scope_rows(
    v1_df: pd.DataFrame,
    v2_df: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    v1_keys = set(zip(v1_df["review_id"], v1_df["gold_label"], strict=True))
    v2_keys = set(zip(v2_df["review_id"], v2_df["gold_label"], strict=True))
    common_keys = v1_keys & v2_keys

    def _mask(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [(review_id, gold_label) in common_keys for review_id, gold_label in zip(df["review_id"], df["gold_label"], strict=True)],
            index=df.index,
        )

    return {
        SCOPE_OWN: (v1_df.copy(), v2_df.copy()),
        SCOPE_COMMON: (v1_df[_mask(v1_df)].copy(), v2_df[_mask(v2_df)].copy()),
    }


def _metrics_rows_for_scope(
    scope: str,
    v1_df: pd.DataFrame,
    v2_df: pd.DataFrame,
    *,
    total_gold_pairs: int,
) -> list[dict[str, Any]]:
    v1_metrics = compute_run_metrics(v1_df, total_gold_pairs=total_gold_pairs)
    v2_metrics = compute_run_metrics(v2_df, total_gold_pairs=total_gold_pairs)
    delta = _delta_dict(v1_metrics, v2_metrics)
    rows = []
    for run_id, payload in ((RUN_ID_V1, v1_metrics), (RUN_ID_V2, v2_metrics), (RUN_ID_DELTA, delta)):
        row = {"scope": scope, "run_id": run_id}
        row.update(payload)
        rows.append(row)
    return rows


def _group_metrics(
    df: pd.DataFrame,
    *,
    group_columns: list[str],
    scope: str,
    run_id: str,
    total_gold_pairs: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["scope", "run_id", *group_columns, "n_pairs", "mae", "accuracy_at_1", "direction_accuracy", "wrong_polarity_rate", "mean_signed_error"])

    rows: list[dict[str, Any]] = []
    for group_values, group in df.groupby(group_columns, sort=True):
        values = group_values if isinstance(group_values, tuple) else (group_values,)
        metrics = compute_run_metrics(group, total_gold_pairs=total_gold_pairs)
        row = {"scope": scope, "run_id": run_id}
        row.update(dict(zip(group_columns, values, strict=True)))
        row["n_pairs"] = int(len(group))
        row["mae"] = metrics["mae"]
        row["accuracy_at_1"] = metrics["accuracy_at_1_0"]
        row["direction_accuracy"] = metrics["direction_accuracy"]
        row["wrong_polarity_rate"] = metrics["wrong_polarity_rate"]
        row["mean_signed_error"] = metrics["mean_signed_error"]
        rows.append(row)
    return pd.DataFrame(rows)


def _hard_cases_top30(df: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["run_id", *df.columns.tolist()])
    out = df.sort_values(["abs_error", "review_id", "gold_label"], ascending=[False, True, True]).head(30).copy()
    out.insert(0, "run_id", run_id)
    return out


def _hard_case_breakdown(df: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["run_id", "dimension", "value", "count", "mean_abs_error_top30", "max_abs_error"])

    rows: list[dict[str, Any]] = []
    mean_abs_error_top30 = float(df["abs_error"].mean())
    max_abs_error = float(df["abs_error"].max())
    for dimension in ("aspect_source", "category_id", "nm_id", "error_type"):
        counts = df[dimension].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "run_id": run_id,
                    "dimension": dimension,
                    "value": value,
                    "count": int(count),
                    "mean_abs_error_top30": mean_abs_error_top30,
                    "max_abs_error": max_abs_error,
                }
            )
    return pd.DataFrame(rows)


def _confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame(0, index=CONFUSION_LABELS, columns=CONFUSION_LABELS)
    if df.empty:
        base.index.name = "gold_rating"
        base.columns.name = "predicted_rating"
        return base
    matrix = pd.crosstab(df["gold_rating"].astype(int), df["pred_round"].astype(int))
    matrix = matrix.reindex(index=CONFUSION_LABELS, columns=CONFUSION_LABELS, fill_value=0)
    matrix.index.name = "gold_rating"
    matrix.columns.name = "predicted_rating"
    return matrix


def _pair_comparison(v1_df: pd.DataFrame, v2_df: pd.DataFrame) -> pd.DataFrame:
    shared_columns = PAIR_KEY_COLUMNS + [
        "review_rating",
        "review_text",
    ]
    select_columns = shared_columns + [
        "predicted_rating",
        "raw_predicted_rating",
        "abs_error",
        "signed_error",
        "aspect_source",
        "n_predicted_items",
        "predicted_keys_json",
        "predicted_aspect_names_json",
        "premise_texts_json",
        "negation_patterns_json",
        "negation_corrected",
        "gold_direction",
        "pred_direction",
        "wrong_polarity",
        "strong_wrong_polarity",
        "too_low",
        "too_high",
        "large_too_low",
        "large_too_high",
        "error_type",
    ]
    left = v1_df[select_columns].rename(columns={column: f"{column}_v1" for column in select_columns if column not in shared_columns})
    right = v2_df[select_columns].rename(columns={column: f"{column}_v2" for column in select_columns if column not in shared_columns})
    merged = left.merge(right, on=shared_columns, how="outer", indicator=True)
    merged["present_in_v1"] = merged["_merge"].isin(["left_only", "both"])
    merged["present_in_v2"] = merged["_merge"].isin(["right_only", "both"])
    merged["delta_predicted_rating_v2_minus_v1"] = merged["predicted_rating_v2"] - merged["predicted_rating_v1"]
    merged["delta_abs_error_v2_minus_v1"] = merged["abs_error_v2"] - merged["abs_error_v1"]
    return merged.drop(columns=["_merge"]).sort_values(["nm_id", "review_id", "gold_label"]).reset_index(drop=True)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def _write_matrix(path: Path, matrix: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, encoding="utf-8")


def _summary_markdown(
    *,
    out_dir: Path,
    own_metrics: pd.DataFrame,
    common_metrics: pd.DataFrame,
    by_product: pd.DataFrame,
    hard_breakdown: pd.DataFrame,
) -> str:
    own = own_metrics[own_metrics["run_id"].isin([RUN_ID_V1, RUN_ID_V2])].set_index("run_id")
    common = common_metrics[common_metrics["run_id"].isin([RUN_ID_V1, RUN_ID_V2])].set_index("run_id")
    top_product_rows = by_product[(by_product["scope"] == SCOPE_OWN) & (by_product["run_id"] == RUN_ID_V2)].sort_values("mae", ascending=False).head(5)
    hard_v1 = hard_breakdown[(hard_breakdown["run_id"] == RUN_ID_V1) & (hard_breakdown["dimension"] == "aspect_source")]
    hard_v2 = hard_breakdown[(hard_breakdown["run_id"] == RUN_ID_V2) & (hard_breakdown["dimension"] == "aspect_source")]
    lines = [
        "# Final V1 vs V2 Sentiment Diagnostics",
        "",
        f"- out_dir: {out_dir}",
        "",
        "## Own passed pairs",
        f"- v1 mae: {own.at[RUN_ID_V1, 'mae']:.4f}",
        f"- v2 mae: {own.at[RUN_ID_V2, 'mae']:.4f}",
        f"- v1 accuracy@1.0: {own.at[RUN_ID_V1, 'accuracy_at_1_0']:.4f}",
        f"- v2 accuracy@1.0: {own.at[RUN_ID_V2, 'accuracy_at_1_0']:.4f}",
        f"- v1 wrong_polarity_rate: {own.at[RUN_ID_V1, 'wrong_polarity_rate']:.4f}",
        f"- v2 wrong_polarity_rate: {own.at[RUN_ID_V2, 'wrong_polarity_rate']:.4f}",
        f"- v1 coverage: {own.at[RUN_ID_V1, 'coverage']:.4f}",
        f"- v2 coverage: {own.at[RUN_ID_V2, 'coverage']:.4f}",
        "",
        "## Common pairs",
        f"- common v1 mae: {common.at[RUN_ID_V1, 'mae']:.4f}",
        f"- common v2 mae: {common.at[RUN_ID_V2, 'mae']:.4f}",
        f"- common v1 accuracy@1.0: {common.at[RUN_ID_V1, 'accuracy_at_1_0']:.4f}",
        f"- common v2 accuracy@1.0: {common.at[RUN_ID_V2, 'accuracy_at_1_0']:.4f}",
        "",
        "## Worst V2 products by own-pair MAE",
    ]
    for row in top_product_rows.itertuples(index=False):
        lines.append(f"- nm_id={row.nm_id} category={row.category_id} n_pairs={row.n_pairs} mae={row.mae:.4f}")
    lines.extend(["", "## Hard cases source mix"])
    for source_df in (hard_v1, hard_v2):
        if source_df.empty:
            continue
        run_id = source_df["run_id"].iloc[0]
        payload = ", ".join(f"{row.value}={row.count}" for row in source_df.itertuples(index=False))
        lines.append(f"- {run_id}: {payload}")
    return "\n".join(lines) + "\n"


def run_comparison(
    *,
    run_v1_dir: Path,
    run_v2_dir: Path,
    out_root: Path,
) -> Path:
    started = datetime.now(timezone.utc)
    artifacts_v1 = _load_run_artifacts(RUN_ID_V1, run_v1_dir.resolve())
    artifacts_v2 = _load_run_artifacts(RUN_ID_V2, run_v2_dir.resolve())
    if artifacts_v1.total_gold_pairs != artifacts_v2.total_gold_pairs:
        raise ValueError("total gold pair count mismatch between runs")

    timestamp = started.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root.resolve() / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    scope_frames = _scope_rows(artifacts_v1.evaluated_pairs, artifacts_v2.evaluated_pairs)
    comparison_rows: list[dict[str, Any]] = []
    by_product_frames: list[pd.DataFrame] = []
    by_category_frames: list[pd.DataFrame] = []
    hard_top_frames: list[pd.DataFrame] = []
    hard_breakdown_frames: list[pd.DataFrame] = []

    for scope, (v1_df, v2_df) in scope_frames.items():
        comparison_rows.extend(
            _metrics_rows_for_scope(
                scope,
                v1_df,
                v2_df,
                total_gold_pairs=artifacts_v1.total_gold_pairs,
            )
        )
        by_product_frames.append(
            _group_metrics(
                v1_df,
                group_columns=["nm_id", "category_id"],
                scope=scope,
                run_id=RUN_ID_V1,
                total_gold_pairs=artifacts_v1.total_gold_pairs,
            )
        )
        by_product_frames.append(
            _group_metrics(
                v2_df,
                group_columns=["nm_id", "category_id"],
                scope=scope,
                run_id=RUN_ID_V2,
                total_gold_pairs=artifacts_v1.total_gold_pairs,
            )
        )
        by_category_frames.append(
            _group_metrics(
                v1_df,
                group_columns=["category_id"],
                scope=scope,
                run_id=RUN_ID_V1,
                total_gold_pairs=artifacts_v1.total_gold_pairs,
            )
        )
        by_category_frames.append(
            _group_metrics(
                v2_df,
                group_columns=["category_id"],
                scope=scope,
                run_id=RUN_ID_V2,
                total_gold_pairs=artifacts_v1.total_gold_pairs,
            )
        )
        _write_matrix(out_dir / f"confusion_matrix_5x5_{RUN_ID_V1}_{scope}.csv", _confusion_matrix(v1_df))
        _write_matrix(out_dir / f"confusion_matrix_5x5_{RUN_ID_V2}_{scope}.csv", _confusion_matrix(v2_df))

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df[["scope", "run_id", *RUN_METRIC_COLUMNS]]

    by_product_df = pd.concat(by_product_frames, ignore_index=True) if by_product_frames else pd.DataFrame(columns=BY_PRODUCT_COLUMNS)
    by_product_df = by_product_df[BY_PRODUCT_COLUMNS]
    by_category_df = pd.concat(by_category_frames, ignore_index=True) if by_category_frames else pd.DataFrame(columns=["scope", "run_id", "category_id", "n_pairs", "mae", "accuracy_at_1", "direction_accuracy", "wrong_polarity_rate", "mean_signed_error"])

    own_hard_v1 = _hard_cases_top30(scope_frames[SCOPE_OWN][0], run_id=RUN_ID_V1)
    own_hard_v2 = _hard_cases_top30(scope_frames[SCOPE_OWN][1], run_id=RUN_ID_V2)
    hard_top_df = pd.concat([own_hard_v1, own_hard_v2], ignore_index=True)
    hard_breakdown_df = pd.concat(
        [
            _hard_case_breakdown(own_hard_v1, run_id=RUN_ID_V1),
            _hard_case_breakdown(own_hard_v2, run_id=RUN_ID_V2),
        ],
        ignore_index=True,
    )

    pair_comparison_df = _pair_comparison(scope_frames[SCOPE_OWN][0], scope_frames[SCOPE_OWN][1])
    nm_diagnostic_df = pair_comparison_df[pair_comparison_df["nm_id"] == TARGET_NM_ID].copy()

    _write_csv(out_dir / "sentiment_metrics_comparison.csv", comparison_df)
    _write_csv(out_dir / "sentiment_metrics_by_product.csv", by_product_df)
    _write_csv(out_dir / "sentiment_metrics_by_category.csv", by_category_df)
    _write_csv(out_dir / "sentiment_hard_cases_top30.csv", hard_top_df)
    _write_csv(out_dir / "sentiment_hard_cases_summary.csv", hard_breakdown_df)
    _write_csv(out_dir / "sentiment_pair_comparison.csv", pair_comparison_df)
    _write_csv(out_dir / f"sentiment_nm_{TARGET_NM_ID}_pair_diagnostics.csv", nm_diagnostic_df)
    _write_csv(out_dir / f"{RUN_ID_V1}_pair_rows.csv", scope_frames[SCOPE_OWN][0])
    _write_csv(out_dir / f"{RUN_ID_V2}_pair_rows.csv", scope_frames[SCOPE_OWN][1])
    _write_csv(out_dir / f"{RUN_ID_V1}_common_pair_rows.csv", scope_frames[SCOPE_COMMON][0])
    _write_csv(out_dir / f"{RUN_ID_V2}_common_pair_rows.csv", scope_frames[SCOPE_COMMON][1])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_v1_dir": str(run_v1_dir.resolve()),
        "run_v2_dir": str(run_v2_dir.resolve()),
        "total_gold_pairs": artifacts_v1.total_gold_pairs,
        "rows_v1_own": int(len(scope_frames[SCOPE_OWN][0])),
        "rows_v2_own": int(len(scope_frames[SCOPE_OWN][1])),
        "rows_common": int(len(scope_frames[SCOPE_COMMON][0])),
        "target_nm_id": TARGET_NM_ID,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        _summary_markdown(
            out_dir=out_dir,
            own_metrics=comparison_df[comparison_df["scope"] == SCOPE_OWN],
            common_metrics=comparison_df[comparison_df["scope"] == SCOPE_COMMON],
            by_product=by_product_df,
            hard_breakdown=hard_breakdown_df,
        ),
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute detailed sentiment diagnostics for final_res_v1 vs final_res_v2")
    parser.add_argument("--run-v1-dir", default=str(RUN_V1_DEFAULT))
    parser.add_argument("--run-v2-dir", default=str(RUN_V2_DEFAULT))
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT_DEFAULT))
    args = parser.parse_args()

    out_dir = run_comparison(
        run_v1_dir=Path(args.run_v1_dir),
        run_v2_dir=Path(args.run_v2_dir),
        out_root=Path(args.out_root),
    )
    print(out_dir)


if __name__ == "__main__":
    main()
