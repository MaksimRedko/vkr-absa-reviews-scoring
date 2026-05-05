from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from metrics import add_error_columns, clip_rating, compute_pair_metrics, compute_product_aggregate_metrics, slice_metric_rows
from report import write_final_report, write_supervised_summary

DEFAULT_OUTPUT_ROOT = THIS_DIR / "results"
EPS = 1e-8


def _latest_results_dir(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and (path / "calibration_dataset.csv").exists())
    if not candidates:
        raise FileNotFoundError(f"no calibration dataset dirs under {root}")
    return candidates[-1]


def _resolve_dataset_filename(results_dir: Path, dataset_filename: str) -> str:
    if dataset_filename:
        return dataset_filename
    dual_name = "calibration_dataset_with_dual_nli.csv"
    if (results_dir / dual_name).exists():
        return dual_name
    return "calibration_dataset.csv"


def _build_feature_manifest(df: pd.DataFrame) -> dict[str, list[str]]:
    manifests: dict[str, list[str]] = {}
    available_base: list[str] = []
    for column in (
        "pos_entailment",
        "pos_neutral",
        "pos_contradiction",
        "neg_entailment",
        "neg_neutral",
        "neg_contradiction",
        "current_raw_rating",
        "current_final_rating",
    ):
        if column in df.columns and not bool(df[column].isna().any()):
            available_base.append(column)

    derived_columns: dict[str, pd.Series] = {}
    if {"pos_entailment", "neg_entailment"}.issubset(df.columns) and not bool(df[["pos_entailment", "neg_entailment"]].isna().any().any()):
        derived_columns["diff_entailment"] = df["pos_entailment"] - df["neg_entailment"]
        derived_columns["log_ratio_entailment"] = np.log(df["pos_entailment"] + EPS) - np.log(df["neg_entailment"] + EPS)
        derived_columns["confidence"] = df[["pos_entailment", "neg_entailment"]].max(axis=1)
    if {"pos_neutral", "neg_neutral"}.issubset(df.columns) and not bool(df[["pos_neutral", "neg_neutral"]].isna().any().any()):
        derived_columns["neutral_mean"] = 0.5 * (df["pos_neutral"] + df["neg_neutral"])

    base_without = [*available_base, *derived_columns.keys(), "aspect_source", "category_id"]
    manifests["without_review_rating"] = base_without
    manifests["with_review_rating"] = [*base_without, "review_rating"] if "review_rating" in df.columns and not bool(df["review_rating"].isna().any()) else base_without
    return manifests, derived_columns


def _make_pipeline(model: Any, *, numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return Pipeline([("preprocessor", ColumnTransformer(transformers=transformers)), ("model", model)])


def _models() -> dict[str, Any]:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42),
        "HuberRegressor": HuberRegressor(max_iter=1000),
    }


def run_supervised(results_dir: Path, *, dataset_filename: str = "") -> None:
    resolved_dataset_filename = _resolve_dataset_filename(results_dir, dataset_filename)
    dataset = pd.read_csv(results_dir / resolved_dataset_filename)
    dataset = dataset[dataset["gold_rating"].notna() & dataset["current_final_rating"].notna()].copy()

    feature_manifest, derived_columns = _build_feature_manifest(dataset)
    for name, series in derived_columns.items():
        dataset[name] = series

    prediction_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    by_product_frames: list[pd.DataFrame] = []
    by_category_frames: list[pd.DataFrame] = []

    unique_nm_ids = sorted(dataset["nm_id"].unique().tolist())
    model_map = _models()

    for feature_set, features in feature_manifest.items():
        baseline_frame = dataset.copy()
        baseline_frame["model_id"] = "baseline_current_formula"
        baseline_frame["feature_set"] = feature_set
        baseline_frame["pred_rating"] = baseline_frame["current_final_rating"].astype(float).map(clip_rating)
        baseline_frame["fold_nm_id"] = baseline_frame["nm_id"]
        baseline_frame = add_error_columns(baseline_frame, pred_col="pred_rating", gold_col="gold_rating")
        prediction_frames.append(baseline_frame)

        numeric_columns = [column for column in features if column not in {"aspect_source", "category_id"}]
        categorical_columns = [column for column in features if column in {"aspect_source", "category_id"}]

        for model_id, model in model_map.items():
            fold_frames: list[pd.DataFrame] = []
            for fold_nm_id in unique_nm_ids:
                train_df = dataset[dataset["nm_id"] != fold_nm_id].copy()
                test_df = dataset[dataset["nm_id"] == fold_nm_id].copy()
                if train_df.empty or test_df.empty:
                    continue

                pipeline = _make_pipeline(model, numeric_columns=numeric_columns, categorical_columns=categorical_columns)
                pipeline.fit(train_df[features], train_df["gold_rating"].astype(float))
                pred = np.asarray(pipeline.predict(test_df[features]), dtype=float)
                fold = test_df.copy()
                fold["model_id"] = model_id
                fold["feature_set"] = feature_set
                fold["fold_nm_id"] = fold_nm_id
                fold["pred_rating"] = [clip_rating(value) for value in pred]
                fold_frames.append(fold)

            if not fold_frames:
                continue
            predicted = pd.concat(fold_frames, ignore_index=True)
            predicted = add_error_columns(predicted, pred_col="pred_rating", gold_col="gold_rating")
            prediction_frames.append(predicted)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    for (model_id, feature_set), group in predictions_df.groupby(["model_id", "feature_set"], sort=True):
        overall = {
            "slice_type": "overall",
            "slice_value": "overall",
            "model_id": model_id,
            "feature_set": feature_set,
        }
        overall.update(compute_pair_metrics(group))
        overall.update(compute_product_aggregate_metrics(group))
        metrics_rows.append(overall)
        by_product_frames.append(slice_metric_rows(group, slice_type="nm_id", group_col="nm_id", model_id=model_id, feature_set=feature_set))
        by_category_frames.append(
            slice_metric_rows(group, slice_type="category_id", group_col="category_id", model_id=model_id, feature_set=feature_set)
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["mae", "model_id", "feature_set"], ascending=[True, True, True]).reset_index(drop=True)
    by_product_df = pd.concat(by_product_frames, ignore_index=True) if by_product_frames else pd.DataFrame()
    by_category_df = pd.concat(by_category_frames, ignore_index=True) if by_category_frames else pd.DataFrame()

    metrics_df.to_csv(results_dir / "supervised_calibration_metrics.csv", index=False, encoding="utf-8")
    by_product_df.to_csv(results_dir / "supervised_calibration_by_product.csv", index=False, encoding="utf-8")
    by_category_df.to_csv(results_dir / "supervised_calibration_by_category.csv", index=False, encoding="utf-8")
    predictions_df.to_csv(results_dir / "supervised_calibration_predictions.csv", index=False, encoding="utf-8")

    write_supervised_summary(
        results_dir / "supervised_calibration_summary.md",
        metrics_df=metrics_df,
        feature_manifest=feature_manifest,
    )

    dryrun_metrics = results_dir / "nli_formula_dryrun_metrics.csv"
    if dryrun_metrics.exists():
        write_final_report(
            results_dir / "sentiment_postprocess_final_report.md",
            dryrun_metrics_path=dryrun_metrics,
            supervised_metrics_path=results_dir / "supervised_calibration_metrics.csv",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised LOPO calibrator on calibration dataset.")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-filename", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _latest_results_dir(Path(args.output_root))
    dataset_filename = _resolve_dataset_filename(results_dir.resolve(), args.dataset_filename)
    run_supervised(results_dir.resolve(), dataset_filename=dataset_filename)
    print(results_dir.resolve())


if __name__ == "__main__":
    main()
