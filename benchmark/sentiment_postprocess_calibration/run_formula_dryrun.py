from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from formulas import build_formula_availability, build_formula_specs
from metrics import add_error_columns, compute_pair_metrics, compute_product_aggregate_metrics, slice_metric_rows
from report import write_dryrun_summary

DEFAULT_OUTPUT_ROOT = THIS_DIR / "results"


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


def run_dry_formulas(results_dir: Path, *, dataset_filename: str = "") -> None:
    resolved_dataset_filename = _resolve_dataset_filename(results_dir, dataset_filename)
    dataset = pd.read_csv(results_dir / resolved_dataset_filename)
    eval_df = dataset[dataset["gold_rating"].notna()].copy()

    availability_df = pd.DataFrame(build_formula_availability(eval_df))
    predictions_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []
    by_source_frames: list[pd.DataFrame] = []
    by_category_frames: list[pd.DataFrame] = []
    by_product_frames: list[pd.DataFrame] = []

    available_map = {
        row["formula_name"]: bool(row["available"])
        for row in availability_df.to_dict(orient="records")
    }

    for spec in build_formula_specs():
        if not available_map.get(spec.name, False):
            continue
        frame = eval_df.copy()
        frame["formula_name"] = spec.name
        frame["pred_rating"] = frame.apply(spec.fn, axis=1)
        frame = add_error_columns(frame, pred_col="pred_rating", gold_col="gold_rating")
        predictions_frames.append(frame)

        overall = {
            "slice_type": "overall",
            "slice_value": "overall",
            "formula_name": spec.name,
        }
        overall.update(compute_pair_metrics(frame))
        overall.update(compute_product_aggregate_metrics(frame))
        metrics_rows.append(overall)

        by_source_frames.append(slice_metric_rows(frame, slice_type="aspect_source", group_col="aspect_source", formula_name=spec.name))
        by_category_frames.append(slice_metric_rows(frame, slice_type="category_id", group_col="category_id", formula_name=spec.name))
        by_product_frames.append(slice_metric_rows(frame, slice_type="nm_id", group_col="nm_id", formula_name=spec.name))

    predictions_df = pd.concat(predictions_frames, ignore_index=True) if predictions_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["mae", "formula_name"], ascending=[True, True]).reset_index(drop=True)
    by_source_df = pd.concat(by_source_frames, ignore_index=True) if by_source_frames else pd.DataFrame()
    by_category_df = pd.concat(by_category_frames, ignore_index=True) if by_category_frames else pd.DataFrame()
    by_product_df = pd.concat(by_product_frames, ignore_index=True) if by_product_frames else pd.DataFrame()

    metrics_df.to_csv(results_dir / "nli_formula_dryrun_metrics.csv", index=False, encoding="utf-8")
    by_source_df.to_csv(results_dir / "nli_formula_dryrun_by_source.csv", index=False, encoding="utf-8")
    by_category_df.to_csv(results_dir / "nli_formula_dryrun_by_category.csv", index=False, encoding="utf-8")
    by_product_df.to_csv(results_dir / "nli_formula_dryrun_by_product.csv", index=False, encoding="utf-8")
    predictions_df.to_csv(results_dir / "nli_formula_dryrun_predictions.csv", index=False, encoding="utf-8")
    write_dryrun_summary(
        results_dir / "nli_formula_dryrun_summary.md",
        metrics_df=metrics_df,
        availability_df=availability_df,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formula dry-run on calibration dataset.")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-filename", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _latest_results_dir(Path(args.output_root))
    dataset_filename = _resolve_dataset_filename(results_dir.resolve(), args.dataset_filename)
    dataset_path = results_dir.resolve() / dataset_filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")
    run_dry_formulas(results_dir.resolve(), dataset_filename=dataset_filename)
    print(results_dir.resolve())


if __name__ == "__main__":
    main()
