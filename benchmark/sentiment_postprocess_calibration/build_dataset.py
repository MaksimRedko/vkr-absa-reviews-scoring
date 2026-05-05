from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from scripts.recompute_manual_audit_metrics import build_audit_base

DEFAULT_DB_PATH = ROOT / "manual_recalc" / "data" / "manual_recalc.sqlite3"
DEFAULT_RUN_DIR = ROOT / "results" / "20260502_171530_traced"
DEFAULT_DATASET_PATH = ROOT / "data" / "dataset_final.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark" / "sentiment_postprocess_calibration" / "results"


def _classify_hypothesis(text: str) -> str:
    lowered = str(text).strip().lower()
    if not lowered:
        return "unknown"
    if "плохо" in lowered or "отриц" in lowered:
        return "neg"
    if "хорошо" in lowered or "положит" in lowered:
        return "pos"
    return "unknown"


def _extract_probability_table(nli_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not {"prediction_id", "p_entailment", "p_neutral", "p_contradiction"}.issubset(nli_df.columns):
        raise ValueError("nli_predictions.parquet does not contain expected three-way probability columns")

    frame = nli_df.copy()
    frame["prediction_id"] = frame["prediction_id"].astype(str)
    frame["hypothesis_polarity"] = frame["hypothesis_text"].map(_classify_hypothesis) if "hypothesis_text" in frame.columns else "unknown"
    frame["hypothesis_polarity"] = frame["hypothesis_polarity"].fillna("unknown")

    row_meta = (
        frame.groupby("prediction_id", dropna=False)
        .agg(
            n_nli_rows=("prediction_id", "size"),
            n_hypothesis_types=("hypothesis_polarity", "nunique"),
        )
        .reset_index()
    )

    prob_rows: list[dict[str, Any]] = []
    unknown_rows = int((frame["hypothesis_polarity"] == "unknown").sum())
    for prediction_id, group in frame.groupby("prediction_id", sort=False):
        row: dict[str, Any] = {
            "prediction_id": str(prediction_id),
            "n_nli_rows": int(len(group)),
        }
        for polarity in ("pos", "neg"):
            subset = group[group["hypothesis_polarity"] == polarity]
            if subset.empty:
                row[f"{polarity}_entailment"] = np.nan
                row[f"{polarity}_neutral"] = np.nan
                row[f"{polarity}_contradiction"] = np.nan
                row[f"{polarity}_hypothesis_text"] = ""
                continue
            top = subset.iloc[0]
            row[f"{polarity}_entailment"] = float(top["p_entailment"])
            row[f"{polarity}_neutral"] = float(top["p_neutral"])
            row[f"{polarity}_contradiction"] = float(top["p_contradiction"])
            row[f"{polarity}_hypothesis_text"] = str(top.get("hypothesis_text", ""))
        if "premise_text" in group.columns:
            row["premise_text"] = str(group.iloc[0].get("premise_text", ""))
        prob_rows.append(row)
    prob_df = pd.DataFrame(prob_rows)
    return prob_df, {
        "unknown_hypothesis_rows": unknown_rows,
        "polarity_counts": frame["hypothesis_polarity"].value_counts(dropna=False).to_dict(),
        "multi_row_prediction_ids": int((row_meta["n_nli_rows"] > 1).sum()),
    }


def build_calibration_dataset(
    *,
    db_path: Path,
    run_dir: Path,
    dataset_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = build_audit_base(db_path=db_path, run_dir=run_dir, dataset_path=dataset_path)
    system = base["system"].copy()
    gold = base["gold"].copy()
    nli = pd.read_parquet(run_dir / "nli_predictions.parquet").copy()
    nli["prediction_id"] = nli["prediction_id"].astype(str)

    tp_mask = (system["manual_decision_norm"] == "TP") & (system["mapped_gold_aspect_norm"] != "")
    tp_pairs = system.loc[tp_mask].copy()
    tp_pairs["prediction_id"] = tp_pairs["prediction_id"].astype(str)

    gold_lookup = gold[["review_id", "gold_aspect", "gold_rating"]].copy()
    gold_lookup["review_id"] = gold_lookup["review_id"].astype(str)
    gold_lookup["gold_aspect"] = gold_lookup["gold_aspect"].astype(str)
    gold_lookup = gold_lookup.rename(columns={"gold_aspect": "mapped_gold_aspect"})

    prob_df, prob_meta = _extract_probability_table(nli)
    merged = tp_pairs.merge(
        gold_lookup,
        left_on=["review_id", "mapped_gold_aspect_norm"],
        right_on=["review_id", "mapped_gold_aspect"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        prob_df,
        on="prediction_id",
        how="left",
        validate="one_to_one",
        indicator="nli_merge_status",
    )

    if "premise_text" not in merged.columns:
        # After merge Pandas may create premise_text_x/premise_text_y.
        for candidate in ("premise_text_x", "premise_text_y"):
            if candidate in merged.columns:
                merged["premise_text"] = merged[candidate]
                break

    renamed = merged.rename(
        columns={
            "mapped_gold_aspect_norm": "mapped_gold_aspect",
            "system_rating": "current_final_rating",
            "raw_rating": "current_raw_rating",
            "system_aspect": "system_aspect",
        }
    )
    for column in (
        "neg_entailment",
        "neg_neutral",
        "neg_contradiction",
        "neg_hypothesis_text",
    ):
        if column not in renamed.columns:
            renamed[column] = np.nan if column != "neg_hypothesis_text" else ""

    dataset = renamed[
        [
            "prediction_id",
            "review_id",
            "nm_id",
            "category_id",
            "aspect_source",
            "system_aspect",
            "mapped_gold_aspect",
            "gold_rating",
            "current_raw_rating",
            "current_final_rating",
            "review_rating",
            "premise_text",
            "pos_hypothesis_text",
            "neg_hypothesis_text",
            "pos_entailment",
            "pos_neutral",
            "pos_contradiction",
            "neg_entailment",
            "neg_neutral",
            "neg_contradiction",
            "n_nli_rows",
            "nli_merge_status",
        ]
    ].copy()
    dataset["nli_merge_status"] = dataset["nli_merge_status"].astype(str)

    duplicates = dataset.duplicated(subset=["prediction_id"], keep=False)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "db_path": str(db_path.resolve()),
        "tp_pairs_found": int(len(tp_pairs)),
        "rows_in_dataset": int(len(dataset)),
        "matched_with_nli": int((dataset["nli_merge_status"] == "both").sum()),
        "lost_without_nli": int((dataset["nli_merge_status"] != "both").sum()),
        "duplicate_prediction_rows": int(duplicates.sum()),
        "missing_gold_rating": int(dataset["gold_rating"].isna().sum()),
        "missing_current_final_rating": int(dataset["current_final_rating"].isna().sum()),
        "missing_field_counts": {column: int(dataset[column].isna().sum()) for column in dataset.columns if dataset[column].dtype != object},
        "probability_metadata": prob_meta,
    }
    return dataset, metadata


def write_validation_report(out_path: Path, *, dataset: pd.DataFrame, metadata: dict[str, Any]) -> None:
    missing_reasons: list[str] = []
    if metadata["lost_without_nli"]:
        missing_reasons.append(f"- no matching NLI row: {metadata['lost_without_nli']}")
    if metadata["missing_gold_rating"]:
        missing_reasons.append(f"- missing gold_rating: {metadata['missing_gold_rating']}")
    if metadata["missing_current_final_rating"]:
        missing_reasons.append(f"- missing current_final_rating: {metadata['missing_current_final_rating']}")
    if not missing_reasons:
        missing_reasons.append("- no row loss after TP + NLI join")

    field_rows = []
    for field in (
        "pos_entailment",
        "pos_neutral",
        "pos_contradiction",
        "neg_entailment",
        "neg_neutral",
        "neg_contradiction",
    ):
        field_rows.append(f"| {field} | {int(dataset[field].notna().sum())} | {int(dataset[field].isna().sum())} |")

    duplicate_sample = dataset[dataset.duplicated(subset=["prediction_id"], keep=False)].head(10)
    lines = [
        "# Dataset Validation Report",
        "",
        f"- generated_at: {metadata['generated_at']}",
        f"- run_dir: {metadata['run_dir']}",
        f"- TP pairs found: {metadata['tp_pairs_found']}",
        f"- rows in calibration dataset: {metadata['rows_in_dataset']}",
        f"- matched with NLI probabilities: {metadata['matched_with_nli']}",
        "",
        "## Lost Rows",
        "",
        *missing_reasons,
        "",
        "## Duplicates / Missing",
        "",
        f"- duplicate prediction_id rows: {metadata['duplicate_prediction_rows']}",
        f"- missing gold_rating: {metadata['missing_gold_rating']}",
        f"- missing current_final_rating: {metadata['missing_current_final_rating']}",
        "",
        "## Probability Fields",
        "",
        "| field | non_null | null |",
        "|---|---:|---:|",
        *field_rows,
        "",
        "## NLI Layout",
        "",
        f"- polarity counts in raw nli_predictions: {metadata['probability_metadata']['polarity_counts']}",
        f"- rows with unknown hypothesis polarity: {metadata['probability_metadata']['unknown_hypothesis_rows']}",
        f"- prediction_ids with multiple NLI rows: {metadata['probability_metadata']['multi_row_prediction_ids']}",
        "",
    ]
    if duplicate_sample.empty:
        lines.extend(["## Duplicate Sample", "", "No duplicate prediction_id rows.", ""])
    else:
        lines.extend(["## Duplicate Sample", "", duplicate_sample.to_markdown(index=False), ""])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calibration dataset for sentiment postprocess experiment.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, metadata = build_calibration_dataset(
        db_path=Path(args.db_path),
        run_dir=Path(args.run_dir),
        dataset_path=Path(args.dataset_path),
    )
    dataset.to_csv(output_dir / "calibration_dataset.csv", index=False, encoding="utf-8")
    write_validation_report(output_dir / "dataset_validation_report.md", dataset=dataset, metadata=metadata)
    print(output_dir)


if __name__ == "__main__":
    main()
