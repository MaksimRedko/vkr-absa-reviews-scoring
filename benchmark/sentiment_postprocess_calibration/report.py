from __future__ import annotations

from pathlib import Path

import pandas as pd


def _as_markdown_table(df: pd.DataFrame, *, max_rows: int = 10) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(max_rows).copy()
    headers = [str(col) for col in view.columns]
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[col]) for col in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_dryrun_summary(
    out_path: Path,
    *,
    metrics_df: pd.DataFrame,
    availability_df: pd.DataFrame,
) -> None:
    best = metrics_df.nsmallest(5, "mae") if not metrics_df.empty and "mae" in metrics_df.columns else pd.DataFrame()
    unavailable = availability_df[~availability_df["available"].astype(bool)].copy() if not availability_df.empty else pd.DataFrame()
    lines = [
        "# NLI Formula Dry-Run Summary",
        "",
        f"- total_formulas: {int(len(availability_df))}",
        f"- available_formulas: {int(availability_df['available'].sum()) if not availability_df.empty else 0}",
        f"- evaluated_rows: {int(len(metrics_df))}",
        "",
        "## Top by MAE",
        "",
        _as_markdown_table(best),
        "",
        "## Unavailable Formulas",
        "",
        _as_markdown_table(unavailable[["formula_name", "reason"]] if not unavailable.empty else unavailable),
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_supervised_summary(
    out_path: Path,
    *,
    metrics_df: pd.DataFrame,
    feature_manifest: dict[str, list[str]],
) -> None:
    best = metrics_df.nsmallest(8, "mae") if not metrics_df.empty and "mae" in metrics_df.columns else pd.DataFrame()
    manifest_lines = [f"- `{name}`: {', '.join(features)}" for name, features in feature_manifest.items()]
    lines = [
        "# Supervised LOPO Calibration Summary",
        "",
        f"- evaluated_models: {int(len(metrics_df))}",
        "",
        "## Feature Sets",
        "",
        *manifest_lines,
        "",
        "## Top by MAE",
        "",
        _as_markdown_table(best),
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    out_path: Path,
    *,
    dryrun_metrics_path: Path,
    supervised_metrics_path: Path,
) -> None:
    dryrun_df = pd.read_csv(dryrun_metrics_path) if dryrun_metrics_path.exists() else pd.DataFrame()
    supervised_df = pd.read_csv(supervised_metrics_path) if supervised_metrics_path.exists() else pd.DataFrame()
    dryrun_best = dryrun_df.nsmallest(3, "mae") if not dryrun_df.empty and "mae" in dryrun_df.columns else pd.DataFrame()
    supervised_best = supervised_df.nsmallest(3, "mae") if not supervised_df.empty and "mae" in supervised_df.columns else pd.DataFrame()
    lines = [
        "# Sentiment Postprocess Calibration Final Report",
        "",
        "## Best Dry-Run Formulas",
        "",
        _as_markdown_table(dryrun_best),
        "",
        "## Best Supervised Models",
        "",
        _as_markdown_table(supervised_best),
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

