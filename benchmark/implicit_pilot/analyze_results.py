from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import dump_json, load_json, resolve_repo_path  # noqa: E402

try:
    from scipy.stats import mannwhitneyu as scipy_mannwhitneyu
except ImportError:  # pragma: no cover
    scipy_mannwhitneyu = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


PRIMARY_SCORE = "score_p_ent"


def _score_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in df.columns if column.startswith("score_")]


def _binary_auc(frame: pd.DataFrame, score_col: str, positive_classes: Iterable[str]) -> float | None:
    positive_set = set(positive_classes)
    subset = frame[frame["class"].isin(positive_set | {"UNRELATED"})].copy()
    if subset.empty:
        return None
    y_true = subset["class"].isin(positive_set).astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, subset[score_col].to_numpy()))


def _bootstrap_auc_ci(
    frame: pd.DataFrame,
    score_col: str,
    positive_classes: Iterable[str],
    iterations: int,
    seed: int,
) -> Dict[str, float | int | None]:
    positive_set = set(positive_classes)
    subset = frame[frame["class"].isin(positive_set | {"UNRELATED"})].copy()
    if subset.empty:
        return {"auc": None, "ci_low": None, "ci_high": None, "effective_iterations": 0}

    y_true = subset["class"].isin(positive_set).astype(int).to_numpy()
    scores = subset[score_col].to_numpy()
    if len(np.unique(y_true)) < 2:
        return {"auc": None, "ci_low": None, "ci_high": None, "effective_iterations": 0}

    auc = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(subset), len(subset))
        sampled_y = y_true[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_scores = scores[indices]
        boot.append(float(roc_auc_score(sampled_y, sampled_scores)))
    if not boot:
        return {"auc": auc, "ci_low": None, "ci_high": None, "effective_iterations": 0}
    return {
        "auc": auc,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "effective_iterations": int(len(boot)),
    }


def _mann_whitney_fallback(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([x, y])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    idx = 0
    while idx < len(combined):
        start = idx
        value = combined[order[idx]]
        while idx < len(combined) and combined[order[idx]] == value:
            idx += 1
        avg_rank = (start + idx + 1) / 2.0
        ranks[order[start:idx]] = avg_rank

    x_ranks = ranks[: len(x)]
    u1 = x_ranks.sum() - len(x) * (len(x) + 1) / 2.0
    mu = len(x) * len(y) / 2.0
    sigma = math.sqrt(len(x) * len(y) * (len(x) + len(y) + 1) / 12.0)
    if sigma == 0:
        return float(u1), 1.0
    z = (u1 - mu) / sigma
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return float(u1), float(p_value)


def _mann_whitney(frame: pd.DataFrame, score_col: str) -> Dict[str, float | int | None]:
    implicit_scores = frame.loc[frame["class"] == "IMPLICIT", score_col].to_numpy(dtype=float)
    unrelated_scores = frame.loc[frame["class"] == "UNRELATED", score_col].to_numpy(dtype=float)
    if len(implicit_scores) == 0 or len(unrelated_scores) == 0:
        return {"u_stat": None, "p_value": None, "n_implicit": int(len(implicit_scores)), "n_unrelated": int(len(unrelated_scores))}
    if scipy_mannwhitneyu is not None:
        stat = scipy_mannwhitneyu(implicit_scores, unrelated_scores, alternative="two-sided")
        return {
            "u_stat": float(stat.statistic),
            "p_value": float(stat.pvalue),
            "n_implicit": int(len(implicit_scores)),
            "n_unrelated": int(len(unrelated_scores)),
        }
    u_stat, p_value = _mann_whitney_fallback(implicit_scores, unrelated_scores)
    return {
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "n_implicit": int(len(implicit_scores)),
        "n_unrelated": int(len(unrelated_scores)),
    }


def _distribution_summary(frame: pd.DataFrame, score_col: str) -> Dict[str, Dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for cls_name, cls_frame in frame.groupby("class"):
        values = cls_frame[score_col].to_numpy(dtype=float)
        summary[str(cls_name)] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)) if len(values) else math.nan,
            "median": float(np.median(values)) if len(values) else math.nan,
            "std": float(np.std(values)) if len(values) else math.nan,
        }
    return summary


def _evaluate_subset(
    frame: pd.DataFrame,
    score_col: str,
    bootstrap_iterations: int,
    seed: int,
    compute_bootstrap: bool,
) -> Dict[str, Any]:
    if compute_bootstrap:
        implicit_ci = _bootstrap_auc_ci(
            frame=frame,
            score_col=score_col,
            positive_classes=["IMPLICIT"],
            iterations=bootstrap_iterations,
            seed=seed,
        )
        mention_ci = _bootstrap_auc_ci(
            frame=frame,
            score_col=score_col,
            positive_classes=["IMPLICIT", "EXPLICIT"],
            iterations=bootstrap_iterations,
            seed=seed + 1,
        )
    else:
        implicit_ci = {
            "auc": _binary_auc(frame, score_col, ["IMPLICIT"]),
            "ci_low": None,
            "ci_high": None,
            "effective_iterations": 0,
        }
        mention_ci = {
            "auc": _binary_auc(frame, score_col, ["IMPLICIT", "EXPLICIT"]),
            "ci_low": None,
            "ci_high": None,
            "effective_iterations": 0,
        }
    mann_whitney = _mann_whitney(frame, score_col)
    return {
        "score_col": score_col,
        "auc_implicit_vs_unrelated": implicit_ci["auc"],
        "auc_implicit_vs_unrelated_ci_low": implicit_ci["ci_low"],
        "auc_implicit_vs_unrelated_ci_high": implicit_ci["ci_high"],
        "auc_implicit_vs_unrelated_bootstrap_n": implicit_ci["effective_iterations"],
        "auc_mention_vs_unrelated": mention_ci["auc"],
        "auc_mention_vs_unrelated_ci_low": mention_ci["ci_low"],
        "auc_mention_vs_unrelated_ci_high": mention_ci["ci_high"],
        "auc_mention_vs_unrelated_bootstrap_n": mention_ci["effective_iterations"],
        "mann_whitney_u": mann_whitney["u_stat"],
        "mann_whitney_p_value": mann_whitney["p_value"],
        "n_implicit": mann_whitney["n_implicit"],
        "n_unrelated": mann_whitney["n_unrelated"],
        "distribution_summary": _distribution_summary(frame, score_col),
    }


def _go_no_go_label(best_auc: float | None) -> str:
    if best_auc is None:
        return "insufficient_data"
    if best_auc > 0.80:
        return "go"
    if best_auc > 0.65:
        return "caveat"
    return "no_go"


def _save_boxplot(frame: pd.DataFrame, score_col: str, output_path: Path) -> bool:
    if plt is None:
        return False
    ordered_classes = ["EXPLICIT", "IMPLICIT", "UNRELATED"]
    series = [
        frame.loc[frame["class"] == cls_name, score_col].to_numpy(dtype=float)
        for cls_name in ordered_classes
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(series, tick_labels=ordered_classes)
    ax.set_ylabel(score_col)
    ax.set_title("Implicit pilot distributions")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def analyze_results(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    analysis_cfg = dict(config.get("analysis", {}))
    results_path = run_dir / "inference_results.csv"
    results_df = pd.read_csv(results_path, dtype={"review_id": str})
    score_columns = _score_columns(results_df)
    bootstrap_iterations = int(analysis_cfg.get("bootstrap_iterations", 1000))
    seed = int(analysis_cfg.get("seed", 42))
    hard_cases_per_group = int(analysis_cfg.get("hard_cases_per_group", 10))

    metrics_rows: list[Dict[str, Any]] = []
    for score_col in score_columns:
        compute_bootstrap = score_col == PRIMARY_SCORE
        for (granularity, hypothesis_id), combo_frame in results_df.groupby(["granularity", "hypothesis_id"]):
            overall = _evaluate_subset(
                combo_frame,
                score_col,
                bootstrap_iterations,
                seed,
                compute_bootstrap=compute_bootstrap,
            )
            metrics_rows.append(
                {
                    "scope": "overall",
                    "aspect_type": "overall",
                    "granularity": str(granularity),
                    "hypothesis_id": str(hypothesis_id),
                    **overall,
                }
            )
            for aspect_type, aspect_frame in combo_frame.groupby("aspect_type"):
                scoped = _evaluate_subset(
                    aspect_frame,
                    score_col,
                    bootstrap_iterations,
                    seed,
                    compute_bootstrap=compute_bootstrap,
                )
                metrics_rows.append(
                    {
                        "scope": "aspect_type",
                        "aspect_type": str(aspect_type),
                        "granularity": str(granularity),
                        "hypothesis_id": str(hypothesis_id),
                        **scoped,
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "analysis_metrics.csv", index=False, encoding="utf-8")

    primary_overall = metrics_df[
        (metrics_df["score_col"] == PRIMARY_SCORE) & (metrics_df["scope"] == "overall")
    ].copy()
    primary_overall = primary_overall.sort_values(
        by="auc_implicit_vs_unrelated",
        ascending=False,
        na_position="last",
    )
    best_primary = primary_overall.iloc[0].to_dict() if not primary_overall.empty else {}

    all_overall = metrics_df[metrics_df["scope"] == "overall"].copy()
    all_overall = all_overall.sort_values(
        by="auc_implicit_vs_unrelated",
        ascending=False,
        na_position="last",
    )
    best_overall = all_overall.iloc[0].to_dict() if not all_overall.empty else {}

    score_variant_rows: list[Dict[str, Any]] = []
    for score_col, score_frame in all_overall.groupby("score_col"):
        best_row = score_frame.iloc[0].to_dict()
        score_variant_rows.append(
            {
                "score_col": str(score_col),
                "best_granularity": str(best_row["granularity"]),
                "best_hypothesis_id": str(best_row["hypothesis_id"]),
                "best_auc_implicit_vs_unrelated": best_row["auc_implicit_vs_unrelated"],
                "best_auc_mention_vs_unrelated": best_row["auc_mention_vs_unrelated"],
            }
        )
    pd.DataFrame(score_variant_rows).to_csv(
        run_dir / "score_variant_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    auc_matrix = primary_overall[
        [
            "granularity",
            "hypothesis_id",
            "auc_implicit_vs_unrelated",
            "auc_implicit_vs_unrelated_ci_low",
            "auc_implicit_vs_unrelated_ci_high",
            "auc_mention_vs_unrelated",
            "auc_mention_vs_unrelated_ci_low",
            "auc_mention_vs_unrelated_ci_high",
            "mann_whitney_p_value",
        ]
    ].copy()
    auc_matrix.to_csv(run_dir / "auc_matrix_primary.csv", index=False, encoding="utf-8")

    aspect_type_matrix = metrics_df[
        (metrics_df["score_col"] == PRIMARY_SCORE) & (metrics_df["scope"] == "aspect_type")
    ][
        [
            "aspect_type",
            "granularity",
            "hypothesis_id",
            "auc_implicit_vs_unrelated",
            "auc_mention_vs_unrelated",
            "mann_whitney_p_value",
        ]
    ].copy()
    aspect_type_matrix.to_csv(run_dir / "auc_by_aspect_type_primary.csv", index=False, encoding="utf-8")

    best_frame = pd.DataFrame()
    boxplot_written = False
    hard_cases_path = run_dir / "hard_cases.csv"
    if best_primary:
        best_frame = results_df[
            (results_df["granularity"] == best_primary["granularity"])
            & (results_df["hypothesis_id"] == best_primary["hypothesis_id"])
        ].copy()
        boxplot_written = _save_boxplot(best_frame, PRIMARY_SCORE, run_dir / "boxplot_best_primary.png")

        implicit_hard = best_frame[best_frame["class"] == "IMPLICIT"].nsmallest(
            hard_cases_per_group, PRIMARY_SCORE
        ).copy()
        implicit_hard["hard_case_group"] = "implicit_low_p_ent"
        unrelated_hard = best_frame[best_frame["class"] == "UNRELATED"].nlargest(
            hard_cases_per_group, PRIMARY_SCORE
        ).copy()
        unrelated_hard["hard_case_group"] = "unrelated_high_p_ent"
        hard_cases = pd.concat([implicit_hard, unrelated_hard], ignore_index=True)
        hard_cases.to_csv(hard_cases_path, index=False, encoding="utf-8")

    summary = {
        "results_path": str(results_path),
        "best_primary": best_primary,
        "best_overall": best_overall,
        "best_primary_go_no_go": _go_no_go_label(best_primary.get("auc_implicit_vs_unrelated")),
        "best_overall_go_no_go": _go_no_go_label(best_overall.get("auc_implicit_vs_unrelated")),
        "boxplot_written": bool(boxplot_written),
        "hard_cases_path": str(hard_cases_path) if hard_cases_path.exists() else None,
    }
    dump_json(run_dir / "analysis_summary.json", summary)

    summary_md = run_dir / "pilot_summary.md"
    best_aspect_rows = pd.DataFrame()
    if best_primary:
        best_aspect_rows = aspect_type_matrix[
            (aspect_type_matrix["granularity"] == best_primary["granularity"])
            & (aspect_type_matrix["hypothesis_id"] == best_primary["hypothesis_id"])
        ].sort_values(by="auc_implicit_vs_unrelated", ascending=False)

    hard_case_preview: list[str] = []
    if hard_cases_path.exists():
        hard_cases_df = pd.read_csv(hard_cases_path, dtype={"review_id": str})
        for _, row in hard_cases_df.head(6).iterrows():
            hard_case_preview.append(
                f"- `{row['hard_case_group']}` | `{row['aspect_id']}` | `{row['p_ent']:.4f}` | {row['premise_text']}"
            )

    summary_lines = [
        "# Implicit NLI Pilot Summary",
        "",
        "## Decision",
        f"- Best primary setting: `{best_primary.get('granularity')}` + `{best_primary.get('hypothesis_id')}`",
        f"- AUC implicit vs unrelated: `{best_primary.get('auc_implicit_vs_unrelated')}`",
        f"- 95% CI: `[{best_primary.get('auc_implicit_vs_unrelated_ci_low')}, {best_primary.get('auc_implicit_vs_unrelated_ci_high')}]`",
        f"- AUC mention vs unrelated: `{best_primary.get('auc_mention_vs_unrelated')}`",
        f"- Mann-Whitney p-value: `{best_primary.get('mann_whitney_p_value')}`",
        f"- Go/no-go (primary): `{summary['best_primary_go_no_go']}`",
        "",
        "## AUC Matrix (Primary Score)",
    ]
    for _, row in auc_matrix.iterrows():
        summary_lines.append(
            f"- `{row['granularity']}` + `{row['hypothesis_id']}` -> "
            f"`AUC_imp={row['auc_implicit_vs_unrelated']:.4f}`, "
            f"`AUC_mention={row['auc_mention_vs_unrelated']:.4f}`, "
            f"`p={row['mann_whitney_p_value']:.4g}`"
        )

    summary_lines.extend(["", "## Aspect Types (Best Primary Setting)"])
    for _, row in best_aspect_rows.iterrows():
        summary_lines.append(
            f"- `{row['aspect_type']}` -> "
            f"`AUC_imp={row['auc_implicit_vs_unrelated']:.4f}`, "
            f"`AUC_mention={row['auc_mention_vs_unrelated']:.4f}`, "
            f"`p={row['mann_whitney_p_value']:.4g}`"
        )

    summary_lines.extend(["", "## Score Variants"])
    for _, row in pd.DataFrame(score_variant_rows).sort_values(
        by="best_auc_implicit_vs_unrelated",
        ascending=False,
    ).iterrows():
        summary_lines.append(
            f"- `{row['score_col']}` -> `{row['best_granularity']}` + `{row['best_hypothesis_id']}` "
            f"with `AUC_imp={row['best_auc_implicit_vs_unrelated']:.4f}`"
        )

    summary_lines.extend(["", "## Hard Case Preview"])
    summary_lines.extend(hard_case_preview or ["- No hard cases exported."])
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze implicit pilot outputs.")
    parser.add_argument(
        "--config",
        default="experiments/implicit_nli_pilot.json",
        help="Path to pilot config JSON.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Results directory containing inference_results.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    run_dir = resolve_repo_path(args.run_dir)
    summary = analyze_results(config, run_dir)
    print(f"[implicit_pilot] summary={summary['best_primary_go_no_go']}")


if __name__ == "__main__":
    main()
