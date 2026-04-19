from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_results import (  # noqa: E402
    _evaluate_subset,
    _go_no_go_label,
)
from common import dump_json, load_json, resolve_repo_path  # noqa: E402


PRIMARY_SCORE = "score_cos"
SCORE_VARIANTS = ["score_cos", "score_cos_contrastive", "score_cos_rank"]


def _load_cosine_results(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "cosine_inference_results.csv", dtype={"review_id": str})


def _metrics_frame(
    results_df: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for score_col in SCORE_VARIANTS:
        for granularity, granularity_frame in results_df.groupby("granularity"):
            overall = _evaluate_subset(
                granularity_frame,
                score_col=score_col,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                compute_bootstrap=True,
            )
            rows.append(
                {
                    "scope": "overall",
                    "aspect_type": "overall",
                    "granularity": str(granularity),
                    "score_variant": score_col,
                    **overall,
                }
            )
            for aspect_type, aspect_frame in granularity_frame.groupby("aspect_type"):
                scoped = _evaluate_subset(
                    aspect_frame,
                    score_col=score_col,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                    compute_bootstrap=True,
                )
                rows.append(
                    {
                        "scope": "aspect_type",
                        "aspect_type": str(aspect_type),
                        "granularity": str(granularity),
                        "score_variant": score_col,
                        **scoped,
                    }
                )
    return pd.DataFrame(rows)


def _load_nli_auc_matrix(nli_run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(nli_run_dir / "auc_matrix_primary.csv")


def _best_rows_by_granularity(matrix: pd.DataFrame, variant_col: str) -> pd.DataFrame:
    ranked = matrix.sort_values(
        by=["granularity", "auc_implicit_vs_unrelated"],
        ascending=[True, False],
        na_position="last",
    )
    return ranked.groupby("granularity", as_index=False).first()


def _write_head_to_head(
    run_dir: Path,
    nli_run_dir: Path,
    cosine_matrix: pd.DataFrame,
) -> None:
    nli_matrix = _load_nli_auc_matrix(nli_run_dir)
    nli_best = _best_rows_by_granularity(nli_matrix, "hypothesis_id")
    cosine_best = _best_rows_by_granularity(cosine_matrix, "score_variant")

    rows: list[Dict[str, Any]] = []
    for _, row in nli_best.iterrows():
        rows.append(
            {
                "granularity": str(row["granularity"]),
                "method": "NLI",
                "best_variant": str(row["hypothesis_id"]),
                "auc_imp_vs_unrel": float(row["auc_implicit_vs_unrelated"]),
                "ci_low": float(row["auc_implicit_vs_unrelated_ci_low"]),
                "ci_high": float(row["auc_implicit_vs_unrelated_ci_high"]),
                "p_value": float(row["mann_whitney_p_value"]),
            }
        )
    for _, row in cosine_best.iterrows():
        rows.append(
            {
                "granularity": str(row["granularity"]),
                "method": "cosine",
                "best_variant": str(row["score_variant"]),
                "auc_imp_vs_unrel": float(row["auc_implicit_vs_unrelated"]),
                "ci_low": float(row["auc_implicit_vs_unrelated_ci_low"]),
                "ci_high": float(row["auc_implicit_vs_unrelated_ci_high"]),
                "p_value": float(row["mann_whitney_p_value"]),
            }
        )

    head_to_head_df = pd.DataFrame(rows).sort_values(
        by=["granularity", "method"],
        ascending=[True, False],
        ignore_index=True,
    )
    head_to_head_df.to_csv(run_dir / "head_to_head.csv", index=False, encoding="utf-8")

    sentence_nli = head_to_head_df[
        (head_to_head_df["granularity"] == "sentence") & (head_to_head_df["method"] == "NLI")
    ]["auc_imp_vs_unrel"].iloc[0]
    sentence_cos = head_to_head_df[
        (head_to_head_df["granularity"] == "sentence") & (head_to_head_df["method"] == "cosine")
    ]["auc_imp_vs_unrel"].iloc[0]
    review_nli = head_to_head_df[
        (head_to_head_df["granularity"] == "review") & (head_to_head_df["method"] == "NLI")
    ]["auc_imp_vs_unrel"].iloc[0]
    review_cos = head_to_head_df[
        (head_to_head_df["granularity"] == "review") & (head_to_head_df["method"] == "cosine")
    ]["auc_imp_vs_unrel"].iloc[0]

    if sentence_cos > sentence_nli and review_cos > review_nli:
        relation = "выше"
        decision = "cosine можно рассматривать как основной matching signal"
    elif sentence_cos < sentence_nli and review_cos < review_nli:
        relation = "ниже"
        decision = "NLI остаётся сильнее; cosine разумно использовать как fallback или feature"
    else:
        relation = "сравнимо"
        decision = "нужен hybrid matching stage с выбором по granularity"

    best_cosine = cosine_best.sort_values(
        by="auc_implicit_vs_unrelated",
        ascending=False,
        na_position="last",
    ).iloc[0]
    paragraph = (
        f"Cosine AUC {relation} с NLI AUC на тех же 90 парах: "
        f"лучший cosine variant — `{best_cosine['score_variant']}` "
        f"(`{best_cosine['granularity']}`, AUC `{best_cosine['auc_implicit_vs_unrelated']:.4f}`), "
        f"тогда как лучший NLI остаётся `experience` "
        f"(`sentence`, AUC `{sentence_nli:.4f}`). Решение: {decision}."
    )
    (run_dir / "head_to_head.md").write_text(paragraph + "\n", encoding="utf-8")


def analyze_cosine(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    analysis_cfg = dict(config.get("analysis", {}))
    bootstrap_iterations = int(analysis_cfg.get("bootstrap_iterations", 1000))
    seed = int(analysis_cfg.get("seed", 42))

    results_df = _load_cosine_results(run_dir)
    metrics_df = _metrics_frame(
        results_df=results_df,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    metrics_df.to_csv(run_dir / "cosine_analysis_metrics.csv", index=False, encoding="utf-8")

    cosine_matrix = metrics_df[
        (metrics_df["scope"] == "overall")
    ][
        [
            "granularity",
            "score_variant",
            "auc_implicit_vs_unrelated",
            "auc_implicit_vs_unrelated_ci_low",
            "auc_implicit_vs_unrelated_ci_high",
            "auc_mention_vs_unrelated",
            "auc_mention_vs_unrelated_ci_low",
            "auc_mention_vs_unrelated_ci_high",
            "mann_whitney_p_value",
        ]
    ].copy()
    cosine_matrix = cosine_matrix.sort_values(
        by=["granularity", "auc_implicit_vs_unrelated"],
        ascending=[True, False],
        na_position="last",
        ignore_index=True,
    )
    cosine_matrix.to_csv(run_dir / "cosine_auc_matrix.csv", index=False, encoding="utf-8")

    aspect_type_matrix = metrics_df[
        metrics_df["scope"] == "aspect_type"
    ][
        [
            "aspect_type",
            "granularity",
            "score_variant",
            "auc_implicit_vs_unrelated",
            "auc_mention_vs_unrelated",
            "mann_whitney_p_value",
        ]
    ].copy()
    aspect_type_matrix = aspect_type_matrix.sort_values(
        by=["granularity", "score_variant", "aspect_type"],
        ignore_index=True,
    )
    aspect_type_matrix.to_csv(run_dir / "cosine_auc_by_aspect_type.csv", index=False, encoding="utf-8")

    best_primary = cosine_matrix[cosine_matrix["score_variant"] == PRIMARY_SCORE].sort_values(
        by="auc_implicit_vs_unrelated",
        ascending=False,
        na_position="last",
    ).iloc[0].to_dict()
    best_overall = cosine_matrix.sort_values(
        by="auc_implicit_vs_unrelated",
        ascending=False,
        na_position="last",
    ).iloc[0].to_dict()

    manifest_path = run_dir / "cosine_manifest.json"
    manifest = load_json(manifest_path)
    nli_run_dir = Path(manifest["source_run_dir"])
    _write_head_to_head(run_dir, nli_run_dir, cosine_matrix)

    summary_lines = [
        "# Cosine Baseline Summary",
        "",
        f"- Best overall variant: `{best_overall['granularity']}` + `{best_overall['score_variant']}`",
        f"- Best overall AUC implicit vs unrelated: `{best_overall['auc_implicit_vs_unrelated']:.4f}`",
        f"- Best overall 95% CI: `[{best_overall['auc_implicit_vs_unrelated_ci_low']:.4f}, {best_overall['auc_implicit_vs_unrelated_ci_high']:.4f}]`",
        f"- Best overall AUC mention vs unrelated: `{best_overall['auc_mention_vs_unrelated']:.4f}`",
        f"- Best overall Mann-Whitney p-value: `{best_overall['mann_whitney_p_value']:.4g}`",
        f"- Best overall go/no-go: `{_go_no_go_label(best_overall['auc_implicit_vs_unrelated'])}`",
        f"- Raw cosine reference: `{best_primary['granularity']}` + `{best_primary['score_variant']}` with `AUC_imp={best_primary['auc_implicit_vs_unrelated']:.4f}`",
        "",
        "## Matrix",
    ]
    for _, row in cosine_matrix.iterrows():
        summary_lines.append(
            f"- `{row['granularity']}` + `{row['score_variant']}` -> "
            f"`AUC_imp={row['auc_implicit_vs_unrelated']:.4f}`, "
            f"`AUC_mention={row['auc_mention_vs_unrelated']:.4f}`, "
            f"`p={row['mann_whitney_p_value']:.4g}`"
        )

    summary_lines.extend(["", "## By Aspect Type"])
    best_combo_aspects = aspect_type_matrix[
        (aspect_type_matrix["granularity"] == best_overall["granularity"])
        & (aspect_type_matrix["score_variant"] == best_overall["score_variant"])
    ]
    for _, row in best_combo_aspects.iterrows():
        summary_lines.append(
            f"- `{row['aspect_type']}` -> "
            f"`AUC_imp={row['auc_implicit_vs_unrelated']:.4f}`, "
            f"`AUC_mention={row['auc_mention_vs_unrelated']:.4f}`, "
            f"`p={row['mann_whitney_p_value']:.4g}`"
        )

    (run_dir / "cosine_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    summary = {
        "best_primary": best_primary,
        "best_overall": best_overall,
        "source_nli_run_dir": str(nli_run_dir),
        "head_to_head_path": str(run_dir / "head_to_head.csv"),
    }
    dump_json(run_dir / "cosine_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cosine baseline outputs.")
    parser.add_argument(
        "--config",
        default="experiments/implicit_nli_pilot.json",
        help="Path to pilot config JSON.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Cosine baseline results directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    run_dir = resolve_repo_path(args.run_dir)
    summary = analyze_cosine(config, run_dir)
    print(f"[implicit_pilot] cosine_summary={_go_no_go_label(summary['best_primary']['auc_implicit_vs_unrelated'])}")


if __name__ == "__main__":
    main()
