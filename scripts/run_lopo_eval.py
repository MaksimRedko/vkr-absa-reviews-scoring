"""
LOPO: для каждого товара test fold = один nm_id, метрики считаются только на нём.

Полный pipeline_results загружается/считается один раз; маппинг — на всех товарах (как в полном eval).
Fold-level baseline comparison: model vs star vs neutral vs product_mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import temporary_config_overrides

from eval_audit_metrics_product import compute_product_level_metrics
from eval_audit_metrics_sentiment import (
    build_synthetic_per_review_baseline,
    compute_review_level_sentiment,
    compute_sentiment_for_per_review_override,
)
from eval_audit_metrics_detection import compute_review_level_detection
from run_eval_audit import (
    load_step12_and_per_review,
    resolve_mapping,
)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="parser/reviews_batches/merged_checked_reviews.csv")
    parser.add_argument("--per-review-json", default=None)
    parser.add_argument("--step12-json", default=None)
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--clusterer", default="aspect", choices=["aspect", "divisive", "mdl_divisive"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping", default="mixed", choices=["manual", "auto", "mixed"])
    parser.add_argument("--auto-threshold", type=float, default=0.3)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    overrides = cfg.get("overrides", {}) if isinstance(cfg.get("overrides"), dict) else {}

    from eval_pipeline import load_markup, run_pipeline_for_ids, set_global_seed

    set_global_seed(args.seed)
    df = load_markup(args.csv_path)
    nm_ids = sorted(df["nm_id"].dropna().unique().astype(int).tolist())

    p12 = Path(args.step12_json) if args.step12_json else None
    pr = Path(args.per_review_json) if args.per_review_json else None

    use_disk = (
        p12
        and pr
        and p12.is_file()
        and pr.is_file()
        and not args.run_pipeline
    )

    if use_disk:
        _, pipeline_results = load_step12_and_per_review(p12, pr)
    else:
        with temporary_config_overrides(overrides):
            pipeline_results = run_pipeline_for_ids(
                nm_ids,
                args.csv_path,
                args.json_path,
                clusterer=args.clusterer,
            )

    with temporary_config_overrides(overrides):
        mapping, mode_label, _, used_auto = resolve_mapping(
            df, pipeline_results, args.mapping, args.auto_threshold, overrides
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "eval_audit" / f"lopo_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    star_pr = build_synthetic_per_review_baseline(df, pipeline_results, mapping, "star")
    neu_pr = build_synthetic_per_review_baseline(df, pipeline_results, mapping, "neutral")
    pm_pr = build_synthetic_per_review_baseline(df, pipeline_results, mapping, "product_mean")

    fold_rows: List[Dict[str, Any]] = []

    for test_id in nm_ids:
        if test_id not in pipeline_results:
            continue
        sub_pr = {test_id: pipeline_results[test_id]}
        sub_df = df[df["nm_id"] == test_id]
        if sub_df.empty:
            continue

        det = compute_review_level_detection(sub_df, sub_pr, mapping)
        sent = compute_review_level_sentiment(sub_df, sub_pr, mapping)
        prod = compute_product_level_metrics(sub_df, sub_pr, mapping)

        s_star = compute_sentiment_for_per_review_override(
            sub_df, {test_id: star_pr.get(test_id, {})}, mapping
        )
        s_neu = compute_sentiment_for_per_review_override(
            sub_df, {test_id: neu_pr.get(test_id, {})}, mapping
        )
        s_pm = compute_sentiment_for_per_review_override(
            sub_df, {test_id: pm_pr.get(test_id, {})}, mapping
        )

        pr_star = {test_id: {**sub_pr[test_id], "per_review": star_pr.get(test_id, {})}}
        pr_neu = {test_id: {**sub_pr[test_id], "per_review": neu_pr.get(test_id, {})}}
        pr_pmean = {test_id: {**sub_pr[test_id], "per_review": pm_pr.get(test_id, {})}}

        p_star = compute_product_level_metrics(sub_df, pr_star, mapping)
        p_neu = compute_product_level_metrics(sub_df, pr_neu, mapping)
        p_pm = compute_product_level_metrics(sub_df, pr_pmean, mapping)

        fold_rows.append(
            {
                "fold_nm_id": test_id,
                "mapping_mode": mode_label,
                "macro_precision": det.macro_precision,
                "macro_recall": det.macro_recall,
                "macro_f1": det.macro_f1,
                "review_mae_continuous_model": sent.mae_continuous_review_macro,
                "review_mae_rounded_model": sent.mae_rounded_review_macro,
                "review_mae_continuous_star": s_star.mae_continuous_review_macro,
                "review_mae_rounded_star": s_star.mae_rounded_review_macro,
                "review_mae_continuous_neutral": s_neu.mae_continuous_review_macro,
                "review_mae_rounded_neutral": s_neu.mae_rounded_review_macro,
                "review_mae_continuous_product_mean": s_pm.mae_continuous_review_macro,
                "review_mae_rounded_product_mean": s_pm.mae_rounded_review_macro,
                "product_mae_matched_all_model": prod.product_mae_matched_all,
                "product_mae_matched_all_star": p_star.product_mae_matched_all,
                "product_mae_matched_all_neutral": p_neu.product_mae_matched_all,
                "product_mae_matched_all_product_mean": p_pm.product_mae_matched_all,
                "product_mae_n_ge_3_model": prod.product_mae_matched_n_true_ge_3,
                "product_mae_n_ge_3_star": p_star.product_mae_matched_n_true_ge_3,
                "product_mae_n_ge_3_neutral": p_neu.product_mae_matched_n_true_ge_3,
                "product_mae_n_ge_3_product_mean": p_pm.product_mae_matched_n_true_ge_3,
                "n_matched_pairs": sent.n_matched_pairs,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "lopo_fold_metrics.csv", index=False)

    if not fold_df.empty:
        num_cols = [
            c
            for c in fold_df.columns
            if c not in ("fold_nm_id", "mapping_mode") and pd.api.types.is_numeric_dtype(fold_df[c])
        ]
        agg = {c: float(fold_df[c].mean()) for c in num_cols}
        agg_row = {"fold_nm_id": "AGGREGATE_MEAN", "mapping_mode": mode_label, **agg}
        pd.DataFrame([agg_row]).to_csv(out_dir / "lopo_aggregate_metrics.csv", index=False)
    else:
        pd.DataFrame({"note": ["no folds"]}).to_csv(out_dir / "lopo_aggregate_metrics.csv", index=False)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mapping_mode": mode_label,
                "nm_ids_used_auto": used_auto,
                "csv_path": args.csv_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[lopo] Wrote {out_dir}")


if __name__ == "__main__":
    main()
