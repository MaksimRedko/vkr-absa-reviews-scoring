"""
Audit-first eval runner: честные review/product метрики, baselines, funnel, confusion.

Режим A: --per-review-json + --step12-json (готовые выходы).
Режим B: если файлов нет или --run-pipeline — вызов eval_pipeline.run_pipeline_for_ids (pipeline не меняется).

eval_pipeline.py не редактируется — только import.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import temporary_config_overrides

from eval_audit_diagnostics import aggregate_funnel_counts, compute_funnel_rows, confusion_pred_to_gold
from eval_audit_metrics_detection import compute_review_level_detection, detection_per_review_to_df
from eval_audit_metrics_product import compute_product_level_metrics, product_rows_to_df
from eval_audit_metrics_sentiment import (
    build_synthetic_per_review_baseline,
    compute_review_level_sentiment,
    compute_sentiment_for_per_review_override,
)


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_step12_and_per_review(
    step12_path: Path,
    per_review_path: Path,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    with open(step12_path, "r", encoding="utf-8") as f:
        step12 = json.load(f)
    with open(per_review_path, "r", encoding="utf-8") as f:
        per_raw = json.load(f)

    pipeline_results: Dict[int, Dict[str, Any]] = {}
    for nm_id_str, info in step12["pipeline_results"].items():
        nm = int(nm_id_str)
        diag = info.get("diagnostics") or {}
        ael = diag.get("aspect_eval_labels") or {}
        pipeline_results[nm] = {
            "aspects": info.get("aspects", []),
            "aspect_keywords": info.get("aspect_keywords", {}),
            "per_review": per_raw.get(nm_id_str, {}),
            "diagnostics": diag,
            "aspect_eval_labels": ael,
        }
    return step12, pipeline_results


def resolve_mapping(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mode: str,
    auto_threshold: float,
    overrides: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Optional[str]]], str, List[int], List[int]]:
    """Returns mapping, mapping_mode label, nm_ids without manual row, nm_ids using auto in this run."""
    from eval_pipeline import MANUAL_MAPPING, _build_auto_mapping

    nm_list = sorted(pipeline_results.keys())
    no_manual = [n for n in nm_list if n not in MANUAL_MAPPING]

    with temporary_config_overrides(overrides):
        if mode == "auto":
            m = _build_auto_mapping(pipeline_results, markup_df, threshold=auto_threshold)
            return m, "auto", no_manual, list(nm_list)

        if mode == "manual":
            m = {nid: dict(MANUAL_MAPPING.get(nid, {})) for nid in nm_list}
            return m, "manual", no_manual, []

        # mixed
        auto_full = _build_auto_mapping(pipeline_results, markup_df, threshold=auto_threshold)
        m = {}
        used_auto: List[int] = []
        for nid in nm_list:
            if nid in MANUAL_MAPPING:
                m[nid] = dict(MANUAL_MAPPING[nid])
            else:
                m[nid] = dict(auto_full.get(nid, {}))
                used_auto.append(nid)
        label = "mixed" if used_auto else "manual"
        return m, label, no_manual, used_auto


def count_pred_aspects_unmapped(
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> int:
    n = 0
    for nm_id, pdata in pipeline_results.items():
        aspects = set(pdata.get("aspects") or [])
        pm = mapping.get(nm_id, {}) or {}
        for pa in aspects:
            if pa not in pm or pm.get(pa) is None:
                n += 1
    return n


def count_reviews_dropped_due_to_mapping(
    markup_df: pd.DataFrame,
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> int:
    """Labeled reviews where product has пустой mapping (нет ни одного pred→true)."""
    c = 0
    for _, row in markup_df.iterrows():
        if row["true_labels_parsed"] is None or (
            isinstance(row["true_labels_parsed"], dict) and not row["true_labels_parsed"]
        ):
            continue
        nm = int(row["nm_id"])
        pm = mapping.get(nm, {})
        if not pm or not any(v is not None for v in pm.values()):
            c += 1
    return c


def build_sample_cases(
    markup_df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
    min_rows: int = 50,
) -> pd.DataFrame:
    from collections import defaultdict

    rows_out: List[Dict[str, Any]] = []

    def mapped_pred_aspects(pred_scores: Dict[str, float], pm: Dict[str, Optional[str]]) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for pa in pred_scores:
            ta = pm.get(pa)
            if ta is not None and ta not in seen:
                seen.add(ta)
                out.append(ta)
        return sorted(out)

    def raw_pred_aspects(pred_scores: Dict[str, float]) -> List[str]:
        return sorted(pred_scores.keys())

    for nm_id, pred_data in pipeline_results.items():
        pm = mapping.get(nm_id, {}) or {}
        per_pr = pred_data.get("per_review") or {}
        grp = markup_df[markup_df["nm_id"] == nm_id]
        for _, row in grp.iterrows():
            tl = row["true_labels_parsed"]
            if not tl:
                continue
            rid = str(row["id"])
            pred_scores = per_pr.get(rid, {})
            A_true = set(tl.keys())
            mp = set(mapped_pred_aspects(pred_scores, pm))
            inter = A_true & mp
            rp = raw_pred_aspects(pred_scores)

            if not pred_scores:
                err = "no_pred_aspects"
            elif not inter:
                err = "wrong_or_missing_mapping"
            else:
                err = "ok_or_partial"

            rows_out.append(
                {
                    "review_id": rid,
                    "product_id": int(nm_id),
                    "true_aspects": json.dumps(sorted(A_true), ensure_ascii=False),
                    "pred_aspects": json.dumps(rp, ensure_ascii=False),
                    "mapped_pred_aspects": json.dumps(sorted(mp), ensure_ascii=False),
                    "intersection_aspects": json.dumps(sorted(inter), ensure_ascii=False),
                    "true_ratings_json": json.dumps({k: tl[k] for k in tl}, ensure_ascii=False),
                    "pred_ratings_json": json.dumps(pred_scores, ensure_ascii=False),
                    "review_star_rating": int(row["rating"]),
                    "error_type": err,
                }
            )

    n_sample = min(min_rows, len(rows_out))
    if n_sample > 0 and len(rows_out) > n_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(rows_out), size=n_sample, replace=False)
        rows_out = [rows_out[i] for i in sorted(idx)]
    return pd.DataFrame(rows_out)


def write_summary_md(
    path: Path,
    meta: Dict[str, Any],
    det: Any,
    sent: Any,
    prod: Any,
    legacy: Optional[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    caveat_auto: bool,
) -> None:
    lines: List[str] = []
    lines.append("# Eval audit summary\n")
    lines.append("## Run metadata\n")
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    if caveat_auto:
        lines.append("### Caveat (auto / mixed mapping)\n")
        lines.append(
            "Метрики, зависящие от автоматического pred→true, **не являются primary-интерпретацией** "
            "без ручной проверки таблицы маппинга.\n"
        )

    lines.append("## Primary\n")
    lines.append("### Review-level detection (macro over reviews)\n")
    lines.append(f"- precision: **{det.macro_precision:.4f}**")
    lines.append(f"- recall: **{det.macro_recall:.4f}**")
    lines.append(f"- F1: **{det.macro_f1:.4f}**\n")

    lines.append("### Review-level sentiment (intersection only)\n")
    lines.append(f"- MAE continuous (macro over reviews): **{sent.mae_continuous_review_macro:.4f}**")
    lines.append(f"- MAE rounded (macro over reviews): **{sent.mae_rounded_review_macro:.4f}**")
    lines.append(f"- n_matched_pairs: {sent.n_matched_pairs}\n")

    lines.append("### Product-level (matched subsets)\n")
    lines.append(f"- MAE matched all: **{prod.product_mae_matched_all}**")
    lines.append(f"- MAE matched n_true≥3: **{prod.product_mae_matched_n_true_ge_3}**")
    lines.append(f"- MAE rounded matched all: **{prod.product_mae_rounded_matched_all}**")
    lines.append(f"- MAE rounded n_true≥3: **{prod.product_mae_rounded_matched_n_true_ge_3}**\n")

    lines.append("### Baseline comparison (same metrics)\n")
    lines.append("| metric | model | star | neutral | product_mean |")
    lines.append("| --- | --- | --- | --- | --- |")
    for br in baseline_rows:
        lines.append(
            f"| {br['metric']} | {br['model']} | {br['star']} | {br['neutral']} | {br['product_mean']} |"
        )
    lines.append("")

    lines.append("## Secondary / diagnostic\n")
    lines.append("### Mapping metadata\n")
    lines.append(f"- mapping_mode: {meta.get('mapping_mode')}")
    lines.append(f"- products_without_manual_mapping: {meta.get('products_without_manual_mapping')}")
    lines.append(f"- reviews_dropped_due_to_mapping: {meta.get('reviews_dropped_due_to_mapping')}")
    lines.append(f"- pred_aspects_unmapped_count: {meta.get('pred_aspects_unmapped_count')}\n")

    lines.append("### Micro detection\n")
    lines.append(f"- micro P/R/F1: {det.micro_precision:.4f} / {det.micro_recall:.4f} / {det.micro_f1:.4f}\n")

    lines.append("### Coverage\n")
    lines.append(f"- reviews_with_no_candidates: {det.reviews_with_no_candidates}")
    lines.append(f"- reviews_with_candidates_but_no_pred_aspects: {det.reviews_with_candidates_but_no_pred_aspects}")
    lines.append(f"- reviews_with_wrong_aspect_only: {det.reviews_with_wrong_aspect_only}")
    lines.append(f"- reviews_with_at_least_one_correct_aspect: {det.reviews_with_at_least_one_correct_aspect}\n")

    if legacy:
        lines.append("### Legacy eval_pipeline metrics (diagnostic)\n")
        lines.append(f"- global_mae_raw: {legacy.get('global_mae_raw')}")
        lines.append(f"- global_mae_calibrated: {legacy.get('global_mae_calibrated')}")
        lines.append(f"- micro_precision / micro_recall: {legacy.get('micro_precision')} / {legacy.get('micro_recall')}")
        lines.append(f"- global_mention_recall_review: {legacy.get('global_mention_recall_review')}\n")

    lines.append("### Files\n")
    lines.append("- `detection_funnel.csv`, `confusion_matrix.csv`, `top_confusions.md`")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="ABSA eval audit (iteration 1)")
    parser.add_argument("--csv-path", default="parser/reviews_batches/merged_checked_reviews.csv")
    parser.add_argument("--per-review-json", default=None, help="eval_per_review.json")
    parser.add_argument("--step12-json", default=None, help="eval_results_step1_2.json")
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Прогнать run_pipeline_for_ids, если JSON нет или всегда при совместном использовании",
    )
    parser.add_argument("--json-path", default=None, help="Длинные отзывы JSON (как в eval_pipeline)")
    parser.add_argument("--clusterer", default="aspect", choices=["aspect", "divisive", "mdl_divisive"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mapping",
        default="mixed",
        choices=["manual", "auto", "mixed"],
        help="manual | auto | mixed (manual + auto для товаров без ручной таблицы)",
    )
    parser.add_argument("--auto-threshold", type=float, default=0.3)
    parser.add_argument("--config", default=None, help="JSON с overrides для OmegaConf (как eval)")
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    overrides = cfg.get("overrides", {}) if isinstance(cfg.get("overrides"), dict) else {}

    from eval_pipeline import load_markup, run_pipeline_for_ids, set_global_seed
    from eval_pipeline import evaluate_with_mapping

    set_global_seed(args.seed)
    df = load_markup(args.csv_path)
    nm_ids = sorted(df["nm_id"].dropna().unique().astype(int).tolist())

    step12_path = Path(args.step12_json) if args.step12_json else None
    per_path = Path(args.per_review_json) if args.per_review_json else None

    use_disk = (
        step12_path
        and per_path
        and step12_path.is_file()
        and per_path.is_file()
        and not args.run_pipeline
    )

    if use_disk:
        print(f"[eval_audit] Mode A: loading {step12_path.name}, {per_path.name}")
        _, pipeline_results = load_step12_and_per_review(step12_path, per_path)
    else:
        print("[eval_audit] Mode B: run_pipeline_for_ids")
        with temporary_config_overrides(overrides):
            pipeline_results = run_pipeline_for_ids(
                nm_ids,
                args.csv_path,
                args.json_path,
                clusterer=args.clusterer,
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "eval_audit" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    with temporary_config_overrides(overrides):
        mapping, mode_label, no_manual, used_auto = resolve_mapping(
            df, pipeline_results, args.mapping, args.auto_threshold, overrides
        )

    caveat = mode_label in ("auto", "mixed") and bool(used_auto or mode_label == "auto")

    meta = {
        "timestamp": ts,
        "csv_path": str(args.csv_path),
        "mapping_mode": mode_label,
        "mapping_cli": args.mapping,
        "products_without_manual_mapping": len(no_manual),
        "nm_ids_without_manual": json.dumps(no_manual),
        "nm_ids_used_auto": json.dumps(used_auto),
        "reviews_dropped_due_to_mapping": count_reviews_dropped_due_to_mapping(df, mapping),
        "pred_aspects_unmapped_count": count_pred_aspects_unmapped(pipeline_results, mapping),
        "clusterer": args.clusterer,
        "seed": args.seed,
    }

    det = compute_review_level_detection(df, pipeline_results, mapping)
    sent = compute_review_level_sentiment(df, pipeline_results, mapping)
    prod = compute_product_level_metrics(df, pipeline_results, mapping)

    star_pr = build_synthetic_per_review_baseline(df, pipeline_results, mapping, "star")
    neu_pr = build_synthetic_per_review_baseline(df, pipeline_results, mapping, "neutral")
    pm_pr = build_synthetic_per_review_baseline(df, pipeline_results, mapping, "product_mean")

    s_star = compute_sentiment_for_per_review_override(df, star_pr, mapping)
    s_neu = compute_sentiment_for_per_review_override(df, neu_pr, mapping)
    s_pm = compute_sentiment_for_per_review_override(df, pm_pr, mapping)

    p_star = compute_product_level_metrics(
        df,
        {k: {**pipeline_results[k], "per_review": star_pr.get(k, {})} for k in pipeline_results},
        mapping,
    )
    p_neu = compute_product_level_metrics(
        df,
        {k: {**pipeline_results[k], "per_review": neu_pr.get(k, {})} for k in pipeline_results},
        mapping,
    )
    p_pmean = compute_product_level_metrics(
        df,
        {k: {**pipeline_results[k], "per_review": pm_pr.get(k, {})} for k in pipeline_results},
        mapping,
    )

    baseline_rows = [
        {
            "metric": "review_sentiment_mae_continuous_macro",
            "model": round(sent.mae_continuous_review_macro, 4),
            "star": round(s_star.mae_continuous_review_macro, 4),
            "neutral": round(s_neu.mae_continuous_review_macro, 4),
            "product_mean": round(s_pm.mae_continuous_review_macro, 4),
        },
        {
            "metric": "review_sentiment_mae_rounded_macro",
            "model": round(sent.mae_rounded_review_macro, 4),
            "star": round(s_star.mae_rounded_review_macro, 4),
            "neutral": round(s_neu.mae_rounded_review_macro, 4),
            "product_mean": round(s_pm.mae_rounded_review_macro, 4),
        },
        {
            "metric": "product_mae_matched_all",
            "model": prod.product_mae_matched_all,
            "star": p_star.product_mae_matched_all,
            "neutral": p_neu.product_mae_matched_all,
            "product_mean": p_pmean.product_mae_matched_all,
        },
        {
            "metric": "product_mae_matched_n_true_ge_3",
            "model": prod.product_mae_matched_n_true_ge_3,
            "star": p_star.product_mae_matched_n_true_ge_3,
            "neutral": p_neu.product_mae_matched_n_true_ge_3,
            "product_mean": p_pmean.product_mae_matched_n_true_ge_3,
        },
    ]

    legacy = None
    try:
        legacy = evaluate_with_mapping(df, pipeline_results, mapping)
    except Exception as e:
        legacy = {"error": str(e)}

    funnel_rows = compute_funnel_rows(df, pipeline_results, mapping)
    funnel_agg = aggregate_funnel_counts(funnel_rows)
    pd.DataFrame([funnel_agg]).to_csv(out_dir / "funnel_counts_summary.csv", index=False)
    mat, top_pairs = confusion_pred_to_gold(df, pipeline_results, mapping)

    write_summary_md(
        out_dir / "summary_metrics.md",
        meta,
        det,
        sent,
        prod,
        legacy if isinstance(legacy, dict) and "error" not in legacy else None,
        baseline_rows,
        caveat,
    )

    summary_csv = {
        **{k: str(v) for k, v in meta.items()},
        "macro_precision": det.macro_precision,
        "macro_recall": det.macro_recall,
        "macro_f1": det.macro_f1,
        "review_mae_continuous_macro": sent.mae_continuous_review_macro,
        "review_mae_rounded_macro": sent.mae_rounded_review_macro,
        "product_mae_matched_all": prod.product_mae_matched_all,
        "product_mae_matched_n_ge_3": prod.product_mae_matched_n_true_ge_3,
    }
    if legacy and "error" not in legacy:
        summary_csv["global_mae_raw"] = legacy.get("global_mae_raw")
        summary_csv["global_mae_calibrated"] = legacy.get("global_mae_calibrated")

    pd.DataFrame([summary_csv]).to_csv(out_dir / "summary_metrics.csv", index=False)

    detection_per_review_to_df(det.per_review).to_csv(
        out_dir / "review_level_detection_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "mae_continuous_macro": sent.mae_continuous_review_macro,
                "rmse_continuous_macro": sent.rmse_continuous_review_macro,
                "mae_rounded_macro": sent.mae_rounded_review_macro,
                "mae_continuous_micro": sent.mae_continuous_micro,
                "n_matched_pairs": sent.n_matched_pairs,
            }
        ]
    ).to_csv(out_dir / "review_level_sentiment_metrics.csv", index=False)

    product_rows_to_df(prod.per_aspect_rows).to_csv(
        out_dir / "per_product_aspect_metrics.csv", index=False
    )

    pp_rows = []
    for nm_id, payload in prod.per_product.items():
        aes = payload.get("aspect_errors", [])
        if aes:
            errs = [x.error_matched for x in aes]
            pp_rows.append(
                {
                    "nm_id": nm_id,
                    "product_mae_matched_mean": float(np.mean(errs)),
                    "n_aspects": len(errs),
                }
            )
    pd.DataFrame(pp_rows).to_csv(out_dir / "per_product_metrics.csv", index=False)

    pd.DataFrame(baseline_rows).to_csv(out_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(funnel_rows).to_csv(out_dir / "detection_funnel.csv", index=False)

    if not mat.empty:
        mat.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8")
    else:
        pd.DataFrame({"note": ["empty"]}).to_csv(out_dir / "confusion_matrix.csv", index=False)

    top_lines = ["# Top confusions (pred aspect name vs gold aspect co-occurrence)\n"]
    for pa, g, k in top_pairs[:15]:
        top_lines.append(f"- `{pa}` vs `{g}`: {k}")
    (out_dir / "top_confusions.md").write_text("\n".join(top_lines), encoding="utf-8")

    build_sample_cases(df, pipeline_results, mapping, min_rows=50).to_csv(
        out_dir / "sample_cases.csv", index=False, encoding="utf-8"
    )

    note = [
        "# eval_audit_note",
        "",
        "## Flow",
        "1. Load markup CSV.",
        "2. Load pipeline JSON (mode A) or run_pipeline_for_ids (mode B).",
        "3. Resolve mapping (manual / auto / mixed).",
        "4. Detection / sentiment / product metrics + baselines.",
        "",
        "## reviews_dropped_due_to_mapping",
        "Счётчик отзывов с непустой разметкой, у которых для товара нет ни одного pred→true в таблице маппинга.",
        "",
        "## Artifacts",
        str(out_dir),
    ]
    (out_dir / "eval_audit_note.md").write_text("\n".join(note), encoding="utf-8")

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if use_disk:
        with open(out_dir / "eval_inputs_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(
                {"per_review_json": str(per_path), "step12_json": str(step12_path)},
                f,
                indent=2,
            )
    print(f"[eval_audit] Wrote {out_dir}")


if __name__ == "__main__":
    main()
