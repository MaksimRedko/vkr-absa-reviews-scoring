from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from configs.configs import config, temporary_config_overrides
from eval_pipeline import (
    MANUAL_MAPPING,
    _build_auto_mapping,
    evaluate_product_ratings,
    evaluate_with_mapping,
    load_markup,
    run_pipeline_for_ids,
    set_global_seed,
)
from src.stages.fraud import NoOpFraud
from experiments.compare_runs import compare_two_runs
from experiments.experiment_manager import ExperimentManager, load_registry


def _load_experiment_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Experiment config must be a JSON object")
    data.setdefault("name", cfg_path.stem)
    data.setdefault("description", "")
    data.setdefault("overrides", {})
    return data


def _resolve_baseline_run_id(current_run_id: str, baseline_arg: str) -> Optional[str]:
    if baseline_arg != "latest":
        return baseline_arg

    entries = load_registry()
    for e in reversed(entries):
        rid = e.get("run_id")
        if rid and rid != current_run_id:
            return rid
    return None


def _summarize_clustering_stats(pipeline_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    per_product: Dict[str, Dict[str, Any]] = {}
    all_cluster_sizes: list[int] = []
    mdl_accepted_total = 0
    mdl_rejected_total = 0

    for nm_id, payload in pipeline_results.items():
        diagnostics = payload.get("diagnostics") or {}
        stats = diagnostics.get("clustering_stats") or {}
        per_product[str(nm_id)] = dict(stats)
        cluster_sizes = [int(size) for size in stats.get("cluster_sizes", [])]
        all_cluster_sizes.extend(cluster_sizes)
        mdl_accepted_total += int(stats.get("mdl_accepted_splits", 0) or 0)
        mdl_rejected_total += int(stats.get("mdl_rejected_splits", 0) or 0)

    if not all_cluster_sizes:
        return {
            "num_clusters": 0,
            "avg_cluster_size": 0.0,
            "median_cluster_size": 0,
            "largest_cluster_size": 0,
            "smallest_cluster_size": 0,
            "mdl_accepted_splits": mdl_accepted_total,
            "mdl_rejected_splits": mdl_rejected_total,
            "by_product": per_product,
        }

    return {
        "num_clusters": int(len(all_cluster_sizes)),
        "avg_cluster_size": float(sum(all_cluster_sizes) / len(all_cluster_sizes)),
        "median_cluster_size": int(sorted(all_cluster_sizes)[len(all_cluster_sizes) // 2]),
        "largest_cluster_size": int(max(all_cluster_sizes)),
        "smallest_cluster_size": int(min(all_cluster_sizes)),
        "mdl_accepted_splits": int(mdl_accepted_total),
        "mdl_rejected_splits": int(mdl_rejected_total),
        "by_product": per_product,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ABSA experiment end-to-end")
    parser.add_argument("--config", required=True, help="Path to experiment JSON")
    parser.add_argument(
        "--baseline",
        default=None,
        help="latest or explicit run_id to compare against",
    )
    parser.add_argument(
        "--csv-path",
        default="parser/reviews_batches/merged_checked_reviews.csv",
        help="Path to markup CSV",
    )
    parser.add_argument(
        "--mapping",
        choices=("auto", "manual"),
        default="auto",
        help="Mapping mode for metrics",
    )
    parser.add_argument("--auto-threshold", type=float, default=0.3)
    parser.add_argument("--json-path", default=None)
    parser.add_argument(
        "--clusterer",
        choices=("aspect", "divisive", "mdl_divisive"),
        default=None,
        help="Переопределить кластеризатор поверх experiment config.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-fraud",
        action="store_true",
        help="Подставить NoOpFraud (все trust_weight=1.0), без encode антифрода",
    )
    args = parser.parse_args()

    exp_cfg = _load_experiment_config(args.config)
    set_global_seed(args.seed)

    exp = ExperimentManager.create(exp_cfg["name"], exp_cfg)
    print(f"[Experiment] description: {exp_cfg.get('description', '')}")
    clusterer_name = str(args.clusterer or exp_cfg.get("clusterer") or "aspect")

    with temporary_config_overrides(exp_cfg.get("overrides", {})) as resolved_cfg:
        with open(exp.run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump(
                OmegaConf.to_container(resolved_cfg, resolve=True),
                f,
                ensure_ascii=False,
                indent=2,
            )

        df = load_markup(args.csv_path)
        stats = df.groupby("nm_id").size().reset_index(name="n")
        nm_ids = stats["nm_id"].astype(int).tolist()

        fraud_stage = NoOpFraud() if args.no_fraud else None
        pipeline_results = run_pipeline_for_ids(
            nm_ids,
            args.csv_path,
            args.json_path,
            fraud_stage=fraud_stage,
            clusterer=clusterer_name,
        )

        step12_payload = {
            "pipeline_results": {
                str(k): {
                    "aspects": v["aspects"],
                    "aspect_keywords": v.get("aspect_keywords", {}),
                    "diagnostics": v.get("diagnostics", {}),
                }
                for k, v in pipeline_results.items()
            }
        }
        with open(exp.run_dir / "eval_results_step1_2.json", "w", encoding="utf-8") as f:
            json.dump(step12_payload, f, ensure_ascii=False, indent=2)

        per_review_dump = {str(k): v["per_review"] for k, v in pipeline_results.items()}
        with open(exp.run_dir / "eval_per_review.json", "w", encoding="utf-8") as f:
            json.dump(per_review_dump, f, ensure_ascii=False, indent=2)

        pipeline_results_for_eval = {
            int(k): {
                "aspects": v["aspects"],
                "aspect_keywords": v.get("aspect_keywords", {}),
                "per_review": per_review_dump.get(str(k), {}),
                "diagnostics": v.get("diagnostics", {}),
            }
            for k, v in pipeline_results.items()
        }

        if args.mapping == "auto":
            active_mapping = _build_auto_mapping(
                pipeline_results_for_eval,
                df,
                threshold=args.auto_threshold,
            )
        else:
            active_mapping = MANUAL_MAPPING

        metrics = evaluate_with_mapping(df, pipeline_results_for_eval, active_mapping)
        product_ratings = evaluate_product_ratings(df, pipeline_results_for_eval, active_mapping)
        metrics["product_ratings"] = product_ratings
        metrics["clustering_stats"] = _summarize_clustering_stats(pipeline_results_for_eval)

        total_nli_pairs = sum(
            int((v.get("diagnostics") or {}).get("nli_pairs_count") or 0)
            for v in pipeline_results_for_eval.values()
        )
        metrics["run_summary"] = {
            "multi_label_threshold": float(config.discovery.multi_label_threshold),
            "multi_label_max_aspects": int(config.discovery.multi_label_max_aspects),
            "pairing_strategy": str(getattr(config.sentiment, "pairing_strategy", "")),
            "nli_pairs_total": total_nli_pairs,
            "mention_recall_review": metrics.get("global_mention_recall_review"),
            "sentence_mae_raw": metrics.get("global_mae_raw"),
            "product_mae_n_ge_3": product_ratings.get("global_product_mae_filtered"),
            "no_fraud": bool(args.no_fraud),
            "clusterer": clusterer_name,
        }

        metrics_name = "eval_metrics_auto.json" if args.mapping == "auto" else "eval_metrics.json"
        with open(exp.run_dir / metrics_name, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    exp.finalize()
    print(f"[Experiment] completed: {exp.run_id}")

    if args.baseline:
        baseline_run_id = _resolve_baseline_run_id(exp.run_id, args.baseline)
        if not baseline_run_id:
            print("[Compare] baseline не найден (registry пуст или только текущий run)")
            return

        baseline_exp = ExperimentManager.load(baseline_run_id)
        baseline_metrics = baseline_exp.metrics()
        current_metrics = exp.metrics()

        if not baseline_metrics or not current_metrics:
            print("[Compare] не удалось загрузить метрики baseline/current")
            return

        print("\n" + compare_two_runs(baseline_metrics, current_metrics))
        print(f"\n[Compare] baseline={baseline_run_id}")
        print(f"[Compare] current={exp.run_id}")


if __name__ == "__main__":
    main()
