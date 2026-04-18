from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_CSV = _ROOT / "benchmark" / "eval_datasets" / "combined_benchmark.csv"
_DEFAULT_MAPPING = "auto"
_DEFAULT_CLUSTERER = "divisive"
_DEFAULT_TARGET_NM_IDS = [1809358565, 165234215]

from eval_pipeline import (  # noqa: E402
    ASPECT_ALIASES,
    MANUAL_MAPPING,
    _collect_eval_aspect_sets,
    _build_auto_mapping,
    load_markup,
    load_pipeline_reviews_from_csv,
    set_global_seed,
)
from run_experiment import temporary_config_overrides  # noqa: E402


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


def _summarize_clustering_stats(
    pipeline_results: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
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

    ordered = sorted(all_cluster_sizes)
    return {
        "num_clusters": int(len(all_cluster_sizes)),
        "avg_cluster_size": float(sum(all_cluster_sizes) / len(all_cluster_sizes)),
        "median_cluster_size": int(ordered[len(ordered) // 2]),
        "largest_cluster_size": int(max(all_cluster_sizes)),
        "smallest_cluster_size": int(min(all_cluster_sizes)),
        "mdl_accepted_splits": int(mdl_accepted_total),
        "mdl_rejected_splits": int(mdl_rejected_total),
        "by_product": per_product,
    }


def _evaluate_recall_only(
    markup_df,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> Dict[str, Any]:
    per_product: Dict[int, Dict[str, Any]] = {}
    all_precision_hits = 0
    all_precision_total = 0
    all_recall_hits = 0
    all_recall_total = 0

    for nm_id, pred_data in pipeline_results.items():
        pred_aspects, pred_eval_aspects, eval_projection = _collect_eval_aspect_sets(pred_data)
        product_mapping = mapping.get(nm_id, {})

        grp = markup_df[markup_df["nm_id"] == nm_id]
        true_aspects_all = set()
        for labels in grp["true_labels_parsed"].dropna():
            true_aspects_all.update(labels.keys())

        mapped_pred = {
            eval_projection.get(pred_aspect, pred_aspect)
            for pred_aspect, true_aspect in product_mapping.items()
            if true_aspect is not None and pred_aspect in pred_aspects
        }
        mapped_true = {
            true_aspect
            for pred_aspect, true_aspect in product_mapping.items()
            if true_aspect is not None and pred_aspect in pred_aspects
        }

        precision_hits = len(mapped_pred)
        precision_total = len(pred_eval_aspects)
        recall_hits = len(mapped_true)
        recall_total = len(true_aspects_all)

        precision = precision_hits / precision_total if precision_total else 0.0
        recall = recall_hits / recall_total if recall_total else 0.0

        per_product[nm_id] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "pred_aspects": sorted(pred_aspects),
            "pred_eval_aspects": sorted(pred_eval_aspects),
            "true_aspects": sorted(true_aspects_all),
            "pred_aspects_count": int(precision_total),
            "true_aspects_count": int(recall_total),
            "matched_predicted_count": int(precision_hits),
            "matched_true_count": int(recall_hits),
        }

        all_precision_hits += precision_hits
        all_precision_total += precision_total
        all_recall_hits += recall_hits
        all_recall_total += recall_total

    macro_precision = (
        float(sum(row["precision"] for row in per_product.values()) / len(per_product))
        if per_product
        else 0.0
    )
    macro_recall = (
        float(sum(row["recall"] for row in per_product.values()) / len(per_product))
        if per_product
        else 0.0
    )
    micro_precision = (
        float(all_precision_hits / all_precision_total) if all_precision_total else 0.0
    )
    micro_recall = (
        float(all_recall_hits / all_recall_total) if all_recall_total else 0.0
    )
    return {
        "per_product": per_product,
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
    }


def _print_summary_table(
    name: str,
    n_venues: int,
    n_reviews: int,
    metrics: Dict[str, Any],
    clustering_stats: Dict[str, Any],
) -> None:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"RECALL-ONLY BENCHMARK: {name} ({n_venues} venues, {n_reviews} reviews)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Macro Precision         {metrics.get('macro_precision', 'N/A')}")
    print(f"Macro Recall            {metrics.get('macro_recall', 'N/A')}")
    print(f"Micro Precision         {metrics.get('micro_precision', 'N/A')}")
    print(f"Micro Recall            {metrics.get('micro_recall', 'N/A')}")
    print(f"Clusters total          {clustering_stats.get('num_clusters', 'N/A')}")
    print()
    print("Per-venue breakdown:")
    per_product = metrics.get("per_product", {}) or {}
    stats_by_product = clustering_stats.get("by_product", {}) or {}
    for nm_id, row in per_product.items():
        kp = stats_by_product.get(str(nm_id), {}).get("num_clusters", "N/A")
        print(
            f"  nm_id={nm_id:<10} "
            f"Recall={row.get('recall', 'N/A')} "
            f"Precision={row.get('precision', 'N/A')} "
            f"Kp={kp}"
        )
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def _make_clusterer(clusterer_name: str, encoder):
    from configs.configs import config
    from src.stages.clustering import (
        AspectClusterer,
        DivisiveClusterer,
        MDLDivisiveClusterer,
    )
    from src.stages.naming import MedoidNamer

    if clusterer_name == "aspect":
        return AspectClusterer(model=encoder)
    if clusterer_name == "divisive":
        return DivisiveClusterer(model=encoder, namer=MedoidNamer())
    if clusterer_name == "mdl_divisive":
        return MDLDivisiveClusterer(
            model=encoder,
            namer=MedoidNamer(),
            use_aicc_correction=bool(
                getattr(config.discovery, "mdl_use_aicc_correction", True)
            ),
            model_penalty_alpha=float(
                getattr(config.discovery, "mdl_model_penalty_alpha", 1.0)
            ),
        )
    raise ValueError(f"Unsupported clusterer={clusterer_name!r}")


def _run_clustering_only_for_ids(
    nm_ids: List[int],
    csv_path: str,
    clusterer_name: str,
) -> Dict[int, Dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    from configs.configs import config
    from src.pipeline import build_aspect_eval_labels
    from src.schemas.models import ReviewInput
    from src.stages.extraction import build_extraction_stage
    from src.stages.pairing import extract_all_with_mapping
    from src.stages.scoring import KeyBERTScorer

    raw_reviews = load_pipeline_reviews_from_csv(csv_path, nm_ids)
    reviews_by_nm: Dict[int, List[dict]] = defaultdict(list)
    for row in raw_reviews:
        reviews_by_nm[int(row["nm_id"])].append(row)

    encoder = SentenceTransformer(config.models.encoder_path)
    extractor = build_extraction_stage()
    scorer = KeyBERTScorer(model=encoder)
    clusterer = _make_clusterer(clusterer_name, encoder)

    results: Dict[int, Dict[str, Any]] = {}
    for nm_id in nm_ids:
        print(f"\n{'=' * 60}")
        print(f"nm_id={nm_id}  (cluster-only recall benchmark)")
        print(f"{'=' * 60}")
        typed_reviews: list[ReviewInput] = []
        for row in reviews_by_nm.get(int(nm_id), []):
            try:
                review = ReviewInput(**row)
            except (TypeError, ValueError):
                continue
            if review.clean_text:
                typed_reviews.append(review)

        if not typed_reviews:
            results[int(nm_id)] = {
                "aspects": [],
                "aspect_keywords": {},
                "diagnostics": {},
            }
            continue

        texts = [review.clean_text for review in typed_reviews]
        review_ids = [review.id for review in typed_reviews]
        all_candidates, _ = extract_all_with_mapping(extractor, texts, review_ids)
        scored_candidates = scorer.score_and_select(all_candidates)
        aspects = clusterer.cluster(scored_candidates)
        diagnostics = {}
        if hasattr(clusterer, "get_diagnostics"):
            maybe_diagnostics = clusterer.get_diagnostics()
            if isinstance(maybe_diagnostics, dict):
                diagnostics = maybe_diagnostics
        aspect_eval_labels = build_aspect_eval_labels(aspects, scored_candidates)
        diagnostics["aspect_eval_labels"] = aspect_eval_labels

        aspect_names = list(aspects.keys())
        aliased_names = [ASPECT_ALIASES[name] for name in aspect_names if name in ASPECT_ALIASES]
        for aspect_name in aspect_names:
            if aspect_name in ASPECT_ALIASES:
                aspect_eval_labels[ASPECT_ALIASES[aspect_name]] = aspect_eval_labels.get(
                    aspect_name,
                    ASPECT_ALIASES[aspect_name],
                )
        results[int(nm_id)] = {
            "aspects": list(dict.fromkeys(aspect_names + aliased_names)),
            "aspect_keywords": {
                name: list(info.keywords) for name, info in aspects.items()
            },
            "aspect_eval_labels": aspect_eval_labels,
            "diagnostics": diagnostics,
        }
        print(
            f"       Review count: {len(typed_reviews)} | "
            f"candidates: {len(all_candidates)} | "
            f"scored: {len(scored_candidates)} | "
            f"Kp: {len(aspects)}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cluster-only recall benchmark without NLI and rating metrics."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Путь к eval CSV (например benchmark/eval_datasets/combined_benchmark.csv).",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help="Тип маппинга аспектов: auto или manual.",
    )
    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=0.3,
        help="Порог косинусной близости для auto mapping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для всех генераторов.",
    )
    parser.add_argument(
        "--clusterer",
        type=str,
        default=_DEFAULT_CLUSTERER,
        choices=["aspect", "divisive", "mdl_divisive"],
        help="Кластеризация для recall-only benchmark.",
    )
    parser.add_argument(
        "--only-target-products",
        action="store_true",
        help=(
            "Запустить benchmark только на фиксированном наборе товаров: "
            f"{_DEFAULT_TARGET_NM_IDS}."
        ),
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    csv_path = str(args.csv)
    markup_df = load_markup(csv_path)
    if args.only_target_products:
        markup_df = markup_df[markup_df["nm_id"].isin(_DEFAULT_TARGET_NM_IDS)].copy()
        if markup_df.empty:
            raise ValueError(f"В CSV нет целевых nm_id: {_DEFAULT_TARGET_NM_IDS}")
        print(f"[recall-benchmark] Ограничение по nm_id: {_DEFAULT_TARGET_NM_IDS}")

    stats = markup_df.groupby("nm_id").size().reset_index(name="n")
    nm_ids: List[int] = stats["nm_id"].astype(int).tolist()
    n_reviews = int(len(markup_df))

    with temporary_config_overrides({}):
        pipeline_results = _run_clustering_only_for_ids(
            nm_ids=nm_ids,
            csv_path=csv_path,
            clusterer_name=args.clusterer,
        )

    if args.mapping == "auto":
        active_mapping = _build_auto_mapping(
            pipeline_results,
            markup_df,
            threshold=args.auto_threshold,
        )
    else:
        active_mapping = MANUAL_MAPPING

    metrics = _evaluate_recall_only(markup_df, pipeline_results, active_mapping)
    clustering_stats = _summarize_clustering_stats(pipeline_results)
    metrics["clustering_stats"] = clustering_stats

    results_dir = _ROOT / "benchmark" / "eval_datasets" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_suffix = f"_{args.clusterer}" if args.clusterer != "aspect" else ""
    limited_suffix = "_limited_products" if args.only_target_products else ""
    dt_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_path = (
        results_dir
        / f"benchmark_recall_results{out_suffix}{limited_suffix}_{dt_name}.yaml"
    )

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "csv_path": csv_path,
        "nm_ids": nm_ids,
        "n_reviews": n_reviews,
        "limited_products_mode": bool(args.only_target_products),
        "limited_products_nm_ids": _DEFAULT_TARGET_NM_IDS if args.only_target_products else None,
        "clusterer": args.clusterer,
        "mapping_mode": args.mapping,
        "auto_threshold": args.auto_threshold if args.mapping == "auto" else None,
        "metrics": metrics,
        "clustering_stats": clustering_stats,
    }
    out_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(_json_safe(payload))),
        encoding="utf-8",
    )
    print(f"\n[recall-benchmark] Результаты записаны: {out_path}")
    _print_summary_table(
        name=f"Recall-only ({args.clusterer})",
        n_venues=len(nm_ids),
        n_reviews=n_reviews,
        metrics=metrics,
        clustering_stats=clustering_stats,
    )


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    if len(sys.argv) <= 1:
        sys.argv = [
            sys.argv[0],
            "--csv",
            str(_DEFAULT_CSV),
            "--mapping",
            _DEFAULT_MAPPING,
            "--clusterer",
            _DEFAULT_CLUSTERER,
        ]
    main()
