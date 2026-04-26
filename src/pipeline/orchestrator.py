from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.discovery.encoder import DiscoveryEncoder
from src.pipeline.reference import e2e
from src.pipeline.stages import (
    s1_extraction,
    s2_encoding,
    s3_vocab_matching,
    s4_discovery,
    s5_nli_sentiment,
    s6_aggregation,
)
from src.pipeline.stages.common import apply_random_seeds, repo_root
from src.pipeline.tracing import ArtifactWriter

ROOT = repo_root()


def load_run_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _write_e2e_compatible_outputs(
    *,
    out_dir: Path,
    reviews: list[Any],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
    aspect_by_id_by_category: dict[str, dict[str, Any]],
    discovery_by_product: dict[int, Any],
    sentiment_by_pair: dict[tuple[str, str], dict[str, float]],
    aggregated: dict[int, dict[str, Any]],
    negation_stats: dict[str, Any],
) -> dict[str, Any]:
    ref = e2e()
    review_a, aggregate_a, hard_a = ref._review_metric_rows(
        reviews,
        term_to_aspects_by_category,
        discovery_by_product,
        sentiment_by_pair,
        include_discovery=False,
    )
    product_a, product_metrics_a = ref._product_metric_rows(
        reviews,
        term_to_aspects_by_category,
        aspect_by_id_by_category,
        discovery_by_product,
        aggregated,
        sentiment_by_pair,
        include_discovery=False,
    )
    aggregate_a.update(product_metrics_a)

    review_b, aggregate_b, hard_b = ref._review_metric_rows(
        reviews,
        term_to_aspects_by_category,
        discovery_by_product,
        sentiment_by_pair,
        include_discovery=True,
    )
    product_b, product_metrics_b = ref._product_metric_rows(
        reviews,
        term_to_aspects_by_category,
        aspect_by_id_by_category,
        discovery_by_product,
        aggregated,
        sentiment_by_pair,
        include_discovery=True,
    )
    aggregate_b.update(product_metrics_b)

    _star_review_df, star_review_metrics, star_product_df, star_product_metrics = ref._star_metrics(reviews)

    per_product_a = ref._per_product_metrics(reviews, review_a, product_a, hard_a)
    per_product_b = ref._per_product_metrics(reviews, review_b, product_b, hard_b)
    per_product_c_rows: list[dict[str, Any]] = []
    review_count_by_product = {nm_id: len(items) for nm_id, items in ref._group_reviews(reviews).items()}
    for (nm_id, category_id), group in star_product_df.groupby(["nm_id", "category_id"]):
        n3 = group[group["n_reviews_with_aspect"] >= ref.PRODUCT_AGGREGATION_MIN_REVIEWS]
        per_product_c_rows.append(
            {
                "nm_id": nm_id,
                "category_id": category_id,
                "n_reviews": review_count_by_product[int(nm_id)],
                "product_mae_n3": float(n3["abs_error"].mean()) if not n3.empty else float("nan"),
                "n_aspects_matched": int(len(group)),
            }
        )
    per_product_c = pd.DataFrame(per_product_c_rows)
    per_product_c["detection_precision"] = float("nan")
    per_product_c["detection_recall"] = float("nan")
    per_product_c["sentiment_mae_review"] = star_review_metrics["sentiment_mae_review"]
    per_product_c["sentiment_mae_review_round"] = star_review_metrics["sentiment_mae_review_round"]

    ref._write_track_csv(out_dir / "metrics_track_a_vocab_only.csv", per_product_a)
    ref._write_track_csv(out_dir / "metrics_track_b_vocab_plus_discovery.csv", per_product_b)
    ref._write_track_csv(out_dir / "metrics_track_c_star_baseline.csv", per_product_c)

    per_aspect = pd.concat([product_b, star_product_df], ignore_index=True)
    per_aspect.to_csv(out_dir / "per_aspect_breakdown.csv", index=False, encoding="utf-8")
    hard_cases = pd.DataFrame(hard_b).sort_values(["abs_error", "review_id"], ascending=[False, True]).head(30)
    hard_cases.to_csv(out_dir / "hard_cases.csv", index=False, encoding="utf-8")
    negation_stats = ref._finalize_negation_stats(negation_stats, hard_b)

    ref._write_predictions(
        out_dir,
        reviews,
        aspect_by_id_by_category,
        discovery_by_product,
        sentiment_by_pair,
        aggregated,
    )
    ref._write_summary(
        out_dir,
        aggregate_a,
        aggregate_b,
        star_review_metrics,
        star_product_metrics,
        per_product_a,
        per_product_b,
        hard_cases,
        negation_stats,
    )
    return {
        "track_a": aggregate_a,
        "track_b": aggregate_b,
        "track_c_review": star_review_metrics,
        "track_c_product": star_product_metrics,
        "negation_correction": negation_stats,
    }


def run_traced_pipeline(
    *,
    config_path: str | Path = "run_config.yaml",
    limit_products: int = 0,
) -> Path:
    config = load_run_config(config_path)
    apply_random_seeds(config.get("seeds", {}))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _resolve(config.get("output_dir", "results")) / f"{timestamp}_traced"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(out_dir)
    logger = e2e().TeeLogger(out_dir / "run_console.log")
    stage_times: dict[str, float] = {}
    artifact_files: dict[str, str] = {}
    started = time.perf_counter()
    stages_completed: list[str] = []

    try:
        logger.log(f"[start] traced_pipeline out_dir={out_dir}")
        shutil.copy2(_resolve(config_path), out_dir / "run_config.yaml")

        reviews = e2e()._load_reviews(_resolve(config["gold_dataset_csv"]))
        if limit_products:
            keep = set(sorted({review.nm_id for review in reviews})[: int(limit_products)])
            reviews = [review for review in reviews if review.nm_id in keep]
            logger.log(f"[smoke] limit_products={limit_products} nm_ids={sorted(keep)}")
        logger.log(f"[load] reviews={len(reviews)} products={len(set(r.nm_id for r in reviews))}")

        categories = {review.category_id for review in reviews}
        aspects_by_category, term_to_aspects_by_category, aspect_by_id_by_category = e2e()._build_hybrid_vocab(
            _resolve(config["core_vocab"]),
            _resolve(config["domain_vocab_dir"]),
            categories,
        )

        t0 = time.perf_counter()
        candidates = s1_extraction.run_stage(reviews, config)
        stage_times["s1"] = time.perf_counter() - t0
        stages_completed.append("s1")
        artifact_files["candidates"] = "candidates.parquet"
        writer.write_dataframe(
            "candidates.parquet",
            candidates[
                [
                    "candidate_id",
                    "review_id",
                    "nm_id",
                    "category_id",
                    "text",
                    "text_lemmatized",
                    "start_offset",
                    "end_offset",
                    "source",
                ]
            ],
            sort_by=["review_id", "start_offset", "candidate_id"],
        )
        logger.log(f"[s1] candidates={len(candidates)}")

        t0 = time.perf_counter()
        encoder = DiscoveryEncoder(
            model_name_or_path=str(config.get("models", {}).get("encoder", "ai-forever/sbert_large_nlu_ru")),
            batch_size=int(config.get("discovery", {}).get("encoder_batch_size", 8)),
        )
        embedding_cache: dict[str, Any] = {}
        enc = s2_encoding.run_stage(
            candidates,
            aspects_by_category,
            config,
            encoder=encoder,
            cache=embedding_cache,
        )
        stage_times["s2"] = time.perf_counter() - t0
        stages_completed.append("s2")
        writer.write_npy("embeddings_candidates.npy", enc["candidate_embeddings"])
        writer.write_csv("embedding_index_candidates.csv", enc["candidate_index"], sort_by=["row_index"])
        writer.write_npy("embeddings_vocab.npy", enc["vocab_embeddings"])
        writer.write_csv("embedding_index_vocab.csv", enc["vocab_index"], sort_by=["row_index"])
        artifact_files.update(
            {
                "embeddings_candidates": "embeddings_candidates.npy",
                "embedding_index_candidates": "embedding_index_candidates.csv",
                "embeddings_vocab": "embeddings_vocab.npy",
                "embedding_index_vocab": "embedding_index_vocab.csv",
            }
        )
        logger.log(f"[s2] candidate_embeddings={enc['candidate_embeddings'].shape} vocab_embeddings={enc['vocab_embeddings'].shape}")

        t0 = time.perf_counter()
        matches = s3_vocab_matching.run_stage(
            reviews,
            candidates,
            term_to_aspects_by_category,
            enc["candidate_vectors_by_id"],
            enc["aspect_vectors_by_category"],
        )
        stage_times["s3"] = time.perf_counter() - t0
        stages_completed.append("s3")
        artifact_files["candidate_matches"] = "candidate_matches.parquet"
        writer.write_dataframe("candidate_matches.parquet", matches, sort_by=["candidate_id", "matched_aspect_id"])
        logger.log(f"[s3] matches={len(matches)}")

        t0 = time.perf_counter()
        disc = s4_discovery.run_stage(
            reviews,
            _resolve(config["discovery_dir"]),
            enc["encoder"],
            enc["cache"],
            phrase_to_cluster_threshold=float(config.get("discovery", {}).get("phrase_to_cluster_threshold", 0.5)),
        )
        discovery_by_product = disc["discovery_by_product"]
        stage_times["s4"] = time.perf_counter() - t0
        stages_completed.append("s4")
        for nm_id, payload in disc["cluster_payloads"].items():
            writer.write_json(f"clusters_{nm_id}.json", payload)
            writer.write_npy(f"cluster_centroids_{nm_id}.npy", disc["centroid_arrays"][nm_id])
        artifact_files["clusters"] = "clusters_<nm_id>.json"
        artifact_files["cluster_centroids"] = "cluster_centroids_<nm_id>.npy"
        logger.log(f"[s4] discovery_products={len(discovery_by_product)}")

        t0 = time.perf_counter()
        sent = s5_nli_sentiment.run_stage(
            reviews,
            aspect_by_id_by_category,
            discovery_by_product,
            logger=logger,
            config=config,
        )
        stage_times["s5"] = time.perf_counter() - t0
        stages_completed.append("s5")
        artifact_files["nli_predictions"] = "nli_predictions.parquet"
        writer.write_dataframe("nli_predictions.parquet", sent["nli_predictions"], sort_by=["review_id", "aspect_source", "aspect_name"])
        logger.log(f"[s5] nli_predictions={len(sent['nli_predictions'])}")

        t0 = time.perf_counter()
        agg = s6_aggregation.run_stage(
            reviews,
            sent["sentiment_by_pair"],
            aspect_by_id_by_category,
            discovery_by_product,
        )
        stage_times["s6"] = time.perf_counter() - t0
        stages_completed.append("s6")
        artifact_files["product_aggregates"] = "product_aggregates.parquet"
        writer.write_dataframe("product_aggregates.parquet", agg["product_aggregates"], sort_by=["nm_id", "aspect_source", "aspect_name"])
        logger.log(f"[s6] product_aggregates={len(agg['product_aggregates'])}")

        metrics_payload = _write_e2e_compatible_outputs(
            out_dir=out_dir,
            reviews=reviews,
            term_to_aspects_by_category=term_to_aspects_by_category,
            aspect_by_id_by_category=aspect_by_id_by_category,
            discovery_by_product=discovery_by_product,
            sentiment_by_pair=sent["sentiment_by_pair"],
            aggregated=agg["aggregated"],
            negation_stats=sent["negation_stats"],
        )

        summary_payload = {
            "status": "OK",
            "out_dir": str(out_dir),
            "elapsed_sec": round(time.perf_counter() - started, 4),
            **metrics_payload,
            "params": {
                "config_path": str(_resolve(config_path)),
                "matching_mode": config.get("matching", {}).get("matching_mode"),
                "cosine_fallback_enabled": config.get("matching", {}).get("cosine_fallback_enabled"),
                "discovery_phrase_to_cluster_threshold": config.get("discovery", {}).get("phrase_to_cluster_threshold"),
                "sentiment_engine_source": config.get("models", {}).get("sentiment_engine_source"),
            },
        }
        writer.write_json("run_summary.json", summary_payload)

        manifest = {
            "run_id": out_dir.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "config": config,
            "model_versions": {
                "encoder": config.get("models", {}).get("encoder"),
                "nli": config.get("models", {}).get("nli"),
            },
            "n_reviews_processed": len(reviews),
            "n_products": len({review.nm_id for review in reviews}),
            "n_categories": len({review.category_id for review in reviews}),
            "stages_completed": stages_completed,
            "artifact_files": artifact_files,
            "elapsed_seconds_per_stage": {key: round(value, 4) for key, value in stage_times.items()},
            "metrics": metrics_payload,
        }
        writer.write_json("MANIFEST.json", manifest)
        logger.log(json.dumps(summary_payload, ensure_ascii=False, indent=2))
        logger.log(f"[done] elapsed={time.perf_counter() - started:.1f}s")
        return out_dir
    finally:
        logger.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run traced ABSA pipeline")
    parser.add_argument("--config", default="run_config.yaml")
    parser.add_argument("--limit-products", type=int, default=0)
    args = parser.parse_args(argv)
    out_dir = run_traced_pipeline(config_path=args.config, limit_products=args.limit_products)
    print(out_dir)


if __name__ == "__main__":
    main()
