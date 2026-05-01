from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.sentiment import common as sentiment_benchmark
from build_manual_audit_queue import build_queue
from src.evaluation.metrics_overall import write_metrics_report
from src.pipeline import orchestrator
from src.pipeline.reference import e2e
from src.pipeline.stages import s6_aggregation
from src.pipeline.tracing import ArtifactWriter

SOURCE_RUN_DEFAULT = ROOT / "results" / "20260425_183110_traced"
FINAL_V1_DEFAULT = ROOT / "results" / "final_res_v1"
FINAL_V2_DEFAULT = ROOT / "results" / "final_res_v2"
MANUAL_AUDIT_V2_DEFAULT = ROOT / "benchmark" / "manual_audit" / "final_v2"
DEFAULT_FINAL_V2_MODE_ID = sentiment_benchmark.MODE_C

STATIC_TRACED_PATTERNS = [
    "candidates.parquet",
    "candidate_matches.parquet",
    "embeddings_candidates.npy",
    "embedding_index_candidates.csv",
    "embeddings_vocab.npy",
    "embedding_index_vocab.csv",
    "clusters_*.json",
    "cluster_centroids_*.npy",
]

MODE_METADATA = {
    sentiment_benchmark.MODE_A: {
        "label": "current baseline",
        "run_name": "final_res_v2_current_baseline",
        "premise_kind": "full_review",
        "hypothesis_mode": "single_hypothesis",
        "hypothesis_template_pos": sentiment_benchmark.SINGLE_HYPOTHESIS_TEMPLATE,
        "hypothesis_template_neg": "",
        "relevance_mode": "p_ent_plus_contra_single_hypothesis",
        "derived_stage": "derive_s5_mode_a",
    },
    sentiment_benchmark.MODE_B: {
        "label": "sentence evidence",
        "run_name": "final_res_v2_sentence_evidence",
        "premise_kind": "sentence_evidence",
        "hypothesis_mode": "dual_hypothesis",
        "hypothesis_template_pos": sentiment_benchmark.DUAL_HYPOTHESIS_POS_TEMPLATE,
        "hypothesis_template_neg": sentiment_benchmark.DUAL_HYPOTHESIS_NEG_TEMPLATE,
        "relevance_mode": "p_pos_plus_p_neg_dual_hypothesis",
        "derived_stage": "derive_s5_mode_b",
    },
    sentiment_benchmark.MODE_C: {
        "label": "window evidence",
        "run_name": "final_res_v2_window_evidence",
        "premise_kind": "window_evidence",
        "hypothesis_mode": "dual_hypothesis",
        "hypothesis_template_pos": sentiment_benchmark.DUAL_HYPOTHESIS_POS_TEMPLATE,
        "hypothesis_template_neg": sentiment_benchmark.DUAL_HYPOTHESIS_NEG_TEMPLATE,
        "relevance_mode": "p_pos_plus_p_neg_dual_hypothesis",
        "derived_stage": "derive_s5_mode_c",
    },
    sentiment_benchmark.MODE_D: {
        "label": "multi evidence",
        "run_name": "final_res_v2_multi_evidence",
        "premise_kind": "multi_sentence_evidence",
        "hypothesis_mode": "dual_hypothesis",
        "hypothesis_template_pos": sentiment_benchmark.DUAL_HYPOTHESIS_POS_TEMPLATE,
        "hypothesis_template_neg": sentiment_benchmark.DUAL_HYPOTHESIS_NEG_TEMPLATE,
        "relevance_mode": "p_pos_plus_p_neg_dual_hypothesis",
        "derived_stage": "derive_s5_mode_d",
    },
    sentiment_benchmark.MODE_D_WEIGHTED: {
        "label": "multi evidence weighted relevance",
        "run_name": "final_res_v2_multi_evidence_weighted_relevance",
        "premise_kind": "multi_sentence_evidence",
        "hypothesis_mode": "dual_hypothesis",
        "hypothesis_template_pos": sentiment_benchmark.DUAL_HYPOTHESIS_POS_TEMPLATE,
        "hypothesis_template_neg": sentiment_benchmark.DUAL_HYPOTHESIS_NEG_TEMPLATE,
        "relevance_mode": "p_pos_plus_p_neg_dual_hypothesis",
        "derived_stage": "derive_s5_mode_d_weighted",
    },
}


def _mode_meta(mode_id: str) -> dict[str, str]:
    if mode_id not in MODE_METADATA:
        raise ValueError(f"unsupported sentiment mode for final_v2: {mode_id}")
    return MODE_METADATA[mode_id]


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


def _assert_within_workspace(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if root == resolved or root in resolved.parents:
        return
    raise ValueError(f"path must stay inside workspace: {path}")


def _reset_dir(path: Path) -> None:
    _assert_within_workspace(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_full_run(source_run_dir: Path, target_dir: Path) -> None:
    _assert_within_workspace(source_run_dir)
    _assert_within_workspace(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_run_dir, target_dir)


def _copy_static_traced_artifacts(source_run_dir: Path, out_dir: Path) -> None:
    for pattern in STATIC_TRACED_PATTERNS:
        for src in source_run_dir.glob(pattern):
            dst = out_dir / src.name
            shutil.copy2(src, dst)


def _load_discovery_from_traced(source_run_dir: Path) -> dict[int, Any]:
    ref = e2e()
    discovery_by_product: dict[int, Any] = {}
    for cluster_json in sorted(source_run_dir.glob("clusters_*.json")):
        payload = json.loads(cluster_json.read_text(encoding="utf-8"))
        nm_id = int(payload["nm_id"])
        category_id = str(payload["category_id"])
        centroid_path = source_run_dir / f"cluster_centroids_{nm_id}.npy"
        centroid_matrix = np.load(centroid_path) if centroid_path.exists() else np.empty((0, 0), dtype=np.float32)

        cluster_rows = sorted(payload.get("clusters", []), key=lambda item: int(item["cluster_id"]))
        clusters: dict[int, Any] = {}
        for index, row in enumerate(cluster_rows):
            cluster_id = int(row["cluster_id"])
            top_pairs = row.get("top_phrases", []) or []
            top_phrases = [str(item[0]) for item in top_pairs if item]
            top_phrase_weights = {
                str(item[0]): int(item[1])
                for item in top_pairs
                if isinstance(item, (list, tuple)) and len(item) >= 2
            }
            centroid = centroid_matrix[index] if index < len(centroid_matrix) else None
            clusters[cluster_id] = ref.ClusterInfo(
                cluster_id=cluster_id,
                top_phrases=top_phrases,
                top_phrase_weights=top_phrase_weights,
                centroid=centroid,
                medoid=str(row.get("medoid_phrase") or ""),
                gold_matches=dict(row.get("gold_matches") or {}),
            )
        discovery_by_product[nm_id] = ref.ProductDiscoveryInfo(
            nm_id=nm_id,
            category_id=category_id,
            clusters=clusters,
        )
    return discovery_by_product


def _hydrate_reviews_from_assignments(context: sentiment_benchmark.BenchmarkContext) -> None:
    for review in context.reviews:
        review.vocab_aspect_ids = set()
        review.discovery_cluster_ids = set()
    for row in context.assignments.itertuples(index=False):
        review = context.reviews_by_id.get(str(row.review_id))
        if review is None:
            continue
        aspect_source = str(row.aspect_source)
        aspect_key = str(row.aspect_key)
        if aspect_source == "vocab":
            review.vocab_aspect_ids.add(aspect_key.split("::", 1)[1])
            continue
        if aspect_source == "discovery":
            parts = aspect_key.split("::")
            if len(parts) >= 3:
                review.discovery_cluster_ids.add(int(parts[2]))


def _find_latest_mode_result(source_run_dir: Path, mode_id: str) -> Path | None:
    root = ROOT / "benchmark" / "sentiment" / mode_id / "results"
    if not root.exists():
        return None
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    source_text = str(source_run_dir.resolve())
    for path in candidates:
        summary_path = path / "summary.json"
        preds_path = path / "nli_predictions.parquet"
        scores_path = path / "review_aspect_scores.csv"
        if not summary_path.exists() or not preds_path.exists() or not scores_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("source_run_dir", "")).strip() == source_text:
            return path
    return None


def _load_or_compute_mode_result(
    context: sentiment_benchmark.BenchmarkContext,
    *,
    mode_id: str,
    mode_run_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path | None]:
    mode_meta = _mode_meta(mode_id)
    chosen_dir = mode_run_dir.resolve() if mode_run_dir is not None else _find_latest_mode_result(context.run_dir, mode_id)
    if chosen_dir is not None:
        predictions = pd.read_parquet(chosen_dir / "nli_predictions.parquet")
        review_scores = pd.read_csv(chosen_dir / "review_aspect_scores.csv")
        summary = json.loads((chosen_dir / "summary.json").read_text(encoding="utf-8"))
        return predictions, review_scores, summary, chosen_dir

    inputs = sentiment_benchmark.build_mode_input_frame(context, mode_id)
    predictions = sentiment_benchmark.run_inference(
        inputs,
        context,
        mode_id,
    )
    aggregation = "weighted_relevance" if mode_id == sentiment_benchmark.MODE_D_WEIGHTED else "max_relevance"
    review_scores = sentiment_benchmark.aggregate_review_aspect_scores(predictions, aggregation=aggregation)
    metrics, _ = sentiment_benchmark.evaluate_review_level(context, review_scores)
    summary = {
        "mode_id": mode_id,
        "source_run_dir": str(context.run_dir),
        "dataset_path": str(context.dataset_path),
        "counts": {
            "input_rows": int(len(inputs)),
            "predictions": int(len(predictions)),
            "kept_after_threshold": int(predictions["passed_relevance_filter"].sum()) if not predictions.empty else 0,
            "review_aspect_scores": int(len(review_scores)),
            "discovery_assignments_without_evidence": int(context.discovery_assignments_without_evidence),
        },
        "metrics": metrics,
        "config": {
            "window_tokens": sentiment_benchmark.DEFAULT_WINDOW_TOKENS,
            "relevance_threshold": sentiment_benchmark.DEFAULT_RELEVANCE_THRESHOLD,
            "temperature": sentiment_benchmark.DEFAULT_TEMPERATURE,
            "aggregation": aggregation,
        },
    }
    return predictions, review_scores, summary, None


def _compat_nli_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "prediction_id",
                "review_id",
                "nm_id",
                "aspect_name",
                "aspect_key",
                "aspect_source",
                "hypothesis_text",
                "hypothesis_pos_text",
                "hypothesis_neg_text",
                "premise_text",
                "premise_kind",
                "evidence_id",
                "p_entailment",
                "p_neutral",
                "p_contradiction",
                "raw_rating",
                "passed_relevance_filter",
                "relevance_filter_value",
                "has_negation_match",
                "negation_correction_applied",
                "final_rating",
            ]
        )

    rows: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        rows.append(
            {
                "prediction_id": sentiment_benchmark.stable_id(str(row.review_id), str(row.aspect_key), str(row.evidence_id)),
                "review_id": str(row.review_id),
                "nm_id": int(row.nm_id),
                "aspect_name": str(row.aspect_name),
                "aspect_key": str(row.aspect_key),
                "aspect_source": str(row.aspect_source),
                "hypothesis_text": str(row.hypothesis_pos_text),
                "hypothesis_pos_text": str(row.hypothesis_pos_text),
                "hypothesis_neg_text": str(row.hypothesis_neg_text),
                "premise_text": str(row.premise_text),
                "premise_kind": str(row.premise_kind),
                "evidence_id": str(row.evidence_id),
                "p_entailment": float(row.p_entailment_pos),
                "p_neutral": float(row.p_neutral),
                "p_contradiction": float(row.p_entailment_neg),
                "raw_rating": float(row.raw_rating),
                "passed_relevance_filter": bool(row.passed_relevance_filter),
                "relevance_filter_value": float(row.relevance_filter_value),
                "has_negation_match": False,
                "negation_correction_applied": False,
                "final_rating": float(row.final_rating),
            }
        )
    return pd.DataFrame(rows)


def _sentiment_by_pair_from_mode_results(
    predictions: pd.DataFrame,
    review_aspect_scores: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, float]]:
    if predictions.empty or review_aspect_scores.empty:
        return {}

    by_selected_evidence = {
        (str(row.review_id), str(row.aspect_key), str(row.evidence_id)): row
        for row in predictions.itertuples(index=False)
    }
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in review_aspect_scores.itertuples(index=False):
        key = (str(row.review_id), str(row.aspect_key))
        pred = by_selected_evidence.get((str(row.review_id), str(row.aspect_key), str(row.selected_evidence_id)))
        if pred is None:
            continue
        rating = float(row.final_rating)
        p_pos = float(pred.p_entailment_pos)
        p_neg = float(pred.p_entailment_neg)
        out[key] = {
            "rating": rating,
            "raw_rating": float(pred.raw_rating),
            "p_ent_pos": p_pos,
            "p_ent_neg": p_neg,
            "p_neutral": float(pred.p_neutral),
            "p_ent_plus_contra": float(pred.relevance_filter_value),
            "polarity": rating - 3.0,
            "raw_polarity": float(pred.raw_rating) - 3.0,
            "negation_corrected": False,
            "negation_pattern": "",
            "negation_hit_lemma": "",
        }
    return out


def _sentiment_by_pair_from_mode_b(
    predictions: pd.DataFrame,
    review_aspect_scores: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, float]]:
    return _sentiment_by_pair_from_mode_results(predictions, review_aspect_scores)


def _zero_negation_stats(
    *,
    total_predictions: int,
    kept_predictions: int,
    reviews: list[Any],
) -> dict[str, Any]:
    per_category: dict[str, int] = {}
    for review in reviews:
        per_category.setdefault(str(review.category_id), 0)
    return {
        "total_predictions": int(total_predictions),
        "kept_predictions": int(kept_predictions),
        "eligible_low_high_rows": 0,
        "corrections_applied": 0,
        "correction_rate": 0.0,
        "inversion_rate": 0.0,
        "per_category": per_category,
    }


def _write_run_log(
    out_dir: Path,
    *,
    source_run_dir: Path,
    sentiment_mode_id: str,
    sentiment_source_dir: Path | None,
    sentiment_summary: dict[str, Any],
    metrics_payload: dict[str, Any],
) -> None:
    mode_meta = _mode_meta(sentiment_mode_id)
    lines = [
        f"[start] freeze_final_results out_dir={out_dir}",
        f"[source] traced_run={source_run_dir}",
        f"[sentiment] mode={sentiment_mode_id}",
        f"[sentiment] label={mode_meta['label']}",
        f"[sentiment] reused_sentiment_dir={sentiment_source_dir if sentiment_source_dir is not None else 'fresh_compute'}",
        f"[sentiment] review_mae={sentiment_summary.get('metrics', {}).get('sentiment_mae_review')}",
        f"[track_a] review_mae={metrics_payload.get('track_a', {}).get('sentiment_mae_review')}",
        f"[track_b] review_mae={metrics_payload.get('track_b', {}).get('sentiment_mae_review')}",
        "[done] final_res_v2 assembled from frozen traced artifacts",
    ]
    (out_dir / "run_console.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _derived_run_config(source_config: dict[str, Any], source_run_dir: Path, sentiment_mode_id: str) -> dict[str, Any]:
    mode_meta = _mode_meta(sentiment_mode_id)
    config = copy.deepcopy(source_config)
    config["run_name"] = mode_meta["run_name"]
    config["derived_from_run_dir"] = str(source_run_dir)
    config.setdefault("sentiment", {})
    config["sentiment"]["mode_id"] = sentiment_mode_id
    config["sentiment"]["premise_kind"] = mode_meta["premise_kind"]
    config["sentiment"]["hypothesis_mode"] = mode_meta["hypothesis_mode"]
    config["sentiment"]["hypothesis_template_pos"] = mode_meta["hypothesis_template_pos"]
    config["sentiment"]["hypothesis_template_neg"] = mode_meta["hypothesis_template_neg"]
    config["sentiment"]["relevance_mode"] = mode_meta["relevance_mode"]
    if mode_meta["hypothesis_mode"] == "dual_hypothesis":
        config["sentiment"]["raw_formula"] = "3 + 2 * (p_pos - p_neg) / (p_pos + p_neg + eps)"
    else:
        config["sentiment"]["raw_formula"] = "single_hypothesis_engine_score"
    config["sentiment"]["negation_correction"] = {"enabled": False}
    return config


def _write_manifest(
    out_dir: Path,
    *,
    source_run_dir: Path,
    source_config: dict[str, Any],
    reviews: list[Any],
    metrics_payload: dict[str, Any],
    sentiment_mode_id: str,
    sentiment_summary: dict[str, Any],
) -> None:
    mode_meta = _mode_meta(sentiment_mode_id)
    manifest = {
        "run_id": out_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "derived_from": {
            "source_run_dir": str(source_run_dir),
            "source_run_id": source_run_dir.name,
            "sentiment_mode_id": sentiment_mode_id,
            "sentiment_mode_label": mode_meta["label"],
        },
        "config": _derived_run_config(source_config, source_run_dir, sentiment_mode_id),
        "n_reviews_processed": len(reviews),
        "n_products": len({review.nm_id for review in reviews}),
        "n_categories": len({review.category_id for review in reviews}),
        "stages_completed": [
            "reuse_s1",
            "reuse_s2",
            "reuse_s3",
            "reuse_s4",
            mode_meta["derived_stage"],
            "derive_s6",
            "evaluate_run",
        ],
        "artifact_files": {
            "candidates": "candidates.parquet",
            "candidate_matches": "candidate_matches.parquet",
            "embeddings_candidates": "embeddings_candidates.npy",
            "embedding_index_candidates": "embedding_index_candidates.csv",
            "embeddings_vocab": "embeddings_vocab.npy",
            "embedding_index_vocab": "embedding_index_vocab.csv",
            "clusters": "clusters_<nm_id>.json",
            "cluster_centroids": "cluster_centroids_<nm_id>.npy",
            "nli_predictions": "nli_predictions.parquet",
            "product_aggregates": "product_aggregates.parquet",
        },
        "metrics": metrics_payload,
        "source_sentiment_summary": sentiment_summary,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_final_results(
    *,
    source_run_dir: Path,
    final_v1_dir: Path,
    final_v2_dir: Path,
    manual_audit_v2_dir: Path | None,
    sentiment_mode_id: str = DEFAULT_FINAL_V2_MODE_ID,
    sentiment_run_dir: Path | None = None,
    skip_v1: bool = False,
    skip_manual_audit: bool = False,
) -> dict[str, Any]:
    source_run_dir = source_run_dir.resolve()
    final_v1_dir = final_v1_dir.resolve()
    final_v2_dir = final_v2_dir.resolve()
    manual_audit_v2_dir = None if manual_audit_v2_dir is None else manual_audit_v2_dir.resolve()

    if not source_run_dir.exists():
        raise FileNotFoundError(f"source traced run not found: {source_run_dir}")
    _mode_meta(sentiment_mode_id)

    if not skip_v1:
        _copy_full_run(source_run_dir, final_v1_dir)

    _reset_dir(final_v2_dir)
    _copy_static_traced_artifacts(source_run_dir, final_v2_dir)

    context = sentiment_benchmark.load_benchmark_context(source_run_dir)
    _hydrate_reviews_from_assignments(context)
    discovery_by_product = _load_discovery_from_traced(source_run_dir)
    predictions, review_scores, sentiment_summary, used_sentiment_dir = _load_or_compute_mode_result(
        context,
        mode_id=sentiment_mode_id,
        mode_run_dir=sentiment_run_dir,
    )

    sentiment_by_pair = _sentiment_by_pair_from_mode_results(predictions, review_scores)
    compat_predictions = _compat_nli_predictions(predictions)

    writer = ArtifactWriter(final_v2_dir)
    writer.write_dataframe("nli_predictions.parquet", compat_predictions, sort_by=["review_id", "aspect_source", "aspect_name"])

    agg = s6_aggregation.run_stage(
        context.reviews,
        sentiment_by_pair,
        context.aspect_by_id_by_category,
        discovery_by_product,
    )
    writer.write_dataframe(
        "product_aggregates.parquet",
        agg["product_aggregates"],
        sort_by=["nm_id", "aspect_source", "aspect_name"],
    )

    negation_stats = _zero_negation_stats(
        total_predictions=len(compat_predictions),
        kept_predictions=len(sentiment_by_pair),
        reviews=context.reviews,
    )
    metrics_payload = orchestrator._write_e2e_compatible_outputs(
        out_dir=final_v2_dir,
        reviews=context.reviews,
        term_to_aspects_by_category=context.term_to_aspects_by_category,
        aspect_by_id_by_category=context.aspect_by_id_by_category,
        discovery_by_product=discovery_by_product,
        sentiment_by_pair=sentiment_by_pair,
        aggregated=agg["aggregated"],
        negation_stats=negation_stats,
    )

    run_summary = {
        "status": "OK",
        "out_dir": str(final_v2_dir),
        "elapsed_sec": None,
        "derived_from_run_dir": str(source_run_dir),
        "derived_sentiment_mode_id": sentiment_mode_id,
        "derived_sentiment_mode_label": _mode_meta(sentiment_mode_id)["label"],
        "reused_sentiment_run_dir": str(used_sentiment_dir) if used_sentiment_dir is not None else None,
        "source_sentiment_metrics": sentiment_summary.get("metrics", {}),
        **metrics_payload,
    }
    writer.write_json("run_summary.json", run_summary)

    source_config = yaml.safe_load((source_run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    (final_v2_dir / "run_config.yaml").write_text(
        yaml.safe_dump(_derived_run_config(source_config, source_run_dir, sentiment_mode_id), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_run_log(
        final_v2_dir,
        source_run_dir=source_run_dir,
        sentiment_mode_id=sentiment_mode_id,
        sentiment_source_dir=used_sentiment_dir,
        sentiment_summary=sentiment_summary,
        metrics_payload=metrics_payload,
    )
    _write_manifest(
        final_v2_dir,
        source_run_dir=source_run_dir,
        source_config=source_config,
        reviews=context.reviews,
        metrics_payload=metrics_payload,
        sentiment_mode_id=sentiment_mode_id,
        sentiment_summary=sentiment_summary,
    )

    metrics_report = write_metrics_report(final_v2_dir)
    if not skip_manual_audit and manual_audit_v2_dir is not None:
        _reset_dir(manual_audit_v2_dir)
        build_queue(
            run_dir=final_v2_dir,
            dataset_path=ROOT / "data" / "dataset_final.csv",
            merge_map_path=ROOT / "aspect_merge_map.json",
            out_dir=manual_audit_v2_dir,
        )

    return {
        "source_run_dir": str(source_run_dir),
        "final_v1_dir": None if skip_v1 else str(final_v1_dir),
        "final_v2_dir": str(final_v2_dir),
        "manual_audit_v2_dir": None if skip_manual_audit or manual_audit_v2_dir is None else str(manual_audit_v2_dir),
        "sentiment_mode_id": sentiment_mode_id,
        "sentiment_run_dir": str(used_sentiment_dir) if used_sentiment_dir is not None else None,
        "sentiment_metrics": sentiment_summary.get("metrics", {}),
        "final_v2_metrics_report": metrics_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze stable final_res_v1 / final_res_v2 from traced artifacts")
    parser.add_argument("--source-run-dir", default=str(SOURCE_RUN_DEFAULT))
    parser.add_argument("--final-v1-dir", default=str(FINAL_V1_DEFAULT))
    parser.add_argument("--final-v2-dir", default=str(FINAL_V2_DEFAULT))
    parser.add_argument("--manual-audit-v2-dir", default=str(MANUAL_AUDIT_V2_DEFAULT))
    parser.add_argument("--sentiment-mode-id", default=DEFAULT_FINAL_V2_MODE_ID)
    parser.add_argument("--sentiment-run-dir", default="")
    parser.add_argument("--skip-v1", action="store_true")
    parser.add_argument("--skip-manual-audit", action="store_true")
    args = parser.parse_args()

    payload = build_final_results(
        source_run_dir=Path(args.source_run_dir),
        final_v1_dir=Path(args.final_v1_dir),
        final_v2_dir=Path(args.final_v2_dir),
        manual_audit_v2_dir=None if args.skip_manual_audit else Path(args.manual_audit_v2_dir),
        sentiment_mode_id=str(args.sentiment_mode_id),
        sentiment_run_dir=Path(args.sentiment_run_dir) if args.sentiment_run_dir else None,
        skip_v1=bool(args.skip_v1),
        skip_manual_audit=bool(args.skip_manual_audit),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
