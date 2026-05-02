from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.sentiment import common as sentiment_benchmark
from scripts import compare_final_sentiment_runs as diagnostics
from scripts import run_phase2_baseline_matching as lexical

MODE_IDS = [
    sentiment_benchmark.MODE_A,
    sentiment_benchmark.MODE_B,
    sentiment_benchmark.MODE_C,
    sentiment_benchmark.MODE_D,
    sentiment_benchmark.MODE_D_WEIGHTED,
]
OUTPUT_ROOT_DEFAULT = ROOT / "benchmark" / "sentiment" / "mode_abcd_diagnostics" / "results"
SCOPE_OWN = diagnostics.SCOPE_OWN
SCOPE_COMMON = diagnostics.SCOPE_COMMON
BY_CATEGORY_COLUMNS = [
    "scope",
    "mode_id",
    "category_id",
    "n_pairs",
    "mae",
    "accuracy_at_1",
    "direction_accuracy",
    "wrong_polarity_rate",
    "mean_signed_error",
]
BY_PRODUCT_COLUMNS = [
    "scope",
    "mode_id",
    "nm_id",
    "category_id",
    "n_pairs",
    "mae",
    "accuracy_at_1",
    "direction_accuracy",
    "wrong_polarity_rate",
    "mean_signed_error",
]
COMPARISON_COLUMNS = [
    "scope",
    "mode_id",
    "legacy_review_mae",
    "legacy_review_mae_round",
    "legacy_vocab_pair_mae",
    "legacy_discovery_pair_mae",
    "legacy_evaluable_pair_coverage",
    "runtime_sec",
    "input_rows",
    "predictions",
    "kept_after_threshold",
    "review_aspect_scores",
    "discovery_assignments_without_evidence",
    *diagnostics.RUN_METRIC_COLUMNS,
]


@dataclass(slots=True)
class BenchmarkModeArtifacts:
    mode_id: str
    benchmark_dir: Path
    source_run_dir: Path
    context: sentiment_benchmark.BenchmarkContext
    review_aspect_scores: pd.DataFrame
    nli_predictions: pd.DataFrame
    evaluated_pairs: pd.DataFrame
    total_gold_pairs: int
    summary_payload: dict[str, Any]


def _latest_result_dir(mode_id: str) -> Path:
    root = ROOT / "benchmark" / "sentiment" / mode_id / "results"
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no result dirs under {root}")
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_load_list(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _json_load_dict(raw: Any) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _pair_key_tuples(df: pd.DataFrame) -> set[tuple[Any, ...]]:
    if df.empty:
        return set()
    columns = diagnostics.PAIR_KEY_COLUMNS
    return {
        tuple(getattr(row, column) for column in columns)
        for row in df[columns].itertuples(index=False)
    }


def common_pair_keys(frames_by_mode: dict[str, pd.DataFrame]) -> set[tuple[Any, ...]]:
    if not frames_by_mode:
        return set()
    iterator = iter(frames_by_mode.values())
    common = _pair_key_tuples(next(iterator))
    for frame in iterator:
        common &= _pair_key_tuples(frame)
    return common


def _filter_to_pair_keys(df: pd.DataFrame, pair_keys: set[tuple[Any, ...]]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    columns = diagnostics.PAIR_KEY_COLUMNS
    mask = pd.Series(
        [
            tuple(getattr(row, column) for column in columns) in pair_keys
            for row in df[columns].itertuples(index=False)
        ],
        index=df.index,
    )
    return df[mask].copy().reset_index(drop=True)


def _prediction_lookup(
    review_aspect_scores: pd.DataFrame,
    nli_predictions: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    by_evidence = {
        str(row.evidence_id): row._asdict()
        for row in nli_predictions.itertuples(index=False)
    }
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in nli_predictions.itertuples(index=False):
        by_pair[(str(row.review_id), str(row.aspect_key))].append(row._asdict())

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in review_aspect_scores.itertuples(index=False):
        review_id = str(row.review_id)
        aspect_key = str(row.aspect_key)
        selected_ids = _json_load_list(getattr(row, "selected_evidence_ids_json", ""))
        evidence_rows = [by_evidence[evidence_id] for evidence_id in selected_ids if evidence_id in by_evidence]
        if not evidence_rows:
            evidence_rows = by_pair.get((review_id, aspect_key), [])
        premise_texts = _unique_preserve([str(item.get("premise_text", "")) for item in evidence_rows if str(item.get("premise_text", "")).strip()])
        premise_kinds = _unique_preserve([str(item.get("premise_kind", "")) for item in evidence_rows if str(item.get("premise_kind", "")).strip()])
        lookup[(review_id, aspect_key)] = {
            "review_id": review_id,
            "nm_id": int(row.nm_id),
            "category_id": str(row.category_id),
            "aspect_key": aspect_key,
            "aspect_name": str(row.aspect_name),
            "aspect_source": str(row.aspect_source),
            "rating": float(row.final_rating),
            "raw_rating": float(row.final_rating),
            "n_evidence_total": int(getattr(row, "n_evidence_total", 0)),
            "n_evidence_kept": int(getattr(row, "n_evidence_kept", 0)),
            "aggregation_method": str(getattr(row, "aggregation_method", "")),
            "selected_evidence_ids": selected_ids,
            "premise_texts": premise_texts,
            "premise_kinds": premise_kinds,
            "gold_matches": _json_load_dict(getattr(row, "gold_matches_json", "{}")),
        }
    return lookup


def _build_pair_evaluations(
    context: sentiment_benchmark.BenchmarkContext,
    review_aspect_scores: pd.DataFrame,
    nli_predictions: pd.DataFrame,
) -> pd.DataFrame:
    by_review_aspect = _prediction_lookup(review_aspect_scores, nli_predictions)
    by_review: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in by_review_aspect.values():
        by_review[str(payload["review_id"])].append(payload)

    rows: list[dict[str, Any]] = []
    for review in context.reviews:
        term_to_aspects = context.term_to_aspects_by_category[review.category_id]
        for gold_label, gold_rating in review.true_labels.items():
            mapped_ids = sorted(term_to_aspects.get(lexical._normalize(gold_label), set()))
            predicted_rows: list[dict[str, Any]] = []
            for aspect_id in mapped_ids:
                payload = by_review_aspect.get((str(review.review_id), f"vocab::{aspect_id}"))
                if payload is not None:
                    predicted_rows.append(payload)
            if not mapped_ids:
                for payload in by_review.get(str(review.review_id), []):
                    if str(payload["aspect_source"]) != "discovery":
                        continue
                    if gold_label in payload["gold_matches"]:
                        predicted_rows.append(payload)
            if not predicted_rows:
                continue

            predicted_rating = sum(float(item["rating"]) for item in predicted_rows) / len(predicted_rows)
            raw_predicted_rating = sum(float(item["raw_rating"]) for item in predicted_rows) / len(predicted_rows)
            premise_texts: list[str] = []
            premise_kinds: list[str] = []
            selected_ids: list[str] = []
            for item in predicted_rows:
                premise_texts.extend(item["premise_texts"])
                premise_kinds.extend(item["premise_kinds"])
                selected_ids.extend(item["selected_evidence_ids"])
            rows.append(
                {
                    "review_id": str(review.review_id),
                    "nm_id": int(review.nm_id),
                    "category_id": str(review.category_id),
                    "review_rating": float(review.rating),
                    "review_text": str(review.text),
                    "gold_label": str(gold_label),
                    "gold_rating": float(gold_rating),
                    "predicted_rating": float(predicted_rating),
                    "raw_predicted_rating": float(raw_predicted_rating),
                    "aspect_source": "discovery" if any(str(item["aspect_source"]) == "discovery" for item in predicted_rows) else "vocab",
                    "n_predicted_items": int(len(predicted_rows)),
                    "predicted_keys_json": json.dumps([str(item["aspect_key"]) for item in predicted_rows], ensure_ascii=False),
                    "predicted_aspect_names_json": json.dumps([str(item["aspect_name"]) for item in predicted_rows], ensure_ascii=False),
                    "premise_texts_json": json.dumps(_unique_preserve(premise_texts), ensure_ascii=False),
                    "premise_kinds_json": json.dumps(_unique_preserve(premise_kinds), ensure_ascii=False),
                    "selected_evidence_ids_json": json.dumps(_unique_preserve(selected_ids), ensure_ascii=False),
                    "n_evidence_total_sum": int(sum(int(item["n_evidence_total"]) for item in predicted_rows)),
                    "n_evidence_kept_sum": int(sum(int(item["n_evidence_kept"]) for item in predicted_rows)),
                    "aggregation_methods_json": json.dumps(_unique_preserve([str(item["aggregation_method"]) for item in predicted_rows]), ensure_ascii=False),
                }
            )
    return pd.DataFrame(rows)


def _load_mode_artifacts(mode_id: str, benchmark_dir: Path) -> BenchmarkModeArtifacts:
    summary_payload = _load_json(benchmark_dir / "summary.json")
    source_run_dir = Path(summary_payload["source_run_dir"]).resolve()
    dataset_path = Path(summary_payload["dataset_path"]).resolve()
    window_tokens = int(summary_payload.get("config", {}).get("window_tokens", sentiment_benchmark.DEFAULT_WINDOW_TOKENS))
    context = sentiment_benchmark.load_benchmark_context(
        run_dir=source_run_dir,
        dataset_path=dataset_path,
        window_tokens=window_tokens,
    )
    review_aspect_scores = pd.read_csv(benchmark_dir / "review_aspect_scores.csv")
    nli_predictions = pd.read_parquet(benchmark_dir / "nli_predictions.parquet")
    evaluated_pairs = _build_pair_evaluations(context, review_aspect_scores, nli_predictions)
    total_gold_pairs = int(sum(len(review.true_labels) for review in context.reviews))
    evaluated_pairs = diagnostics.enrich_pair_rows(evaluated_pairs, total_gold_pairs=total_gold_pairs)
    return BenchmarkModeArtifacts(
        mode_id=mode_id,
        benchmark_dir=benchmark_dir,
        source_run_dir=source_run_dir,
        context=context,
        review_aspect_scores=review_aspect_scores,
        nli_predictions=nli_predictions,
        evaluated_pairs=evaluated_pairs,
        total_gold_pairs=total_gold_pairs,
        summary_payload=summary_payload,
    )


def _comparison_row(artifacts: BenchmarkModeArtifacts, *, scope: str, df: pd.DataFrame) -> dict[str, Any]:
    payload = artifacts.summary_payload
    row = {
        "scope": scope,
        "mode_id": artifacts.mode_id,
        "legacy_review_mae": payload["metrics"]["sentiment_mae_review"],
        "legacy_review_mae_round": payload["metrics"]["sentiment_mae_review_round"],
        "legacy_vocab_pair_mae": payload["metrics"]["sentiment_mae_vocab_pairs"],
        "legacy_discovery_pair_mae": payload["metrics"]["sentiment_mae_discovery_pairs"],
        "legacy_evaluable_pair_coverage": payload["metrics"]["evaluable_pair_coverage"],
        "runtime_sec": payload["runtime_sec"],
        "input_rows": payload["counts"]["input_rows"],
        "predictions": payload["counts"]["predictions"],
        "kept_after_threshold": payload["counts"]["kept_after_threshold"],
        "review_aspect_scores": payload["counts"]["review_aspect_scores"],
        "discovery_assignments_without_evidence": payload["counts"]["discovery_assignments_without_evidence"],
    }
    row.update(diagnostics.compute_run_metrics(df, total_gold_pairs=artifacts.total_gold_pairs))
    return row


def _group_metrics(
    df: pd.DataFrame,
    *,
    group_columns: list[str],
    scope: str,
    mode_id: str,
    total_gold_pairs: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["scope", "mode_id", *group_columns, "n_pairs", "mae", "accuracy_at_1", "direction_accuracy", "wrong_polarity_rate", "mean_signed_error"])
    rows: list[dict[str, Any]] = []
    for group_values, group in df.groupby(group_columns, sort=True):
        values = group_values if isinstance(group_values, tuple) else (group_values,)
        metrics = diagnostics.compute_run_metrics(group, total_gold_pairs=total_gold_pairs)
        row = {"scope": scope, "mode_id": mode_id}
        row.update(dict(zip(group_columns, values, strict=True)))
        row["n_pairs"] = int(len(group))
        row["mae"] = metrics["mae"]
        row["accuracy_at_1"] = metrics["accuracy_at_1_0"]
        row["direction_accuracy"] = metrics["direction_accuracy"]
        row["wrong_polarity_rate"] = metrics["wrong_polarity_rate"]
        row["mean_signed_error"] = metrics["mean_signed_error"]
        rows.append(row)
    return pd.DataFrame(rows)


def _pair_presence_frame(frames_by_mode: dict[str, pd.DataFrame]) -> pd.DataFrame:
    union_keys: set[tuple[Any, ...]] = set()
    for frame in frames_by_mode.values():
        union_keys |= _pair_key_tuples(frame)
    rows: list[dict[str, Any]] = []
    key_columns = diagnostics.PAIR_KEY_COLUMNS
    for key in sorted(union_keys):
        row = dict(zip(key_columns, key, strict=True))
        present_count = 0
        for mode_id, frame in frames_by_mode.items():
            present = key in _pair_key_tuples(frame)
            row[f"present_in_{mode_id}"] = present
            present_count += int(present)
        row["n_modes_present"] = present_count
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_markdown(
    *,
    out_dir: Path,
    comparison_df: pd.DataFrame,
    hard_breakdown_df: pd.DataFrame,
) -> str:
    own = comparison_df[comparison_df["scope"] == SCOPE_OWN].copy()
    common = comparison_df[comparison_df["scope"] == SCOPE_COMMON].copy()
    own_mae = own.sort_values("mae")[["mode_id", "mae", "accuracy_at_1_0", "wrong_polarity_rate", "coverage"]]
    own_rmse = own.sort_values("rmse")[["mode_id", "rmse", "strong_wrong_polarity_rate"]]
    common_mae = common.sort_values("mae")[["mode_id", "mae", "accuracy_at_1_0", "wrong_polarity_rate"]]
    lines = [
        "# Sentiment Benchmark A/B/C/D Diagnostics",
        "",
        f"- out_dir: {out_dir}",
        "",
        "## Own pairs by MAE",
    ]
    for row in own_mae.itertuples(index=False):
        lines.append(
            f"- {row.mode_id}: mae={row.mae:.4f}, acc@1={row.accuracy_at_1_0:.4f}, wrong_polarity={row.wrong_polarity_rate:.4f}, coverage={row.coverage:.4f}"
        )
    lines.extend(["", "## Own pairs by RMSE"])
    for row in own_rmse.itertuples(index=False):
        lines.append(
            f"- {row.mode_id}: rmse={row.rmse:.4f}, strong_wrong_polarity={row.strong_wrong_polarity_rate:.4f}"
        )
    lines.extend(["", "## Common pairs by MAE"])
    for row in common_mae.itertuples(index=False):
        lines.append(
            f"- {row.mode_id}: mae={row.mae:.4f}, acc@1={row.accuracy_at_1_0:.4f}, wrong_polarity={row.wrong_polarity_rate:.4f}"
        )
    lines.extend(["", "## Hard-case source mix"])
    hard_source = hard_breakdown_df[hard_breakdown_df["dimension"] == "aspect_source"]
    for mode_id in MODE_IDS:
        subset = hard_source[hard_source["mode_id"] == mode_id]
        if subset.empty:
            continue
        payload = ", ".join(f"{row.value}={row.count}" for row in subset.itertuples(index=False))
        lines.append(f"- {mode_id}: {payload}")
    return "\n".join(lines) + "\n"


def run_comparison(
    *,
    benchmark_dirs: dict[str, Path],
    out_root: Path,
) -> Path:
    started = datetime.now(timezone.utc)
    artifacts_by_mode = {
        mode_id: _load_mode_artifacts(mode_id, benchmark_dirs[mode_id].resolve())
        for mode_id in MODE_IDS
    }
    total_gold_pairs = {artifact.total_gold_pairs for artifact in artifacts_by_mode.values()}
    if len(total_gold_pairs) != 1:
        raise ValueError("total gold pair count mismatch across modes")

    timestamp = started.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root.resolve() / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    own_frames = {mode_id: artifact.evaluated_pairs.copy() for mode_id, artifact in artifacts_by_mode.items()}
    common_keys_all = common_pair_keys(own_frames)
    common_frames = {
        mode_id: _filter_to_pair_keys(frame, common_keys_all)
        for mode_id, frame in own_frames.items()
    }
    frames_by_scope = {
        SCOPE_OWN: own_frames,
        SCOPE_COMMON: common_frames,
    }

    comparison_rows: list[dict[str, Any]] = []
    by_product_frames: list[pd.DataFrame] = []
    by_category_frames: list[pd.DataFrame] = []
    hard_top_frames: list[pd.DataFrame] = []
    hard_breakdown_frames: list[pd.DataFrame] = []

    for scope, frames in frames_by_scope.items():
        for mode_id in MODE_IDS:
            artifact = artifacts_by_mode[mode_id]
            frame = frames[mode_id]
            comparison_rows.append(_comparison_row(artifact, scope=scope, df=frame))
            by_product_frames.append(
                _group_metrics(
                    frame,
                    group_columns=["nm_id", "category_id"],
                    scope=scope,
                    mode_id=mode_id,
                    total_gold_pairs=artifact.total_gold_pairs,
                )
            )
            by_category_frames.append(
                _group_metrics(
                    frame,
                    group_columns=["category_id"],
                    scope=scope,
                    mode_id=mode_id,
                    total_gold_pairs=artifact.total_gold_pairs,
                )
            )
            diagnostics._write_matrix(out_dir / f"confusion_matrix_5x5_{mode_id}_{scope}.csv", diagnostics._confusion_matrix(frame))
            diagnostics._write_csv(out_dir / f"{mode_id}_{scope}_pair_rows.csv", frame)

    for mode_id in MODE_IDS:
        own_frame = own_frames[mode_id]
        top = diagnostics._hard_cases_top30(own_frame, run_id=mode_id).rename(columns={"run_id": "mode_id"})
        hard_top_frames.append(top)
        hard_breakdown = diagnostics._hard_case_breakdown(top.rename(columns={"mode_id": "run_id"}), run_id=mode_id)
        hard_breakdown = hard_breakdown.rename(columns={"run_id": "mode_id"})
        hard_breakdown_frames.append(hard_breakdown)

    comparison_df = pd.DataFrame(comparison_rows)[COMPARISON_COLUMNS].sort_values(["scope", "mae", "mode_id"]).reset_index(drop=True)
    by_product_df = pd.concat(by_product_frames, ignore_index=True)[BY_PRODUCT_COLUMNS].sort_values(["scope", "mode_id", "mae", "nm_id"], ascending=[True, True, False, True]).reset_index(drop=True)
    by_category_df = pd.concat(by_category_frames, ignore_index=True)[BY_CATEGORY_COLUMNS].sort_values(["scope", "mode_id", "mae", "category_id"], ascending=[True, True, False, True]).reset_index(drop=True)
    hard_top_df = pd.concat(hard_top_frames, ignore_index=True).sort_values(["mode_id", "abs_error", "review_id"], ascending=[True, False, True]).reset_index(drop=True)
    hard_breakdown_df = pd.concat(hard_breakdown_frames, ignore_index=True).sort_values(["mode_id", "dimension", "count", "value"], ascending=[True, True, False, True]).reset_index(drop=True)
    pair_presence_df = _pair_presence_frame(own_frames).sort_values(["n_modes_present", "nm_id", "review_id", "gold_label"], ascending=[True, True, True, True]).reset_index(drop=True)

    diagnostics._write_csv(out_dir / "sentiment_metrics_comparison.csv", comparison_df)
    diagnostics._write_csv(out_dir / "sentiment_metrics_by_product.csv", by_product_df)
    diagnostics._write_csv(out_dir / "sentiment_metrics_by_category.csv", by_category_df)
    diagnostics._write_csv(out_dir / "sentiment_hard_cases_top30.csv", hard_top_df)
    diagnostics._write_csv(out_dir / "sentiment_hard_cases_summary.csv", hard_breakdown_df)
    diagnostics._write_csv(out_dir / "sentiment_pair_presence.csv", pair_presence_df)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_dirs": {mode_id: str(path.resolve()) for mode_id, path in benchmark_dirs.items()},
        "total_gold_pairs": int(next(iter(total_gold_pairs))),
        "common_pair_count": int(len(common_keys_all)),
        "own_pair_counts": {mode_id: int(len(own_frames[mode_id])) for mode_id in MODE_IDS},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        _summary_markdown(
            out_dir=out_dir,
            comparison_df=comparison_df,
            hard_breakdown_df=hard_breakdown_df,
        ),
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved A/B/C/D sentiment benchmark runs with richer diagnostics")
    parser.add_argument("--mode-a-dir", default=str(_latest_result_dir(sentiment_benchmark.MODE_A)))
    parser.add_argument("--mode-b-dir", default=str(_latest_result_dir(sentiment_benchmark.MODE_B)))
    parser.add_argument("--mode-c-dir", default=str(_latest_result_dir(sentiment_benchmark.MODE_C)))
    parser.add_argument("--mode-d-dir", default=str(_latest_result_dir(sentiment_benchmark.MODE_D)))
    parser.add_argument("--mode-d-weighted-dir", default=str(_latest_result_dir(sentiment_benchmark.MODE_D_WEIGHTED)))
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT_DEFAULT))
    args = parser.parse_args()

    benchmark_dirs = {
        sentiment_benchmark.MODE_A: Path(args.mode_a_dir),
        sentiment_benchmark.MODE_B: Path(args.mode_b_dir),
        sentiment_benchmark.MODE_C: Path(args.mode_c_dir),
        sentiment_benchmark.MODE_D: Path(args.mode_d_dir),
        sentiment_benchmark.MODE_D_WEIGHTED: Path(args.mode_d_weighted_dir),
    }
    out_dir = run_comparison(
        benchmark_dirs=benchmark_dirs,
        out_root=Path(args.out_root),
    )
    print(out_dir)


if __name__ == "__main__":
    main()
