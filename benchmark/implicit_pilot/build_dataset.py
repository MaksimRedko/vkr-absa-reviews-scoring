from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    aspect_terms,
    clean_text,
    dump_json,
    encode_json_list,
    has_lexical_overlap,
    load_json,
    load_markup_frame,
    make_run_dir,
    matched_terms,
    profile_all_aspects,
    profile_selected_aspects,
    sample_records,
    seed_random,
    select_sentence_candidate,
    split_sentences,
)


def _build_record(row: pd.Series, aspect_cfg: Dict[str, Any], cls_name: str) -> Dict[str, Any]:
    review_text = clean_text(str(row["review_text"]))
    terms = aspect_terms(aspect_cfg)
    sentence_candidates = split_sentences(review_text)
    matched = matched_terms(review_text, terms)
    record = {
        "review_id": str(row["id"]),
        "product_id": int(row["nm_id"]),
        "rating": int(row["rating"]),
        "aspect_id": str(aspect_cfg["name"]),
        "aspect_type": str(aspect_cfg["type"]),
        "class": cls_name,
        "expected_p_ent": "high" if cls_name in {"EXPLICIT", "IMPLICIT"} else "low",
        "review_text": review_text,
        "sentence_text": select_sentence_candidate(review_text, terms),
        "sentence_candidates_json": encode_json_list(sentence_candidates),
        "lexicon_terms_json": encode_json_list(terms),
        "matched_terms_json": encode_json_list(matched),
        "lexical_overlap": bool(matched),
        "true_labels_json": json.dumps(row["true_labels_parsed"], ensure_ascii=False),
        "needs_manual_review": cls_name == "IMPLICIT",
    }
    return record


def _sample_aspect_rows(
    df: pd.DataFrame,
    aspect_cfg: Dict[str, Any],
    per_class_target: int,
    rng_seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    aspect_name = str(aspect_cfg["name"])
    terms = aspect_terms(aspect_cfg)
    positive_mask = df["true_labels_parsed"].apply(lambda labels: aspect_name in labels)
    overlap_mask = df["review_text"].apply(lambda text: has_lexical_overlap(text, terms))

    explicit_df = df[positive_mask & overlap_mask]
    implicit_df = df[positive_mask & (~overlap_mask)]
    unrelated_df = df[(~positive_mask) & (~overlap_mask)]

    rng = seed_random(rng_seed)
    sampled_explicit = sample_records(
        [_build_record(row, aspect_cfg, "EXPLICIT") for _, row in explicit_df.iterrows()],
        per_class_target,
        rng,
    )
    sampled_implicit = sample_records(
        [_build_record(row, aspect_cfg, "IMPLICIT") for _, row in implicit_df.iterrows()],
        per_class_target,
        rng,
    )
    sampled_unrelated = sample_records(
        [_build_record(row, aspect_cfg, "UNRELATED") for _, row in unrelated_df.iterrows()],
        per_class_target,
        rng,
    )

    stats = {
        "aspect": aspect_name,
        "aspect_type": str(aspect_cfg["type"]),
        "per_class_target": int(per_class_target),
        "explicit_pool": int(len(explicit_df)),
        "implicit_pool": int(len(implicit_df)),
        "unrelated_pool": int(len(unrelated_df)),
        "explicit_selected": int(len(sampled_explicit)),
        "implicit_selected": int(len(sampled_implicit)),
        "unrelated_selected": int(len(sampled_unrelated)),
        "lexicon_terms": terms,
    }
    return sampled_explicit + sampled_implicit + sampled_unrelated, stats


def build_dataset(config: Dict[str, Any], run_dir: Path) -> Path:
    dataset_cfg = dict(config.get("dataset", {}))
    csv_path = dataset_cfg["csv_path"]
    aspects = list(dataset_cfg["aspects"])
    per_class_target = int(dataset_cfg.get("samples_per_class", 10))
    seed = int(dataset_cfg.get("seed", 42))

    df = load_markup_frame(csv_path)
    profile_payload = {
        "csv_path": csv_path,
        "rows": int(len(df)),
        "all_aspects": profile_all_aspects(df),
        "selected_aspects": profile_selected_aspects(df, aspects),
    }
    dump_json(run_dir / "aspect_profile.json", profile_payload)

    all_records: list[Dict[str, Any]] = []
    sampling_stats: list[Dict[str, Any]] = []
    for idx, aspect_cfg in enumerate(aspects):
        aspect_records, aspect_stats = _sample_aspect_rows(
            df=df,
            aspect_cfg=aspect_cfg,
            per_class_target=per_class_target,
            rng_seed=seed + idx,
        )
        all_records.extend(aspect_records)
        sampling_stats.append(aspect_stats)

    dataset_df = pd.DataFrame(all_records)
    dataset_df = dataset_df.sort_values(
        by=["aspect_type", "aspect_id", "class", "review_id"],
        ignore_index=True,
    )
    output_path = run_dir / "pilot_dataset.csv"
    dataset_df.to_csv(output_path, index=False, encoding="utf-8")

    summary = {
        "dataset_rows": int(len(dataset_df)),
        "samples_per_class": per_class_target,
        "class_counts": dataset_df["class"].value_counts().sort_index().to_dict(),
        "aspect_counts": dataset_df["aspect_id"].value_counts().sort_index().to_dict(),
        "sampling": sampling_stats,
    }
    dump_json(run_dir / "dataset_summary.json", summary)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build implicit NLI pilot dataset.")
    parser.add_argument(
        "--config",
        default="experiments/implicit_nli_pilot.json",
        help="Path to pilot config JSON.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Existing results directory. If omitted, a new one is created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(prefix=config.get("name", "implicit_nli_pilot"))
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = build_dataset(config, run_dir)
    manifest = {
        "stage": "build_dataset",
        "config_path": str(args.config),
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
    }
    dump_json(run_dir / "build_manifest.json", manifest)
    print(f"[implicit_pilot] dataset={dataset_path}")


if __name__ == "__main__":
    main()
