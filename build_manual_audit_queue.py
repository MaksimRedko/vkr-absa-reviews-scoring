#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a full manual-audit queue from traced ABSA run artifacts.

Expected files:
  <run_dir>/nli_predictions.parquet
  <run_dir>/candidate_matches.parquet            optional, for vocab evidence
  <run_dir>/clusters_<nm_id>.json                optional, for discovery evidence
  data/dataset_final.csv                         gold labels + review text
  aspect_merge_map.json                          optional, compact family mapping

Output:
  manual_audit_queue_full.csv
  manual_audit_queue_passed.csv
  manual_audit_queue_sample.csv
  manual_audit_summary.md

Usage from repo root:
  python scripts/build_manual_audit_queue.py \
    --run-dir results/20260425_183110_traced \
    --dataset data/dataset_final.csv \
    --merge-map aspect_merge_map.json \
    --out-dir benchmark/manual_audit/final_v1
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_true_labels(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}
    try:
        obj = ast.literal_eval(text)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[str(k).strip()] = float(v)
        except Exception:
            continue
    return out


def counts_to_markdown(series: pd.Series, label: str) -> str:
    counts = series.value_counts(dropna=False)
    lines = [f"| {label} | count |", "|---|---:|"]
    for key, value in counts.items():
        rendered = "" if pd.isna(key) else str(key)
        lines.append(f"| {rendered} | {int(value)} |")
    return "\n".join(lines)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    required = {"id", "nm_id", "category", "rating", "full_text", "true_labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {sorted(missing)}")
    df = df.copy()
    df["review_id"] = df["id"].astype(str)
    df["true_labels_dict"] = df["true_labels"].apply(parse_true_labels)
    df["gold_aspects"] = df["true_labels_dict"].apply(lambda d: "|".join(sorted(d)))
    df["gold_aspect_ratings"] = df["true_labels_dict"].apply(
        lambda d: json.dumps(d, ensure_ascii=False, sort_keys=True)
    )
    return df[[
        "review_id", "nm_id", "category", "rating", "full_text",
        "gold_aspects", "gold_aspect_ratings", "true_labels_dict",
    ]]


def load_merge_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def aspect_key_from_row(row: pd.Series) -> str:
    source = str(row.get("aspect_source", ""))
    name = str(row.get("aspect_name", ""))
    if source == "vocab":
        return name
    return name


def exact_gold_hit(aspect_name: str, gold_dict: dict[str, float]) -> str:
    # Strict exact only. Manual reviewer can override later.
    if aspect_name in gold_dict:
        return aspect_name
    return ""


def rating_for_exact_hit(aspect_name: str, gold_dict: dict[str, float]) -> float | None:
    if aspect_name in gold_dict:
        return float(gold_dict[aspect_name])
    return None


def load_cluster_lookup(run_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    """
    Keys:
      (nm_id, aspect_name/medoid/matched gold label) -> cluster info
    This is fuzzy support for manual audit, not scoring.
    """
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for path in run_dir.glob("clusters_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        nm_id = int(payload.get("nm_id", path.stem.split("_")[-1]))
        clusters = payload.get("clusters") or payload.get("cluster_summaries") or []
        for c in clusters:
            info = {
                "cluster_id": c.get("cluster_id"),
                "medoid_phrase": c.get("medoid_phrase") or c.get("medoid") or "",
                "top_phrases": c.get("top_phrases", []),
                "matched_to_gold_aspect": c.get("matched_to_gold_aspect"),
                "is_novel": c.get("is_novel"),
                "gold_matches": c.get("gold_matches", {}),
            }
            keys = set()
            if info["medoid_phrase"]:
                keys.add(str(info["medoid_phrase"]))
            if info["matched_to_gold_aspect"]:
                keys.add(str(info["matched_to_gold_aspect"]))
            for g in (info.get("gold_matches") or {}).keys():
                keys.add(str(g))
            for k in keys:
                out[(nm_id, k)] = info
    return out


def compact_family(aspect_name: str, merge_map: dict[str, str]) -> str:
    return merge_map.get(aspect_name, "")


def make_cluster_evidence(nm_id: int, aspect_name: str, lookup: dict[tuple[int, str], dict[str, Any]]) -> str:
    info = lookup.get((int(nm_id), aspect_name))
    if not info:
        return ""
    phrases = info.get("top_phrases") or []
    # top_phrases can be list[list[str, int]]
    rendered: list[str] = []
    for item in phrases[:10]:
        if isinstance(item, (list, tuple)) and item:
            rendered.append(f"{item[0]}:{item[1] if len(item) > 1 else ''}")
        else:
            rendered.append(str(item))
    return "; ".join(rendered)


def build_queue(run_dir: Path, dataset_path: Path, merge_map_path: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    nli_path = run_dir / "nli_predictions.parquet"
    if not nli_path.exists():
        raise FileNotFoundError(f"missing {nli_path}")

    nli = pd.read_parquet(nli_path)
    data = load_dataset(dataset_path)
    merge_map = load_merge_map(merge_map_path)
    cluster_lookup = load_cluster_lookup(run_dir)

    # Keep original NLI columns; join review text/gold.
    df = nli.merge(
        data,
        on=["review_id", "nm_id"],
        how="left",
        validate="many_to_one",
    )

    if "passed_relevance_filter" in df.columns:
        df["passed_relevance_filter"] = df["passed_relevance_filter"].astype(bool)
    else:
        df["passed_relevance_filter"] = True

    df["manual_predicted_aspect"] = df["aspect_name"].astype(str)
    df["manual_compact_family"] = df["manual_predicted_aspect"].apply(lambda x: compact_family(x, merge_map))
    df["auto_exact_gold_hit"] = df.apply(
        lambda r: exact_gold_hit(str(r["aspect_name"]), r["true_labels_dict"] if isinstance(r["true_labels_dict"], dict) else {}),
        axis=1,
    )
    df["auto_exact_gold_rating"] = df.apply(
        lambda r: rating_for_exact_hit(str(r["aspect_name"]), r["true_labels_dict"] if isinstance(r["true_labels_dict"], dict) else {}),
        axis=1,
    )
    df["auto_abs_error_if_exact"] = df.apply(
        lambda r: abs(float(r["final_rating"]) - float(r["auto_exact_gold_rating"]))
        if pd.notna(r.get("auto_exact_gold_rating")) else None,
        axis=1,
    )

    df["cluster_evidence"] = df.apply(
        lambda r: make_cluster_evidence(int(r["nm_id"]), str(r["aspect_name"]), cluster_lookup)
        if str(r.get("aspect_source", "")) == "discovery" else "",
        axis=1,
    )

    # Blank columns for human audit.
    df["manual_gold_aspect"] = ""
    df["manual_decision"] = ""      # TP / FP / UNCLEAR / DUPLICATE / OUT_OF_SCOPE
    df["manual_sentiment_decision"] = ""  # OK / WRONG_POLARITY / TOO_HIGH / TOO_LOW / NOT_EVALUATED
    df["manual_error_type"] = ""    # wrong_aspect / too_broad / too_narrow / missing_gold / sentiment_context / negation / etc.
    df["manual_comment"] = ""

    preferred_cols = [
        "prediction_id",
        "review_id",
        "nm_id",
        "category",
        "rating",
        "aspect_source",
        "aspect_name",
        "manual_predicted_aspect",
        "manual_compact_family",
        "passed_relevance_filter",
        "final_rating",
        "raw_rating",
        "p_entailment",
        "p_neutral",
        "p_contradiction",
        "relevance_filter_value",
        "has_negation_match",
        "negation_correction_applied",
        "gold_aspects",
        "gold_aspect_ratings",
        "auto_exact_gold_hit",
        "auto_exact_gold_rating",
        "auto_abs_error_if_exact",
        "cluster_evidence",
        "premise_text",
        "full_text",
        "hypothesis_text",
        "manual_gold_aspect",
        "manual_decision",
        "manual_sentiment_decision",
        "manual_error_type",
        "manual_comment",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols and c != "true_labels_dict"]

    full = df[cols].sort_values(
        by=["passed_relevance_filter", "auto_abs_error_if_exact", "nm_id", "review_id", "aspect_source", "aspect_name"],
        ascending=[False, False, True, True, True, True],
        na_position="last",
    )
    full.to_csv(out_dir / "manual_audit_queue_full.csv", index=False, encoding="utf-8-sig")

    passed = full[full["passed_relevance_filter"] == True].copy()
    passed.to_csv(out_dir / "manual_audit_queue_passed.csv", index=False, encoding="utf-8-sig")

    # Balanced-ish sample: worst exact errors + random-like deterministic head per category/source.
    worst = passed[pd.notna(passed.get("auto_abs_error_if_exact"))].head(200)
    by_group = (
        passed.groupby(["category", "aspect_source"], dropna=False, group_keys=False)
        .head(30)
    )
    sample = pd.concat([worst, by_group], ignore_index=True).drop_duplicates(subset=["prediction_id"])
    sample.to_csv(out_dir / "manual_audit_queue_sample.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        "# Manual audit queue summary",
        "",
        f"- Source run: `{run_dir}`",
        f"- NLI rows total: {len(full)}",
        f"- Passed relevance filter: {len(passed)}",
        "",
        "## By aspect_source",
        "",
        counts_to_markdown(passed["aspect_source"], "aspect_source"),
        "",
        "## By category",
        "",
        counts_to_markdown(passed["category"], "category"),
        "",
        "## Exact auto-hit against gold aspect names",
        "",
        f"- Exact hits: {(passed['auto_exact_gold_hit'].astype(str) != '').sum()}",
        f"- Exact hit share among passed rows: {((passed['auto_exact_gold_hit'].astype(str) != '').mean() if len(passed) else 0):.4f}",
        "",
        "## How to use",
        "",
        "1. Start with `manual_audit_queue_sample.csv`.",
        "2. Fill `manual_gold_aspect`, `manual_decision`, `manual_sentiment_decision`, `manual_error_type`, `manual_comment`.",
        "3. Then continue with `manual_audit_queue_passed.csv` if time allows.",
        "4. Use `manual_audit_queue_full.csv` only if you also want to inspect rows rejected by relevance filter.",
        "",
        "Recommended `manual_decision` values: `TP`, `FP`, `UNCLEAR`, `DUPLICATE`, `OUT_OF_SCOPE`.",
        "Recommended sentiment values: `OK`, `WRONG_POLARITY`, `TOO_HIGH`, `TOO_LOW`, `NOT_EVALUATED`.",
    ]
    (out_dir / "manual_audit_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--merge-map", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    build_queue(
        run_dir=Path(args.run_dir),
        dataset_path=Path(args.dataset),
        merge_map_path=Path(args.merge_map) if args.merge_map else None,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
