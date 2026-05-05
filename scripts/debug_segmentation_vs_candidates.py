from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import configs as _cfg_module
from src.stages.extraction import CandidateExtractor

_cfg_module.config.discovery.dependency_filter_enabled = False  # type: ignore[attr-defined]


def _parse_true_labels(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, dict):
        return []
    return sorted(str(k).strip() for k in parsed.keys() if str(k).strip())


def _split_segments_rule_based(text: str) -> list[str]:
    parts = re.split(
        r"[.!?;\n\r]+|(?:\s+-\s+)|(?:\s+—\s+)|(?:,\s+но\s+)|(?:,\s+а\s+)|(?:,\s+однако\s+)",
        text,
    )
    return [p.strip() for p in parts if p and p.strip()]


def _extract_candidates_with_sentences(text: str, extractor: CandidateExtractor) -> list[dict]:
    cleaned = extractor._clean(text)
    sentences = extractor._split_sentences(cleaned)
    out = []
    for idx, sent in enumerate(sentences):
        cands = extractor._candidates_from_sentence(sent)
        out.append(
            {
                "unit_index": idx,
                "unit_text": sent,
                "candidates": sorted({c.span for c in cands if str(c.span).strip()}),
            }
        )
    return out


def _extract_candidates_with_segments(text: str, extractor: CandidateExtractor) -> list[dict]:
    cleaned = extractor._clean(text)
    segments = _split_segments_rule_based(cleaned)
    out = []
    for idx, seg in enumerate(segments):
        cands = extractor._candidates_from_sentence(seg)
        out.append(
            {
                "unit_index": idx,
                "unit_text": seg,
                "candidates": sorted({c.span for c in cands if str(c.span).strip()}),
            }
        )
    return out


def _load_rows(dataset_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv, dtype={"id": str})
    required = {"id", "nm_id", "category", "full_text", "true_labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--review-id", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--category", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = _load_rows(ROOT / args.dataset_csv)

    if args.review_id:
        df = df[df["id"] == args.review_id]
    elif args.category:
        df = df[df["category"] == args.category].head(args.limit)
    else:
        df = df.head(args.limit)

    extractor = CandidateExtractor(ngram_range=(1, 2), min_word_length=3)

    traces = []
    for _, row in df.iterrows():
        text = str(row["full_text"]).strip()
        if not text:
            continue

        sentence_units = _extract_candidates_with_sentences(text, extractor)
        segment_units = _extract_candidates_with_segments(text, extractor)

        sentence_union = sorted(
            {cand for unit in sentence_units for cand in unit["candidates"]}
        )
        segment_union = sorted(
            {cand for unit in segment_units for cand in unit["candidates"]}
        )

        traces.append(
            {
                "review_id": str(row["id"]),
                "nm_id": int(row["nm_id"]),
                "category": str(row["category"]),
                "gold_labels": _parse_true_labels(row["true_labels"]),
                "text": text,
                "sentence_split_units": sentence_units,
                "segment_split_units": segment_units,
                "sentence_union_candidates": sentence_union,
                "segment_union_candidates": segment_union,
                "only_in_sentence_union": sorted(set(sentence_union) - set(segment_union)),
                "only_in_segment_union": sorted(set(segment_union) - set(sentence_union)),
                "same_union": sentence_union == segment_union,
            }
        )

    payload = json.dumps(traces, ensure_ascii=False, indent=2)

    if args.out:
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"saved to {out_path}")
    else:
        print(payload)


if __name__ == "__main__":
    main()