from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


BATCHES_DIR = Path("benchmark") / "batches"
LABELS_DIR = Path("benchmark") / "labels"
RAW_DIR = Path("benchmark") / "raw"
REVIEWS_BY_VENUE_DIR = RAW_DIR / "reviews_by_venue"


def _clip_score(v: Any) -> int:
    try:
        iv = int(v)
    except Exception:
        return 3
    if iv < 1:
        return 1
    if iv > 5:
        return 5
    return iv


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_meta_for_batch(batch_path: Path, meta_path: Path) -> List[Dict[str, Any]]:
    meta = _load_json(meta_path)
    stem = batch_path.stem  # e.g. venue_001_batch_01
    batches = meta.get("batches", {})
    if stem not in batches:
        raise ValueError(f"Batch '{stem}' not found in meta {meta_path}")
    entries = batches[stem]
    if not isinstance(entries, list):
        raise ValueError(f"Invalid meta format for batch '{stem}'")
    return entries


def parse_single_batch(batch_path: Path, meta_path: Path) -> Dict[str, Dict[str, int]]:
    labels_raw = _load_json(batch_path)
    meta_entries = _load_meta_for_batch(batch_path, meta_path)

    if not isinstance(labels_raw, list):
        raise ValueError(f"LLM labels file {batch_path} must contain a JSON array")

    if len(labels_raw) != len(meta_entries):
        raise ValueError(
            f"Batch size mismatch for {batch_path}: "
            f"{len(labels_raw)} labels vs {len(meta_entries)} meta entries"
        )

    parsed: Dict[str, Dict[str, int]] = {}
    for obj, meta in zip(labels_raw, meta_entries):
        review_id = meta["review_id"]
        if obj is None:
            parsed[review_id] = {}
            continue
        if not isinstance(obj, dict):
            # Плохой формат — считаем как пустой объект
            parsed[review_id] = {}
            continue
        clean: Dict[str, int] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            score = _clip_score(v)
            clean[k] = score
        parsed[review_id] = clean

    return parsed


def _collect_stats_for_venue(venue_id: str, labels: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    reviews_path = REVIEWS_BY_VENUE_DIR / f"{venue_id}.json"
    reviews = _load_json(reviews_path)
    total_reviews = len(reviews)

    labeled_reviews = 0
    empty_reviews = 0
    unique_aspects = set()
    aspect_counts: Counter[str] = Counter()

    for rid, asp_dict in labels.items():
        if not asp_dict:
            empty_reviews += 1
            continue
        labeled_reviews += 1
        for asp in asp_dict.keys():
            unique_aspects.add(asp)
            aspect_counts[asp] += 1

    return {
        "total_reviews": total_reviews,
        "labeled_reviews": labeled_reviews,
        "empty_reviews": empty_reviews,
        "unique_aspects": sorted(unique_aspects),
        "aspect_counts": {k: int(v) for k, v in aspect_counts.items()},
    }


def parse_for_venue(venue_id: str) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(LABELS_DIR.glob(f"{venue_id}_batch_*.json"))
    if not batch_files:
        print(f"[parse_llm_labels] No batch label files for venue {venue_id}")
        return

    meta_path = BATCHES_DIR / f"{venue_id}_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"Meta file not found for venue {venue_id}: {meta_path}")

    aggregated: Dict[str, Dict[str, int]] = {}
    for batch_path in batch_files:
        batch_parsed = parse_single_batch(batch_path, meta_path)
        aggregated.update(batch_parsed)

    stats = _collect_stats_for_venue(venue_id, aggregated)

    out_path = LABELS_DIR / f"{venue_id}_parsed.json"
    payload = {
        "venue_id": venue_id,
        "labels": aggregated,
        "stats": stats,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[parse_llm_labels] Parsed labels for {venue_id} → {out_path}")


def parse_all_venues() -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    selected_path = RAW_DIR / "selected_venues.json"
    if not selected_path.is_file():
        raise SystemExit(
            f"selected_venues.json not found at {selected_path}. "
            "Run benchmark/download_yandex_maps.py first."
        )
    with selected_path.open("r", encoding="utf-8") as f:
        venues = json.load(f)
    for v in venues:
        parse_for_venue(v["venue_id"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse LLM labels into structured per-venue format."
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Путь к одному JSON-файлу с ответом LLM для батча.",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Путь к meta JSON (обычно benchmark/batches/venue_XXX_meta.json).",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        help="ID заведения (venue_XXX) для парсинга всех его батчей.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Парсить все заведения из selected_venues.json.",
    )
    args = parser.parse_args()

    if args.batch:
        if not args.meta:
            raise SystemExit("--meta is required when --batch is specified")
        batch_path = Path(args.batch)
        meta_path = Path(args.meta)
        venue_id = batch_path.stem.split("_batch_")[0]
        LABELS_DIR.mkdir(parents=True, exist_ok=True)

        parsed = parse_single_batch(batch_path, meta_path)
        stats = _collect_stats_for_venue(venue_id, parsed)
        out_path = LABELS_DIR / f"{venue_id}_parsed.json"
        payload = {
            "venue_id": venue_id,
            "labels": parsed,
            "stats": stats,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[parse_llm_labels] Parsed single batch for {venue_id} → {out_path}")
        return

    if args.venue:
        parse_for_venue(args.venue)
        return

    if args.all:
        parse_all_venues()
        return

    parser.print_help()


if __name__ == "__main__":
    main()

