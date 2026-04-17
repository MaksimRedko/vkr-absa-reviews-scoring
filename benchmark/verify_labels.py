from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


RAW_DIR = Path("benchmark") / "raw"
REVIEWS_BY_VENUE_DIR = RAW_DIR / "reviews_by_venue"
LABELS_DIR = Path("benchmark") / "labels"
VERIFIED_DIR = Path("benchmark") / "verified"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clip_score(v: Any) -> int:
    try:
        iv = int(v)
    except Exception:
        return 3
    return max(1, min(5, iv))


def _interactive_review_loop(
    venue_id: str,
    reviews: Dict[str, Dict[str, Any]],
    labels: Dict[str, Dict[str, int]],
    sample_rate: float,
    seed: int,
) -> None:
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)

    review_ids = list(labels.keys())
    n_sample = max(1, int(len(review_ids) * sample_rate))
    rng = random.Random(seed)
    rng.shuffle(review_ids)
    sample_ids = review_ids[:n_sample]

    edits: List[Dict[str, Any]] = []
    agreed = 0
    skipped = 0

    for rid in sample_ids:
        rev = reviews.get(rid, {})
        rating = rev.get("rating", "")
        text = str(rev.get("text", "")).strip()
        current = labels.get(rid, {})

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(
            f"Venue: {venue_id} | Review: {rid} | Rating: {rating}"
        )
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Текст: {text!r}")
        print()
        print(f"LLM разметка: {json.dumps(current, ensure_ascii=False)}")
        print()
        print("[Enter] = согласен  |  [e] = редактировать  |  [s] = skip  |  [q] = выход")
        choice = input("> ").strip().lower()

        if choice == "q":
            break
        if choice == "s":
            skipped += 1
            continue
        if choice == "e":
            print("Введи исправленную разметку (JSON):")
            new_raw = input("> ").strip()
            try:
                obj = json.loads(new_raw) if new_raw else {}
                if not isinstance(obj, dict):
                    print("Ожидался JSON-объект, правка пропущена.")
                    continue
                clean = {str(k): _clip_score(v) for k, v in obj.items()}
                edits.append(
                    {
                        "review_id": rid,
                        "original": current,
                        "corrected": clean,
                    }
                )
                labels[rid] = clean
            except json.JSONDecodeError:
                print("Не удалось распарсить JSON, правка пропущена.")
                continue
        else:
            agreed += 1

    total_checked = agreed + len(edits) + skipped
    agreement_rate = agreed / total_checked if total_checked else 0.0

    out_payload = {
        "venue_id": venue_id,
        "total_checked": total_checked,
        "agreed": agreed,
        "edited": len(edits),
        "skipped": skipped,
        "agreement_rate": round(agreement_rate, 4),
        "edits": edits,
    }

    out_path = VERIFIED_DIR / f"{venue_id}_verification.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"[verify_labels] Verification results saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive tool for manual verification of parsed LLM labels."
    )
    parser.add_argument(
        "--venue",
        type=str,
        required=True,
        help="ID заведения (venue_XXX).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.15,
        help="Доля отзывов для проверки (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для случайного выбора отзывов (default: 42).",
    )
    args = parser.parse_args()

    parsed_path = LABELS_DIR / f"{args.venue}_parsed.json"
    if not parsed_path.is_file():
        raise SystemExit(
            f"Parsed labels file not found for venue {args.venue}: {parsed_path}"
        )
    parsed_payload = _load_json(parsed_path)
    labels = parsed_payload.get("labels", {})

    reviews_path = REVIEWS_BY_VENUE_DIR / f"{args.venue}.json"
    if not reviews_path.is_file():
        raise SystemExit(
            f"Reviews file not found for venue {args.venue}: {reviews_path}"
        )
    reviews_list = _load_json(reviews_path)
    reviews = {r["id"]: r for r in reviews_list}

    _interactive_review_loop(
        venue_id=args.venue,
        reviews=reviews,
        labels=labels,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

