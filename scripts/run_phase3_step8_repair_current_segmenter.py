from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._phase3_step8_old_segmenter_snapshot import RuleBasedClauseSegmenter as OldSegmenter
from src.stages.segmentation import RuleBasedClauseSegmenter as NewSegmenter

CATEGORIES: tuple[str, ...] = (
    "physical_goods",
    "consumables",
    "hospitality",
    "services",
)

PATTERNS: tuple[tuple[str, str], ...] = (
    ("contrast_no", r"\b\u043d\u043e\b"),
    ("contrast_odnako", r"\b\u043e\u0434\u043d\u0430\u043a\u043e\b"),
    ("contrast_zato", r"\b\u0437\u0430\u0442\u043e\b"),
    ("contrast_hotya", r"\b\u0445\u043e\u0442\u044f\b"),
    ("contrast_pri_etom", r"\u043f\u0440\u0438 \u044d\u0442\u043e\u043c"),
    ("contrast_v_to_zhe_vremya", r"\u0432 \u0442\u043e \u0436\u0435 \u0432\u0440\u0435\u043c\u044f"),
    ("contrast_side_a", r"\u0441 \u043e\u0434\u043d\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u044b"),
    ("contrast_side_b", r"\u0441 \u0434\u0440\u0443\u0433\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u044b"),
    ("abbr_tdotd", r"\u0442\.\u0434\."),
    ("abbr_tdotp", r"\u0442\.\u043f\."),
    ("abbr_g", r"\b\u0433\."),
    ("abbr_ul", r"\b\u0443\u043b\."),
    ("abbr_str", r"\b\u0441\u0442\u0440\."),
    ("ellipsis", r"(?:\.{3,}|\u2026)"),
    ("decimal", r"\d+[\.,]\d+"),
    ("label_adv", r"\u0434\u043e\u0441\u0442\u043e\u0438\u043d\u0441\u0442\u0432\u0430\s*:"),
    ("label_dis", r"\u043d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043a\u0438\s*:"),
)

CONTRAST_LABELS: tuple[str, ...] = (
    "contrast_no",
    "contrast_odnako",
    "contrast_zato",
    "contrast_hotya",
    "contrast_pri_etom",
    "contrast_v_to_zhe_vremya",
    "contrast_side_a",
    "contrast_side_b",
)

LABEL_LABELS: tuple[str, ...] = ("label_adv", "label_dis")

TRIVIAL_SEGMENT_RE = re.compile(
    r"^[\s,;:.!?\u2026-]*(?:"
    r"\u043d\u043e|"
    r"\u0438|"
    r"\u0430|"
    r"\u0438\u043b\u0438|"
    r"\u043e\u0434\u043d\u0430\u043a\u043e|"
    r"\u0437\u0430\u0442\u043e|"
    r"\u0445\u043e\u0442\u044f|"
    r"\u043f\u0440\u0438 \u044d\u0442\u043e\u043c|"
    r"\u0432 \u0442\u043e \u0436\u0435 \u0432\u0440\u0435\u043c\u044f|"
    r"\u0441 \u043e\u0434\u043d\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u044b|"
    r"\u0441 \u0434\u0440\u0443\u0433\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u044b"
    r")[\s,;:.!?\u2026-]*$",
    re.I,
)
TOKEN_RE = re.compile(r"\w+", re.U)


@dataclass(frozen=True)
class ReviewCase:
    review_id: str
    category: str
    original_text: str


def _segment_text(seg: Any, text: str) -> list[str]:
    return [s.text for s in seg.split(text)]


def _segment_json(parts: list[str]) -> str:
    return json.dumps(parts, ensure_ascii=False)


def _feature_flags(text: str) -> dict[str, bool]:
    lower = str(text).lower()
    flags: dict[str, bool] = {}
    for label, pattern in PATTERNS:
        flags[label] = bool(re.search(pattern, lower, flags=re.I))
    return flags


def _score_case(text: str) -> float:
    flags = _feature_flags(text)
    contrast_hits = sum(1 for key in CONTRAST_LABELS if flags[key])
    aux_hits = sum(1 for key, value in flags.items() if value and key not in CONTRAST_LABELS)
    comma_count = str(text).count(",")
    length_score = min(len(str(text)) / 220.0, 3.0)
    return contrast_hits * 4.0 + aux_hits * 2.0 + min(comma_count, 6) * 0.3 + length_score


def _bad_join_count(parts: list[str]) -> int:
    total = 0
    for part in parts:
        lower = part.lower()
        if "\u0434\u043e\u0441\u0442\u043e\u0438\u043d\u0441\u0442\u0432\u0430:" in lower and "\u043d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043a\u0438:" in lower:
            total += 2
        for marker in (
            "\u043d\u043e",
            "\u043e\u0434\u043d\u0430\u043a\u043e",
            "\u0437\u0430\u0442\u043e",
            "\u0445\u043e\u0442\u044f",
            "\u043f\u0440\u0438 \u044d\u0442\u043e\u043c",
            "\u0432 \u0442\u043e \u0436\u0435 \u0432\u0440\u0435\u043c\u044f",
            "\u0441 \u043e\u0434\u043d\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u044b",
            "\u0441 \u0434\u0440\u0443\u0433\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u044b",
        ):
            idx = lower.find(marker)
            if idx > 0 and idx < len(lower) - len(marker) - 2:
                total += 1
    return total


def _old_fail_score(text: str, old_seg: Any) -> float:
    parts = _segment_text(old_seg, text)
    flags = _feature_flags(text)
    score = float(_bad_join_count(parts)) * 3.0
    if flags["label_adv"] or flags["label_dis"]:
        score += 2.0
    if flags["ellipsis"]:
        score += 1.0
    if any(flags[key] for key in CONTRAST_LABELS):
        score += 1.5
    score += min(len(str(text)) / 400.0, 3.0)
    return score


def _pick_diagnostic_slice(df: pd.DataFrame) -> list[ReviewCase]:
    old_seg = OldSegmenter()
    picked: list[ReviewCase] = []
    for category in CATEGORIES:
        sub = df[df["category"] == category].copy()
        sub = sub[sub["full_text"].fillna("").str.strip().ne("")].copy()
        sub["score"] = sub["full_text"].map(lambda value: _old_fail_score(str(value), old_seg))
        sub["text_len"] = sub["full_text"].map(lambda value: len(str(value)))
        sub = sub.sort_values(["score", "text_len", "id"], ascending=[False, False, True])
        top = sub.head(10)
        if len(top) < 10:
            raise ValueError(f"Not enough reviews for category={category}")
        for _, row in top.iterrows():
            picked.append(
                ReviewCase(
                    review_id=str(row["id"]),
                    category=category,
                    original_text=str(row["full_text"]),
                )
            )
    return picked


def _word_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _has_trivial_segment(parts: list[str]) -> bool:
    return any(TRIVIAL_SEGMENT_RE.match(part.strip().lower() or "") for part in parts)


def _has_too_short_noise(parts: list[str]) -> bool:
    return any(_word_count(part) <= 1 for part in parts)


def _is_mixed_case(text: str, parts: list[str]) -> bool:
    flags = _feature_flags(text)
    if any(flags[key] for key in CONTRAST_LABELS) and len(parts) <= 1:
        return True
    if any(flags[key] for key in LABEL_LABELS) and len(parts) <= 1:
        return True
    return False


def _classify_case(text: str, old_parts: list[str], new_parts: list[str]) -> str:
    old_bad = _bad_join_count(old_parts)
    new_bad = _bad_join_count(new_parts)
    old_trivial = _has_trivial_segment(old_parts) or _has_too_short_noise(old_parts)
    new_trivial = _has_trivial_segment(new_parts) or _has_too_short_noise(new_parts)

    if new_trivial and not old_trivial:
        return "still_oversegmented"
    if new_bad < old_bad:
        return "fixed_cases"
    if old_parts != new_parts and not new_trivial and new_bad <= old_bad:
        return "fixed_cases"
    if new_bad > 0:
        return "still_undersegmented"
    return "ambiguous_cases"


def _format_examples(rows: list[dict[str, Any]], limit: int) -> str:
    if not rows:
        return "none\n"
    lines: list[str] = []
    for row in rows[:limit]:
        lines.append(f"- `{row['review_id']}` [{row['category']}]")
        lines.append(f"  - text: {row['original_text'].strip()}")
        lines.append(f"  - old: {' | '.join(row['old_segments'])}")
        lines.append(f"  - new: {' | '.join(row['new_segments'])}")
    return "\n".join(lines) + "\n"


def run(dataset_csv: Path, out_root: Path) -> Path:
    df = pd.read_csv(dataset_csv, dtype={"id": str})
    required = {"id", "category", "full_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    old_seg = OldSegmenter()
    new_seg = NewSegmenter()
    cases = _pick_diagnostic_slice(df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {
        "fixed_cases": [],
        "still_undersegmented": [],
        "still_oversegmented": [],
        "ambiguous_cases": [],
    }

    for case in cases:
        old_parts = _segment_text(old_seg, case.original_text)
        new_parts = _segment_text(new_seg, case.original_text)
        bucket = _classify_case(case.original_text, old_parts, new_parts)
        row = {
            "review_id": case.review_id,
            "category": case.category,
            "original_text": case.original_text,
            "old_segments": old_parts,
            "new_segments": new_parts,
            "old_n_segments": len(old_parts),
            "new_n_segments": len(new_parts),
            "bucket": bucket,
            "improved": bucket == "fixed_cases",
        }
        rows.append(row)
        buckets[bucket].append(row)

    csv_rows = [
        {
            "review_id": row["review_id"],
            "category": row["category"],
            "original_text": row["original_text"],
            "old_segments": _segment_json(row["old_segments"]),
            "new_segments": _segment_json(row["new_segments"]),
            "old_n_segments": row["old_n_segments"],
            "new_n_segments": row["new_n_segments"],
        }
        for row in rows
    ]
    pd.DataFrame(csv_rows).to_csv(out_dir / "segmenter_before_after.csv", index=False, encoding="utf-8")

    failure_lines: list[str] = []
    for name in ("still_undersegmented", "still_oversegmented", "fixed_cases", "ambiguous_cases"):
        failure_lines.append(f"## {name}\n")
        failure_lines.append(_format_examples(buckets[name], 10))
    (out_dir / "segmenter_failure_cases.md").write_text("\n".join(failure_lines), encoding="utf-8")

    improved = sum(1 for row in rows if row["improved"])
    mixed_old = sum(_bad_join_count(row["old_segments"]) for row in rows)
    mixed_new = sum(_bad_join_count(row["new_segments"]) for row in rows)
    improved_share = improved / len(rows) if rows else 0.0
    decision = "PASS" if improved_share >= 0.60 and mixed_new < mixed_old else "FAIL"

    summary_lines = [
        "# phase3_step8_repair_current_segmenter",
        "",
        "Диагностическая выборка: 40 реальных отзывов, по 10 на категорию, отобраны как failure-prone кейсы старого segmenter-а.",
        "",
        "## A. Что поменяли",
        "- добавили contrast markers: `с одной стороны`, `с другой стороны`",
        "- добавили label split для `Достоинства:` / `Недостатки:` / `Преимущества:` / `Комментарий:`",
        "- перестали превращать многоточие в жёсткую точку-разделитель",
        "- расширили защиту сокращений: `г.`, `ул.`, `стр.`",
        "- ослабили contrast split guard для коротких оценочных фрагментов (`1/1` content)",
        "- добавили merge для тривиальных marker-only хвостов",
        "",
        "## B. Что стало лучше",
        _format_examples(buckets["fixed_cases"], 10),
        "## C. Что всё ещё плохо",
        _format_examples(buckets["still_undersegmented"] + buckets["still_oversegmented"], 10),
        "## D. Decision",
        f"- improved_reviews = {improved}/{len(rows)} ({improved_share:.1%})",
        f"- mixed_old = {mixed_old}",
        f"- mixed_new = {mixed_new}",
        f"- decision = {decision}",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="phase3_step8_repair_current_segmenter")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase3_step8_repair_current_segmenter")
    args = parser.parse_args()

    out_dir = run(ROOT / args.dataset_csv, ROOT / args.out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
