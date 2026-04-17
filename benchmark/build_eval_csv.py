from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from data_sources import load_wb_reviews, load_ym_reviews  # type: ignore


RAW_DIR = Path("benchmark") / "raw"
REVIEWS_BY_VENUE_DIR = RAW_DIR / "reviews_by_venue"
LABELS_DIR = Path("benchmark") / "labels"
VERIFIED_DIR = Path("benchmark") / "verified"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dict_to_python_literal(d: Dict[str, int]) -> str:
    if not d:
        return "{}"
    items = [f"'{k}': {int(v)}" for k, v in d.items()]
    return "{" + ", ".join(items) + "}"


def _stable_nm_id_for_ym(prefix: str) -> int:
    # Детерминированный int для ym_<venue_key>, чтобы можно было смешивать с WB.
    # Диапазон 1_500_000_000..1_999_999_999
    h = int(hashlib.md5(prefix.encode("utf-8")).hexdigest()[:8], 16)
    return 1_500_000_000 + (h % 500_000_000)


def build_eval_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    parsed_files = sorted(LABELS_DIR.glob("*_parsed.json"))
    if not parsed_files:
        raise SystemExit("No *_parsed.json files found in benchmark/labels.")

    for parsed_path in parsed_files:
        payload = _load_json(parsed_path)
        labels: Dict[str, Dict[str, int]] = payload.get("labels", {}) or {}
        if not labels:
            continue

        prefix = parsed_path.stem.replace("_parsed", "")

        if prefix.startswith("wb_"):
            nm_id = int(prefix.split("_", 1)[1])
            wb_df = load_wb_reviews(nm_id, limit=None)
            if wb_df.empty:
                continue

            wb_df = wb_df.copy()
            wb_df["id"] = wb_df["id"].astype(str)
            wb_df["text"] = wb_df["full_text"].fillna("").astype(str)
            empty_mask = wb_df["text"].str.strip() == ""
            wb_df.loc[empty_mask, "text"] = (
                wb_df.loc[empty_mask, "pros"].fillna("").astype(str)
                + " "
                + wb_df.loc[empty_mask, "cons"].fillna("").astype(str)
            ).str.strip()

            by_id = {str(r["id"]): r for _, r in wb_df.iterrows()}
            for rid, lab in labels.items():
                if rid not in by_id:
                    continue
                rec = by_id[rid]
                rows.append(
                    {
                        "id": rid,
                        "nm_id": int(nm_id),
                        "rating": int(rec.get("rating", 0)),
                        "created_date": str(rec.get("created_date", "2025-01-01")),
                        "full_text": str(rec.get("text", "")).strip(),
                        "pros": str(rec.get("pros", "") or "").strip(),
                        "cons": str(rec.get("cons", "") or "").strip(),
                        "true_labels": _dict_to_python_literal(lab or {}),
                    }
                )

        elif prefix.startswith("ym_"):
            venue_key = prefix.split("_", 1)[1]
            nm_id = _stable_nm_id_for_ym(prefix)
            ym_df = load_ym_reviews(venue_key, limit=None)
            if ym_df.empty:
                continue
            ym_df = ym_df.copy()
            ym_df["id"] = ym_df["id"].astype(str)
            by_id = {str(r["id"]): r for _, r in ym_df.iterrows()}

            for rid, lab in labels.items():
                if rid not in by_id:
                    continue
                rec = by_id[rid]
                rows.append(
                    {
                        "id": rid,
                        "nm_id": int(nm_id),
                        "rating": int(rec.get("rating", 0)),
                        "created_date": "2025-01-01",
                        "full_text": str(rec.get("text", "")).strip(),
                        "pros": "",
                        "cons": "",
                        "true_labels": _dict_to_python_literal(lab or {}),
                    }
                )

        else:
            # Неподдерживаемый префикс — пропускаем
            continue

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build eval CSV compatible with eval_pipeline.load_markup()."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Путь к выходному CSV (например benchmark/eval_datasets/yandex_maps_benchmark.csv).",
    )
    args = parser.parse_args()

    rows = build_eval_rows()
    if not rows:
        raise SystemExit("No rows to write. Make sure parsed labels are available.")

    df = pd.DataFrame(rows)

    # Лёгкая проверка, что true_labels парсится через ast.literal_eval
    try:
        _ = df["true_labels"].apply(ast.literal_eval)
    except Exception as e:
        raise SystemExit(f"true_labels column is not compatible with ast.literal_eval: {e}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[build_eval_csv] Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

