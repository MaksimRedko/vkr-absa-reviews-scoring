"""
Загрузчики размеченных отзывов из трёх источников:
  - parser/razmetka/checked_reviews.csv       (WB, JSON-метки)
  - parser/reviews_batches/merged_checked_reviews.csv  (WB, JSON-метки)
  - benchmark/eval_datasets/combined_benchmark.csv     (Yandex, Python-dict-метки)
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class AnnotatedReview:
    review_id: str
    product_id: int
    text: str
    true_aspects: frozenset[str]  # ключи из true_labels (только названия, без оценок)
    source: str  # "wb" | "yandex"


# ---------------------------------------------------------------------------
# Внутренние хелперы
# ---------------------------------------------------------------------------

def _parse_labels(raw: object) -> frozenset[str]:
    """Парсит true_labels в виде JSON-строки или Python-dict-строки."""
    if raw is None:
        return frozenset()
    if isinstance(raw, float):
        return frozenset()  # NaN
    if isinstance(raw, dict):
        return frozenset(str(k) for k in raw.keys())
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "{}"):
        return frozenset()
    # Сначала JSON (двойные кавычки), затем ast (одинарные)
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            return frozenset(str(k) for k in d.keys())
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return frozenset(str(k) for k in d.keys())
    except (ValueError, SyntaxError):
        pass
    return frozenset()


def _row_text(row: pd.Series) -> str:
    return str(row.get("full_text") or "").strip()


# ---------------------------------------------------------------------------
# Публичные загрузчики
# ---------------------------------------------------------------------------

def load_wb_checked(
    path: str | Path,
    *,
    only_checked: bool = True,
    product_ids: list[int] | None = None,
) -> list[AnnotatedReview]:
    """checked_reviews.csv — JSON-формат меток, есть колонка is_checked."""
    df = pd.read_csv(path, dtype={"nm_id": int})
    if only_checked and "is_checked" in df.columns:
        df = df[df["is_checked"].astype(str).str.strip().str.lower() == "true"]
    if product_ids is not None:
        df = df[df["nm_id"].isin(product_ids)]
    records: list[AnnotatedReview] = []
    for _, row in df.iterrows():
        aspects = _parse_labels(row.get("true_labels"))
        text = _row_text(row)
        if not aspects or not text:
            continue
        records.append(AnnotatedReview(
            review_id=str(row["id"]),
            product_id=int(row["nm_id"]),
            text=text,
            true_aspects=aspects,
            source="wb",
        ))
    return records


def load_wb_merged(
    path: str | Path,
    *,
    only_checked: bool = True,
    product_ids: list[int] | None = None,
) -> list[AnnotatedReview]:
    """merged_checked_reviews.csv — идентичный формат checked_reviews."""
    return load_wb_checked(path, only_checked=only_checked, product_ids=product_ids)


def load_yandex_benchmark(
    path: str | Path,
    product_ids: list[int],
) -> list[AnnotatedReview]:
    """combined_benchmark.csv — Python-dict-формат меток (одиночные кавычки)."""
    df = pd.read_csv(path, dtype={"nm_id": int})
    df = df[df["nm_id"].isin(product_ids)]
    records: list[AnnotatedReview] = []
    for _, row in df.iterrows():
        aspects = _parse_labels(row.get("true_labels"))
        text = _row_text(row)
        if not aspects or not text:
            continue
        records.append(AnnotatedReview(
            review_id=str(row["id"]),
            product_id=int(row["nm_id"]),
            text=text,
            true_aspects=aspects,
            source="yandex",
        ))
    return records


def load_all_annotated(
    wb_checked_path: str | Path,
    wb_merged_path: str | Path,
    yandex_benchmark_path: str | Path,
    wb_product_ids: list[int],
    yandex_product_ids: list[int],
) -> list[AnnotatedReview]:
    """Собирает отзывы из всех источников, дедуплицирует по review_id."""
    wb1 = load_wb_checked(wb_checked_path, product_ids=wb_product_ids)
    wb2 = load_wb_merged(wb_merged_path, product_ids=wb_product_ids)
    yndx = load_yandex_benchmark(yandex_benchmark_path, yandex_product_ids)

    seen: set[str] = set()
    out: list[AnnotatedReview] = []
    for r in wb1 + wb2 + yndx:
        if r.review_id not in seen:
            seen.add(r.review_id)
            out.append(r)
    return out
