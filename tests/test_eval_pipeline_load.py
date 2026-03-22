"""
Тесты загрузки разметки и отзывов для eval_pipeline (без тяжёлого пайплайна).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from eval_pipeline import load_markup, load_pipeline_reviews_from_csv

ROOT = Path(__file__).resolve().parents[1]
MERGED_CSV = ROOT / "parser" / "reviews_batches" / "merged_checked_reviews.csv"

# Товары из merged_checked_reviews (300 отзывов: по 100 на nm_id)
@pytest.mark.parametrize("nm_id", [15430704, 619500952, 54581151])
def test_load_pipeline_reviews_count_per_product(nm_id: int) -> None:
    assert MERGED_CSV.is_file(), f"нет файла {MERGED_CSV}"
    rows = load_pipeline_reviews_from_csv(str(MERGED_CSV), [nm_id])
    assert len(rows) == 100
    assert all(r["nm_id"] == nm_id for r in rows)
    assert all(isinstance(r["id"], str) and r["id"] for r in rows)
    assert all(1 <= r["rating"] <= 5 for r in rows)


@pytest.mark.parametrize("nm_id", [15430704, 619500952, 54581151])
def test_load_markup_true_labels_parsed(nm_id: int) -> None:
    df = load_markup(str(MERGED_CSV))
    g = df[df["nm_id"] == nm_id]
    assert len(g) == 100
    parsed = g["true_labels_parsed"].dropna()
    assert len(parsed) == 100


def test_load_pipeline_all_three_products() -> None:
    nms = [15430704, 619500952, 54581151]
    rows = load_pipeline_reviews_from_csv(str(MERGED_CSV), nms)
    assert len(rows) == 300
