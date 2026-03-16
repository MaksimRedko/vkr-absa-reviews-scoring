"""Загрузчик отзывов из SQLite БД"""

from __future__ import annotations

import sqlite3
from typing import List

import pandas as pd

from src.schemas.models import ReviewInput


class DataLoader:
    """Загрузка отзывов из dataset.db"""

    def __init__(self, db_path: str = "data/dataset.db"):
        self.db_path = db_path

    def get_top_products(self, limit: int = 5) -> pd.DataFrame:
        """Товары с наибольшим количеством отзывов."""
        query = """
        SELECT nm_id, COUNT(*) as review_count
        FROM reviews
        GROUP BY nm_id
        ORDER BY review_count DESC
        LIMIT ?
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn, params=(limit,))
        except Exception as e:
            print(f"Ошибка БД: {e}")
            return pd.DataFrame()

    def load_reviews(self, nm_id: int, limit: int = 500) -> List[ReviewInput]:
        """Загружает отзывы для конкретного товара."""
        query = """
        SELECT id, nm_id, rating, created_date, full_text, pros, cons
        FROM reviews
        WHERE nm_id = ?
        LIMIT ?
        """
        valid = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(nm_id, limit))

            for _, row in df.iterrows():
                try:
                    review = ReviewInput(**row.to_dict())
                    if not review.clean_text:
                        continue
                    valid.append(review)
                except Exception:
                    continue

        except Exception as e:
            print(f"Ошибка загрузки отзывов: {e}")

        return valid
