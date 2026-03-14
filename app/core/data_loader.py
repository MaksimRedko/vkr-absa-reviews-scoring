import sqlite3
import pandas as pd
from typing import List
from app.schemas.models import ReviewInput


class DataLoader:
    def __init__(self, db_path: str = "dataset.db"):
        self.db_path = db_path

    def get_top_products(self, limit: int = 5) -> pd.DataFrame:
        """
        Ищет товары с наибольшим количеством отзывов.
        Нужно, чтобы выбрать популярный товар для анализа.
        """
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
            print(f"❌ Ошибка БД: {e}")
            return pd.DataFrame()

    def load_reviews_for_product(self, nm_id: int, limit: int = 1000) -> List[ReviewInput]:
        """
        Загружает отзывы для конкретного товара.
        """
        # Берем только нужные колонки
        query = """
        SELECT id, nm_id, rating, created_date, full_text, pros, cons
        FROM reviews
        WHERE nm_id = ?
        LIMIT ?
        """

        valid_reviews = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(nm_id, limit))

            # Превращаем строки таблицы в объекты
            for _, row in df.iterrows():
                try:
                    # Валидация через Pydantic
                    review = ReviewInput(**row.to_dict())

                    # Если после склейки текст пустой - пропускаем
                    if not review.clean_text:
                        continue

                    valid_reviews.append(review)
                except Exception as val_err:
                    continue  # Битая строка, пропускаем

        except Exception as e:
            print(f"❌ Ошибка загрузки отзывов: {e}")

        return valid_reviews