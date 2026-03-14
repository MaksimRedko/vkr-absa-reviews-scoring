from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class ReviewInput(BaseModel):
    """
    Схема одного отзыва.
    Проверяет типы данных и склеивает текст.
    """
    id: str
    nm_id: int
    rating: int = Field(..., ge=1, le=5)  # Оценка строго от 1 до 5
    created_date: datetime

    # Эти поля могут быть пустыми в базе
    full_text: Optional[str] = None
    pros: Optional[str] = None
    cons: Optional[str] = None

    @field_validator('created_date', mode='before')
    def parse_date(cls, v):
        # Если дата пришла строкой, Pydantic сам разберется, но на всякий случай
        return v

    @property
    def clean_text(self) -> str:
        """
        Магия склейки. Превращает 3 колонки в один текст для нейросети.
        """
        parts = []
        # Если есть плюсы, добавляем метку "Достоинства:"
        if self.pros and len(str(self.pros)) > 2:
            parts.append(f"Достоинства: {str(self.pros).strip()}")

        # Если есть минусы
        if self.cons and len(str(self.cons)) > 2:
            parts.append(f"Недостатки: {str(self.cons).strip()}")

        # Если есть основной текст
        if self.full_text and len(str(self.full_text)) > 2:
            parts.append(f"Комментарий: {str(self.full_text).strip()}")

        # Склеиваем через пробел
        return " ".join(parts)