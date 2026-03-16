"""Pydantic-схемы для данных отзывов"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ReviewInput(BaseModel):
    """
    Схема одного отзыва из БД.
    Склеивает pros/cons/full_text в единый текст для пайплайна.
    """
    id: str
    nm_id: int
    rating: int = Field(..., ge=1, le=5)
    created_date: datetime

    full_text: Optional[str] = None
    pros: Optional[str] = None
    cons: Optional[str] = None

    @field_validator("created_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        return v

    @property
    def clean_text(self) -> str:
        """Склейка трёх текстовых полей в один текст для анализа."""
        parts = []
        if self.pros and len(str(self.pros)) > 2:
            parts.append(f"Достоинства: {str(self.pros).strip()}")
        if self.cons and len(str(self.cons)) > 2:
            parts.append(f"Недостатки: {str(self.cons).strip()}")
        if self.full_text and len(str(self.full_text)) > 2:
            parts.append(f"Комментарий: {str(self.full_text).strip()}")
        return " ".join(parts)
