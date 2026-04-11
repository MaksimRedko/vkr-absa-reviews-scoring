"""Pydantic-схемы для данных отзывов"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from pydantic import BaseModel, Field, field_validator


class ReviewInput(BaseModel):
    """
    Схема одного отзыва из БД.
    Склеивает pros/cons/full_text в единый текст для пайплайна.
    """
    id: str
    nm_id: int
    rating: int = Field(..., ge=1, le=5)
    created_date: Optional[datetime] = None

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


@dataclass
class Candidate:
    span: str
    sentence: str
    token_indices: tuple[int, int]


@dataclass
class ScoredCandidate:
    span: str
    score: float
    sentence: str
    embedding: np.ndarray


@dataclass
class AspectInfo:
    keywords: List[str]
    centroid_embedding: np.ndarray
    keyword_weights: List[float] = field(default_factory=list)
    nli_label: str = ""


@dataclass
class SentimentResult:
    review_id: str
    aspect: str
    sentence: str
    score: float
    p_ent_pos: float
    p_ent_neg: float
    confidence: float = 1.0


@dataclass
class EvalData:
    aspects: Dict[str, AspectInfo]
    sentiment_results: List[SentimentResult]
    sentence_to_review: Dict[str, str]
    trust_weights: List[float]
    per_review: Dict[str, Dict[str, float]]
    aspect_keywords: Dict[str, List[str]]


@dataclass
class AspectScore:
    name: str
    score: float
    raw_mean: float
    controversy: float
    mentions: int
    effective_mentions: float


@dataclass
class AggregationResult:
    aspects: Dict[str, AspectScore] = field(default_factory=dict)
    covariance_matrix: Optional[np.ndarray] = None
    aspect_order: List[str] = field(default_factory=list)
