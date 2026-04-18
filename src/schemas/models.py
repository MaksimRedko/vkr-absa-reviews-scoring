"""Pydantic-схемы для данных отзывов"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional

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
        """Текст отзыва для анализа."""
        return (self.full_text or "").strip()
        # parts = []
        # if self.pros and len(str(self.pros)) > 2:
        #     parts.append(str(self.pros).strip())
        # if self.cons and len(str(self.cons)) > 2:
        #     parts.append(str(self.cons).strip())
        # if self.full_text and len(str(self.full_text)) > 2:
        #     parts.append(str(self.full_text).strip())
        # return " ".join(parts)


@dataclass
class Candidate:
    span: str
    sentence: str
    token_indices: tuple[int, int]
    review_id: str = ""
    candidate_id: str = ""
    source_span: Optional[str] = None
    head_lemma: str = ""
    modifier_text: str = ""
    modifier_lemma: str = ""
    modifier_type: Optional[Literal["amod", "xcomp", "copular", "predicative", "event", "nominal"]] = None
    dep_label: str = ""


@dataclass
class ScoredCandidate:
    span: str
    score: float
    sentence: str
    embedding: np.ndarray
    review_id: str = ""
    candidate_id: str = ""
    source_span: Optional[str] = None


@dataclass
class AspectInfo:
    keywords: List[str]
    centroid_embedding: np.ndarray
    keyword_weights: List[float] = field(default_factory=list)
    nli_label: str = ""


@dataclass
class SentimentPair:
    review_id: str
    sentence: str
    aspect: str
    nli_label: str
    weight: float = 1.0


@dataclass
class SentimentResult:
    review_id: str
    aspect: str
    sentence: str
    score: float
    # NLI v4 (single hypothesis): entailment / contradiction из одного трёхклассового softmax
    p_ent_pos: float
    p_ent_neg: float
    confidence: float = 1.0


@dataclass
class AggregationInput:
    review_id: str
    aspects: Dict[str, float]
    fraud_weight: float
    date: Optional[datetime] = None


@dataclass
class PairingMetadata:
    anchor_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    candidate_assignments: Dict[str, str] = field(default_factory=dict)


@dataclass
class PairingContext:
    review_text_by_id: Dict[str, str]
    sentence_to_review: Dict[str, str]
    scored_candidates: List[ScoredCandidate]
    aspects: Dict[str, AspectInfo]
    metadata: PairingMetadata = field(default_factory=PairingMetadata)
    multi_label_threshold: float = 0.0
    multi_label_max_aspects: int = 0


@dataclass
class EvalData:
    aspects: Dict[str, AspectInfo]
    sentiment_results: List[SentimentResult]
    sentence_to_review: Dict[str, str]
    trust_weights: List[float]
    per_review: Dict[str, Dict[str, float]]
    aspect_keywords: Dict[str, List[str]]
    diagnostics: Dict[str, object] = field(default_factory=dict)


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
