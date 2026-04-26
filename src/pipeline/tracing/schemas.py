from __future__ import annotations

from pydantic import BaseModel


class CandidateArtifact(BaseModel):
    candidate_id: str
    review_id: str
    nm_id: int
    category_id: str
    text: str
    text_lemmatized: str
    start_offset: int
    end_offset: int
    source: str


class CandidateMatchArtifact(BaseModel):
    candidate_id: str
    matched_aspect_id: str | None
    match_method: str | None
    match_score: float
    matched_lemmas: list[str]
    cosine_similarity: float
    is_unmatched: bool


class NliPredictionArtifact(BaseModel):
    prediction_id: str
    review_id: str
    nm_id: int
    aspect_name: str
    aspect_source: str
    hypothesis_text: str
    premise_text: str
    p_entailment: float
    p_neutral: float
    p_contradiction: float
    raw_rating: float
    passed_relevance_filter: bool
    relevance_filter_value: float
    has_negation_match: bool
    negation_correction_applied: bool
    final_rating: float


class ProductAggregateArtifact(BaseModel):
    nm_id: int
    aspect_name: str
    aspect_source: str
    n_reviews_contributing: int
    contributing_review_ids: list[str]
    raw_mean_rating: float
    shrunken_rating: float
    variance: float
    shrinkage_strength: float
    gold_rating: float | None
    abs_error: float | None
