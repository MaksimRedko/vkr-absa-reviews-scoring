"""
stages.py — Абстрактные базовые классы (ABCs) для каждой стадии ABSA-пайплайна.

Контракты фиксируют входные и выходные типы каждого узла.
Любую стадию можно заменить альтернативной реализацией:
достаточно унаследовать соответствующий ABC и реализовать абстрактный метод —
пайплайн подхватит без изменений в остальном коде.

  ┌──────────────────┬────────────────────────────────┬──────────────────────────┐
  │ ABC              │ Вход                           │ Выход                    │
  ├──────────────────┼────────────────────────────────┼──────────────────────────┤
  │ FraudStage       │ List[str]  (тексты отзывов)    │ List[float]  (веса)      │
  │ ExtractionStage  │ str  (текст одного отзыва)     │ List[Candidate]          │
  │ ScoringStage     │ List[Candidate]                │ List[ScoredCandidate]    │
  │ ClusteringStage  │ List[ScoredCandidate]          │ Dict[str, AspectInfo]    │
  │ SentimentStage   │ List[SentimentPair]            │ List[SentimentResult]    │
  │ AggregationStage │ List[AggregationInput]         │ AggregationResult        │
  └──────────────────┴────────────────────────────────┴──────────────────────────┘

SentimentPair    = Tuple[review_id, sentence, aspect_name, nli_label, weight]
AggregationInput = Dict{"review_id", "aspects": Dict[str,float], "fraud_weight", "date"}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from src.discovery.candidates import Candidate
from src.discovery.clusterer import AspectInfo
from src.discovery.scorer import ScoredCandidate
from src.math.engine import AggregationResult
from src.sentiment.engine import SentimentResult

SentimentPair    = Tuple[str, str, str, str, float]  # (review_id, sentence, aspect, nli_label, weight)
AggregationInput = Dict                              # {"review_id", "aspects", "fraud_weight", "date"}


class FraudStage(ABC):
    """
    Стадия 1: вычисление весов доверия.
    texts[i] → weights[i] ∈ [0.01, 1.0], порядок сохраняется.
    """
    @abstractmethod
    def calculate_trust_weights(self, texts: List[str]) -> List[float]: ...


class ExtractionStage(ABC):
    """
    Стадия 2: морфологическая экстракция кандидатов из одного отзыва.
    raw_text → List[Candidate]  (n-граммы, прошедшие POS-фильтр)
    """
    @abstractmethod
    def extract(self, raw_text: str) -> List[Candidate]: ...


class ScoringStage(ABC):
    """
    Стадия 3: семантический скоринг + MMR-диверсификация.
    List[Candidate] → List[ScoredCandidate]  (с embedding для кластеризации)
    """
    @abstractmethod
    def score_and_select(self, candidates: List[Candidate]) -> List[ScoredCandidate]: ...


class ClusteringStage(ABC):
    """
    Стадия 4: группировка кандидатов в аспект-кластеры.
    List[ScoredCandidate] → Dict[str, AspectInfo]

    Замена: любой алгоритм кластеризации с тем же контрактом
    (например, LDA, BERTopic, GMM) подключается без изменений в pipeline.py.
    """
    @abstractmethod
    def cluster(self, candidates: List[ScoredCandidate]) -> Dict[str, AspectInfo]: ...


class SentimentStage(ABC):
    """
    Стадия 5: NLI-сентимент для пар (sentence, aspect).
    List[SentimentPair] → List[SentimentResult],  score ∈ [1.0, 5.0]
    """
    @abstractmethod
    def batch_analyze(self, pairs: List[SentimentPair]) -> List[SentimentResult]: ...


class AggregationStage(ABC):
    """
    Стадия 6: байесовская агрегация оценок по аспектам.
    List[AggregationInput] → AggregationResult  (AspectScore + ковариация)
    """
    @abstractmethod
    def aggregate(self, inputs: List[AggregationInput]) -> AggregationResult: ...
