"""
stages.py — Абстрактные базовые классы (ABCs) для каждой стадии ABSA-пайплайна.

Контракты фиксируют входные и выходные типы каждого узла.
Любую стадию можно заменить альтернативной реализацией:
достаточно унаследовать соответствующий ABC и реализовать его единственный
абстрактный метод — пайплайн подхватит без других изменений.

Таблица контрактов:
  ┌────────────────────┬──────────────────────────────────┬──────────────────────────┐
  │ Стадия             │ Вход                             │ Выход                    │
  ├────────────────────┼──────────────────────────────────┼──────────────────────────┤
  │ FraudStage         │ List[str]  (тексты отзывов)      │ List[float]  (веса)      │
  │ ExtractionStage    │ str  (сырой текст отзыва)        │ List[Candidate]           │
  │ ScoringStage       │ List[Candidate]                  │ List[ScoredCandidate]    │
  │ ClusteringStage    │ List[ScoredCandidate]            │ Dict[str, AspectInfo]    │
  │ SentimentStage     │ List[SentimentPair]              │ List[SentimentResult]    │
  │ AggregationStage   │ List[AggregationInput]           │ AggregationResult        │
  └────────────────────┴──────────────────────────────────┴──────────────────────────┘

SentimentPair   = Tuple[review_id, sentence, aspect_name, nli_label, weight]
AggregationInput = Dict{"review_id", "aspects": Dict[str, float], "fraud_weight", "date"}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

# Импортируем только типы — модули уже существуют, циклических импортов нет.
from src.discovery.candidates import Candidate
from src.discovery.clusterer import AspectInfo
from src.discovery.scorer import ScoredCandidate
from src.math.engine import AggregationResult
from src.sentiment.engine import SentimentResult

# Псевдонимы типов для читаемости сигнатур
SentimentPair = Tuple[str, str, str, str, float]   # (review_id, sentence, aspect, nli_label, weight)
AggregationInput = Dict                             # {"review_id", "aspects", "fraud_weight", "date"}


# ── Стадия 1: Фрод-детекция ───────────────────────────────────────────────────

class FraudStage(ABC):
    """
    Вычисляет веса доверия для списка текстов отзывов.

    Контракт:
        texts[i] → trust_weights[i]  (один вес на один отзыв, порядок сохраняется)
        Значения: [0.01, 1.0]
    """

    @abstractmethod
    def calculate_trust_weights(self, texts: List[str]) -> List[float]:
        """
        texts: список чистых текстов отзывов (в том же порядке, что ReviewInput).
        Возвращает список весов [0.01, 1.0] той же длины.
        """
        ...


# ── Стадия 2: Извлечение кандидатов ──────────────────────────────────────────

class ExtractionStage(ABC):
    """
    Извлекает аспект-кандидаты из сырого текста одного отзыва.

    Контракт:
        raw_text → List[Candidate]  (морфо-фильтрованные n-граммы)
    """

    @abstractmethod
    def extract(self, raw_text: str) -> List[Candidate]:
        """
        raw_text: склеенный текст отзыва (pros + cons + full_text).
        Возвращает кандидатов из всех предложений текста.
        """
        ...


# ── Стадия 3: Семантический скоринг + MMR ─────────────────────────────────────

class ScoringStage(ABC):
    """
    Семантически отбирает кандидатов из всех отзывов сразу.

    Контракт:
        List[Candidate]  →  List[ScoredCandidate]
        Фильтрует по cosine_threshold, применяет MMR-диверсификацию.
        ScoredCandidate содержит embedding (np.ndarray) для кластеризации.
    """

    @abstractmethod
    def score_and_select(self, candidates: List[Candidate]) -> List[ScoredCandidate]:
        """
        candidates: плоский список кандидатов со всех отзывов.
        Возвращает отфильтрованный и ранжированный список ScoredCandidate.
        """
        ...


# ── Стадия 4: Кластеризация аспектов ─────────────────────────────────────────

class ClusteringStage(ABC):
    """
    Группирует scored_candidates в именованные аспект-кластеры.

    Контракт:
        List[ScoredCandidate]  →  Dict[str, AspectInfo]
        Ключ — имя аспекта (якорь или residual-метка).
        AspectInfo содержит: keywords, centroid_embedding, nli_label.

    Замена этой стадии: любой алгоритм кластеризации, принимающий
    List[ScoredCandidate] и возвращающий Dict[str, AspectInfo],
    будет корректно подхвачен пайплайном и eval-скриптами.
    """

    @abstractmethod
    def cluster(self, candidates: List[ScoredCandidate]) -> Dict[str, AspectInfo]:
        """
        candidates: отобранные и векторизованные кандидаты.
        Возвращает словарь {aspect_name: AspectInfo}.
        Пустой словарь — допустимый результат (нет аспектов).
        """
        ...


# ── Стадия 5: NLI Sentiment ───────────────────────────────────────────────────

class SentimentStage(ABC):
    """
    Оценивает тональность каждой (sentence, aspect) пары через NLI.

    Контракт:
        List[SentimentPair]  →  List[SentimentResult]
        SentimentPair = (review_id, sentence, aspect_name, nli_label, weight)
        SentimentResult.score ∈ [1.0, 5.0]
    """

    @abstractmethod
    def batch_analyze(self, pairs: List[SentimentPair]) -> List[SentimentResult]:
        """
        pairs: список пар для NLI-инференса.
        Возвращает SentimentResult для каждой пары той же длины.
        """
        ...


# ── Стадия 6: Математическая агрегация ───────────────────────────────────────

class AggregationStage(ABC):
    """
    Агрегирует сентимент-оценки в итоговые метрики аспектов.

    Контракт:
        List[AggregationInput]  →  AggregationResult
        AggregationInput: {"review_id": str, "aspects": Dict[str, float],
                            "fraud_weight": float, "date": datetime}
        AggregationResult.aspects: Dict[str, AspectScore] с байесовским сглаживанием.
    """

    @abstractmethod
    def aggregate(self, inputs: List[AggregationInput]) -> AggregationResult:
        """
        inputs: список записей по отзывам с взвешенными аспект-оценками.
        Возвращает AggregationResult с финальными скорами и ковариацией.
        """
        ...
