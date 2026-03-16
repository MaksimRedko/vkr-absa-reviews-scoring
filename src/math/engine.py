"""
Математическое ядро v2 (RatingMathEngine)

Агрегирует аспектные NLI-оценки с учётом:
  - Весов доверия (AntiFraud trust_weight)
  - Временного экспоненциального затухания
  - Байесовского сглаживания с нейтральным априором (3.0)
  - Ledoit-Wolf shrinkage-ковариации для portfolio variance
  - Дисперсии как UI-алерта (controversy), без влияния на score

Ключевые отличия от v1:
  4.1  prior = 3.0 (нейтральный MaxEnt), а не global_avg_rating
       C = min(median_mentions, C_max)
  4.2  Ledoit-Wolf ковариация для персонализации (portfolio variance)
  4.3  variance_penalty = 0.0 → controversy только в UI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sklearn.covariance import LedoitWolf

from configs.configs import config


@dataclass
class AspectScore:
    """Итоговые метрики одного аспекта"""
    name: str
    score: float               # Байесовски сглаженная оценка [1, 5]
    raw_mean: float            # Чистое взвешенное среднее
    controversy: float         # Взвешенное стандартное отклонение (для UI-алерта)
    mentions: int              # Число упоминаний (до фильтрации ботов)
    effective_mentions: float  # Сумма весов (после AntiFraud * Time)


@dataclass
class AggregationResult:
    """Полный результат агрегации по товару"""
    aspects: Dict[str, AspectScore] = field(default_factory=dict)
    covariance_matrix: Optional[np.ndarray] = None  # Ledoit-Wolf Σ (K×K)
    aspect_order: List[str] = field(default_factory=list)  # Порядок осей в Σ


class RatingMathEngine:
    """
    Байесовская агрегация аспектных оценок v2.

    Формула для аспекта k:
        C = min(median_mentions, C_max)
        bayesian_score_k = (N_k / (N_k + C)) * weighted_mean_k + (C / (N_k + C)) * prior

    Где:
        N_k   — сумма весов по аспекту k (trust_weight * time_weight)
        prior — нейтральный априор 3.0 (MaxEnt)
        C_max — из конфига (по умолчанию 3)
    """

    def __init__(self):
        self.k_time = 1.0 / config.math.time_decay_days
        self.prior = config.math.prior_mean
        self.c_max = config.math.prior_strength_max
        self.variance_penalty = config.math.variance_penalty

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def aggregate(self, reviews_data: List[Dict]) -> AggregationResult:
        """
        Главный метод агрегации.

        Args:
            reviews_data: список словарей вида
                {
                    "review_id": str,
                    "aspects": {"Качество": 4.7, "Логистика": 1.3, ...},
                    "fraud_weight": float,  # от AntiFraudEngine
                    "date": datetime | None,
                }

        Returns:
            AggregationResult с оценками по каждому аспекту и ковариационной матрицей.
        """
        current_date = datetime.now()

        # 1. Разнести оценки по корзинам аспектов
        aspect_buckets: Dict[str, List[dict]] = {}

        for review in reviews_data:
            w_fraud = review.get("fraud_weight", 1.0)
            w_time = self._time_weight(review.get("date"), current_date)
            w = w_fraud * w_time

            if w < self.variance_penalty if self.variance_penalty > 0 else w < 0.01:
                continue

            for aspect_name, score in review.get("aspects", {}).items():
                if aspect_name not in aspect_buckets:
                    aspect_buckets[aspect_name] = []
                aspect_buckets[aspect_name].append({"score": score, "weight": w})

        # 2. C = min(median_mentions, C_max)
        counts = [len(v) for v in aspect_buckets.values()]
        median_mentions = float(np.median(counts)) if counts else 2.0
        c_strength = min(median_mentions, self.c_max)

        # 3. Считаем метрики по каждому аспекту
        aspects: Dict[str, AspectScore] = {}

        for name, items in aspect_buckets.items():
            scores = np.array([x["score"] for x in items])
            weights = np.array([x["weight"] for x in items])

            n_k = float(np.sum(weights))

            if n_k == 0:
                weighted_mean = self.prior
            else:
                weighted_mean = float(np.average(scores, weights=weights))

            # Взвешенное стандартное отклонение
            if len(scores) > 1 and n_k > 0:
                variance = float(np.average((scores - weighted_mean) ** 2, weights=weights))
                std_dev = float(np.sqrt(variance))
            else:
                std_dev = 0.0

            # Байесовское сглаживание с нейтральным априором
            bayesian = (n_k / (n_k + c_strength)) * weighted_mean + \
                       (c_strength / (n_k + c_strength)) * self.prior

            # variance_penalty = 0.0 → не влияет на score
            final_score = bayesian - (self.variance_penalty * std_dev)
            final_score = max(1.0, min(5.0, final_score))

            aspects[name] = AspectScore(
                name=name,
                score=round(final_score, 2),
                raw_mean=round(weighted_mean, 2),
                controversy=round(std_dev, 2),
                mentions=len(scores),
                effective_mentions=round(n_k, 2),
            )

        # 4. Ledoit-Wolf ковариация
        aspect_order = sorted(aspects.keys())
        cov_matrix = self._compute_covariance(reviews_data, aspect_order, current_date)

        return AggregationResult(
            aspects=aspects,
            covariance_matrix=cov_matrix,
            aspect_order=aspect_order,
        )

    def calculate_personal_rating(
        self,
        result: AggregationResult,
        user_weights: Dict[str, float],
    ) -> float:
        """
        Персональный рейтинг на основе пользовательских весов.
        user_weights: {"Качество": 1.0, "Цена": 0.0, ...} — от 0 до 1.
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for name, asp in result.aspects.items():
            u_w = user_weights.get(name, 1.0)
            weighted_sum += asp.score * u_w
            total_weight += u_w

        if total_weight == 0:
            return self.prior

        return round(weighted_sum / total_weight, 2)

    def calculate_portfolio_variance(
        self,
        result: AggregationResult,
        user_weights: Dict[str, float],
    ) -> float:
        """
        Portfolio variance с учётом Ledoit-Wolf ковариации.
        Позволяет оценить «неопределённость» персональной оценки.

        Var_portfolio = w^T Σ w / (w^T w)^2
        где w — вектор пользовательских весов.
        """
        if result.covariance_matrix is None or len(result.aspect_order) < 2:
            return 0.0

        w = np.array([user_weights.get(name, 1.0) for name in result.aspect_order])
        total = np.sum(w)
        if total == 0:
            return 0.0

        w_norm = w / total
        portfolio_var = float(w_norm @ result.covariance_matrix @ w_norm)
        return round(max(0.0, portfolio_var), 4)

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _time_weight(self, review_date: Optional[datetime], current_date: datetime) -> float:
        """Экспоненциальное затухание: e^(-k * delta_days)"""
        if not review_date:
            return 1.0
        delta = max(0, (current_date - review_date).days)
        return float(np.exp(-self.k_time * delta))

    def _compute_covariance(
        self,
        reviews_data: List[Dict],
        aspect_order: List[str],
        current_date: datetime,
    ) -> Optional[np.ndarray]:
        """
        Строит Ledoit-Wolf shrinkage ковариационную матрицу.
        Использует только отзывы, в которых >= 2 аспектов.
        Fallback на диагональную матрицу, если таких отзывов < 5.
        """
        if len(aspect_order) < 2:
            return None

        aspect_idx = {name: i for i, name in enumerate(aspect_order)}
        k = len(aspect_order)

        rows = []
        for review in reviews_data:
            w_fraud = review.get("fraud_weight", 1.0)
            if w_fraud < 0.01:
                continue

            asp = review.get("aspects", {})
            present = [a for a in aspect_order if a in asp]
            if len(present) < 2:
                continue

            row = np.full(k, np.nan)
            for a in present:
                row[aspect_idx[a]] = asp[a]
            rows.append(row)

        if len(rows) < 5:
            return self._diagonal_fallback(reviews_data, aspect_order, current_date)

        X = np.array(rows)
        col_means = np.nanmean(X, axis=0)
        for j in range(k):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_means[j]

        try:
            lw = LedoitWolf().fit(X)
            return lw.covariance_
        except Exception:
            return self._diagonal_fallback(reviews_data, aspect_order, current_date)

    def _diagonal_fallback(
        self,
        reviews_data: List[Dict],
        aspect_order: List[str],
        current_date: datetime,  # noqa: ARG002
    ) -> np.ndarray:
        """Диагональная ковариационная матрица (fallback)."""
        k = len(aspect_order)
        variances = np.ones(k)

        aspect_scores: Dict[str, list] = {a: [] for a in aspect_order}
        for review in reviews_data:
            if review.get("fraud_weight", 1.0) < 0.01:
                continue
            for a in aspect_order:
                if a in review.get("aspects", {}):
                    aspect_scores[a].append(review["aspects"][a])

        for i, a in enumerate(aspect_order):
            if len(aspect_scores[a]) > 1:
                variances[i] = float(np.var(aspect_scores[a]))

        return np.diag(variances)


# ------------------------------------------------------------------
# Тест
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    engine = RatingMathEngine()

    mock_data = [
        {
            "review_id": "r1",
            "fraud_weight": 0.01,  # БОТ
            "date": datetime(2026, 3, 8),
            "aspects": {"Качество": 5.0, "Цена": 5.0},
        },
        {
            "review_id": "r2",
            "fraud_weight": 0.95,
            "date": datetime(2025, 1, 1),  # Старый (~1 год)
            "aspects": {"Качество": 2.0},
        },
        {
            "review_id": "r3",
            "fraud_weight": 0.98,
            "date": datetime(2026, 3, 8),  # Свежий
            "aspects": {"Качество": 5.0, "Цена": 1.0},
        },
        {
            "review_id": "r4",
            "fraud_weight": 0.90,
            "date": datetime(2026, 2, 1),
            "aspects": {"Качество": 4.0, "Цена": 2.0, "Логистика": 5.0},
        },
        {
            "review_id": "r5",
            "fraud_weight": 0.85,
            "date": datetime(2026, 1, 15),
            "aspects": {"Качество": 3.5, "Логистика": 4.0},
        },
    ]

    print("Запуск математического ядра v2\n")

    result = engine.aggregate(mock_data)

    print("Результаты по аспектам:")
    for name in sorted(result.aspects.keys()):
        a = result.aspects[name]
        print(f"  {name}: score={a.score}, raw_mean={a.raw_mean}, "
              f"controversy={a.controversy}, mentions={a.mentions}, "
              f"effective_N={a.effective_mentions}")

    print(f"\nКовариационная матрица ({' x '.join(result.aspect_order)}):")
    if result.covariance_matrix is not None:
        print(np.round(result.covariance_matrix, 3))
    else:
        print("  (не вычислена — мало данных)")

    # Тест персонализации
    prefs = {"Качество": 1.0, "Цена": 0.0, "Логистика": 0.5}
    personal = engine.calculate_personal_rating(result, prefs)
    port_var = engine.calculate_portfolio_variance(result, prefs)
    print(f"\nПерсональный рейтинг (Качество=1, Цена=0, Логистика=0.5): {personal}")
    print(f"Portfolio variance: {port_var}")

    # Проверка критериев
    print("\nКритерии приёмки:")
    q_score = result.aspects.get("Качество")
    if q_score:
        prior_pull = abs(q_score.score - q_score.raw_mean)
        print(f"  Бот (fraud=0.01) отсечён: {'OK' if 'r1' not in str(mock_data) or q_score.mentions <= 4 else 'FAIL'}")
        print(f"  'Качество' score={q_score.score} (prior=3.0 подтягивает): "
              f"{'OK' if q_score.score < q_score.raw_mean + 0.01 else 'FAIL'}")
        print(f"  controversy={q_score.controversy} присутствует, но не влияет на score: "
              f"{'OK' if config.math.variance_penalty == 0.0 else 'FAIL'}")
