import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime


class RatingMathEngine:
    """
    Математическое ядро системы.
    Агрегирует оценки нейросети, применяет временное затухание,
    анти-фрод веса и байесовское сглаживание.
    """

    def __init__(self, time_decay_days: int = 365, variance_penalty: float = 0.5):
        # Коэффициент затухания: через 365 дней вес отзыва упадет в ~2.7 раза
        self.k_time = 1 / time_decay_days
        # Сила штрафа за противоречивость (чем больше, тем сильнее наказываем за споры)
        self.variance_penalty = variance_penalty

    def _calculate_time_weight(self, review_date: datetime, current_date: datetime) -> float:
        """Экспоненциальное затухание по времени."""
        if not review_date:
            return 1.0

        delta_days = (current_date - review_date).days
        delta_days = max(0, delta_days)  # Защита от будущего

        # Формула: e^(-k * t)
        return np.exp(-self.k_time * delta_days)

    def aggregate_aspect_scores(
            self,
            reviews_data: List[Dict],
            global_avg_rating: float = 3.0
    ) -> Dict[str, Dict]:
        """
        Главный метод агрегации.

        Args:
            reviews_data: Список словарей, где каждый словарь - это один отзыв:
                          {
                            "aspects": {"Экран": 5.0, "Цена": 1.0},
                            "fraud_weight": 0.05,  (от AntiFraudEngine)
                            "date": datetime(...)
                          }
            global_avg_rating: Средний рейтинг товара (или 3.0 если неизвестно).

        Returns:
            Словарь с итоговыми метриками по каждому аспекту.
        """
        current_date = datetime.now()

        # 1. Трансформируем список отзывов в удобную структуру по аспектам
        # aspect_buckets = { "Экран": [ {"score": 5.0, "weight": 0.8}, ... ] }
        aspect_buckets = {}

        for review in reviews_data:
            # Считаем итоговый вес отзыва: АнтиФрод * Время
            w_fraud = review.get("fraud_weight", 1.0)
            w_time = self._calculate_time_weight(review.get("date"), current_date)
            final_weight = w_fraud * w_time

            # Если вес отзыва околонулевой (бот), пропускаем его влияние
            if final_weight < 0.01:
                continue

            # Раскидываем оценки по корзинам аспектов
            for aspect_name, score in review.get("aspects", {}).items():
                if aspect_name not in aspect_buckets:
                    aspect_buckets[aspect_name] = []

                aspect_buckets[aspect_name].append({
                    "score": score,
                    "weight": final_weight
                })

        # 2. Считаем статистики для каждого аспекта
        results = {}

        # Вычисляем порог доверия (m) как медиану количества упоминаний
        # Если аспектов мало, берем дефолт = 2
        counts = [len(v) for v in aspect_buckets.values()]
        m_threshold = np.median(counts) if counts else 2.0

        for aspect, items in aspect_buckets.items():
            scores = np.array([x["score"] for x in items])
            weights = np.array([x["weight"] for x in items])

            # v - сумма весов (количество "эффективных" голосов)
            v_sum = np.sum(weights)

            # Взвешенное среднее (Weighted Mean)
            if v_sum == 0:
                weighted_mean = 3.0
            else:
                weighted_mean = np.average(scores, weights=weights)

            # Дисперсия (разброс мнений)
            # Если отзыв один, дисперсия 0
            if len(scores) > 1:
                variance = np.average((scores - weighted_mean) ** 2, weights=weights)
                std_dev = np.sqrt(variance)
            else:
                std_dev = 0.0

            # --- ФОРМУЛА БАЙЕСА ---
            # R = (v / (v + m)) * R_aspect + (m / (v + m)) * C
            # Подтягиваем оценку к глобальному среднему, если голосов мало
            bayesian_score = (v_sum / (v_sum + m_threshold)) * weighted_mean + \
                             (m_threshold / (v_sum + m_threshold)) * global_avg_rating

            # --- ШТРАФ ЗА ПРОТИВОРЕЧИВОСТЬ ---
            # Чем выше разброс мнений (std_dev), тем ниже итоговый рейтинг
            final_score = bayesian_score - (self.variance_penalty * std_dev)

            # Клиппинг (чтобы не уйти за границы 1..5)
            final_score = max(1.0, min(5.0, final_score))

            results[aspect] = {
                "score": round(final_score, 2),  # Итоговая оценка для диаграммы
                "raw_mean": round(weighted_mean, 2),  # Чистая средняя (для справки)
                "controversy": round(std_dev, 2),  # Индекс противоречивости
                "mentions": len(scores)  # Сколько людей упомянуло
            }

        return results

    def calculate_personal_rating(self, system_results: Dict, user_weights: Dict) -> float:
        """
        Считает одну цифру (Персональный рейтинг) на основе ползунков юзера.
        user_weights: { "Экран": 1.0, "Цена": 0.0 } (от 0 до 1)
        """
        weighted_sum = 0.0
        total_user_weight = 0.0

        for aspect, metrics in system_results.items():
            # Если юзер не настроил этот аспект, берем вес 0.5 (средняя важность)
            # Или можно брать 1.0, как договоримся.
            u_weight = user_weights.get(aspect, 1.0)

            weighted_sum += metrics["score"] * u_weight
            total_user_weight += u_weight

        if total_user_weight == 0:
            return 0.0

        return round(weighted_sum / total_user_weight, 2)


# --- ИНТЕГРАЦИОННЫЙ ТЕСТ (СБОРКА ВСЕГО) ---
if __name__ == "__main__":
    math_engine = RatingMathEngine()

    # Симулируем данные, которые прошли через ВСЕ наши модули
    # Отзыв 1 (Бот) - fraud_weight низкий
    # Отзыв 2 (Старый) - дата старая
    # Отзыв 3 (Свежий) - всё ок

    mock_processed_data = [
        {
            "id": 1,
            "fraud_weight": 0.01,  # БОТ!
            "date": datetime(2026, 3, 8),
            "aspects": {"Качество": 5.0, "Цена": 5.0}  # Бот всё хвалит
        },
        {
            "id": 2,
            "fraud_weight": 0.95,
            "date": datetime(2025, 1, 1),  # Старый (год назад)
            "aspects": {"Качество": 2.0}  # Раньше было плохо
        },
        {
            "id": 3,
            "fraud_weight": 0.98,
            "date": datetime(2026, 3, 8),  # Свежий
            "aspects": {"Качество": 5.0, "Цена": 1.0}  # Сейчас качество супер, но дорого
        }
    ]

    print("\n🧮 Запуск Математического Ядра...\n")

    # 1. Агрегация
    results = math_engine.aggregate_aspect_scores(mock_processed_data, global_avg_rating=4.0)

    print("📊 Результаты по аспектам:")
    for asp, metrics in results.items():
        print(
            f"   🔹 {asp}: {metrics['score']} (Сырое: {metrics['raw_mean']}, Споры: {metrics['controversy']}, Упом: {metrics['mentions']})")

    print("\n💡 Объяснение:")
    print(
        "   'Качество' получило высокую оценку, потому что свежий отзыв (5.0) перевесил старый (2.0), а бот (5.0) был проигнорирован.")

    # 2. Персонализация
    # Юзер говорит: "Мне плевать на Цену, мне важно Качество"
    user_prefs = {"Качество": 1.0, "Цена": 0.0}
    personal_score = math_engine.calculate_personal_rating(results, user_prefs)

    print(f"\n🎯 Персональный рейтинг (Важно только Качество): {personal_score}")