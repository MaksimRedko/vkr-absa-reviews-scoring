import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os


class AntiFraudEngine:
    """
    Модуль выявления накруток и дубликатов.
    Рассчитывает 'Trust Score' (Коэффициент доверия) для каждого отзыва
    на основе его уникальности и информативности.
    """

    def __init__(self):
        # Используем ту же легкую модель, что и для кластеризации
        # Отключаем ворнинги
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        model_name = 'cointegrated/rubert-tiny2'

        print(f"⏳ Инициализация AntiFraudEngine (модель: {model_name})...")
        try:
            self.model = SentenceTransformer(model_name)
            print("✅ AntiFraudEngine готов.")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise e

    def _calculate_length_penalty(self, texts: List[str]) -> np.ndarray:
        """
        Штраф за длину. Короткие отзывы получают низкий вес.
        Используем сигмоиду: вес резко падает, если слов < 5.
        """
        weights = []
        for text in texts:
            # Считаем количество слов
            word_count = len(text.split())

            # Формула: 1 / (1 + e^(-0.5 * (x - 4)))
            # 1 слово -> вес 0.18
            # 3 слова -> вес 0.37
            # 5 слов  -> вес 0.62
            # 10 слов -> вес 0.95
            weight = 1 / (1 + np.exp(-0.8 * (word_count - 4)))
            weights.append(weight)

        return np.array(weights)

    def _calculate_uniqueness_penalty(self, embeddings: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Штраф за дубликаты.
        Строит матрицу косинусного сходства. Если отзыв похож на другой > threshold,
        его вес снижается.
        """
        n = len(embeddings)
        if n < 2:
            return np.ones(n)

        # 1. Считаем матрицу NxN (все со всеми)
        sim_matrix = cosine_similarity(embeddings)

        # 2. Обнуляем диагональ (чтобы отзыв не считался дубликатом самого себя)
        np.fill_diagonal(sim_matrix, 0)

        # 3. Для каждого отзыва находим МАКСИМАЛЬНОЕ сходство с любым другим
        max_similarities = np.max(sim_matrix, axis=1)

        # 4. Рассчитываем вес уникальности:
        # Если сходство 0.1 (уникален) -> вес 1.0
        # Если сходство 0.99 (копия) -> вес 0.01
        # Используем квадрат для более жесткого штрафа спамеров
        uniqueness_weights = 1.0 - (max_similarities ** 2)

        # Защита от отрицательных чисел (на всякий случай)
        return np.maximum(uniqueness_weights, 0.01)

    def calculate_trust_weights(self, reviews: List[str]) -> List[float]:
        """
        Главный метод.
        Принимает список текстов отзывов.
        Возвращает список весов [0.0 - 1.0] для каждого отзыва.
        """
        if not reviews:
            return []

        # 1. Векторизация
        embeddings = self.model.encode(reviews)

        # 2. Расчет факторов
        len_weights = self._calculate_length_penalty(reviews)
        uniq_weights = self._calculate_uniqueness_penalty(embeddings)

        # 3. Итоговый вес (произведение факторов)
        # Trust = Длина * Уникальность
        final_weights = len_weights * uniq_weights

        return final_weights.tolist()


# --- ТЕСТОВЫЙ ЗАПУСК ---
if __name__ == "__main__":
    fraud_engine = AntiFraudEngine()

    mock_reviews = [
        "Отличный товар, всем рекомендую, быстрая доставка!",  # Отзыв А (Оригинал)
        "Норм",  # Отзыв Б (Слишком короткий)
        "Отличный товар, всем рекомендую, быстрая доставка!",  # Отзыв В (Полная копия А)
        "Товар отличный, рекомендую всем, доставка быстрая.",  # Отзыв Г (Рерайт А)
        "Купил, распаковал, пользуюсь. Пока нареканий нет, но коробка была немного помята. За свои деньги топ.",
        "Член просто огромный. В мою пизденку засовывается отлично. Я сквиртила всю ночь",
        # Отзыв Д (Качественный)
    ]

    print(f"\n🕵️ Анализ {len(mock_reviews)} отзывов на фрод...\n")
    weights = fraud_engine.calculate_trust_weights(mock_reviews)

    for i, (text, w) in enumerate(zip(mock_reviews, weights)):
        status = "✅ Живой" if w > 0.5 else "❌ Подозрительный"
        print(f"Review #{i + 1}: {status} (Weight: {w:.4f})")
        print(f"Text: '{text}'")
        print("-" * 40)