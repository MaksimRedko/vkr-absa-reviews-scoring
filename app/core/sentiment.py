import torch
from transformers import pipeline
from typing import List, Dict
import numpy as np


class SentimentEngine:
    """
    Модуль целевого анализа тональности (Targeted Sentiment Analysis).
    Использует подход Zero-Shot NLI с мягкой формулой скоринга.
    """

    def __init__(self, model_name: str = "cointegrated/rubert-base-cased-nli-twoway"):
        print(f"⏳ Инициализация SentimentEngine (модель: {model_name})...")

        # Определяем девайс (GPU если есть, иначе CPU)
        device = 0 if torch.cuda.is_available() else -1

        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device
            )
            print("✅ SentimentEngine готов.")
        except Exception as e:
            print(f"❌ Ошибка загрузки NLI модели: {e}")
            raise e

    def batch_analyze(self, text: str, aspects: List[str]) -> Dict[str, float]:
        """
        Прогоняет один отзыв по всем найденным аспектам.
        Возвращает словарь { "Аспект": Оценка (1.0 - 5.0) }
        """
        results = {}

        # Для каждого аспекта формируем пару гипотез
        # Мы убрали "Нейтрально", чтобы заставить модель определиться с вектором эмоции
        for aspect in aspects:
            # Формулировки стали проще и понятнее для модели
            aspect_lower = aspect.lower()
            pos_label = f"хороший {aspect_lower}"
            neg_label = f"плохой {aspect_lower}"
            candidate_labels = [pos_label, neg_label]

            try:
                output = self.classifier(text, candidate_labels, multi_label=False)

                # Достаем вероятности
                scores = dict(zip(output['labels'], output['scores']))
                pos_score = scores.get(pos_label, 0.0)
                neg_score = scores.get(neg_label, 0.0)

                # --- НОВАЯ ФОРМУЛА ---
                # База 3.0. Сдвигаем вверх или вниз в зависимости от разницы уверенности.
                # Множитель 2.0 растягивает диапазон до [1.0, 5.0].
                delta = pos_score - neg_score
                raw_score = 3.0 + (delta * 2.0)

                # Обрезаем границы (на случай экстремальной уверенности)
                final_score = max(1.0, min(5.0, raw_score))

                results[aspect] = round(final_score, 2)

            except Exception as e:
                print(f"⚠️ Ошибка при анализе аспекта '{aspect}': {e}")
                results[aspect] = 3.0

        return results


# --- ТЕСТОВЫЙ ЗАПУСК ---
if __name__ == "__main__":
    engine = SentimentEngine()

    # Сложные кейсы с контекстом
    test_cases = [
        ("Салфетки высохли через неделю, просто ужас!", ["Качество", "Внешний вид"]),
        ("Коробка была мятая, но сами салфетки выглядят стильно.", ["Логистика", "Внешний вид"]),
        ("Стоят копейки, а чистят отлично.", ["Цена", "Эффективность"]),
        ("Ну такое, пойдет, но могло быть лучше.", ["Общее впечатление"])  # Нейтральный кейс
    ]

    print("\n🧠 Тестирование NLI Sentiment (v2 - Soft Scoring)...\n")

    for text, aspects in test_cases:
        print(f"📝 Отзыв: '{text}'")
        scores = engine.batch_analyze(text, aspects)
        for asp, score in scores.items():
            # Рисуем бар для наглядности
            bar_len = int(score * 2)
            bar = "🟩" * bar_len + "⬜" * (10 - bar_len)
            print(f"   {bar} {asp}: {score}")
        print("-" * 40)