import pandas as pd
from typing import List, Dict
import time

# Импортируем наши кирпичики
from app.core.data_loader import DataLoader
from app.core.npl_parser import NLPProcessor
from app.core.clustering import AspectDiscoveryEngine
from app.core.sentiment import SentimentEngine
from app.core.anti_fraud import AntiFraudEngine
from app.core.math_engine import RatingMathEngine
from app.schemas.models import ReviewInput


class ProductAnalyzerService:
    """
    Оркестратор. Связывает все ML-модули в единый пайплайн.
    """

    def __init__(self, db_path: str = "dataset.db"):
        print("\n🚀 Запуск сервиса анализа (Загрузка нейросетей)...")
        self.loader = DataLoader(db_path)

        # Инициализируем ядра (это займет пару секунд)
        self.nlp = NLPProcessor()
        self.clustering = AspectDiscoveryEngine()  # Грузит rubert-tiny2
        self.sentiment = SentimentEngine()  # Грузит NLI
        self.fraud = AntiFraudEngine()  # Грузит rubert-tiny2 (кэшированный)
        self.math = RatingMathEngine()
        print("🚀 Сервис готов к работе!\n")

    def analyze_product(self, nm_id: int, limit: int = 50) -> Dict:
        """
        Полный цикл анализа одного товара.
        limit=50, чтобы не ждать NLI полчаса на тестах.
        """
        start_time = time.time()

        # 1. Загрузка данных
        print(f"📦 [1/6] Загрузка отзывов для товара {nm_id}...")
        reviews: List[ReviewInput] = self.loader.load_reviews_for_product(nm_id, limit)
        if not reviews:
            return {"error": "Нет отзывов"}

        # Превращаем в тексты
        raw_texts = [r.clean_text for r in reviews]

        # 2. Анти-фрод (Считаем веса доверия)
        print(f"🕵️ [2/6] Поиск накруток среди {len(reviews)} отзывов...")
        fraud_weights = self.fraud.calculate_trust_weights(raw_texts)

        # 3. NLP Препроцессинг + Сбор кандидатов
        print(f"🧠 [3/6] Лингвистический анализ...")
        all_lemmas = []
        processed_reviews = []

        for i, text in enumerate(raw_texts):
            res = self.nlp.process_review(text)
            all_lemmas.extend(res['lemmas'])

            processed_reviews.append({
                "id": reviews[i].id,
                "text": text,
                "date": reviews[i].created_date,
                "fraud_weight": fraud_weights[i],
                "sentences": res['sentences']  # Для NLI
            })

        # 4. Поиск Аспектов (Кластеризация)
        print(f"💎 [4/6] Кластеризация аспектов...")
        # Берем топ-150 кандидатов через TF-IDF внутри nlp_parser (надо было добавить метод, но пока возьмем raw lemmas)
        # Для простоты кидаем все леммы, DBSCAN разберется
        aspect_clusters = self.clustering.discover_aspects(all_lemmas)

        # Получаем чистый список названий аспектов ["Цена", "Качество"]
        aspect_names = list(aspect_clusters.keys())
        print(f"   -> Найдено {len(aspect_names)} аспектов: {aspect_names}")

        # 5. Оценка Тональности (NLI - Самое долгое)
        print(f"❤️ [5/6] Оценка тональности (NLI)...")
        reviews_with_scores = []

        for review in processed_reviews:
            # Оцениваем отзыв целиком или по предложениям.
            # Для скорости оценим текст целиком относительно аспектов.
            scores = self.sentiment.batch_analyze(review['text'], aspect_names)

            review["aspects"] = scores
            reviews_with_scores.append(review)

        # 6. Математическая агрегация
        print(f"🧮 [6/6] Финальный расчет рейтинга...")
        final_stats = self.math.aggregate_aspect_scores(
            reviews_with_scores,
            global_avg_rating=4.0  # Можно посчитать среднее из reviews.rating
        )

        total_time = time.time() - start_time

        return {
            "product_id": nm_id,
            "reviews_processed": len(reviews),
            "processing_time": round(total_time, 2),
            "aspects": final_stats
        }