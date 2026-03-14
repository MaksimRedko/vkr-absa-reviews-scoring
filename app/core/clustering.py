import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


class AspectDiscoveryEngine:
    UNIVERSAL_ANCHORS_WORDS = {
        "Цена": ["цена", "стоимость", "деньги", "выгода", "дорого", "дешево", "скидка"],
        "Качество": ["качество", "надежность", "сборка", "материал", "прочность", "брак", "хлипкий", "сломался"],
        "Внешний вид": ["дизайн", "красота", "внешность", "стиль", "цвет", "выглядит", "симпатичный"],
        "Удобство": ["удобство", "комфорт", "эргономика", "тяжелый", "легкий", "размер", "габарит"],
        "Функциональность": ["мощность", "скорость", "работа", "производительность", "тормозит", "лагает", "функция"],
        "Логистика": ["доставка", "коробка", "продавец", "упаковка", "курьер", "срок", "приехал"],
        "Органолептика": ["вкус", "запах", "аромат", "звук", "шум", "ощупь"]
    }

    def __init__(self):
        # Отключаем ворнинги
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        # Просто указываем имя.
        # Библиотека сама найдет её в кэше или скачает ПРАВИЛЬНУЮ версию с modules.json
        model_name = 'cointegrated/rubert-tiny2'

        print(f"⏳ Инициализация AspectDiscoveryEngine (модель: {model_name})...")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"❌ Ошибка загрузки модели. Проверь интернет или VPN. Ошибка: {e}")
            raise e

        # Предрасчет якорей
        self.anchor_embeddings = {}
        for anchor_name, words in self.UNIVERSAL_ANCHORS_WORDS.items():
            embs = self.model.encode(words)
            centroid = np.mean(embs, axis=0)
            self.anchor_embeddings[anchor_name] = centroid / np.linalg.norm(centroid)

        print("✅ AspectDiscoveryEngine готов.")

    def _get_embeddings(self, words: List[str]) -> np.ndarray:
        embeddings = self.model.encode(words)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def _name_cluster(self, cluster_words: List[str], cluster_embeddings: np.ndarray) -> str:
        cluster_centroid = np.mean(cluster_embeddings, axis=0)
        cluster_centroid = cluster_centroid / np.linalg.norm(cluster_centroid)

        best_anchor = None
        best_sim = -1.0

        for anchor_name, anchor_vector in self.anchor_embeddings.items():
            sim = np.dot(cluster_centroid, anchor_vector)
            if sim > best_sim:
                best_sim = sim
                best_anchor = anchor_name

        # Порог семантической близости (0.40 - оптимально для tiny-bert)
        if best_sim > 0.40:
            return best_anchor

        distances = np.linalg.norm(cluster_embeddings - cluster_centroid, axis=1)
        center_word_idx = np.argmin(distances)
        return cluster_words[center_word_idx].capitalize()

    def discover_aspects(self, candidate_words: List[str]) -> Dict[str, List[str]]:
        words = list(set(candidate_words))
        if len(words) < 2:
            return {"Общее": words}

        embeddings = self._get_embeddings(words)

        # ЗАМЕНА НА DBSCAN (Стабильнее для малых данных)
        # eps=0.6: Если расстояние между словами < 0.6 (косинус > 0.8), они слипаются
        # min_samples=2: Достаточно 2 слов, чтобы создать кластер
        clusterer = DBSCAN(eps=0.4, min_samples=2, metric='euclidean')
        labels = clusterer.fit_predict(embeddings)

        discovered_aspects = {}

        for label in set(labels):
            if label == -1:
                continue  # Мусор

            idxs = np.where(labels == label)[0]
            cluster_words = [words[i] for i in idxs]
            cluster_embs = embeddings[idxs]

            aspect_name = self._name_cluster(cluster_words, cluster_embs)

            if aspect_name in discovered_aspects:
                aspect_name = f"{aspect_name}_{label}"

            discovered_aspects[aspect_name] = cluster_words

        return discovered_aspects


if __name__ == "__main__":
    # Тестовые данные
    mock_candidates = [
        "сломался", "трещина", "хлипкий", "развалился", "пластик",
        "красивый", "уродливый", "дизайн", "выглядит", "цвет", "яркий",
        "салфетка", "тряпка", "туба", "крышка",
        "кошка", "луна", "суббота"
    ]

    engine = AspectDiscoveryEngine()

    print(f"\n🧠 Анализ {len(mock_candidates)} кандидатов...\n")
    aspects = engine.discover_aspects(mock_candidates)

    for aspect, words in aspects.items():
        print(f"📌 Аспект: [{aspect}]")
        print(f"   Слова: {words}\n")