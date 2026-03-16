"""
Модуль AntiFraud (v2)

Рассчитывает вес доверия [0.0, 1.0] для каждого отзыва на основе:
  1. Длины отзыва (сигмоида по количеству слов)
  2. Уникальности (штраф за дубликаты / ботов)

Ключевое отличие от v1:
  - Загрузка модели из локального пути (config.models.encoder_path)
  - Логарифмическая субаддитивная агрегация для кластеров ботов:
    если группа отзывов попарно похожа > uniqueness_threshold,
    их суммарный эффективный вес = ln(1 + count) вместо линейного count.
    Это обрезает влияние ботов-спамеров, не трогая уникальные отзывы.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from configs.configs import config


@dataclass
class TrustResult:
    """Результат оценки доверия для одного отзыва"""
    index: int
    text: str
    trust_weight: float        # Итоговый вес [0.0, 1.0]
    length_weight: float       # Компонента «длина»
    uniqueness_weight: float   # Компонента «уникальность»
    bot_cluster_id: int        # -1 если не в кластере ботов, иначе ID кластера


class AntiFraudEngine:
    """
    AntiFraud v2.

    Алгоритм:
      1. Векторизация отзывов (rubert-tiny2, локально).
      2. Штраф за длину: sigmoid по числу слов.
      3. Штраф за дубликаты: построение кластеров ботов через Union-Find
         по попарной косинусной матрице (>= uniqueness_threshold).
      4. Логарифмическая субаддитивная агрегация внутри кластера:
         weight_i = base_weight_i * ln(2) / ln(1 + cluster_size)
         Одиночный отзыв: ln(2)/ln(2) = 1.0 (без штрафа).
         Кластер из 10 копий: ln(2)/ln(11) ≈ 0.29 — все получают ~0.29 от base.
      5. Клиппинг до [min_trust_weight, 1.0].
    """

    def __init__(self):
        self.model = SentenceTransformer(config.models.encoder_path)
        self.uniqueness_threshold: float = config.fraud.uniqueness_threshold
        self.sim_noise_floor: float = config.fraud.sim_noise_floor
        self.min_trust_weight: float = config.fraud.min_trust_weight
        self.length_k: float = config.fraud.length_sigmoid_k
        self.length_x0: float = config.fraud.length_sigmoid_x0

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def calculate_trust_weights(self, reviews: List[str]) -> List[float]:
        """
        Принимает список текстов. Возвращает список весов [0.0, 1.0].
        Быстрый метод — только веса без деталей.
        """
        results = self.analyze(reviews)
        return [r.trust_weight for r in results]

    def analyze(self, reviews: List[str]) -> List[TrustResult]:
        """
        Полный анализ: возвращает TrustResult с деталями по каждому отзыву.
        """
        if not reviews:
            return []

        embeddings = self.model.encode(reviews, show_progress_bar=False)

        len_weights = self._length_weights(reviews)
        uniq_weights, bot_cluster_ids = self._uniqueness_weights(embeddings)

        results = []
        for i, text in enumerate(reviews):
            final = float(np.clip(
                len_weights[i] * uniq_weights[i],
                self.min_trust_weight,
                1.0,
            ))
            results.append(TrustResult(
                index=i,
                text=text,
                trust_weight=final,
                length_weight=float(len_weights[i]),
                uniqueness_weight=float(uniq_weights[i]),
                bot_cluster_id=int(bot_cluster_ids[i]),
            ))

        return results

    # ------------------------------------------------------------------
    # Компонента 1: длина
    # ------------------------------------------------------------------

    def _length_weights(self, reviews: List[str]) -> np.ndarray:
        """
        Сигмоидный штраф за длину.
        Параметры из конфига:
          k   = 0.8  (крутизна)
          x0  = 4    (точка перегиба, слова)
        Формула: 1 / (1 + exp(-k * (words - x0)))
        """
        weights = np.array([
            1.0 / (1.0 + math.exp(-self.length_k * (len(t.split()) - self.length_x0)))
            for t in reviews
        ])
        return weights

    # ------------------------------------------------------------------
    # Компонента 2: уникальность + кластеры ботов
    # ------------------------------------------------------------------

    def _uniqueness_weights(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Двухуровневый штраф за дубликаты.

        Уровень 1 — базовый:
          Для каждого отзыва берём максимальное косинусное сходство с любым другим.
          Сходство ниже sim_noise_floor (0.5) — фоновый шум, штрафа нет.
          adjusted_sim = max(0, (max_sim - noise_floor) / (1 - noise_floor))
          base_uniq = 1 - adjusted_sim²
          Точная копия (max_sim=1.0) → adjusted=1.0 → base_uniq=0 → клипп до min_trust_weight.
          Нормальный отзыв (max_sim=0.58) → adjusted=0.16 → base_uniq≈0.97.

        Уровень 2 — логарифмическая субаддитивная агрегация (новый):
          Union-Find строит кластеры отзывов с sim >= threshold.
          Для кластера размером S: log_factor = ln(2) / ln(1 + S)
            S=1 (одиночный): factor = 1.0 (нет штрафа)
            S=2 (пара копий): factor ≈ 0.631
            S=5:              factor ≈ 0.431
            S=10:             factor ≈ 0.289
          Это дополнительно срезает вес координированных бот-кампаний
          (много «похожих, но не идентичных» отзывов).

        Итог: uniqueness_weight = clip(base_uniq * log_factor, min_trust_weight, 1.0)
        """
        n = len(embeddings)
        if n < 2:
            return np.ones(n), np.full(n, -1, dtype=int)

        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0.0)

        # --- Уровень 1: базовый штраф с noise floor ---
        max_similarities = np.max(sim_matrix, axis=1)
        noise = self.sim_noise_floor
        adjusted = np.maximum(0.0, (max_similarities - noise) / (1.0 - noise))
        base_uniq = 1.0 - (adjusted ** 2)

        # --- Уровень 2: Union-Find кластеры ---
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.uniqueness_threshold:
                    union(i, j)

        root_to_cluster: dict[int, int] = {}
        cluster_id_counter = 0
        cluster_ids = np.full(n, -1, dtype=int)
        for i in range(n):
            root = find(i)
            if root not in root_to_cluster:
                root_to_cluster[root] = cluster_id_counter
                cluster_id_counter += 1
            cluster_ids[i] = root_to_cluster[root]

        cluster_sizes: dict[int, int] = {}
        for cid in cluster_ids:
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

        # --- Объединяем оба уровня ---
        ln2 = math.log(2)
        uniqueness_weights = np.empty(n)
        for i in range(n):
            s = cluster_sizes[cluster_ids[i]]
            log_factor = ln2 / math.log(1 + s) if s > 1 else 1.0
            combined = float(np.clip(base_uniq[i] * log_factor, self.min_trust_weight, 1.0))
            uniqueness_weights[i] = combined

        # cluster_id = -1 для одиночных (s == 1)
        for i in range(n):
            if cluster_sizes[cluster_ids[i]] == 1:
                cluster_ids[i] = -1

        return uniqueness_weights, cluster_ids


# ------------------------------------------------------------------
# Тест
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    engine = AntiFraudEngine()

    mock_reviews = [
        "Отличный товар, всем рекомендую, быстрая доставка!",        # A — оригинал
        "Норм",                                                        # Б — слишком короткий
        "Отличный товар, всем рекомендую, быстрая доставка!",        # В — полная копия A
        "Товар отличный, рекомендую всем, доставка быстрая.",        # Г — рерайт A
        "Купил, распаковал, пользуюсь. Пока нареканий нет, "
        "но коробка была немного помята. За свои деньги топ.",        # Д — качественный
    ]

    print(f"\nАнализ {len(mock_reviews)} отзывов на фрод...\n")
    results = engine.analyze(mock_reviews)

    for r in results:
        status = "Живой" if r.trust_weight > 0.5 else "Подозрительный"
        cluster_info = f"кластер {r.bot_cluster_id}" if r.bot_cluster_id >= 0 else "одиночный"
        print(f"Review #{r.index + 1}: [{status}] weight={r.trust_weight:.4f} "
              f"(len={r.length_weight:.3f}, uniq={r.uniqueness_weight:.3f}, {cluster_info})")
        print(f"  Text: '{r.text[:80]}'")
        print()

    print("Критерий приёмки:")
    w = [r.trust_weight for r in results]
    print(f"  Бот (В — полная копия):    {w[2]:.4f}  {'OK (< 0.1)' if w[2] < 0.1 else 'FAIL'}")
    print(f"  Живой (Д — качественный): {w[4]:.4f}  {'OK (> 0.7)' if w[4] > 0.7 else 'FAIL'}")
