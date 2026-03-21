"""
AspectClusterer v2: Anchor-First Assignment + Residual Discovery.

Architecture (matched filter analogy):
  Pass 1 — Anchor Assignment: каждый span → argmax cos(span, anchor).
           Если cos ≥ τ_anchor И НЕ junk → назначаем на якорь.
           Детерминированный, стабильный, ловит все стандартные аспекты.

  Pass 2 — Residual Discovery: остатки → UMAP → HDBSCAN.
           Находит новые аспекты, которых нет в якорях.

  Pass 3 — Фильтрация: min_mentions + анти-якоря.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import hdbscan
import numpy as np
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from configs.configs import config
from src.discovery.scorer import ScoredCandidate


STOP_SPANS: set[str] = set()

# ── Макро-якоря v2: расширенные описания для точного matching ────────────
# Каждый якорь — 5-10 фраз из реального языка отзывов.
# Больше фраз → плотнее облако в embedding space → точнее assignment.
# "Логистика" и "Упаковка" РАЗДЕЛЕНЫ (раньше были слиты).

MACRO_ANCHORS: Dict[str, List[str]] = {
    "Цена": [
        "цена", "стоимость", "деньги", "дорого", "дешево",
        "за такие деньги", "ценник", "переплата", "соотношение цена",
    ],
    "Качество": [
        "качество", "надежность", "сборка", "материал", "брак",
        "качество материала", "плохое качество", "хорошее качество",
        "качество изготовления", "некачественный",
    ],
    "Внешний вид": [
        "дизайн", "красота", "внешность", "стиль", "цвет",
        "красивый", "выглядит", "смотрится", "внешний вид",
    ],
    "Удобство": [
        "удобство", "комфорт", "эргономика", "удобно пользоваться",
        "удобно держать", "удобный", "неудобный", "под рукой",
        "удобно носить", "комфортно",
    ],
    "Функциональность": [
        "мощность", "скорость", "производительность", "функция",
        "работает", "механизм", "кнопка", "функционал",
    ],
    "Логистика": [
        "доставка", "курьер", "доставили быстро", "пришло быстро",
        "долго шло", "срок доставки", "транспортировка",
    ],
    "Упаковка": [
        "упаковка", "пакет", "коробка", "упаковано",
        "фирменная упаковка", "пришло в пакете", "целая упаковка",
    ],
    "Соответствие": [
        "размер подошёл", "соответствует описанию", "как на фото",
        "не соответствует", "размер", "маломерит",
        "соответствие", "размер в размер",
    ],
    "Органолептика": [
        "вкус", "запах", "аромат", "звук", "пахнет",
        "неприятный запах", "вонь", "вкусный",
    ],
    "Содержание": [
        "содержание", "текст", "шрифт", "сюжет", "интересно читать",
        "информация", "контент",
    ],
    "Состав": [
        "состав", "ингредиенты", "натуральный состав",
        "хороший состав", "без добавок",
    ],
    "Здоровье": [
        "здоровье", "аллергия", "реакция", "самочувствие",
        "шерсть", "стул", "переносимость",
    ],
    "Поедаемость": [
        "ест с удовольствием", "нравится коту", "не ест",
        "привередливый", "вкусный корм", "поедаемость",
    ],
    "Свежесть": [
        "срок годности", "свежий", "свежесть",
        "не просроченный", "дата изготовления",
    ],
    "Комфорт": [
        "удобно носить", "комфортно", "приятно к телу",
        "неудобно", "натирает", "мягкий материал",
        "приятная ткань", "ткань", "материал приятный",
    ],
    "Продавец": [
        "продавец", "магазин", "обслуживание",
        "ответ продавца", "отношение продавца",
    ],
    "Состояние": [
        "пришло в плохом состоянии", "брак", "дефект",
        "нитки торчат", "швы порваны", "повреждённый",
        "возврат", "бракованный",
    ],
    "Вместимость": [
        "вместимость", "помещается", "влезает",
        "мало места", "ёмкость",
    ],
}

ANTI_ANCHORS: Dict[str, List[str]] = {
    "_emotion": [
        "радость", "восторг", "счастье", "доволен", "мечта", "кайф",
        "разочарование", "ужас", "класс", "топ", "огонь", "пушка",
        "бомба", "рад", "довольна", "восторге", "круто", "супер",
        "помощник", "напарник", "здоровье", "жизнь", "рабочий",
    ],
    "_buyer": [
        "ребёнок", "дочь", "сын", "жена", "муж", "друг", "подарок",
        "новичок", "подросток", "брат", "внук", "сестра", "племянник",
        "дети", "ребенка", "дочери", "сыну", "мужу", "другу", "внуку",
    ],
    "_time": [
        "год", "месяц", "день", "неделя", "раз", "первый", "второй",
        "начало", "срок", "уже", "пока", "время",
    ],
    "_meta": [
        "товар", "покупка", "заказ", "вещь", "штука", "вариант",
        "покупкой", "заказом", "вещи",
        "описание", "фото", "видео", "карточка", "отзыв",
    ],
    "_overall": [
        "гитара", "инструмент", "телефон", "модель", "устройство",
        "аппарат", "продукт", "изделие",
        "штучка", "игрушка",
    ],
}


@dataclass
class AspectInfo:
    keywords: List[str]
    centroid_embedding: np.ndarray


class AspectClusterer:
    def __init__(self, model: SentenceTransformer | None = None):
        self.model = model or SentenceTransformer(config.models.encoder_path)
        self.min_samples: int = config.discovery.hdbscan_min_samples
        self.anchor_threshold: float = config.discovery.anchor_similarity_threshold
        self.anti_anchor_threshold: float = config.discovery.anti_anchor_threshold
        self.merge_threshold: float = config.discovery.cluster_merge_threshold

        self.umap_n_components: int = config.discovery.umap_n_components
        self.umap_min_dist: float = config.discovery.umap_min_dist
        self.umap_metric: str = config.discovery.umap_metric

        # Порог для Pass 1 (anchor assignment).
        # Читаем из конфига если есть, иначе дефолт 0.55
        self.anchor_assign_threshold: float = getattr(
            config.discovery, "anchor_assign_threshold", 0.55
        )
        # FIX2: Минимальный margin между top-1 и top-2 anchor.
        # Если margin < δ — спан "confused", уходит в residual.
        self.anchor_assign_margin: float = getattr(
            config.discovery, "anchor_assign_margin", 0.03
        )

        self._anchor_embeddings: dict[str, np.ndarray] = {}
        self._anti_anchor_embeddings: dict[str, np.ndarray] = {}
        self._build_anchor_embeddings()
        self._build_anti_anchor_embeddings()

    def _build_anchor_embeddings(self) -> None:
        for name, words in MACRO_ANCHORS.items():
            embs = self.model.encode(words, show_progress_bar=False)
            self._anchor_embeddings[name] = np.mean(embs, axis=0)

    def _build_anti_anchor_embeddings(self) -> None:
        for name, words in ANTI_ANCHORS.items():
            embs = self.model.encode(words, show_progress_bar=False)
            self._anti_anchor_embeddings[name] = np.mean(embs, axis=0)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def cluster(
        self, candidates: List[ScoredCandidate], min_mentions: int = 2
    ) -> Dict[str, AspectInfo]:
        """
        Anchor-First Assignment + Residual Discovery.

        v3 fixes:
          FIX1: Auto product-name stop-spans (top-frequency single words → filter)
          FIX2: Margin-based assignment (margin < δ → residual)
          FIX3: Domain-aware post-filter (anchors with few mentions → residual)

        Pass 0: Auto-detect product name stop-spans.
        Pass 1: Anchor assignment with margin check.
        Pass 2: Residual → UMAP → HDBSCAN.
        Pass 3: Domain-aware filtering + min_mentions.
        """
        if not candidates:
            return {}

        # ── Pass 0: Auto stop-spans ──────────────────────────────────
        # Самые частые single-word спаны (count > 5% всех кандидатов)
        # — скорее всего название товара. Фильтруем ДО агрегации.
        auto_stops = self._detect_product_stops(candidates)
        local_stops = STOP_SPANS | auto_stops  # не мутируем глобальный set

        span_data = self._aggregate_spans(candidates, min_mentions=1, stop_spans=local_stops)
        if not span_data:
            return {}

        spans = list(span_data.keys())
        embeddings = np.stack([span_data[s]["embedding"] for s in spans])
        counts = np.array([span_data[s]["count"] for s in spans])

        # ── Pass 1: Anchor Assignment with margin ────────────────────
        anchor_names = list(self._anchor_embeddings.keys())
        anchor_matrix = np.stack([self._anchor_embeddings[n] for n in anchor_names])

        # Cosine: |spans| × |anchors|
        sim_matrix = cosine_similarity(embeddings, anchor_matrix)

        assigned: Dict[str, List[int]] = {n: [] for n in anchor_names}
        residual_indices: List[int] = []

        for i in range(len(spans)):
            sorted_idx = np.argsort(sim_matrix[i])[::-1]
            best_j = int(sorted_idx[0])
            best_sim = float(sim_matrix[i, best_j])
            second_sim = float(sim_matrix[i, sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
            margin = best_sim - second_sim

            # FIX2: margin check — confused spans go to residual
            if best_sim >= self.anchor_assign_threshold and margin >= self.anchor_assign_margin:
                if self._is_junk_span(embeddings[i]):
                    residual_indices.append(i)
                else:
                    assigned[anchor_names[best_j]].append(i)
            else:
                residual_indices.append(i)

        # Собираем anchor-кластеры
        result: Dict[str, AspectInfo] = {}
        for anchor_name, indices in assigned.items():
            if not indices:
                continue
            cluster_spans = [spans[i] for i in indices]
            cluster_embs = embeddings[indices]
            total_mentions = int(counts[indices].sum())

            if total_mentions < min_mentions:
                residual_indices.extend(indices)
                continue

            centroid = np.mean(cluster_embs, axis=0)
            result[anchor_name] = AspectInfo(
                keywords=cluster_spans,
                centroid_embedding=centroid,
            )

        # ── Pass 2: Residual Discovery (HDBSCAN) ────────────────────
        if len(residual_indices) >= 5:
            res_spans = [spans[i] for i in residual_indices]
            res_embs = embeddings[residual_indices]
            res_counts = counts[residual_indices]

            residual_aspects = self._cluster_residuals(
                res_spans, res_embs, res_counts, min_mentions
            )

            for name, info in residual_aspects.items():
                if name in result:
                    existing = result[name]
                    result[name] = AspectInfo(
                        keywords=existing.keywords + info.keywords,
                        centroid_embedding=np.mean(
                            np.vstack([
                                existing.centroid_embedding.reshape(1, -1),
                                info.centroid_embedding.reshape(1, -1),
                            ]),
                            axis=0,
                        ).flatten(),
                    )
                else:
                    result[name] = info

        # ── Pass 3: Domain-aware filtering ───────────────────────────
        filtered: Dict[str, AspectInfo] = {}
        for name, info in result.items():
            total = sum(
                span_data[s]["count"] for s in info.keywords if s in span_data
            )
            if total >= min_mentions:
                filtered[name] = info

        return filtered

    # ------------------------------------------------------------------
    # FIX1: Auto-detect product name stop-spans
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_product_stops(
        candidates: List[ScoredCandidate],
        freq_threshold: float = 0.05,
    ) -> set:
        """Находит спаны-названия товара: single words с частотой > threshold.

        Логика: если одно слово встречается в >5% всех кандидатов —
        это скорее всего название товара ("книга", "портсигар", "корм"),
        а не аспект. Такие спаны засоряют anchor assignment.
        """
        if not candidates:
            return set()

        total = len(candidates)
        word_counts: dict[str, int] = {}
        for c in candidates:
            # Только single-word спаны (не биграммы)
            if " " not in c.span.strip():
                w = c.span.strip().lower()
                word_counts[w] = word_counts.get(w, 0) + 1

        stops = set()
        for word, count in word_counts.items():
            if count / total >= freq_threshold and len(word) >= 3:
                stops.add(word)
                # Добавляем и оригинальный регистр из кандидатов
                for c in candidates:
                    if c.span.strip().lower() == word:
                        stops.add(c.span.strip())

        return stops

    # ------------------------------------------------------------------
    # Pass 2: HDBSCAN на остатках (residuals)
    # ------------------------------------------------------------------
    def _cluster_residuals(
        self,
        spans: List[str],
        embeddings: np.ndarray,
        counts: np.ndarray,
        min_mentions: int,
    ) -> Dict[str, AspectInfo]:
        """UMAP → HDBSCAN на residual spans. Именование через якоря."""
        N = len(spans)
        min_cluster_size = max(3, min(8, N // 50))
        if N < min_cluster_size:
            return {}

        n_neighbors = max(2, min(15, N // 10))

        reducer = umap.UMAP(
            n_components=min(self.umap_n_components, N - 2),
            n_neighbors=min(n_neighbors, N - 1),
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            n_jobs=1,
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced)

        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(idx)

        if not clusters:
            return {}

        result: Dict[str, AspectInfo] = {}
        umap_centroids: Dict[str, np.ndarray] = {}

        for label, indices in clusters.items():
            cluster_spans = [spans[i] for i in indices]
            cluster_embs = embeddings[indices]
            centroid = np.mean(cluster_embs, axis=0)

            umap_embs = reduced[indices]
            umap_centroid = np.mean(umap_embs, axis=0)

            # Фильтрация мусорных кластеров
            if self._is_junk_cluster(centroid):
                continue

            total_mentions = int(counts[indices].sum())
            if total_mentions < min_mentions:
                continue

            name = self._name_cluster(centroid, cluster_spans, cluster_embs)

            if name in result:
                existing = result[name]
                result[name] = AspectInfo(
                    keywords=existing.keywords + cluster_spans,
                    centroid_embedding=np.mean(
                        np.vstack([
                            existing.centroid_embedding.reshape(1, -1),
                            centroid.reshape(1, -1),
                        ]),
                        axis=0,
                    ).flatten(),
                )
                umap_centroids[name] = np.mean(
                    np.vstack([
                        umap_centroids[name].reshape(1, -1),
                        umap_centroid.reshape(1, -1),
                    ]),
                    axis=0,
                ).flatten()
            else:
                result[name] = AspectInfo(
                    keywords=cluster_spans,
                    centroid_embedding=centroid,
                )
                umap_centroids[name] = umap_centroid

        # Пост-кластерный мерж
        result = self._merge_similar_clusters(result, umap_centroids)
        return result

    # ------------------------------------------------------------------
    # Junk detection
    # ------------------------------------------------------------------
    def _is_junk_span(self, embedding: np.ndarray) -> bool:
        """Проверяет один span: ближе к анти-якорю чем к любому якорю?"""
        emb = embedding.reshape(1, -1)

        max_anti_sim = max(
            cosine_similarity(emb, anti_emb.reshape(1, -1))[0, 0]
            for anti_emb in self._anti_anchor_embeddings.values()
        )
        max_anchor_sim = max(
            cosine_similarity(emb, anchor_emb.reshape(1, -1))[0, 0]
            for anchor_emb in self._anchor_embeddings.values()
        )

        return max_anti_sim > max_anchor_sim + self.anti_anchor_threshold

    def _is_junk_cluster(self, centroid: np.ndarray) -> bool:
        """Проверяет центроид кластера."""
        centroid_norm = centroid.reshape(1, -1)

        max_anti_sim = max(
            cosine_similarity(centroid_norm, anti_emb.reshape(1, -1))[0, 0]
            for anti_emb in self._anti_anchor_embeddings.values()
        )
        max_anchor_sim = max(
            cosine_similarity(centroid_norm, anchor_emb.reshape(1, -1))[0, 0]
            for anchor_emb in self._anchor_embeddings.values()
        )

        return max_anti_sim > max_anchor_sim + self.anti_anchor_threshold

    # ------------------------------------------------------------------
    # Агрегация спанов
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_spans(
        candidates: List[ScoredCandidate], min_mentions: int,
        stop_spans: set | None = None,
    ) -> dict:
        stops = stop_spans or STOP_SPANS
        agg: dict[str, dict] = {}
        for c in candidates:
            if c.span in stops:
                continue
            if c.span not in agg:
                agg[c.span] = {"count": 0, "embedding": c.embedding, "score_sum": 0.0}
            agg[c.span]["count"] += 1
            agg[c.span]["score_sum"] += c.score

        if min_mentions > 1:
            agg = {s: d for s, d in agg.items() if d["count"] >= min_mentions}
        return agg

    # ------------------------------------------------------------------
    # Пост-кластерный мерж близких кластеров (евклид в UMAP R5)
    # ------------------------------------------------------------------
    def _merge_similar_clusters(
        self, clusters: Dict[str, AspectInfo], umap_centroids: Dict[str, np.ndarray]
    ) -> Dict[str, AspectInfo]:
        if len(clusters) < 2:
            return clusters

        names = list(clusters.keys())

        while True:
            best_pair = None
            best_dist = float("inf")

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if names[i] not in umap_centroids or names[j] not in umap_centroids:
                        continue
                    dist = float(np.linalg.norm(
                        umap_centroids[names[i]] - umap_centroids[names[j]]
                    ))
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (names[i], names[j])

            if best_dist > self.merge_threshold or best_pair is None:
                break

            name_i, name_j = best_pair

            if len(clusters[name_i].keywords) >= len(clusters[name_j].keywords):
                keep_name, drop_name = name_i, name_j
            else:
                keep_name, drop_name = name_j, name_i

            keep = clusters[keep_name]
            drop = clusters[drop_name]

            clusters[keep_name] = AspectInfo(
                keywords=keep.keywords + drop.keywords,
                centroid_embedding=keep.centroid_embedding,
            )
            if keep_name in umap_centroids and drop_name in umap_centroids:
                umap_centroids[keep_name] = np.mean(
                    np.vstack([
                        umap_centroids[keep_name].reshape(1, -1),
                        umap_centroids[drop_name].reshape(1, -1),
                    ]),
                    axis=0,
                ).flatten()

            del clusters[drop_name]
            if drop_name in umap_centroids:
                del umap_centroids[drop_name]
            names = list(clusters.keys())

        return clusters

    # ------------------------------------------------------------------
    # Именование кластера (для residual discovery)
    # ------------------------------------------------------------------
    def _name_cluster(
        self,
        centroid: np.ndarray,
        spans: List[str],
        embeddings: np.ndarray,
    ) -> str:
        centroid_norm = centroid.reshape(1, -1)

        anchor_sims: list[tuple[str, float]] = []
        for name, anchor_emb in self._anchor_embeddings.items():
            sim = cosine_similarity(centroid_norm, anchor_emb.reshape(1, -1))[0, 0]
            anchor_sims.append((name, float(sim)))

        anchor_sims.sort(key=lambda x: x[1], reverse=True)
        best_name, best_sim = anchor_sims[0]
        second_sim = anchor_sims[1][1] if len(anchor_sims) > 1 else 0.0
        margin = best_sim - second_sim

        if margin >= self.anchor_threshold:
            return best_name

        sims = cosine_similarity(centroid_norm, embeddings)[0]
        medoid_idx = int(np.argmax(sims))
        return spans[medoid_idx]

    # ------------------------------------------------------------------
    # Fallback: если слишком мало данных для HDBSCAN
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_single_cluster(span_data: dict) -> Dict[str, AspectInfo]:
        if not span_data:
            return {}
        spans = list(span_data.keys())
        embs = np.stack([span_data[s]["embedding"] for s in spans])
        centroid = np.mean(embs, axis=0)
        return {
            "Общее": AspectInfo(keywords=spans, centroid_embedding=centroid)
        }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from src.discovery.candidates import CandidateExtractor
    from src.discovery.scorer import KeyBERTScorer

    extractor = CandidateExtractor()
    scorer = KeyBERTScorer()

    reviews = [
        "Доставка быстрая, курьер вежливый. Упаковка хорошая.",
        "Пришло быстро, упаковка целая, курьер позвонил заранее.",
        "Доставили за два дня, упаковано отлично.",
        "Качество материала отличное, сборка на уровне.",
        "Материал приятный, качество сборки радует.",
        "Сборка крепкая, материал хороший, брака нет.",
        "Экран яркий, цвета насыщенные.",
        "Дисплей отличный, матрица хорошая.",
        "Экран шикарный, разрешение высокое.",
        "Цена адекватная, за такие деньги — отлично.",
        "Стоимость нормальная, не дорого.",
        "За такую цену — отличное качество.",
    ]

    all_scored: list[ScoredCandidate] = []
    for text in reviews:
        cands = extractor.extract(text)
        scored = scorer.score_and_select(cands)
        all_scored.extend(scored)

    print(f"Всего scored-кандидатов: {len(all_scored)}\n")

    clusterer = AspectClusterer(model=scorer.model)
    aspects = clusterer.cluster(all_scored, min_mentions=1)

    for name, info in aspects.items():
        print(f"[{name}]")
        print(f"  keywords: {info.keywords}")
        print()