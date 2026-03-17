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

MACRO_ANCHORS: Dict[str, List[str]] = {
    "Цена": ["цена", "стоимость", "деньги", "дорого", "дешево"],
    "Качество": ["качество", "надежность", "сборка", "материал", "брак"],
    "Внешний вид": ["дизайн", "красота", "внешность", "стиль", "цвет"],
    "Удобство": ["удобство", "комфорт", "эргономика", "размер"],
    "Функциональность": ["мощность", "скорость", "производительность", "функция"],
    "Логистика": ["доставка", "упаковка", "курьер", "срок"],
    "Органолептика": ["вкус", "запах", "аромат", "звук"],
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
        if not candidates:
            return {}

        span_data = self._aggregate_spans(candidates, min_mentions)
        N = len(span_data)

        min_cluster_size = max(3, min(8, N // 300))
        if N < min_cluster_size:
            return self._fallback_single_cluster(span_data)

        adaptive_n_neighbors = max(5, min(25, N // 100))

        spans = list(span_data.keys())
        embeddings = np.stack([span_data[s]["embedding"] for s in spans])

        n_neighbors = min(adaptive_n_neighbors, N - 1)
        reducer = umap.UMAP(
            n_components=min(self.umap_n_components, N - 2),
            n_neighbors=max(2, n_neighbors),
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
            return self._fallback_single_cluster(span_data)

        result: Dict[str, AspectInfo] = {}
        umap_centroids: Dict[str, np.ndarray] = {}
        
        # Первый проход: мержим HDBSCAN-кластеры с одинаковыми именами
        for label, indices in clusters.items():
            cluster_spans = [spans[i] for i in indices]
            cluster_embs = embeddings[indices]
            centroid = np.mean(cluster_embs, axis=0)

            umap_embs = reduced[indices]
            umap_centroid = np.mean(umap_embs, axis=0)

            name = self._name_cluster(centroid, cluster_spans, cluster_embs)

            if name in result:
                existing = result[name]
                merged_kw = existing.keywords + cluster_spans
                result[name] = AspectInfo(
                    keywords=merged_kw,
                    centroid_embedding=np.mean(
                        np.vstack([existing.centroid_embedding.reshape(1, -1), centroid.reshape(1, -1)]),
                        axis=0,
                    ).flatten(),
                )
                umap_centroids[name] = np.mean(
                    np.vstack([umap_centroids[name].reshape(1, -1), umap_centroid.reshape(1, -1)]),
                    axis=0,
                ).flatten()
            else:
                result[name] = AspectInfo(
                    keywords=cluster_spans,
                    centroid_embedding=centroid,
                )
                umap_centroids[name] = umap_centroid

        # Второй проход: фильтруем мусорные кластеры через анти-якоря
        filtered: Dict[str, AspectInfo] = {}
        filtered_umap: Dict[str, np.ndarray] = {}
        for name, info in result.items():
            if not self._is_junk_cluster(info.centroid_embedding):
                filtered[name] = info
                filtered_umap[name] = umap_centroids[name]

        # Третий проход: пост-кластерный мерж близких кластеров
        filtered = self._merge_similar_clusters(filtered, filtered_umap)
        return filtered

    # ------------------------------------------------------------------
    # Агрегация спанов
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_spans(
        candidates: List[ScoredCandidate], min_mentions: int
    ) -> dict:
        agg: dict[str, dict] = {}
        for c in candidates:
            if c.span in STOP_SPANS:
                continue
            if c.span not in agg:
                agg[c.span] = {"count": 0, "embedding": c.embedding, "score_sum": 0.0}
            agg[c.span]["count"] += 1
            agg[c.span]["score_sum"] += c.score

        if min_mentions > 1:
            agg = {s: d for s, d in agg.items() if d["count"] >= min_mentions}
        return agg

    # ------------------------------------------------------------------
    # Фильтрация мусорных кластеров через анти-якоря
    # ------------------------------------------------------------------
    def _is_junk_cluster(self, centroid: np.ndarray) -> bool:
        centroid_norm = centroid.reshape(1, -1)
        
        # Max sim к анти-якорям
        max_anti_sim = max(
            cosine_similarity(centroid_norm, anti_emb.reshape(1, -1))[0, 0]
            for anti_emb in self._anti_anchor_embeddings.values()
        )
        
        # Max sim к макро-якорям
        max_anchor_sim = max(
            cosine_similarity(centroid_norm, anchor_emb.reshape(1, -1))[0, 0]
            for anchor_emb in self._anchor_embeddings.values()
        )
        
        # Отбросить, если анти-якорь ближе чем макро-якорь + margin
        return max_anti_sim > max_anchor_sim + self.anti_anchor_threshold

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
            umap_centroids[keep_name] = np.mean(
                np.vstack([
                    umap_centroids[keep_name].reshape(1, -1),
                    umap_centroids[drop_name].reshape(1, -1),
                ]),
                axis=0,
            ).flatten()

            del clusters[drop_name]
            del umap_centroids[drop_name]
            names = list(clusters.keys())

        return clusters

    # ------------------------------------------------------------------
    # Именование кластера
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
