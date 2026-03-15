from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import hdbscan
import numpy as np
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


@dataclass
class AspectInfo:
    keywords: List[str]
    centroid_embedding: np.ndarray


class AspectClusterer:
    def __init__(self, model: SentenceTransformer | None = None):
        self.model = model or SentenceTransformer(config.models.encoder_path)
        self.min_cluster_size: int = config.discovery.cluster_min_size
        self.anchor_threshold: float = config.discovery.anchor_similarity_threshold

        self._anchor_embeddings: dict[str, np.ndarray] = {}
        self._build_anchor_embeddings()

    def _build_anchor_embeddings(self) -> None:
        for name, words in MACRO_ANCHORS.items():
            embs = self.model.encode(words, show_progress_bar=False)
            self._anchor_embeddings[name] = np.mean(embs, axis=0)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def cluster(
        self, candidates: List[ScoredCandidate], min_mentions: int = 2
    ) -> Dict[str, AspectInfo]:
        if not candidates:
            return {}

        span_data = self._aggregate_spans(candidates, min_mentions)
        if len(span_data) < self.min_cluster_size:
            return self._fallback_single_cluster(span_data)

        spans = list(span_data.keys())
        embeddings = np.stack([span_data[s]["embedding"] for s in spans])

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)

        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(idx)

        if not clusters:
            return self._fallback_single_cluster(span_data)

        result: Dict[str, AspectInfo] = {}
        for label, indices in clusters.items():
            cluster_spans = [spans[i] for i in indices]
            cluster_embs = embeddings[indices]
            centroid = np.mean(cluster_embs, axis=0)

            name = self._name_cluster(centroid, cluster_spans, cluster_embs)

            if name in result:
                existing = result[name]
                merged_kw = existing.keywords + cluster_spans
                merged_embs = np.vstack([
                    existing.centroid_embedding.reshape(1, -1).repeat(len(existing.keywords), axis=0)
                    if len(existing.keywords) > 0 else existing.centroid_embedding.reshape(1, -1),
                    cluster_embs,
                ])
                result[name] = AspectInfo(
                    keywords=merged_kw,
                    centroid_embedding=np.mean(
                        np.vstack([existing.centroid_embedding.reshape(1, -1), centroid.reshape(1, -1)]),
                        axis=0,
                    ).flatten(),
                )
            else:
                result[name] = AspectInfo(
                    keywords=cluster_spans,
                    centroid_embedding=centroid,
                )

        return result

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
