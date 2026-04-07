"""
AspectClusterer: Anchor-First + Residual HDBSCAN (baseline).

  Pass 1 — Anchor assignment: cos к якорям, margin; junk → residual.
  Pass 2 — Residual → UMAP → HDBSCAN → _name_cluster (margin из config).
  UMAP-merge близких residual-кластеров (cluster_merge_threshold).
  Pass 3 — min_mentions filter.
  nli_label: для NLI гипотезы; medoid-кластеры → argmax cos(centroid, anchors).
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import hdbscan
import numpy as np
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from configs.configs import config
from src.schemas.models import AspectInfo, ScoredCandidate
from src.stages.contracts import ClusteringStage


STOP_SPANS: set[str] = set()

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


class AspectClusterer(ClusteringStage):
    def __init__(self, model: SentenceTransformer | None = None):
        self.model = model or SentenceTransformer(config.models.encoder_path)
        self.min_samples: int = config.discovery.hdbscan_min_samples
        self.anchor_threshold: float = float(config.discovery.anchor_similarity_threshold)
        self.anti_anchor_threshold: float = config.discovery.anti_anchor_threshold
        self.merge_threshold: float = config.discovery.cluster_merge_threshold

        self.anchor_assign_threshold: float = float(
            getattr(config.discovery, "anchor_assign_threshold", 0.55)
        )
        self.anchor_assign_margin: float = float(
            getattr(config.discovery, "anchor_assign_margin", 0.03)
        )

        self.umap_n_components: int = config.discovery.umap_n_components
        self.umap_min_dist: float = config.discovery.umap_min_dist
        self.umap_metric: str = config.discovery.umap_metric

        self._anchor_embeddings: dict[str, np.ndarray] = {}
        self._anti_anchor_embeddings: dict[str, np.ndarray] = {}
        self._load_or_build_anchor_embeddings()

        self.last_assignment_counts: Dict[str, int] = {}
        self.last_residual_medoid_names: List[str] = []
        self.last_nli_medoid_diagnostics: List[str] = []

    def _build_anchor_embeddings(self) -> None:
        for name, words in MACRO_ANCHORS.items():
            embs = self.model.encode(words, show_progress_bar=False)
            self._anchor_embeddings[name] = np.mean(embs, axis=0)

    def _build_anti_anchor_embeddings(self) -> None:
        for name, words in ANTI_ANCHORS.items():
            embs = self.model.encode(words, show_progress_bar=False)
            self._anti_anchor_embeddings[name] = np.mean(embs, axis=0)

    def _anchors_signature(self) -> str:
        payload = {
            "model": str(config.models.encoder_path),
            "macro": MACRO_ANCHORS,
            "anti": ANTI_ANCHORS,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _anchors_cache_path(self) -> Path:
        sig = self._anchors_signature()
        cache_dir = Path("data") / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"anchor_embeddings_{sig}.npz"

    def _load_or_build_anchor_embeddings(self) -> None:
        cache_path = self._anchors_cache_path()
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            self._anchor_embeddings = {
                name: cached[f"macro__{name}"] for name in MACRO_ANCHORS
            }
            self._anti_anchor_embeddings = {
                name: cached[f"anti__{name}"] for name in ANTI_ANCHORS
            }
            return

        self._build_anchor_embeddings()
        self._build_anti_anchor_embeddings()

        payload: dict[str, np.ndarray] = {}
        payload.update({f"macro__{k}": v for k, v in self._anchor_embeddings.items()})
        payload.update({f"anti__{k}": v for k, v in self._anti_anchor_embeddings.items()})
        np.savez_compressed(cache_path, **payload)

    def cluster(
        self, candidates: List[ScoredCandidate], min_mentions: int = 2
    ) -> Dict[str, AspectInfo]:
        self.last_assignment_counts = {}
        self.last_residual_medoid_names = []
        self.last_nli_medoid_diagnostics = []

        if not candidates:
            return {}

        auto_stops = self._detect_product_stops(candidates)
        local_stops = STOP_SPANS | auto_stops

        span_data = self._aggregate_spans(candidates, min_mentions=1, stop_spans=local_stops)
        if not span_data:
            return {}

        aspect_medoid: Dict[str, bool] = {}

        spans = list(span_data.keys())
        embeddings = np.stack([span_data[s]["embedding"] for s in spans])
        counts = np.array([span_data[s]["count"] for s in spans])

        anchor_names = list(self._anchor_embeddings.keys())
        anchor_matrix = np.stack([self._anchor_embeddings[n] for n in anchor_names])

        sim_matrix = cosine_similarity(embeddings, anchor_matrix)

        assigned: Dict[str, List[int]] = {n: [] for n in anchor_names}
        residual_indices: List[int] = []

        for i in range(len(spans)):
            sorted_idx = np.argsort(sim_matrix[i])[::-1]
            best_j = int(sorted_idx[0])
            best_sim = float(sim_matrix[i, best_j])
            second_sim = float(sim_matrix[i, sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
            margin = best_sim - second_sim

            if self._is_junk_span(embeddings[i]):
                residual_indices.append(i)
                continue

            if (
                best_sim >= self.anchor_assign_threshold
                and margin >= self.anchor_assign_margin
            ):
                assigned[anchor_names[best_j]].append(i)
            else:
                residual_indices.append(i)

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
                keyword_weights=[1.0] * len(cluster_spans),
                nli_label="",
            )

        for _nm in result:
            aspect_medoid[_nm] = False

        n_confident = sum(len(info.keywords) for info in result.values())
        n_residual_pass1 = len(residual_indices)

        if len(residual_indices) >= 5:
            res_spans = [spans[i] for i in residual_indices]
            res_embs = embeddings[residual_indices]
            res_counts = counts[residual_indices]

            residual_aspects, residual_medoid = self._cluster_residuals(
                res_spans, res_embs, res_counts, min_mentions
            )

            for name, info in residual_aspects.items():
                m = residual_medoid.get(name, False)
                if name in result:
                    existing = result[name]
                    kw_w = existing.keyword_weights or [1.0] * len(existing.keywords)
                    add_w = info.keyword_weights or [1.0] * len(info.keywords)
                    result[name] = AspectInfo(
                        keywords=existing.keywords + info.keywords,
                        centroid_embedding=np.mean(
                            np.vstack([
                                existing.centroid_embedding.reshape(1, -1),
                                info.centroid_embedding.reshape(1, -1),
                            ]),
                            axis=0,
                        ).flatten(),
                        keyword_weights=kw_w + add_w,
                        nli_label="",
                    )
                    aspect_medoid[name] = aspect_medoid.get(name, False) or m
                else:
                    result[name] = info
                    aspect_medoid[name] = m

        filtered: Dict[str, AspectInfo] = {}
        for name, info in result.items():
            total = sum(
                span_data[s]["count"] for s in info.keywords if s in span_data
            )
            if total >= min_mentions:
                filtered[name] = info

        self._apply_nli_labels(
            filtered, aspect_medoid, anchor_names, anchor_matrix
        )

        self.last_assignment_counts = {
            "confident": n_confident,
            "residual": n_residual_pass1,
        }
        self.last_residual_medoid_names = sorted(
            k for k in filtered if aspect_medoid.get(k, False)
        )

        nli_txt = (
            "  nli medoid routing:\n    "
            + "\n    ".join(self.last_nli_medoid_diagnostics)
            if self.last_nli_medoid_diagnostics
            else ""
        )
        print(
            f"[AspectClusterer] pass1 confident_spans={n_confident}, "
            f"residual_spans={n_residual_pass1}\n"
            f"{nli_txt}"
            f"  residual medoid names (final): {self.last_residual_medoid_names}"
        )

        return filtered

    def _apply_nli_labels(
        self,
        filtered: Dict[str, AspectInfo],
        aspect_medoid: Dict[str, bool],
        anchor_names: List[str],
        anchor_matrix: np.ndarray,
    ) -> None:
        for name, info in filtered.items():
            if not aspect_medoid.get(name, False):
                info.nli_label = name
                continue

            c = info.centroid_embedding.reshape(1, -1)
            sims = cosine_similarity(c, anchor_matrix)[0]
            bi = int(np.argmax(sims))
            cos_v = float(sims[bi])
            nli = anchor_names[bi]
            info.nli_label = nli
            self.last_nli_medoid_diagnostics.append(
                f"medoid '{name}' → nli_label '{nli}' (cos={cos_v:.2f})"
            )

    @staticmethod
    def _detect_product_stops(
        candidates: List[ScoredCandidate],
        freq_threshold: float = 0.05,
    ) -> set:
        if not candidates:
            return set()

        total = len(candidates)
        word_counts: dict[str, int] = {}
        for c in candidates:
            if " " not in c.span.strip():
                w = c.span.strip().lower()
                word_counts[w] = word_counts.get(w, 0) + 1

        stops = set()
        for word, count in word_counts.items():
            if count / total >= freq_threshold and len(word) >= 3:
                stops.add(word)
                for c in candidates:
                    if c.span.strip().lower() == word:
                        stops.add(c.span.strip())

        return stops

    def _cluster_residuals(
        self,
        spans: List[str],
        embeddings: np.ndarray,
        counts: np.ndarray,
        min_mentions: int,
    ) -> Tuple[Dict[str, AspectInfo], Dict[str, bool]]:
        N = len(spans)
        min_cluster_size = max(3, min(8, N // 50))
        if N < min_cluster_size:
            return {}, {}

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
            return {}, {}

        result: Dict[str, AspectInfo] = {}
        umap_centroids: Dict[str, np.ndarray] = {}
        aspect_medoid: Dict[str, bool] = {}

        for _label, indices in clusters.items():
            cluster_spans = [spans[i] for i in indices]
            cluster_embs = embeddings[indices]
            centroid = np.mean(cluster_embs, axis=0)

            umap_embs = reduced[indices]
            umap_centroid = np.mean(umap_embs, axis=0)

            if self._is_junk_cluster(centroid):
                continue

            total_mentions = int(counts[indices].sum())
            if total_mentions < min_mentions:
                continue

            name, is_medoid = self._name_cluster(centroid, cluster_spans, cluster_embs)

            if name in result:
                existing = result[name]
                ex_kw = existing.keyword_weights or [1.0] * len(existing.keywords)
                result[name] = AspectInfo(
                    keywords=existing.keywords + cluster_spans,
                    centroid_embedding=np.mean(
                        np.vstack([
                            existing.centroid_embedding.reshape(1, -1),
                            centroid.reshape(1, -1),
                        ]),
                        axis=0,
                    ).flatten(),
                    keyword_weights=ex_kw + [1.0] * len(cluster_spans),
                    nli_label="",
                )
                umap_centroids[name] = np.mean(
                    np.vstack([
                        umap_centroids[name].reshape(1, -1),
                        umap_centroid.reshape(1, -1),
                    ]),
                    axis=0,
                ).flatten()
                aspect_medoid[name] = aspect_medoid.get(name, False) or is_medoid
            else:
                result[name] = AspectInfo(
                    keywords=cluster_spans,
                    centroid_embedding=centroid,
                    keyword_weights=[1.0] * len(cluster_spans),
                    nli_label="",
                )
                umap_centroids[name] = umap_centroid
                aspect_medoid[name] = is_medoid

        result = self._merge_similar_clusters(result, umap_centroids, aspect_medoid)
        return result, aspect_medoid

    def _merge_similar_clusters(
        self,
        clusters: Dict[str, AspectInfo],
        umap_centroids: Dict[str, np.ndarray],
        aspect_medoid: Optional[Dict[str, bool]] = None,
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

            kw_w_k = keep.keyword_weights or [1.0] * len(keep.keywords)
            kw_w_d = drop.keyword_weights or [1.0] * len(drop.keywords)
            wk = float(len(keep.keywords))
            wd = float(len(drop.keywords))
            wsum = wk + wd
            merged_c = (
                (wk * keep.centroid_embedding + wd * drop.centroid_embedding) / wsum
                if wsum > 0
                else keep.centroid_embedding
            )
            clusters[keep_name] = AspectInfo(
                keywords=keep.keywords + drop.keywords,
                centroid_embedding=np.asarray(merged_c).flatten(),
                keyword_weights=kw_w_k + kw_w_d,
                nli_label="",
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
            if aspect_medoid is not None:
                aspect_medoid[keep_name] = aspect_medoid.get(keep_name, False) or aspect_medoid.get(
                    drop_name, False
                )
                if drop_name in aspect_medoid:
                    del aspect_medoid[drop_name]
            names = list(clusters.keys())

        return clusters

    def _name_cluster(
        self,
        centroid: np.ndarray,
        spans: List[str],
        embeddings: np.ndarray,
    ) -> Tuple[str, bool]:
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
            return best_name, False

        sims = cosine_similarity(centroid_norm, embeddings)[0]
        medoid_idx = int(np.argmax(sims))
        return spans[medoid_idx], True

    def _is_junk_span(self, embedding: np.ndarray) -> bool:
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

    @staticmethod
    def _fallback_single_cluster(span_data: dict) -> Dict[str, AspectInfo]:
        if not span_data:
            return {}
        spans = list(span_data.keys())
        embs = np.stack([span_data[s]["embedding"] for s in spans])
        centroid = np.mean(embs, axis=0)
        return {
            "Общее": AspectInfo(
                keywords=spans,
                centroid_embedding=centroid,
                keyword_weights=[1.0] * len(spans),
                nli_label="Общее",
            )
        }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from src.stages.extraction import CandidateExtractor
    from src.stages.scoring import KeyBERTScorer

    extractor = CandidateExtractor()
    scorer = KeyBERTScorer()

    reviews = [
        "Доставка быстрая, курьер вежливый. Упаковка хорошая.",
        "Пришло быстро, упаковка целая, курьер позвонил заранее.",
    ]

    all_scored: list[ScoredCandidate] = []
    for text in reviews:
        cands = extractor.extract(text)
        scored = scorer.score_and_select(cands)
        all_scored.extend(scored)

    clusterer = AspectClusterer(model=scorer.model)
    aspects = clusterer.cluster(all_scored, min_mentions=1)

    for name, info in aspects.items():
        print(f"[{name}] nli={info.nli_label!r}")
        print(f"  keywords: {info.keywords[:5]}")
