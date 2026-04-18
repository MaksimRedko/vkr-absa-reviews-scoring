"""
AspectClusterer: Anchor-First + Residual HDBSCAN (baseline).

  Pass 1 — Anchor assignment: cos к якорям, margin; junk → residual.
  Pass 2 — Residual → UMAP → HDBSCAN → _name_cluster (margin из config).
  UMAP-merge близких residual-кластеров (cluster_merge_threshold).
  Pass 3 — min_mentions filter.
  nli_label: для NLI гипотезы; medoid-кластеры → argmax cos(centroid, anchors).
"""

from __future__ import annotations

import heapq
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import umap
from scipy.stats import normaltest
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from configs.configs import config
from src.schemas.models import AspectInfo, PairingMetadata, ScoredCandidate
from src.stages.contracts import ClusteringStage
from src.stages.naming import ClusterNamer, MedoidNamer


STOP_SPANS: set[str] = set()
MDL_NORMALITY_PVALUE_THRESHOLD = 0.20


def _build_assignment_maps(
    aspects: Dict[str, AspectInfo],
    candidates: List[ScoredCandidate],
) -> tuple[dict[str, str], dict[str, str]]:
    span_to_cluster: dict[str, str] = {}
    for cluster_name, info in aspects.items():
        for keyword in info.keywords:
            span_to_cluster[str(keyword)] = str(cluster_name)

    candidate_to_cluster: dict[str, str] = {}
    for candidate in candidates:
        candidate_id = str(getattr(candidate, "candidate_id", "") or "")
        if not candidate_id:
            continue
        cluster_name = span_to_cluster.get(str(candidate.span), "")
        if cluster_name:
            candidate_to_cluster[candidate_id] = cluster_name

    return span_to_cluster, candidate_to_cluster


def _build_pairing_metadata(
    cluster_centroids: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    candidate_assignments: Dict[str, str],
) -> PairingMetadata:
    resolved_embeddings = (
        dict(cluster_centroids) if cluster_centroids else dict(anchor_embeddings)
    )
    return PairingMetadata(
        anchor_embeddings=resolved_embeddings,
        candidate_assignments=dict(candidate_assignments),
    )


@dataclass
class MDLDelta:
    delta_l_model: float
    delta_l_data: float
    delta_l_total: float


@dataclass
class MDLTreeNode:
    node_id: int
    indices: np.ndarray
    centroid: np.ndarray
    sum_sq_dist: float
    depth: int
    is_leaf: bool = True
    split_reason: str = ""
    parent_id: Optional[int] = None
    children_ids: tuple[int, int] | None = None
    split_diagnostics: dict[str, Any] = field(default_factory=dict)


def compute_delta_L(
    n_parent: int,
    V_parent: float,
    n1: int,
    V1: float,
    n2: int,
    V2: float,
    d: int,
    epsilon: float = 1e-12,
    use_aicc_correction: bool = False,
    model_penalty_alpha: float = 1.0,
) -> MDLDelta:
    """MDL gain from splitting one Gaussian cluster into two."""
    if min(n_parent, n1, n2, d) <= 0:
        raise ValueError("n_parent, n1, n2, d must be positive")

    sigma2_parent = float(V_parent) / float(n_parent * d) + float(epsilon)
    sigma2_child1 = float(V1) / float(n1 * d) + float(epsilon)
    sigma2_child2 = float(V2) / float(n2 * d) + float(epsilon)

    log_size_term = (
        math.log2(float(n_parent)) - math.log2(float(n1)) - math.log2(float(n2))
    )
    if use_aicc_correction:
        effective_k = int(d + 2)
        denom_parent = int(n_parent - effective_k - 1)
        denom_child1 = int(n1 - effective_k - 1)
        denom_child2 = int(n2 - effective_k - 1)
        if min(denom_parent, denom_child1, denom_child2) <= 0:
            delta_l_model = float("-inf")
        else:
            aicc_correction = (
                0.5
                * effective_k
                * (effective_k + 1)
                * math.log2(math.e)
                * (
                    (1.0 / float(denom_parent))
                    - (1.0 / float(denom_child1))
                    - (1.0 / float(denom_child2))
                )
            )
            delta_l_model = 0.5 * effective_k * log_size_term + aicc_correction
    else:
        delta_l_model = 0.5 * (d + 1) * log_size_term
    delta_l_data = 0.5 * d * (
        float(n_parent) * math.log2(sigma2_parent)
        - float(n1) * math.log2(sigma2_child1)
        - float(n2) * math.log2(sigma2_child2)
    )
    if use_aicc_correction:
        delta_l_total = float(delta_l_model + delta_l_data)
    else:
        delta_l_total = float(float(model_penalty_alpha) * delta_l_model + delta_l_data)
    return MDLDelta(
        delta_l_model=float(delta_l_model),
        delta_l_data=float(delta_l_data),
        delta_l_total=delta_l_total,
    )

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
        self._cluster_centroids: dict[str, np.ndarray] = {}
        self._anti_anchor_embeddings: dict[str, np.ndarray] = {}
        self._load_or_build_anchor_embeddings()

        self.last_assignment_counts: Dict[str, int] = {}
        self.last_residual_medoid_names: List[str] = []
        self.last_nli_medoid_diagnostics: List[str] = []
        self.last_span_assignments: Dict[str, str] = {}
        self.last_candidate_assignments: Dict[str, str] = {}

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
        self._cluster_centroids = {}
        self.last_assignment_counts = {}
        self.last_residual_medoid_names = []
        self.last_nli_medoid_diagnostics = []
        self.last_span_assignments = {}
        self.last_candidate_assignments = {}

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

        self.last_span_assignments, self.last_candidate_assignments = (
            _build_assignment_maps(filtered, candidates)
        )
        self._cluster_centroids = {
            name: np.asarray(info.centroid_embedding).flatten()
            for name, info in filtered.items()
        }
        return filtered

    def get_pairing_metadata(self) -> PairingMetadata:
        return _build_pairing_metadata(
            cluster_centroids=getattr(self, "_cluster_centroids", {}),
            anchor_embeddings=getattr(self, "_anchor_embeddings", {}),
            candidate_assignments=self.last_candidate_assignments,
        )

    def get_diagnostics(self) -> Dict[str, object]:
        diagnostics: Dict[str, object] = {}
        if self.last_assignment_counts:
            diagnostics["anchor_assignment_counts"] = dict(self.last_assignment_counts)
        diagnostics["residual_medoid_names"] = list(self.last_residual_medoid_names)
        diagnostics["nli_medoid_diagnostics"] = list(self.last_nli_medoid_diagnostics)
        diagnostics["clustering_stats"] = {
            "num_clusters": int(len(getattr(self, "_cluster_centroids", {}))),
        }
        return diagnostics

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
                agg[c.span] = {
                    "count": 0,
                    "embedding_sum": np.asarray(c.embedding, dtype=np.float32).copy(),
                    "score_sum": 0.0,
                }
            else:
                agg[c.span]["embedding_sum"] += np.asarray(c.embedding, dtype=np.float32)
            agg[c.span]["count"] += 1
            agg[c.span]["score_sum"] += c.score

        if min_mentions > 1:
            agg = {s: d for s, d in agg.items() if d["count"] >= min_mentions}
        for span, data in agg.items():
            data["embedding"] = data["embedding_sum"] / max(int(data["count"]), 1)
            data.pop("embedding_sum", None)
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


class DivisiveClusterer(ClusteringStage):
    def __init__(
        self,
        model: SentenceTransformer | None = None,
        min_variance_ratio: float = 0.05,
        max_clusters: int = 20,
        min_cluster_size: int = 3,
        umap_n_components: int = 5,
        umap_min_dist: float = 0.0,
        umap_metric: str = "cosine",
        random_state: int = 42,
        namer: ClusterNamer | None = None,
    ):
        self.model = model or SentenceTransformer(config.models.encoder_path)
        self.min_variance_ratio = float(min_variance_ratio)
        self.max_clusters = int(max_clusters)
        self.min_cluster_size = int(min_cluster_size)
        self.umap_n_components = int(umap_n_components)
        self.umap_min_dist = float(umap_min_dist)
        self.umap_metric = str(umap_metric)
        self.random_state = int(random_state)
        self.merge_threshold: float = float(config.discovery.cluster_merge_threshold)
        self.anti_anchor_threshold: float = float(config.discovery.anti_anchor_threshold)
        self.namer: ClusterNamer = namer or MedoidNamer()

        self._anchor_embeddings: dict[str, np.ndarray] = {}
        self._cluster_centroids: dict[str, np.ndarray] = {}
        self._anti_anchor_embeddings: dict[str, np.ndarray] = {}
        self._build_anti_anchor_embeddings()

        self.last_n_splits: int = 0
        self.last_n_rejected: int = 0
        self.last_leaf_variances: list[float] = []
        self.last_split_history: list[tuple[int, int, int, float]] = []
        self.last_span_assignments: Dict[str, str] = {}
        self.last_candidate_assignments: Dict[str, str] = {}

    def _build_anti_anchor_embeddings(self) -> None:
        for name, words in ANTI_ANCHORS.items():
            embs = self.model.encode(words, show_progress_bar=False)
            self._anti_anchor_embeddings[name] = np.mean(embs, axis=0)

    @staticmethod
    def _detect_product_stops(
        candidates: List[ScoredCandidate],
        freq_threshold: float = 0.05,
    ) -> set:
        if not candidates:
            return set()

        total = len(candidates)
        word_counts: dict[str, int] = {}
        for cand in candidates:
            if " " not in cand.span.strip():
                word = cand.span.strip().lower()
                word_counts[word] = word_counts.get(word, 0) + 1

        stops: set[str] = set()
        for word, count in word_counts.items():
            if count / total >= freq_threshold and len(word) >= 3:
                stops.add(word)
                for cand in candidates:
                    if cand.span.strip().lower() == word:
                        stops.add(cand.span.strip())
        return stops

    @staticmethod
    def _aggregate_spans(
        candidates: List[ScoredCandidate],
        min_mentions: int,
        stop_spans: set | None = None,
    ) -> dict:
        stops = stop_spans or STOP_SPANS
        agg: dict[str, dict] = {}
        for cand in candidates:
            if cand.span in stops:
                continue
            if cand.span not in agg:
                agg[cand.span] = {
                    "count": 0,
                    "embedding_sum": np.asarray(cand.embedding, dtype=np.float32).copy(),
                    "score_sum": 0.0,
                }
            else:
                agg[cand.span]["embedding_sum"] += np.asarray(cand.embedding, dtype=np.float32)
            agg[cand.span]["count"] += 1
            agg[cand.span]["score_sum"] += cand.score

        if min_mentions > 1:
            agg = {span: data for span, data in agg.items() if data["count"] >= min_mentions}
        for span, data in agg.items():
            data["embedding"] = data["embedding_sum"] / max(int(data["count"]), 1)
            data.pop("embedding_sum", None)
        return agg

    @staticmethod
    def _cluster_variance(points: np.ndarray) -> float:
        if points.size == 0:
            return 0.0
        centroid = np.mean(points, axis=0)
        sq = np.sum((points - centroid) ** 2, axis=1)
        return float(np.mean(sq))

    def _merge_similar_clusters_cosine(
        self,
        clusters: Dict[str, AspectInfo],
        span_data: dict[str, dict],
    ) -> Dict[str, AspectInfo]:
        if len(clusters) < 2:
            return clusters

        names = list(clusters.keys())
        while True:
            best_pair: tuple[str, str] | None = None
            best_sim = -1.0

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ci = clusters[names[i]].centroid_embedding.reshape(1, -1)
                    cj = clusters[names[j]].centroid_embedding.reshape(1, -1)
                    sim = float(cosine_similarity(ci, cj)[0, 0])
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (names[i], names[j])

            if best_pair is None or best_sim <= self.merge_threshold:
                break

            a, b = best_pair
            cluster_a = clusters[a]
            cluster_b = clusters[b]
            keep, drop = (a, b) if len(cluster_a.keywords) >= len(cluster_b.keywords) else (b, a)

            keep_info = clusters[keep]
            drop_info = clusters[drop]
            combined_keywords = keep_info.keywords + drop_info.keywords
            unique_keywords = list(dict.fromkeys(combined_keywords))
            embs = np.stack([span_data[s]["embedding"] for s in unique_keywords])
            centroid = np.mean(embs, axis=0)
            keyword_weights = [float(span_data[s]["count"]) for s in unique_keywords]
            clusters[keep] = AspectInfo(
                keywords=unique_keywords,
                centroid_embedding=centroid,
                keyword_weights=keyword_weights,
                nli_label=keep,
            )
            del clusters[drop]
            names = list(clusters.keys())

        return clusters

    def cluster(self, candidates: List[ScoredCandidate]) -> Dict[str, AspectInfo]:
        self.last_n_splits = 0
        self.last_n_rejected = 0
        self.last_leaf_variances = []
        self.last_split_history = []
        self._anchor_embeddings = {}
        self._cluster_centroids = {}
        self.last_span_assignments = {}
        self.last_candidate_assignments = {}

        if not candidates:
            return {}

        auto_stops = self._detect_product_stops(candidates)
        local_stops = STOP_SPANS | auto_stops
        span_data = self._aggregate_spans(candidates, min_mentions=1, stop_spans=local_stops)
        if len(span_data) < self.min_cluster_size:
            return {}

        spans = list(span_data.keys())
        embeddings_312 = np.stack([span_data[s]["embedding"] for s in spans])
        n_spans = len(spans)

        n_neighbors = max(2, min(15, n_spans // 10))
        reducer = umap.UMAP(
            n_components=min(self.umap_n_components, max(2, n_spans - 1)),
            n_neighbors=min(n_neighbors, n_spans - 1),
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            n_jobs=1,
            random_state=self.random_state,
        )
        embeddings_umap = reducer.fit_transform(embeddings_312)

        tree: dict[int, dict] = {
            0: {
                "id": 0,
                "indices": list(range(n_spans)),
                "variance": self._cluster_variance(embeddings_umap),
                "final": False,
            }
        }
        leaf_ids: set[int] = {0}
        next_id = 1

        while len(leaf_ids) < self.max_clusters:
            splittable = [
                cid for cid in leaf_ids
                if (not tree[cid]["final"]) and len(tree[cid]["indices"]) >= 2 * self.min_cluster_size
            ]
            if not splittable:
                break

            target_id = max(splittable, key=lambda cid: tree[cid]["variance"])
            target = tree[target_id]
            target_indices = target["indices"]
            points = embeddings_umap[target_indices]

            km = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(points)
            local_c1 = [i for i, lb in enumerate(labels) if lb == 0]
            local_c2 = [i for i, lb in enumerate(labels) if lb == 1]
            if not local_c1 or not local_c2:
                tree[target_id]["final"] = True
                self.last_n_rejected += 1
                continue

            c1_indices = [target_indices[i] for i in local_c1]
            c2_indices = [target_indices[i] for i in local_c2]
            v_parent = float(target["variance"])
            v1 = self._cluster_variance(embeddings_umap[c1_indices])
            v2 = self._cluster_variance(embeddings_umap[c2_indices])

            if v_parent <= 1e-12:
                rho = 0.0
            else:
                weighted_child = (
                    len(c1_indices) * v1 + len(c2_indices) * v2
                ) / float(len(target_indices))
                rho = 1.0 - (weighted_child / v_parent)

            child1_id = next_id
            child2_id = next_id + 1
            if rho > self.min_variance_ratio:
                leaf_ids.remove(target_id)
                tree[child1_id] = {
                    "id": child1_id,
                    "indices": c1_indices,
                    "variance": v1,
                    "final": False,
                }
                tree[child2_id] = {
                    "id": child2_id,
                    "indices": c2_indices,
                    "variance": v2,
                    "final": False,
                }
                leaf_ids.add(child1_id)
                leaf_ids.add(child2_id)
                self.last_n_splits += 1
                self.last_split_history.append((target_id, child1_id, child2_id, float(rho)))
                next_id += 2
            else:
                tree[target_id]["final"] = True
                self.last_n_rejected += 1

            if all(tree[cid]["final"] for cid in leaf_ids):
                break

        final_leaf_ids = list(leaf_ids)
        self.last_leaf_variances = [float(tree[cid]["variance"]) for cid in final_leaf_ids]

        aspects: Dict[str, AspectInfo] = {}
        for cid in final_leaf_ids:
            indices = tree[cid]["indices"]
            if len(indices) < self.min_cluster_size:
                continue

            cluster_embs_312 = embeddings_312[indices]
            centroid_312 = np.mean(cluster_embs_312, axis=0)

            cluster_spans = [spans[i] for i in indices]
            sims = cosine_similarity(cluster_embs_312, centroid_312.reshape(1, -1)).reshape(-1)
            medoid_local_idx = int(np.argmax(sims))
            name = cluster_spans[medoid_local_idx]

            keyword_weights = [float(span_data[s]["count"]) for s in cluster_spans]
            info = AspectInfo(
                keywords=cluster_spans,
                centroid_embedding=centroid_312,
                keyword_weights=keyword_weights,
                nli_label=name,
            )

            if name in aspects:
                merged_keywords = list(dict.fromkeys(aspects[name].keywords + info.keywords))
                merged_embs = np.stack([span_data[s]["embedding"] for s in merged_keywords])
                merged_centroid = np.mean(merged_embs, axis=0)
                merged_weights = [float(span_data[s]["count"]) for s in merged_keywords]
                aspects[name] = AspectInfo(
                    keywords=merged_keywords,
                    centroid_embedding=merged_centroid,
                    keyword_weights=merged_weights,
                    nli_label=name,
                )
            else:
                aspects[name] = info

        aspects = self._merge_similar_clusters_cosine(aspects, span_data)
        aspects = self.namer.rename(aspects)

        self._cluster_centroids = {
            name: np.asarray(info.centroid_embedding).flatten()
            for name, info in aspects.items()
        }
        self._anchor_embeddings = dict(self._cluster_centroids)
        return aspects

    def get_pairing_metadata(self) -> PairingMetadata:
        return _build_pairing_metadata(
            cluster_centroids=self._cluster_centroids,
            anchor_embeddings=self._anchor_embeddings,
            candidate_assignments=self.last_candidate_assignments,
        )

    def get_diagnostics(self) -> Dict[str, object]:
        return {
            "clustering_stats": {
                "num_clusters": int(len(self._cluster_centroids)),
                "mdl_accepted_splits": int(self.last_n_splits),
                "mdl_rejected_splits": int(self.last_n_rejected),
                "cluster_sizes": [],
            },
            "leaf_variances": list(self.last_leaf_variances),
            "split_history": list(self.last_split_history),
        }


class MDLDivisiveClusterer(DivisiveClusterer):
    def __init__(
        self,
        model: SentenceTransformer | None = None,
        min_cluster_size: Optional[int] = 3,
        kmeans_restarts: int = 10,
        kmeans_max_iter: int = 50,
        epsilon: float = 1e-12,
        use_aicc_correction: bool = False,
        model_penalty_alpha: float = 1.0,
        max_depth: Optional[int] = None,
        max_clusters: Optional[int] = None,
        namer: ClusterNamer | None = None,
    ):
        self.model = model
        self.min_cluster_size = (
            None if min_cluster_size is None else int(min_cluster_size)
        )
        self.kmeans_restarts = int(kmeans_restarts)
        self.kmeans_max_iter = int(kmeans_max_iter)
        self.epsilon = float(epsilon)
        self.use_aicc_correction = bool(use_aicc_correction)
        self.model_penalty_alpha = float(model_penalty_alpha)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.max_clusters = None if max_clusters is None else int(max_clusters)
        self.umap_n_components = int(config.discovery.umap_n_components)
        self.umap_min_dist = float(config.discovery.umap_min_dist)
        self.umap_metric = str(config.discovery.umap_metric)
        self.merge_threshold = float(config.discovery.cluster_merge_threshold)
        self.namer: ClusterNamer = namer or MedoidNamer()

        self._anchor_embeddings: dict[str, np.ndarray] = {}
        self._cluster_centroids: dict[str, np.ndarray] = {}

        self.last_n_splits: int = 0
        self.last_n_rejected: int = 0
        self.last_leaf_variances: list[float] = []
        self.last_split_history: list[tuple[int, int, int, float]] = []
        self.last_tree_nodes: list[MDLTreeNode] = []
        self.last_tree_summary: dict[str, Any] = {}
        self.last_split_reason_histogram: dict[str, int] = {}
        self.last_clustering_stats: dict[str, Any] = {}
        self.last_node_logs: list[str] = []
        self.last_sigma_sq_min: float = 0.0
        self.last_tree_dimension: int = 0
        self.last_effective_min_cluster_size: int = 0

    def _resolve_min_cluster_size(self, d: int) -> int:
        if self.min_cluster_size is not None:
            return int(self.min_cluster_size)
        return int(max(3, int(d) + 2))

    @staticmethod
    def _estimate_sigma_sq_min(embeddings: np.ndarray, k: int = 5) -> float:
        n_points, d = embeddings.shape
        if n_points <= 1 or d <= 0:
            return 0.0

        kth = min(k, n_points - 1)
        distances = pairwise_distances(embeddings, metric="euclidean")
        kth_nearest = np.partition(distances, kth, axis=1)[:, kth]
        sigma_sq_min = (float(np.percentile(kth_nearest, 5)) ** 2) / float(d)
        return float(max(sigma_sq_min, 0.0))

    @staticmethod
    def _sum_sq_dist(points: np.ndarray, centroid: Optional[np.ndarray] = None) -> float:
        if points.size == 0:
            return 0.0
        center = np.mean(points, axis=0) if centroid is None else centroid
        sq = np.sum((points - center) ** 2, axis=1)
        return float(np.sum(sq))

    def _make_node(
        self,
        node_id: int,
        indices: np.ndarray,
        embeddings: np.ndarray,
        depth: int,
        parent_id: Optional[int] = None,
    ) -> MDLTreeNode:
        node_points = embeddings[indices]
        centroid = np.mean(node_points, axis=0)
        sum_sq_dist = self._sum_sq_dist(node_points, centroid=centroid)
        return MDLTreeNode(
            node_id=node_id,
            indices=np.asarray(indices, dtype=int),
            centroid=np.asarray(centroid).flatten(),
            sum_sq_dist=float(sum_sq_dist),
            depth=depth,
            parent_id=parent_id,
        )

    def _best_bisect(self, points: np.ndarray) -> tuple[np.ndarray, float, float] | None:
        best_labels: np.ndarray | None = None
        best_total_v = float("inf")
        best_v1 = 0.0
        best_v2 = 0.0

        for restart in range(self.kmeans_restarts):
            km = KMeans(
                n_clusters=2,
                init="k-means++",
                n_init=1,
                max_iter=self.kmeans_max_iter,
                random_state=restart,
            )
            labels = km.fit_predict(points)
            mask0 = labels == 0
            mask1 = labels == 1
            if (not np.any(mask0)) or (not np.any(mask1)):
                continue
            v1 = self._sum_sq_dist(points[mask0])
            v2 = self._sum_sq_dist(points[mask1])
            total_v = float(v1 + v2)
            if total_v < best_total_v:
                best_total_v = total_v
                best_labels = labels.copy()
                best_v1 = float(v1)
                best_v2 = float(v2)

        if best_labels is None:
            return None
        return best_labels, best_v1, best_v2

    def _log_node(
        self,
        node: MDLTreeNode,
        n1: int,
        n2: int,
        v1: float,
        v2: float,
        delta: Optional[MDLDelta],
        decision: str,
    ) -> None:
        lines = [
            f"[MDL] Node n={len(node.indices)}, V={node.sum_sq_dist:.4f}, "
            f"V/n={node.sum_sq_dist / max(len(node.indices), 1):.4f}",
            f"  Bisect attempted: n1={n1}, n2={n2}, V1={v1:.4f}, V2={v2:.4f}",
        ]
        if delta is not None:
            lines.extend(
                [
                    f"  dL_model = {delta.delta_l_model:+.4f}",
                    f"  dL_data  = {delta.delta_l_data:+.4f}",
                    f"  dL_total = {delta.delta_l_total:+.4f} -> {decision}",
                ]
            )
        else:
            lines.append(f"  Decision  = {decision}")
        message = "\n".join(lines)
        self.last_node_logs.append(message)
        print(message)

    def _merge_leaf_groups_by_mdl(
        self,
        leaf_nodes: list[MDLTreeNode],
        embeddings: np.ndarray,
    ) -> list[np.ndarray]:
        groups = [
            np.asarray(node.indices, dtype=int).copy()
            for node in leaf_nodes
            if len(node.indices) >= self._resolve_min_cluster_size(int(embeddings.shape[1]))
        ]
        if len(groups) < 2:
            return groups

        d = int(embeddings.shape[1])
        while len(groups) > 1:
            best_pair: tuple[int, int] | None = None
            best_delta_total = float("inf")

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    idx_i = groups[i]
                    idx_j = groups[j]
                    parent_indices = np.concatenate([idx_i, idx_j])
                    parent_points = embeddings[parent_indices]
                    parent_v = self._sum_sq_dist(parent_points)
                    child_v1 = self._sum_sq_dist(embeddings[idx_i])
                    child_v2 = self._sum_sq_dist(embeddings[idx_j])
                    pair_labels = np.concatenate(
                        [
                            np.zeros(len(idx_i), dtype=int),
                            np.ones(len(idx_j), dtype=int),
                        ]
                    )
                    delta = compute_delta_L(
                        n_parent=len(parent_indices),
                        V_parent=parent_v,
                        n1=len(idx_i),
                        V1=child_v1,
                        n2=len(idx_j),
                        V2=child_v2,
                        d=d,
                        epsilon=self.epsilon,
                        use_aicc_correction=self.use_aicc_correction,
                        model_penalty_alpha=self.model_penalty_alpha,
                    )
                    normality_pvalue = self._split_normality_pvalue(parent_points, pair_labels)
                    if (
                        delta.delta_l_total < best_delta_total
                        and normality_pvalue is not None
                        and normality_pvalue >= MDL_NORMALITY_PVALUE_THRESHOLD
                    ):
                        best_delta_total = float(delta.delta_l_total)
                        best_pair = (i, j)

            if best_pair is None or best_delta_total > 0.0:
                break

            i, j = best_pair
            merged = np.concatenate([groups[i], groups[j]])
            groups[i] = np.asarray(merged, dtype=int)
            del groups[j]

        return groups

    @staticmethod
    def _split_normality_pvalue(points: np.ndarray, labels: np.ndarray) -> Optional[float]:
        if len(points) < 8:
            return None
        mask0 = labels == 0
        mask1 = labels == 1
        if (not np.any(mask0)) or (not np.any(mask1)):
            return None

        axis = np.mean(points[mask1], axis=0) - np.mean(points[mask0], axis=0)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            return 1.0

        projection = (points - np.mean(points, axis=0)) @ (axis / axis_norm)
        try:
            _, pvalue = normaltest(projection)
        except Exception:
            return None
        return float(pvalue)

    def _compute_clustering_stats(self, leaf_nodes: list[MDLTreeNode]) -> dict[str, Any]:
        leaf_sizes = [int(len(node.indices)) for node in leaf_nodes]
        if not leaf_sizes:
            return {
                "num_clusters": 0,
                "avg_cluster_size": 0.0,
                "median_cluster_size": 0,
                "largest_cluster_size": 0,
                "smallest_cluster_size": 0,
                "mdl_accepted_splits": int(self.last_n_splits),
                "mdl_rejected_splits": int(self.last_n_rejected),
                "cluster_sizes": [],
            }

        return {
            "num_clusters": int(len(leaf_sizes)),
            "avg_cluster_size": float(np.mean(leaf_sizes)),
            "median_cluster_size": int(np.median(leaf_sizes)),
            "largest_cluster_size": int(max(leaf_sizes)),
            "smallest_cluster_size": int(min(leaf_sizes)),
            "mdl_accepted_splits": int(self.last_n_splits),
            "mdl_rejected_splits": int(self.last_n_rejected),
            "cluster_sizes": leaf_sizes,
        }

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "clustering_stats": dict(self.last_clustering_stats),
            "tree_summary": dict(self.last_tree_summary),
            "split_reasons_histogram": dict(self.last_split_reason_histogram),
            "mdl_node_logs": list(self.last_node_logs),
            "sigma_sq_min": float(self.last_sigma_sq_min),
            "tree_dimension": int(self.last_tree_dimension),
            "effective_min_cluster_size": int(self.last_effective_min_cluster_size),
            "use_aicc_correction": bool(self.use_aicc_correction),
            "model_penalty_alpha": float(self.model_penalty_alpha),
        }

    def fit(
        self,
        embeddings: np.ndarray,
        spans: List[str],
        span_data: dict[str, dict],
        original_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, AspectInfo]:
        self.last_n_splits = 0
        self.last_n_rejected = 0
        self.last_leaf_variances = []
        self.last_split_history = []
        self.last_tree_nodes = []
        self.last_tree_summary = {}
        self.last_split_reason_histogram = {}
        self.last_clustering_stats = {}
        self.last_node_logs = []
        self.last_sigma_sq_min = 0.0
        self.last_tree_dimension = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
        self.last_effective_min_cluster_size = 0
        self._anchor_embeddings = {}
        self._cluster_centroids = {}

        if embeddings.size == 0:
            return {}
        point_embeddings = embeddings if original_embeddings is None else original_embeddings
        sigma_sq_min = self._estimate_sigma_sq_min(embeddings)
        self.last_sigma_sq_min = float(sigma_sq_min)

        n_points, d = embeddings.shape
        effective_min_cluster_size = self._resolve_min_cluster_size(d)
        self.last_effective_min_cluster_size = int(effective_min_cluster_size)
        nodes: dict[int, MDLTreeNode] = {}
        queue: list[tuple[float, int, int]] = []
        next_node_id = 0
        next_uid = 0

        root = self._make_node(
            node_id=next_node_id,
            indices=np.arange(n_points, dtype=int),
            embeddings=embeddings,
            depth=0,
        )
        nodes[root.node_id] = root
        heapq.heappush(
            queue,
            (-root.sum_sq_dist / max(len(root.indices), 1), next_uid, root.node_id),
        )
        next_node_id += 1
        next_uid += 1
        leaf_count = 1

        while queue:
            _, _, node_id = heapq.heappop(queue)
            node = nodes[node_id]

            if len(node.indices) < 2 * effective_min_cluster_size:
                node.is_leaf = True
                node.split_reason = "min_size"
                continue

            safety_hit = (
                (self.max_depth is not None and node.depth >= self.max_depth)
                or (self.max_clusters is not None and leaf_count >= self.max_clusters)
            )
            if safety_hit:
                node.is_leaf = True
                node.split_reason = "max_depth"
                continue

            sigma_sq_parent = float(node.sum_sq_dist) / float(len(node.indices) * d)
            if sigma_sq_parent < sigma_sq_min:
                node.is_leaf = True
                node.split_reason = "variance_floor"
                node.split_diagnostics = {
                    "n": int(len(node.indices)),
                    "V": float(node.sum_sq_dist),
                    "sigma_sq_parent": float(sigma_sq_parent),
                    "sigma_sq_min": float(sigma_sq_min),
                }
                continue

            points = embeddings[node.indices]
            best_split = self._best_bisect(points)
            if best_split is None:
                node.is_leaf = True
                node.split_reason = "degenerate_split"
                self._log_node(
                    node=node,
                    n1=0,
                    n2=0,
                    v1=0.0,
                    v2=0.0,
                    delta=None,
                    decision="DEGENERATE",
                )
                continue

            labels, v1, v2 = best_split
            mask0 = labels == 0
            mask1 = labels == 1
            n1 = int(np.sum(mask0))
            n2 = int(np.sum(mask1))

            if min(n1, n2) < effective_min_cluster_size:
                node.is_leaf = True
                node.split_reason = "degenerate_split"
                self._log_node(
                    node=node,
                    n1=n1,
                    n2=n2,
                    v1=v1,
                    v2=v2,
                    delta=None,
                    decision="DEGENERATE",
                )
                continue

            delta = compute_delta_L(
                n_parent=len(node.indices),
                V_parent=node.sum_sq_dist,
                n1=n1,
                V1=v1,
                n2=n2,
                V2=v2,
                d=d,
                epsilon=self.epsilon,
                use_aicc_correction=self.use_aicc_correction,
                model_penalty_alpha=self.model_penalty_alpha,
            )

            node.split_diagnostics = {
                "n": int(len(node.indices)),
                "V": float(node.sum_sq_dist),
                "V_per_point": float(node.sum_sq_dist / max(len(node.indices), 1)),
                "n1": n1,
                "n2": n2,
                "V1": float(v1),
                "V2": float(v2),
                "delta_l_model": float(delta.delta_l_model),
                "delta_l_data": float(delta.delta_l_data),
                "delta_l_total": float(delta.delta_l_total),
            }
            normality_pvalue = self._split_normality_pvalue(points, labels)
            node.split_diagnostics["split_normality_pvalue"] = normality_pvalue

            if delta.delta_l_total > 0.0 and (
                normality_pvalue is None or normality_pvalue < MDL_NORMALITY_PVALUE_THRESHOLD
            ):
                child1 = self._make_node(
                    node_id=next_node_id,
                    indices=node.indices[mask0],
                    embeddings=embeddings,
                    depth=node.depth + 1,
                    parent_id=node.node_id,
                )
                child2 = self._make_node(
                    node_id=next_node_id + 1,
                    indices=node.indices[mask1],
                    embeddings=embeddings,
                    depth=node.depth + 1,
                    parent_id=node.node_id,
                )
                nodes[child1.node_id] = child1
                nodes[child2.node_id] = child2
                node.is_leaf = False
                node.split_reason = "mdl_accepted"
                node.children_ids = (child1.node_id, child2.node_id)

                heapq.heappush(
                    queue,
                    (-child1.sum_sq_dist / max(len(child1.indices), 1), next_uid, child1.node_id),
                )
                next_uid += 1
                heapq.heappush(
                    queue,
                    (-child2.sum_sq_dist / max(len(child2.indices), 1), next_uid, child2.node_id),
                )
                next_uid += 1
                next_node_id += 2
                leaf_count += 1
                self.last_n_splits += 1
                self.last_split_history.append(
                    (
                        node.node_id,
                        child1.node_id,
                        child2.node_id,
                        float(delta.delta_l_total),
                    )
                )
                self._log_node(
                    node=node,
                    n1=n1,
                    n2=n2,
                    v1=v1,
                    v2=v2,
                    delta=delta,
                    decision="ACCEPTED",
                )
            else:
                node.is_leaf = True
                node.split_reason = "mdl_rejected"
                self.last_n_rejected += 1
                self._log_node(
                    node=node,
                    n1=n1,
                    n2=n2,
                    v1=v1,
                    v2=v2,
                    delta=delta,
                    decision="REJECTED",
                )

        self.last_tree_nodes = [nodes[node_id] for node_id in sorted(nodes)]
        leaf_nodes = [node for node in self.last_tree_nodes if node.is_leaf]
        self.last_leaf_variances = [
            float(node.sum_sq_dist / max(len(node.indices), 1)) for node in leaf_nodes
        ]
        histogram: dict[str, int] = {}
        for node in self.last_tree_nodes:
            if node.split_reason:
                histogram[node.split_reason] = histogram.get(node.split_reason, 0) + 1
        self.last_split_reason_histogram = histogram
        self.last_tree_summary = {
            "total_nodes": int(len(self.last_tree_nodes)),
            "leaves": int(len(leaf_nodes)),
            "max_depth": int(max((node.depth for node in self.last_tree_nodes), default=0)),
            "split_reasons_histogram": dict(histogram),
            "sigma_sq_min": float(sigma_sq_min),
            "tree_dimension": int(d),
            "effective_min_cluster_size": int(effective_min_cluster_size),
            "use_aicc_correction": bool(self.use_aicc_correction),
            "model_penalty_alpha": float(self.model_penalty_alpha),
        }
        self.last_clustering_stats = self._compute_clustering_stats(leaf_nodes)

        print("[MDL] Tree summary:")
        print(f"  Total nodes: {self.last_tree_summary['total_nodes']}")
        print(f"  Leaves: {self.last_tree_summary['leaves']}")
        print(f"  Max depth: {self.last_tree_summary['max_depth']}")
        print(f"  Tree dimension: {self.last_tree_summary['tree_dimension']}")
        print(
            f"  effective_min_cluster_size: "
            f"{self.last_tree_summary['effective_min_cluster_size']}"
        )
        print(f"  sigma_sq_min: {self.last_tree_summary['sigma_sq_min']:.6f}")
        print(
            f"  use_aicc_correction: {self.last_tree_summary['use_aicc_correction']}"
        )
        print(f"  model_penalty_alpha: {self.last_tree_summary['model_penalty_alpha']:.3f}")
        print("  Split reasons histogram:")
        for reason in (
            "mdl_accepted",
            "mdl_rejected",
            "min_size",
            "degenerate_split",
            "variance_floor",
            "max_depth",
        ):
            print(f"    {reason:<17} {histogram.get(reason, 0)}")

        merged_groups = self._merge_leaf_groups_by_mdl(leaf_nodes, embeddings)
        self.last_clustering_stats = self._compute_clustering_stats(
            [
                MDLTreeNode(
                    node_id=-1,
                    indices=np.asarray(indices, dtype=int),
                    centroid=np.mean(point_embeddings[indices], axis=0),
                    sum_sq_dist=self._sum_sq_dist(point_embeddings[indices]),
                    depth=0,
                )
                for indices in merged_groups
            ]
        )

        aspects: Dict[str, AspectInfo] = {}
        for group_indices in merged_groups:
            indices = group_indices.tolist()
            if len(indices) < effective_min_cluster_size:
                continue

            cluster_embs = point_embeddings[indices]
            centroid = np.mean(cluster_embs, axis=0)
            cluster_spans = [spans[i] for i in indices]
            sims = cosine_similarity(cluster_embs, centroid.reshape(1, -1)).reshape(-1)
            medoid_local_idx = int(np.argmax(sims))
            name = cluster_spans[medoid_local_idx]
            keyword_weights = [float(span_data[s]["count"]) for s in cluster_spans]
            info = AspectInfo(
                keywords=cluster_spans,
                centroid_embedding=centroid,
                keyword_weights=keyword_weights,
                nli_label=name,
            )

            if name in aspects:
                merged_keywords = list(dict.fromkeys(aspects[name].keywords + info.keywords))
                merged_embs = np.stack([span_data[s]["embedding"] for s in merged_keywords])
                merged_centroid = np.mean(merged_embs, axis=0)
                merged_weights = [float(span_data[s]["count"]) for s in merged_keywords]
                aspects[name] = AspectInfo(
                    keywords=merged_keywords,
                    centroid_embedding=merged_centroid,
                    keyword_weights=merged_weights,
                    nli_label=name,
                )
            else:
                aspects[name] = info

        aspects = self._merge_similar_clusters_cosine(aspects, span_data)
        aspects = self.namer.rename(aspects)
        self._cluster_centroids = {
            name: np.asarray(info.centroid_embedding).flatten()
            for name, info in aspects.items()
        }
        self._anchor_embeddings = dict(self._cluster_centroids)
        return aspects

    def cluster(self, candidates: List[ScoredCandidate]) -> Dict[str, AspectInfo]:
        self._anchor_embeddings = {}
        self._cluster_centroids = {}
        self.last_span_assignments = {}
        self.last_candidate_assignments = {}
        if not candidates:
            self.last_clustering_stats = self._compute_clustering_stats([])
            self.last_tree_summary = {
                "total_nodes": 0,
                "leaves": 0,
                "max_depth": 0,
                "split_reasons_histogram": {},
            }
            self.last_split_reason_histogram = {}
            self.last_tree_nodes = []
            self.last_node_logs = []
            self.last_effective_min_cluster_size = 0
            return {}

        auto_stops = self._detect_product_stops(candidates)
        local_stops = STOP_SPANS | auto_stops
        span_data = self._aggregate_spans(candidates, min_mentions=1, stop_spans=local_stops)
        spans = list(span_data.keys())
        if not spans:
            self.last_clustering_stats = self._compute_clustering_stats([])
            self.last_tree_summary = {
                "total_nodes": 0,
                "leaves": 0,
                "max_depth": 0,
                "split_reasons_histogram": {},
            }
            self.last_split_reason_histogram = {}
            self.last_tree_nodes = []
            self.last_node_logs = []
            self.last_effective_min_cluster_size = 0
            return {}

        original_embeddings = np.stack([span_data[s]["embedding"] for s in spans])
        n_points = len(spans)
        tree_embeddings = original_embeddings
        prospective_tree_d = int(original_embeddings.shape[1])

        if n_points >= 5:
            prospective_tree_d = int(min(self.umap_n_components, max(2, n_points - 1)))
        effective_min_cluster_size = self._resolve_min_cluster_size(prospective_tree_d)

        if n_points >= max(5, effective_min_cluster_size + 2):
            n_neighbors = max(2, min(15, n_points // 10))
            reducer = umap.UMAP(
                n_components=min(self.umap_n_components, max(2, n_points - 1)),
                n_neighbors=min(n_neighbors, n_points - 1),
                min_dist=self.umap_min_dist,
                metric=self.umap_metric,
                n_jobs=1,
                random_state=42,
            )
            tree_embeddings = reducer.fit_transform(original_embeddings)

        aspects = self.fit(
            embeddings=tree_embeddings,
            spans=spans,
            span_data=span_data,
            original_embeddings=original_embeddings,
        )
        self.last_span_assignments, self.last_candidate_assignments = (
            _build_assignment_maps(aspects, candidates)
        )
        return aspects


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
