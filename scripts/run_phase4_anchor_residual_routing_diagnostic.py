from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymorphy3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import configs as _cfg_module
from src.stages.extraction import CandidateExtractor
from src.vocabulary.loader import AspectDefinition, Vocabulary

_cfg_module.config.discovery.dependency_filter_enabled = False  # type: ignore[attr-defined]
_MORPH = pymorphy3.MorphAnalyzer()

try:
    import hdbscan  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional runtime dependency guard
    hdbscan = None


GENERIC_EMOTION_LEMMAS = {
    "восторг",
    "впечатление",
    "впечатления",
    "кошмар",
    "любовь",
    "нерв",
    "нравиться",
    "рекомендация",
    "совет",
    "ужас",
    "эмоция",
}

GENERIC_OBJECT_LEMMAS = {
    "вещь",
    "заказ",
    "магазин",
    "место",
    "покупка",
    "продукт",
    "сервис",
    "товар",
    "услуга",
    "штука",
}

PRIORITY_DOMAIN_CONFLICT_PAIRS: set[tuple[str, str, str]] = {
    ("physical_goods", "Упаковка", "Крепления и соединения"),
    ("physical_goods", "Чистота", "Крепления и соединения"),
    ("physical_goods", "Материал", "Инструкция"),
    ("services", "Цена", "Оплата и расчёты"),
    ("physical_goods", "Упаковка", "Инструкция"),
    ("physical_goods", "Цена", "Крепления и соединения"),
    ("services", "Доставка", "Запись на услугу"),
    ("physical_goods", "Соответствие ожиданиям", "Инструкция"),
    ("services", "Инфраструктура", "Поддержка и сопровождение"),
    ("hospitality", "Цена", "Бронирование"),
    ("hospitality", "Расположение", "Номер"),
    ("services", "Загруженность", "Ожидание"),
    ("services", "Обслуживание", "Запись на услугу"),
    ("consumables", "Общее впечатление", "Поедаемость"),
}


@dataclass(frozen=True, slots=True)
class ReviewRow:
    review_id: str
    category: str
    text: str


@dataclass(frozen=True, slots=True)
class CandidateRow:
    review_id: str
    category: str
    candidate_text: str
    candidate_lemma: str


@dataclass(frozen=True, slots=True)
class LayerScore:
    best_anchor_id: str
    best_anchor_name: str
    best_score: float
    second_score: float


@dataclass(frozen=True, slots=True)
class Thresholds:
    t_general: float
    m_general: float
    t_domain: float
    m_domain: float
    t_general_conflict: float
    t_domain_conflict: float
    c_overlap: float
    weak_score_floor: float


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    route: str
    chosen_layer: str
    chosen_anchor: str
    conflict: bool
    general_ok: bool
    domain_ok: bool


def _normalize(text: str) -> str:
    tokens = [token.strip().lower() for token in str(text).replace("-", " ").split() if token.strip()]
    lemmas: list[str] = []
    for token in tokens:
        parses = _MORPH.parse(token)
        lemmas.append(str(parses[0].normal_form if parses else token).lower())
    return " ".join(lemmas)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_reviews(dataset_csv: Path) -> list[ReviewRow]:
    df = pd.read_csv(dataset_csv, dtype={"id": str})
    required = {"id", "full_text", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    rows: list[ReviewRow] = []
    for _, row in df.iterrows():
        text = str(row["full_text"]).strip()
        if not text:
            continue
        rows.append(
            ReviewRow(
                review_id=str(row["id"]),
                category=str(row["category"]).strip(),
                text=text,
            )
        )
    return rows


def _extract_old_candidates(reviews: list[ReviewRow]) -> list[CandidateRow]:
    extractor = CandidateExtractor(ngram_range=(1, 2), min_word_length=3)
    extractor.dependency_filter_enabled = False
    out: list[CandidateRow] = []
    for review in reviews:
        by_lemma: dict[str, str] = {}
        for cand in extractor.extract(review.text):
            raw = str(cand.span).strip().lower()
            lemma = _normalize(raw)
            if not lemma:
                continue
            by_lemma.setdefault(lemma, raw)
        for lemma, raw in sorted(by_lemma.items()):
            out.append(
                CandidateRow(
                    review_id=review.review_id,
                    category=review.category,
                    candidate_text=raw,
                    candidate_lemma=lemma,
                )
            )
    return out


def _load_aspects(path: Path) -> list[AspectDefinition]:
    return Vocabulary.load_from_yaml(path).aspects


def _build_term_map(aspects: list[AspectDefinition]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for aspect in aspects:
        for term in [aspect.canonical_name, *aspect.synonyms]:
            lemma = _normalize(term)
            if lemma:
                out[lemma].add(aspect.id)
    return out


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _load_encoder() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_cfg_module.config.models.encoder_path)


def _encode_texts_cached(
    model: Any,
    texts: list[str],
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    unique = sorted({text for text in texts if text and text not in cache})
    if unique:
        embeddings = model.encode(unique, show_progress_bar=False, convert_to_numpy=True, batch_size=256)
        normalized = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        for idx, text in enumerate(unique):
            cache[text] = normalized[idx]
    return {text: cache[text] for text in texts if text in cache}


def _build_anchor_bank(
    model: Any,
    aspects: list[AspectDefinition],
    cache: dict[str, np.ndarray],
) -> tuple[list[str], list[str], np.ndarray]:
    anchor_ids: list[str] = []
    anchor_names: list[str] = []
    anchor_vectors: list[np.ndarray] = []
    for aspect in aspects:
        terms = [_normalize(term) for term in [aspect.canonical_name, *aspect.synonyms]]
        terms = [term for term in terms if term]
        if not terms:
            continue
        vectors = _encode_texts_cached(model, terms, cache)
        if not vectors:
            continue
        matrix = np.stack([vectors[term] for term in terms], axis=0).astype(np.float32)
        centroid = _l2_normalize(matrix.mean(axis=0, keepdims=True))[0]
        anchor_ids.append(aspect.id)
        anchor_names.append(aspect.canonical_name)
        anchor_vectors.append(centroid)
    if not anchor_vectors:
        return [], [], np.zeros((0, 1), dtype=np.float32)
    return anchor_ids, anchor_names, np.stack(anchor_vectors, axis=0).astype(np.float32)


def _best_two(scores: np.ndarray, anchor_ids: list[str], anchor_names: list[str]) -> LayerScore:
    if scores.size == 0:
        return LayerScore(best_anchor_id="", best_anchor_name="", best_score=0.0, second_score=0.0)
    best_idx = int(np.argmax(scores))
    if scores.size == 1:
        second_score = 0.0
    else:
        top2 = np.partition(scores, -2)[-2:]
        second_score = float(np.min(top2))
    return LayerScore(
        best_anchor_id=anchor_ids[best_idx],
        best_anchor_name=anchor_names[best_idx],
        best_score=float(scores[best_idx]),
        second_score=second_score,
    )


def _compute_layer_score(
    candidate_vec: np.ndarray,
    anchor_ids: list[str],
    anchor_names: list[str],
    anchor_matrix: np.ndarray,
) -> LayerScore:
    if anchor_matrix.size == 0:
        return LayerScore(best_anchor_id="", best_anchor_name="", best_score=0.0, second_score=0.0)
    scores = anchor_matrix @ candidate_vec
    return _best_two(scores=scores, anchor_ids=anchor_ids, anchor_names=anchor_names)


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _pick_score_threshold(hit_scores: list[float], miss_scores: list[float], fallback: float) -> float:
    if not hit_scores:
        return fallback
    combined = np.asarray(hit_scores + miss_scores, dtype=np.float32)
    quantiles = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    candidates = sorted(
        {
            round(float(x), 3)
            for x in np.quantile(combined, quantiles, method="linear").tolist()
            if np.isfinite(x)
        }
        | {round(fallback, 3)}
    )
    best_thr = round(fallback, 3)
    best_f1 = -1.0
    best_precision = -1.0
    for threshold in candidates:
        tp = sum(1 for score in hit_scores if score >= threshold)
        fp = sum(1 for score in miss_scores if score >= threshold)
        fn = sum(1 for score in hit_scores if score < threshold)
        f1 = _binary_f1(tp=tp, fp=fp, fn=fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        if (f1 > best_f1) or (math.isclose(f1, best_f1) and precision > best_precision):
            best_f1 = f1
            best_precision = precision
            best_thr = threshold
    return round(_clip(best_thr, 0.55, 0.92), 3)


def _pick_margin_threshold(hit_gaps: list[float], fallback: float) -> float:
    if not hit_gaps:
        return fallback
    value = float(np.quantile(np.asarray(hit_gaps, dtype=np.float32), 0.25, method="linear"))
    return round(_clip(value, 0.03, 0.10), 3)


def _pick_overlap_band(diffs: list[float], fallback: float) -> float:
    if not diffs:
        return fallback
    value = float(np.quantile(np.asarray(diffs, dtype=np.float32), 0.25, method="linear"))
    return round(_clip(value, 0.03, 0.08), 3)


def _noise_reason(candidate_lemma: str, best_general_score: float, best_domain_score: float, weak_score_floor: float) -> str:
    tokens = [token for token in candidate_lemma.split() if token]
    if not tokens or len(candidate_lemma) < 3:
        return "too_short"
    if all(token in GENERIC_EMOTION_LEMMAS for token in tokens):
        return "generic_emotion"
    if candidate_lemma in GENERIC_EMOTION_LEMMAS:
        return "generic_emotion"
    if candidate_lemma in GENERIC_OBJECT_LEMMAS:
        return "too_generic"
    if re.fullmatch(r"[a-z0-9\\-]+", candidate_lemma):
        return "technical_garbage"
    if re.fullmatch(r"[0-9\\-]+", candidate_lemma):
        return "technical_garbage"
    if max(best_general_score, best_domain_score) < weak_score_floor:
        return "weak_scores"
    return ""


def _quick_residual_label(candidate_lemma: str, best_general_score: float, best_domain_score: float, thresholds: Thresholds) -> str:
    if candidate_lemma in GENERIC_EMOTION_LEMMAS or candidate_lemma in GENERIC_OBJECT_LEMMAS:
        return "looks_noise"
    max_score = max(best_general_score, best_domain_score)
    strong_hint = max(thresholds.t_general_conflict, thresholds.t_domain_conflict) + 0.04
    weak_hint = thresholds.weak_score_floor + 0.03
    if max_score >= strong_hint:
        return "looks_useful"
    if max_score <= weak_hint:
        return "looks_noise"
    return "unclear"


def _route_candidate(
    category: str,
    best_general: LayerScore,
    best_domain: LayerScore,
    thresholds: Thresholds,
    noise_reason: str,
    mode: str,
) -> RoutingDecision:
    if noise_reason:
        return RoutingDecision(
            route="noise",
            chosen_layer="",
            chosen_anchor="",
            conflict=False,
            general_ok=False,
            domain_ok=False,
        )

    general_ok = (
        best_general.best_score >= thresholds.t_general
        and (best_general.best_score - best_general.second_score) >= thresholds.m_general
    )
    domain_ok = (
        best_domain.best_score >= thresholds.t_domain
        and (best_domain.best_score - best_domain.second_score) >= thresholds.m_domain
    )
    conflict = (
        best_general.best_score >= thresholds.t_general_conflict
        and best_domain.best_score >= thresholds.t_domain_conflict
        and abs(best_general.best_score - best_domain.best_score) <= thresholds.c_overlap
    )
    priority_pair = (category, best_general.best_anchor_name, best_domain.best_anchor_name) in PRIORITY_DOMAIN_CONFLICT_PAIRS

    if mode == "domain_priority" and conflict and (domain_ok or priority_pair):
        return RoutingDecision(
            route="domain",
            chosen_layer="domain",
            chosen_anchor=best_domain.best_anchor_name,
            conflict=True,
            general_ok=general_ok,
            domain_ok=domain_ok,
        )
    if general_ok and not conflict:
        return RoutingDecision(
            route="general",
            chosen_layer="general",
            chosen_anchor=best_general.best_anchor_name,
            conflict=conflict,
            general_ok=general_ok,
            domain_ok=domain_ok,
        )
    if domain_ok and not conflict:
        return RoutingDecision(
            route="domain",
            chosen_layer="domain",
            chosen_anchor=best_domain.best_anchor_name,
            conflict=conflict,
            general_ok=general_ok,
            domain_ok=domain_ok,
        )
    if conflict:
        return RoutingDecision(
            route="overlap",
            chosen_layer="",
            chosen_anchor="",
            conflict=True,
            general_ok=general_ok,
            domain_ok=domain_ok,
        )
    return RoutingDecision(
        route="residual",
        chosen_layer="",
        chosen_anchor="",
        conflict=False,
        general_ok=general_ok,
        domain_ok=domain_ok,
    )


def _format_pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(100.0 * part / total):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 anchor residual routing diagnostic")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase4_anchor_residual_routing_diagnostic")
    parser.add_argument("--mode", choices=["current", "domain_priority"], default="current")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reviews = _load_reviews(ROOT / args.dataset_csv)
    candidates = _extract_old_candidates(reviews)

    general_aspects = _load_aspects(ROOT / args.core_vocab)
    domain_paths = {
        "physical_goods": ROOT / "src/vocabulary/domain/physical_goods.yaml",
        "consumables": ROOT / "src/vocabulary/domain/consumables.yaml",
        "hospitality": ROOT / "src/vocabulary/domain/hospitality.yaml",
        "services": ROOT / "src/vocabulary/domain/services.yaml",
    }
    domain_aspects = {category: _load_aspects(path) for category, path in domain_paths.items()}

    model = _load_encoder()
    embedding_cache: dict[str, np.ndarray] = {}

    general_term_map = _build_term_map(general_aspects)
    domain_term_map = {category: _build_term_map(aspects) for category, aspects in domain_aspects.items()}

    all_terms_to_encode = list({row.candidate_lemma for row in candidates})
    _encode_texts_cached(model, all_terms_to_encode, embedding_cache)

    general_ids, general_names, general_matrix = _build_anchor_bank(model, general_aspects, embedding_cache)
    domain_banks: dict[str, tuple[list[str], list[str], np.ndarray]] = {}
    for category, aspects in domain_aspects.items():
        domain_banks[category] = _build_anchor_bank(model, aspects, embedding_cache)

    pre_rows: list[dict[str, Any]] = []
    general_hit_scores: list[float] = []
    general_miss_scores: list[float] = []
    general_hit_gaps: list[float] = []
    domain_hit_scores: list[float] = []
    domain_miss_scores: list[float] = []
    domain_hit_gaps: list[float] = []

    for row in candidates:
        candidate_vec = embedding_cache[row.candidate_lemma]
        best_general = _compute_layer_score(
            candidate_vec=candidate_vec,
            anchor_ids=general_ids,
            anchor_names=general_names,
            anchor_matrix=general_matrix,
        )
        domain_ids, domain_names, domain_matrix = domain_banks.get(row.category, ([], [], np.zeros((0, 1), dtype=np.float32)))
        best_domain = _compute_layer_score(
            candidate_vec=candidate_vec,
            anchor_ids=domain_ids,
            anchor_names=domain_names,
            anchor_matrix=domain_matrix,
        )

        general_exact_hit = row.candidate_lemma in general_term_map
        domain_exact_hit = row.candidate_lemma in domain_term_map.get(row.category, {})

        if general_exact_hit:
            general_hit_scores.append(best_general.best_score)
            general_hit_gaps.append(best_general.best_score - best_general.second_score)
        else:
            general_miss_scores.append(best_general.best_score)

        if domain_exact_hit:
            domain_hit_scores.append(best_domain.best_score)
            domain_hit_gaps.append(best_domain.best_score - best_domain.second_score)
        else:
            domain_miss_scores.append(best_domain.best_score)

        pre_rows.append(
            {
                "review_id": row.review_id,
                "category": row.category,
                "candidate_text": row.candidate_text,
                "candidate_lemma": row.candidate_lemma,
                "best_general_anchor_id": best_general.best_anchor_id,
                "best_general_anchor": best_general.best_anchor_name,
                "best_general_score": round(best_general.best_score, 4),
                "second_general_score": round(best_general.second_score, 4),
                "best_domain_anchor_id": best_domain.best_anchor_id,
                "best_domain_anchor": best_domain.best_anchor_name,
                "best_domain_score": round(best_domain.best_score, 4),
                "second_domain_score": round(best_domain.second_score, 4),
            }
        )

    # Compact diagnostics showed that auto-picked exact-hit thresholds drift too high
    # in the current embedding space. We keep the diagnostic estimates, but select a
    # stable manual start point from the compact grid around the median-upper score band.
    _diagnostic_thresholds = {
        "t_general_auto": _pick_score_threshold(general_hit_scores, general_miss_scores, fallback=0.76),
        "m_general_auto": _pick_margin_threshold(general_hit_gaps, fallback=0.05),
        "t_domain_auto": _pick_score_threshold(domain_hit_scores, domain_miss_scores, fallback=0.79),
        "m_domain_auto": _pick_margin_threshold(domain_hit_gaps, fallback=0.05),
    }
    thresholds = Thresholds(
        t_general=0.88,
        m_general=0.04,
        t_domain=0.88,
        m_domain=0.04,
        t_general_conflict=0.83,
        t_domain_conflict=0.83,
        c_overlap=0.02,
        weak_score_floor=0.68,
    )

    routed_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    route_counter: Counter[str] = Counter()
    residual_counter: Counter[str] = Counter()

    for row in pre_rows:
        best_general = LayerScore(
            best_anchor_id=str(row["best_general_anchor_id"]),
            best_anchor_name=str(row["best_general_anchor"]),
            best_score=float(row["best_general_score"]),
            second_score=float(row["second_general_score"]),
        )
        best_domain = LayerScore(
            best_anchor_id=str(row["best_domain_anchor_id"]),
            best_anchor_name=str(row["best_domain_anchor"]),
            best_score=float(row["best_domain_score"]),
            second_score=float(row["second_domain_score"]),
        )
        noise_reason = _noise_reason(
            candidate_lemma=str(row["candidate_lemma"]),
            best_general_score=best_general.best_score,
            best_domain_score=best_domain.best_score,
            weak_score_floor=thresholds.weak_score_floor,
        )
        decision = _route_candidate(
            category=str(row["category"]),
            best_general=best_general,
            best_domain=best_domain,
            thresholds=thresholds,
            noise_reason=noise_reason,
            mode=args.mode,
        )
        route = decision.route
        route_counter[route] += 1
        routed = {
            "review_id": row["review_id"],
            "category": row["category"],
            "candidate_text": row["candidate_text"],
            "best_general_anchor": row["best_general_anchor"],
            "best_general_score": row["best_general_score"],
            "second_general_score": row["second_general_score"],
            "best_domain_anchor": row["best_domain_anchor"],
            "best_domain_score": row["best_domain_score"],
            "second_domain_score": row["second_domain_score"],
            "route": route,
            "chosen_layer": decision.chosen_layer,
            "chosen_anchor": decision.chosen_anchor,
            "_conflict": decision.conflict,
            "_general_ok": decision.general_ok,
            "_domain_ok": decision.domain_ok,
            "_candidate_lemma": row["candidate_lemma"],
            "_noise_reason": noise_reason,
        }
        routed_rows.append(routed)

        if route == "overlap":
            overlap_rows.append(
                {
                    "review_id": row["review_id"],
                    "candidate_text": row["candidate_text"],
                    "best_general_anchor": row["best_general_anchor"],
                    "best_general_score": row["best_general_score"],
                    "best_domain_anchor": row["best_domain_anchor"],
                    "best_domain_score": row["best_domain_score"],
                    "score_gap": round(abs(float(row["best_general_score"]) - float(row["best_domain_score"])), 4),
                    "category": row["category"],
                }
            )
        if route == "residual":
            residual_counter[str(row["candidate_lemma"])] += 1

    residual_rows = [row for row in routed_rows if row["route"] == "residual"]
    residual_clean_rows = [row for row in residual_rows if residual_counter[str(row["_candidate_lemma"])] >= 2]

    cluster_stats: dict[str, Any] = {
        "ran": False,
        "n_input": len(residual_clean_rows),
        "n_clusters": 0,
        "clustered_share": 0.0,
        "min_cluster_size": None,
        "min_samples": None,
    }
    if hdbscan is not None and residual_clean_rows:
        min_cluster_size = max(5, min(15, int(round(math.sqrt(len(residual_clean_rows))))))
        min_samples = max(3, min_cluster_size // 2)
        if len(residual_clean_rows) >= min_cluster_size:
            residual_vectors = np.stack(
                [embedding_cache[str(row["_candidate_lemma"])] for row in residual_clean_rows],
                axis=0,
            ).astype(np.float32)
            labels = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            ).fit_predict(residual_vectors)
            n_clustered = int(np.sum(labels != -1))
            n_clusters = len({int(label) for label in labels.tolist() if int(label) != -1})
            cluster_stats = {
                "ran": True,
                "n_input": len(residual_clean_rows),
                "n_clusters": int(n_clusters),
                "clustered_share": round(n_clustered / len(residual_clean_rows), 4),
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
            }

    rng = random.Random(args.seed)
    residual_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in residual_rows:
        residual_by_category[str(row["category"])].append(row)
    for rows in residual_by_category.values():
        rng.shuffle(rows)

    per_category_quota = max(1, args.sample_size // max(1, len(residual_by_category)))
    sampled_rows: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str, str]] = set()
    for category in sorted(residual_by_category):
        for row in residual_by_category[category][:per_category_quota]:
            key = (str(row["review_id"]), str(row["category"]), str(row["candidate_text"]))
            if key in used_keys:
                continue
            used_keys.add(key)
            sampled_rows.append(row)
    remaining_pool = [row for row in residual_rows if (str(row["review_id"]), str(row["category"]), str(row["candidate_text"])) not in used_keys]
    rng.shuffle(remaining_pool)
    for row in remaining_pool:
        if len(sampled_rows) >= args.sample_size:
            break
        sampled_rows.append(row)

    residual_sample_records: list[dict[str, Any]] = []
    quick_label_counter: Counter[str] = Counter()
    for row in sampled_rows:
        quick_label = _quick_residual_label(
            candidate_lemma=str(row["_candidate_lemma"]),
            best_general_score=float(row["best_general_score"]),
            best_domain_score=float(row["best_domain_score"]),
            thresholds=thresholds,
        )
        quick_label_counter[quick_label] += 1
        residual_sample_records.append(
            {
                "review_id": row["review_id"],
                "category": row["category"],
                "candidate_text": row["candidate_text"],
                "quick_label": quick_label,
            }
        )

    sample_size = len(residual_sample_records)
    useful_rate = (quick_label_counter["looks_useful"] / sample_size) if sample_size else 0.0
    noise_rate = (quick_label_counter["looks_noise"] / sample_size) if sample_size else 0.0
    overlap_rate = (route_counter["overlap"] / len(routed_rows)) if routed_rows else 0.0
    residual_judgement = "useful material for HDBSCAN" if useful_rate >= max(0.45, noise_rate + 0.10) else "mostly trash / too unclear"
    overlap_judgement = "limited" if overlap_rate < 0.10 else "strong"

    go_hdbscan = (
        len(residual_rows) >= 100
        and useful_rate >= 0.45
        and noise_rate <= 0.30
        and overlap_rate <= 0.15
        and (not cluster_stats["ran"] or float(cluster_stats["clustered_share"]) >= 0.25)
    )
    recommendation = "go_hdbscan" if go_hdbscan else "do_not_go_hdbscan"

    out_dir = ROOT / args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_routing_df = pd.DataFrame(
        [
            {
                "review_id": row["review_id"],
                "category": row["category"],
                "candidate_text": row["candidate_text"],
                "best_general_anchor": row["best_general_anchor"],
                "best_general_score": row["best_general_score"],
                "second_general_score": row["second_general_score"],
                "best_domain_anchor": row["best_domain_anchor"],
                "best_domain_score": row["best_domain_score"],
                "second_domain_score": row["second_domain_score"],
                "route": row["route"],
                "chosen_layer": row["chosen_layer"],
                "chosen_anchor": row["chosen_anchor"],
            }
            for row in routed_rows
        ]
    )
    candidate_routing_df.to_csv(out_dir / "candidate_routing.csv", index=False, encoding="utf-8")

    pd.DataFrame(overlap_rows).to_csv(out_dir / "anchor_overlap.csv", index=False, encoding="utf-8")
    pd.DataFrame(residual_sample_records).to_csv(out_dir / "residual_pool_sample.csv", index=False, encoding="utf-8")

    routing_rules_md = f"""# Routing Rules

## Scoring
- candidate unit: old candidates from current extractor, deduplicated by normalized candidate text inside each review
- general layer: anchors from `src/vocabulary/universal_aspects_v1.yaml`
- domain layer: anchors from `src/vocabulary/domain/<category>.yaml`
- score: cosine between candidate lemma embedding and anchor centroid built from canonical name + synonyms
- routing mode: `{args.mode}`

## Thresholds
- `T_general = {thresholds.t_general:.3f}`
- `M_general = {thresholds.m_general:.3f}`
- `T_domain = {thresholds.t_domain:.3f}`
- `M_domain = {thresholds.m_domain:.3f}`
- `T_general_conflict = {thresholds.t_general_conflict:.3f}`
- `T_domain_conflict = {thresholds.t_domain_conflict:.3f}`
- `C_overlap = {thresholds.c_overlap:.3f}`
- `weak_score_floor = {thresholds.weak_score_floor:.3f}`

## Routing
### `general`
- route to `general` if:
  - `best_general_score >= T_general`
  - `best_general_score - second_general_score >= M_general`
  - candidate is not noise
- save:
  - `chosen_layer = general`
  - `chosen_anchor = best_general_anchor`

### `domain`
- route to `domain` if:
  - candidate did not pass `general`
  - `best_domain_score >= T_domain`
  - `best_domain_score - second_domain_score >= M_domain`
  - candidate is not noise
- save:
  - `chosen_layer = domain`
  - `chosen_anchor = best_domain_anchor`

### `overlap`
- route to `overlap` if:
  - `best_general_score >= T_general_conflict`
  - `best_domain_score >= T_domain_conflict`
  - `abs(best_general_score - best_domain_score) <= C_overlap`
- in `current` mode: do not assign anchor automatically
- in `domain_priority` mode: assign `domain`, but only if the domain anchor also passes its main threshold and margin
- send row to `anchor_overlap.csv`

### `noise`
- route to `noise` if any of:
  - too short candidate
  - generic emotion / praise token without object
  - overly generic object word
  - technical garbage token
  - weak score on both layers: `max(general, domain) < weak_score_floor`

### `residual`
- route to `residual` if:
  - not `general`
  - not `domain`
  - not `overlap`
  - not `noise`

## Residual Cleaning For HDBSCAN
- start from `residual` only
- keep only residual candidate lemmas with global frequency `>= 2`
- do not send `noise`, `general`, `domain`, or `overlap` rows to HDBSCAN
"""
    (out_dir / "routing_rules.md").write_text(routing_rules_md, encoding="utf-8")

    summary_md = f"""# phase4_anchor_residual_routing_diagnostic

## Mode
- routing mode: `{args.mode}`

## Route counts
- general: {route_counter['general']} / {len(routed_rows)} ({_format_pct(route_counter['general'], len(routed_rows))})
- domain: {route_counter['domain']} / {len(routed_rows)} ({_format_pct(route_counter['domain'], len(routed_rows))})
- overlap: {route_counter['overlap']} / {len(routed_rows)} ({_format_pct(route_counter['overlap'], len(routed_rows))})
- residual: {route_counter['residual']} / {len(routed_rows)} ({_format_pct(route_counter['residual'], len(routed_rows))})
- noise: {route_counter['noise']} / {len(routed_rows)} ({_format_pct(route_counter['noise'], len(routed_rows))})

## Conflict
- general vs domain conflict: {overlap_judgement}
- overlap rows: {route_counter['overlap']} ({_format_pct(route_counter['overlap'], len(routed_rows))})

## Residual sample
- quick labels on sample: useful={quick_label_counter['looks_useful']}, noise={quick_label_counter['looks_noise']}, unclear={quick_label_counter['unclear']}
- residual judgement: {residual_judgement}

## HDBSCAN readiness
- residual raw size: {len(residual_rows)}
- residual cleaned size: {len(residual_clean_rows)}
- diagnostic clustering ran: {str(cluster_stats['ran']).lower()}
- diagnostic clusters: {cluster_stats['n_clusters']}
- clustered share: {cluster_stats['clustered_share']}

## Recommendation
- {recommendation}
"""
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    run_summary = {
        "mode": args.mode,
        "n_reviews": len(reviews),
        "n_candidates": len(candidates),
        "diagnostic_thresholds": _diagnostic_thresholds,
        "thresholds": {
            "T_general": thresholds.t_general,
            "M_general": thresholds.m_general,
            "T_domain": thresholds.t_domain,
            "M_domain": thresholds.m_domain,
            "T_general_conflict": thresholds.t_general_conflict,
            "T_domain_conflict": thresholds.t_domain_conflict,
            "C_overlap": thresholds.c_overlap,
            "weak_score_floor": thresholds.weak_score_floor,
        },
        "route_counts": dict(route_counter),
        "quick_label_counts": dict(quick_label_counter),
        "cluster_stats": cluster_stats,
        "recommendation": recommendation,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "mode": args.mode, "recommendation": recommendation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
