from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymorphy3

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import temporary_config_overrides
from scripts import run_phase2_baseline_matching as lexical
from src.discovery.encoder import DiscoveryEncoder
from src.schemas.models import AggregationInput, SentimentPair
from src.stages.aggregation import RatingMathEngine
from src.vocabulary.loader import AspectDefinition

NLI_TEMPERATURE = 0.7
SENTIMENT_RATING_MIN = 1.0
SENTIMENT_RATING_MAX = 5.0
SENTIMENT_RELEVANCE_THRESHOLD = 0.2
SENTIMENT_RELEVANCE_MODE = "p_ent_plus_p_contra_faad23a"
REFERENCE_FAAD23A_REVIEW_MAE = 0.7116
NEGATION_RAW_RATING_MAX = 2.0
NEGATION_REVIEW_RATING_MIN = 4.0
DISCOVERY_PHRASE_TO_CLUSTER_THRESHOLD = 0.5
DISCOVERY_TO_GOLD_MATCH_THRESHOLD = 0.65
PRODUCT_AGGREGATION_MIN_REVIEWS = 3
USE_LEDOIT_WOLF_SHRINKAGE = True
V4_SENTIMENT_ENGINE_PATH = (
    ROOT
    / ".opencode"
    / "artifacts"
    / "sentiment_search_20260425"
    / "sentiment_faad23a_v4_single_hypothesis.py"
)

REFERENCE_DETECTION = {
    "precision": 0.4806,
    "recall": 0.4130,
    "f1": 0.4251,
}

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_MORPH = pymorphy3.MorphAnalyzer()
_LEMMA_CACHE: dict[str, tuple[str, ...]] = {}
_POSITIVE_ABSENCE_LEMMAS = {
    "аромат",
    "брак",
    "вонь",
    "горечь",
    "грязь",
    "дефект",
    "задержка",
    "запах",
    "мусор",
    "насекомое",
    "ожидание",
    "очередь",
    "плесень",
    "повреждение",
    "поломка",
    "проблема",
    "пыль",
    "пятно",
    "ржавчина",
    "скол",
    "таракан",
    "царапина",
    "шум",
}


@dataclass(slots=True)
class ReviewRecord:
    review_id: str
    nm_id: int
    category_id: str
    source: str
    text: str
    rating: float
    true_labels: dict[str, float]
    candidate_lemmas: set[str] = field(default_factory=set)
    candidate_surfaces_by_lemma: dict[str, list[str]] = field(default_factory=dict)
    vocab_aspect_ids: set[str] = field(default_factory=set)
    unmatched_phrases: list[str] = field(default_factory=list)
    discovery_cluster_ids: set[int] = field(default_factory=set)


@dataclass(slots=True)
class ClusterInfo:
    cluster_id: int
    top_phrases: list[str]
    top_phrase_weights: dict[str, int]
    centroid: np.ndarray | None = None
    medoid: str = ""
    gold_matches: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ProductDiscoveryInfo:
    nm_id: int
    category_id: str
    clusters: dict[int, ClusterInfo]


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._fh.write(message + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _parse_true_labels(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in parsed.items():
        key_s = str(key).strip()
        if not key_s:
            continue
        try:
            out[key_s] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _load_reviews(path: Path) -> list[ReviewRecord]:
    df = pd.read_csv(path, dtype={"id": str})
    required = {"nm_id", "id", "rating", "full_text", "true_labels", "source", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    reviews: list[ReviewRecord] = []
    for _, row in df.iterrows():
        text = str(row["full_text"]).strip()
        if not text:
            continue
        reviews.append(
            ReviewRecord(
                review_id=str(row["id"]),
                nm_id=int(row["nm_id"]),
                category_id=str(row["category"]).strip(),
                source=str(row["source"]).strip(),
                text=text,
                rating=float(row["rating"]),
                true_labels=_parse_true_labels(row["true_labels"]),
            )
        )
    return reviews


def _build_hybrid_vocab(
    core_vocab_path: Path,
    domain_dir: Path,
    categories: set[str],
) -> tuple[
    dict[str, list[AspectDefinition]],
    dict[str, dict[str, set[str]]],
    dict[str, dict[str, AspectDefinition]],
]:
    aspects_by_category: dict[str, list[AspectDefinition]] = {}
    term_to_aspects_by_category: dict[str, dict[str, set[str]]] = {}
    aspect_by_id_by_category: dict[str, dict[str, AspectDefinition]] = {}
    for category in sorted(categories):
        domain_path = domain_dir / f"{category}.yaml"
        paths = [core_vocab_path] + ([domain_path] if domain_path.exists() else [])
        aspects = lexical._build_vocabulary(paths)
        term_to_aspects, _ = lexical._term_indexes(aspects)
        aspects_by_category[category] = aspects
        term_to_aspects_by_category[category] = term_to_aspects
        aspect_by_id_by_category[category] = {aspect.id: aspect for aspect in aspects}
    return aspects_by_category, term_to_aspects_by_category, aspect_by_id_by_category


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (matrix / norms).astype(np.float32, copy=False)


def _encode_cached(
    encoder: DiscoveryEncoder,
    texts: list[str],
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    unique = sorted({str(text).strip() for text in texts if str(text).strip() and str(text).strip() not in cache})
    if unique:
        encoded = _l2_normalize(encoder.encode(unique))
        for text, vector in zip(unique, encoded, strict=True):
            cache[text] = vector.astype(np.float32, copy=False)
    return {text: cache[text] for text in texts if text in cache}


def _load_discovery(
    discovery_dir: Path,
    encoder: DiscoveryEncoder,
    cache: dict[str, np.ndarray],
) -> dict[int, ProductDiscoveryInfo]:
    by_product: dict[int, ProductDiscoveryInfo] = {}
    for path in sorted(discovery_dir.glob("product_*_filtered.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        nm_id = int(payload["nm_id"])
        category_id = str(payload["category_id"])
        clusters: dict[int, ClusterInfo] = {}
        for item in payload.get("cluster_summaries", []):
            top_pairs = item.get("top_phrases", [])
            top_phrases = [str(pair[0]).strip() for pair in top_pairs if pair and str(pair[0]).strip()]
            if not top_phrases:
                continue
            vectors_by_phrase = _encode_cached(encoder, top_phrases, cache)
            matrix = np.vstack([vectors_by_phrase[phrase] for phrase in top_phrases])
            centroid = _l2_normalize(matrix.mean(axis=0, keepdims=True))[0]
            sim = matrix @ matrix.T
            medoid_index = int(np.argmax(sim.mean(axis=1)))
            clusters[int(item["cluster_id"])] = ClusterInfo(
                cluster_id=int(item["cluster_id"]),
                top_phrases=top_phrases,
                top_phrase_weights={str(pair[0]): int(pair[1]) for pair in top_pairs if pair},
                centroid=centroid,
                medoid=top_phrases[medoid_index],
            )
        by_product[nm_id] = ProductDiscoveryInfo(
            nm_id=nm_id,
            category_id=category_id,
            clusters=clusters,
        )
    return by_product


def _extract_and_match_reviews(
    reviews: list[ReviewRecord],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
    discovery_by_product: dict[int, ProductDiscoveryInfo],
    encoder: DiscoveryEncoder,
    cache: dict[str, np.ndarray],
    logger: TeeLogger,
) -> None:
    extractor = lexical.CandidateExtractor(ngram_range=(1, 2), min_word_length=3)
    extractor.dependency_filter_enabled = False
    reviews_by_product: dict[int, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        reviews_by_product[review.nm_id].append(review)

    total_started = time.perf_counter()
    product_durations: list[float] = []
    products = sorted(reviews_by_product)
    for index, nm_id in enumerate(products, start=1):
        product_started = time.perf_counter()
        product_reviews = reviews_by_product[nm_id]
        category_id = product_reviews[0].category_id
        term_to_aspects = term_to_aspects_by_category[category_id]
        discovery = discovery_by_product.get(nm_id)
        cluster_ids = sorted(discovery.clusters) if discovery else []
        cluster_matrix = (
            np.vstack([discovery.clusters[cid].centroid for cid in cluster_ids]).astype(np.float32)
            if discovery and cluster_ids
            else np.empty((0, encoder.embedding_dim), dtype=np.float32)
        )

        for r_index, review in enumerate(product_reviews, start=1):
            candidates = extractor.extract(review.text)
            surfaces_by_lemma: dict[str, list[str]] = defaultdict(list)
            for candidate in candidates:
                lemma = lexical._normalize(candidate.span)
                if lemma:
                    surfaces_by_lemma[lemma].append(candidate.span)
            review.candidate_surfaces_by_lemma = dict(surfaces_by_lemma)
            review.candidate_lemmas = set(surfaces_by_lemma)

            matched_terms = lexical._match_terms(review.candidate_lemmas, term_to_aspects, "lexical_only")
            vocab_ids: set[str] = set()
            for term in matched_terms:
                vocab_ids.update(term_to_aspects[term])
            review.vocab_aspect_ids = vocab_ids

            unmatched_lemmas = sorted(review.candidate_lemmas - set(matched_terms))
            unmatched_phrases = [
                review.candidate_surfaces_by_lemma[lemma][0]
                for lemma in unmatched_lemmas
                if review.candidate_surfaces_by_lemma.get(lemma)
            ]
            review.unmatched_phrases = unmatched_phrases

            if unmatched_phrases and cluster_matrix.size:
                vectors = _encode_cached(encoder, unmatched_phrases, cache)
                for phrase in unmatched_phrases:
                    vector = vectors.get(phrase)
                    if vector is None:
                        continue
                    scores = cluster_matrix @ vector
                    best_idx = int(np.argmax(scores))
                    best_score = float(scores[best_idx])
                    if best_score >= DISCOVERY_PHRASE_TO_CLUSTER_THRESHOLD:
                        review.discovery_cluster_ids.add(int(cluster_ids[best_idx]))

            elapsed_product = time.perf_counter() - product_started
            avg_review = elapsed_product / max(r_index, 1)
            remain_product = avg_review * (len(product_reviews) - r_index)
            elapsed_total = time.perf_counter() - total_started
            done_products_fraction = (index - 1 + r_index / len(product_reviews)) / len(products)
            eta_total = elapsed_total * (1.0 / max(done_products_fraction, 1e-9) - 1.0)
            if r_index == 1 or r_index == len(product_reviews) or r_index % 25 == 0:
                logger.log(
                    f"[detect] product {index}/{len(products)} nm_id={nm_id} "
                    f"review {r_index}/{len(product_reviews)} "
                    f"product_elapsed={elapsed_product:.1f}s "
                    f"product_eta={remain_product:.1f}s total_eta={eta_total:.1f}s"
                )

        product_durations.append(time.perf_counter() - product_started)


def _compute_discovery_gold_matches(
    reviews: list[ReviewRecord],
    discovery_by_product: dict[int, ProductDiscoveryInfo],
    encoder: DiscoveryEncoder,
    cache: dict[str, np.ndarray],
) -> None:
    gold_by_product: dict[int, set[str]] = defaultdict(set)
    for review in reviews:
        gold_by_product[review.nm_id].update(review.true_labels)

    for nm_id, discovery in discovery_by_product.items():
        gold_labels = sorted(gold_by_product.get(nm_id, set()))
        if not gold_labels:
            continue
        gold_vectors = _encode_cached(encoder, gold_labels, cache)
        for cluster in discovery.clusters.values():
            if not cluster.top_phrases:
                continue
            phrase_vectors = _encode_cached(encoder, cluster.top_phrases, cache)
            phrase_matrix = np.vstack([phrase_vectors[p] for p in cluster.top_phrases])
            raw_scores: dict[str, float] = {}
            for gold_label in gold_labels:
                gold_vec = gold_vectors.get(gold_label)
                if gold_vec is None:
                    continue
                score = float(np.max(phrase_matrix @ gold_vec))
                if score >= DISCOVERY_TO_GOLD_MATCH_THRESHOLD:
                    raw_scores[gold_label] = score
            cluster.gold_matches = raw_scores

        best_by_gold: dict[str, tuple[int, float]] = {}
        for cluster in discovery.clusters.values():
            for gold_label, score in cluster.gold_matches.items():
                if gold_label not in best_by_gold or score > best_by_gold[gold_label][1]:
                    best_by_gold[gold_label] = (cluster.cluster_id, score)
        for cluster in discovery.clusters.values():
            cluster.gold_matches = {
                gold_label: score
                for gold_label, score in cluster.gold_matches.items()
                if best_by_gold.get(gold_label, (-1, -1.0))[0] == cluster.cluster_id
            }


def _load_v4_sentiment_engine_class() -> Any:
    if not V4_SENTIMENT_ENGINE_PATH.exists():
        raise FileNotFoundError(f"v4 sentiment engine copy not found: {V4_SENTIMENT_ENGINE_PATH}")
    spec = importlib.util.spec_from_file_location("sentiment_faad23a_v4", V4_SENTIMENT_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load v4 sentiment engine from {V4_SENTIMENT_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SentimentEngine


def _lemmas(text: str) -> tuple[str, ...]:
    cached = _LEMMA_CACHE.get(text)
    if cached is not None:
        return cached
    out: list[str] = []
    for token in _WORD_RE.findall(text.lower()):
        if not any(ch.isalpha() for ch in token):
            continue
        out.append(_MORPH.parse(token)[0].normal_form)
    result = tuple(out)
    _LEMMA_CACHE[text] = result
    return result


def _aspect_lemmas(terms: list[str]) -> set[str]:
    lemmas: set[str] = set()
    for term in terms:
        for lemma in _lemmas(term):
            if len(lemma) > 2:
                lemmas.add(lemma)
    return lemmas


def _negation_match(text: str, aspect_lemmas: set[str]) -> tuple[bool, str, str]:
    absence_lemmas = aspect_lemmas & _POSITIVE_ABSENCE_LEMMAS
    if not absence_lemmas:
        return False, "", ""

    tokens = _lemmas(text)
    n_tokens = len(tokens)
    for index, token in enumerate(tokens):
        prev_window = set(tokens[max(0, index - 4) : index])
        next_window = set(tokens[index + 1 : index + 5])

        if token in {"нет", "без"} and next_window & absence_lemmas:
            hit = sorted(next_window & absence_lemmas)[0]
            return True, token, hit
        if token == "нет" and prev_window & absence_lemmas:
            hit = sorted(prev_window & absence_lemmas)[0]
            return True, "reverse нет", hit
        if token == "не" and index + 1 < n_tokens and tokens[index + 1] == "быть":
            after = set(tokens[index + 2 : index + 6])
            hit_set = (after | prev_window) & absence_lemmas
            if hit_set:
                return True, "не было", sorted(hit_set)[0]
        if token == "не" and index + 1 < n_tokens and tokens[index + 1] == "иметь":
            after = set(tokens[index + 2 : index + 7])
            if after & absence_lemmas:
                return True, "не имеет", sorted(after & absence_lemmas)[0]
        if token == "отсутствовать" and (next_window | prev_window) & absence_lemmas:
            hit = sorted((next_window | prev_window) & absence_lemmas)[0]
            return True, "отсутствует", hit
        if token == "никакой" and next_window & absence_lemmas:
            hit = sorted(next_window & absence_lemmas)[0]
            return True, "никакой", hit

    return False, "", ""


def _sentiment_scores(pairs: list[SentimentPair], logger: TeeLogger) -> dict[tuple[str, str], dict[str, float]]:
    if not pairs:
        return {}
    overrides = {
        "sentiment": {
            "temperature": NLI_TEMPERATURE,
            "hypothesis_template_pos": "{aspect} — это хорошо",
            "relevance_threshold": SENTIMENT_RELEVANCE_THRESHOLD,
        }
    }
    logger.log(
        f"[sentiment] pairs={len(pairs)} nli_calls={len(pairs)} "
        f"T={NLI_TEMPERATURE} relevance=P_ent+P_contra>={SENTIMENT_RELEVANCE_THRESHOLD}"
    )
    started = time.perf_counter()
    with temporary_config_overrides(overrides):
        engine_cls = _load_v4_sentiment_engine_class()
        engine = engine_cls()
        tuple_pairs = [
            (pair.review_id, pair.sentence, pair.aspect, pair.nli_label, pair.weight)
            for pair in pairs
        ]
        results = engine.batch_analyze(tuple_pairs)

    out: dict[tuple[str, str], dict[str, float]] = {}
    skipped = 0
    for result in results:
        p_ent = float(result.p_ent_pos)
        p_contra = float(result.p_ent_neg)
        relevance = p_ent + p_contra
        if relevance < SENTIMENT_RELEVANCE_THRESHOLD:
            skipped += 1
            continue
        rating = float(np.clip(result.score, SENTIMENT_RATING_MIN, SENTIMENT_RATING_MAX))
        out[(str(result.review_id), str(result.aspect))] = {
            "rating": rating,
            "raw_rating": rating,
            "p_ent_pos": p_ent,
            "p_ent_neg": p_contra,
            "p_ent_plus_contra": relevance,
            "polarity": rating - 3.0,
            "raw_polarity": rating - 3.0,
            "negation_corrected": False,
            "negation_pattern": "",
            "negation_hit_lemma": "",
        }
    logger.log(
        f"[sentiment] kept={len(out)} skipped={skipped} "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )
    return out


def _apply_negation_corrections(
    sentiment_by_pair: dict[tuple[str, str], dict[str, float]],
    reviews: list[ReviewRecord],
    aspect_by_id_by_category: dict[str, dict[str, AspectDefinition]],
    discovery_by_product: dict[int, ProductDiscoveryInfo],
    logger: TeeLogger,
) -> dict[str, Any]:
    review_by_id = {review.review_id: review for review in reviews}
    per_category: dict[str, int] = defaultdict(int)
    applied = 0
    eligible_low_high = 0

    for (review_id, aspect_key), scores in sentiment_by_pair.items():
        review = review_by_id.get(review_id)
        if review is None:
            continue
        raw_rating = float(scores.get("raw_rating", scores["rating"]))
        if raw_rating > NEGATION_RAW_RATING_MAX or review.rating < NEGATION_REVIEW_RATING_MIN:
            continue
        eligible_low_high += 1

        terms: list[str] = []
        if aspect_key.startswith("vocab::"):
            aspect_id = aspect_key.split("::", 1)[1]
            aspect = aspect_by_id_by_category.get(review.category_id, {}).get(aspect_id)
            if aspect is not None:
                terms = [aspect.canonical_name, *aspect.synonyms]
        elif aspect_key.startswith("discovery::"):
            parts = aspect_key.split("::")
            if len(parts) == 3:
                discovery = discovery_by_product.get(int(parts[1]))
                cluster = discovery.clusters.get(int(parts[2])) if discovery else None
                if cluster is not None:
                    terms = [cluster.medoid]
        if not terms:
            continue

        matched, pattern, hit_lemma = _negation_match(review.text, _aspect_lemmas(terms))
        if not matched:
            continue

        corrected = float(np.clip(6.0 - raw_rating, SENTIMENT_RATING_MIN, SENTIMENT_RATING_MAX))
        scores["rating"] = corrected
        scores["polarity"] = corrected - 3.0
        scores["negation_corrected"] = True
        scores["negation_pattern"] = pattern
        scores["negation_hit_lemma"] = hit_lemma
        applied += 1
        per_category[review.category_id] += 1

    total = len(sentiment_by_pair)
    stats = {
        "total_predictions": total,
        "eligible_low_high_predictions": eligible_low_high,
        "corrections_applied": applied,
        "correction_rate": applied / total if total else 0.0,
        "inversion_rate": applied / eligible_low_high if eligible_low_high else 0.0,
        "per_category": dict(sorted(per_category.items())),
    }
    logger.log(
        "[negation] total_predictions={total} eligible_low_high={eligible} "
        "corrections={applied} inversion_rate={inversion:.2%}".format(
            total=total,
            eligible=eligible_low_high,
            applied=applied,
            inversion=stats["inversion_rate"],
        )
    )
    return stats


def _build_sentiment_pairs(
    reviews: list[ReviewRecord],
    aspect_by_id_by_category: dict[str, dict[str, AspectDefinition]],
    discovery_by_product: dict[int, ProductDiscoveryInfo],
) -> list[SentimentPair]:
    pairs: list[SentimentPair] = []
    seen: set[tuple[str, str]] = set()
    for review in reviews:
        aspect_by_id = aspect_by_id_by_category[review.category_id]
        for aspect_id in sorted(review.vocab_aspect_ids):
            key = (review.review_id, f"vocab::{aspect_id}")
            if key in seen:
                continue
            seen.add(key)
            canonical = aspect_by_id.get(aspect_id).canonical_name if aspect_id in aspect_by_id else aspect_id
            pairs.append(
                SentimentPair(
                    review_id=review.review_id,
                    sentence=review.text,
                    aspect=f"vocab::{aspect_id}",
                    nli_label=canonical,
                    weight=1.0,
                )
            )
        discovery = discovery_by_product.get(review.nm_id)
        if not discovery:
            continue
        for cluster_id in sorted(review.discovery_cluster_ids):
            cluster = discovery.clusters.get(cluster_id)
            if cluster is None:
                continue
            key = (review.review_id, f"discovery::{review.nm_id}::{cluster_id}")
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                SentimentPair(
                    review_id=review.review_id,
                    sentence=review.text,
                    aspect=f"discovery::{review.nm_id}::{cluster_id}",
                    nli_label=cluster.medoid,
                    weight=1.0,
                )
            )
    return pairs


def _prf(pred: set[str], true: set[str]) -> tuple[float, float, float]:
    if not pred and not true:
        return 1.0, 1.0, 1.0
    if not pred or not true:
        return 0.0, 0.0, 0.0
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _gold_vocab_ids(review: ReviewRecord, term_to_aspects: dict[str, set[str]]) -> set[str]:
    out: set[str] = set()
    for gold_label in review.true_labels:
        out.update(term_to_aspects.get(lexical._normalize(gold_label), set()))
    return out


def _gold_unmapped_labels(review: ReviewRecord, term_to_aspects: dict[str, set[str]]) -> set[str]:
    out: set[str] = set()
    for gold_label in review.true_labels:
        if not term_to_aspects.get(lexical._normalize(gold_label), set()):
            out.add(gold_label)
    return out


def _review_metric_rows(
    reviews: list[ReviewRecord],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
    discovery_by_product: dict[int, ProductDiscoveryInfo],
    sentiment_by_pair: dict[tuple[str, str], dict[str, float]],
    include_discovery: bool,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hard_case_rows: list[dict[str, Any]] = []
    review_maes: list[float] = []
    review_round_maes: list[float] = []
    discovery_errors: list[float] = []
    vocab_errors: list[float] = []

    for review in reviews:
        term_to_aspects = term_to_aspects_by_category[review.category_id]
        true_vocab = _gold_vocab_ids(review, term_to_aspects)
        pred_vocab = set(review.vocab_aspect_ids)
        pred = {f"vocab::{aspect_id}" for aspect_id in pred_vocab}
        true = {f"vocab::{aspect_id}" for aspect_id in true_vocab}

        if include_discovery:
            true_unmapped = _gold_unmapped_labels(review, term_to_aspects)
            true.update(f"gold::{label}" for label in true_unmapped)
            discovery = discovery_by_product.get(review.nm_id)
            if discovery:
                for cluster_id in review.discovery_cluster_ids:
                    cluster = discovery.clusters.get(cluster_id)
                    if cluster is None:
                        continue
                    for gold_label in cluster.gold_matches:
                        if gold_label in true_unmapped:
                            pred.add(f"gold::{gold_label}")

        precision, recall, f1 = _prf(pred, true)
        rows.append(
            {
                "review_id": review.review_id,
                "nm_id": review.nm_id,
                "category_id": review.category_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_pred": len(pred),
                "n_true": len(true),
            }
        )

        errors: list[float] = []
        round_errors: list[float] = []
        for gold_label, gold_rating in review.true_labels.items():
            mapped_ids = sorted(term_to_aspects.get(lexical._normalize(gold_label), set()))
            predicted: list[tuple[str, str, float, float, bool]] = []
            for aspect_id in mapped_ids:
                key = (review.review_id, f"vocab::{aspect_id}")
                if key in sentiment_by_pair:
                    predicted.append(
                        (
                            "vocab",
                            aspect_id,
                            sentiment_by_pair[key]["rating"],
                            sentiment_by_pair[key].get("raw_rating", sentiment_by_pair[key]["rating"]),
                            bool(sentiment_by_pair[key].get("negation_corrected", False)),
                        )
                    )
            if include_discovery and not mapped_ids:
                discovery = discovery_by_product.get(review.nm_id)
                if discovery:
                    for cluster_id in review.discovery_cluster_ids:
                        cluster = discovery.clusters.get(cluster_id)
                        if cluster is None or gold_label not in cluster.gold_matches:
                            continue
                        key = (review.review_id, f"discovery::{review.nm_id}::{cluster_id}")
                        if key in sentiment_by_pair:
                            predicted.append(
                                (
                                    "discovery",
                                    str(cluster_id),
                                    sentiment_by_pair[key]["rating"],
                                    sentiment_by_pair[key].get("raw_rating", sentiment_by_pair[key]["rating"]),
                                    bool(sentiment_by_pair[key].get("negation_corrected", False)),
                                )
                            )
            if not predicted:
                continue
            pred_rating = float(np.mean([item[2] for item in predicted]))
            raw_pred_rating = float(np.mean([item[3] for item in predicted]))
            error = abs(pred_rating - float(gold_rating))
            raw_error = abs(raw_pred_rating - float(gold_rating))
            errors.append(error)
            round_errors.append(abs(round(pred_rating) - float(gold_rating)))
            if any(item[0] == "discovery" for item in predicted):
                discovery_errors.append(error)
            else:
                vocab_errors.append(error)
            hard_case_rows.append(
                {
                    "review_id": review.review_id,
                    "nm_id": review.nm_id,
                    "category_id": review.category_id,
                    "aspect": gold_label,
                    "aspect_source": "discovery" if any(item[0] == "discovery" for item in predicted) else "vocab",
                    "gold_rating": float(gold_rating),
                    "predicted_rating": round(pred_rating, 4),
                    "raw_predicted_rating": round(raw_pred_rating, 4),
                    "abs_error": round(error, 4),
                    "raw_abs_error": round(raw_error, 4),
                    "negation_correction_applied": any(item[4] for item in predicted),
                    "review_rating": review.rating,
                    "review_text": review.text,
                }
            )
        if errors:
            review_maes.append(float(np.mean(errors)))
            review_round_maes.append(float(np.mean(round_errors)))

    df = pd.DataFrame(rows)
    metrics = {
        "detection_precision": float(df["precision"].mean()) if not df.empty else 0.0,
        "detection_recall": float(df["recall"].mean()) if not df.empty else 0.0,
        "detection_f1": float(df["f1"].mean()) if not df.empty else 0.0,
        "sentiment_mae_review": float(np.mean(review_maes)) if review_maes else np.nan,
        "sentiment_mae_review_round": float(np.mean(review_round_maes)) if review_round_maes else np.nan,
        "sentiment_mae_vocab_pairs": float(np.mean(vocab_errors)) if vocab_errors else np.nan,
        "sentiment_mae_discovery_pairs": float(np.mean(discovery_errors)) if discovery_errors else np.nan,
        "n_sentiment_review_matches": int(len(review_maes)),
        "n_discovery_sentiment_pairs": int(len(discovery_errors)),
    }
    return df, metrics, hard_case_rows


def _aggregate_product_scores(
    reviews: list[ReviewRecord],
    sentiment_by_pair: dict[tuple[str, str], dict[str, float]],
    aspect_by_id_by_category: dict[str, dict[str, AspectDefinition]],
    discovery_by_product: dict[int, ProductDiscoveryInfo],
) -> dict[int, dict[str, Any]]:
    by_product: dict[int, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        by_product[review.nm_id].append(review)

    math_engine = RatingMathEngine()
    aggregated: dict[int, dict[str, Any]] = {}
    for nm_id, product_reviews in sorted(by_product.items()):
        inputs: list[AggregationInput] = []
        for review in product_reviews:
            aspects: dict[str, float] = {}
            aspect_by_id = aspect_by_id_by_category[review.category_id]
            for aspect_id in sorted(review.vocab_aspect_ids):
                key = (review.review_id, f"vocab::{aspect_id}")
                if key in sentiment_by_pair:
                    name = aspect_by_id[aspect_id].canonical_name if aspect_id in aspect_by_id else aspect_id
                    aspects[f"vocab::{aspect_id}::{name}"] = float(sentiment_by_pair[key]["rating"])
            discovery = discovery_by_product.get(nm_id)
            if discovery:
                for cluster_id in sorted(review.discovery_cluster_ids):
                    cluster = discovery.clusters.get(cluster_id)
                    key = (review.review_id, f"discovery::{nm_id}::{cluster_id}")
                    if cluster is not None and key in sentiment_by_pair:
                        aspects[f"discovery::{cluster_id}::{cluster.medoid}"] = float(sentiment_by_pair[key]["rating"])
            if aspects:
                inputs.append(
                    AggregationInput(
                        review_id=review.review_id,
                        aspects=aspects,
                        fraud_weight=1.0,
                        date=None,
                    )
                )
        result = math_engine.aggregate(inputs)
        aggregated[nm_id] = {
            "raw": result,
            "scores": {name: float(score.score) for name, score in result.aspects.items()},
        }
    return aggregated


def _product_metric_rows(
    reviews: list[ReviewRecord],
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
    aspect_by_id_by_category: dict[str, dict[str, AspectDefinition]],
    discovery_by_product: dict[int, ProductDiscoveryInfo],
    aggregated: dict[int, dict[str, Any]],
    sentiment_by_pair: dict[tuple[str, str], dict[str, float]],
    include_discovery: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    by_product: dict[int, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        by_product[review.nm_id].append(review)

    rows: list[dict[str, Any]] = []
    for nm_id, product_reviews in sorted(by_product.items()):
        category_id = product_reviews[0].category_id
        term_to_aspects = term_to_aspects_by_category[category_id]
        aspect_by_id = aspect_by_id_by_category[category_id]
        product_scores = aggregated.get(nm_id, {}).get("scores", {})
        gold_scores_by_label: dict[str, list[float]] = defaultdict(list)
        for review in product_reviews:
            for label, score in review.true_labels.items():
                gold_scores_by_label[label].append(float(score))

        for gold_label, gold_scores in sorted(gold_scores_by_label.items()):
            mapped_ids = sorted(term_to_aspects.get(lexical._normalize(gold_label), set()))
            pred_scores: list[tuple[str, str, float, bool]] = []
            for aspect_id in mapped_ids:
                canonical = aspect_by_id[aspect_id].canonical_name if aspect_id in aspect_by_id else aspect_id
                key = f"vocab::{aspect_id}::{canonical}"
                if key in product_scores:
                    corrected = any(
                        sentiment_by_pair.get((review.review_id, f"vocab::{aspect_id}"), {}).get(
                            "negation_corrected", False
                        )
                        for review in product_reviews
                    )
                    pred_scores.append(("vocab", canonical, float(product_scores[key]), corrected))
            if include_discovery and not mapped_ids:
                discovery = discovery_by_product.get(nm_id)
                if discovery:
                    for cluster in discovery.clusters.values():
                        if gold_label not in cluster.gold_matches:
                            continue
                        key = f"discovery::{cluster.cluster_id}::{cluster.medoid}"
                        if key in product_scores:
                            corrected = any(
                                sentiment_by_pair.get(
                                    (review.review_id, f"discovery::{nm_id}::{cluster.cluster_id}"),
                                    {},
                                ).get("negation_corrected", False)
                                for review in product_reviews
                            )
                            pred_scores.append(("discovery", cluster.medoid, float(product_scores[key]), corrected))
            if not pred_scores:
                continue
            predicted = float(np.mean([item[2] for item in pred_scores]))
            gold = float(np.mean(gold_scores))
            rows.append(
                {
                    "nm_id": nm_id,
                    "category_id": category_id,
                    "aspect_source": "discovery" if any(item[0] == "discovery" for item in pred_scores) else "vocab",
                    "aspect_name": gold_label,
                    "n_reviews_with_aspect": len(gold_scores),
                    "predicted_rating": round(predicted, 4),
                    "gold_rating": round(gold, 4),
                    "abs_error": round(abs(predicted - gold), 4),
                    "negation_correction_applied": any(item[3] for item in pred_scores),
                }
            )

    df = pd.DataFrame(rows)
    n3 = df[df["n_reviews_with_aspect"] >= PRODUCT_AGGREGATION_MIN_REVIEWS] if not df.empty else df
    metrics = {
        "product_mae_n3": float(n3["abs_error"].mean()) if n3 is not None and not n3.empty else np.nan,
        "n_aspects_matched": int(len(df)),
        "n_aspects_matched_n3": int(len(n3)) if n3 is not None else 0,
    }
    return df, metrics


def _star_metrics(reviews: list[ReviewRecord]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []
    review_maes: list[float] = []
    review_round_maes: list[float] = []
    by_product_label: dict[tuple[int, str, str], list[tuple[float, float]]] = defaultdict(list)
    for review in reviews:
        errors: list[float] = []
        round_errors: list[float] = []
        for label, gold in review.true_labels.items():
            error = abs(float(review.rating) - float(gold))
            errors.append(error)
            round_errors.append(abs(round(float(review.rating)) - float(gold)))
            by_product_label[(review.nm_id, review.category_id, label)].append((float(review.rating), float(gold)))
            review_rows.append(
                {
                    "review_id": review.review_id,
                    "nm_id": review.nm_id,
                    "category_id": review.category_id,
                    "aspect": label,
                    "predicted_rating": float(review.rating),
                    "gold_rating": float(gold),
                    "abs_error": error,
                }
            )
        if errors:
            review_maes.append(float(np.mean(errors)))
            review_round_maes.append(float(np.mean(round_errors)))

    product_rows: list[dict[str, Any]] = []
    for (nm_id, category_id, label), pairs in sorted(by_product_label.items()):
        pred = float(np.mean([p for p, _ in pairs]))
        gold = float(np.mean([g for _, g in pairs]))
        product_rows.append(
            {
                "nm_id": nm_id,
                "category_id": category_id,
                "aspect_source": "star",
                "aspect_name": label,
                "n_reviews_with_aspect": len(pairs),
                "predicted_rating": round(pred, 4),
                "gold_rating": round(gold, 4),
                "abs_error": round(abs(pred - gold), 4),
                "negation_correction_applied": False,
            }
        )
    review_df = pd.DataFrame(review_rows)
    product_df = pd.DataFrame(product_rows)
    n3 = product_df[product_df["n_reviews_with_aspect"] >= PRODUCT_AGGREGATION_MIN_REVIEWS] if not product_df.empty else product_df
    review_metrics = {
        "sentiment_mae_review": float(np.mean(review_maes)) if review_maes else np.nan,
        "sentiment_mae_review_round": float(np.mean(review_round_maes)) if review_round_maes else np.nan,
    }
    product_metrics = {
        "product_mae_n3": float(n3["abs_error"].mean()) if not n3.empty else np.nan,
        "n_aspects_matched": int(len(product_df)),
    }
    return review_df, review_metrics, product_df, product_metrics


def _per_product_metrics(
    reviews: list[ReviewRecord],
    review_df: pd.DataFrame,
    product_df: pd.DataFrame,
    review_hard_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    n_reviews = {nm_id: len(list(group)) for nm_id, group in _group_reviews(reviews).items()}
    cats = {review.nm_id: review.category_id for review in reviews}
    sentiment_df = pd.DataFrame(review_hard_rows)
    rows: list[dict[str, Any]] = []
    for nm_id in sorted(cats):
        rsub = review_df[review_df["nm_id"] == nm_id] if not review_df.empty else pd.DataFrame()
        ssub = sentiment_df[sentiment_df["nm_id"] == nm_id] if not sentiment_df.empty else pd.DataFrame()
        psub = product_df[product_df["nm_id"] == nm_id] if not product_df.empty else pd.DataFrame()
        psub_n3 = psub[psub["n_reviews_with_aspect"] >= PRODUCT_AGGREGATION_MIN_REVIEWS] if not psub.empty else psub
        rows.append(
            {
                "nm_id": nm_id,
                "category_id": cats[nm_id],
                "n_reviews": n_reviews[nm_id],
                "detection_precision": float(rsub["precision"].mean()) if not rsub.empty else np.nan,
                "detection_recall": float(rsub["recall"].mean()) if not rsub.empty else np.nan,
                "sentiment_mae_review": float(ssub["abs_error"].mean()) if not ssub.empty else np.nan,
                "sentiment_mae_review_round": float(abs(ssub["predicted_rating"].round() - ssub["gold_rating"]).mean()) if not ssub.empty else np.nan,
                "product_mae_n3": float(psub_n3["abs_error"].mean()) if not psub_n3.empty else np.nan,
                "n_aspects_matched": int(len(psub)),
            }
        )
    return pd.DataFrame(rows)


def _group_reviews(reviews: list[ReviewRecord]) -> dict[int, list[ReviewRecord]]:
    out: dict[int, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        out[review.nm_id].append(review)
    return dict(out)


def _write_predictions(
    out_dir: Path,
    reviews: list[ReviewRecord],
    aspect_by_id_by_category: dict[str, dict[str, AspectDefinition]],
    discovery_by_product: dict[int, ProductDiscoveryInfo],
    sentiment_by_pair: dict[tuple[str, str], dict[str, float]],
    aggregated: dict[int, dict[str, Any]],
) -> None:
    for nm_id, product_reviews in _group_reviews(reviews).items():
        category_id = product_reviews[0].category_id
        aspect_by_id = aspect_by_id_by_category[category_id]
        discovery = discovery_by_product.get(nm_id)
        review_items: list[dict[str, Any]] = []
        for review in product_reviews:
            vocab_items: list[dict[str, Any]] = []
            for aspect_id in sorted(review.vocab_aspect_ids):
                key = (review.review_id, f"vocab::{aspect_id}")
                if key not in sentiment_by_pair:
                    continue
                aspect = aspect_by_id.get(aspect_id)
                vocab_items.append(
                    {
                        "aspect_id": aspect_id,
                        "aspect": aspect.canonical_name if aspect else aspect_id,
                        "rating": round(float(sentiment_by_pair[key]["rating"]), 4),
                        "raw_rating": round(float(sentiment_by_pair[key].get("raw_rating", sentiment_by_pair[key]["rating"])), 4),
                        "polarity": round(float(sentiment_by_pair[key]["polarity"]), 4),
                        "negation_corrected": bool(sentiment_by_pair[key].get("negation_corrected", False)),
                        "negation_pattern": sentiment_by_pair[key].get("negation_pattern", ""),
                        "negation_hit_lemma": sentiment_by_pair[key].get("negation_hit_lemma", ""),
                    }
                )
            discovery_items: list[dict[str, Any]] = []
            if discovery:
                for cluster_id in sorted(review.discovery_cluster_ids):
                    cluster = discovery.clusters.get(cluster_id)
                    key = (review.review_id, f"discovery::{nm_id}::{cluster_id}")
                    if cluster is None or key not in sentiment_by_pair:
                        continue
                    discovery_items.append(
                        {
                            "cluster_id": cluster_id,
                            "medoid": cluster.medoid,
                            "rating": round(float(sentiment_by_pair[key]["rating"]), 4),
                            "raw_rating": round(float(sentiment_by_pair[key].get("raw_rating", sentiment_by_pair[key]["rating"])), 4),
                            "polarity": round(float(sentiment_by_pair[key]["polarity"]), 4),
                            "negation_corrected": bool(sentiment_by_pair[key].get("negation_corrected", False)),
                            "negation_pattern": sentiment_by_pair[key].get("negation_pattern", ""),
                            "negation_hit_lemma": sentiment_by_pair[key].get("negation_hit_lemma", ""),
                            "gold_matches": cluster.gold_matches,
                        }
                    )
            review_items.append(
                {
                    "review_id": review.review_id,
                    "rating": review.rating,
                    "vocabulary_aspects": vocab_items,
                    "discovery_aspects": discovery_items,
                }
            )
        agg_scores = aggregated.get(nm_id, {}).get("scores", {})
        payload = {
            "nm_id": nm_id,
            "category": category_id,
            "reviews": review_items,
            "product_aggregated": {
                "vocabulary": {
                    key.split("::", 2)[2]: value
                    for key, value in agg_scores.items()
                    if key.startswith("vocab::")
                },
                "discovery": {
                    key.split("::", 2)[2]: value
                    for key, value in agg_scores.items()
                    if key.startswith("discovery::")
                },
            },
        }
        (out_dir / f"predictions_{nm_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    try:
        if np.isnan(float(value)):
            return "N/A"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.4f}"


def _write_summary(
    out_dir: Path,
    aggregate_a: dict[str, Any],
    aggregate_b: dict[str, Any],
    aggregate_c_review: dict[str, Any],
    aggregate_c_product: dict[str, Any],
    per_product_a: pd.DataFrame,
    per_product_b: pd.DataFrame,
    hard_cases: pd.DataFrame,
    negation_stats: dict[str, Any],
) -> None:
    discovery_added = aggregate_b.get("n_aspects_matched", 0) - aggregate_a.get("n_aspects_matched", 0)
    vocab_mae = aggregate_b.get("sentiment_mae_vocab_pairs")
    disc_mae = aggregate_b.get("sentiment_mae_discovery_pairs")
    a_review = aggregate_a.get("sentiment_mae_review")
    star_review = aggregate_c_review.get("sentiment_mae_review")
    b_review = aggregate_b.get("sentiment_mae_review")
    star_product = aggregate_c_product.get("product_mae_n3")
    b_product = aggregate_b.get("product_mae_n3")
    sanity_diff = (
        float(a_review) - REFERENCE_FAAD23A_REVIEW_MAE
        if a_review is not None and not np.isnan(float(a_review))
        else np.nan
    )
    sanity_status = (
        "regression: inspect before accepting"
        if not np.isnan(sanity_diff) and abs(sanity_diff) > 0.05
        else "close to reference"
    )
    sanity_range = (
        "works in checked range"
        if a_review is not None and 0.65 <= float(a_review) <= 0.75
        else "outside checked range"
    )
    consumables_mae = np.nan
    if not per_product_a.empty and "category_id" in per_product_a.columns:
        consumables_rows = per_product_a[per_product_a["category_id"] == "consumables"]
        if not consumables_rows.empty:
            consumables_mae = float(consumables_rows["sentiment_mae_review"].mean())
    correction_count = int(negation_stats.get("corrections_applied", 0))
    correction_target_status = (
        "inside broad target"
        if 50 <= correction_count <= 150
        else "below broad target"
        if correction_count < 50
        else "above broad target"
    )
    lines = [
        "# Final End-to-End Pipeline Results",
        "",
        "## Aggregate metrics across 16 products",
        "",
        "| Metric | Vocab Only | Vocab + Discovery | Star Baseline |",
        "|---|---:|---:|---:|",
        f"| Detection Precision | {_fmt(aggregate_a.get('detection_precision'))} | {_fmt(aggregate_b.get('detection_precision'))} | N/A |",
        f"| Detection Recall | {_fmt(aggregate_a.get('detection_recall'))} | {_fmt(aggregate_b.get('detection_recall'))} | N/A |",
        f"| Detection F1 | {_fmt(aggregate_a.get('detection_f1'))} | {_fmt(aggregate_b.get('detection_f1'))} | N/A |",
        f"| Sentiment MAE (review) | {_fmt(aggregate_a.get('sentiment_mae_review'))} | {_fmt(aggregate_b.get('sentiment_mae_review'))} | {_fmt(star_review)} |",
        f"| Sentiment MAE (round) | {_fmt(aggregate_a.get('sentiment_mae_review_round'))} | {_fmt(aggregate_b.get('sentiment_mae_review_round'))} | {_fmt(aggregate_c_review.get('sentiment_mae_review_round'))} |",
        f"| Product MAE (n>=3) | {_fmt(aggregate_a.get('product_mae_n3'))} | {_fmt(aggregate_b.get('product_mae_n3'))} | {_fmt(star_product)} |",
        "",
        "## Per-category breakdown",
        "",
    ]
    if not per_product_b.empty:
        cat_rows = []
        for category, group in per_product_b.groupby("category_id"):
            cat_rows.append(
                "| {cat} | {p} | {r} | {s} | {pm} |".format(
                    cat=category,
                    p=_fmt(group["detection_precision"].mean()),
                    r=_fmt(group["detection_recall"].mean()),
                    s=_fmt(group["sentiment_mae_review"].mean()),
                    pm=_fmt(group["product_mae_n3"].mean()),
                )
            )
        lines.extend(
            [
                "| Category | Detection P | Detection R | Review MAE | Product MAE n>=3 |",
                "|---|---:|---:|---:|---:|",
                *cat_rows,
                "",
            ]
        )
    lines.extend(
        [
            "## Comparison: did discovery help?",
            "",
            f"Discovery added {int(discovery_added)} matched product-aspect rows.",
            f"Sentiment quality on discovery aspects: MAE = {_fmt(disc_mae)} vs {_fmt(vocab_mae)} on vocabulary aspects.",
            "",
            "## Comparison: did we beat star baseline?",
            "",
            f"On Sentiment MAE (review-level): {'yes' if b_review < star_review else 'no'} by {_fmt(abs((b_review or 0) - (star_review or 0)))}.",
            f"On Product MAE: {'yes' if b_product < star_product else 'no'} by {_fmt(abs((b_product or 0) - (star_product or 0)))}.",
            "",
            "## Sanity check vs old baseline",
            "",
            "Old code (commit faad23a):",
            f"- Vocab-only sentiment MAE review: {_fmt(REFERENCE_FAAD23A_REVIEW_MAE)} (reference)",
            "",
            "New code (this run):",
            f"- Vocab-only sentiment MAE review: {_fmt(a_review)}",
            f"- Difference: {_fmt(a_review)} - {_fmt(REFERENCE_FAAD23A_REVIEW_MAE)} = {_fmt(sanity_diff)}",
            f"- Status: {sanity_status}; {sanity_range}.",
            "",
            "## Negation correction stats",
            "",
            f"Total predictions: {int(negation_stats.get('total_predictions', 0))}",
            "Corrections applied: {n} ({rate})".format(
                n=int(negation_stats.get("corrections_applied", 0)),
                rate=_fmt(float(negation_stats.get("correction_rate", 0.0)) * 100.0) + "%",
            ),
            f"Avg MAE before correction: {_fmt(negation_stats.get('avg_mae_before_correction'))}",
            f"Avg MAE after correction: {_fmt(negation_stats.get('avg_mae_after_correction'))}",
            f"Improvement: {_fmt(negation_stats.get('mae_improvement'))}",
            f"Inversion rate: {_fmt(float(negation_stats.get('inversion_rate', 0.0)) * 100.0)}%",
            f"Correction target status: {correction_target_status}",
            "",
            "Per-category corrections:",
            *[
                f"- {category}: {count} corrections"
                for category, count in sorted(negation_stats.get("per_category", {}).items())
            ],
            "",
            "## Negation sanity check",
            "",
            f"- Vocab-only sentiment MAE: {_fmt(a_review)} (expected 0.72-0.85)",
            f"- Consumables MAE: {_fmt(consumables_mae)} (expected <0.50)",
            f"- Inversion rate: {_fmt(float(negation_stats.get('inversion_rate', 0.0)) * 100.0)}% (expected <12%)",
            f"- Corrections applied: {correction_count} (target 50-150; hard lower check is >=30)",
            "",
            "## Hard cases (10 worst predictions)",
            "",
        ]
    )
    if hard_cases.empty:
        lines.append("_none_")
    else:
        lines.extend(
            [
                "| review_id | nm_id | aspect | source | gold | pred | abs_error |",
                "|---|---:|---|---|---:|---:|---:|",
            ]
        )
        for _, row in hard_cases.head(10).iterrows():
            lines.append(
                f"| {row['review_id']} | {row['nm_id']} | {row['aspect']} | {row['aspect_source']} | "
                f"{_fmt(row['gold_rating'])} | {_fmt(row['predicted_rating'])} | {_fmt(row['abs_error'])} |"
            )
    (out_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_track_csv(path: Path, per_product: pd.DataFrame) -> None:
    columns = [
        "nm_id",
        "category_id",
        "n_reviews",
        "detection_precision",
        "detection_recall",
        "sentiment_mae_review",
        "sentiment_mae_review_round",
        "product_mae_n3",
        "n_aspects_matched",
    ]
    per_product.to_csv(path, index=False, encoding="utf-8", columns=columns)


def _finalize_negation_stats(negation_stats: dict[str, Any], hard_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(negation_stats)
    hard_df = pd.DataFrame(hard_rows)
    if hard_df.empty or "raw_abs_error" not in hard_df.columns:
        out["avg_mae_before_correction"] = np.nan
        out["avg_mae_after_correction"] = np.nan
        out["mae_improvement"] = np.nan
        return out

    before = float(hard_df["raw_abs_error"].mean())
    after = float(hard_df["abs_error"].mean())
    out["avg_mae_before_correction"] = before
    out["avg_mae_after_correction"] = after
    out["mae_improvement"] = before - after
    return out


def run(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / args.out_dir / f"{timestamp}_final_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(out_dir / "run_console.log")
    started = time.perf_counter()
    try:
        logger.log(f"[start] final_e2e out_dir={out_dir}")
        reviews = _load_reviews(ROOT / args.dataset_csv)
        if args.limit_products:
            keep = set(sorted({review.nm_id for review in reviews})[: int(args.limit_products)])
            reviews = [review for review in reviews if review.nm_id in keep]
            logger.log(f"[smoke] limit_products={args.limit_products} nm_ids={sorted(keep)}")
        logger.log(f"[load] reviews={len(reviews)} products={len(set(r.nm_id for r in reviews))}")

        categories = {review.category_id for review in reviews}
        _, term_to_aspects_by_category, aspect_by_id_by_category = _build_hybrid_vocab(
            ROOT / args.core_vocab,
            ROOT / args.domain_vocab_dir,
            categories,
        )
        logger.log(f"[load] vocab_categories={sorted(categories)}")

        encoder = DiscoveryEncoder(batch_size=int(args.discovery_batch_size))
        embedding_cache: dict[str, np.ndarray] = {}
        discovery_by_product = _load_discovery(ROOT / args.discovery_dir, encoder, embedding_cache)
        logger.log(f"[load] discovery_products={len(discovery_by_product)} source={args.discovery_dir}")

        _extract_and_match_reviews(
            reviews=reviews,
            term_to_aspects_by_category=term_to_aspects_by_category,
            discovery_by_product=discovery_by_product,
            encoder=encoder,
            cache=embedding_cache,
            logger=logger,
        )
        _compute_discovery_gold_matches(reviews, discovery_by_product, encoder, embedding_cache)
        logger.log("[discovery] gold matches computed")

        pairs = _build_sentiment_pairs(reviews, aspect_by_id_by_category, discovery_by_product)
        sentiment_by_pair = _sentiment_scores(pairs, logger)
        negation_stats = _apply_negation_corrections(
            sentiment_by_pair,
            reviews,
            aspect_by_id_by_category,
            discovery_by_product,
            logger,
        )

        aggregated = _aggregate_product_scores(
            reviews,
            sentiment_by_pair,
            aspect_by_id_by_category,
            discovery_by_product,
        )
        logger.log("[aggregation] product scores computed")

        review_a, aggregate_a, hard_a = _review_metric_rows(
            reviews,
            term_to_aspects_by_category,
            discovery_by_product,
            sentiment_by_pair,
            include_discovery=False,
        )
        product_a, product_metrics_a = _product_metric_rows(
            reviews,
            term_to_aspects_by_category,
            aspect_by_id_by_category,
            discovery_by_product,
            aggregated,
            sentiment_by_pair,
            include_discovery=False,
        )
        aggregate_a.update(product_metrics_a)

        review_b, aggregate_b, hard_b = _review_metric_rows(
            reviews,
            term_to_aspects_by_category,
            discovery_by_product,
            sentiment_by_pair,
            include_discovery=True,
        )
        product_b, product_metrics_b = _product_metric_rows(
            reviews,
            term_to_aspects_by_category,
            aspect_by_id_by_category,
            discovery_by_product,
            aggregated,
            sentiment_by_pair,
            include_discovery=True,
        )
        aggregate_b.update(product_metrics_b)

        _star_review_df, star_review_metrics, star_product_df, star_product_metrics = _star_metrics(reviews)

        per_product_a = _per_product_metrics(reviews, review_a, product_a, hard_a)
        per_product_b = _per_product_metrics(reviews, review_b, product_b, hard_b)
        per_product_c_rows: list[dict[str, Any]] = []
        review_count_by_product = {nm_id: len(items) for nm_id, items in _group_reviews(reviews).items()}
        for (nm_id, category_id), group in star_product_df.groupby(["nm_id", "category_id"]):
            n3 = group[group["n_reviews_with_aspect"] >= PRODUCT_AGGREGATION_MIN_REVIEWS]
            per_product_c_rows.append(
                {
                    "nm_id": nm_id,
                    "category_id": category_id,
                    "n_reviews": review_count_by_product[int(nm_id)],
                    "product_mae_n3": float(n3["abs_error"].mean()) if not n3.empty else np.nan,
                    "n_aspects_matched": int(len(group)),
                }
            )
        per_product_c = pd.DataFrame(per_product_c_rows)
        per_product_c["detection_precision"] = np.nan
        per_product_c["detection_recall"] = np.nan
        per_product_c["sentiment_mae_review"] = star_review_metrics["sentiment_mae_review"]
        per_product_c["sentiment_mae_review_round"] = star_review_metrics["sentiment_mae_review_round"]

        _write_track_csv(out_dir / "metrics_track_a_vocab_only.csv", per_product_a)
        _write_track_csv(out_dir / "metrics_track_b_vocab_plus_discovery.csv", per_product_b)
        _write_track_csv(out_dir / "metrics_track_c_star_baseline.csv", per_product_c)

        per_aspect = pd.concat([product_b, star_product_df], ignore_index=True)
        per_aspect.to_csv(out_dir / "per_aspect_breakdown.csv", index=False, encoding="utf-8")
        hard_cases = pd.DataFrame(hard_b).sort_values(["abs_error", "review_id"], ascending=[False, True]).head(30)
        hard_cases.to_csv(out_dir / "hard_cases.csv", index=False, encoding="utf-8")
        negation_stats = _finalize_negation_stats(negation_stats, hard_b)

        _write_predictions(
            out_dir,
            reviews,
            aspect_by_id_by_category,
            discovery_by_product,
            sentiment_by_pair,
            aggregated,
        )
        _write_summary(
            out_dir,
            aggregate_a,
            aggregate_b,
            star_review_metrics,
            star_product_metrics,
            per_product_a,
            per_product_b,
            hard_cases,
            negation_stats,
        )

        summary_payload = {
            "status": "OK",
            "out_dir": str(out_dir),
            "elapsed_sec": round(time.perf_counter() - started, 4),
            "track_a": aggregate_a,
            "track_b": aggregate_b,
            "track_c_review": star_review_metrics,
            "track_c_product": star_product_metrics,
            "negation_correction": negation_stats,
            "params": {
                "NLI_TEMPERATURE": NLI_TEMPERATURE,
                "SENTIMENT_RATING_MIN": SENTIMENT_RATING_MIN,
                "SENTIMENT_RATING_MAX": SENTIMENT_RATING_MAX,
                "SENTIMENT_RELEVANCE_THRESHOLD": SENTIMENT_RELEVANCE_THRESHOLD,
                "SENTIMENT_RELEVANCE_MODE": SENTIMENT_RELEVANCE_MODE,
                "NEGATION_RAW_RATING_MAX": NEGATION_RAW_RATING_MAX,
                "NEGATION_REVIEW_RATING_MIN": NEGATION_REVIEW_RATING_MIN,
                "SENTIMENT_ENGINE_SOURCE": str(V4_SENTIMENT_ENGINE_PATH),
                "SENTIMENT_HYPOTHESIS_TEMPLATE_POS": "{aspect} — это хорошо",
                "SENTIMENT_SCORE_FORMULA": "5*P(entailment)+3*P(neutral)+1*P(contradiction)",
                "DISCOVERY_PHRASE_TO_CLUSTER_THRESHOLD": DISCOVERY_PHRASE_TO_CLUSTER_THRESHOLD,
                "DISCOVERY_TO_GOLD_MATCH_THRESHOLD": DISCOVERY_TO_GOLD_MATCH_THRESHOLD,
                "PRODUCT_AGGREGATION_MIN_REVIEWS": PRODUCT_AGGREGATION_MIN_REVIEWS,
                "USE_LEDOIT_WOLF_SHRINKAGE": USE_LEDOIT_WOLF_SHRINKAGE,
            },
        }
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.log(json.dumps(summary_payload, ensure_ascii=False, indent=2))

        det_p = round(float(aggregate_a["detection_precision"]), 4)
        det_r = round(float(aggregate_a["detection_recall"]), 4)
        det_f = round(float(aggregate_a["detection_f1"]), 4)
        logger.log(
            "[sanity] detection "
            f"P/R/F1={det_p:.4f}/{det_r:.4f}/{det_f:.4f} "
            f"reference={REFERENCE_DETECTION['precision']:.4f}/{REFERENCE_DETECTION['recall']:.4f}/{REFERENCE_DETECTION['f1']:.4f}"
        )
        logger.log(f"[done] elapsed={time.perf_counter() - started:.1f}s")
        return out_dir
    finally:
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Final end-to-end ABSA pipeline runner")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--domain-vocab-dir", default="src/vocabulary/domain")
    parser.add_argument("--discovery-dir", default="benchmark/discovery/results/20260424_231742_v3")
    parser.add_argument("--out-dir", default="benchmark/end_to_end/results")
    parser.add_argument("--discovery-batch-size", type=int, default=8)
    parser.add_argument("--limit-products", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

