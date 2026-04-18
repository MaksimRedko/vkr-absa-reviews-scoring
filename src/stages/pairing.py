from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from src.schemas.models import AspectInfo, Candidate, ScoredCandidate
    from src.stages.contracts import ExtractionStage

SentimentPair = Tuple[str, str, str, str, float]


def _resolve_product_anchors(
    aspects: Dict[str, "AspectInfo"],
    anchor_embeddings: Dict[str, np.ndarray],
) -> set[str]:
    product_anchors: set[str] = set()
    for asp_name, info in aspects.items():
        nli = (info.nli_label or asp_name).strip() or asp_name
        if nli in anchor_embeddings:
            product_anchors.add(nli)
        if asp_name in anchor_embeddings:
            product_anchors.add(asp_name)
    return product_anchors


def extract_all_with_mapping(
    extractor: ExtractionStage,
    texts: List[str],
    review_ids: List[str],
) -> Tuple[List[Candidate], Dict[str, str]]:
    all_candidates: List[Candidate] = []
    sentence_to_review: Dict[str, str] = {}

    for text, review_id in zip(texts, review_ids):
        candidates = extractor.extract(text)
        for idx, cand in enumerate(candidates):
            cand.review_id = review_id
            if not getattr(cand, "candidate_id", ""):
                cand.candidate_id = f"{review_id}:{idx}"
            sentence_to_review[cand.sentence.strip()] = review_id
            sentence_to_review[cand.sentence.lower().strip()] = review_id
        all_candidates.extend(candidates)

    return all_candidates, sentence_to_review


def build_sentiment_pairs(
    scored_candidates: List[ScoredCandidate],
    aspects: Dict[str, AspectInfo],
    sentence_to_review: Dict[str, str],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float,
    max_aspects: int,
) -> List[SentimentPair]:
    """
    Multi-label: cos(span, anchor) >= threshold -> NLI-пара (до max_aspects якорей).
    product_anchors — из результата кластеризации (имена якорей / nli_label).
    (review_id, sentence, aspect_name, nli_label, weight); здесь aspect_name = nli_label = якорь.
    """
    if not aspects or not scored_candidates:
        return []

    anchor_names = list(anchor_embeddings.keys())
    anchor_matrix = np.stack([anchor_embeddings[n] for n in anchor_names])

    product_anchors = _resolve_product_anchors(aspects, anchor_embeddings)

    seen_pairs: set[Tuple[str, str, str]] = set()
    pairs: List[SentimentPair] = []

    for cand in scored_candidates:
        emb = np.asarray(cand.embedding, dtype=np.float64).reshape(1, -1)
        sims = cosine_similarity(emb, anchor_matrix)[0]

        candidates_anchors: List[Tuple[str, float]] = []
        for idx, sim in enumerate(sims):
            aname = anchor_names[idx]
            if sim >= threshold and aname in product_anchors:
                candidates_anchors.append((aname, float(sim)))

        candidates_anchors.sort(key=lambda x: x[1], reverse=True)
        candidates_anchors = candidates_anchors[:max_aspects]

        if not candidates_anchors:
            continue

        review_id = sentence_to_review.get(
            cand.sentence.strip(),
            sentence_to_review.get(cand.sentence.lower().strip(), "unknown"),
        )

        for aname, sim in candidates_anchors:
            pair_key = (review_id, cand.sentence, aname)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            pairs.append((review_id, cand.sentence, aname, aname, float(sim)))

    return pairs


def build_review_level_pairs(
    review_text_by_id: Dict[str, str],
    scored_candidates: List["ScoredCandidate"],
    candidate_assignments: Dict[str, str],
    aspects: Dict[str, "AspectInfo"],
) -> List[SentimentPair]:
    """
    Review-level режим: одна NLI-пара на (review, aspect), но только для аспектов,
    в которые реально попали кандидаты из этого отзыва.
    Вес = 1.0, premise = clean_text отзыва.
    """
    if not aspects or not review_text_by_id or not scored_candidates or not candidate_assignments:
        return []

    aspect_to_nli: Dict[str, str] = {
        aspect_name: (info.nli_label or aspect_name).strip() or aspect_name
        for aspect_name, info in aspects.items()
    }

    review_aspects: Dict[str, Dict[str, str]] = defaultdict(dict)
    for candidate in scored_candidates:
        candidate_id = str(getattr(candidate, "candidate_id", "") or "")
        review_id = str(getattr(candidate, "review_id", "") or "")
        if not candidate_id or not review_id:
            continue
        aspect_name = candidate_assignments.get(candidate_id, "")
        if not aspect_name:
            continue
        nli_label = aspect_to_nli.get(aspect_name, aspect_name)
        review_aspects[review_id][aspect_name] = nli_label

    pairs: List[SentimentPair] = []
    for review_id, aspect_map in review_aspects.items():
        text = review_text_by_id.get(review_id, "")
        if not text:
            continue
        for aspect_name in sorted(aspect_map):
            pairs.append((review_id, text, aspect_name, aspect_map[aspect_name], 1.0))
    return pairs
