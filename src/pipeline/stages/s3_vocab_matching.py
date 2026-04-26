from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scripts import run_phase2_baseline_matching as lexical


def _cosine_for_aspect(
    candidate_id: str,
    aspect_id: str,
    category_id: str,
    candidate_vectors_by_id: dict[str, np.ndarray],
    aspect_vectors_by_category: dict[str, dict[str, np.ndarray]],
) -> float:
    cand = candidate_vectors_by_id.get(candidate_id)
    asp = aspect_vectors_by_category.get(category_id, {}).get(aspect_id)
    if cand is None or asp is None:
        return float("nan")
    return float(np.dot(cand, asp))


def _best_cosine(
    candidate_id: str,
    category_id: str,
    candidate_vectors_by_id: dict[str, np.ndarray],
    aspect_vectors_by_category: dict[str, dict[str, np.ndarray]],
) -> float:
    cand = candidate_vectors_by_id.get(candidate_id)
    aspects = aspect_vectors_by_category.get(category_id, {})
    if cand is None or not aspects:
        return float("nan")
    matrix = np.vstack(list(aspects.values()))
    return float(np.max(matrix @ cand))


def run_stage(
    reviews: list[Any],
    candidates: pd.DataFrame,
    term_to_aspects_by_category: dict[str, dict[str, set[str]]],
    candidate_vectors_by_id: dict[str, np.ndarray],
    aspect_vectors_by_category: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows_by_review = {
        review_id: group.copy()
        for review_id, group in candidates.groupby("review_id", sort=False)
    } if not candidates.empty else {}

    for review in reviews:
        term_to_aspects = term_to_aspects_by_category[review.category_id]
        matched_terms = lexical._match_terms(review.candidate_lemmas, term_to_aspects, "lexical_only")
        matched_term_set = set(matched_terms)
        vocab_ids: set[str] = set()
        for term in matched_terms:
            vocab_ids.update(term_to_aspects[term])
        review.vocab_aspect_ids = vocab_ids
        unmatched_lemmas = sorted(review.candidate_lemmas - matched_term_set)
        review.unmatched_phrases = [
            review.candidate_surfaces_by_lemma[lemma][0]
            for lemma in unmatched_lemmas
            if review.candidate_surfaces_by_lemma.get(lemma)
        ]

        for row in rows_by_review.get(review.review_id, pd.DataFrame()).itertuples(index=False):
            lemma = str(row.text_lemmatized)
            aspect_ids = sorted(term_to_aspects.get(lemma, set())) if lemma in matched_term_set else []
            if aspect_ids:
                for aspect_id in aspect_ids:
                    rows.append(
                        {
                            "candidate_id": row.candidate_id,
                            "matched_aspect_id": aspect_id,
                            "match_method": "lexical",
                            "match_score": 1.0,
                            "matched_lemmas": [lemma],
                            "cosine_similarity": _cosine_for_aspect(
                                row.candidate_id,
                                aspect_id,
                                review.category_id,
                                candidate_vectors_by_id,
                                aspect_vectors_by_category,
                            ),
                            "is_unmatched": False,
                        }
                    )
            else:
                rows.append(
                    {
                        "candidate_id": row.candidate_id,
                        "matched_aspect_id": None,
                        "match_method": None,
                        "match_score": float("nan"),
                        "matched_lemmas": [],
                        "cosine_similarity": _best_cosine(
                            row.candidate_id,
                            review.category_id,
                            candidate_vectors_by_id,
                            aspect_vectors_by_category,
                        ),
                        "is_unmatched": True,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "matched_aspect_id",
            "match_method",
            "match_score",
            "matched_lemmas",
            "cosine_similarity",
            "is_unmatched",
        ],
    )
