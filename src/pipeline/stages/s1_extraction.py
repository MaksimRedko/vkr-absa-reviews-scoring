from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from scripts import run_phase2_baseline_matching as lexical
from src.pipeline.stages.common import stable_id


def _find_offset(clean_text: str, span: str, cursors: dict[str, int]) -> int:
    key = span.lower()
    start_from = cursors.get(key, -1) + 1
    offset = clean_text.lower().find(key, start_from)
    if offset < 0:
        offset = clean_text.lower().find(key)
    cursors[key] = offset
    return int(offset)


def run_stage(reviews: list[Any], config: dict[str, Any]) -> pd.DataFrame:
    ext_cfg = config.get("extraction", {})
    ngram = tuple(ext_cfg.get("ngram_range", [1, 2]))
    extractor = lexical.CandidateExtractor(
        ngram_range=(int(ngram[0]), int(ngram[1])),
        min_word_length=int(ext_cfg.get("min_word_length", 3)),
    )
    extractor.dependency_filter_enabled = bool(ext_cfg.get("dependency_filter_enabled", False))

    rows: list[dict[str, Any]] = []
    for review in reviews:
        candidates = extractor.extract(review.text)
        clean_text = extractor._clean(review.text)
        cursors: dict[str, int] = {}
        surfaces_by_lemma: dict[str, list[str]] = defaultdict(list)

        for candidate in candidates:
            lemma = lexical._normalize(candidate.span)
            if not lemma:
                continue
            start = _find_offset(clean_text, candidate.span, cursors)
            end = start + len(candidate.span) if start >= 0 else -1
            candidate_id = stable_id(review.review_id, lemma, start)
            surfaces_by_lemma[lemma].append(candidate.span)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "review_id": review.review_id,
                    "nm_id": int(review.nm_id),
                    "category_id": review.category_id,
                    "text": candidate.span,
                    "text_lemmatized": lemma,
                    "start_offset": int(start),
                    "end_offset": int(end),
                    "source": "ngram",
                    "sentence": candidate.sentence,
                }
            )

        review.candidate_surfaces_by_lemma = dict(surfaces_by_lemma)
        review.candidate_lemmas = set(surfaces_by_lemma)

    columns = [
        "candidate_id",
        "review_id",
        "nm_id",
        "category_id",
        "text",
        "text_lemmatized",
        "start_offset",
        "end_offset",
        "source",
        "sentence",
    ]
    return pd.DataFrame(rows, columns=columns)
