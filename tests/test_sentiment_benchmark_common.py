from __future__ import annotations

import pandas as pd

from benchmark.sentiment.common import (
    aggregate_review_aspect_scores,
    find_phrase_occurrences,
    resolve_sentence_text,
)


def test_find_phrase_occurrences_prefers_longer_match() -> None:
    text = "Зажигалка хорошая, зажигалка огонь."
    hits = find_phrase_occurrences(text, ["зажигалка", "зажигалка хорошая"])

    assert len(hits) == 2
    assert hits[0]["phrase_template"] == "зажигалка хорошая"
    assert hits[1]["phrase_template"] == "зажигалка"


def test_resolve_sentence_text_falls_back_to_lemma_overlap() -> None:
    text = "Катушка сломалась почти сразу. Доставка быстрая."
    sentence = resolve_sentence_text(
        text,
        evidence_text="катушка качество",
        evidence_lemma_text="катушка качество",
        start_offset=-1,
        end_offset=-1,
    )

    assert sentence == "Катушка сломалась почти сразу."


def test_aggregate_review_aspect_scores_prefers_max_relevance() -> None:
    predictions = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "nm_id": 1,
                "category_id": "physical_goods",
                "aspect_key": "vocab::coil",
                "aspect_name": "Катушка",
                "aspect_source": "vocab",
                "evidence_id": "e1",
                "gold_matches_json": "{}",
                "passed_relevance_filter": True,
                "relevance_filter_value": 0.9,
                "raw_rating": 1.2,
            },
            {
                "review_id": "r1",
                "nm_id": 1,
                "category_id": "physical_goods",
                "aspect_key": "vocab::coil",
                "aspect_name": "Катушка",
                "aspect_source": "vocab",
                "evidence_id": "e2",
                "gold_matches_json": "{}",
                "passed_relevance_filter": True,
                "relevance_filter_value": 0.3,
                "raw_rating": 4.7,
            },
        ]
    )

    scores = aggregate_review_aspect_scores(predictions, aggregation="max_relevance")

    assert len(scores) == 1
    assert scores.iloc[0]["final_rating"] == 1.2
    assert scores.iloc[0]["selected_evidence_id"] == "e1"
