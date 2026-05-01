from __future__ import annotations

import pandas as pd

from benchmark.sentiment.common import (
    MODE_A,
    MODE_B,
    MODE_C,
    _build_assignment_mode_frame,
    aggregate_review_aspect_scores,
    assert_shared_single_mode_lengths,
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


def test_assignment_mode_frames_keep_same_row_count() -> None:
    assignments = pd.DataFrame(
        [
            {
                "assignment_id": "a1",
                "review_id": "r1",
                "nm_id": 1,
                "category_id": "physical_goods",
                "aspect_key": "vocab::coil",
                "aspect_name": "Катушка",
                "aspect_source": "vocab",
                "candidate_id": "c1",
                "cluster_id": None,
                "evidence_text": "Катушка сломалась почти сразу",
                "evidence_lemma_text": "катушка сломаться почти сразу",
                "sentence_text": "Катушка сломалась почти сразу.",
                "window_text": "Катушка сломалась почти сразу",
                "review_text": "Катушка сломалась почти сразу. Доставка быстрая.",
                "start_offset": 0,
                "end_offset": 27,
                "gold_matches_json": "{}",
            },
            {
                "assignment_id": "a2",
                "review_id": "r1",
                "nm_id": 1,
                "category_id": "physical_goods",
                "aspect_key": "vocab::delivery",
                "aspect_name": "Доставка",
                "aspect_source": "vocab",
                "candidate_id": "c2",
                "cluster_id": None,
                "evidence_text": "Доставка быстрая",
                "evidence_lemma_text": "доставка быстрый",
                "sentence_text": "Доставка быстрая.",
                "window_text": "Доставка быстрая",
                "review_text": "Катушка сломалась почти сразу. Доставка быстрая.",
                "start_offset": 30,
                "end_offset": 47,
                "gold_matches_json": "{}",
            },
        ]
    )

    assert_shared_single_mode_lengths(assignments)
    frame_a = _build_assignment_mode_frame(assignments, MODE_A)
    frame_b = _build_assignment_mode_frame(assignments, MODE_B)
    frame_c = _build_assignment_mode_frame(assignments, MODE_C)

    assert len(frame_a) == len(frame_b) == len(frame_c) == 2
    assert frame_a.iloc[0]["premise_text"] == "Катушка сломалась почти сразу. Доставка быстрая."
    assert frame_b.iloc[0]["premise_text"] == "Катушка сломалась почти сразу."
    assert frame_c.iloc[0]["premise_text"] == "Катушка сломалась почти сразу"
