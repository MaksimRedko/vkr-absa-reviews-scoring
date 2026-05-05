from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.sentiment.common import (
    MODE_A,
    MODE_B,
    MODE_C,
    MODE_D,
    BenchmarkContext,
    _build_assignment_mode_frame,
    aggregate_review_aspect_scores,
    assert_shared_single_mode_lengths,
    build_mode_input_frame,
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
                "cluster_id": None,
                "review_text": "Катушка сломалась почти сразу. Доставка быстрая.",
                "sentence_text": "Катушка сломалась почти сразу.",
                "window_text": "Катушка сломалась почти сразу",
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
                "cluster_id": None,
                "review_text": "Катушка сломалась почти сразу. Доставка быстрая.",
                "sentence_text": "Доставка быстрая.",
                "window_text": "Доставка быстрая",
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


def test_shared_artifact_path_uses_assignments_for_a_and_evidence_for_bcd() -> None:
    assignments = pd.DataFrame(
        [
            {
                "assignment_id": "a1",
                "review_id": "r1",
                "nm_id": 1,
                "category_id": "physical_goods",
                "aspect_key": "vocab::coil",
                "aspect_name": "coil",
                "aspect_source": "vocab",
                "cluster_id": None,
                "review_text": "coil broke quickly. rattles.",
                "gold_matches_json": "{}",
            },
            {
                "assignment_id": "a2",
                "review_id": "r2",
                "nm_id": 2,
                "category_id": "physical_goods",
                "aspect_key": "vocab::delivery",
                "aspect_name": "delivery",
                "aspect_source": "vocab",
                "cluster_id": None,
                "review_text": "delivery was fast.",
                "gold_matches_json": "{}",
            },
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "evidence_id": "e1",
                "assignment_id": "a1",
                "review_id": "r1",
                "nm_id": 1,
                "category_id": "physical_goods",
                "aspect_key": "vocab::coil",
                "aspect_name": "coil",
                "aspect_source": "vocab",
                "cluster_id": None,
                "review_text": "coil broke quickly. rattles.",
                "gold_matches_json": "{}",
                "candidate_id": "c1",
                "evidence_text": "coil broke",
                "evidence_lemma_text": "coil break",
                "sentence_text": "coil broke quickly.",
                "window_text": "coil broke quickly",
                "start_offset": 0,
                "end_offset": 10,
                "start_sort": 0,
                "evidence_len": 10,
            },
            {
                "evidence_id": "e2",
                "assignment_id": "a1",
                "review_id": "r1",
                "nm_id": 1,
                "category_id": "physical_goods",
                "aspect_key": "vocab::coil",
                "aspect_name": "coil",
                "aspect_source": "vocab",
                "cluster_id": None,
                "review_text": "coil broke quickly. rattles.",
                "gold_matches_json": "{}",
                "candidate_id": "c2",
                "evidence_text": "rattles",
                "evidence_lemma_text": "rattle",
                "sentence_text": "rattles.",
                "window_text": "rattles",
                "start_offset": 20,
                "end_offset": 27,
                "start_sort": 20,
                "evidence_len": 7,
            },
            {
                "evidence_id": "e3",
                "assignment_id": "a2",
                "review_id": "r2",
                "nm_id": 2,
                "category_id": "physical_goods",
                "aspect_key": "vocab::delivery",
                "aspect_name": "delivery",
                "aspect_source": "vocab",
                "cluster_id": None,
                "review_text": "delivery was fast.",
                "gold_matches_json": "{}",
                "candidate_id": "c3",
                "evidence_text": "delivery was fast",
                "evidence_lemma_text": "delivery be fast",
                "sentence_text": "delivery was fast.",
                "window_text": "delivery was fast",
                "start_offset": 0,
                "end_offset": 17,
                "start_sort": 0,
                "evidence_len": 17,
            },
        ]
    )
    context = BenchmarkContext(
        run_dir=Path("."),
        dataset_path=Path("."),
        run_config={},
        reviews=[],
        reviews_by_id={},
        term_to_aspects_by_category={},
        aspect_by_id_by_category={},
        assignments=assignments,
        evidence=evidence,
        discovery_assignment_count=0,
        discovery_assignments_without_evidence=0,
    )

    frame_a = build_mode_input_frame(context, MODE_A)
    frame_b = build_mode_input_frame(context, MODE_B)
    frame_c = build_mode_input_frame(context, MODE_C)
    frame_d = build_mode_input_frame(context, MODE_D)

    assert len(frame_a) == 2
    assert len(frame_b) == 2
    assert len(frame_c) == 2
    assert len(frame_d) == 3
    assert frame_b["assignment_id"].tolist() == ["a1", "a2"]
    assert frame_d["assignment_id"].tolist().count("a1") == 2
