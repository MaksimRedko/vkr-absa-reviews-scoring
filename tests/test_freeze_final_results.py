from __future__ import annotations

import pandas as pd

from scripts import freeze_final_results as freeze


def test_compat_nli_predictions_renames_dual_hypothesis_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "nm_id": 11,
                "aspect_key": "vocab::quality",
                "aspect_name": "Качество",
                "aspect_source": "vocab",
                "premise_text": "Катушка сломалась почти сразу",
                "premise_kind": "sentence",
                "evidence_id": "ev1",
                "hypothesis_pos_text": "В этом фрагменте качество оценивается положительно",
                "hypothesis_neg_text": "В этом фрагменте качество оценивается отрицательно",
                "p_entailment_pos": 0.1,
                "p_entailment_neg": 0.7,
                "p_neutral": 0.2,
                "relevance_filter_value": 0.8,
                "passed_relevance_filter": True,
                "raw_rating": 1.5,
                "final_rating": 1.5,
            }
        ]
    )

    result = freeze._compat_nli_predictions(frame)

    row = result.iloc[0]
    assert row["aspect_key"] == "vocab::quality"
    assert row["hypothesis_text"] == frame.iloc[0]["hypothesis_pos_text"]
    assert row["p_entailment"] == frame.iloc[0]["p_entailment_pos"]
    assert row["p_contradiction"] == frame.iloc[0]["p_entailment_neg"]
    assert bool(row["negation_correction_applied"]) is False


def test_sentiment_by_pair_uses_selected_evidence_row() -> None:
    predictions = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "aspect_key": "discovery::11::2",
                "evidence_id": "ev_bad",
                "p_entailment_pos": 0.2,
                "p_entailment_neg": 0.5,
                "p_neutral": 0.3,
                "relevance_filter_value": 0.7,
                "raw_rating": 2.0,
            },
            {
                "review_id": "r1",
                "aspect_key": "discovery::11::2",
                "evidence_id": "ev_good",
                "p_entailment_pos": 0.85,
                "p_entailment_neg": 0.05,
                "p_neutral": 0.10,
                "relevance_filter_value": 0.9,
                "raw_rating": 4.7,
            },
        ]
    )
    review_scores = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "aspect_key": "discovery::11::2",
                "selected_evidence_id": "ev_good",
                "final_rating": 4.8,
            }
        ]
    )

    result = freeze._sentiment_by_pair_from_mode_b(predictions, review_scores)

    scores = result[("r1", "discovery::11::2")]
    assert scores["rating"] == 4.8
    assert scores["raw_rating"] == 4.7
    assert scores["p_ent_pos"] == 0.85
    assert scores["p_ent_neg"] == 0.05
