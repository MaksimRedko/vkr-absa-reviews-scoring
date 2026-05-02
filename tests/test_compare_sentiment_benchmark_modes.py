from __future__ import annotations

import pandas as pd

from scripts import compare_sentiment_benchmark_modes as compare


def test_mode_ids_include_weighted_mode() -> None:
    assert "mode_d_multi_evidence_weighted_relevance" in compare.MODE_IDS


def test_common_pair_keys_requires_presence_in_all_modes() -> None:
    frames = {
        "a": pd.DataFrame(
            [
                {"review_id": "r1", "nm_id": 1, "category_id": "c1", "gold_label": "g1", "gold_rating": 5.0},
                {"review_id": "r2", "nm_id": 1, "category_id": "c1", "gold_label": "g2", "gold_rating": 4.0},
            ]
        ),
        "b": pd.DataFrame(
            [
                {"review_id": "r1", "nm_id": 1, "category_id": "c1", "gold_label": "g1", "gold_rating": 5.0},
                {"review_id": "r3", "nm_id": 2, "category_id": "c2", "gold_label": "g3", "gold_rating": 3.0},
            ]
        ),
        "c": pd.DataFrame(
            [
                {"review_id": "r1", "nm_id": 1, "category_id": "c1", "gold_label": "g1", "gold_rating": 5.0},
            ]
        ),
        "d": pd.DataFrame(
            [
                {"review_id": "r1", "nm_id": 1, "category_id": "c1", "gold_label": "g1", "gold_rating": 5.0},
                {"review_id": "r4", "nm_id": 3, "category_id": "c3", "gold_label": "g4", "gold_rating": 2.0},
            ]
        ),
    }

    common = compare.common_pair_keys(frames)

    assert common == {("r1", 1, "c1", "g1", 5.0)}


def test_filter_to_pair_keys_keeps_only_requested_rows() -> None:
    df = pd.DataFrame(
        [
            {"review_id": "r1", "nm_id": 1, "category_id": "c1", "gold_label": "g1", "gold_rating": 5.0, "predicted_rating": 4.5},
            {"review_id": "r2", "nm_id": 2, "category_id": "c2", "gold_label": "g2", "gold_rating": 3.0, "predicted_rating": 3.2},
        ]
    )
    filtered = compare._filter_to_pair_keys(df, {("r2", 2, "c2", "g2", 3.0)})

    assert filtered.shape[0] == 1
    assert filtered.iloc[0]["review_id"] == "r2"
