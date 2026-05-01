from __future__ import annotations

import math

import pandas as pd

from scripts import compare_final_sentiment_runs as compare


def test_direction_thresholds() -> None:
    assert compare.gold_direction(1.0) == "negative"
    assert compare.gold_direction(3.0) == "neutral"
    assert compare.gold_direction(5.0) == "positive"

    assert compare.predicted_direction(2.74) == "negative"
    assert compare.predicted_direction(2.75) == "neutral"
    assert compare.predicted_direction(3.25) == "neutral"
    assert compare.predicted_direction(3.26) == "positive"


def test_compute_run_metrics_basic_values() -> None:
    df = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "nm_id": 1,
                "category_id": "c1",
                "gold_label": "a1",
                "gold_rating": 5.0,
                "predicted_rating": 4.2,
                "aspect_source": "vocab",
            },
            {
                "review_id": "r2",
                "nm_id": 1,
                "category_id": "c1",
                "gold_label": "a2",
                "gold_rating": 1.0,
                "predicted_rating": 4.5,
                "aspect_source": "discovery",
            },
            {
                "review_id": "r3",
                "nm_id": 2,
                "category_id": "c2",
                "gold_label": "a3",
                "gold_rating": 3.0,
                "predicted_rating": 3.1,
                "aspect_source": "vocab",
            },
        ]
    )
    enriched = compare.enrich_pair_rows(df, total_gold_pairs=10)
    metrics = compare.compute_run_metrics(enriched, total_gold_pairs=10)

    assert math.isclose(metrics["mae"], (0.8 + 3.5 + 0.1) / 3.0, rel_tol=1e-9)
    assert math.isclose(metrics["accuracy_at_1_0"], 2.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(metrics["wrong_polarity_rate"], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(metrics["strong_wrong_polarity_rate"], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(metrics["vocab_accuracy_at_1"], 1.0, rel_tol=1e-9)
    assert math.isclose(metrics["discovery_accuracy_at_1"], 0.0, rel_tol=1e-9)
    assert math.isclose(metrics["coverage"], 0.3, rel_tol=1e-9)
    assert math.isclose(metrics["n_pairs"], 3.0, rel_tol=1e-9)
    assert math.isclose(metrics["n_products"], 2.0, rel_tol=1e-9)


def test_error_type_priority_prefers_strong_wrong_polarity() -> None:
    row = pd.Series(
        {
            "strong_wrong_polarity": True,
            "wrong_polarity": True,
            "large_too_low": False,
            "large_too_high": True,
            "too_low": False,
            "too_high": True,
        }
    )
    assert compare._classify_error_type(row) == "strong_wrong_polarity"
