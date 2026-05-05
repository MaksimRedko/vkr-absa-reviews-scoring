import math

import pandas as pd

from scripts import recompute_manual_audit_metrics as audit


def test_normalize_text_treats_none_sentinel_as_empty() -> None:
    assert audit._normalize_text(None) == ""
    assert audit._normalize_text(float("nan")) == ""
    assert audit._normalize_text(" NONE ") == ""
    assert audit._normalize_text("Цена") == "Цена"


def test_detection_row_matches_requested_formulas() -> None:
    system_df = pd.DataFrame(
        {
            "review_id": ["r1", "r1", "r2", "r3", "r3"],
            "manual_decision_norm": ["TP", "FP", "UNCLEAR", "DUPLICATE", "OUT_OF_SCOPE"],
        }
    )
    gold_df = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "status_norm": ["FOUND", "FN", "FN"],
        }
    )

    row = audit.compute_detection_row(
        slice_type="overall",
        slice_value="overall",
        system_df=system_df,
        gold_df=gold_df,
        allow_recall=True,
        allow_precision=True,
    )

    assert row["tp"] == 1
    assert row["fp"] == 1
    assert row["fn"] == 2
    assert row["unclear"] == 1
    assert row["duplicate"] == 1
    assert math.isclose(row["manual_precision_strict"], 1 / 3)
    assert math.isclose(row["manual_precision_soft"], 1 / 2)
    assert math.isclose(row["manual_recall"], 1 / 3)
    assert math.isclose(row["manual_f1_strict"], 1 / 3)
    assert math.isclose(row["manual_f1_soft"], 0.4)


def test_sentiment_metrics_and_error_labels() -> None:
    pairs = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "gold_rating": [5.0, 1.0, 3.0],
            "predicted_rating": [4.6, 4.2, 2.7],
        }
    )
    enriched = audit.enrich_sentiment_pairs(pairs)
    row = audit.compute_sentiment_row(slice_type="overall", slice_value="overall", pairs=enriched)

    assert row["n_pairs"] == 3
    assert row["manual_wrong_polarity_rate"] > 0.0
    assert row["manual_strong_wrong_polarity_rate"] > 0.0
    assert row["manual_rmse"] >= row["manual_sentiment_mae"]
    assert set(enriched["error_type"]) >= {"near_miss", "strong_wrong_polarity"}


def test_fuzzy_vector_mapping_interpolates_fractional_ratings() -> None:
    vec = audit.rating_to_fuzzy_sentiment_vector(4.5)
    assert all(math.isclose(float(value), expected) for value, expected in zip(vec, [0.0, 0.25, 0.75], strict=True))

    vec = audit.rating_to_fuzzy_sentiment_vector(3.1)
    assert all(math.isclose(float(value), expected) for value, expected in zip(vec, [0.0, 0.95, 0.05], strict=True))

    vec = audit.rating_to_fuzzy_sentiment_vector(1.4)
    assert all(math.isclose(float(value), expected) for value, expected in zip(vec, [0.8, 0.2, 0.0], strict=True))


def test_vector_sentiment_metrics_capture_collapse_flip_and_underestimate() -> None:
    pairs = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "gold_rating": [5.0, 1.0, 5.0],
            "predicted_rating": [3.0, 5.0, 4.0],
        }
    )
    enriched = audit.enrich_vector_sentiment_pairs(audit.enrich_sentiment_pairs(pairs))
    row = audit.compute_vector_sentiment_row(slice_type="overall", slice_value="overall", pairs=enriched)

    assert row["n_pairs"] == 3
    assert row["n_gold_strong_polar_pairs"] == 3
    assert row["neutral_collapse_rate"] == 1 / 3
    assert row["polarity_flip_rate"] == 1 / 3
    assert row["n_same_polar_direction_pairs"] == 1
    assert row["intensity_underestimate_rate"] == 1.0
    assert row["mean_vector_l1_distance"] > 0.0
    assert row["mean_cosine_similarity"] < 1.0
