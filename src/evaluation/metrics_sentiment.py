from __future__ import annotations

from typing import Any


def sentiment_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    track = summary.get("track_a", {})
    neg = summary.get("negation_correction", {})
    return {
        "sentiment_mae_review": float(track.get("sentiment_mae_review", float("nan"))),
        "sentiment_mae_round": float(track.get("sentiment_mae_review_round", float("nan"))),
        "inversion_rate": float(neg.get("inversion_rate", float("nan"))),
    }
