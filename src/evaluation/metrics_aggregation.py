from __future__ import annotations

from typing import Any


def aggregation_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    track = summary.get("track_a", {})
    star = summary.get("track_c_product", {})
    return {
        "product_mae_n3": float(track.get("product_mae_n3", float("nan"))),
        "star_baseline_product_mae_n3": float(star.get("product_mae_n3", float("nan"))),
    }
