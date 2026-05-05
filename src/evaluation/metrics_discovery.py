from __future__ import annotations

from typing import Any


def discovery_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    track = summary.get("track_b", {})
    return {
        "track_b_detection_precision": float(track.get("detection_precision", float("nan"))),
        "track_b_detection_recall": float(track.get("detection_recall", float("nan"))),
        "track_b_detection_f1": float(track.get("detection_f1", float("nan"))),
    }
