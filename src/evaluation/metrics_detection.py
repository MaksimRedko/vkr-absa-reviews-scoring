from __future__ import annotations

from typing import Any


def detection_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    track = summary.get("track_a", {})
    return {
        "detection_precision_track_a": float(track.get("detection_precision", float("nan"))),
        "detection_recall_track_a": float(track.get("detection_recall", float("nan"))),
        "detection_f1_track_a": float(track.get("detection_f1", float("nan"))),
    }
