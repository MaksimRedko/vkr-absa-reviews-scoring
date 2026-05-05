from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd


def parse_true_labels(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key).strip()] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def load_gold(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str})
    frame["true_labels_parsed"] = frame["true_labels"].map(parse_true_labels)
    return frame
