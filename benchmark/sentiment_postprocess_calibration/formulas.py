from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

EPS = 1e-8


def _clip_rating(value: float) -> float:
    return float(min(5.0, max(1.0, value)))


def _as_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value))))


@dataclass(frozen=True, slots=True)
class FormulaSpec:
    name: str
    required_fields: tuple[str, ...]
    fn: Callable[[Mapping[str, Any]], float]
    description: str


def f0_current(row: Mapping[str, Any]) -> float:
    return _clip_rating(_as_float(row, "current_final_rating"))


def f1_pos_full_distribution(row: Mapping[str, Any]) -> float:
    rating = (
        1.0 * _as_float(row, "pos_contradiction")
        + 3.0 * _as_float(row, "pos_neutral")
        + 5.0 * _as_float(row, "pos_entailment")
    )
    return _clip_rating(rating)


def f2_neg_full_distribution(row: Mapping[str, Any]) -> float:
    rating = (
        5.0 * _as_float(row, "neg_contradiction")
        + 3.0 * _as_float(row, "neg_neutral")
        + 1.0 * _as_float(row, "neg_entailment")
    )
    return _clip_rating(rating)


def f3_avg_full_distribution(row: Mapping[str, Any]) -> float:
    return _clip_rating((f1_pos_full_distribution(row) + f2_neg_full_distribution(row)) / 2.0)


def f4_dual_diff(row: Mapping[str, Any]) -> float:
    pos = _as_float(row, "pos_entailment")
    neg = _as_float(row, "neg_entailment")
    return _clip_rating(3.0 + 2.0 * (pos - neg))


def f5_dual_ratio(row: Mapping[str, Any]) -> float:
    pos = _as_float(row, "pos_entailment")
    neg = _as_float(row, "neg_entailment")
    return _clip_rating(3.0 + 2.0 * (pos - neg) / (pos + neg + EPS))


def make_f6_dual_logratio(temp: float) -> Callable[[Mapping[str, Any]], float]:
    def _formula(row: Mapping[str, Any]) -> float:
        score = math.log(_as_float(row, "pos_entailment") + EPS) - math.log(_as_float(row, "neg_entailment") + EPS)
        return _clip_rating(1.0 + 4.0 * _sigmoid(score / temp))

    return _formula


def f7_dual_full_nli(row: Mapping[str, Any]) -> float:
    pos_signal = _as_float(row, "pos_entailment") + _as_float(row, "neg_contradiction")
    neg_signal = _as_float(row, "neg_entailment") + _as_float(row, "pos_contradiction")
    neutral_signal = 0.5 * (_as_float(row, "pos_neutral") + _as_float(row, "neg_neutral"))
    total = pos_signal + neg_signal + neutral_signal + EPS
    pos_norm = pos_signal / total
    neg_norm = neg_signal / total
    neutral_norm = neutral_signal / total
    return _clip_rating(5.0 * pos_norm + 3.0 * neutral_norm + 1.0 * neg_norm)


def make_f8_dual_full_nli_power(alpha: float) -> Callable[[Mapping[str, Any]], float]:
    def _formula(row: Mapping[str, Any]) -> float:
        pos_signal = _as_float(row, "pos_entailment") + _as_float(row, "neg_contradiction")
        neg_signal = _as_float(row, "neg_entailment") + _as_float(row, "pos_contradiction")
        neutral_signal = 0.5 * (_as_float(row, "pos_neutral") + _as_float(row, "neg_neutral"))
        pos_signal = pos_signal**alpha
        neg_signal = neg_signal**alpha
        neutral_signal = neutral_signal**alpha
        total = pos_signal + neg_signal + neutral_signal + EPS
        pos_norm = pos_signal / total
        neg_norm = neg_signal / total
        neutral_norm = neutral_signal / total
        return _clip_rating(5.0 * pos_norm + 3.0 * neutral_norm + 1.0 * neg_norm)

    return _formula


def make_f9_center_expansion(gamma: float) -> Callable[[Mapping[str, Any]], float]:
    def _formula(row: Mapping[str, Any]) -> float:
        return _clip_rating(3.0 + gamma * (_as_float(row, "current_final_rating") - 3.0))

    return _formula


def build_formula_specs() -> list[FormulaSpec]:
    specs = [
        FormulaSpec(
            name="F0_current",
            required_fields=("current_final_rating",),
            fn=f0_current,
            description="Current baseline final rating.",
        ),
        FormulaSpec(
            name="F1_pos_full_distribution",
            required_fields=("pos_entailment", "pos_neutral", "pos_contradiction"),
            fn=f1_pos_full_distribution,
            description="Single positive-hypothesis expected rating from full distribution.",
        ),
        FormulaSpec(
            name="F2_neg_full_distribution",
            required_fields=("neg_entailment", "neg_neutral", "neg_contradiction"),
            fn=f2_neg_full_distribution,
            description="Negative-hypothesis expected rating from full distribution.",
        ),
        FormulaSpec(
            name="F3_avg_full_distribution",
            required_fields=(
                "pos_entailment",
                "pos_neutral",
                "pos_contradiction",
                "neg_entailment",
                "neg_neutral",
                "neg_contradiction",
            ),
            fn=f3_avg_full_distribution,
            description="Average of F1 and F2.",
        ),
        FormulaSpec(
            name="F4_dual_diff",
            required_fields=("pos_entailment", "neg_entailment"),
            fn=f4_dual_diff,
            description="Linear difference between positive and negative entailment.",
        ),
        FormulaSpec(
            name="F5_dual_ratio",
            required_fields=("pos_entailment", "neg_entailment"),
            fn=f5_dual_ratio,
            description="Relative entailment difference.",
        ),
        FormulaSpec(
            name="F7_dual_full_nli",
            required_fields=(
                "pos_entailment",
                "pos_neutral",
                "pos_contradiction",
                "neg_entailment",
                "neg_neutral",
                "neg_contradiction",
            ),
            fn=f7_dual_full_nli,
            description="Three-way normalized dual-hypothesis rating.",
        ),
    ]
    for temp in (0.5, 0.7, 1.0, 1.3, 1.5, 2.0):
        specs.append(
            FormulaSpec(
                name=f"F6_dual_logratio_T{str(temp).replace('.', '_')}",
                required_fields=("pos_entailment", "neg_entailment"),
                fn=make_f6_dual_logratio(temp),
                description=f"Log-ratio with sigmoid temperature T={temp}.",
            )
        )
    for alpha in (0.5, 0.7, 1.0, 1.3, 1.5, 2.0):
        specs.append(
            FormulaSpec(
                name=f"F8_dual_full_nli_power_alpha{str(alpha).replace('.', '_')}",
                required_fields=(
                    "pos_entailment",
                    "pos_neutral",
                    "pos_contradiction",
                    "neg_entailment",
                    "neg_neutral",
                    "neg_contradiction",
                ),
                fn=make_f8_dual_full_nli_power(alpha),
                description=f"Power-transformed dual full NLI with alpha={alpha}.",
            )
        )
    for gamma in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
        specs.append(
            FormulaSpec(
                name=f"F9_center_expansion_gamma{str(gamma).replace('.', '_')}",
                required_fields=("current_final_rating",),
                fn=make_f9_center_expansion(gamma),
                description=f"Linear center expansion around 3 with gamma={gamma}.",
            )
        )
    return specs


def build_formula_availability(df) -> list[dict[str, Any]]:
    availability: list[dict[str, Any]] = []
    for spec in build_formula_specs():
        missing_columns = [column for column in spec.required_fields if column not in df.columns]
        null_columns = []
        if not missing_columns:
            null_columns = [column for column in spec.required_fields if bool(df[column].isna().any())]
        is_available = not missing_columns and not null_columns
        reason = ""
        if missing_columns:
            reason = f"missing columns: {', '.join(missing_columns)}"
        elif null_columns:
            null_parts = [f"{column}={int(df[column].isna().sum())}" for column in null_columns]
            reason = f"null values in required columns: {', '.join(null_parts)}"
        availability.append(
            {
                "formula_name": spec.name,
                "available": bool(is_available),
                "required_fields": ",".join(spec.required_fields),
                "reason": reason,
                "description": spec.description,
            }
        )
    return availability


def get_formula_spec_map() -> dict[str, FormulaSpec]:
    return {spec.name: spec for spec in build_formula_specs()}


__all__ = [
    "EPS",
    "FormulaSpec",
    "build_formula_availability",
    "build_formula_specs",
    "get_formula_spec_map",
]
