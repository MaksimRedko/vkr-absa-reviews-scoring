from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from configs.configs import temporary_config_overrides
from src.pipeline.reference import e2e
from src.pipeline.stages.common import stable_id


def _aspect_label_for_pair(pair: Any) -> str:
    return str(pair.nli_label)


def run_stage(
    reviews: list[Any],
    aspect_by_id_by_category: dict[str, dict[str, Any]],
    discovery_by_product: dict[int, Any],
    *,
    logger: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    pairs = e2e()._build_sentiment_pairs(reviews, aspect_by_id_by_category, discovery_by_product)
    sent_cfg = config.get("sentiment", {})
    threshold = float(sent_cfg.get("relevance_filter_threshold", 0.2))
    overrides = {
        "sentiment": {
            "temperature": float(sent_cfg.get("nli_temperature", 0.7)),
            "hypothesis_template_pos": "{aspect} — это хорошо",
            "relevance_threshold": threshold,
        }
    }

    logger.log(
        f"[sentiment] pairs={len(pairs)} nli_calls={len(pairs)} "
        f"T={overrides['sentiment']['temperature']} relevance=P_ent+P_contra>={threshold}"
    )
    with temporary_config_overrides(overrides):
        engine_cls = e2e()._load_v4_sentiment_engine_class()
        engine = engine_cls()
        tuple_pairs = [
            (pair.review_id, pair.sentence, pair.aspect, pair.nli_label, pair.weight)
            for pair in pairs
        ]
        results = engine.batch_analyze(tuple_pairs)

    sentiment_by_pair: dict[tuple[str, str], dict[str, float]] = {}
    raw_rows: list[dict[str, Any]] = []
    skipped = 0
    pair_by_key = {(pair.review_id, pair.aspect): pair for pair in pairs}
    for result in results:
        key = (str(result.review_id), str(result.aspect))
        pair = pair_by_key[key]
        p_ent = float(result.p_ent_pos)
        p_contra = float(result.p_ent_neg)
        p_neu = float(max(0.0, 1.0 - p_ent - p_contra))
        relevance = p_ent + p_contra
        passed = relevance >= threshold
        rating = float(np.clip(result.score, 1.0, 5.0))
        if passed:
            sentiment_by_pair[key] = {
                "rating": rating,
                "raw_rating": rating,
                "p_ent_pos": p_ent,
                "p_ent_neg": p_contra,
                "p_neutral": p_neu,
                "p_ent_plus_contra": relevance,
                "polarity": rating - 3.0,
                "raw_polarity": rating - 3.0,
                "negation_corrected": False,
                "negation_pattern": "",
                "negation_hit_lemma": "",
            }
        else:
            skipped += 1
        raw_rows.append(
            {
                "prediction_id": stable_id(result.review_id, result.aspect),
                "review_id": str(result.review_id),
                "nm_id": None,
                "aspect_name": _aspect_label_for_pair(pair),
                "aspect_key": str(result.aspect),
                "aspect_source": "discovery" if str(result.aspect).startswith("discovery::") else "vocab",
                "hypothesis_text": "{aspect} — это хорошо".format(aspect=pair.nli_label),
                "premise_text": str(result.sentence),
                "p_entailment": p_ent,
                "p_neutral": p_neu,
                "p_contradiction": p_contra,
                "raw_rating": rating,
                "passed_relevance_filter": bool(passed),
                "relevance_filter_value": relevance,
                "has_negation_match": False,
                "negation_correction_applied": False,
                "final_rating": rating,
            }
        )
    logger.log(f"[sentiment] kept={len(sentiment_by_pair)} skipped={skipped}")

    negation_stats = e2e()._apply_negation_corrections(
        sentiment_by_pair,
        reviews,
        aspect_by_id_by_category,
        discovery_by_product,
        logger,
    )
    review_by_id = {review.review_id: review for review in reviews}
    for row in raw_rows:
        review = review_by_id.get(row["review_id"])
        row["nm_id"] = int(review.nm_id) if review is not None else None
        scores = sentiment_by_pair.get((row["review_id"], row["aspect_key"]))
        if scores:
            row["has_negation_match"] = bool(scores.get("negation_pattern"))
            row["negation_correction_applied"] = bool(scores.get("negation_corrected", False))
            row["final_rating"] = float(scores["rating"])
    frame = pd.DataFrame(raw_rows)
    if "aspect_key" in frame.columns:
        frame = frame.drop(columns=["aspect_key"])
    return {
        "pairs": pairs,
        "sentiment_by_pair": sentiment_by_pair,
        "negation_stats": negation_stats,
        "nli_predictions": frame,
    }
