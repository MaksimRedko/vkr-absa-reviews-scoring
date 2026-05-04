from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_access import ReviewRecord


def load_few_shot_examples(config_dir: Path) -> list[dict[str, Any]]:
    path = config_dir / "few_shot_examples.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompt_template(config_dir: Path) -> str:
    path = config_dir / "prompt_template.md"
    if not path.exists():
        return "{few_shot}\n\n{batch_json}"
    return path.read_text(encoding="utf-8")


def build_batch_prompt(config_dir: Path, reviews: list[ReviewRecord]) -> str:
    template = load_prompt_template(config_dir)
    few_shot = load_few_shot_examples(config_dir)
    payload = {
        "items": [
            {
                "review_id": review.review_id,
                "nm_id": review.nm_id,
                "category": review.category,
                "rating": review.review_rating,
                "full_text": review.full_text,
                "gold_aspects": {item.aspect: item.rating for item in review.gold_aspects},
                "system_aspects": [
                    {
                        "prediction_id": item.prediction_id,
                        "aspect_name": item.aspect_name,
                        "aspect_source": item.aspect_source,
                        "final_rating": item.final_rating,
                        "premise_text": item.premise_text,
                        "hypothesis_text": item.hypothesis_text,
                        "evidence_fragments": [fragment.get("text", "") for fragment in item.evidence_fragments[:5]],
                        "cluster_phrases": item.cluster_phrases[:8],
                    }
                    for item in review.system_aspects
                ],
            }
            for review in reviews
        ]
    }
    few_shot_block = json.dumps(few_shot, ensure_ascii=False, indent=2)
    batch_block = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        template
        .replace("{few_shot}", few_shot_block)
        .replace("{batch_json}", batch_block)
    )
