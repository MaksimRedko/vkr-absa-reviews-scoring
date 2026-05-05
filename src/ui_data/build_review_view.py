from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _load_dataset_text(run_dir: Path, review_id: str) -> str:
    config_path = run_dir / "run_config.yaml"
    dataset = Path("data/dataset_final.csv")
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset = Path(cfg.get("gold_dataset_csv", dataset))
    if not dataset.is_absolute():
        dataset = Path(__file__).resolve().parents[2] / dataset
    if dataset.exists():
        df = pd.read_csv(dataset, dtype={"id": str})
        row = df[df["id"] == str(review_id)]
        if not row.empty:
            return str(row.iloc[0].get("full_text", ""))
    return ""


def build_review_view(run_dir: str | Path, review_id: str) -> dict[str, Any]:
    run_path = Path(run_dir)
    text = _load_dataset_text(run_path, review_id)
    spans: list[dict[str, Any]] = []
    discovered: list[dict[str, Any]] = []
    aggregated: dict[str, float] = {}

    candidates_path = run_path / "candidates.parquet"
    matches_path = run_path / "candidate_matches.parquet"
    nli_path = run_path / "nli_predictions.parquet"
    if candidates_path.exists() and matches_path.exists():
        cands = pd.read_parquet(candidates_path)
        matches = pd.read_parquet(matches_path)
        view = cands[cands["review_id"].astype(str) == str(review_id)].merge(matches, on="candidate_id", how="left")
        ratings = pd.DataFrame()
        if nli_path.exists():
            ratings = pd.read_parquet(nli_path)
        for _, row in view.iterrows():
            aspect = row.get("matched_aspect_id")
            if pd.isna(aspect):
                continue
            rating = None
            if not ratings.empty:
                hit = ratings[
                    (ratings["review_id"].astype(str) == str(review_id))
                    & (ratings["aspect_source"] == "vocab")
                ]
                if not hit.empty:
                    rating = float(hit.iloc[0]["final_rating"])
            spans.append(
                {
                    "start": int(row["start_offset"]),
                    "end": int(row["end_offset"]),
                    "text": row["text"],
                    "aspect": str(aspect),
                    "match_method": row.get("match_method"),
                    "rating": rating,
                    "color": "green" if rating is None or rating >= 3 else "red",
                }
            )

    for path in run_path.glob("predictions_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for review in payload.get("reviews", []):
            if str(review.get("review_id")) != str(review_id):
                continue
            for item in review.get("discovery_aspects", []):
                discovered.append({"medoid": item.get("medoid"), "rating": item.get("rating")})
            aggregated = payload.get("product_aggregated", {}).get("vocabulary", {})
            break

    return {
        "review_id": review_id,
        "text": text,
        "spans": spans,
        "discovered_aspects": discovered,
        "aggregated": aggregated,
    }
