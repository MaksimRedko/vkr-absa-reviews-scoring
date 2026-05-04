from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class GoldAspect:
    aspect: str
    rating: float


@dataclass(frozen=True)
class SystemAspect:
    prediction_id: str
    review_id: str
    nm_id: int
    category: str
    review_rating: float
    aspect_name: str
    aspect_source: str
    final_rating: float
    raw_rating: float | None
    premise_text: str
    hypothesis_text: str
    p_entailment: float | None
    p_neutral: float | None
    p_contradiction: float | None
    passed_relevance_filter: bool
    has_negation_match: bool
    negation_correction_applied: bool
    evidence_fragments: list[dict[str, Any]]
    cluster_phrases: list[str]


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    nm_id: int
    category: str
    review_rating: float
    full_text: str
    gold_aspects: list[GoldAspect]
    system_aspects: list[SystemAspect]


def parse_true_labels(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key).strip()] = float(value)
        except Exception:
            continue
    return out


def _default_dataset_path(root: Path) -> Path:
    return root / "data" / "dataset_final.csv"


def _default_run_dir(root: Path) -> Path:
    traced = sorted((root / "results").glob("*_traced"))
    return traced[-1] if traced else root / "results"


def discover_run_dirs(root: Path) -> list[Path]:
    runs = [path for path in (root / "results").glob("*_traced") if path.is_dir()]
    return sorted(runs, key=lambda path: path.name, reverse=True)


def load_review_records(root: Path, run_dir: Path, dataset_path: Path) -> list[ReviewRecord]:
    dataset = pd.read_csv(dataset_path, dtype={"id": str})
    required = {"id", "nm_id", "category", "rating", "full_text", "true_labels"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {sorted(missing)}")
    dataset = dataset.copy()
    dataset["review_id"] = dataset["id"].astype(str)
    dataset["gold_dict"] = dataset["true_labels"].apply(parse_true_labels)

    nli = pd.read_parquet(run_dir / "nli_predictions.parquet")
    candidates = pd.read_parquet(run_dir / "candidates.parquet")

    assignments_path = run_dir / "aspect_review_assignments.parquet"
    evidence_path = run_dir / "aspect_review_evidence.parquet"
    assignments = pd.read_parquet(assignments_path) if assignments_path.exists() else pd.DataFrame()
    evidence = pd.read_parquet(evidence_path) if evidence_path.exists() else pd.DataFrame()

    vocab_name_to_id = load_vocab_name_to_id(root)
    cluster_info = load_cluster_info(run_dir)
    evidence_lookup = build_evidence_lookup(assignments, evidence, candidates, vocab_name_to_id, cluster_info)

    merged = nli.merge(
        dataset[["review_id", "nm_id", "category", "rating", "full_text", "gold_dict"]],
        on=["review_id", "nm_id"],
        how="left",
        validate="many_to_one",
    )

    reviews: list[ReviewRecord] = []
    for review_id, review_rows in merged.groupby("review_id", sort=False):
        row0 = review_rows.iloc[0]
        gold_dict = row0["gold_dict"] if isinstance(row0["gold_dict"], dict) else {}
        gold_aspects = [
            GoldAspect(aspect=name, rating=float(score))
            for name, score in sorted(gold_dict.items())
        ]
        system_aspects: list[SystemAspect] = []
        for _, row in review_rows.sort_values(by=["aspect_source", "aspect_name"]).iterrows():
            evidence_key = (str(row["review_id"]), str(row["aspect_source"]), str(row["aspect_name"]), int(row["nm_id"]))
            evidence_pack = evidence_lookup.get(evidence_key, {})
            system_aspects.append(
                SystemAspect(
                    prediction_id=str(row["prediction_id"]),
                    review_id=str(row["review_id"]),
                    nm_id=int(row["nm_id"]),
                    category=str(row["category"]),
                    review_rating=float(row["rating"]),
                    aspect_name=str(row["aspect_name"]),
                    aspect_source=str(row["aspect_source"]),
                    final_rating=float(row["final_rating"]),
                    raw_rating=float(row["raw_rating"]) if pd.notna(row.get("raw_rating")) else None,
                    premise_text=str(row.get("premise_text", "")),
                    hypothesis_text=str(row.get("hypothesis_text", "")),
                    p_entailment=float(row["p_entailment"]) if pd.notna(row.get("p_entailment")) else None,
                    p_neutral=float(row["p_neutral"]) if pd.notna(row.get("p_neutral")) else None,
                    p_contradiction=float(row["p_contradiction"]) if pd.notna(row.get("p_contradiction")) else None,
                    passed_relevance_filter=bool(row.get("passed_relevance_filter", True)),
                    has_negation_match=bool(row.get("has_negation_match", False)),
                    negation_correction_applied=bool(row.get("negation_correction_applied", False)),
                    evidence_fragments=evidence_pack.get("fragments", []),
                    cluster_phrases=evidence_pack.get("cluster_phrases", []),
                )
            )
        reviews.append(
            ReviewRecord(
                review_id=str(review_id),
                nm_id=int(row0["nm_id"]),
                category=str(row0["category"]),
                review_rating=float(row0["rating"]),
                full_text=str(row0["full_text"]),
                gold_aspects=gold_aspects,
                system_aspects=system_aspects,
            )
        )

    return sorted(reviews, key=lambda item: (item.nm_id, item.review_id))


def load_vocab_name_to_id(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    vocab_roots = [root / "src" / "vocabulary" / "universal_aspects_v1.yaml"]
    vocab_roots.extend(sorted((root / "src" / "vocabulary" / "domain").glob("*.yaml")))
    for path in vocab_roots:
        if not path.exists():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item in payload.get("aspects", []):
            canonical = str(item.get("canonical_name", "")).strip()
            aspect_id = str(item.get("id", "")).strip()
            if canonical and aspect_id:
                out[canonical] = aspect_id
    return out


def load_cluster_info(run_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for path in run_dir.glob("clusters_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        nm_id = int(payload.get("nm_id", path.stem.split("_")[-1]))
        for cluster in payload.get("clusters", []):
            cluster_id = str(cluster.get("cluster_id"))
            top_phrases = []
            for item in cluster.get("top_phrases", [])[:10]:
                if isinstance(item, (list, tuple)) and item:
                    phrase = str(item[0])
                    count = item[1] if len(item) > 1 else ""
                    top_phrases.append(f"{phrase}:{count}")
                else:
                    top_phrases.append(str(item))
            info = {
                "cluster_id": cluster_id,
                "medoid_phrase": str(cluster.get("medoid_phrase", "") or cluster.get("medoid", "")),
                "top_phrases": top_phrases,
            }
            if info["medoid_phrase"]:
                out[(nm_id, info["medoid_phrase"])] = info
            out[(nm_id, cluster_id)] = info
    return out


def build_evidence_lookup(
    assignments: pd.DataFrame,
    evidence: pd.DataFrame,
    candidates: pd.DataFrame,
    vocab_name_to_id: dict[str, str],
    cluster_info: dict[tuple[int, str], dict[str, Any]],
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    if assignments.empty or evidence.empty:
        return {}
    deduped_candidates = candidates.drop_duplicates(subset=["candidate_id"], keep="last")
    candidate_lookup = deduped_candidates.set_index("candidate_id").to_dict(orient="index")

    fragments_by_assignment: dict[str, list[dict[str, Any]]] = {}
    for _, row in evidence.iterrows():
        candidate = candidate_lookup.get(str(row["candidate_id"]), {})
        fragment = {
            "candidate_id": str(row["candidate_id"]),
            "text": str(candidate.get("text", "")),
            "text_lemmatized": str(candidate.get("text_lemmatized", "")),
            "start_offset": int(row.get("start_offset", candidate.get("start_offset", -1))),
            "end_offset": int(row.get("end_offset", candidate.get("end_offset", -1))),
            "source": str(candidate.get("source", "")),
        }
        fragments_by_assignment.setdefault(str(row["assignment_id"]), []).append(fragment)

    out: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for _, row in assignments.iterrows():
        review_id = str(row["review_id"])
        aspect_type = str(row["aspect_type"])
        aspect_id = str(row["aspect_id"])
        fragments = fragments_by_assignment.get(str(row["assignment_id"]), [])
        nm_id = None
        if fragments:
            first_candidate_id = fragments[0]["candidate_id"]
            candidate_row = candidate_lookup.get(first_candidate_id, {})
            nm_id = int(candidate_row.get("nm_id", 0))
        if aspect_type == "vocab":
            aspect_name = next((name for name, vocab_id in vocab_name_to_id.items() if vocab_id == aspect_id), aspect_id)
            key = (review_id, "vocab", aspect_name, int(nm_id or 0))
            out[key] = {"fragments": fragments, "cluster_phrases": []}
            continue
        cluster = cluster_info.get((int(nm_id or 0), aspect_id), {})
        aspect_name = str(cluster.get("medoid_phrase", aspect_id))
        key = (review_id, "discovery", aspect_name, int(nm_id or 0))
        out[key] = {
            "fragments": fragments,
            "cluster_phrases": list(cluster.get("top_phrases", [])),
        }
    return out


def highlight_text(text: str, fragments: list[dict[str, Any]]) -> str:
    valid = [
        fragment
        for fragment in fragments
        if int(fragment.get("start_offset", -1)) >= 0 and int(fragment.get("end_offset", -1)) > int(fragment.get("start_offset", -1))
    ]
    if not text or not valid:
        return text
    valid.sort(key=lambda item: (int(item["start_offset"]), int(item["end_offset"])))
    parts: list[str] = []
    cursor = 0
    for fragment in valid:
        start = max(cursor, int(fragment["start_offset"]))
        end = int(fragment["end_offset"])
        if start >= end or start >= len(text):
            continue
        parts.append(escape(text[cursor:start]))
        parts.append(f"<mark>{escape(text[start:end])}</mark>")
        cursor = max(cursor, end)
    parts.append(escape(text[cursor:]))
    return "".join(parts)


def default_paths(root: Path) -> tuple[Path, Path]:
    return _default_run_dir(root), _default_dataset_path(root)
