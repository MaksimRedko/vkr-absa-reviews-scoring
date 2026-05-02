from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymorphy3
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import temporary_config_overrides
from scripts import run_phase2_baseline_matching as lexical
from src.pipeline.stages.common import stable_id
from src.schemas.models import SentimentPair

DEFAULT_RUN_DIR = ROOT / "results" / "20260425_183110_traced"
DEFAULT_WINDOW_TOKENS = 6
DEFAULT_RELEVANCE_THRESHOLD = 0.2
DEFAULT_TEMPERATURE = 0.7

MODE_A = "mode_a_current_baseline"
MODE_B = "mode_b_sentence_evidence"
MODE_C = "mode_c_window_evidence"
MODE_D = "mode_d_multi_evidence"
MODE_D_WEIGHTED = "mode_d_multi_evidence_weighted_relevance"

SINGLE_HYPOTHESIS_TEMPLATE = "{aspect} — это хорошо"
DUAL_HYPOTHESIS_POS_TEMPLATE = "В этом фрагменте {aspect} оценивается положительно"
DUAL_HYPOTHESIS_NEG_TEMPLATE = "В этом фрагменте {aspect} оценивается отрицательно"

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;])\s+")
_MORPH = pymorphy3.MorphAnalyzer()
_LEMMA_CACHE: dict[str, tuple[str, ...]] = {}
_REFERENCE_MODULE: Any | None = None


@dataclass(slots=True)
class BenchmarkContext:
    run_dir: Path
    dataset_path: Path
    run_config: dict[str, Any]
    reviews: list[Any]
    reviews_by_id: dict[str, Any]
    term_to_aspects_by_category: dict[str, dict[str, set[str]]]
    aspect_by_id_by_category: dict[str, dict[str, Any]]
    assignments: pd.DataFrame
    evidence: pd.DataFrame
    discovery_assignment_count: int
    discovery_assignments_without_evidence: int


def _reference() -> Any:
    global _REFERENCE_MODULE
    if _REFERENCE_MODULE is not None:
        return _REFERENCE_MODULE
    from src.pipeline.reference import e2e

    _REFERENCE_MODULE = e2e()
    return _REFERENCE_MODULE


def _resolve_path(base_dir: Path, raw_path: str | None, fallback: Path | None = None) -> Path:
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        return path
    if fallback is None:
        raise ValueError("path is required")
    return fallback.resolve()


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"run_config.yaml not found in {run_dir}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _lemmas(text: str) -> tuple[str, ...]:
    cached = _LEMMA_CACHE.get(text)
    if cached is not None:
        return cached
    out: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if not any(ch.isalpha() for ch in token):
            continue
        out.append(_MORPH.parse(token)[0].normal_form)
    result = tuple(out)
    _LEMMA_CACHE[text] = result
    return result


def sentence_spans(text: str) -> list[dict[str, Any]]:
    text = str(text)
    if not text.strip():
        return []
    spans: list[dict[str, Any]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.start()
        chunk = text[start:end].strip()
        if chunk:
            left = text.find(chunk, start, end)
            spans.append({"text": chunk, "start": left, "end": left + len(chunk)})
        start = match.end()
    tail = text[start:].strip()
    if tail:
        left = text.find(tail, start)
        spans.append({"text": tail, "start": left, "end": left + len(tail)})
    if not spans:
        return [{"text": text.strip(), "start": 0, "end": len(text.strip())}]
    return spans


def tokenize_with_spans(text: str) -> list[dict[str, Any]]:
    return [
        {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "lemma": _MORPH.parse(match.group(0).lower())[0].normal_form,
        }
        for match in _TOKEN_RE.finditer(str(text))
    ]


def resolve_sentence_text(
    review_text: str,
    evidence_text: str,
    evidence_lemma_text: str,
    start_offset: int,
    end_offset: int,
) -> str:
    spans = sentence_spans(review_text)
    if not spans:
        return str(review_text).strip()

    if start_offset >= 0:
        for span in spans:
            if span["start"] <= start_offset < span["end"]:
                return str(span["text"]).strip()

    lowered_text = str(review_text).casefold()
    lowered_evidence = str(evidence_text).strip().casefold()
    if lowered_evidence:
        idx = lowered_text.find(lowered_evidence)
        if idx >= 0:
            for span in spans:
                if span["start"] <= idx < span["end"]:
                    return str(span["text"]).strip()

    evidence_lemmas = set(_lemmas(evidence_lemma_text or evidence_text))
    best_span = spans[0]
    best_score = -1
    best_coverage = -1.0
    for span in spans:
        sentence_lemmas = set(_lemmas(span["text"]))
        overlap = evidence_lemmas & sentence_lemmas
        score = len(overlap)
        coverage = score / max(len(evidence_lemmas), 1)
        if score > best_score or (score == best_score and coverage > best_coverage):
            best_span = span
            best_score = score
            best_coverage = coverage
    return str(best_span["text"]).strip()


def extract_window_text(
    review_text: str,
    evidence_text: str,
    evidence_lemma_text: str,
    start_offset: int,
    end_offset: int,
    window_tokens: int,
    sentence_fallback: str,
) -> str:
    tokens = tokenize_with_spans(review_text)
    if not tokens:
        return sentence_fallback.strip()

    anchor_indices: list[int] = []
    if start_offset >= 0 and end_offset > start_offset:
        anchor_indices = [
            index
            for index, token in enumerate(tokens)
            if token["start"] < end_offset and token["end"] > start_offset
        ]

    if not anchor_indices and evidence_text.strip():
        lowered_text = str(review_text).casefold()
        lowered_evidence = str(evidence_text).strip().casefold()
        idx = lowered_text.find(lowered_evidence)
        if idx >= 0:
            phrase_end = idx + len(evidence_text)
            anchor_indices = [
                index
                for index, token in enumerate(tokens)
                if token["start"] < phrase_end and token["end"] > idx
            ]

    if not anchor_indices:
        evidence_lemmas = set(_lemmas(evidence_lemma_text or evidence_text))
        anchor_indices = [
            index
            for index, token in enumerate(tokens)
            if token["lemma"] in evidence_lemmas
        ]

    if not anchor_indices:
        return sentence_fallback.strip()

    left = max(0, min(anchor_indices) - int(window_tokens))
    right = min(len(tokens) - 1, max(anchor_indices) + int(window_tokens))
    return str(review_text[tokens[left]["start"] : tokens[right]["end"]]).strip()


def find_phrase_occurrences(review_text: str, phrases: list[str]) -> list[dict[str, Any]]:
    lowered_text = str(review_text).casefold()
    raw_hits: list[dict[str, Any]] = []
    seen_phrases: set[str] = set()
    unique_phrases: list[str] = []
    for phrase in phrases:
        cleaned = str(phrase).strip()
        lowered = cleaned.casefold()
        if not cleaned or lowered in seen_phrases:
            continue
        seen_phrases.add(lowered)
        unique_phrases.append(cleaned)

    for rank, phrase in enumerate(unique_phrases):
        lowered_phrase = phrase.casefold()
        start = 0
        while True:
            idx = lowered_text.find(lowered_phrase, start)
            if idx < 0:
                break
            raw_hits.append(
                {
                    "start": idx,
                    "end": idx + len(phrase),
                    "phrase": str(review_text[idx : idx + len(phrase)]),
                    "phrase_template": phrase,
                    "rank": rank,
                    "token_len": len(_TOKEN_RE.findall(phrase)),
                    "char_len": len(phrase),
                }
            )
            start = idx + 1

    raw_hits.sort(key=lambda item: (item["start"], -item["token_len"], -item["char_len"], item["rank"]))
    selected: list[dict[str, Any]] = []
    for hit in raw_hits:
        overlap_index = next(
            (
                index
                for index, existing in enumerate(selected)
                if not (hit["end"] <= existing["start"] or hit["start"] >= existing["end"])
            ),
            None,
        )
        if overlap_index is None:
            selected.append(hit)
            continue
        existing = selected[overlap_index]
        if hit["start"] != existing["start"]:
            continue
        hit_key = (hit["token_len"], hit["char_len"], -hit["rank"])
        existing_key = (existing["token_len"], existing["char_len"], -existing["rank"])
        if hit_key > existing_key:
            selected[overlap_index] = hit
    return sorted(selected, key=lambda item: item["start"])


def _load_clusters(run_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    clusters: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(run_dir.glob("clusters_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        nm_id = int(payload["nm_id"])
        for cluster in payload.get("clusters", []):
            cluster_id = int(cluster["cluster_id"])
            clusters[(nm_id, cluster_id)] = cluster
    return clusters


def _load_prediction_payloads(run_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("predictions_*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _build_assignment_frame(
    reviews_by_id: dict[str, Any],
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    aspect_by_id_by_category: dict[str, dict[str, Any]],
    prediction_payloads: list[dict[str, Any]],
) -> tuple[pd.DataFrame, int]:
    matched = matches[matches["matched_aspect_id"].notna()].copy()
    vocab_rows: list[dict[str, Any]] = []
    if not matched.empty:
        merged = candidates.merge(matched, on="candidate_id", how="inner")
        merged = merged.sort_values(["review_id", "matched_aspect_id", "start_offset", "candidate_id"])
        for row in merged.itertuples(index=False):
            if str(row.review_id) not in reviews_by_id:
                continue
            review = reviews_by_id[str(row.review_id)]
            aspect_id = str(row.matched_aspect_id)
            aspect = aspect_by_id_by_category[review.category_id].get(aspect_id)
            aspect_name = aspect.canonical_name if aspect is not None else aspect_id
            vocab_rows.append(
                {
                    "assignment_id": stable_id(row.review_id, "vocab", aspect_id),
                    "review_id": str(row.review_id),
                    "nm_id": int(row.nm_id),
                    "category_id": str(row.category_id),
                    "aspect_key": f"vocab::{aspect_id}",
                    "aspect_name": str(aspect_name),
                    "aspect_source": "vocab",
                    "review_text": str(review.text),
                    "gold_matches_json": "{}",
                }
            )
    assignments = pd.DataFrame(vocab_rows).drop_duplicates(subset=["review_id", "aspect_key"])

    discovery_rows: list[dict[str, Any]] = []
    discovery_assignment_count = 0
    for payload in prediction_payloads:
        nm_id = int(payload["nm_id"])
        category_id = str(payload["category"])
        for review_payload in payload.get("reviews", []):
            review_id = str(review_payload["review_id"])
            if review_id not in reviews_by_id:
                continue
            review = reviews_by_id[review_id]
            for item in review_payload.get("discovery_aspects", []):
                cluster_id = int(item["cluster_id"])
                discovery_assignment_count += 1
                discovery_rows.append(
                    {
                        "assignment_id": stable_id(review_id, "discovery", nm_id, cluster_id),
                        "review_id": review_id,
                        "nm_id": nm_id,
                        "category_id": category_id,
                        "aspect_key": f"discovery::{nm_id}::{cluster_id}",
                        "aspect_name": str(item["medoid"]),
                        "aspect_source": "discovery",
                        "review_text": str(review.text),
                        "gold_matches_json": json.dumps(item.get("gold_matches", {}), ensure_ascii=False, sort_keys=True),
                    }
                )
    discovery_df = pd.DataFrame(discovery_rows).drop_duplicates(subset=["review_id", "aspect_key"])
    if assignments.empty:
        combined = discovery_df
    elif discovery_df.empty:
        combined = assignments
    else:
        combined = pd.concat([assignments, discovery_df], ignore_index=True)
    return combined.sort_values(["review_id", "aspect_source", "aspect_name"]).reset_index(drop=True), discovery_assignment_count


def _build_vocab_evidence_rows(
    reviews_by_id: dict[str, Any],
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    aspect_by_id_by_category: dict[str, dict[str, Any]],
    window_tokens: int,
) -> list[dict[str, Any]]:
    evidence_rows: list[dict[str, Any]] = []
    matched = matches[matches["matched_aspect_id"].notna()].copy()
    if matched.empty:
        return evidence_rows
    merged = candidates.merge(matched, on="candidate_id", how="inner")
    merged = merged.sort_values(["review_id", "matched_aspect_id", "start_offset", "candidate_id"])
    for row in merged.itertuples(index=False):
        if str(row.review_id) not in reviews_by_id:
            continue
        review = reviews_by_id[str(row.review_id)]
        aspect_id = str(row.matched_aspect_id)
        aspect = aspect_by_id_by_category[review.category_id].get(aspect_id)
        aspect_name = aspect.canonical_name if aspect is not None else aspect_id
        start_offset = int(row.start_offset)
        end_offset = int(row.end_offset)
        sentence_text = resolve_sentence_text(
            review.text,
            str(row.text),
            str(row.text_lemmatized),
            start_offset,
            end_offset,
        )
        window_text = extract_window_text(
            review.text,
            str(row.text),
            str(row.text_lemmatized),
            start_offset,
            end_offset,
            window_tokens,
            sentence_text,
        )
        evidence_rows.append(
            {
                "evidence_id": stable_id(row.review_id, aspect_id, row.candidate_id),
                "review_id": str(row.review_id),
                "nm_id": int(row.nm_id),
                "category_id": str(row.category_id),
                "aspect_key": f"vocab::{aspect_id}",
                "aspect_name": str(aspect_name),
                "aspect_source": "vocab",
                "candidate_id": str(row.candidate_id),
                "cluster_id": None,
                "evidence_text": str(row.text),
                "evidence_lemma_text": str(row.text_lemmatized),
                "sentence_text": sentence_text,
                "window_text": window_text,
                "review_text": str(review.text),
                "start_offset": start_offset,
                "end_offset": end_offset,
                "gold_matches_json": "{}",
            }
        )
    return evidence_rows


def _build_discovery_evidence_rows(
    reviews_by_id: dict[str, Any],
    prediction_payloads: list[dict[str, Any]],
    clusters: dict[tuple[int, int], dict[str, Any]],
    window_tokens: int,
) -> list[dict[str, Any]]:
    evidence_rows: list[dict[str, Any]] = []
    for payload in prediction_payloads:
        nm_id = int(payload["nm_id"])
        category_id = str(payload["category"])
        for review_payload in payload.get("reviews", []):
            review_id = str(review_payload["review_id"])
            if review_id not in reviews_by_id:
                continue
            review = reviews_by_id[review_id]
            for item in review_payload.get("discovery_aspects", []):
                cluster_id = int(item["cluster_id"])
                cluster = clusters.get((nm_id, cluster_id))
                if cluster is None:
                    continue
                phrases = [str(pair[0]) for pair in cluster.get("top_phrases", []) if pair]
                if cluster.get("medoid_phrase"):
                    phrases.append(str(cluster["medoid_phrase"]))
                hits = find_phrase_occurrences(review.text, phrases)
                for index, hit in enumerate(hits):
                    sentence_text = resolve_sentence_text(
                        review.text,
                        str(hit["phrase"]),
                        str(hit["phrase_template"]),
                        int(hit["start"]),
                        int(hit["end"]),
                    )
                    window_text = extract_window_text(
                        review.text,
                        str(hit["phrase"]),
                        str(hit["phrase_template"]),
                        int(hit["start"]),
                        int(hit["end"]),
                        window_tokens,
                        sentence_text,
                    )
                    evidence_rows.append(
                        {
                            "evidence_id": stable_id(review_id, "discovery", nm_id, cluster_id, index, hit["start"]),
                            "review_id": review_id,
                            "nm_id": nm_id,
                            "category_id": category_id,
                            "aspect_key": f"discovery::{nm_id}::{cluster_id}",
                            "aspect_name": str(item["medoid"]),
                            "aspect_source": "discovery",
                            "candidate_id": None,
                            "cluster_id": cluster_id,
                            "evidence_text": str(hit["phrase"]),
                            "evidence_lemma_text": str(hit["phrase_template"]),
                            "sentence_text": sentence_text,
                            "window_text": window_text,
                            "review_text": str(review.text),
                            "start_offset": int(hit["start"]),
                            "end_offset": int(hit["end"]),
                            "gold_matches_json": json.dumps(item.get("gold_matches", {}), ensure_ascii=False, sort_keys=True),
                        }
                    )
    return evidence_rows


def _resolve_assignment_metadata(
    reviews_by_id: dict[str, Any],
    review_id: str,
    aspect_type: str,
    aspect_id: str,
    aspect_by_id_by_category: dict[str, dict[str, Any]],
    clusters: dict[tuple[int, int], dict[str, Any]],
    assignment_id: str,
) -> dict[str, Any] | None:
    review = reviews_by_id.get(review_id)
    if review is None:
        return None
    if aspect_type == "vocab":
        aspect = aspect_by_id_by_category[review.category_id].get(aspect_id)
        aspect_name = aspect.canonical_name if aspect is not None else aspect_id
        aspect_key = f"vocab::{aspect_id}"
        cluster_id = None
        gold_matches_json = "{}"
    elif aspect_type == "discovery":
        cluster_id = int(aspect_id)
        cluster = clusters.get((int(review.nm_id), cluster_id), {})
        aspect_name = str(cluster.get("medoid_phrase") or cluster_id)
        aspect_key = f"discovery::{int(review.nm_id)}::{cluster_id}"
        gold_matches_json = json.dumps(cluster.get("gold_matches", {}), ensure_ascii=False, sort_keys=True)
    else:
        raise ValueError(f"unsupported aspect_type: {aspect_type}")
    return {
        "assignment_id": assignment_id,
        "review_id": review_id,
        "nm_id": int(review.nm_id),
        "category_id": str(review.category_id),
        "aspect_key": aspect_key,
        "aspect_name": str(aspect_name),
        "aspect_source": aspect_type,
        "cluster_id": cluster_id,
        "review_text": str(review.text),
        "gold_matches_json": gold_matches_json,
    }


def _build_assignment_rows_from_artifact(
    reviews_by_id: dict[str, Any],
    assignments: pd.DataFrame,
    aspect_by_id_by_category: dict[str, dict[str, Any]],
    clusters: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if assignments.empty:
        return rows
    ordered = assignments.sort_values(["review_id", "aspect_type", "aspect_id", "assignment_id"])
    for row in ordered.itertuples(index=False):
        payload = _resolve_assignment_metadata(
            reviews_by_id,
            str(row.review_id),
            str(row.aspect_type),
            str(row.aspect_id),
            aspect_by_id_by_category,
            clusters,
            str(row.assignment_id),
        )
        if payload is not None:
            rows.append(payload)
    return rows


def _empty_assignment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "assignment_id",
            "review_id",
            "nm_id",
            "category_id",
            "aspect_key",
            "aspect_name",
            "aspect_source",
            "cluster_id",
            "review_text",
            "gold_matches_json",
        ]
    )


def _build_evidence_rows_from_artifact(
    reviews_by_id: dict[str, Any],
    candidates: pd.DataFrame,
    evidence: pd.DataFrame,
    aspect_by_id_by_category: dict[str, dict[str, Any]],
    clusters: dict[tuple[int, int], dict[str, Any]],
    window_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if evidence.empty or candidates.empty:
        return rows
    fragments = candidates[
        [
            "candidate_id",
            "review_id",
            "text",
            "text_lemmatized",
            "start_offset",
            "end_offset",
        ]
    ].copy()
    merged = evidence.merge(
        fragments,
        on=["candidate_id", "review_id", "start_offset", "end_offset"],
        how="inner",
    )
    merged = merged.sort_values(["review_id", "aspect_type", "aspect_id", "assignment_id", "start_offset", "candidate_id"])
    for row in merged.itertuples(index=False):
        review_id = str(row.review_id)
        review = reviews_by_id.get(review_id)
        if review is None:
            continue
        metadata = _resolve_assignment_metadata(
            reviews_by_id,
            review_id,
            str(row.aspect_type),
            str(row.aspect_id),
            aspect_by_id_by_category,
            clusters,
            str(row.assignment_id),
        )
        if metadata is None:
            continue
        start_offset = int(row.start_offset)
        end_offset = int(row.end_offset)
        sentence_text = resolve_sentence_text(
            review.text,
            str(row.text),
            str(row.text_lemmatized),
            start_offset,
            end_offset,
        )
        window_text = extract_window_text(
            review.text,
            str(row.text),
            str(row.text_lemmatized),
            start_offset,
            end_offset,
            window_tokens,
            sentence_text,
        )
        rows.append(
            {
                "evidence_id": str(row.evidence_id),
                **metadata,
                "candidate_id": str(row.candidate_id),
                "evidence_text": str(row.text),
                "evidence_lemma_text": str(row.text_lemmatized),
                "sentence_text": sentence_text,
                "window_text": window_text,
                "start_offset": start_offset,
                "end_offset": end_offset,
            }
        )
    return rows


def _empty_evidence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "evidence_id",
            "assignment_id",
            "review_id",
            "nm_id",
            "category_id",
            "aspect_key",
            "aspect_name",
            "aspect_source",
            "cluster_id",
            "review_text",
            "gold_matches_json",
            "candidate_id",
            "evidence_text",
            "evidence_lemma_text",
            "sentence_text",
            "window_text",
            "start_offset",
            "end_offset",
        ]
    )


def _build_assignment_mode_frame(
    assignments: pd.DataFrame,
    mode_id: str,
) -> pd.DataFrame:
    frame = assignments.copy()
    if frame.empty:
        return frame
    if mode_id == MODE_A:
        frame["premise_text"] = frame["review_text"]
        frame["premise_kind"] = "full_review"
    elif mode_id == MODE_B:
        frame["premise_text"] = frame["sentence_text"]
        frame["premise_kind"] = "sentence"
    elif mode_id == MODE_C:
        frame["premise_text"] = frame["window_text"]
        frame["premise_kind"] = "window"
    elif mode_id in {MODE_D, MODE_D_WEIGHTED}:
        frame["premise_text"] = frame["sentence_text"]
        frame["premise_kind"] = "multi_sentence"
    else:
        raise ValueError(f"unsupported mode_id: {mode_id}")
    return frame.reset_index(drop=True)


def assert_shared_single_mode_lengths(assignments: pd.DataFrame) -> None:
    counts = {
        MODE_A: len(_build_assignment_mode_frame(assignments, MODE_A)),
        MODE_B: len(_build_assignment_mode_frame(assignments, MODE_B)),
        MODE_C: len(_build_assignment_mode_frame(assignments, MODE_C)),
    }
    if len(set(counts.values())) != 1:
        raise AssertionError(f"A/B/C input size mismatch: {counts}")


def load_benchmark_context(
    run_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    dataset_path: str | Path | None = None,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
) -> BenchmarkContext:
    run_dir = Path(run_dir).resolve()
    run_config = _load_run_config(run_dir)
    dataset = _resolve_path(run_dir, str(dataset_path) if dataset_path is not None else run_config.get("gold_dataset_csv"))
    reference = _reference()
    reviews = reference._load_reviews(dataset)
    reviews_by_id = {str(review.review_id): review for review in reviews}

    categories = {review.category_id for review in reviews}
    core_vocab_path = _resolve_path(run_dir, run_config.get("core_vocab"))
    domain_vocab_dir = _resolve_path(run_dir, run_config.get("domain_vocab_dir"))
    _, term_to_aspects_by_category, aspect_by_id_by_category = reference._build_hybrid_vocab(
        core_vocab_path,
        domain_vocab_dir,
        categories,
    )

    candidates = pd.read_parquet(run_dir / "candidates.parquet")
    matches = pd.read_parquet(run_dir / "candidate_matches.parquet")
    prediction_payloads = _load_prediction_payloads(run_dir)
    clusters = _load_clusters(run_dir)

    assignment_artifact_path = run_dir / "aspect_review_assignments.parquet"
    evidence_artifact_path = run_dir / "aspect_review_evidence.parquet"
    if assignment_artifact_path.exists() and evidence_artifact_path.exists():
        raw_assignments = pd.read_parquet(assignment_artifact_path)
        raw_evidence = pd.read_parquet(evidence_artifact_path)
        assignment_rows = _build_assignment_rows_from_artifact(
            reviews_by_id,
            raw_assignments,
            aspect_by_id_by_category,
            clusters,
        )
        evidence_rows = _build_evidence_rows_from_artifact(
            reviews_by_id,
            candidates,
            raw_evidence,
            aspect_by_id_by_category,
            clusters,
            window_tokens,
        )
        assignments = pd.DataFrame(assignment_rows) if assignment_rows else _empty_assignment_frame()
        evidence = pd.DataFrame(evidence_rows) if evidence_rows else _empty_evidence_frame()
        evidence["has_valid_offset"] = evidence["start_offset"].fillna(-1).astype(int) >= 0
        evidence["start_sort"] = np.where(
            evidence["has_valid_offset"],
            evidence["start_offset"].fillna(0).astype(int),
            10**9,
        )
        evidence["evidence_len"] = evidence["evidence_text"].fillna("").astype(str).str.len()
        evidence = evidence.sort_values(
            ["review_id", "aspect_key", "start_sort", "evidence_len", "evidence_id"],
            ascending=[True, True, True, False, True],
        ).reset_index(drop=True)
        assignments = assignments.sort_values(
            ["review_id", "aspect_key", "assignment_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        single_evidence = select_single_evidence_rows(evidence)
        counts = {
            MODE_A: len(assignments),
            MODE_B: len(single_evidence),
            MODE_C: len(single_evidence),
        }
        if len(set(counts.values())) != 1:
            raise AssertionError(f"A/B/C input size mismatch: {counts}")
        return BenchmarkContext(
            run_dir=run_dir,
            dataset_path=dataset,
            run_config=run_config,
            reviews=reviews,
            reviews_by_id=reviews_by_id,
            term_to_aspects_by_category=term_to_aspects_by_category,
            aspect_by_id_by_category=aspect_by_id_by_category,
            assignments=assignments,
            evidence=evidence,
            discovery_assignment_count=int((raw_assignments["aspect_type"] == "discovery").sum()) if "aspect_type" in raw_assignments.columns else 0,
            discovery_assignments_without_evidence=0,
        )

    assignments, discovery_assignment_count = _build_assignment_frame(
        reviews_by_id,
        candidates,
        matches,
        aspect_by_id_by_category,
        prediction_payloads,
    )

    evidence_rows = _build_vocab_evidence_rows(
        reviews_by_id,
        candidates,
        matches,
        aspect_by_id_by_category,
        window_tokens,
    )
    evidence_rows.extend(
        _build_discovery_evidence_rows(
            reviews_by_id,
            prediction_payloads,
            clusters,
            window_tokens,
        )
    )
    evidence = pd.DataFrame(evidence_rows)
    if evidence.empty:
        evidence = pd.DataFrame(
            columns=[
                "evidence_id",
                "review_id",
                "nm_id",
                "category_id",
                "aspect_key",
                "aspect_name",
                "aspect_source",
                "candidate_id",
                "cluster_id",
                "evidence_text",
                "evidence_lemma_text",
                "sentence_text",
                "window_text",
                "review_text",
                "start_offset",
                "end_offset",
                "gold_matches_json",
            ]
        )
    evidence["has_valid_offset"] = evidence["start_offset"].fillna(-1).astype(int) >= 0
    evidence["start_sort"] = np.where(
        evidence["has_valid_offset"],
        evidence["start_offset"].fillna(0).astype(int),
        10**9,
    )
    evidence["evidence_len"] = evidence["evidence_text"].fillna("").astype(str).str.len()
    evidence = evidence.sort_values(
        ["review_id", "aspect_key", "start_sort", "evidence_len", "evidence_id"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)

    discovery_evidence_keys = {
        (str(row.review_id), str(row.aspect_key))
        for row in evidence.itertuples(index=False)
        if str(row.aspect_source) == "discovery"
    }
    discovery_assignments_without_evidence = 0
    for row in assignments.itertuples(index=False):
        if str(row.aspect_source) != "discovery":
            continue
        if (str(row.review_id), str(row.aspect_key)) not in discovery_evidence_keys:
            discovery_assignments_without_evidence += 1

    return BenchmarkContext(
        run_dir=run_dir,
        dataset_path=dataset,
        run_config=run_config,
        reviews=reviews,
        reviews_by_id=reviews_by_id,
        term_to_aspects_by_category=term_to_aspects_by_category,
        aspect_by_id_by_category=aspect_by_id_by_category,
        assignments=assignments,
        evidence=evidence,
        discovery_assignment_count=discovery_assignment_count,
        discovery_assignments_without_evidence=discovery_assignments_without_evidence,
    )


def select_single_evidence_rows(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return evidence.copy()
    ordered = evidence.sort_values(
        ["review_id", "aspect_key", "start_sort", "evidence_len", "evidence_id"],
        ascending=[True, True, True, False, True],
    )
    return ordered.drop_duplicates(subset=["review_id", "aspect_key"], keep="first").reset_index(drop=True)


def build_mode_input_frame(
    context: BenchmarkContext,
    mode_id: str,
) -> pd.DataFrame:
    if "assignment_id" in context.assignments.columns and "evidence_id" in context.evidence.columns:
        if mode_id == MODE_A:
            return _build_assignment_mode_frame(context.assignments, mode_id)
        if mode_id == MODE_B:
            frame = select_single_evidence_rows(context.evidence)
            frame["premise_text"] = frame["sentence_text"]
            frame["premise_kind"] = "sentence"
            return frame.reset_index(drop=True)
        if mode_id == MODE_C:
            frame = select_single_evidence_rows(context.evidence)
            frame["premise_text"] = frame["window_text"]
            frame["premise_kind"] = "window"
            return frame.reset_index(drop=True)
        if mode_id in {MODE_D, MODE_D_WEIGHTED}:
            frame = context.evidence.copy()
            frame["premise_text"] = frame["sentence_text"]
            frame["premise_kind"] = "multi_sentence"
            return frame.reset_index(drop=True)

    if mode_id == MODE_A:
        frame = context.assignments.copy()
        frame["premise_text"] = frame["review_text"]
        frame["premise_kind"] = "full_review"
        frame["evidence_count_hint"] = 1
        return frame.reset_index(drop=True)

    if mode_id == MODE_B:
        frame = select_single_evidence_rows(context.evidence)
        frame["premise_text"] = frame["sentence_text"]
        frame["premise_kind"] = "sentence"
        return frame.reset_index(drop=True)

    if mode_id == MODE_C:
        frame = select_single_evidence_rows(context.evidence)
        frame["premise_text"] = frame["window_text"]
        frame["premise_kind"] = "window"
        return frame.reset_index(drop=True)

    if mode_id in {MODE_D, MODE_D_WEIGHTED}:
        frame = context.evidence.copy()
        frame["premise_text"] = frame["sentence_text"]
        frame["premise_kind"] = "multi_sentence"
        return frame.reset_index(drop=True)

    raise ValueError(f"unsupported mode_id: {mode_id}")


def _load_single_hypothesis_engine_class(run_dir: Path, run_config: dict[str, Any]) -> Any:
    raw_path = run_config.get("models", {}).get("sentiment_engine_source")
    if not raw_path:
        raise ValueError("models.sentiment_engine_source is missing in run_config")
    engine_path = _resolve_path(run_dir, raw_path)
    spec = importlib.util.spec_from_file_location("sentiment_single_hypothesis_benchmark", engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load single-hypothesis engine from {engine_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SentimentEngine


def _run_single_hypothesis(
    frame: pd.DataFrame,
    context: BenchmarkContext,
    *,
    temperature: float,
    relevance_threshold: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    sent_cfg = context.run_config.get("sentiment", {})
    engine_cls = _load_single_hypothesis_engine_class(context.run_dir, context.run_config)
    overrides = {
        "sentiment": {
            "temperature": float(temperature),
            "hypothesis_template_pos": SINGLE_HYPOTHESIS_TEMPLATE,
            "persistent_nli_cache_enabled": bool(sent_cfg.get("persistent_nli_cache_enabled", True)),
            "persistent_nli_cache_path": str(sent_cfg.get("persistent_nli_cache_path", "./cache/nli_global.sqlite3")),
        }
    }
    tuples = [
        (str(row.review_id), str(row.premise_text), str(row.aspect_key), str(row.aspect_name), 1.0)
        for row in frame.itertuples(index=False)
    ]
    with temporary_config_overrides(overrides):
        engine = engine_cls()
        results = engine.batch_analyze(tuples)
        cache_stats = engine.get_cache_stats() if hasattr(engine, "get_cache_stats") else {}

    rows: list[dict[str, Any]] = []
    for source_row, result in zip(frame.itertuples(index=False), results, strict=True):
        p_ent = float(result.p_ent_pos)
        p_contra = float(result.p_ent_neg)
        p_neutral = float(max(0.0, 1.0 - p_ent - p_contra))
        relevance = p_ent + p_contra
        rows.append(
            {
                "mode_id": MODE_A,
                "engine_variant": "single_hypothesis_v4",
                "evidence_id": str(source_row.assignment_id),
                "review_id": str(source_row.review_id),
                "nm_id": int(source_row.nm_id),
                "category_id": str(source_row.category_id),
                "aspect_key": str(source_row.aspect_key),
                "aspect_name": str(source_row.aspect_name),
                "aspect_source": str(source_row.aspect_source),
                "premise_kind": str(source_row.premise_kind),
                "premise_text": str(source_row.premise_text),
                "hypothesis_pos_text": SINGLE_HYPOTHESIS_TEMPLATE.format(aspect=str(source_row.aspect_name)),
                "hypothesis_neg_text": "",
                "p_entailment_pos": p_ent,
                "p_entailment_neg": p_contra,
                "p_neutral": p_neutral,
                "relevance_filter_value": relevance,
                "passed_relevance_filter": bool(relevance >= relevance_threshold),
                "raw_rating": float(result.score),
                "final_rating": float(result.score),
                "gold_matches_json": str(source_row.gold_matches_json),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["cache_stats"] = cache_stats
    return out


def _run_dual_hypothesis(
    frame: pd.DataFrame,
    context: BenchmarkContext,
    *,
    temperature: float,
    relevance_threshold: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    sent_cfg = context.run_config.get("sentiment", {})
    from src.stages.sentiment import SentimentEngine

    overrides = {
        "sentiment": {
            "temperature": float(temperature),
            "hypothesis_template_pos": DUAL_HYPOTHESIS_POS_TEMPLATE,
            "hypothesis_template_neg": DUAL_HYPOTHESIS_NEG_TEMPLATE,
            "persistent_nli_cache_enabled": bool(sent_cfg.get("persistent_nli_cache_enabled", True)),
            "persistent_nli_cache_path": str(sent_cfg.get("persistent_nli_cache_path", "./cache/nli_global.sqlite3")),
        }
    }
    pairs = [
        SentimentPair(
            review_id=str(row.review_id),
            sentence=str(row.premise_text),
            aspect=str(row.aspect_key),
            nli_label=str(row.aspect_name),
            weight=1.0,
        )
        for row in frame.itertuples(index=False)
    ]
    with temporary_config_overrides(overrides):
        engine = SentimentEngine()
        results = engine.batch_analyze(pairs)
        cache_stats = engine.get_cache_stats() if hasattr(engine, "get_cache_stats") else {}

    rows: list[dict[str, Any]] = []
    for source_row, result in zip(frame.itertuples(index=False), results, strict=True):
        p_pos = float(result.p_ent_pos)
        p_neg = float(result.p_ent_neg)
        relevance = p_pos + p_neg
        rows.append(
            {
                "mode_id": str(frame.iloc[0]["mode_id"]) if "mode_id" in frame.columns and not frame.empty else "",
                "engine_variant": "dual_hypothesis_v5",
                "evidence_id": str(source_row.evidence_id),
                "review_id": str(source_row.review_id),
                "nm_id": int(source_row.nm_id),
                "category_id": str(source_row.category_id),
                "aspect_key": str(source_row.aspect_key),
                "aspect_name": str(source_row.aspect_name),
                "aspect_source": str(source_row.aspect_source),
                "premise_kind": str(source_row.premise_kind),
                "premise_text": str(source_row.premise_text),
                "hypothesis_pos_text": DUAL_HYPOTHESIS_POS_TEMPLATE.format(aspect=str(source_row.aspect_name)),
                "hypothesis_neg_text": DUAL_HYPOTHESIS_NEG_TEMPLATE.format(aspect=str(source_row.aspect_name)),
                "p_entailment_pos": p_pos,
                "p_entailment_neg": p_neg,
                "p_neutral": float(max(0.0, 1.0 - p_pos - p_neg)),
                "relevance_filter_value": relevance,
                "passed_relevance_filter": bool(relevance >= relevance_threshold),
                "raw_rating": float(result.score),
                "final_rating": float(result.score),
                "gold_matches_json": str(source_row.gold_matches_json),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["cache_stats"] = cache_stats
    return out


def run_inference(
    frame: pd.DataFrame,
    context: BenchmarkContext,
    mode_id: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    frame["mode_id"] = mode_id
    if mode_id == MODE_A:
        return _run_single_hypothesis(
            frame,
            context,
            temperature=temperature,
            relevance_threshold=relevance_threshold,
        )
    return _run_dual_hypothesis(
        frame,
        context,
        temperature=temperature,
        relevance_threshold=relevance_threshold,
    )


def aggregate_review_aspect_scores(
    predictions: pd.DataFrame,
    *,
    aggregation: str = "max_relevance",
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    kept = predictions[predictions["passed_relevance_filter"]].copy()
    if kept.empty:
        return pd.DataFrame(
            columns=[
                "review_id",
                "nm_id",
                "category_id",
                "aspect_key",
                "aspect_name",
                "aspect_source",
                "final_rating",
                "n_evidence_total",
                "n_evidence_kept",
                "aggregation_method",
                "selected_evidence_id",
                "selected_evidence_ids_json",
                "gold_matches_json",
            ]
        )

    total_counts = predictions.groupby(["review_id", "aspect_key"]).size().to_dict()
    rows: list[dict[str, Any]] = []
    for (review_id, aspect_key), group in kept.groupby(["review_id", "aspect_key"], sort=False):
        group = group.sort_values(
            ["relevance_filter_value", "raw_rating", "evidence_id"],
            ascending=[False, False, True],
        )
        top = group.iloc[0]
        if aggregation == "weighted_relevance" and len(group) > 1:
            weights = group["relevance_filter_value"].astype(float).to_numpy()
            rating = float(np.average(group["raw_rating"].astype(float).to_numpy(), weights=weights))
        else:
            rating = float(top["raw_rating"])
        rows.append(
            {
                "review_id": str(review_id),
                "nm_id": int(top["nm_id"]),
                "category_id": str(top["category_id"]),
                "aspect_key": str(aspect_key),
                "aspect_name": str(top["aspect_name"]),
                "aspect_source": str(top["aspect_source"]),
                "final_rating": rating,
                "n_evidence_total": int(total_counts.get((review_id, aspect_key), len(group))),
                "n_evidence_kept": int(len(group)),
                "aggregation_method": aggregation,
                "selected_evidence_id": str(top["evidence_id"]),
                "selected_evidence_ids_json": json.dumps(group["evidence_id"].astype(str).tolist(), ensure_ascii=False),
                "gold_matches_json": str(top["gold_matches_json"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["review_id", "aspect_source", "aspect_name"]).reset_index(drop=True)


def evaluate_review_level(
    context: BenchmarkContext,
    review_aspect_scores: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    by_review_aspect = {
        (str(row.review_id), str(row.aspect_key)): row
        for row in review_aspect_scores.itertuples(index=False)
    }
    by_review: dict[str, list[Any]] = defaultdict(list)
    for row in review_aspect_scores.itertuples(index=False):
        by_review[str(row.review_id)].append(row)

    hard_case_rows: list[dict[str, Any]] = []
    review_maes: list[float] = []
    review_round_maes: list[float] = []
    vocab_errors: list[float] = []
    discovery_errors: list[float] = []
    total_gold_pairs = 0
    evaluable_gold_pairs = 0

    for review in context.reviews:
        term_to_aspects = context.term_to_aspects_by_category[review.category_id]
        errors: list[float] = []
        round_errors: list[float] = []
        for gold_label, gold_rating in review.true_labels.items():
            total_gold_pairs += 1
            mapped_ids = sorted(term_to_aspects.get(lexical._normalize(gold_label), set()))
            predicted: list[tuple[str, str, float, int]] = []
            for aspect_id in mapped_ids:
                key = (review.review_id, f"vocab::{aspect_id}")
                row = by_review_aspect.get(key)
                if row is not None:
                    predicted.append(
                        (
                            "vocab",
                            aspect_id,
                            float(row.final_rating),
                            int(row.n_evidence_kept),
                        )
                    )
            if not mapped_ids:
                for row in by_review.get(review.review_id, []):
                    if str(row.aspect_source) != "discovery":
                        continue
                    gold_matches = json.loads(str(row.gold_matches_json) or "{}")
                    if gold_label in gold_matches:
                        predicted.append(
                            (
                                "discovery",
                                str(row.aspect_key),
                                float(row.final_rating),
                                int(row.n_evidence_kept),
                            )
                        )
            if not predicted:
                continue
            evaluable_gold_pairs += 1
            predicted_rating = float(np.mean([item[2] for item in predicted]))
            error = abs(predicted_rating - float(gold_rating))
            errors.append(error)
            round_errors.append(abs(round(predicted_rating) - float(gold_rating)))
            if any(item[0] == "discovery" for item in predicted):
                discovery_errors.append(error)
                aspect_source = "discovery"
            else:
                vocab_errors.append(error)
                aspect_source = "vocab"
            hard_case_rows.append(
                {
                    "review_id": review.review_id,
                    "nm_id": int(review.nm_id),
                    "category_id": str(review.category_id),
                    "aspect_name": str(gold_label),
                    "aspect_source": aspect_source,
                    "gold_rating": float(gold_rating),
                    "predicted_rating": round(predicted_rating, 4),
                    "abs_error": round(error, 4),
                    "review_rating": float(review.rating),
                    "review_text": str(review.text),
                }
            )
        if errors:
            review_maes.append(float(np.mean(errors)))
            review_round_maes.append(float(np.mean(round_errors)))

    hard_cases = pd.DataFrame(hard_case_rows)
    if not hard_cases.empty:
        hard_cases = hard_cases.sort_values(["abs_error", "review_id"], ascending=[False, True]).reset_index(drop=True)

    metrics = {
        "sentiment_mae_review": float(np.mean(review_maes)) if review_maes else np.nan,
        "sentiment_mae_review_round": float(np.mean(review_round_maes)) if review_round_maes else np.nan,
        "sentiment_mae_vocab_pairs": float(np.mean(vocab_errors)) if vocab_errors else np.nan,
        "sentiment_mae_discovery_pairs": float(np.mean(discovery_errors)) if discovery_errors else np.nan,
        "n_sentiment_review_matches": int(len(review_maes)),
        "n_evaluable_gold_pairs": int(evaluable_gold_pairs),
        "n_total_gold_pairs": int(total_gold_pairs),
        "evaluable_pair_coverage": float(evaluable_gold_pairs / total_gold_pairs) if total_gold_pairs else 0.0,
    }
    return metrics, hard_cases


def _summary_markdown(
    *,
    mode_id: str,
    runtime_sec: float,
    metrics: dict[str, Any],
    counts: dict[str, Any],
    config: dict[str, Any],
    cache_stats: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"# {mode_id}",
        "",
        "## Metrics",
        f"- review_mae: {metrics['sentiment_mae_review']:.4f}" if not np.isnan(metrics["sentiment_mae_review"]) else "- review_mae: nan",
        f"- review_mae_round: {metrics['sentiment_mae_review_round']:.4f}" if not np.isnan(metrics["sentiment_mae_review_round"]) else "- review_mae_round: nan",
        f"- vocab_pair_mae: {metrics['sentiment_mae_vocab_pairs']:.4f}" if not np.isnan(metrics["sentiment_mae_vocab_pairs"]) else "- vocab_pair_mae: nan",
        f"- discovery_pair_mae: {metrics['sentiment_mae_discovery_pairs']:.4f}" if not np.isnan(metrics["sentiment_mae_discovery_pairs"]) else "- discovery_pair_mae: nan",
        f"- evaluable_pair_coverage: {metrics['evaluable_pair_coverage']:.4f}",
        "",
        "## Counts",
        f"- input_rows: {counts['input_rows']}",
        f"- predictions: {counts['predictions']}",
        f"- kept_after_threshold: {counts['kept_after_threshold']}",
        f"- review_aspect_scores: {counts['review_aspect_scores']}",
        f"- discovery_assignments_without_evidence: {counts['discovery_assignments_without_evidence']}",
        "",
        "## Config",
        f"- relevance_threshold: {config['relevance_threshold']}",
        f"- temperature: {config['temperature']}",
        f"- aggregation: {config['aggregation']}",
        f"- window_tokens: {config['window_tokens']}",
        "",
        f"- runtime_sec: {runtime_sec:.2f}",
    ]
    if cache_stats:
        lines.extend(
            [
                "",
                "## NLI cache",
                f"- memory_hits: {int(cache_stats.get('memory_hits', 0))}",
                f"- persistent_hits: {int(cache_stats.get('persistent_hits', 0))}",
                f"- misses: {int(cache_stats.get('misses', 0))}",
                f"- writes: {int(cache_stats.get('writes', 0))}",
            ]
        )
    return "\n".join(lines) + "\n"


def run_mode(
    mode_id: str,
    *,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    dataset_path: str | Path | None = None,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
    aggregation: str = "max_relevance",
) -> Path:
    started = time.perf_counter()
    context = load_benchmark_context(
        run_dir=run_dir,
        dataset_path=dataset_path,
        window_tokens=window_tokens,
    )
    inputs = build_mode_input_frame(context, mode_id)
    predictions = run_inference(
        inputs,
        context,
        mode_id,
        temperature=temperature,
        relevance_threshold=relevance_threshold,
    )
    cache_stats = dict(predictions.attrs.get("cache_stats", {}))
    review_aspect_scores = aggregate_review_aspect_scores(
        predictions,
        aggregation=aggregation,
    )
    metrics, hard_cases = evaluate_review_level(context, review_aspect_scores)

    mode_root = ROOT / "benchmark" / "sentiment" / mode_id
    out_dir = mode_root / "results" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs.to_parquet(out_dir / "inputs.parquet", index=False)
    predictions.to_parquet(out_dir / "nli_predictions.parquet", index=False)
    review_aspect_scores.to_csv(out_dir / "review_aspect_scores.csv", index=False, encoding="utf-8")
    hard_cases.to_csv(out_dir / "hard_cases.csv", index=False, encoding="utf-8")

    counts = {
        "input_rows": int(len(inputs)),
        "predictions": int(len(predictions)),
        "kept_after_threshold": int(predictions["passed_relevance_filter"].sum()) if not predictions.empty else 0,
        "review_aspect_scores": int(len(review_aspect_scores)),
        "discovery_assignments_without_evidence": int(context.discovery_assignments_without_evidence),
    }
    payload = {
        "mode_id": mode_id,
        "source_run_dir": str(context.run_dir),
        "dataset_path": str(context.dataset_path),
        "counts": counts,
        "metrics": metrics,
        "config": {
            "window_tokens": int(window_tokens),
            "relevance_threshold": float(relevance_threshold),
            "temperature": float(temperature),
            "aggregation": str(aggregation),
        },
        "nli_cache": cache_stats,
        "runtime_sec": round(time.perf_counter() - started, 4),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        _summary_markdown(
            mode_id=mode_id,
            runtime_sec=payload["runtime_sec"],
            metrics=metrics,
            counts=counts,
            config=payload["config"],
            cache_stats=cache_stats,
        ),
        encoding="utf-8",
    )
    return out_dir
