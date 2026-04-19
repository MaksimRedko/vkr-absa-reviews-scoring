from __future__ import annotations

import ast
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import pymorphy3


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "benchmark" / "implicit_pilot" / "results"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

_MORPH = pymorphy3.MorphAnalyzer()


def resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_json(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {resolved}")
    return payload


def dump_json(path: str | Path, payload: Any) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def make_run_dir(prefix: str = "implicit_nli") -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / f"{run_id}_{prefix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_true_labels(raw: Any) -> Dict[str, float]:
    if pd.isna(raw):
        return {}
    text = str(raw).strip()
    if not text or text in {"nan", "{}"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): float(value) for key, value in parsed.items()}


def review_text_from_row(row: pd.Series) -> str:
    full_text = str(row.get("full_text", "") or "").strip()
    if full_text and full_text.lower() != "nan":
        return full_text

    parts: list[str] = []
    for key in ("pros", "cons"):
        value = str(row.get(key, "") or "").strip()
        if value and value.lower() != "nan":
            parts.append(value)
    return " ".join(parts).strip()


def load_markup_frame(csv_path: str | Path) -> pd.DataFrame:
    resolved = resolve_repo_path(csv_path)
    df = pd.read_csv(resolved, dtype={"id": str})
    df["true_labels_parsed"] = df["true_labels"].apply(parse_true_labels)
    df["review_text"] = df.apply(review_text_from_row, axis=1)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df = df[df["review_text"].str.strip() != ""].copy()
    return df


def clean_text(text: str) -> str:
    normalized = str(text or "").replace("\\n", " ").replace("\n", " ")
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()
    return normalized


def split_sentences(text: str) -> List[str]:
    normalized = clean_text(text)
    if not normalized:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    return parts if parts else [normalized]


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(clean_text(text))]


def lemmatize_token(token: str) -> str:
    parses = _MORPH.parse(token)
    if not parses:
        return token.lower()
    return str(parses[0].normal_form or token).lower()


def lemmatize_text(text: str) -> List[str]:
    return [lemmatize_token(token) for token in tokenize(text)]


def term_lemma_sequences(terms: Sequence[str]) -> List[tuple[str, ...]]:
    sequences: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for term in terms:
        seq = tuple(lemmatize_text(term))
        if not seq or seq in seen:
            continue
        seen.add(seq)
        sequences.append(seq)
    return sequences


def _contains_subsequence(tokens: Sequence[str], sequence: Sequence[str]) -> bool:
    if not tokens or not sequence or len(sequence) > len(tokens):
        return False
    end = len(tokens) - len(sequence) + 1
    for idx in range(end):
        if tuple(tokens[idx : idx + len(sequence)]) == tuple(sequence):
            return True
    return False


def matched_terms(text: str, terms: Sequence[str]) -> List[str]:
    tokens = lemmatize_text(text)
    matches: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for term in terms:
        sequence = tuple(lemmatize_text(term))
        if not sequence or sequence in seen:
            continue
        seen.add(sequence)
        if _contains_subsequence(tokens, sequence):
            matches.append(term)
    return matches


def has_lexical_overlap(text: str, terms: Sequence[str]) -> bool:
    return bool(matched_terms(text, terms))


def score_sentence_for_lexicon(sentence: str, terms: Sequence[str]) -> tuple[int, int, int]:
    tokens = lemmatize_text(sentence)
    token_set = set(tokens)
    sequences = term_lemma_sequences(terms)
    phrase_hits = 0
    token_hits = 0
    for sequence in sequences:
        if _contains_subsequence(tokens, sequence):
            phrase_hits += 1
            token_hits += len(set(sequence))
        else:
            token_hits += sum(1 for token in set(sequence) if token in token_set)
    return phrase_hits, token_hits, len(tokens)


def select_sentence_candidate(review_text: str, terms: Sequence[str]) -> str:
    sentences = split_sentences(review_text)
    if not sentences:
        return clean_text(review_text)
    return max(sentences, key=lambda sentence: score_sentence_for_lexicon(sentence, terms))


def aspect_terms(aspect_cfg: Dict[str, Any]) -> List[str]:
    terms = [str(aspect_cfg["name"])]
    for term in aspect_cfg.get("synonyms", []):
        text = str(term).strip()
        if text:
            terms.append(text)
    return terms


def seed_random(seed: int) -> random.Random:
    return random.Random(int(seed))


def sample_records(records: Sequence[Dict[str, Any]], size: int, rng: random.Random) -> List[Dict[str, Any]]:
    if size >= len(records):
        return list(records)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    picked = sorted(indices[:size])
    return [records[idx] for idx in picked]


def profile_selected_aspects(df: pd.DataFrame, aspects: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: list[Dict[str, Any]] = []
    for aspect_cfg in aspects:
        aspect_name = str(aspect_cfg["name"])
        terms = aspect_terms(aspect_cfg)
        positive = df["true_labels_parsed"].apply(
            lambda labels, aspect_name=aspect_name: aspect_name in labels
        )
        positive_df = df[positive]
        explicit = positive_df["review_text"].apply(
            lambda text, terms=terms: has_lexical_overlap(text, terms)
        )
        summary.append(
            {
                "aspect": aspect_name,
                "aspect_type": str(aspect_cfg.get("type", "unknown")),
                "total_positive": int(len(positive_df)),
                "explicit_positive": int(explicit.sum()),
                "implicit_positive": int((~explicit).sum()),
                "unrelated_pool": int(
                    (
                        (~positive)
                        & (
                            ~df["review_text"].apply(
                                lambda text, terms=terms: has_lexical_overlap(text, terms)
                            )
                        )
                    ).sum()
                ),
                "lexicon_terms": terms,
            }
        )
    return summary


def profile_all_aspects(df: pd.DataFrame) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for labels in df["true_labels_parsed"]:
        for aspect in labels:
            counts[aspect] = counts.get(aspect, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"aspect": aspect, "count": count} for aspect, count in ranked]


def encode_json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False)
