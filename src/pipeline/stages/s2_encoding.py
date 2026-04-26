from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scripts import run_phase2_baseline_matching as lexical
from src.discovery.encoder import DiscoveryEncoder


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (matrix / norms).astype(np.float32, copy=False)


def _encode_cached(
    encoder: DiscoveryEncoder,
    texts: list[str],
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    unique = sorted({str(text).strip() for text in texts if str(text).strip() and str(text).strip() not in cache})
    if unique:
        vectors = _l2_normalize(encoder.encode(unique))
        for text, vector in zip(unique, vectors, strict=True):
            cache[text] = vector.astype(np.float32, copy=False)
    return {text: cache[text] for text in texts if text in cache}


def run_stage(
    candidates: pd.DataFrame,
    aspects_by_category: dict[str, list[Any]],
    config: dict[str, Any],
    *,
    encoder: DiscoveryEncoder | None = None,
    cache: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    model_cfg = config.get("models", {})
    disc_cfg = config.get("discovery", {})
    encoder = encoder or DiscoveryEncoder(
        model_name_or_path=str(model_cfg.get("encoder", "ai-forever/sbert_large_nlu_ru")),
        batch_size=int(disc_cfg.get("encoder_batch_size", 8)),
    )
    cache = cache if cache is not None else {}

    cand_texts = candidates["text_lemmatized"].fillna("").astype(str).tolist() if not candidates.empty else []
    cand_vectors_by_text = _encode_cached(encoder, cand_texts, cache)
    cand_rows: list[np.ndarray] = []
    cand_index_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(candidates.itertuples(index=False)):
        text = str(row.text_lemmatized)
        vector = cand_vectors_by_text.get(text)
        if vector is None:
            vector = np.zeros(encoder.embedding_dim, dtype=np.float32)
        cand_rows.append(vector)
        cand_index_rows.append({"candidate_id": row.candidate_id, "row_index": row_index})
    candidate_embeddings = (
        np.vstack(cand_rows).astype(np.float32)
        if cand_rows
        else np.empty((0, encoder.embedding_dim), dtype=np.float32)
    )

    vocab_rows: list[np.ndarray] = []
    vocab_index_rows: list[dict[str, Any]] = []
    aspect_vectors_by_category: dict[str, dict[str, np.ndarray]] = {}
    for category, aspects in sorted(aspects_by_category.items()):
        aspect_vectors_by_category[category] = {}
        for aspect in aspects:
            raw_terms = [aspect.canonical_name] + list(aspect.synonyms)
            terms = [lexical._normalize(term) for term in raw_terms]
            terms = [term for term in terms if term]
            vectors = _encode_cached(encoder, terms, cache)
            if vectors:
                matrix = np.vstack([vectors[term] for term in terms])
                centroid = _l2_normalize(matrix.mean(axis=0, keepdims=True))[0]
            else:
                centroid = np.zeros(encoder.embedding_dim, dtype=np.float32)
            aspect_vectors_by_category[category][aspect.id] = centroid
            vocab_index_rows.append(
                {
                    "aspect_id": aspect.id,
                    "category_id": category,
                    "row_index": len(vocab_rows),
                }
            )
            vocab_rows.append(centroid)
    vocab_embeddings = (
        np.vstack(vocab_rows).astype(np.float32)
        if vocab_rows
        else np.empty((0, encoder.embedding_dim), dtype=np.float32)
    )

    return {
        "encoder": encoder,
        "cache": cache,
        "candidate_embeddings": candidate_embeddings,
        "candidate_index": pd.DataFrame(cand_index_rows, columns=["candidate_id", "row_index"]),
        "vocab_embeddings": vocab_embeddings,
        "vocab_index": pd.DataFrame(vocab_index_rows, columns=["aspect_id", "category_id", "row_index"]),
        "candidate_vectors_by_id": {
            row["candidate_id"]: candidate_embeddings[int(row["row_index"])]
            for row in cand_index_rows
        },
        "aspect_vectors_by_category": aspect_vectors_by_category,
    }
