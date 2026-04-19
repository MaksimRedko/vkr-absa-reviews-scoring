from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from configs.configs import config  # noqa: E402
from common import clean_text, dump_json, make_run_dir, resolve_repo_path  # noqa: E402


SOFTMAX_TAU = 0.1

INLINE_VOCAB: Dict[str, Dict[str, Any]] = {
    "Чистота": {
        "dataset_aspect": "Чистота",
        "canonical": "Чистота",
        "aspect_type": "sensory",
        "synonyms": [
            "чистота",
            "чисто",
            "грязно",
            "грязь",
            "уборка",
            "пыль",
            "грязный",
            "чистый",
            "убрано",
            "не убрано",
            "мусор",
            "пятно",
            "пятна",
            "аккуратный",
            "опрятный",
        ],
    },
    "Размер": {
        "dataset_aspect": "Размер",
        "canonical": "Размер",
        "aspect_type": "functional",
        "synonyms": [
            "размер",
            "размеры",
            "большой",
            "маленький",
            "мелкий",
            "крупный",
            "широкий",
            "узкий",
            "длинный",
            "короткий",
            "высота",
            "ширина",
            "длина",
            "подошёл",
            "не подошёл",
            "габариты",
        ],
    },
    "Впечатление": {
        "dataset_aspect": "Впечатление",
        "canonical": "Общее впечатление",
        "aspect_type": "abstract",
        "synonyms": [
            "впечатление",
            "впечатления",
            "в целом",
            "общее",
            "понравилось",
            "не понравилось",
            "рекомендую",
            "не рекомендую",
            "советую",
            "не советую",
            "доволен",
            "недоволен",
            "хорошо",
            "плохо",
            "отлично",
            "ужасно",
        ],
    },
}


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, dtype={"review_id": str})
    return df.fillna("")


def _load_vocabulary(vocab_path: Path | None) -> Dict[str, Dict[str, Any]]:
    if vocab_path is None or not vocab_path.is_file():
        return INLINE_VOCAB

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read external vocabulary yaml") from exc

    with vocab_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    entries: Iterable[Any]
    if isinstance(payload, dict) and isinstance(payload.get("aspects"), list):
        entries = payload["aspects"]
    elif isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.values()
    else:
        raise ValueError(f"Unsupported vocabulary format in {vocab_path}")

    vocab: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        dataset_aspect = str(
            entry.get("dataset_aspect")
            or entry.get("name")
            or entry.get("canonical")
            or ""
        ).strip()
        if not dataset_aspect:
            continue
        vocab[dataset_aspect] = {
            "dataset_aspect": dataset_aspect,
            "canonical": str(entry.get("canonical") or dataset_aspect).strip(),
            "aspect_type": str(entry.get("aspect_type") or entry.get("type") or ""),
            "synonyms": [str(item).strip() for item in entry.get("synonyms", []) if str(item).strip()],
        }
    if not vocab:
        raise ValueError(f"No usable aspect entries found in {vocab_path}")
    return vocab


def _aspect_terms(aspect_meta: Dict[str, Any]) -> List[str]:
    terms = [str(aspect_meta["canonical"]).strip()]
    dataset_aspect = str(aspect_meta["dataset_aspect"]).strip()
    if dataset_aspect and dataset_aspect not in terms:
        terms.append(dataset_aspect)
    for synonym in aspect_meta.get("synonyms", []):
        text = str(synonym).strip()
        if text and text not in terms:
            terms.append(text)
    return terms


def _encode_unique_texts(model: SentenceTransformer, texts: Iterable[str]) -> Dict[str, np.ndarray]:
    unique_texts = sorted({clean_text(text) for text in texts if clean_text(text)})
    if not unique_texts:
        return {}
    embeddings = model.encode(
        unique_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    normalized = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    return {text: normalized[idx] for idx, text in enumerate(unique_texts)}


def _build_centroids(
    vocab: Dict[str, Dict[str, Any]],
    embedding_cache: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    centroids: Dict[str, np.ndarray] = {}
    for aspect_id, aspect_meta in vocab.items():
        term_vectors = np.stack([embedding_cache[text] for text in _aspect_terms(aspect_meta)])
        centroid = term_vectors.mean(axis=0, keepdims=True).astype(np.float32)
        centroids[aspect_id] = _l2_normalize(centroid)[0]
    return centroids


def _softmax_rank(scores: np.ndarray, target_index: int, tau: float) -> float:
    scaled = scores / tau
    shifted = scaled - np.max(scaled)
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    return float(probs[target_index])


def run_cosine_baseline(
    dataset_path: Path,
    run_dir: Path,
    vocab_path: Path | None = None,
) -> Path:
    dataset_df = _load_dataset(dataset_path)
    vocab = _load_vocabulary(vocab_path)

    missing_aspects = sorted(set(dataset_df["aspect_id"]) - set(vocab))
    if missing_aspects:
        raise ValueError(f"Missing cosine vocabulary for aspects: {missing_aspects}")

    model = SentenceTransformer(config.models.encoder_path)

    premise_texts: List[str] = []
    for column in ("review_text", "sentence_text"):
        premise_texts.extend(dataset_df[column].astype(str).tolist())

    synonym_texts: List[str] = []
    for aspect_meta in vocab.values():
        synonym_texts.extend(_aspect_terms(aspect_meta))

    embedding_cache = _encode_unique_texts(model, [*premise_texts, *synonym_texts])
    centroids = _build_centroids(vocab, embedding_cache)
    aspect_order = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[aspect_id] for aspect_id in aspect_order])

    results: list[Dict[str, Any]] = []
    for granularity, premise_column in (("review", "review_text"), ("sentence", "sentence_text")):
        for _, row in dataset_df.iterrows():
            premise_text = clean_text(str(row[premise_column]))
            premise_embedding = embedding_cache[premise_text]
            cosine_scores = centroid_matrix @ premise_embedding
            target_aspect = str(row["aspect_id"])
            target_index = aspect_order.index(target_aspect)
            target_score = float(cosine_scores[target_index])
            competitor_scores = np.delete(cosine_scores, target_index)
            contrastive_score = float(target_score - np.max(competitor_scores))
            rank_score = _softmax_rank(cosine_scores, target_index, SOFTMAX_TAU)

            payload = {column: row[column] for column in dataset_df.columns}
            payload.update(
                {
                    "granularity": granularity,
                    "premise_text": premise_text,
                    "aspect_canonical": str(vocab[target_aspect]["canonical"]),
                    "score_cos": target_score,
                    "score_cos_contrastive": contrastive_score,
                    "score_cos_rank": rank_score,
                }
            )
            results.append(payload)

    results_df = pd.DataFrame(results)
    output_path = run_dir / "cosine_inference_results.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8")

    manifest = {
        "dataset_path": str(dataset_path),
        "source_run_dir": str(dataset_path.parent),
        "output_path": str(output_path),
        "vocabulary_path": str(vocab_path) if vocab_path and vocab_path.is_file() else None,
        "encoder_path": str(config.models.encoder_path),
        "softmax_tau": SOFTMAX_TAU,
        "aspect_order": aspect_order,
        "num_rows": int(len(results_df)),
    }
    dump_json(run_dir / "cosine_manifest.json", manifest)

    copied_dataset_path = run_dir / "pilot_dataset.csv"
    if copied_dataset_path.resolve() != dataset_path.resolve():
        shutil.copyfile(dataset_path, copied_dataset_path)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cosine baseline on implicit pilot dataset.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to existing pilot_dataset.csv.",
    )
    parser.add_argument(
        "--vocabulary-path",
        default="src/vocabulary/universal_aspects_v1.yaml",
        help="Optional vocabulary yaml. Falls back to inline pilot vocabulary when absent.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Existing results directory. If omitted, a new cosine_baseline run dir is created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset_path)
    vocab_path = resolve_repo_path(args.vocabulary_path) if args.vocabulary_path else None
    run_dir = resolve_repo_path(args.run_dir) if args.run_dir else make_run_dir(prefix="cosine_baseline")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_cosine_baseline(dataset_path=dataset_path, run_dir=run_dir, vocab_path=vocab_path)
    print(f"[implicit_pilot] cosine={output_path}")


if __name__ == "__main__":
    main()
