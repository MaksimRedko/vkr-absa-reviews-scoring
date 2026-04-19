from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.stages.sentiment import SentimentEngine  # noqa: E402

from common import clean_text, dump_json, load_json, resolve_repo_path  # noqa: E402


def _score_column_name(lambda_value: float) -> str:
    formatted = f"{lambda_value:.2f}".rstrip("0").rstrip(".")
    return f"score_ent_minus_neu_lambda_{formatted.replace('.', '_')}"


def _load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    resolved = resolve_repo_path(dataset_path)
    df = pd.read_csv(resolved, dtype={"review_id": str})
    for column in ("sentence_candidates_json", "lexicon_terms_json", "matched_terms_json"):
        if column in df.columns:
            df[column] = df[column].fillna("[]").apply(json.loads)
    return df


def _collect_probabilities(
    engine: SentimentEngine,
    premises: List[str],
    hypotheses: List[str],
) -> torch.Tensor:
    logits = engine._forward_logits_tensor(premises, hypotheses)  # pylint: disable=protected-access
    return torch.softmax(logits / engine.temperature, dim=1).cpu()


def _review_level_results(
    engine: SentimentEngine,
    dataset_df: pd.DataFrame,
    template_id: str,
    template_text: str,
    lambda_values: List[float],
) -> List[Dict[str, Any]]:
    premises = [clean_text(text) for text in dataset_df["review_text"].tolist()]
    hypotheses = [template_text.format(aspect=aspect) for aspect in dataset_df["aspect_id"].tolist()]
    probs = _collect_probabilities(engine, premises, hypotheses).numpy()

    results: list[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(dataset_df.iterrows()):
        p_ent = float(probs[idx, engine.ent_idx])
        p_neu = float(probs[idx, engine.neu_idx])
        p_contr = float(probs[idx, engine.contra_idx])
        payload = {
            "review_id": str(row["review_id"]),
            "product_id": int(row["product_id"]),
            "rating": int(row["rating"]),
            "aspect_id": str(row["aspect_id"]),
            "aspect_type": str(row["aspect_type"]),
            "class": str(row["class"]),
            "expected_p_ent": str(row["expected_p_ent"]),
            "granularity": "review",
            "hypothesis_id": template_id,
            "hypothesis_text": template_text.format(aspect=row["aspect_id"]),
            "premise_text": premises[idx],
            "selected_sentence_index": -1,
            "candidate_sentence_count": int(len(row.get("sentence_candidates_json", []))),
            "lexical_overlap": bool(row.get("lexical_overlap", False)),
            "needs_manual_review": bool(row.get("needs_manual_review", False)),
            "p_ent": p_ent,
            "p_neu": p_neu,
            "p_contr": p_contr,
            "score_p_ent": p_ent,
            "score_ent_minus_contr": p_ent - p_contr,
        }
        for lambda_value in lambda_values:
            payload[_score_column_name(lambda_value)] = p_ent - (lambda_value * p_neu)
        results.append(payload)
    return results


def _sentence_level_results(
    engine: SentimentEngine,
    dataset_df: pd.DataFrame,
    template_id: str,
    template_text: str,
    lambda_values: List[float],
) -> List[Dict[str, Any]]:
    premises: list[str] = []
    hypotheses: list[str] = []
    row_refs: list[tuple[int, int]] = []

    for row_idx, (_, row) in enumerate(dataset_df.iterrows()):
        sentence_candidates = list(row.get("sentence_candidates_json", []))
        cleaned = [clean_text(sentence) for sentence in sentence_candidates if clean_text(sentence)]
        if not cleaned:
            cleaned = [clean_text(str(row["sentence_text"] or row["review_text"]))]
        hypothesis = template_text.format(aspect=row["aspect_id"])
        for sent_idx, sentence in enumerate(cleaned):
            premises.append(sentence)
            hypotheses.append(hypothesis)
            row_refs.append((row_idx, sent_idx))

    probs = _collect_probabilities(engine, premises, hypotheses).numpy()

    grouped: dict[int, list[Dict[str, Any]]] = {}
    for expanded_idx, (row_idx, sentence_idx) in enumerate(row_refs):
        grouped.setdefault(row_idx, []).append(
            {
                "sentence_idx": sentence_idx,
                "premise_text": premises[expanded_idx],
                "p_ent": float(probs[expanded_idx, engine.ent_idx]),
                "p_neu": float(probs[expanded_idx, engine.neu_idx]),
                "p_contr": float(probs[expanded_idx, engine.contra_idx]),
            }
        )

    results: list[Dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(dataset_df.iterrows()):
        candidates = grouped[row_idx]
        best = max(
            candidates,
            key=lambda item: (
                item["p_ent"],
                item["p_ent"] - item["p_neu"],
                item["p_ent"] - item["p_contr"],
                -item["sentence_idx"],
            ),
        )
        payload = {
            "review_id": str(row["review_id"]),
            "product_id": int(row["product_id"]),
            "rating": int(row["rating"]),
            "aspect_id": str(row["aspect_id"]),
            "aspect_type": str(row["aspect_type"]),
            "class": str(row["class"]),
            "expected_p_ent": str(row["expected_p_ent"]),
            "granularity": "sentence",
            "hypothesis_id": template_id,
            "hypothesis_text": template_text.format(aspect=row["aspect_id"]),
            "premise_text": best["premise_text"],
            "selected_sentence_index": int(best["sentence_idx"]),
            "candidate_sentence_count": int(len(candidates)),
            "lexical_overlap": bool(row.get("lexical_overlap", False)),
            "needs_manual_review": bool(row.get("needs_manual_review", False)),
            "p_ent": float(best["p_ent"]),
            "p_neu": float(best["p_neu"]),
            "p_contr": float(best["p_contr"]),
            "score_p_ent": float(best["p_ent"]),
            "score_ent_minus_contr": float(best["p_ent"] - best["p_contr"]),
        }
        for lambda_value in lambda_values:
            payload[_score_column_name(lambda_value)] = float(
                best["p_ent"] - (lambda_value * best["p_neu"])
            )
        results.append(payload)
    return results


def run_inference(config: Dict[str, Any], run_dir: Path) -> Path:
    dataset_path = run_dir / "pilot_dataset.csv"
    dataset_df = _load_dataset(dataset_path)
    inference_cfg = dict(config.get("inference", {}))
    templates = list(inference_cfg["hypothesis_templates"])
    granularities = list(inference_cfg.get("granularities", ["review", "sentence"]))
    lambda_values = [float(value) for value in inference_cfg.get("lambda_values", [0.0, 0.5, 1.0])]

    engine = SentimentEngine()

    results: list[Dict[str, Any]] = []
    for template in templates:
        template_id = str(template["id"])
        template_text = str(template["text"])
        if "review" in granularities:
            results.extend(
                _review_level_results(
                    engine=engine,
                    dataset_df=dataset_df,
                    template_id=template_id,
                    template_text=template_text,
                    lambda_values=lambda_values,
                )
            )
        if "sentence" in granularities:
            results.extend(
                _sentence_level_results(
                    engine=engine,
                    dataset_df=dataset_df,
                    template_id=template_id,
                    template_text=template_text,
                    lambda_values=lambda_values,
                )
            )

    results_df = pd.DataFrame(results)
    output_path = run_dir / "inference_results.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8")

    metadata = {
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "templates": templates,
        "granularities": granularities,
        "lambda_values": lambda_values,
        "model_path": str(engine.tokenizer.name_or_path),
        "temperature": float(engine.temperature),
        "ent_idx": int(engine.ent_idx),
        "neu_idx": int(engine.neu_idx),
        "contra_idx": int(engine.contra_idx),
        "num_rows": int(len(results_df)),
    }
    dump_json(run_dir / "inference_manifest.json", metadata)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLI inference for implicit pilot.")
    parser.add_argument(
        "--config",
        default="experiments/implicit_nli_pilot.json",
        help="Path to pilot config JSON.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Results directory created by build_dataset.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    run_dir = resolve_repo_path(args.run_dir)
    output_path = run_inference(config, run_dir)
    print(f"[implicit_pilot] inference={output_path}")


if __name__ == "__main__":
    main()
