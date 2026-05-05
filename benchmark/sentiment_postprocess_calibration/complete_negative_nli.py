from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import config

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = THIS_DIR / "results"
EPS = 1e-12


def _latest_results_dir(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and (path / "calibration_dataset.csv").exists())
    if not candidates:
        raise FileNotFoundError(f"no calibration dataset dirs under {root}")
    return candidates[-1]


def _label_indices(id2label: dict, num_labels: int) -> tuple[int, int, int]:
    ent_idx = 0
    neu_idx = 1 if num_labels > 1 else 0
    contra_idx = 0
    for idx, label in id2label.items():
        lab = str(label).lower()
        if lab == "entailment":
            ent_idx = int(idx)
        if lab == "neutral":
            neu_idx = int(idx)
        if lab == "contradiction":
            contra_idx = int(idx)
    if num_labels >= 3 and len({ent_idx, neu_idx, contra_idx}) == 3:
        return ent_idx, neu_idx, contra_idx
    if num_labels >= 3:
        return 2, 1, 0
    return ent_idx, neu_idx, contra_idx


def _format_negative_hypothesis(aspect_name: str, template: str) -> str:
    return template.format(aspect=str(aspect_name))


def complete_negative_nli(
    *,
    results_dir: Path,
    input_filename: str,
    output_filename: str,
    batch_size: int,
) -> Path:
    input_path = results_dir / input_filename
    if not input_path.exists():
        raise FileNotFoundError(f"input dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = {
        "prediction_id",
        "review_id",
        "premise_text",
        "system_aspect",
        "pos_entailment",
        "pos_neutral",
        "pos_contradiction",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"input dataset is missing required columns: {', '.join(missing_columns)}")
    if df["prediction_id"].astype(str).duplicated().any():
        raise ValueError("input has duplicate prediction_id")

    neg_template = str(getattr(config.sentiment, "hypothesis_template_neg", "{aspect} — это плохо"))
    if "{aspect}" not in neg_template:
        neg_template = "{aspect} — это плохо"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.models.nli_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.models.nli_path, local_files_only=True).to(device)
    model.eval()

    hf_cfg = AutoConfig.from_pretrained(config.models.nli_path, local_files_only=True)
    ent_idx, neu_idx, contra_idx = _label_indices(hf_cfg.id2label, int(hf_cfg.num_labels))
    temperature = float(getattr(config.sentiment, "temperature", 1.0))

    premises = df["premise_text"].astype(str).tolist()
    aspects = df["system_aspect"].astype(str).tolist()
    neg_hypotheses = [_format_negative_hypothesis(aspect, neg_template) for aspect in aspects]

    neg_entailment = np.zeros(len(df), dtype=np.float64)
    neg_neutral = np.zeros(len(df), dtype=np.float64)
    neg_contradiction = np.zeros(len(df), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            enc = tokenizer(
                premises[start:end],
                neg_hypotheses[start:end],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()
            neg_entailment[start:end] = probs[:, ent_idx]
            neg_neutral[start:end] = probs[:, neu_idx]
            neg_contradiction[start:end] = probs[:, contra_idx]

    out = df.copy()
    out["neg_hypothesis_text"] = neg_hypotheses
    out["neg_entailment"] = neg_entailment
    out["neg_neutral"] = neg_neutral
    out["neg_contradiction"] = neg_contradiction

    # Invariants required by experiment.
    if len(out) != len(df):
        raise AssertionError("row count changed")
    if out["prediction_id"].astype(str).duplicated().any():
        raise AssertionError("duplicate prediction_id after completion")
    for col in ("neg_entailment", "neg_neutral", "neg_contradiction"):
        if out[col].isna().any():
            raise AssertionError(f"null values found in {col}")
    for col in ("pos_entailment", "pos_neutral", "pos_contradiction"):
        if not np.allclose(out[col].astype(float).values, df[col].astype(float).values, atol=0.0, rtol=0.0):
            raise AssertionError(f"positive probabilities changed in column {col}")
    if not out["prediction_id"].astype(str).equals(df["prediction_id"].astype(str)):
        raise AssertionError("prediction_id list changed")
    if not out["review_id"].astype(str).equals(df["review_id"].astype(str)):
        raise AssertionError("review_id list changed")
    if not out["premise_text"].astype(str).equals(df["premise_text"].astype(str)):
        raise AssertionError("premise_text changed")

    sum_probs = out["neg_entailment"] + out["neg_neutral"] + out["neg_contradiction"]
    if not np.all(np.abs(sum_probs - 1.0) < 1e-4 + EPS):
        raise AssertionError("negative probabilities do not sum to ~1")

    output_path = results_dir / output_filename
    out.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete missing negative NLI probabilities for calibration dataset.")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--input-filename", default="calibration_dataset.csv")
    parser.add_argument("--output-filename", default="calibration_dataset_with_dual_nli.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _latest_results_dir(Path(args.output_root))
    out = complete_negative_nli(
        results_dir=results_dir.resolve(),
        input_filename=args.input_filename,
        output_filename=args.output_filename,
        batch_size=int(args.batch_size),
    )
    print(out)


if __name__ == "__main__":
    main()
