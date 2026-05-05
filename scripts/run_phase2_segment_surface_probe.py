from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pymorphy3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import configs as _cfg_module
from src.stages.extraction import CandidateExtractor
from src.stages.segmentation import RuleBasedClauseSegmenter
from src.vocabulary.loader import AspectDefinition, Vocabulary

_cfg_module.config.discovery.dependency_filter_enabled = False  # type: ignore[attr-defined]
_MORPH = pymorphy3.MorphAnalyzer()

MODES: tuple[str, ...] = ("A", "B1", "B2")
PAIRWISE_MODE_PAIRS: tuple[tuple[str, str], ...] = (("A", "B1"), ("A", "B2"), ("B1", "B2"))


@dataclass(frozen=True, slots=True)
class ReviewRow:
    review_id: str
    nm_id: int
    category: str
    text: str
    gold_labels_lemma: frozenset[str]


@dataclass(frozen=True, slots=True)
class ModePrediction:
    terms: set[str]
    aspects: set[str]


def _normalize(text: str) -> str:
    tokens = [t.strip().lower() for t in str(text).replace("-", " ").split() if t.strip()]
    lemmas: list[str] = []
    for token in tokens:
        parses = _MORPH.parse(token)
        lemmas.append(str(parses[0].normal_form if parses else token).lower())
    return " ".join(lemmas)


def _parse_true_labels(raw: Any) -> frozenset[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return frozenset()
    txt = str(raw).strip()
    if not txt or txt.lower() in {"nan", "none", "{}"}:
        return frozenset()
    try:
        parsed = ast.literal_eval(txt)
    except (ValueError, SyntaxError):
        return frozenset()
    if not isinstance(parsed, dict):
        return frozenset()
    return frozenset(_normalize(str(k).strip()) for k in parsed.keys() if str(k).strip())


def _load_rows(dataset_csv: Path, category: str) -> list[ReviewRow]:
    df = pd.read_csv(dataset_csv, dtype={"id": str})
    if "category" not in df.columns:
        raise ValueError("dataset must contain 'category' column")
    if category != "all":
        df = df[df["category"] == category]

    out: list[ReviewRow] = []
    for _, row in df.iterrows():
        text = str(row.get("full_text", "")).strip()
        if not text:
            continue
        out.append(
            ReviewRow(
                review_id=str(row["id"]),
                nm_id=int(row["nm_id"]),
                category=str(row["category"]).strip(),
                text=text,
                gold_labels_lemma=_parse_true_labels(row.get("true_labels")),
            )
        )
    return out


def _build_hybrid_aspects_by_category(core_vocab: Path, domain_by_category: dict[str, Path]) -> dict[str, list[AspectDefinition]]:
    out: dict[str, list[AspectDefinition]] = {}
    for category, domain_path in domain_by_category.items():
        seen: set[str] = set()
        aspects: list[AspectDefinition] = []
        for p in (core_vocab, domain_path):
            vocab = Vocabulary.load_from_yaml(p)
            for a in vocab.aspects:
                if a.id in seen:
                    continue
                seen.add(a.id)
                aspects.append(a)
        out[category] = aspects
    return out


def _term_to_aspects(aspects: list[AspectDefinition]) -> dict[str, set[str]]:
    m: dict[str, set[str]] = defaultdict(set)
    for a in aspects:
        for term in [a.canonical_name] + a.synonyms:
            lemma = _normalize(term)
            if lemma:
                m[lemma].add(a.id)
    return m


def _lexical_map_to_aspects(terms_lemma: set[str], term2asp: dict[str, set[str]]) -> set[str]:
    out: set[str] = set()
    for term in terms_lemma:
        out.update(term2asp.get(term, set()))
    return out


def _prf(pred: set[str], true: set[str]) -> tuple[float, float, float]:
    if not pred and not true:
        return 1.0, 1.0, 1.0
    if not pred or not true:
        return 0.0, 0.0, 0.0
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true) if true else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def _candidate_lemmas_from_text(text: str, extractor: CandidateExtractor) -> set[str]:
    return {_normalize(c.span) for c in extractor.extract(text) if str(c.span).strip()}


def _segment_surface_terms(segment_text: str) -> set[str]:
    lemmas = [x for x in _normalize(segment_text).split() if x]
    out: set[str] = set(lemmas)
    for n in (2, 3):
        if len(lemmas) < n:
            continue
        for i in range(0, len(lemmas) - n + 1):
            out.add(" ".join(lemmas[i : i + n]))
    return out


def _predict_a_review(text: str, extractor: CandidateExtractor, term2asp: dict[str, set[str]]) -> ModePrediction:
    terms = _candidate_lemmas_from_text(text, extractor)
    return ModePrediction(terms=terms, aspects=_lexical_map_to_aspects(terms, term2asp))


def _predict_b1_segments(text: str, extractor: CandidateExtractor, segmenter: RuleBasedClauseSegmenter, term2asp: dict[str, set[str]]) -> tuple[ModePrediction, int]:
    segments = segmenter.split(text)
    all_terms: set[str] = set()
    all_aspects: set[str] = set()
    for seg in segments:
        seg_terms = _candidate_lemmas_from_text(seg.text, extractor)
        all_terms.update(seg_terms)
        all_aspects.update(_lexical_map_to_aspects(seg_terms, term2asp))
    return ModePrediction(terms=all_terms, aspects=all_aspects), len(segments)


def _predict_b2_segments(text: str, segmenter: RuleBasedClauseSegmenter, term2asp: dict[str, set[str]]) -> ModePrediction:
    segments = segmenter.split(text)
    all_terms: set[str] = set()
    all_aspects: set[str] = set()
    for seg in segments:
        seg_terms = _segment_surface_terms(seg.text)
        all_terms.update(seg_terms)
        all_aspects.update(_lexical_map_to_aspects(seg_terms, term2asp))
    return ModePrediction(terms=all_terms, aspects=all_aspects)


def _macro(vals: list[tuple[float, float, float]]) -> dict[str, float]:
    if not vals:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = sum(x[0] for x in vals) / len(vals)
    r = sum(x[1] for x in vals) / len(vals)
    f = sum(x[2] for x in vals) / len(vals)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def run_experiment(rows: list[ReviewRow], hybrid_aspects_by_category: dict[str, list[AspectDefinition]], out_dir: Path) -> None:
    term2asp_by_category = {cat: _term_to_aspects(aspects) for cat, aspects in hybrid_aspects_by_category.items()}

    extractor = CandidateExtractor(ngram_range=(1, 2), min_word_length=3)
    extractor.dependency_filter_enabled = False
    segmenter = RuleBasedClauseSegmenter()

    per_mode_metrics: dict[str, list[tuple[float, float, float]]] = {m: [] for m in MODES}
    per_mode_category_metrics: dict[str, dict[str, list[tuple[float, float, float]]]] = {
        m: defaultdict(list) for m in MODES
    }
    pred_count_by_mode: dict[str, list[int]] = {m: [] for m in MODES}
    fp_by_mode: dict[str, Counter[str]] = {m: Counter() for m in MODES}
    fn_by_mode: dict[str, Counter[str]] = {m: Counter() for m in MODES}
    pairwise_diff_counter: Counter[str] = Counter()

    review_rows: list[dict[str, Any]] = []

    for r in rows:
        term2asp = term2asp_by_category.get(r.category)
        if term2asp is None:
            raise ValueError(f"No vocabulary loaded for category='{r.category}'")

        gold_aspects = _lexical_map_to_aspects(set(r.gold_labels_lemma), term2asp)

        pred_a = _predict_a_review(r.text, extractor, term2asp)
        pred_b1, n_segments = _predict_b1_segments(r.text, extractor, segmenter, term2asp)
        pred_b2 = _predict_b2_segments(r.text, segmenter, term2asp)

        preds = {
            "A": pred_a.aspects,
            "B1": pred_b1.aspects,
            "B2": pred_b2.aspects,
        }

        for mode, pred_aspects in preds.items():
            p, rc, f1 = _prf(pred_aspects, gold_aspects)
            per_mode_metrics[mode].append((p, rc, f1))
            per_mode_category_metrics[mode][r.category].append((p, rc, f1))
            pred_count_by_mode[mode].append(len(pred_aspects))

            for aid in (pred_aspects - gold_aspects):
                fp_by_mode[mode][aid] += 1
            for aid in (gold_aspects - pred_aspects):
                fn_by_mode[mode][aid] += 1

        for left, right in PAIRWISE_MODE_PAIRS:
            if preds[left] != preds[right]:
                pairwise_diff_counter[f"{left}_vs_{right}"] += 1

        review_rows.append(
            {
                "review_id": r.review_id,
                "nm_id": r.nm_id,
                "category": r.category,
                "n_segments": n_segments,
                "gold_aspect_ids_json": json.dumps(sorted(gold_aspects), ensure_ascii=False),
                "A_pred_aspect_ids_json": json.dumps(sorted(pred_a.aspects), ensure_ascii=False),
                "B1_pred_aspect_ids_json": json.dumps(sorted(pred_b1.aspects), ensure_ascii=False),
                "B2_pred_aspect_ids_json": json.dumps(sorted(pred_b2.aspects), ensure_ascii=False),
                "A_pred_count": len(pred_a.aspects),
                "B1_pred_count": len(pred_b1.aspects),
                "B2_pred_count": len(pred_b2.aspects),
            }
        )

    n_reviews = len(rows)
    macro = {mode: _macro(vals) for mode, vals in per_mode_metrics.items()}

    comparison_table_rows: list[dict[str, Any]] = []
    for mode in MODES:
        comparison_table_rows.append(
            {
                "mode": mode,
                "macro_precision": macro[mode]["precision"],
                "macro_recall": macro[mode]["recall"],
                "macro_f1": macro[mode]["f1"],
                "avg_predicted_aspects_per_review": round(sum(pred_count_by_mode[mode]) / n_reviews, 4) if n_reviews else 0.0,
            }
        )

    base = macro["A"]
    delta_rows: list[dict[str, Any]] = []
    for mode in ("B1", "B2"):
        delta_rows.append(
            {
                "mode": mode,
                "delta_precision_vs_A": round(macro[mode]["precision"] - base["precision"], 4),
                "delta_recall_vs_A": round(macro[mode]["recall"] - base["recall"], 4),
                "delta_f1_vs_A": round(macro[mode]["f1"] - base["f1"], 4),
            }
        )

    per_category_rows: list[dict[str, Any]] = []
    categories = sorted({r.category for r in rows})
    for mode in MODES:
        for cat in categories:
            m = _macro(per_mode_category_metrics[mode][cat])
            per_category_rows.append(
                {
                    "mode": mode,
                    "category": cat,
                    "macro_precision": m["precision"],
                    "macro_recall": m["recall"],
                    "macro_f1": m["f1"],
                    "n_reviews": len(per_mode_category_metrics[mode][cat]),
                }
            )

    pairwise_diff_share = {
        pair: round(pairwise_diff_counter[pair] / n_reviews, 4) if n_reviews else 0.0
        for pair in [f"{a}_vs_{b}" for a, b in PAIRWISE_MODE_PAIRS]
    }

    fp_rows: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []
    for mode in MODES:
        fp_rows.extend(
            {"mode": mode, "aspect_id": aid, "fp_count": int(cnt)}
            for aid, cnt in fp_by_mode[mode].most_common(50)
        )
        fn_rows.extend(
            {"mode": mode, "aspect_id": aid, "fn_count": int(cnt)}
            for aid, cnt in fn_by_mode[mode].most_common(50)
        )

    summary = {
        "experiment": "phase2_step2b_segment_surface_matching",
        "n_reviews": n_reviews,
        "metric_space": "review_level_lexically_mappable_gold_subset",
        "matching_definition": "lexical-only exact lemma / exact lemma-sequence",
        "modes": {
            "A": "CandidateExtractor(review) + lexical-only",
            "B1": "RuleBasedClauseSegmenter(review) -> CandidateExtractor(segment) -> union + lexical-only",
            "B2": "RuleBasedClauseSegmenter(review) -> segment normalized lemma unigrams/2-3grams -> union + lexical-only",
        },
        "comparison_table": comparison_table_rows,
        "delta_vs_A": delta_rows,
        "pairwise_pred_diff_share": pairwise_diff_share,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(comparison_table_rows).to_csv(out_dir / "comparison_table.csv", index=False, encoding="utf-8")
    pd.DataFrame(delta_rows).to_csv(out_dir / "delta_vs_A.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_category_rows).to_csv(out_dir / "per_category_breakdown.csv", index=False, encoding="utf-8")
    pd.DataFrame(review_rows).to_csv(out_dir / "review_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(fp_rows).to_csv(out_dir / "top_false_positives_by_aspect.csv", index=False, encoding="utf-8")
    pd.DataFrame(fn_rows).to_csv(out_dir / "top_false_negatives_by_aspect.csv", index=False, encoding="utf-8")
    (out_dir / "pairwise_diff_share.json").write_text(json.dumps(pairwise_diff_share, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="phase2_step2b_segment_surface_matching")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--category", default="all", help="all|physical_goods|consumables|hospitality|services")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase2_step2b_segment_surface_matching")
    args = parser.parse_args()

    domain_vocab_by_category = {
        "physical_goods": ROOT / "src/vocabulary/domain/physical_goods.yaml",
        "consumables": ROOT / "src/vocabulary/domain/consumables.yaml",
        "hospitality": ROOT / "src/vocabulary/domain/hospitality.yaml",
        "services": ROOT / "src/vocabulary/domain/services.yaml",
    }

    rows = _load_rows(ROOT / args.dataset_csv, args.category)
    if not rows:
        raise ValueError("No rows loaded for experiment.")

    hybrid_aspects_by_category = _build_hybrid_aspects_by_category(ROOT / args.core_vocab, domain_vocab_by_category)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / args.out_dir / ts
    run_experiment(rows, hybrid_aspects_by_category, out_dir)
    print(str(out_dir))


if __name__ == "__main__":
    main()
