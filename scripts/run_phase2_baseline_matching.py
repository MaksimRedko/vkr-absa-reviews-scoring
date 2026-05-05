from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymorphy3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import configs as _cfg_module
from src.stages.extraction import CandidateExtractor
from src.vocabulary.loader import AspectDefinition, Vocabulary

_cfg_module.config.discovery.dependency_filter_enabled = False  # type: ignore[attr-defined]
_MORPH = pymorphy3.MorphAnalyzer()


@dataclass(frozen=True, slots=True)
class ReviewSample:
    review_id: str
    nm_id: int
    category: str
    text: str
    true_labels_raw: frozenset[str]
    true_labels_lemma: frozenset[str]


@dataclass(slots=True)
class CosineFilterState:
    model: Any
    term_embedding_cache: dict[str, np.ndarray]
    aspect_anchor_by_category: dict[str, dict[str, np.ndarray]]


def _normalize(text: str) -> str:
    tokens = [token.strip().lower() for token in str(text).replace("-", " ").split() if token.strip()]
    lemmas: list[str] = []
    for token in tokens:
        parses = _MORPH.parse(token)
        lemmas.append(str(parses[0].normal_form if parses else token).lower())
    return " ".join(lemmas)


def _parse_true_labels(raw: Any) -> frozenset[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return frozenset()
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return frozenset()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return frozenset()
    if not isinstance(parsed, dict):
        return frozenset()
    return frozenset(str(k).strip() for k in parsed.keys() if str(k).strip())


def _load_reviews(dataset_csv: Path) -> list[ReviewSample]:
    df = pd.read_csv(dataset_csv, dtype={"id": str})
    required = {"nm_id", "id", "full_text", "true_labels", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    rows: list[ReviewSample] = []
    for _, row in df.iterrows():
        text = str(row["full_text"]).strip()
        if not text:
            continue
        true_raw = _parse_true_labels(row["true_labels"])
        true_lemma = frozenset(_normalize(x) for x in true_raw)
        rows.append(
            ReviewSample(
                review_id=str(row["id"]),
                nm_id=int(row["nm_id"]),
                category=str(row["category"]).strip(),
                text=text,
                true_labels_raw=true_raw,
                true_labels_lemma=true_lemma,
            )
        )
    return rows


def _build_vocabulary(paths: list[Path]) -> list[AspectDefinition]:
    seen: set[str] = set()
    out: list[AspectDefinition] = []
    for path in paths:
        vocab = Vocabulary.load_from_yaml(path)
        for asp in vocab.aspects:
            if asp.id in seen:
                continue
            seen.add(asp.id)
            out.append(asp)
    return out


def _term_indexes(aspects: list[AspectDefinition]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    term_to_aspects: dict[str, set[str]] = defaultdict(set)
    aspect_to_terms: dict[str, set[str]] = defaultdict(set)
    for asp in aspects:
        terms = [asp.canonical_name] + asp.synonyms
        for term in terms:
            lemma = _normalize(term)
            if not lemma:
                continue
            term_to_aspects[lemma].add(asp.id)
            aspect_to_terms[asp.id].add(lemma)
    return term_to_aspects, aspect_to_terms


def _extract_candidates(text: str, extractor: CandidateExtractor) -> set[str]:
    cands = extractor.extract(text)
    return {cand.span for cand in cands if str(cand.span).strip()}


def _split_segments_rule_based(text: str) -> list[str]:
    # Simple reproducible segmentation by punctuation and discourse separators.
    parts = re.split(r"[.!?;\n\r]+|(?:\s+-\s+)|(?:\s+—\s+)|(?:,\s+но\s+)|(?:,\s+а\s+)|(?:,\s+однако\s+)", text)
    return [p.strip() for p in parts if p and p.strip()]


def _extract_segment_lemma_units(text: str, ngram_range: tuple[int, int] = (1, 2), min_word_length: int = 3) -> set[str]:
    # Keep signature for backward compatibility; not used in phase2_step2.
    units: set[str] = set()
    for seg in _split_segments_rule_based(text):
        lemmas = [x for x in _normalize(seg).split() if x and len(x) >= min_word_length]
        lo, hi = ngram_range
        for n in range(lo, hi + 1):
            if n <= 0 or len(lemmas) < n:
                continue
            for i in range(0, len(lemmas) - n + 1):
                units.add(" ".join(lemmas[i : i + n]))
    return units


def _extract_candidate_lemmas_by_unit(text: str, extractor: CandidateExtractor, unit_of_analysis: str) -> set[str]:
    if unit_of_analysis == "candidates":
        spans = _extract_candidates(text, extractor)
        out = {_normalize(span) for span in spans}
        out.discard("")
        return out
    if unit_of_analysis == "segments":
        # Same extractor, but applied per rule-based segment.
        out: set[str] = set()
        for seg in _split_segments_rule_based(text):
            spans = _extract_candidates(seg, extractor)
            for span in spans:
                lemma = _normalize(span)
                if lemma:
                    out.add(lemma)
        return out
    raise ValueError(f"unsupported unit_of_analysis: {unit_of_analysis}")


def _eval_prf(pred: set[str], true: set[str]) -> tuple[float, float, float]:
    if not pred and not true:
        return 1.0, 1.0, 1.0
    if not pred or not true:
        return 0.0, 0.0, 0.0
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true) if true else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def _is_subtuple(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    if not short:
        return False
    if len(short) > len(long):
        return False
    window = len(short)
    for i in range(0, len(long) - window + 1):
        if long[i : i + window] == short:
            return True
    return False


def _is_relaxed_lexical_match(candidate_term: str, vocab_term: str) -> bool:
    c = tuple(x for x in candidate_term.split() if x)
    v = tuple(x for x in vocab_term.split() if x)
    if not c or not v:
        return False
    if c == v:
        return True
    if len(c) == 1 and c[0] in v:
        return True
    if len(v) == 1 and v[0] in c:
        return True
    if _is_subtuple(c, v) or _is_subtuple(v, c):
        return True
    return False


def _match_terms(
    candidate_lemmas: set[str],
    term_to_aspects: dict[str, set[str]],
    matching_mode: str,
) -> list[str]:
    vocab_terms = set(term_to_aspects.keys())
    if matching_mode == "lexical_only":
        return sorted(candidate_lemmas & vocab_terms)
    if matching_mode == "relaxed_lexical":
        matched: set[str] = set(candidate_lemmas & vocab_terms)
        for cand in candidate_lemmas:
            for vocab_term in vocab_terms:
                if vocab_term in matched:
                    continue
                if _is_relaxed_lexical_match(cand, vocab_term):
                    matched.add(vocab_term)
        return sorted(matched)
    raise ValueError(f"unsupported matching_mode: {matching_mode}")


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _load_encoder_once() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_cfg_module.config.models.encoder_path)


def _encode_texts_cached(
    model: Any,
    texts: list[str],
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    unique = sorted({x for x in texts if x and x not in cache})
    if unique:
        embeddings = model.encode(unique, show_progress_bar=False, convert_to_numpy=True)
        normalized = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        for idx, text in enumerate(unique):
            cache[text] = normalized[idx]
    return {text: cache[text] for text in texts if text in cache}


def _build_aspect_anchors(
    model: Any,
    aspects: list[AspectDefinition],
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    anchors: dict[str, np.ndarray] = {}
    for asp in aspects:
        terms_raw = [asp.canonical_name] + list(asp.synonyms)
        terms = [t for t in (_normalize(x) for x in terms_raw) if t]
        if not terms:
            continue
        vectors = _encode_texts_cached(model, terms, cache)
        if not vectors:
            continue
        mat = np.stack([vectors[t] for t in terms], axis=0).astype(np.float32)
        centroid = _l2_normalize(mat.mean(axis=0, keepdims=True))[0]
        anchors[asp.id] = centroid
    return anchors


def _apply_cosine_filter_to_aspects(
    matched_terms: list[str],
    term_to_aspects: dict[str, set[str]],
    anchors: dict[str, np.ndarray],
    model: Any,
    cache: dict[str, np.ndarray],
    tau: float,
) -> set[str]:
    if not matched_terms:
        return set()
    candidate_to_aspects = {term: set(term_to_aspects.get(term, set())) for term in matched_terms}
    candidate_vectors = _encode_texts_cached(model, matched_terms, cache)
    out: set[str] = set()
    for term, aspect_ids in candidate_to_aspects.items():
        vec = candidate_vectors.get(term)
        if vec is None:
            continue
        for aid in aspect_ids:
            anchor = anchors.get(aid)
            if anchor is None:
                continue
            score = float(np.dot(vec, anchor))
            if score >= tau:
                out.add(aid)
    return out


def _apply_cosine_margin_gate_to_aspects(
    matched_terms: list[str],
    term_to_aspects: dict[str, set[str]],
    anchors: dict[str, np.ndarray],
    model: Any,
    cache: dict[str, np.ndarray],
    tau_s: float,
    tau_m: float,
) -> set[str]:
    if not matched_terms or not anchors:
        return set()
    candidate_vectors = _encode_texts_cached(model, matched_terms, cache)
    anchor_ids = sorted(anchors.keys())
    anchor_matrix = np.stack([anchors[aid] for aid in anchor_ids], axis=0).astype(np.float32)
    anchor_index_by_id = {aid: idx for idx, aid in enumerate(anchor_ids)}
    out: set[str] = set()
    for term in matched_terms:
        lexical_aspects = term_to_aspects.get(term, set())
        if not lexical_aspects:
            continue
        lexical_indices = [anchor_index_by_id[aid] for aid in lexical_aspects if aid in anchor_index_by_id]
        if not lexical_indices:
            continue
        vec = candidate_vectors.get(term)
        if vec is None:
            continue
        scores = anchor_matrix @ vec
        if scores.size == 0:
            continue
        lexical_scores = scores[lexical_indices]
        best_local_pos = int(np.argmax(lexical_scores))
        best_lexical_score = float(lexical_scores[best_local_pos])
        best_lexical_idx = lexical_indices[best_local_pos]
        best_lexical_aspect = anchor_ids[best_lexical_idx]

        nonlexical_mask = np.ones(scores.shape[0], dtype=bool)
        nonlexical_mask[lexical_indices] = False
        if bool(np.any(nonlexical_mask)):
            best_nonlexical_score = float(np.max(scores[nonlexical_mask]))
        else:
            # No non-lexical competitors in this category space.
            best_nonlexical_score = -1.0

        margin = best_lexical_score - best_nonlexical_score
        if best_lexical_score >= tau_s and margin >= tau_m:
            out.add(best_lexical_aspect)
    return out


def run_experiment(
    *,
    run_name: str,
    reviews: list[ReviewSample],
    core_vocab_path: Path,
    domain_vocab_by_category: dict[str, Path],
    use_hybrid: bool,
    unit_of_analysis: str,
    matching_mode: str,
    cosine_tau: float,
    cosine_tau_s: float,
    cosine_tau_m: float,
    output_root: Path,
    debug_sample_review_ids: set[str] | None = None,
) -> Path:
    extractor = CandidateExtractor(ngram_range=(1, 2), min_word_length=3)
    extractor.dependency_filter_enabled = False

    core_aspects = _build_vocabulary([core_vocab_path])
    domain_aspects_cache: dict[str, list[AspectDefinition]] = {}
    cosine_state: CosineFilterState | None = None

    def _aspects_for_category(category: str) -> list[AspectDefinition]:
        if not use_hybrid:
            return core_aspects
        if category not in domain_aspects_cache:
            domain_path = domain_vocab_by_category.get(category)
            paths = [core_vocab_path] + ([domain_path] if domain_path else [])
            domain_aspects_cache[category] = _build_vocabulary(paths)
        return domain_aspects_cache[category]

    if matching_mode in {"cosine_filter", "cosine_margin_gate"}:
        model = _load_encoder_once()
        embedding_cache: dict[str, np.ndarray] = {}
        if use_hybrid:
            category_list = sorted(set(rev.category for rev in reviews))
            anchors_by_category = {
                cat: _build_aspect_anchors(model, _aspects_for_category(cat), embedding_cache)
                for cat in category_list
            }
        else:
            anchors = _build_aspect_anchors(model, core_aspects, embedding_cache)
            anchors_by_category = defaultdict(dict)
            for rev in reviews:
                anchors_by_category[rev.category] = anchors
        cosine_state = CosineFilterState(
            model=model,
            term_embedding_cache=embedding_cache,
            aspect_anchor_by_category=dict(anchors_by_category),
        )

    all_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    per_category_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    fp_by_aspect: Counter[str] = Counter()
    fn_by_aspect: Counter[str] = Counter()

    for review in reviews:
        aspects = _aspects_for_category(review.category)
        term_to_aspects, _aspect_to_terms = _term_indexes(aspects)
        candidate_lemmas = _extract_candidate_lemmas_by_unit(review.text, extractor, unit_of_analysis)

        if matching_mode in {"cosine_filter", "cosine_margin_gate"}:
            matched_terms = _match_terms(candidate_lemmas, term_to_aspects, "lexical_only")
        else:
            matched_terms = _match_terms(candidate_lemmas, term_to_aspects, matching_mode)

        # multi-label mapping: one term can map to multiple aspects
        if matching_mode == "cosine_filter":
            if cosine_state is None:
                raise RuntimeError("cosine_state was not initialized")
            anchors = cosine_state.aspect_anchor_by_category.get(review.category, {})
            pred_aspect_ids = _apply_cosine_filter_to_aspects(
                matched_terms=matched_terms,
                term_to_aspects=term_to_aspects,
                anchors=anchors,
                model=cosine_state.model,
                cache=cosine_state.term_embedding_cache,
                tau=cosine_tau,
            )
        elif matching_mode == "cosine_margin_gate":
            if cosine_state is None:
                raise RuntimeError("cosine_state was not initialized")
            anchors = cosine_state.aspect_anchor_by_category.get(review.category, {})
            pred_aspect_ids = _apply_cosine_margin_gate_to_aspects(
                matched_terms=matched_terms,
                term_to_aspects=term_to_aspects,
                anchors=anchors,
                model=cosine_state.model,
                cache=cosine_state.term_embedding_cache,
                tau_s=cosine_tau_s,
                tau_m=cosine_tau_m,
            )
        else:
            pred_aspect_ids = set()
            for term in matched_terms:
                pred_aspect_ids.update(term_to_aspects[term])

        # Explicit gold mapping: gold label -> aspect id via lexical term index.
        true_aspect_ids: set[str] = set()
        for gold_label in review.true_labels_lemma:
            true_aspect_ids.update(term_to_aspects.get(gold_label, set()))

        p, r, f1 = _eval_prf(pred_aspect_ids, true_aspect_ids)
        per_category_rows[review.category].append((p, r, f1))

        # diagnostics by aspect id in the same space (pred vs gold-mapped aspect ids)
        for aid in (pred_aspect_ids - true_aspect_ids):
            fp_by_aspect[aid] += 1
        for aid in (true_aspect_ids - pred_aspect_ids):
            fn_by_aspect[aid] += 1

        all_rows.append(
            {
                "review_id": review.review_id,
                "nm_id": review.nm_id,
                "category": review.category,
                "predicted_aspect_ids_json": json.dumps(sorted(pred_aspect_ids), ensure_ascii=False),
                "true_aspect_ids_json": json.dumps(sorted(true_aspect_ids), ensure_ascii=False),
                "matched_terms_json": json.dumps(matched_terms, ensure_ascii=False),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }
        )
        if debug_sample_review_ids and review.review_id in debug_sample_review_ids:
            matched_term_to_aspects = {
                term: sorted(term_to_aspects.get(term, set())) for term in matched_terms
            }
            debug_rows.append(
                {
                    "review_id": review.review_id,
                    "nm_id": review.nm_id,
                    "category": review.category,
                    "unit_of_analysis": unit_of_analysis,
                    "text_preview": review.text[:500],
                    "candidate_lemmas_sorted": sorted(candidate_lemmas),
                    "matched_terms_sorted": matched_terms,
                    "matched_term_to_aspect_ids": matched_term_to_aspects,
                    "predicted_aspect_ids_sorted": sorted(pred_aspect_ids),
                    "gold_labels_lemma_sorted": sorted(review.true_labels_lemma),
                    "gold_aspect_ids_sorted": sorted(true_aspect_ids),
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f1, 4),
                }
            )

    review_df = pd.DataFrame(all_rows)
    macro_p = float(review_df["precision"].mean()) if not review_df.empty else 0.0
    macro_r = float(review_df["recall"].mean()) if not review_df.empty else 0.0
    macro_f1 = float(review_df["f1"].mean()) if not review_df.empty else 0.0

    per_category_records: list[dict[str, Any]] = []
    for cat, vals in sorted(per_category_rows.items()):
        ps = [x[0] for x in vals]
        rs = [x[1] for x in vals]
        fs = [x[2] for x in vals]
        per_category_records.append(
            {
                "category": cat,
                "n_reviews": len(vals),
                "macro_precision": round(sum(ps) / len(ps), 4),
                "macro_recall": round(sum(rs) / len(rs), 4),
                "macro_f1": round(sum(fs) / len(fs), 4),
            }
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"{run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    review_df.to_csv(out_dir / "review_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_category_records).to_csv(out_dir / "per_category_breakdown.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"aspect_id": k, "fp_count": int(v)} for k, v in fp_by_aspect.most_common()]).to_csv(
        out_dir / "top_false_positives_by_aspect.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame([{"aspect_id": k, "fn_count": int(v)} for k, v in fn_by_aspect.most_common()]).to_csv(
        out_dir / "top_false_negatives_by_aspect.csv", index=False, encoding="utf-8"
    )
    if debug_rows:
        (out_dir / "debug_review_traces.json").write_text(
            json.dumps(debug_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "run_name": run_name,
        "matching_definition": matching_mode,
        "cosine_tau": round(float(cosine_tau), 4) if matching_mode == "cosine_filter" else None,
        "cosine_tau_s": round(float(cosine_tau_s), 4) if matching_mode == "cosine_margin_gate" else None,
        "cosine_tau_m": round(float(cosine_tau_m), 4) if matching_mode == "cosine_margin_gate" else None,
        "mapping_definition": "multi-label",
        "unit_of_analysis": unit_of_analysis,
        "n_reviews": int(len(review_df)),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "use_hybrid": use_hybrid,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 baseline lexical matching runner")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase2_step1_baseline_matching_current_extractor")
    parser.add_argument(
        "--experiment",
        choices=["phase2_step1", "phase2_step2", "phase2_step3", "phase2_step4", "phase2_step5"],
        default="phase2_step1",
    )
    parser.add_argument("--tau", type=float, default=0.35, help="Cosine threshold for phase2_step4")
    parser.add_argument("--tau-s", type=float, default=0.9, help="Top1 cosine threshold for phase2_step5")
    parser.add_argument("--tau-m", type=float, default=0.03, help="Cosine margin threshold for phase2_step5")
    parser.add_argument("--debug-samples", type=int, default=0, help="Number of random reviews to trace end-to-end")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for debug sample selection")
    args = parser.parse_args()

    reviews = _load_reviews(ROOT / args.dataset_csv)
    domain_vocab_by_category = {
        "physical_goods": ROOT / "src/vocabulary/domain/physical_goods.yaml",
        "consumables": ROOT / "src/vocabulary/domain/consumables.yaml",
        "hospitality": ROOT / "src/vocabulary/domain/hospitality.yaml",
        "services": ROOT / "src/vocabulary/domain/services.yaml",
    }
    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    debug_sample_review_ids: set[str] | None = None
    if args.debug_samples > 0:
        rng = random.Random(args.seed)
        sampled = rng.sample(reviews, min(args.debug_samples, len(reviews)))
        debug_sample_review_ids = {r.review_id for r in sampled}

    if args.experiment == "phase2_step1":
        run_a = run_experiment(
            run_name="run_A_core_only",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=False,
            unit_of_analysis="candidates",
            matching_mode="lexical_only",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
        run_b = run_experiment(
            run_name="run_B_hybrid",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=True,
            unit_of_analysis="candidates",
            matching_mode="lexical_only",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
    elif args.experiment == "phase2_step2":
        run_a = run_experiment(
            run_name="run_A_candidates_hybrid",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=True,
            unit_of_analysis="candidates",
            matching_mode="lexical_only",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
        run_b = run_experiment(
            run_name="run_B_segments_hybrid",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=True,
            unit_of_analysis="segments",
            matching_mode="lexical_only",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
    elif args.experiment == "phase2_step3":
        run_a = run_experiment(
            run_name="run_A_lexical_only_hybrid",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=True,
            unit_of_analysis="candidates",
            matching_mode="lexical_only",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
        run_b = run_experiment(
            run_name="run_B_relaxed_lexical_hybrid",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=True,
            unit_of_analysis="candidates",
            matching_mode="relaxed_lexical",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
    elif args.experiment == "phase2_step4":
        run_a = run_experiment(
            run_name="run_A_lexical_only_hybrid",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=True,
            unit_of_analysis="candidates",
            matching_mode="lexical_only",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
        run_b = run_experiment(
            run_name="run_B_lexical_plus_cosine_filter_hybrid",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=True,
            unit_of_analysis="candidates",
            matching_mode="cosine_filter",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
    else:
        run_a = run_experiment(
            run_name="run_A_lexical_only_hybrid",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=True,
            unit_of_analysis="candidates",
            matching_mode="lexical_only",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
        run_b = run_experiment(
            run_name="run_B_lexical_plus_cosine_margin_gate_hybrid",
            reviews=reviews,
            core_vocab_path=ROOT / args.core_vocab,
            domain_vocab_by_category=domain_vocab_by_category,
            use_hybrid=True,
            unit_of_analysis="candidates",
            matching_mode="cosine_margin_gate",
            cosine_tau=args.tau,
            cosine_tau_s=args.tau_s,
            cosine_tau_m=args.tau_m,
            output_root=out_root,
            debug_sample_review_ids=debug_sample_review_ids,
        )
    print(json.dumps({"run_A_dir": str(run_a), "run_B_dir": str(run_b)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
