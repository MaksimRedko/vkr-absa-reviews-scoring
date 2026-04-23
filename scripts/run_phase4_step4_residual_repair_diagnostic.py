from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_phase4_anchor_residual_routing_diagnostic as routing
from src.stages.extraction import CandidateExtractor


REPAIR_META_NOISE_LEMMAS = {
    "раз",
    "уже",
    "после",
    "при",
    "быть",
    "человек",
    "год",
}

BAD_TOP_TERM_LEMMAS = REPAIR_META_NOISE_LEMMAS | {
    "качество",
    "размер",
    "доставка",
    "персонал",
}


@dataclass(frozen=True, slots=True)
class CandidateContext:
    review_id: str
    category: str
    candidate_text: str
    candidate_lemma: str
    sentence_text: str
    context_window_text: str


@dataclass(frozen=True, slots=True)
class ExactAnchorMaps:
    general: dict[str, set[str]]
    domain_by_category: dict[str, dict[str, set[str]]]
    any_domain: dict[str, set[str]]


@dataclass(frozen=True, slots=True)
class RouteResult:
    route: str
    chosen_layer: str
    chosen_anchor: str
    reason: str
    conflict: bool
    general_ok: bool
    domain_ok: bool


def _extract_old_candidates_with_context(
    reviews: list[routing.ReviewRow],
    context_tokens: int,
) -> list[CandidateContext]:
    extractor = CandidateExtractor(ngram_range=(1, 2), min_word_length=3)
    extractor.dependency_filter_enabled = False
    out: list[CandidateContext] = []
    for review in reviews:
        by_lemma: dict[str, CandidateContext] = {}
        for cand in extractor.extract(review.text):
            raw = str(cand.span).strip().lower()
            lemma = routing._normalize(raw)
            if not lemma:
                continue
            sentence_text = str(cand.sentence).strip()
            sentence_tokens = extractor._tokenize(sentence_text)
            start, end = cand.token_indices
            left = max(0, int(start) - context_tokens)
            right = min(len(sentence_tokens), int(end) + context_tokens)
            context_window_text = " ".join(sentence_tokens[left:right]) if sentence_tokens else sentence_text
            by_lemma.setdefault(
                lemma,
                CandidateContext(
                    review_id=review.review_id,
                    category=review.category,
                    candidate_text=raw,
                    candidate_lemma=lemma,
                    sentence_text=sentence_text,
                    context_window_text=context_window_text,
                ),
            )
        out.extend(by_lemma[lemma] for lemma in sorted(by_lemma))
    return out


def _build_exact_anchor_map(aspects: list[routing.AspectDefinition]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for aspect in aspects:
        for term in [aspect.canonical_name, *aspect.synonyms]:
            lemma = routing._normalize(term)
            if lemma:
                out[lemma].add(aspect.canonical_name)
    return dict(out)


def _build_exact_anchor_maps(
    general_aspects: list[routing.AspectDefinition],
    domain_aspects: dict[str, list[routing.AspectDefinition]],
) -> ExactAnchorMaps:
    domain_by_category = {
        category: _build_exact_anchor_map(aspects)
        for category, aspects in domain_aspects.items()
    }
    any_domain: dict[str, set[str]] = defaultdict(set)
    for category, term_map in domain_by_category.items():
        for lemma, names in term_map.items():
            for name in names:
                any_domain[lemma].add(f"{category}::{name}")
    return ExactAnchorMaps(
        general=_build_exact_anchor_map(general_aspects),
        domain_by_category=domain_by_category,
        any_domain=dict(any_domain),
    )


def _first_anchor(names: set[str]) -> str:
    return sorted(names)[0] if names else ""


def _is_repair_meta_noise(candidate_lemma: str) -> bool:
    tokens = [token for token in str(candidate_lemma).split() if token]
    if not tokens:
        return True
    return candidate_lemma in REPAIR_META_NOISE_LEMMAS or all(
        token in REPAIR_META_NOISE_LEMMAS for token in tokens
    )


def _route_repair_candidate(
    candidate: CandidateContext,
    best_general: routing.LayerScore,
    best_domain: routing.LayerScore,
    thresholds: routing.Thresholds,
    noise_reason: str,
    exact_maps: ExactAnchorMaps,
) -> RouteResult:
    domain_exact = exact_maps.domain_by_category.get(candidate.category, {}).get(candidate.candidate_lemma, set())
    general_exact = exact_maps.general.get(candidate.candidate_lemma, set())
    if domain_exact:
        return RouteResult(
            route="domain",
            chosen_layer="domain",
            chosen_anchor=_first_anchor(domain_exact),
            reason="exact_domain_anchor",
            conflict=False,
            general_ok=False,
            domain_ok=True,
        )
    if general_exact:
        return RouteResult(
            route="general",
            chosen_layer="general",
            chosen_anchor=_first_anchor(general_exact),
            reason="exact_general_anchor",
            conflict=False,
            general_ok=True,
            domain_ok=False,
        )
    if noise_reason:
        return RouteResult(
            route="noise",
            chosen_layer="",
            chosen_anchor="",
            reason=noise_reason,
            conflict=False,
            general_ok=False,
            domain_ok=False,
        )
    if _is_repair_meta_noise(candidate.candidate_lemma):
        return RouteResult(
            route="noise",
            chosen_layer="",
            chosen_anchor="",
            reason="repair_meta_noise",
            conflict=False,
            general_ok=False,
            domain_ok=False,
        )
    decision = routing._route_candidate(
        category=candidate.category,
        best_general=best_general,
        best_domain=best_domain,
        thresholds=thresholds,
        noise_reason="",
        mode="domain_priority",
    )
    return RouteResult(
        route=decision.route,
        chosen_layer=decision.chosen_layer,
        chosen_anchor=decision.chosen_anchor,
        reason="score_routing",
        conflict=decision.conflict,
        general_ok=decision.general_ok,
        domain_ok=decision.domain_ok,
    )


def _route_baseline_candidate(
    candidate: CandidateContext,
    best_general: routing.LayerScore,
    best_domain: routing.LayerScore,
    thresholds: routing.Thresholds,
    noise_reason: str,
) -> RouteResult:
    decision = routing._route_candidate(
        category=candidate.category,
        best_general=best_general,
        best_domain=best_domain,
        thresholds=thresholds,
        noise_reason=noise_reason,
        mode="domain_priority",
    )
    return RouteResult(
        route=decision.route,
        chosen_layer=decision.chosen_layer,
        chosen_anchor=decision.chosen_anchor,
        reason=noise_reason or "score_routing",
        conflict=decision.conflict,
        general_ok=decision.general_ok,
        domain_ok=decision.domain_ok,
    )


def _residual_clean_rows(rows: list[dict[str, Any]], route_col: str) -> list[dict[str, Any]]:
    residual = [row for row in rows if row[route_col] == "residual"]
    counts = Counter(str(row["candidate_lemma"]) for row in residual)
    return [row for row in residual if counts[str(row["candidate_lemma"])] >= 2]


def _share(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _single_token_share(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    single = sum(1 for row in rows if len(str(row["candidate_lemma"]).split()) == 1)
    return _share(single, len(rows))


def _anchor_leakage_rows(
    rows: list[dict[str, Any]],
    route_col: str,
    exact_maps: ExactAnchorMaps,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row[route_col] != "residual":
            continue
        lemma = str(row["candidate_lemma"])
        general_names = exact_maps.general.get(lemma, set())
        domain_names = exact_maps.domain_by_category.get(str(row["category"]), {}).get(lemma, set())
        any_domain_names = exact_maps.any_domain.get(lemma, set())
        if not (general_names or domain_names):
            continue
        out.append(
            {
                "stage": "baseline" if route_col == "baseline_route" else "repair_v1",
                "review_id": row["review_id"],
                "category": row["category"],
                "candidate_text": row["candidate_text"],
                "candidate_lemma": lemma,
                "context_window_text": row["context_window_text"],
                "exact_general_anchors": " | ".join(sorted(general_names)),
                "exact_category_domain_anchors": " | ".join(sorted(domain_names)),
                "exact_any_domain_anchors": " | ".join(sorted(any_domain_names)),
                "baseline_route": row["baseline_route"],
                "repair_route": row["repair_route"],
                "best_general_anchor": row["best_general_anchor"],
                "best_general_score": row["best_general_score"],
                "best_domain_anchor": row["best_domain_anchor"],
                "best_domain_score": row["best_domain_score"],
            }
        )
    return out


def _top_terms(
    rows: list[dict[str, Any]],
    stage: str,
    limit: int,
) -> list[dict[str, Any]]:
    counts = Counter(str(row["candidate_lemma"]) for row in rows)
    examples: dict[str, str] = {}
    for row in rows:
        examples.setdefault(str(row["candidate_lemma"]), str(row["candidate_text"]))
    total = sum(counts.values())
    out: list[dict[str, Any]] = []
    for rank, (lemma, count) in enumerate(counts.most_common(limit), start=1):
        out.append(
            {
                "stage": stage,
                "rank": rank,
                "candidate_lemma": lemma,
                "count": count,
                "share": _share(count, total),
                "example_candidate_text": examples.get(lemma, ""),
                "is_known_bad_top_term": str(lemma in BAD_TOP_TERM_LEMMAS).lower(),
            }
        )
    return out


def _sample_residual_after(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    for category_rows in by_category.values():
        rng.shuffle(category_rows)

    sampled: list[dict[str, Any]] = []
    used: set[tuple[str, str, str]] = set()
    quota = max(1, sample_size // max(1, len(by_category)))
    for category in sorted(by_category):
        for row in by_category[category][:quota]:
            key = (str(row["review_id"]), str(row["category"]), str(row["candidate_text"]))
            if key in used:
                continue
            used.add(key)
            sampled.append(row)

    rest = [
        row
        for row in rows
        if (str(row["review_id"]), str(row["category"]), str(row["candidate_text"])) not in used
    ]
    rng.shuffle(rest)
    for row in rest:
        if len(sampled) >= sample_size:
            break
        sampled.append(row)

    return sampled


def _quick_label_after(row: dict[str, Any], thresholds: routing.Thresholds) -> str:
    if _is_repair_meta_noise(str(row["candidate_lemma"])):
        return "looks_noise"
    return routing._quick_residual_label(
        candidate_lemma=str(row["candidate_lemma"]),
        best_general_score=float(row["best_general_score"]),
        best_domain_score=float(row["best_domain_score"]),
        thresholds=thresholds,
    )


def _route_count_table(rows: list[dict[str, Any]], route_col: str) -> dict[str, int]:
    counts = Counter(str(row[route_col]) for row in rows)
    return {route: counts[route] for route in ["general", "domain", "overlap", "residual", "noise"]}


def _stage_stats(
    rows: list[dict[str, Any]],
    route_col: str,
    exact_maps: ExactAnchorMaps,
) -> dict[str, Any]:
    residual_raw = [row for row in rows if row[route_col] == "residual"]
    residual_clean = _residual_clean_rows(rows, route_col)
    leakage_raw = _anchor_leakage_rows(rows, route_col, exact_maps)
    clean_keys = {
        (str(row["review_id"]), str(row["category"]), str(row["candidate_lemma"]))
        for row in residual_clean
    }
    leakage_clean = [
        row
        for row in leakage_raw
        if (str(row["review_id"]), str(row["category"]), str(row["candidate_lemma"])) in clean_keys
    ]
    top10 = _top_terms(residual_clean, "baseline" if route_col == "baseline_route" else "repair_v1", 10)
    top10_bad = sum(1 for row in top10 if row["candidate_lemma"] in BAD_TOP_TERM_LEMMAS)
    return {
        "route_counts": _route_count_table(rows, route_col),
        "residual_raw_n": len(residual_raw),
        "residual_clean_n": len(residual_clean),
        "residual_raw_unique_lemmas": len({str(row["candidate_lemma"]) for row in residual_raw}),
        "residual_clean_unique_lemmas": len({str(row["candidate_lemma"]) for row in residual_clean}),
        "single_token_raw_share": _single_token_share(residual_raw),
        "single_token_clean_share": _single_token_share(residual_clean),
        "exact_anchor_leakage_raw_n": len(leakage_raw),
        "exact_anchor_leakage_clean_n": len(leakage_clean),
        "exact_anchor_leakage_clean_share": _share(len(leakage_clean), len(residual_clean)),
        "bad_terms_in_top10": top10_bad,
    }


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _write_summary(
    out_dir: Path,
    rows: list[dict[str, Any]],
    baseline_stats: dict[str, Any],
    repair_stats: dict[str, Any],
    sample_after: list[dict[str, Any]],
    recommendation: str,
) -> None:
    sample_counts = Counter(str(row["quick_label"]) for row in sample_after)
    total = len(rows)
    repair_clean = _residual_clean_rows(rows, "repair_route")
    repair_top10 = ", ".join(
        str(row["candidate_lemma"]) for row in _top_terms(repair_clean, "repair_v1", 10)
    )
    lines = [
        "# phase4_step4_residual_repair_diagnostic",
        "",
        "## Verdict",
        f"- recommendation: `{recommendation}`",
        "- HDBSCAN was not run.",
        "",
        "## Route Counts",
        "| stage | general | domain | overlap | residual | noise |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, stats in [("baseline", baseline_stats), ("repair_v1", repair_stats)]:
        counts = stats["route_counts"]
        lines.append(
            f"| {stage} | {counts['general']} | {counts['domain']} | {counts['overlap']} | "
            f"{counts['residual']} | {counts['noise']} |"
        )

    lines.extend(
        [
            "",
            "## Residual Quality",
            "| metric | baseline | repair_v1 |",
            "|---|---:|---:|",
            f"| residual raw | {baseline_stats['residual_raw_n']} | {repair_stats['residual_raw_n']} |",
            f"| residual clean | {baseline_stats['residual_clean_n']} | {repair_stats['residual_clean_n']} |",
            f"| clean unique lemmas | {baseline_stats['residual_clean_unique_lemmas']} | {repair_stats['residual_clean_unique_lemmas']} |",
            f"| exact anchor leakage clean | {baseline_stats['exact_anchor_leakage_clean_n']} | {repair_stats['exact_anchor_leakage_clean_n']} |",
            f"| exact anchor leakage clean share | {_format_pct(baseline_stats['exact_anchor_leakage_clean_share'])} | {_format_pct(repair_stats['exact_anchor_leakage_clean_share'])} |",
            f"| single-token clean share | {_format_pct(baseline_stats['single_token_clean_share'])} | {_format_pct(repair_stats['single_token_clean_share'])} |",
            f"| bad terms in top-10 | {baseline_stats['bad_terms_in_top10']} | {repair_stats['bad_terms_in_top10']} |",
            "",
            "## Residual Sample After",
            f"- sample size: {len(sample_after)}",
            f"- looks_useful: {sample_counts['looks_useful']}",
            f"- unclear: {sample_counts['unclear']}",
            f"- looks_noise: {sample_counts['looks_noise']}",
            "",
            "## Interpretation",
        ]
    )

    leakage_fixed = repair_stats["exact_anchor_leakage_clean_n"] == 0
    top_terms_fixed = repair_stats["bad_terms_in_top10"] == 0
    single_token_delta = baseline_stats["single_token_clean_share"] - repair_stats["single_token_clean_share"]
    lines.append(f"- exact anchor leakage fixed: {str(leakage_fixed).lower()}")
    lines.append(f"- bad top terms removed: {str(top_terms_fixed).lower()}")
    lines.append(f"- single-token clean share delta: {_format_pct(single_token_delta)}")
    lines.append(f"- repair top-10 residual terms: {repair_top10}")
    lines.append("- residual is cleaner, but still mostly context-free one-token object/entity lemmas")
    lines.append(f"- total candidates compared: {total}")
    (out_dir / "residual_repair_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 step 4 residual repair diagnostic")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase4_step4_residual_repair_diagnostic")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--top-limit", type=int, default=40)
    parser.add_argument("--context-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reviews = routing._load_reviews(ROOT / args.dataset_csv)
    candidates = _extract_old_candidates_with_context(reviews, context_tokens=args.context_tokens)

    general_aspects = routing._load_aspects(ROOT / args.core_vocab)
    domain_paths = {
        "physical_goods": ROOT / "src/vocabulary/domain/physical_goods.yaml",
        "consumables": ROOT / "src/vocabulary/domain/consumables.yaml",
        "hospitality": ROOT / "src/vocabulary/domain/hospitality.yaml",
        "services": ROOT / "src/vocabulary/domain/services.yaml",
    }
    domain_aspects = {category: routing._load_aspects(path) for category, path in domain_paths.items()}
    exact_maps = _build_exact_anchor_maps(general_aspects, domain_aspects)

    model = routing._load_encoder()
    embedding_cache: dict[str, np.ndarray] = {}
    routing._encode_texts_cached(model, list({candidate.candidate_lemma for candidate in candidates}), embedding_cache)

    general_ids, general_names, general_matrix = routing._build_anchor_bank(model, general_aspects, embedding_cache)
    domain_banks = {
        category: routing._build_anchor_bank(model, aspects, embedding_cache)
        for category, aspects in domain_aspects.items()
    }

    thresholds = routing.Thresholds(
        t_general=0.88,
        m_general=0.04,
        t_domain=0.88,
        m_domain=0.04,
        t_general_conflict=0.83,
        t_domain_conflict=0.83,
        c_overlap=0.02,
        weak_score_floor=0.68,
    )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_vec = embedding_cache[candidate.candidate_lemma]
        best_general = routing._compute_layer_score(
            candidate_vec=candidate_vec,
            anchor_ids=general_ids,
            anchor_names=general_names,
            anchor_matrix=general_matrix,
        )
        domain_ids, domain_names, domain_matrix = domain_banks.get(
            candidate.category,
            ([], [], np.zeros((0, 1), dtype=np.float32)),
        )
        best_domain = routing._compute_layer_score(
            candidate_vec=candidate_vec,
            anchor_ids=domain_ids,
            anchor_names=domain_names,
            anchor_matrix=domain_matrix,
        )
        noise_reason = routing._noise_reason(
            candidate_lemma=candidate.candidate_lemma,
            best_general_score=best_general.best_score,
            best_domain_score=best_domain.best_score,
            weak_score_floor=thresholds.weak_score_floor,
        )
        baseline = _route_baseline_candidate(candidate, best_general, best_domain, thresholds, noise_reason)
        repair = _route_repair_candidate(candidate, best_general, best_domain, thresholds, noise_reason, exact_maps)
        exact_general = exact_maps.general.get(candidate.candidate_lemma, set())
        exact_domain = exact_maps.domain_by_category.get(candidate.category, {}).get(candidate.candidate_lemma, set())
        rows.append(
            {
                "review_id": candidate.review_id,
                "category": candidate.category,
                "candidate_text": candidate.candidate_text,
                "candidate_lemma": candidate.candidate_lemma,
                "context_window_text": candidate.context_window_text,
                "sentence_text": candidate.sentence_text,
                "best_general_anchor": best_general.best_anchor_name,
                "best_general_score": round(best_general.best_score, 4),
                "second_general_score": round(best_general.second_score, 4),
                "best_domain_anchor": best_domain.best_anchor_name,
                "best_domain_score": round(best_domain.best_score, 4),
                "second_domain_score": round(best_domain.second_score, 4),
                "exact_general_anchors": " | ".join(sorted(exact_general)),
                "exact_category_domain_anchors": " | ".join(sorted(exact_domain)),
                "baseline_route": baseline.route,
                "baseline_chosen_layer": baseline.chosen_layer,
                "baseline_chosen_anchor": baseline.chosen_anchor,
                "baseline_reason": baseline.reason,
                "repair_route": repair.route,
                "repair_chosen_layer": repair.chosen_layer,
                "repair_chosen_anchor": repair.chosen_anchor,
                "repair_reason": repair.reason,
                "route_changed": str(baseline.route != repair.route).lower(),
            }
        )

    baseline_clean = _residual_clean_rows(rows, "baseline_route")
    repair_clean = _residual_clean_rows(rows, "repair_route")
    baseline_stats = _stage_stats(rows, "baseline_route", exact_maps)
    repair_stats = _stage_stats(rows, "repair_route", exact_maps)

    sample_after_rows = _sample_residual_after(repair_clean, sample_size=args.sample_size, seed=args.seed)
    sample_after: list[dict[str, Any]] = []
    for row in sample_after_rows:
        quick_label = _quick_label_after(row, thresholds)
        enriched = {
            "review_id": row["review_id"],
            "category": row["category"],
            "candidate_text": row["candidate_text"],
            "candidate_lemma": row["candidate_lemma"],
            "context_window_text": row["context_window_text"],
            "best_general_anchor": row["best_general_anchor"],
            "best_general_score": row["best_general_score"],
            "best_domain_anchor": row["best_domain_anchor"],
            "best_domain_score": row["best_domain_score"],
            "quick_label": quick_label,
        }
        sample_after.append(enriched)

    sample_counts = Counter(str(row["quick_label"]) for row in sample_after)
    sample_useful_share = _share(sample_counts["looks_useful"], len(sample_after))
    sample_noise_share = _share(sample_counts["looks_noise"], len(sample_after))
    single_token_delta = baseline_stats["single_token_clean_share"] - repair_stats["single_token_clean_share"]
    recommendation = (
        "return_to_hdbscan"
        if repair_stats["exact_anchor_leakage_clean_n"] == 0
        and repair_stats["bad_terms_in_top10"] == 0
        and (single_token_delta >= 0.05 or repair_stats["single_token_clean_share"] <= 0.70)
        and sample_useful_share >= 0.45
        and sample_noise_share <= 0.20
        else "kill_residual_branch"
    )

    out_dir = ROOT / args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_dir / "residual_before_after.csv", index=False, encoding="utf-8")

    leakage_rows = _anchor_leakage_rows(rows, "baseline_route", exact_maps)
    leakage_rows.extend(_anchor_leakage_rows(rows, "repair_route", exact_maps))
    pd.DataFrame(leakage_rows).to_csv(out_dir / "exact_anchor_leakage.csv", index=False, encoding="utf-8")

    top_rows = _top_terms(baseline_clean, "baseline", args.top_limit)
    top_rows.extend(_top_terms(repair_clean, "repair_v1", args.top_limit))
    pd.DataFrame(top_rows).to_csv(out_dir / "residual_top_terms_before_after.csv", index=False, encoding="utf-8")
    pd.DataFrame(sample_after).to_csv(out_dir / "residual_sample_after.csv", index=False, encoding="utf-8")

    _write_summary(
        out_dir=out_dir,
        rows=rows,
        baseline_stats=baseline_stats,
        repair_stats=repair_stats,
        sample_after=sample_after,
        recommendation=recommendation,
    )

    run_summary = {
        "experiment": "phase4_step4_residual_repair_diagnostic",
        "n_reviews": len(reviews),
        "n_candidates": len(candidates),
        "thresholds": {
            "T_general": thresholds.t_general,
            "M_general": thresholds.m_general,
            "T_domain": thresholds.t_domain,
            "M_domain": thresholds.m_domain,
            "T_general_conflict": thresholds.t_general_conflict,
            "T_domain_conflict": thresholds.t_domain_conflict,
            "C_overlap": thresholds.c_overlap,
            "weak_score_floor": thresholds.weak_score_floor,
        },
        "baseline_stats": baseline_stats,
        "repair_stats": repair_stats,
        "sample_counts": dict(sample_counts),
        "recommendation": recommendation,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "recommendation": recommendation,
                "baseline_residual_clean": baseline_stats["residual_clean_n"],
                "repair_residual_clean": repair_stats["residual_clean_n"],
                "baseline_leakage_clean": baseline_stats["exact_anchor_leakage_clean_n"],
                "repair_leakage_clean": repair_stats["exact_anchor_leakage_clean_n"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
