from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hdbscan  # type: ignore[import-not-found]
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_phase4_anchor_residual_routing_diagnostic as routing
import scripts.run_phase4_step4_residual_repair_diagnostic as repair


CONTENT_STOP_LEMMAS = {
    "а",
    "без",
    "бы",
    "быть",
    "в",
    "во",
    "вот",
    "все",
    "всё",
    "где",
    "для",
    "до",
    "его",
    "ее",
    "её",
    "если",
    "еще",
    "ещё",
    "и",
    "или",
    "их",
    "как",
    "который",
    "ли",
    "на",
    "над",
    "не",
    "ни",
    "но",
    "о",
    "об",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "по",
    "под",
    "после",
    "при",
    "про",
    "раз",
    "с",
    "со",
    "так",
    "такой",
    "там",
    "то",
    "только",
    "тот",
    "тут",
    "у",
    "уже",
    "чем",
    "что",
    "это",
    "этот",
}

GENERIC_CONTEXT_LEMMAS = CONTENT_STOP_LEMMAS | routing.GENERIC_EMOTION_LEMMAS | routing.GENERIC_OBJECT_LEMMAS | {
    "день",
    "год",
    "звезда",
    "минус",
    "плюс",
    "покупка",
    "проблема",
    "раз",
    "человек",
    "штука",
}


@dataclass(frozen=True, slots=True)
class ClusterRecord:
    cluster_id: int
    size: int
    medoid_text: str
    top_candidate_lemmas: str
    top_context_terms: str
    nearest_general_anchor: str
    nearest_domain_anchor: str
    nearest_general_score: float
    nearest_domain_score: float
    dominant_categories: str
    heuristic_label: str
    heuristic_comment: str
    typical_context_snippets: list[str]
    is_duplicate_general: bool
    is_duplicate_domain: bool


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", str(text).lower(), flags=re.UNICODE)
    lemmas = [routing._normalize(token) for token in tokens]
    return [
        lemma
        for lemma in lemmas
        if lemma
        and len(lemma) >= 3
        and lemma not in CONTENT_STOP_LEMMAS
        and not lemma.isdigit()
    ]


def _cluster_text_for_row(row: dict[str, Any]) -> str:
    context = str(row["context_window_text"]).strip()
    sentence = str(row["sentence_text"]).strip()
    candidate = str(row["candidate_text"]).strip().lower()
    chosen = context
    if len(_content_tokens(chosen)) < 3:
        chosen = sentence
    if chosen.strip().lower() == candidate:
        chosen = sentence
    if candidate and candidate not in chosen.lower():
        chosen = sentence if candidate in sentence.lower() else context
    return re.sub(r"\s{2,}", " ", chosen).strip()


def _build_repaired_residual_rows(
    dataset_csv: Path,
    core_vocab: Path,
    context_tokens: int,
) -> tuple[
    list[dict[str, Any]],
    tuple[list[str], list[str], np.ndarray],
    tuple[list[str], list[str], np.ndarray],
    Any,
]:
    reviews = routing._load_reviews(dataset_csv)
    candidates = repair._extract_old_candidates_with_context(reviews, context_tokens=context_tokens)

    general_aspects = routing._load_aspects(core_vocab)
    domain_paths = {
        "physical_goods": ROOT / "src/vocabulary/domain/physical_goods.yaml",
        "consumables": ROOT / "src/vocabulary/domain/consumables.yaml",
        "hospitality": ROOT / "src/vocabulary/domain/hospitality.yaml",
        "services": ROOT / "src/vocabulary/domain/services.yaml",
    }
    domain_aspects = {category: routing._load_aspects(path) for category, path in domain_paths.items()}
    exact_maps = repair._build_exact_anchor_maps(general_aspects, domain_aspects)

    model = routing._load_encoder()
    embedding_cache: dict[str, np.ndarray] = {}
    routing._encode_texts_cached(model, list({candidate.candidate_lemma for candidate in candidates}), embedding_cache)

    general_ids, general_names, general_matrix = routing._build_anchor_bank(model, general_aspects, embedding_cache)
    domain_banks = {
        category: routing._build_anchor_bank(model, aspects, embedding_cache)
        for category, aspects in domain_aspects.items()
    }

    all_domain_ids: list[str] = []
    all_domain_names: list[str] = []
    all_domain_vectors: list[np.ndarray] = []
    for category, (domain_ids, domain_names, domain_matrix) in domain_banks.items():
        for idx, anchor_id in enumerate(domain_ids):
            all_domain_ids.append(f"{category}::{anchor_id}")
            all_domain_names.append(f"{category}::{domain_names[idx]}")
            all_domain_vectors.append(domain_matrix[idx])
    all_domain_matrix = (
        np.stack(all_domain_vectors, axis=0).astype(np.float32)
        if all_domain_vectors
        else np.zeros((0, 1), dtype=np.float32)
    )

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
    residual_counter: Counter[str] = Counter()
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
        decision = repair._route_repair_candidate(
            candidate=candidate,
            best_general=best_general,
            best_domain=best_domain,
            thresholds=thresholds,
            noise_reason=noise_reason,
            exact_maps=exact_maps,
        )
        if decision.route != "residual":
            continue
        cluster_text = _cluster_text_for_row(
            {
                "context_window_text": candidate.context_window_text,
                "sentence_text": candidate.sentence_text,
                "candidate_text": candidate.candidate_text,
            }
        )
        content_tokens = _content_tokens(cluster_text)
        if len(content_tokens) < 3:
            continue
        if cluster_text.strip().lower() == candidate.candidate_text.strip().lower():
            continue
        if candidate.candidate_text.strip().lower() not in cluster_text.lower():
            continue
        row = {
            "review_id": candidate.review_id,
            "category": candidate.category,
            "candidate_text": candidate.candidate_text,
            "candidate_lemma": candidate.candidate_lemma,
            "cluster_text": cluster_text,
            "context_content_terms": " ".join(content_tokens),
            "best_general_anchor": best_general.best_anchor_name,
            "best_general_score": round(best_general.best_score, 4),
            "best_domain_anchor": best_domain.best_anchor_name,
            "best_domain_score": round(best_domain.best_score, 4),
        }
        rows.append(row)
        residual_counter[candidate.candidate_lemma] += 1

    residual_clean = [
        row
        for row in rows
        if residual_counter[str(row["candidate_lemma"])] >= 2
    ]
    return (
        residual_clean,
        (general_ids, general_names, general_matrix),
        (all_domain_ids, all_domain_names, all_domain_matrix),
        model,
    )


def _dominant_categories(rows: list[dict[str, Any]], limit: int = 3) -> tuple[str, float, int]:
    counts = Counter(str(row["category"]) for row in rows)
    total = sum(counts.values())
    if total == 0:
        return "", 0.0, 0
    top = counts.most_common(limit)
    dominant_name, dominant_count = top[0]
    return ", ".join(f"{name}:{count}" for name, count in top), dominant_count / total, len(counts)


def _top_candidate_lemmas(rows: list[dict[str, Any]], limit: int = 20) -> str:
    counts = Counter(str(row["candidate_lemma"]) for row in rows)
    return ", ".join(lemma for lemma, _ in counts.most_common(limit))


def _top_context_terms(rows: list[dict[str, Any]], limit: int = 20) -> tuple[str, float]:
    counts: Counter[str] = Counter()
    generic = 0
    total = 0
    for row in rows:
        terms = str(row["context_content_terms"]).split()
        counts.update(terms)
        total += len(terms)
        generic += sum(1 for term in terms if term in GENERIC_CONTEXT_LEMMAS)
    return ", ".join(term for term, _ in counts.most_common(limit)), (generic / total if total else 0.0)


def _typical_contexts(rows: list[dict[str, Any]], sims: np.ndarray, limit: int = 15) -> list[str]:
    grouped: dict[str, dict[str, Any]] = {}
    for row, sim in zip(rows, sims.tolist(), strict=True):
        text = str(row["cluster_text"])
        info = grouped.setdefault(text, {"count": 0, "best_sim": -1.0})
        info["count"] += 1
        info["best_sim"] = max(float(info["best_sim"]), float(sim))
    ranked = sorted(grouped.items(), key=lambda item: (-int(item[1]["count"]), -float(item[1]["best_sim"]), item[0]))
    return [text for text, _ in ranked[:limit]]


def _label_cluster(
    size: int,
    dominant_share: float,
    n_categories: int,
    generic_ratio: float,
    nearest_general_score: float,
    nearest_domain_score: float,
    top_candidate_lemmas: str,
    top_context_terms: str,
) -> tuple[str, str, bool, bool]:
    duplicate_general = nearest_general_score >= 0.82
    duplicate_domain = nearest_domain_score >= 0.82
    if duplicate_general or duplicate_domain:
        return (
            "duplicate_existing_anchor",
            "context centroid is close to an existing anchor",
            duplicate_general,
            duplicate_domain,
        )
    lemma_set = {lemma.strip() for lemma in top_candidate_lemmas.split(",") if lemma.strip()}
    term_set = {term.strip() for term in top_context_terms.split(",") if term.strip()}
    if generic_ratio >= 0.35:
        return "noise_cluster", "generic context terms dominate", duplicate_general, duplicate_domain
    if len(lemma_set) <= 2 and size >= 20:
        return "noise_cluster", "cluster is mostly the same leftover object word", duplicate_general, duplicate_domain
    if dominant_share < 0.55 and n_categories >= 3:
        return "too_mixed", "cluster spans too many categories", duplicate_general, duplicate_domain
    if size >= 10 and dominant_share >= 0.60 and len(term_set) >= 5:
        return "useful_new_aspect", "context snippets form a coherent residual theme", duplicate_general, duplicate_domain
    return "unclear", "cluster needs manual inspection", duplicate_general, duplicate_domain


def _run_hdbscan(vectors: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(vectors)


def _build_records(
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    labels: np.ndarray,
    general_bank: tuple[list[str], list[str], np.ndarray],
    domain_bank: tuple[list[str], list[str], np.ndarray],
) -> list[ClusterRecord]:
    valid_labels = [int(label) for label in labels.tolist() if int(label) != -1]
    label_sizes = Counter(valid_labels)
    ranked_labels = [label for label, _ in sorted(label_sizes.items(), key=lambda item: (-item[1], item[0]))]
    label_to_cluster_id = {label: idx for idx, label in enumerate(ranked_labels)}
    general_ids, general_names, general_matrix = general_bank
    domain_ids, domain_names, domain_matrix = domain_bank
    records: list[ClusterRecord] = []

    for original_label in ranked_labels:
        member_indices = np.where(labels == original_label)[0]
        member_rows = [rows[int(idx)] for idx in member_indices.tolist()]
        cluster_vectors = vectors[member_indices]
        centroid = _l2_normalize(cluster_vectors.mean(axis=0, keepdims=True))[0]
        sims = cluster_vectors @ centroid
        medoid_idx = int(np.argmax(sims))
        medoid_text = str(member_rows[medoid_idx]["cluster_text"])
        top_candidate_lemmas = _top_candidate_lemmas(member_rows)
        top_context_terms, generic_ratio = _top_context_terms(member_rows)
        dominant_categories, dominant_share, n_categories = _dominant_categories(member_rows)
        nearest_general = routing._compute_layer_score(
            candidate_vec=centroid,
            anchor_ids=general_ids,
            anchor_names=general_names,
            anchor_matrix=general_matrix,
        )
        nearest_domain = routing._compute_layer_score(
            candidate_vec=centroid,
            anchor_ids=domain_ids,
            anchor_names=domain_names,
            anchor_matrix=domain_matrix,
        )
        label, comment, is_dup_general, is_dup_domain = _label_cluster(
            size=len(member_rows),
            dominant_share=dominant_share,
            n_categories=n_categories,
            generic_ratio=generic_ratio,
            nearest_general_score=float(nearest_general.best_score),
            nearest_domain_score=float(nearest_domain.best_score),
            top_candidate_lemmas=top_candidate_lemmas,
            top_context_terms=top_context_terms,
        )
        records.append(
            ClusterRecord(
                cluster_id=label_to_cluster_id[original_label],
                size=len(member_rows),
                medoid_text=medoid_text,
                top_candidate_lemmas=top_candidate_lemmas,
                top_context_terms=top_context_terms,
                nearest_general_anchor=nearest_general.best_anchor_name,
                nearest_domain_anchor=nearest_domain.best_anchor_name,
                nearest_general_score=round(float(nearest_general.best_score), 4),
                nearest_domain_score=round(float(nearest_domain.best_score), 4),
                dominant_categories=dominant_categories,
                heuristic_label=label,
                heuristic_comment=comment,
                typical_context_snippets=_typical_contexts(member_rows, sims=sims, limit=15),
                is_duplicate_general=is_dup_general,
                is_duplicate_domain=is_dup_domain,
            )
        )
    return records


def _render_top_clusters(records: list[ClusterRecord], limit: int) -> str:
    lines = ["# Contextual HDBSCAN Top Clusters", ""]
    for record in records[:limit]:
        lines.append(f"## Cluster {record.cluster_id}")
        lines.append(f"- size: {record.size}")
        lines.append(f"- medoid: {record.medoid_text}")
        lines.append(f"- dominant_categories: {record.dominant_categories}")
        lines.append(f"- top_candidate_lemmas: {record.top_candidate_lemmas}")
        lines.append(f"- top_context_terms: {record.top_context_terms}")
        lines.append(f"- nearest_general_anchor: {record.nearest_general_anchor} ({record.nearest_general_score:.4f})")
        lines.append(f"- nearest_domain_anchor: {record.nearest_domain_anchor} ({record.nearest_domain_score:.4f})")
        lines.append(f"- heuristic_label: {record.heuristic_label}")
        lines.append(f"- comment: {record.heuristic_comment}")
        lines.append("- typical_context_snippets:")
        for snippet in record.typical_context_snippets[:15]:
            lines.append(f"  - {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 step 5 contextual residual HDBSCAN probe")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase4_step5_contextual_residual_hdbscan_probe")
    parser.add_argument("--context-tokens", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--fallback-min-cluster-size", type=int, default=8)
    parser.add_argument("--fallback-min-samples", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    residual_rows, general_bank, domain_bank, model = _build_repaired_residual_rows(
        dataset_csv=ROOT / args.dataset_csv,
        core_vocab=ROOT / args.core_vocab,
        context_tokens=args.context_tokens,
    )
    if not residual_rows:
        raise RuntimeError("contextual repaired residual is empty")

    context_texts = [str(row["cluster_text"]) for row in residual_rows]
    embeddings = model.encode(context_texts, show_progress_bar=False, convert_to_numpy=True, batch_size=128)
    vectors = _l2_normalize(np.asarray(embeddings, dtype=np.float32))

    labels = _run_hdbscan(vectors, args.min_cluster_size, args.min_samples)
    valid_labels = [int(label) for label in labels.tolist() if int(label) != -1]
    fallback_used = False
    if not valid_labels:
        labels = _run_hdbscan(vectors, args.fallback_min_cluster_size, args.fallback_min_samples)
        valid_labels = [int(label) for label in labels.tolist() if int(label) != -1]
        fallback_used = True

    records = _build_records(residual_rows, vectors, labels, general_bank, domain_bank)
    clustered_share = len(valid_labels) / len(residual_rows) if residual_rows else 0.0
    top_records = records[: args.top_k]
    top_label_counts = Counter(record.heuristic_label for record in top_records)
    duplicate_noise_count = top_label_counts["duplicate_existing_anchor"] + top_label_counts["noise_cluster"]
    keep_branch = top_label_counts["useful_new_aspect"] >= 5 and duplicate_noise_count < len(top_records) / 2
    decision = "keep_contextual_hdbscan_branch" if keep_branch else "kill_contextual_hdbscan_branch"

    out_dir = ROOT / args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {
            "cluster_id": record.cluster_id,
            "size": record.size,
            "medoid_text": record.medoid_text,
            "top_candidate_lemmas": record.top_candidate_lemmas,
            "top_context_terms": record.top_context_terms,
            "nearest_general_anchor": record.nearest_general_anchor,
            "nearest_domain_anchor": record.nearest_domain_anchor,
            "nearest_general_score": record.nearest_general_score,
            "nearest_domain_score": record.nearest_domain_score,
            "dominant_categories": record.dominant_categories,
            "heuristic_label": record.heuristic_label,
            "heuristic_comment": record.heuristic_comment,
        }
        for record in records
    ]
    duplicate_rows = [
        {
            "cluster_id": record.cluster_id,
            "nearest_general_anchor": record.nearest_general_anchor,
            "nearest_domain_anchor": record.nearest_domain_anchor,
            "is_duplicate_general": str(record.is_duplicate_general).lower(),
            "is_duplicate_domain": str(record.is_duplicate_domain).lower(),
            "comment": record.heuristic_comment,
        }
        for record in records
    ]
    manual_label_rows = [
        {
            "cluster_id": record.cluster_id,
            "label": record.heuristic_label,
            "comment": record.heuristic_comment,
        }
        for record in top_records
    ]

    pd.DataFrame(summary_rows).to_csv(out_dir / "contextual_hdbscan_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(duplicate_rows).to_csv(out_dir / "contextual_duplicate_check.csv", index=False, encoding="utf-8")
    pd.DataFrame(manual_label_rows).to_csv(out_dir / "contextual_manual_labels.csv", index=False, encoding="utf-8")
    (out_dir / "contextual_top_clusters.md").write_text(
        _render_top_clusters(records, limit=args.top_k),
        encoding="utf-8",
    )

    min_cluster_size = args.fallback_min_cluster_size if fallback_used else args.min_cluster_size
    min_samples = args.fallback_min_samples if fallback_used else args.min_samples
    summary_md = f"""# phase4_step5_contextual_residual_hdbscan_probe

## Setup
- input: repair_v1 residual_clean only
- embedding text: `cluster_text` = candidate context window, not `candidate_lemma`
- residual rows after context filter: {len(residual_rows)}
- HDBSCAN min_cluster_size: {min_cluster_size}
- HDBSCAN min_samples: {min_samples}
- fallback used: {str(fallback_used).lower()}
- clustered share: {clustered_share:.4f}

## Clusters
- total clusters: {len(records)}
- top-{args.top_k} useful_new_aspect: {top_label_counts['useful_new_aspect']}
- top-{args.top_k} duplicate_existing_anchor: {top_label_counts['duplicate_existing_anchor']}
- top-{args.top_k} too_mixed: {top_label_counts['too_mixed']}
- top-{args.top_k} noise_cluster: {top_label_counts['noise_cluster']}
- top-{args.top_k} unclear: {top_label_counts['unclear']}

## Verdict
- contextual residual after HDBSCAN looks like: {"a source of new aspects" if keep_branch else "mostly duplicates / mixed signal / noise"}
- {decision}

## Interpretation
- largest cluster size: {records[0].size if records else 0} / {len(residual_rows)}
- context embeddings avoid pure one-word clustering, but can still collapse into broad mixed review-window clusters.
- useful signal exists only if top clusters split into several coherent themes, not one giant mixed bucket.
"""
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    run_summary = {
        "out_dir": str(out_dir),
        "input_rows": len(residual_rows),
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "fallback_used": fallback_used,
        "n_clusters": len(records),
        "clustered_share": round(clustered_share, 4),
        "top_label_counts": dict(top_label_counts),
        "decision": decision,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
