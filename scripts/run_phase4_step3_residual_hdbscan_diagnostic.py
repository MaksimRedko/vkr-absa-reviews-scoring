from __future__ import annotations

import argparse
import json
import math
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


STOP_LEMMAS = {
    "быть",
    "без",
    "более",
    "весь",
    "вот",
    "все",
    "всё",
    "где",
    "для",
    "его",
    "если",
    "еще",
    "ещё",
    "или",
    "как",
    "который",
    "ли",
    "меня",
    "мой",
    "над",
    "нам",
    "нет",
    "них",
    "она",
    "они",
    "оно",
    "под",
    "при",
    "про",
    "самый",
    "свой",
    "так",
    "такой",
    "только",
    "тут",
    "уже",
    "хоть",
    "чем",
    "что",
    "это",
    "этот",
}

GENERIC_CLUSTER_LEMMAS = routing.GENERIC_EMOTION_LEMMAS | routing.GENERIC_OBJECT_LEMMAS | {
    "вещь",
    "вообще",
    "заказ",
    "магазин",
    "место",
    "недостаток",
    "отзыв",
    "плюс",
    "покупка",
    "проблема",
    "продукт",
    "просто",
    "товар",
    "услуга",
    "штука",
}


@dataclass(frozen=True, slots=True)
class ClusterRecord:
    cluster_id: int
    size: int
    medoid_text: str
    top_terms: str
    nearest_general_anchor: str
    nearest_domain_anchor: str
    nearest_general_score: float
    nearest_domain_score: float
    dominant_categories: str
    heuristic_label: str
    heuristic_comment: str
    typical_candidates: list[str]
    is_duplicate_general: bool
    is_duplicate_domain: bool


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _tokenize_candidate(candidate_lemma: str) -> list[str]:
    return [
        token
        for token in str(candidate_lemma).split()
        if token and len(token) >= 3 and token not in STOP_LEMMAS
    ]


def _dominant_categories(rows: list[dict[str, Any]], limit: int = 3) -> tuple[str, float, int]:
    counts = Counter(str(row["category"]) for row in rows)
    total = sum(counts.values())
    if total == 0:
        return "", 0.0, 0
    top = counts.most_common(limit)
    dominant_category, dominant_count = top[0]
    rendered = ", ".join(f"{name}:{count}" for name, count in top)
    return rendered, dominant_count / total, len(counts)


def _top_terms(rows: list[dict[str, Any]], limit: int = 20) -> tuple[str, float]:
    counter: Counter[str] = Counter()
    generic_count = 0
    total_count = 0
    for row in rows:
        tokens = _tokenize_candidate(str(row["candidate_lemma"]))
        counter.update(tokens)
        total_count += len(tokens)
        generic_count += sum(1 for token in tokens if token in GENERIC_CLUSTER_LEMMAS)
    top_terms = ", ".join(term for term, _ in counter.most_common(limit))
    generic_ratio = (generic_count / total_count) if total_count else 0.0
    return top_terms, generic_ratio


def _typical_candidates(rows: list[dict[str, Any]], sims: np.ndarray, limit: int = 20) -> list[str]:
    grouped: dict[str, dict[str, Any]] = {}
    for row, sim in zip(rows, sims.tolist(), strict=True):
        lemma = str(row["candidate_lemma"])
        bucket = grouped.setdefault(
            lemma,
            {
                "count": 0,
                "best_sim": -1.0,
                "texts": Counter(),
            },
        )
        bucket["count"] += 1
        bucket["best_sim"] = max(bucket["best_sim"], float(sim))
        bucket["texts"][str(row["candidate_text"])] += 1
    ranked = sorted(
        grouped.items(),
        key=lambda item: (-int(item[1]["count"]), -float(item[1]["best_sim"]), item[0]),
    )
    out: list[str] = []
    for _, info in ranked[:limit]:
        best_text, _ = info["texts"].most_common(1)[0]
        out.append(str(best_text))
    return out


def _cluster_label_comment(
    size: int,
    dominant_share: float,
    n_categories: int,
    generic_ratio: float,
    nearest_general_score: float,
    nearest_domain_score: float,
    top_terms: str,
) -> tuple[str, str, bool, bool]:
    duplicate_general = nearest_general_score >= 0.86
    duplicate_domain = nearest_domain_score >= 0.86
    if duplicate_general or duplicate_domain:
        comment = "cluster centroid is too close to an existing anchor"
        return "duplicate_existing_anchor", comment, duplicate_general, duplicate_domain
    if generic_ratio >= 0.45:
        comment = "generic/object-heavy terms dominate the cluster"
        return "noise_cluster", comment, duplicate_general, duplicate_domain
    if dominant_share < 0.55 and n_categories >= 3:
        comment = "members are spread across categories; theme looks mixed"
        return "too_mixed", comment, duplicate_general, duplicate_domain
    if size >= 12 and generic_ratio <= 0.20 and dominant_share >= 0.60 and top_terms:
        comment = "coherent residual theme with no strong anchor duplicate"
        return "useful_new_aspect", comment, duplicate_general, duplicate_domain
    comment = "cluster is coherent enough to inspect, but signal is still ambiguous"
    return "unclear", comment, duplicate_general, duplicate_domain


def _build_residual_clean_rows(
    dataset_csv: Path,
    core_vocab: Path,
    mode: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, np.ndarray],
    tuple[list[str], list[str], np.ndarray],
    tuple[list[str], list[str], np.ndarray],
]:
    reviews = routing._load_reviews(dataset_csv)
    candidates = routing._extract_old_candidates(reviews)

    general_aspects = routing._load_aspects(core_vocab)
    domain_paths = {
        "physical_goods": ROOT / "src/vocabulary/domain/physical_goods.yaml",
        "consumables": ROOT / "src/vocabulary/domain/consumables.yaml",
        "hospitality": ROOT / "src/vocabulary/domain/hospitality.yaml",
        "services": ROOT / "src/vocabulary/domain/services.yaml",
    }
    domain_aspects = {category: routing._load_aspects(path) for category, path in domain_paths.items()}

    model = routing._load_encoder()
    embedding_cache: dict[str, np.ndarray] = {}
    routing._encode_texts_cached(model, list({row.candidate_lemma for row in candidates}), embedding_cache)

    general_ids, general_names, general_matrix = routing._build_anchor_bank(model, general_aspects, embedding_cache)
    domain_banks: dict[str, tuple[list[str], list[str], np.ndarray]] = {}
    for category, aspects in domain_aspects.items():
        domain_banks[category] = routing._build_anchor_bank(model, aspects, embedding_cache)

    combined_domain_ids: list[str] = []
    combined_domain_names: list[str] = []
    combined_domain_vectors: list[np.ndarray] = []
    for category, (domain_ids, domain_names, domain_matrix) in domain_banks.items():
        for idx, anchor_id in enumerate(domain_ids):
            combined_domain_ids.append(f"{category}::{anchor_id}")
            combined_domain_names.append(f"{category}::{domain_names[idx]}")
            combined_domain_vectors.append(domain_matrix[idx])
    combined_domain_matrix = (
        np.stack(combined_domain_vectors, axis=0).astype(np.float32)
        if combined_domain_vectors
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

    residual_rows: list[dict[str, Any]] = []
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
        decision = routing._route_candidate(
            category=candidate.category,
            best_general=best_general,
            best_domain=best_domain,
            thresholds=thresholds,
            noise_reason=noise_reason,
            mode=mode,
        )
        if decision.route != "residual":
            continue
        row = {
            "review_id": candidate.review_id,
            "category": candidate.category,
            "candidate_text": candidate.candidate_text,
            "candidate_lemma": candidate.candidate_lemma,
            "best_general_anchor": best_general.best_anchor_name,
            "best_general_score": round(best_general.best_score, 4),
            "best_domain_anchor": best_domain.best_anchor_name,
            "best_domain_score": round(best_domain.best_score, 4),
        }
        residual_rows.append(row)
        residual_counter[candidate.candidate_lemma] += 1

    residual_clean_rows = [
        row
        for row in residual_rows
        if residual_counter[str(row["candidate_lemma"])] >= 2
    ]
    return (
        residual_clean_rows,
        embedding_cache,
        (general_ids, general_names, general_matrix),
        (combined_domain_ids, combined_domain_names, combined_domain_matrix),
    )


def _render_top_clusters_md(records: list[ClusterRecord], limit: int = 20) -> str:
    lines = ["# HDBSCAN Top Clusters", ""]
    for record in records[:limit]:
        lines.append(f"## Cluster {record.cluster_id}")
        lines.append(f"- size: {record.size}")
        lines.append(f"- medoid: {record.medoid_text}")
        lines.append(f"- dominant_categories: {record.dominant_categories}")
        lines.append(f"- top_terms: {record.top_terms}")
        lines.append(
            f"- nearest_general_anchor: {record.nearest_general_anchor} ({record.nearest_general_score:.4f})"
        )
        lines.append(
            f"- nearest_domain_anchor: {record.nearest_domain_anchor} ({record.nearest_domain_score:.4f})"
        )
        lines.append(f"- heuristic_label: {record.heuristic_label}")
        lines.append(f"- comment: {record.heuristic_comment}")
        lines.append("- typical_candidates:")
        for candidate in record.typical_candidates[:20]:
            lines.append(f"  - {candidate}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 step 3 residual-only HDBSCAN diagnostic")
    parser.add_argument("--dataset-csv", default="data/dataset_final.csv")
    parser.add_argument("--core-vocab", default="src/vocabulary/universal_aspects_v1.yaml")
    parser.add_argument("--out-dir", default=".opencode/artifacts/phase4_step3_residual_hdbscan")
    parser.add_argument("--mode", choices=["current", "domain_priority"], default="domain_priority")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cluster-size", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    residual_clean_rows, embedding_cache, general_bank, all_domain_bank = _build_residual_clean_rows(
        dataset_csv=ROOT / args.dataset_csv,
        core_vocab=ROOT / args.core_vocab,
        mode=args.mode,
    )
    if not residual_clean_rows:
        raise RuntimeError("residual_clean is empty; cannot run HDBSCAN diagnostic")

    vectors = np.stack(
        [embedding_cache[str(row["candidate_lemma"])] for row in residual_clean_rows],
        axis=0,
    ).astype(np.float32)
    n_rows = len(residual_clean_rows)
    auto_min_cluster_size = max(5, min(15, int(round(math.sqrt(n_rows)))))
    min_cluster_size = args.min_cluster_size or auto_min_cluster_size
    min_samples = args.min_samples or max(3, min_cluster_size // 2)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(vectors)

    valid_labels = [int(label) for label in labels.tolist() if int(label) != -1]
    label_sizes = Counter(valid_labels)
    ranked_labels = [label for label, _ in sorted(label_sizes.items(), key=lambda item: (-item[1], item[0]))]
    label_to_cluster_id = {label: idx for idx, label in enumerate(ranked_labels)}

    general_ids, general_names, general_matrix = general_bank
    domain_ids, domain_names, domain_matrix = all_domain_bank

    records: list[ClusterRecord] = []
    summary_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    manual_label_rows: list[dict[str, Any]] = []

    for original_label in ranked_labels:
        member_indices = np.where(labels == original_label)[0]
        member_rows = [residual_clean_rows[int(idx)] for idx in member_indices.tolist()]
        cluster_vectors = vectors[member_indices]
        centroid = _l2_normalize(cluster_vectors.mean(axis=0, keepdims=True))[0]
        sims = cluster_vectors @ centroid
        medoid_idx = int(np.argmax(sims))
        medoid_text = str(member_rows[medoid_idx]["candidate_text"])
        top_terms, generic_ratio = _top_terms(member_rows)
        dominant_categories, dominant_share, n_categories = _dominant_categories(member_rows)
        typical_candidates = _typical_candidates(member_rows, sims=sims, limit=20)

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

        heuristic_label, heuristic_comment, duplicate_general, duplicate_domain = _cluster_label_comment(
            size=len(member_rows),
            dominant_share=dominant_share,
            n_categories=n_categories,
            generic_ratio=generic_ratio,
            nearest_general_score=float(nearest_general.best_score),
            nearest_domain_score=float(nearest_domain.best_score),
            top_terms=top_terms,
        )

        cluster_id = label_to_cluster_id[original_label]
        record = ClusterRecord(
            cluster_id=cluster_id,
            size=len(member_rows),
            medoid_text=medoid_text,
            top_terms=top_terms,
            nearest_general_anchor=nearest_general.best_anchor_name,
            nearest_domain_anchor=nearest_domain.best_anchor_name,
            nearest_general_score=round(float(nearest_general.best_score), 4),
            nearest_domain_score=round(float(nearest_domain.best_score), 4),
            dominant_categories=dominant_categories,
            heuristic_label=heuristic_label,
            heuristic_comment=heuristic_comment,
            typical_candidates=typical_candidates,
            is_duplicate_general=duplicate_general,
            is_duplicate_domain=duplicate_domain,
        )
        records.append(record)

        summary_rows.append(
            {
                "cluster_id": record.cluster_id,
                "size": record.size,
                "medoid_text": record.medoid_text,
                "top_terms": record.top_terms,
                "nearest_general_anchor": record.nearest_general_anchor,
                "nearest_domain_anchor": record.nearest_domain_anchor,
                "nearest_general_score": record.nearest_general_score,
                "nearest_domain_score": record.nearest_domain_score,
            }
        )
        duplicate_rows.append(
            {
                "cluster_id": record.cluster_id,
                "nearest_general_anchor": record.nearest_general_anchor,
                "nearest_domain_anchor": record.nearest_domain_anchor,
                "is_duplicate_general": str(record.is_duplicate_general).lower(),
                "is_duplicate_domain": str(record.is_duplicate_domain).lower(),
                "comment": record.heuristic_comment,
            }
        )
        if record.cluster_id < args.top_k:
            manual_label_rows.append(
                {
                    "cluster_id": record.cluster_id,
                    "label": record.heuristic_label,
                    "comment": record.heuristic_comment,
                }
            )

    top_label_counts = Counter(str(row["label"]) for row in manual_label_rows)
    keep_branch = (
        top_label_counts["useful_new_aspect"] >= 5
        and (top_label_counts["duplicate_existing_anchor"] + top_label_counts["noise_cluster"]) < args.top_k / 2
    )
    decision = "keep_hdbscan_branch" if keep_branch else "kill_hdbscan_branch"

    clustered_share = (len(valid_labels) / n_rows) if n_rows else 0.0
    out_dir = ROOT / args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summary_rows).to_csv(out_dir / "hdbscan_cluster_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(duplicate_rows).to_csv(out_dir / "hdbscan_duplicate_check.csv", index=False, encoding="utf-8")
    pd.DataFrame(manual_label_rows).to_csv(out_dir / "hdbscan_manual_labels.csv", index=False, encoding="utf-8")
    (out_dir / "hdbscan_top_clusters.md").write_text(_render_top_clusters_md(records, limit=args.top_k), encoding="utf-8")

    summary_md = f"""# phase4_step3_residual_only_hdbscan

## Setup
- routing mode: `{args.mode}`
- residual_clean rows: {n_rows}
- HDBSCAN min_cluster_size: {min_cluster_size}
- HDBSCAN min_samples: {min_samples}
- clustered share: {clustered_share:.4f}

## Clusters
- total clusters: {len(records)}
- top-{args.top_k} useful_new_aspect: {top_label_counts['useful_new_aspect']}
- top-{args.top_k} duplicate_existing_anchor: {top_label_counts['duplicate_existing_anchor']}
- top-{args.top_k} too_mixed: {top_label_counts['too_mixed']}
- top-{args.top_k} noise_cluster: {top_label_counts['noise_cluster']}
- top-{args.top_k} unclear: {top_label_counts['unclear']}

## Verdict
- residual after HDBSCAN looks like: {"a source of new aspects" if keep_branch else "mostly duplicates / mixed signal / noise"}
- {decision}
"""
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    run_summary = {
        "out_dir": str(out_dir),
        "routing_mode": args.mode,
        "residual_clean_rows": n_rows,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_clusters": len(records),
        "clustered_share": round(clustered_share, 4),
        "top_label_counts": dict(top_label_counts),
        "decision": decision,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
