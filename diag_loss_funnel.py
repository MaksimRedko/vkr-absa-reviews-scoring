"""
Loss Funnel Analysis: на каком шаге pipeline теряются mentions?

Для каждого (review_id, true_aspect) из разметки проверяем:
  Stage A: Есть ли в review хотя бы 1 candidate (после POS extraction)?
  Stage B: Прошёл ли хоть один relevant candidate через KeyBERT (cosine ≥ 0.45)?
  Stage C: Прошёл ли через MMR top-5?
  Stage D: Попал ли в кластер, который маппится на true_aspect?
  Stage E: Дошла ли пара (review_id, aspect) до NLI scoring?

Результат: воронка потерь по стадиям.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")

def _parse_labels(val) -> Optional[Dict[str, float]]:
    if pd.isna(val) or str(val).strip() in ("", "nan", "{}"):
        return None
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, dict) and parsed:
            return {k: float(v) for k, v in parsed.items()}
    except (ValueError, SyntaxError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Воронка потерь: где теряются true aspects")
    parser.add_argument(
        "--csv-path",
        default="parser/reviews_batches/merged_checked_reviews.csv",
        help="Разметка (true_labels + id)",
    )
    parser.add_argument(
        "--json-path",
        default=None,
        help="Если задан и файл есть — отзывы из JSON; иначе из --csv-path",
    )
    args = parser.parse_args()

    # ── Загрузка ────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    from configs.configs import config
    from src.discovery.candidates import CandidateExtractor
    from src.discovery.scorer import KeyBERTScorer
    from src.discovery.clusterer import AspectClusterer, MACRO_ANCHORS
    from src.schemas.models import ReviewInput

    df = pd.read_csv(args.csv_path, dtype={"id": str})
    df["true_labels_parsed"] = df["true_labels"].apply(_parse_labels)

    reviews_by_nm: Dict[int, List[ReviewInput]] = defaultdict(list)
    if args.json_path and os.path.isfile(args.json_path):
        with open(args.json_path, "r", encoding="utf-8") as f:
            all_reviews_raw = json.load(f)
        for r in all_reviews_raw:
            try:
                ri = ReviewInput(**r)
                if ri.clean_text:
                    reviews_by_nm[ri.nm_id].append(ri)
            except Exception:
                continue
    else:
        from eval_pipeline import load_pipeline_reviews_from_csv

        nm_ids = sorted(df["nm_id"].unique().tolist())
        raw = load_pipeline_reviews_from_csv(args.csv_path, [int(x) for x in nm_ids])
        for r in raw:
            try:
                ri = ReviewInput(**r)
                if ri.clean_text:
                    reviews_by_nm[ri.nm_id].append(ri)
            except Exception:
                continue
        if args.json_path:
            print(
                f"[diag_loss_funnel] json_path={args.json_path!r} не найден — отзывы из CSV"
            )

    encoder = SentenceTransformer(config.models.encoder_path)
    extractor = CandidateExtractor()
    scorer = KeyBERTScorer(model=encoder)
    clusterer = AspectClusterer(model=encoder)

    # Encode anchor embeddings для проверки relevance
    anchor_embeddings: Dict[str, np.ndarray] = {}
    for name, words in MACRO_ANCHORS.items():
        embs = encoder.encode(words, show_progress_bar=False)
        anchor_embeddings[name] = np.mean(embs, axis=0)

    # Маппинг true_aspect → ближайший anchor (для сравнения)
    # Некоторые true aspects = anchor names, некоторые нет
    true_to_anchor: Dict[str, str] = {}
    all_true_aspects = set()
    for labels in df["true_labels_parsed"].dropna():
        all_true_aspects.update(labels.keys())

    anchor_names = list(anchor_embeddings.keys())
    anchor_matrix = np.stack([anchor_embeddings[n] for n in anchor_names])

    for ta in all_true_aspects:
        if ta in anchor_embeddings:
            true_to_anchor[ta] = ta
        else:
            # Encode true aspect name, find nearest anchor
            ta_emb = encoder.encode([ta], show_progress_bar=False)[0]
            sims = cosine_similarity(ta_emb.reshape(1, -1), anchor_matrix)[0]
            best_idx = int(np.argmax(sims))
            true_to_anchor[ta] = anchor_names[best_idx]

    print("=" * 70)
    print("TRUE ASPECT → ANCHOR MAPPING")
    print("=" * 70)
    for ta, anchor in sorted(true_to_anchor.items()):
        tag = "identity" if ta == anchor else f"mapped"
        print(f"  {ta:25s} → {anchor:20s} ({tag})")

    # ── Основной цикл по товарам ────────────────────────────────────────
    nm_ids = sorted(df["nm_id"].unique())

    # Глобальные счётчики
    global_counts = Counter()

    for nm_id in nm_ids:
        reviews = reviews_by_nm.get(nm_id, [])
        if not reviews:
            continue

        review_by_id = {r.id: r for r in reviews}
        grp = df[df["nm_id"] == nm_id]
        texts = [r.clean_text for r in reviews]

        # ── Stage A: CandidateExtractor ──────────────────────────────────
        # Извлекаем candidates per review
        candidates_per_review: Dict[str, list] = {}
        all_candidates = []
        for review in reviews:
            cands = extractor.extract(review.clean_text)
            candidates_per_review[review.id] = cands
            all_candidates.extend(cands)

        # Encode all candidate spans (unique)
        unique_spans = list({c.span for c in all_candidates})
        if unique_spans:
            span_embs_raw = encoder.encode(unique_spans, show_progress_bar=False)
            span_to_emb = dict(zip(unique_spans, span_embs_raw))
        else:
            span_to_emb = {}

        # ── Stage B+C: KeyBERT + MMR ─────────────────────────────────────
        scored_candidates = scorer.score_and_select(all_candidates)
        scored_spans_set = {c.span for c in scored_candidates}

        # Build sentence_to_review from candidates
        sentence_to_review = {}
        for review in reviews:
            for c in candidates_per_review[review.id]:
                sentence_to_review[c.sentence.strip()] = review.id
                sentence_to_review[c.sentence.lower().strip()] = review.id

        # ── Stage D: Clustering ──────────────────────────────────────────
        aspects = clusterer.cluster(scored_candidates)
        aspect_names = list(aspects.keys())

        # Build span_to_aspect from cluster keywords
        span_to_cluster: Dict[str, str] = {}
        for asp_name, info in aspects.items():
            for kw in info.keywords:
                span_to_cluster[kw] = asp_name

        # For each cluster, find nearest anchor (for mapping to true)
        cluster_to_anchor: Dict[str, str] = {}
        for asp_name, info in aspects.items():
            if asp_name in anchor_embeddings:
                cluster_to_anchor[asp_name] = asp_name
            else:
                centroid = info.centroid_embedding.reshape(1, -1)
                sims = cosine_similarity(centroid, anchor_matrix)[0]
                best_idx = int(np.argmax(sims))
                cluster_to_anchor[asp_name] = anchor_names[best_idx]

        # ── Stage E: NLI pairs (simulated) ───────────────────────────────
        # Build set of (review_id, cluster_name) that would get NLI score
        nli_pairs: set = set()
        for cand in scored_candidates:
            cluster_name = span_to_cluster.get(cand.span)
            if not cluster_name:
                continue
            rid = sentence_to_review.get(
                cand.sentence.strip(),
                sentence_to_review.get(cand.sentence.lower().strip(), "unknown"),
            )
            if rid != "unknown":
                nli_pairs.add((rid, cluster_name))

        # ── Trace each (review_id, true_aspect) ─────────────────────────
        product_counts = Counter()

        for _, row in grp.iterrows():
            true_labels = row["true_labels_parsed"]
            if not true_labels:
                continue

            rid = str(row["id"]).strip()
            review = review_by_id.get(rid)
            if not review:
                continue

            cands = candidates_per_review.get(rid, [])
            target_anchor = None  # will be set per aspect

            for true_asp, true_score in true_labels.items():
                target_anchor = true_to_anchor.get(true_asp, true_asp)
                if target_anchor not in anchor_embeddings:
                    product_counts["no_anchor_for_true"] += 1
                    global_counts["no_anchor_for_true"] += 1
                    continue

                target_emb = anchor_embeddings[target_anchor]
                product_counts["total"] += 1
                global_counts["total"] += 1

                # Stage A: any candidate at all for this review?
                if not cands:
                    product_counts["lost_A_no_candidates"] += 1
                    global_counts["lost_A_no_candidates"] += 1
                    continue

                # Stage B: any candidate semantically relevant to true_aspect?
                # Check: max cosine(candidate_span_emb, target_anchor_emb)
                relevant_found = False
                best_cand_sim = 0.0
                for c in cands:
                    if c.span in span_to_emb:
                        sim = cosine_similarity(
                            span_to_emb[c.span].reshape(1, -1),
                            target_emb.reshape(1, -1)
                        )[0, 0]
                        best_cand_sim = max(best_cand_sim, sim)
                        if sim >= 0.3:
                            relevant_found = True
                            break

                if not relevant_found:
                    product_counts["lost_B_no_relevant_candidate"] += 1
                    global_counts["lost_B_no_relevant_candidate"] += 1
                    continue

                # Stage C: did any relevant candidate survive KeyBERT + MMR?
                survived_scoring = False
                for c in cands:
                    if c.span in scored_spans_set and c.span in span_to_emb:
                        sim = cosine_similarity(
                            span_to_emb[c.span].reshape(1, -1),
                            target_emb.reshape(1, -1)
                        )[0, 0]
                        if sim >= 0.3:
                            survived_scoring = True
                            break

                if not survived_scoring:
                    product_counts["lost_C_keybert_mmr"] += 1
                    global_counts["lost_C_keybert_mmr"] += 1
                    continue

                # Stage D: is the span in a cluster that maps to target_anchor?
                correct_cluster = False
                for c in cands:
                    if c.span in scored_spans_set and c.span in span_to_cluster:
                        cluster_name = span_to_cluster[c.span]
                        mapped_anchor = cluster_to_anchor.get(cluster_name)
                        if mapped_anchor == target_anchor:
                            correct_cluster = True
                            break

                if not correct_cluster:
                    product_counts["lost_D_wrong_cluster"] += 1
                    global_counts["lost_D_wrong_cluster"] += 1
                    continue

                # Stage E: did (review_id, correct_cluster) reach NLI?
                reached_nli = False
                for c in cands:
                    if c.span in scored_spans_set and c.span in span_to_cluster:
                        cluster_name = span_to_cluster[c.span]
                        mapped_anchor = cluster_to_anchor.get(cluster_name)
                        if mapped_anchor == target_anchor:
                            if (rid, cluster_name) in nli_pairs:
                                reached_nli = True
                                break

                if not reached_nli:
                    product_counts["lost_E_no_nli_pair"] += 1
                    global_counts["lost_E_no_nli_pair"] += 1
                    continue

                product_counts["survived"] += 1
                global_counts["survived"] += 1

        # ── Print per-product ────────────────────────────────────────────
        total = product_counts["total"]
        if total == 0:
            continue

        print(f"\n{'='*70}")
        print(f"nm_id={nm_id}  (mentions={total})")
        print(f"{'='*70}")

        stages = [
            ("A. No candidates at all", "lost_A_no_candidates"),
            ("B. No relevant candidate (cos<0.3 to anchor)", "lost_B_no_relevant_candidate"),
            ("C. Lost at KeyBERT/MMR scoring", "lost_C_keybert_mmr"),
            ("D. Wrong cluster (maps to different anchor)", "lost_D_wrong_cluster"),
            ("E. No NLI pair (sentence→review mismatch)", "lost_E_no_nli_pair"),
        ]

        remaining = total
        for label, key in stages:
            lost = product_counts[key]
            pct = lost / total * 100
            remaining -= lost
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {label:50s}  {lost:4d} ({pct:5.1f}%)  |{bar}|")

        survived = product_counts["survived"]
        pct_s = survived / total * 100
        print(f"  {'SURVIVED':50s}  {survived:4d} ({pct_s:5.1f}%)")

        if product_counts.get("no_anchor_for_true", 0):
            print(f"  (skipped {product_counts['no_anchor_for_true']} mentions — no anchor for true aspect)")

    # ── Global summary ───────────────────────────────────────────────────
    total = global_counts["total"]
    survived = global_counts["survived"]

    print(f"\n{'='*70}")
    print(f"GLOBAL LOSS FUNNEL  (total mentions = {total})")
    print(f"{'='*70}")

    stages = [
        ("A. No candidates at all", "lost_A_no_candidates"),
        ("B. No relevant candidate (cos<0.3 to anchor)", "lost_B_no_relevant_candidate"),
        ("C. Lost at KeyBERT/MMR scoring", "lost_C_keybert_mmr"),
        ("D. Wrong cluster (maps to different anchor)", "lost_D_wrong_cluster"),
        ("E. No NLI pair (sentence→review mismatch)", "lost_E_no_nli_pair"),
    ]

    remaining = total
    cumulative_lost = 0
    for label, key in stages:
        lost = global_counts[key]
        pct = lost / total * 100 if total else 0
        cumulative_lost += lost
        remaining_now = total - cumulative_lost
        print(f"  {label:50s}  {lost:4d} ({pct:5.1f}%)  remaining: {remaining_now}")

    pct_s = survived / total * 100 if total else 0
    print(f"\n  SURVIVED → NLI scoring:  {survived}/{total}  ({pct_s:.1f}%)")
    print(f"  LOST total:              {total - survived}/{total}  ({100 - pct_s:.1f}%)")

    # ── Top bottleneck ───────────────────────────────────────────────────
    stage_losses = [(key, global_counts[key]) for _, key in stages]
    stage_losses.sort(key=lambda x: x[1], reverse=True)
    top_key, top_val = stage_losses[0]
    top_pct = top_val / total * 100 if total else 0
    print(f"\n  BIGGEST BOTTLENECK: {top_key} — {top_val} mentions ({top_pct:.1f}%)")


if __name__ == "__main__":
    main()
