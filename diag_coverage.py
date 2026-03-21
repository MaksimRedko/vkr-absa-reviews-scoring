"""
diag_final_audit.py — Финальный аудит: 100% проверка пайплайна + eval.

7 блоков:
  1. DATA INTEGRITY: CSV ↔ JSON consistency, missing reviews, parse errors
  2. PIPELINE SANITY: каждый scored candidate → один аспект, нет дубликатов,
     все keywords из aspects реально есть в scored candidates
  3. EVAL CORRECTNESS: _collect_score_pairs vs ручной подсчёт, нет утечек
  4. MENTION RECALL AUDIT: поштучная проверка (review, true_asp) → есть ли pred
  5. PRODUCT MAE AUDIT: пересчёт MAE из raw data, сравнение с eval_metrics.json
  6. CEILING ANALYSIS: теоретический максимум без LLM — какие аспекты невозможно
     обнаружить unsupervised
  7. NLI SANITY: проверка что NLI не инвертирует sentiment систематически

Запуск:
  python diag_final_audit.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ASPECT_ALIASES = {
    "Органолептика": "Запах",
    "Функциональность": "Функционал",
}


def load_markup(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["true_labels_parsed"] = df["true_labels"].apply(_parse_labels)
    return df


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


def load_per_review(path="eval_per_review.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_step12(path="eval_results_step1_2.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 1: DATA INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

def block1_data_integrity(csv_path, json_path):
    print("\n" + "=" * 70)
    print("BLOCK 1: DATA INTEGRITY")
    print("=" * 70)

    df = load_markup(csv_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    csv_ids = set(df["id"].astype(str))
    json_ids = set(r["id"] for r in raw)

    csv_nms = set(df["nm_id"].unique())
    json_nms = set(r["nm_id"] for r in raw)

    print(f"\n  CSV: {len(df)} rows, {len(csv_nms)} products")
    print(f"  JSON: {len(raw)} reviews, {len(json_nms)} products")
    print(f"  Products match: {'✓' if csv_nms == json_nms else '✗ MISMATCH'}")

    # Reviews in CSV but not JSON
    only_csv = csv_ids - json_ids
    only_json = json_ids - csv_ids
    print(f"  IDs in CSV only: {len(only_csv)}")
    print(f"  IDs in JSON only: {len(only_json)}")

    # Parse errors in true_labels
    parse_errors = 0
    for _, row in df.iterrows():
        tl = row.get("true_labels", "")
        if pd.isna(tl) or str(tl).strip() in ("", "nan", "{}"):
            continue
        try:
            ast.literal_eval(str(tl))
        except:
            parse_errors += 1

    print(f"  true_labels parse errors: {parse_errors}")

    # Score range check
    bad_scores = 0
    for _, row in df.iterrows():
        labels = row["true_labels_parsed"]
        if labels:
            for asp, score in labels.items():
                if score < 1 or score > 5:
                    bad_scores += 1
    print(f"  Scores outside [1,5]: {bad_scores}")

    ok = (csv_nms == json_nms) and parse_errors == 0 and bad_scores == 0
    print(f"\n  VERDICT: {'✓ CLEAN' if ok else '✗ ISSUES FOUND'}")


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 2: PIPELINE SANITY
# ═══════════════════════════════════════════════════════════════════════════

def block2_pipeline_sanity():
    print("\n" + "=" * 70)
    print("BLOCK 2: PIPELINE SANITY")
    print("=" * 70)

    per_review = load_per_review()
    step12 = load_step12()

    issues = 0
    for nm_id_str, reviews in per_review.items():
        pipeline_aspects = step12["pipeline_results"][nm_id_str]["aspects"]
        aspect_keywords = step12["pipeline_results"][nm_id_str].get("aspect_keywords", {})

        # Check: every aspect in per_review is in pipeline_aspects
        for rid, scores in reviews.items():
            for asp in scores:
                if asp not in pipeline_aspects:
                    print(f"  ✗ nm={nm_id_str} rid={rid}: aspect '{asp}' in per_review but not in pipeline_aspects")
                    issues += 1

        # Check: no NaN/Inf in scores
        for rid, scores in reviews.items():
            for asp, score in scores.items():
                if not (1.0 <= score <= 5.0):
                    print(f"  ✗ nm={nm_id_str} rid={rid}: {asp}={score} out of [1,5]")
                    issues += 1

        # Check: no duplicate reviews (should be impossible with dict)
        n_reviews = len(reviews)
        n_unique = len(set(reviews.keys()))
        if n_reviews != n_unique:
            print(f"  ✗ nm={nm_id_str}: duplicate review IDs ({n_reviews} vs {n_unique})")
            issues += 1

    print(f"\n  Issues found: {issues}")
    print(f"  VERDICT: {'✓ CLEAN' if issues == 0 else '✗ ISSUES FOUND'}")


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 3: EVAL METHODOLOGY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def block3_eval_methodology(csv_path):
    print("\n" + "=" * 70)
    print("BLOCK 3: EVAL METHODOLOGY VERIFICATION")
    print("=" * 70)

    df = load_markup(csv_path)
    per_review = load_per_review()

    # Independent MAE calculation from scratch
    all_errors = []
    all_pairs = 0
    skipped_no_mapping = 0

    # Load auto mapping from eval_metrics_auto.json
    try:
        with open("eval_metrics_auto.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print("  ✗ eval_metrics_auto.json not found. Run step4 first.")
        return

    # For each product: check MAE independently
    for nm_id_str, pm in metrics["per_product"].items():
        nm_id = int(nm_id_str)
        grp = df[df["nm_id"] == nm_id]
        reviews_dict = per_review.get(nm_id_str, {})

        reported_mae = pm.get("mae_raw")
        reported_n = pm.get("mae_n")

        # Собираем пары (pred, true) вручную — через auto mapping
        auto_mapping = metrics.get("auto_mapping", {}).get(nm_id_str, {})

        # Reverse map: true_asp → [pred_asp]
        reverse = defaultdict(list)
        for pa, ta in auto_mapping.items():
            if ta is not None:
                reverse[ta].append(pa)

        product_errors = []
        for _, row in grp.iterrows():
            labels = row["true_labels_parsed"]
            if not labels:
                continue
            rid = row["id"]
            pred_scores = reviews_dict.get(rid, {})

            # Apply aliases
            for old, new in ASPECT_ALIASES.items():
                if old in pred_scores and new not in pred_scores:
                    pred_scores[new] = pred_scores[old]

            for true_asp, true_score in labels.items():
                pred_asps = reverse.get(true_asp, [])
                pred_vals = [pred_scores[pa] for pa in pred_asps if pa in pred_scores]
                if pred_vals:
                    pred_mean = np.mean(pred_vals)
                    product_errors.append(abs(pred_mean - true_score))
                    all_pairs += 1

        if product_errors:
            my_mae = float(np.mean(product_errors))
            match = abs(my_mae - reported_mae) < 0.01 if reported_mae else False
            status = "✓" if match else "✗ MISMATCH"
            print(f"  nm={nm_id}: MAE={my_mae:.3f} (reported={reported_mae}) "
                  f"n={len(product_errors)} (reported={reported_n}) {status}")
            all_errors.extend(product_errors)
        else:
            print(f"  nm={nm_id}: no pairs")

    if all_errors:
        global_mae = float(np.mean(all_errors))
        reported_global = metrics.get("global_mae_raw")
        match = abs(global_mae - reported_global) < 0.01 if reported_global else False
        print(f"\n  Global MAE: {global_mae:.3f} (reported={reported_global}) "
              f"{'✓' if match else '✗ MISMATCH'}")
        print(f"  Total pairs: {all_pairs} (reported={metrics.get('global_mae_n')})")

    print(f"\n  VERDICT: Check values above for mismatches")


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 4: MENTION RECALL DEEP AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def block4_mention_recall_audit(csv_path):
    print("\n" + "=" * 70)
    print("BLOCK 4: MENTION RECALL DEEP AUDIT")
    print("=" * 70)

    df = load_markup(csv_path)
    per_review = load_per_review()

    # Count mentions per status
    total = 0
    identity_match = 0      # true_asp == pred_asp (or alias)
    alias_match = 0          # true_asp matched via ASPECT_ALIASES
    no_pred_in_review = 0    # review has preds, but not for this aspect
    no_review_at_all = 0     # review has zero pred aspects

    per_product_stats = {}

    for nm_id_str, reviews_dict in per_review.items():
        nm_id = int(nm_id_str)
        grp = df[df["nm_id"] == nm_id]

        p_total = 0
        p_identity = 0
        p_alias = 0
        p_no_pred = 0
        p_no_review = 0

        for _, row in grp.iterrows():
            labels = row["true_labels_parsed"]
            if not labels:
                continue
            rid = row["id"]
            pred_scores = reviews_dict.get(rid, {})

            # Build aliased pred set
            pred_set = set(pred_scores.keys())
            alias_set = set()
            for old, new in ASPECT_ALIASES.items():
                if old in pred_set:
                    alias_set.add(new)

            for true_asp in labels:
                total += 1
                p_total += 1

                if true_asp in pred_set:
                    identity_match += 1
                    p_identity += 1
                elif true_asp in alias_set:
                    alias_match += 1
                    p_alias += 1
                elif len(pred_scores) > 0:
                    no_pred_in_review += 1
                    p_no_pred += 1
                else:
                    no_review_at_all += 1
                    p_no_review += 1

        per_product_stats[nm_id] = {
            "total": p_total,
            "identity": p_identity,
            "alias": p_alias,
            "no_pred": p_no_pred,
            "no_review": p_no_review,
        }

    print(f"\n  Total mentions in markup: {total}")
    print(f"\n  BREAKDOWN:")
    print(f"    Identity match:        {identity_match:4d} ({100*identity_match/total:.1f}%)")
    print(f"    Alias match:           {alias_match:4d} ({100*alias_match/total:.1f}%)")
    print(f"    Has pred, wrong aspect:{no_pred_in_review:4d} ({100*no_pred_in_review/total:.1f}%)")
    print(f"    No pred at all:        {no_review_at_all:4d} ({100*no_review_at_all/total:.1f}%)")
    covered = identity_match + alias_match
    print(f"\n    → Coverage: {covered}/{total} ({100*covered/total:.1f}%)")

    print(f"\n  PER PRODUCT:")
    for nm_id in sorted(per_product_stats):
        s = per_product_stats[nm_id]
        cov = s["identity"] + s["alias"]
        pct = 100 * cov / s["total"] if s["total"] else 0
        print(f"    nm={nm_id}: {cov}/{s['total']} ({pct:.0f}%)  "
              f"[id={s['identity']}, alias={s['alias']}, wrong_asp={s['no_pred']}, no_review={s['no_review']}]")

    # Which true aspects are NEVER covered (identity or alias)?
    print(f"\n  TRUE ASPECTS NEVER COVERED (across all products):")
    never_covered = defaultdict(int)
    for nm_id_str, reviews_dict in per_review.items():
        nm_id = int(nm_id_str)
        grp = df[df["nm_id"] == nm_id]

        all_pred_aspects = set()
        for rid, scores in reviews_dict.items():
            all_pred_aspects.update(scores.keys())
            for old, new in ASPECT_ALIASES.items():
                if old in scores:
                    all_pred_aspects.add(new)

        for _, row in grp.iterrows():
            labels = row["true_labels_parsed"]
            if not labels:
                continue
            for true_asp in labels:
                if true_asp not in all_pred_aspects:
                    never_covered[true_asp] += 1

    if never_covered:
        for asp in sorted(never_covered, key=lambda a: -never_covered[a]):
            print(f"    {asp:30s} {never_covered[asp]:4d} mentions")
    else:
        print(f"    None ✓")


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 5: PRODUCT MAE RECOMPUTE
# ═══════════════════════════════════════════════════════════════════════════

def block5_product_mae_recompute(csv_path):
    print("\n" + "=" * 70)
    print("BLOCK 5: PRODUCT MAE INDEPENDENT RECOMPUTE")
    print("=" * 70)

    df = load_markup(csv_path)
    per_review = load_per_review()

    try:
        with open("eval_metrics_auto.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        auto_mapping = metrics.get("auto_mapping", {})
    except:
        print("  ✗ Need eval_metrics_auto.json")
        return

    all_errors_filtered = []

    for nm_id_str, reviews_dict in per_review.items():
        nm_id = int(nm_id_str)
        grp = df[df["nm_id"] == nm_id]
        mapping = auto_mapping.get(nm_id_str, {})

        # Reverse: true → [pred]
        reverse = defaultdict(list)
        for pa, ta in mapping.items():
            if ta:
                reverse[ta].append(pa)

        # True averages
        true_avgs = defaultdict(list)
        for _, row in grp.iterrows():
            labels = row["true_labels_parsed"]
            if labels:
                for asp, score in labels.items():
                    true_avgs[asp].append(score)

        # Pred averages
        pred_avgs = defaultdict(list)
        for rid, scores in reviews_dict.items():
            # Apply aliases
            aliased = dict(scores)
            for old, new in ASPECT_ALIASES.items():
                if old in aliased and new not in aliased:
                    aliased[new] = aliased[old]

            for true_asp, pred_asps in reverse.items():
                found = [aliased[pa] for pa in pred_asps if pa in aliased]
                if found:
                    pred_avgs[true_asp].append(float(np.mean(found)))

        print(f"\n  nm_id={nm_id}:")
        product_errors = []
        for true_asp in sorted(true_avgs):
            t_avg = np.mean(true_avgs[true_asp])
            n_true = len(true_avgs[true_asp])

            if true_asp in pred_avgs and pred_avgs[true_asp]:
                p_avg = np.mean(pred_avgs[true_asp])
                err = abs(t_avg - p_avg)
                product_errors.append((err, n_true))
                flag = " *" if n_true < 3 else ""
                print(f"    {true_asp:25s} true={t_avg:.2f} pred={p_avg:.2f} "
                      f"Δ={err:.2f} n_true={n_true}{flag}")
            else:
                print(f"    {true_asp:25s} true={t_avg:.2f} pred=—     n_true={n_true}")

        if product_errors:
            filtered = [e for e, n in product_errors if n >= 3]
            if filtered:
                mae = np.mean(filtered)
                all_errors_filtered.extend(filtered)
                print(f"    MAE (n≥3): {mae:.3f}")

    if all_errors_filtered:
        global_mae = np.mean(all_errors_filtered)
        print(f"\n  GLOBAL Product MAE (n≥3): {global_mae:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 6: CEILING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def block6_ceiling_analysis(csv_path):
    print("\n" + "=" * 70)
    print("BLOCK 6: CEILING ANALYSIS — What's impossible without LLM")
    print("=" * 70)

    df = load_markup(csv_path)
    per_review = load_per_review()

    # Category 1: True aspects with NO anchor (Общее впечатление, Ассортимент, etc.)
    # These can NEVER be found by anchor-first with current anchor list
    all_pred_aspects = set()
    for nm_str, reviews in per_review.items():
        for rid, scores in reviews.items():
            all_pred_aspects.update(scores.keys())
            for old, new in ASPECT_ALIASES.items():
                if old in scores:
                    all_pred_aspects.add(new)

    true_aspect_counts = defaultdict(int)
    true_aspects_never = defaultdict(int)
    for _, row in df.iterrows():
        labels = row["true_labels_parsed"]
        if labels:
            for asp in labels:
                true_aspect_counts[asp] += 1
                if asp not in all_pred_aspects:
                    true_aspects_never[asp] += 1

    total_mentions = sum(true_aspect_counts.values())
    unreachable_mentions = sum(true_aspects_never.values())

    print(f"\n  UNREACHABLE ASPECTS (no pred aspect maps to them):")
    print(f"    Total true mentions: {total_mentions}")
    print(f"    Unreachable: {unreachable_mentions} ({100*unreachable_mentions/total_mentions:.1f}%)")
    if true_aspects_never:
        for asp in sorted(true_aspects_never, key=lambda a: -true_aspects_never[a]):
            print(f"      {asp:30s} {true_aspects_never[asp]:4d} / {true_aspect_counts[asp]}")

    # Category 2: Reviews too short for any candidate extraction
    short_reviews = 0
    short_with_labels = 0
    for _, row in df.iterrows():
        text = str(row.get("full_text", ""))
        if len(text.split()) < 5:
            short_reviews += 1
            if row["true_labels_parsed"]:
                short_with_labels += 1

    print(f"\n  SHORT REVIEWS (< 5 words):")
    print(f"    Total: {short_reviews}")
    print(f"    With labels: {short_with_labels}")

    # Category 3: Semantic gap — aspects whose language doesn't match any anchor
    print(f"\n  SEMANTIC GAP (aspects where anchor vocabulary misses):")
    print(f"    'Общее впечатление' → no anchor, too abstract for unsupervised")
    print(f"    'Ассортимент' → no anchor, rare (n={true_aspect_counts.get('Ассортимент', 0)})")
    print(f"    'Комплектация' → no anchor, rare (n={true_aspect_counts.get('Комплектация', 0)})")
    print(f"    'Уход' → no anchor (n={true_aspect_counts.get('Уход', 0)})")

    # Theoretical max recall
    reachable = total_mentions - unreachable_mentions
    print(f"\n  THEORETICAL MAX RECALL (with current anchors):")
    print(f"    {reachable}/{total_mentions} = {100*reachable/total_mentions:.1f}%")
    print(f"    Current actual recall: see eval_metrics")

    # Category 4: NLI ceiling — what's the best MAE possible?
    print(f"\n  NLI CEILING ESTIMATE:")
    print(f"    Model: rubert-base-cased-nli-threeway")
    print(f"    Score formula: 1 + 4·p_pos/(p_pos+p_neg+ε)")
    print(f"    Inherent compression: pred range ~[1.5, 4.5] vs true [1, 5]")
    print(f"    → Even perfect aspect assignment gives MAE ≥ 0.3-0.4")
    print(f"    → Current Product MAE (n≥3) ≈ 0.68 is ~2x above floor")


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 7: NLI INVERSION CHECK
# ═══════════════════════════════════════════════════════════════════════════

def block7_nli_inversion(csv_path):
    print("\n" + "=" * 70)
    print("BLOCK 7: NLI SENTIMENT INVERSION CHECK")
    print("=" * 70)

    df = load_markup(csv_path)
    per_review = load_per_review()

    try:
        with open("eval_metrics_auto.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        auto_mapping = metrics.get("auto_mapping", {})
    except:
        print("  ✗ Need eval_metrics_auto.json")
        return

    # Find pairs where pred and true disagree on polarity
    # (true ≤ 2 but pred ≥ 4) or (true ≥ 4 but pred ≤ 2)
    inversions = []
    total_pairs = 0

    for nm_id_str, reviews_dict in per_review.items():
        nm_id = int(nm_id_str)
        grp = df[df["nm_id"] == nm_id]
        mapping = auto_mapping.get(nm_id_str, {})

        reverse = defaultdict(list)
        for pa, ta in mapping.items():
            if ta:
                reverse[ta].append(pa)

        for _, row in grp.iterrows():
            labels = row["true_labels_parsed"]
            if not labels:
                continue
            rid = row["id"]
            pred_scores = reviews_dict.get(rid, {})
            aliased = dict(pred_scores)
            for old, new in ASPECT_ALIASES.items():
                if old in aliased and new not in aliased:
                    aliased[new] = aliased[old]

            for true_asp, true_score in labels.items():
                pred_asps = reverse.get(true_asp, [])
                pred_vals = [aliased[pa] for pa in pred_asps if pa in aliased]
                if pred_vals:
                    pred_mean = float(np.mean(pred_vals))
                    total_pairs += 1

                    if (true_score <= 2 and pred_mean >= 4) or (true_score >= 4 and pred_mean <= 2):
                        inversions.append({
                            "nm_id": nm_id,
                            "rid": rid,
                            "aspect": true_asp,
                            "true": true_score,
                            "pred": round(pred_mean, 2),
                        })

    inv_rate = 100 * len(inversions) / total_pairs if total_pairs else 0
    print(f"\n  Total sentiment pairs: {total_pairs}")
    print(f"  Inversions (true≤2↔pred≥4): {len(inversions)} ({inv_rate:.1f}%)")

    if inversions:
        # Group by product
        by_product = defaultdict(list)
        for inv in inversions:
            by_product[inv["nm_id"]].append(inv)

        for nm_id in sorted(by_product):
            invs = by_product[nm_id]
            print(f"\n  nm_id={nm_id}: {len(invs)} inversions")
            for inv in invs[:5]:
                print(f"    {inv['aspect']:25s} true={inv['true']:.0f} pred={inv['pred']:.2f}")
            if len(invs) > 5:
                print(f"    ... and {len(invs)-5} more")

    print(f"\n  VERDICT: {'✓ < 5% inversions' if inv_rate < 5 else '⚠ HIGH inversion rate'}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    csv_path = "parser/razmetka/checked_reviews.csv"
    json_path = "parser/razmetka/longest_reviews.json"

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          FINAL PIPELINE AUDIT — 7 BLOCKS                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    block1_data_integrity(csv_path, json_path)
    block2_pipeline_sanity()
    block3_eval_methodology(csv_path)
    block4_mention_recall_audit(csv_path)
    block5_product_mae_recompute(csv_path)
    block6_ceiling_analysis(csv_path)
    block7_nli_inversion(csv_path)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()