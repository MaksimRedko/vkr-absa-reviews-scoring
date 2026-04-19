"""
Matching Audit Runner: forensic audit for candidate -> aspect assignment.
Iteration 2: Focus on detection bottleneck.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import temporary_config_overrides
from eval_pipeline import load_markup, MANUAL_MAPPING, _build_auto_mapping
from src.schemas.models import ReviewInput, ScoredCandidate

def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def get_matching_data_from_df(df: pd.DataFrame, clusterer_name: str) -> Dict[int, Dict[str, Any]]:
    from src.factories import build_clustering_stage
    from src.pipeline import ABSAPipeline
    from src.stages.pairing import extract_all_with_mapping
    from configs.configs import config

    encoder = SentenceTransformer(config.models.encoder_path)
    pipeline = ABSAPipeline(encoder=encoder)
    pipeline.clusterer = build_clustering_stage(encoder=encoder, name=clusterer_name)

    results = {}
    nm_ids = df['nm_id'].unique()

    for nm_id in nm_ids:
        grp = df[df["nm_id"] == nm_id]
        reviews = []
        for _, row in grp.iterrows():
            ft = row.get("full_text")
            pr = row.get("pros")
            cn = row.get("cons")
            reviews.append(ReviewInput(
                nm_id=int(row["nm_id"]),
                id=str(row["id"]),
                rating=int(row["rating"]),
                created_date=str(row["created_date"]).strip(),
                full_text="" if pd.isna(ft) else str(ft),
                pros="" if pd.isna(pr) else str(pr),
                cons="" if pd.isna(cn) else str(cn),
            ))
        
        texts = [r.clean_text for r in reviews]
        review_ids = [r.id for r in reviews]

        all_candidates, sentence_to_review = extract_all_with_mapping(
            pipeline.candidate_extractor, texts, review_ids
        )
        scored_candidates = pipeline.scorer.score_and_select(all_candidates)
        aspects = pipeline.clusterer.cluster(scored_candidates)
        pairing_meta = pipeline.clusterer.get_pairing_metadata()

        results[int(nm_id)] = {
            'scored_candidates': scored_candidates,
            'aspects': aspects,
            'anchor_embeddings': pairing_meta.anchor_embeddings,
            'candidate_assignments': pairing_meta.candidate_assignments,
            'review_ids': review_ids,
            'sentence_to_review': sentence_to_review
        }
    return results

def run_matching_audit(
    df: pd.DataFrame,
    pipeline_results: Dict[int, Dict[str, Any]],
    mapping: Dict[int, Dict[str, Optional[str]]],
    out_dir: Path
):
    all_rows = []
    
    for nm_id, data in pipeline_results.items():
        pm = mapping.get(nm_id, {})
        scored_candidates: List[ScoredCandidate] = data['scored_candidates']
        anchor_embeddings: Dict[str, np.ndarray] = data['anchor_embeddings']
        candidate_assignments: Dict[str, str] = data['candidate_assignments']
        sentence_to_review: Dict[str, str] = data['sentence_to_review']
        
        if not anchor_embeddings or not scored_candidates:
            continue
            
        anchor_names = list(anchor_embeddings.keys())
        anchor_matrix = np.stack([anchor_embeddings[n] for n in anchor_names])
        
        markup_grp = df[df['nm_id'] == nm_id]
        review_to_true = {str(row['id']): row['true_labels_parsed'] for _, row in markup_grp.iterrows()}

        for cand in scored_candidates:
            emb = np.asarray(cand.embedding, dtype=np.float64).reshape(1, -1)
            sims = cosine_similarity(emb, anchor_matrix)[0]
            
            sorted_indices = np.argsort(sims)[::-1]
            top5_indices = sorted_indices[:5]
            
            top_aspects = []
            for idx in top5_indices:
                aname = anchor_names[idx]
                score = float(sims[idx])
                top_aspects.append({
                    'anchor': aname,
                    'mapped_true': pm.get(aname),
                    'score': score
                })
            
            rid = sentence_to_review.get(cand.sentence.strip(), 
                                        sentence_to_review.get(cand.sentence.lower().strip(), "unknown"))
            true_labels = review_to_true.get(rid) or {}
            true_aspects_review = sorted(true_labels.keys())
            
            pred_aspect_orig = candidate_assignments.get(str(getattr(cand, "candidate_id", "")), "")
            pred_aspect_mapped = pm.get(pred_aspect_orig) if pred_aspect_orig else None
            
            margin = float(sims[sorted_indices[0]] - (sims[sorted_indices[1]] if len(sims)>1 else 0.0))
            
            all_rows.append({
                'product_id': nm_id,
                'review_id': rid,
                'candidate_id': getattr(cand, "candidate_id", ""),
                'candidate_text': getattr(cand, "source_span", cand.span),
                'candidate_score': cand.score,
                'predicted_aspect_raw': pred_aspect_orig,
                'predicted_aspect_mapped': pred_aspect_mapped,
                'true_aspects_review': json.dumps(true_aspects_review, ensure_ascii=False),
                'top1_anchor': top_aspects[0]['anchor'],
                'top1_mapped': top_aspects[0]['mapped_true'],
                'top1_score': top_aspects[0]['score'],
                'top2_anchor': top_aspects[1]['anchor'] if len(top_aspects) > 1 else None,
                'top2_score': top_aspects[1]['score'] if len(top_aspects) > 1 else None,
                'margin': margin,
                'top5_json': json.dumps(top_aspects, ensure_ascii=False),
                'is_correct_top1': top_aspects[0]['mapped_true'] in true_aspects_review if top_aspects[0]['mapped_true'] else False
            })
            
    debug_df = pd.DataFrame(all_rows)
    debug_df.to_csv(out_dir / "candidate_assignment_debug.csv", index=False, encoding="utf-8")
    
    run_ablations(debug_df, df, mapping, out_dir)
    run_slice_summary(debug_df, df, mapping, out_dir)
    run_aspect_failure_analysis(debug_df, out_dir)
    run_confidence_buckets(debug_df, out_dir)
    run_margin_analysis(debug_df, out_dir)
    run_sample_cases(debug_df, out_dir)
    write_matching_summary_md(out_dir, debug_df)

def run_ablations(debug_df: pd.DataFrame, markup_df: pd.DataFrame, mapping: Dict[int, Dict[str, Optional[str]]], out_dir: Path):
    thresholds = [0.0, 0.5, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]
    threshold_results = []
    review_to_true = {str(row['id']): set((row['true_labels_parsed'] or {}).keys()) for _, row in markup_df.iterrows()}
    all_rids = set(review_to_true.keys())

    for tau in thresholds:
        review_preds = {}
        for rid, grp in debug_df.groupby('review_id'):
            preds = set()
            for _, row in grp.iterrows():
                if row['top1_score'] >= tau and row['top1_mapped']:
                    preds.add(row['top1_mapped'])
            review_preds[rid] = preds
        metrics = compute_metrics_for_preds(review_preds, review_to_true, all_rids)
        metrics['threshold'] = tau
        threshold_results.append(metrics)
    pd.DataFrame(threshold_results).to_csv(out_dir / "threshold_ablation.csv", index=False)
    
    top_k_results = []
    for k in [1, 2, 3]:
        review_preds = {}
        for rid, grp in debug_df.groupby('review_id'):
            preds = set()
            for _, row in grp.iterrows():
                top5 = json.loads(row['top5_json'])
                for i in range(min(k, len(top5))):
                    if top5[i]['mapped_true']:
                        preds.add(top5[i]['mapped_true'])
            review_preds[rid] = preds
        metrics = compute_metrics_for_preds(review_preds, review_to_true, all_rids)
        metrics['k'] = k
        top_k_results.append(metrics)
    pd.DataFrame(top_k_results).to_csv(out_dir / "topk_ablation.csv", index=False)

def compute_metrics_for_preds(review_preds, review_to_true, all_rids):
    precisions = []
    recalls = []
    tp_micro = fp_micro = fn_micro = 0
    reviews_with_candidates_but_no_pred = 0
    reviews_with_wrong_only = 0
    reviews_with_at_least_one_correct = 0
    
    for rid in all_rids:
        true_set = review_to_true[rid]
        if not true_set: continue
        pred_set = review_preds.get(rid, set())
        if not pred_set:
            reviews_with_candidates_but_no_pred += 1
            recalls.append(0.0)
            fn_micro += len(true_set)
            continue
        inter = pred_set & true_set
        p = len(inter) / len(pred_set) if pred_set else 0
        r = len(inter) / len(true_set) if true_set else 0
        precisions.append(p)
        recalls.append(r)
        tp_micro += len(inter)
        fp_micro += len(pred_set - true_set)
        fn_micro += len(true_set - pred_set)
        if not inter: reviews_with_wrong_only += 1
        else: reviews_with_at_least_one_correct += 1
            
    macro_p = np.mean(precisions) if precisions else 0
    macro_r = np.mean(recalls) if recalls else 0
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0
    micro_p = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
    micro_r = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    
    return {
        'macro_precision': macro_p, 'macro_recall': macro_r, 'macro_f1': macro_f1,
        'micro_precision': micro_p, 'micro_recall': micro_r, 'micro_f1': micro_f1,
        'reviews_with_candidates_but_no_pred_aspects': reviews_with_candidates_but_no_pred,
        'reviews_with_wrong_aspect_only': reviews_with_wrong_only,
        'reviews_with_at_least_one_correct_aspect': reviews_with_at_least_one_correct
    }

def run_slice_summary(debug_df, markup_df, mapping, out_dir):
    manual_nm_ids = set(MANUAL_MAPPING.keys())
    slices = {
        'all': debug_df['product_id'].unique(),
        'manual-only': [nid for nid in debug_df['product_id'].unique() if nid in manual_nm_ids],
        'auto-only': [nid for nid in debug_df['product_id'].unique() if nid not in manual_nm_ids]
    }
    review_to_true = {str(row['id']): set((row['true_labels_parsed'] or {}).keys()) for _, row in markup_df.iterrows()}
    rows = []
    for name, ids in slices.items():
        if len(ids) == 0: continue
        slice_debug = debug_df[debug_df['product_id'].isin(ids)]
        slice_markup = markup_df[markup_df['nm_id'].isin(ids)]
        all_rids = set(str(rid) for rid in slice_markup['id'])
        review_preds = {}
        for rid, grp in slice_debug.groupby('review_id'):
            preds = set(grp['predicted_aspect_mapped'].dropna())
            review_preds[rid] = preds
        metrics = compute_metrics_for_preds(review_preds, review_to_true, all_rids)
        metrics['slice'] = name
        rids_with_cands = set(slice_debug['review_id'])
        metrics['reviews_with_no_candidates'] = len(all_rids - rids_with_cands)
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(out_dir / "matching_slice_summary.csv", index=False)

def run_aspect_failure_analysis(debug_df, out_dir):
    aspect_rows = []
    all_preds = debug_df['predicted_aspect_mapped'].dropna().unique()
    all_true_in_debug = set()
    for tr_json in debug_df['true_aspects_review']:
        all_true_in_debug.update(json.loads(tr_json))
    all_aspect_names = sorted(set(all_preds) | all_true_in_debug)
    
    for asp in all_aspect_names:
        pred_mask = debug_df['predicted_aspect_mapped'] == asp
        pred_count = int(pred_mask.sum())
        correct_count = 0
        confusions = Counter()
        scores = []
        margins = []
        for _, row in debug_df[pred_mask].iterrows():
            true_set = set(json.loads(row['true_aspects_review']))
            if asp in true_set: correct_count += 1
            else:
                for t in true_set: confusions[t] += 1
            scores.append(row['top1_score'])
            margins.append(row['margin'])
        precision = correct_count / pred_count if pred_count > 0 else 0
        true_total = true_caught = 0
        for _, row in debug_df.iterrows():
            true_set = set(json.loads(row['true_aspects_review']))
            if asp in true_set:
                true_total += 1
                if row['predicted_aspect_mapped'] == asp: true_caught += 1
        recall = true_caught / true_total if true_total > 0 else 0
        most_common = ", ".join([f"{k}({v})" for k, v in confusions.most_common(3)])
        comment = ""
        if pred_count > 20 and precision < 0.2: comment = "attractor_aspect"
        if true_total > 20 and recall < 0.1: comment = "starved_aspect"
        if pred_count > 10 and precision < 0.5 and most_common: comment += " confused_aspect"
        aspect_rows.append({
            'aspect_name': asp, 'predicted_count': pred_count, 'matched_true_count': true_caught,
            'true_total_mentions': true_total, 'precision_like': precision, 'recall_like': recall,
            'most_common_confusions': most_common, 'avg_top1_score': np.mean(scores) if scores else 0,
            'avg_margin_top1_top2': np.mean(margins) if margins else 0, 'comment': comment.strip()
        })
    asp_df = pd.DataFrame(aspect_rows)
    asp_df.to_csv(out_dir / "aspect_failure_analysis.csv", index=False)
    lines = ["# Aspect failure analysis\n", "| aspect | pred | true_mentions | P | R | confusions | comment |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in asp_df.sort_values('predicted_count', ascending=False).iterrows():
        lines.append(f"| {r['aspect_name']} | {r['predicted_count']} | {r['true_total_mentions']} | {r['precision_like']:.2f} | {r['recall_like']:.2f} | {r['most_common_confusions']} | {r['comment']} |")
    (out_dir / "aspect_failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")

def run_confidence_buckets(debug_df, out_dir):
    bins = [0, 0.7, 0.8, 0.85, 0.9, 0.92, 1.0]
    labels = ['low', 'mid', 'high', 'v_high', 'extreme', 'max']
    debug_df['conf_bucket'] = pd.cut(debug_df['top1_score'], bins=bins, labels=labels)
    summary = debug_df.groupby('conf_bucket', observed=False).agg({
        'is_correct_top1': ['count', 'mean'], 'top1_score': 'mean'
    }).reset_index()
    summary.columns = ['bucket', 'count', 'precision', 'avg_score']
    summary.to_csv(out_dir / "confidence_buckets.csv", index=False)
    lines = ["# Confidence buckets\n", "| bucket | count | precision | avg_score |", "| --- | --- | --- | --- |"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['bucket']} | {r['count']} | {r['precision']:.2f} | {r['avg_score']:.3f} |")
    (out_dir / "confidence_buckets.md").write_text("\n".join(lines), encoding="utf-8")

def run_margin_analysis(debug_df, out_dir):
    bins = [0, 0.01, 0.03, 0.05, 0.1, 1.0]
    labels = ['near_tie', 'small', 'medium', 'large', 'very_large']
    debug_df['margin_bucket'] = pd.cut(debug_df['margin'], bins=bins, labels=labels)
    summary = debug_df.groupby('margin_bucket', observed=False).agg({
        'is_correct_top1': ['count', 'mean'], 'margin': 'mean'
    }).reset_index()
    summary.columns = ['bucket', 'count', 'precision', 'avg_margin']
    summary.to_csv(out_dir / "margin_analysis.csv", index=False)
    lines = ["# Margin analysis\n", "| bucket | count | precision | avg_margin |", "| --- | --- | --- | --- |"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['bucket']} | {r['count']} | {r['precision']:.2f} | {r['avg_margin']:.3f} |")
    (out_dir / "margin_analysis.md").write_text("\n".join(lines), encoding="utf-8")

def run_sample_cases(debug_df, out_dir):
    samples = []
    samples.append(debug_df[debug_df['is_correct_top1'] & (debug_df['top1_score'] > 0.6)].head(20))
    samples.append(debug_df[~debug_df['is_correct_top1'] & (debug_df['top1_score'] > 0.6)].head(20))
    samples.append(debug_df[debug_df['margin'] < 0.02].head(20))
    def check_top2_correct(row):
        top5 = json.loads(row['top5_json'])
        if len(top5) < 2: return False
        true_set = set(json.loads(row['true_aspects_review']))
        return top5[1]['mapped_true'] in true_set
    debug_df['is_top2_correct'] = debug_df.apply(check_top2_correct, axis=1)
    samples.append(debug_df[~debug_df['is_correct_top1'] & debug_df['is_top2_correct']].head(20))
    samples.append(debug_df.sample(min(20, len(debug_df))))
    pd.concat(samples).drop_duplicates().to_csv(out_dir / "sample_candidate_cases.csv", index=False)

def write_matching_summary_md(out_dir, debug_df):
    lines = ["# Matching Audit Summary\n", "## Overview\n"]
    lines.append(f"- Total candidates processed: {len(debug_df)}")
    lines.append(f"- Global candidate-level Top-1 Precision: {debug_df['is_correct_top1'].mean():.4f}")
    lines.append(f"- Average margin: {debug_df['margin'].mean():.4f}")
    lines.append("\n## Primary matching metrics (from summary scripts)\n")
    lines.append("See `matching_slice_summary.csv` and `threshold_ablation.csv` for detailed review-level metrics.\n")
    (out_dir / "matching_summary.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-paths", nargs="+", default=[
        "benchmark/eval_datasets/combined_benchmark.csv",
        "parser/razmetka/checked_reviews.csv",
        "parser/reviews_batches/merged_checked_reviews.csv"
    ])
    parser.add_argument("--clusterer", default="aspect", choices=["aspect", "divisive", "mdl_divisive"])
    parser.add_argument("--auto-threshold", type=float, default=0.3)
    args = parser.parse_args()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "matching_audit" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    for path in args.csv_paths:
        if os.path.exists(path): dfs.append(load_markup(path))
    if not dfs:
        print("No CSV files found!"); return
    df = pd.concat(dfs).drop_duplicates(subset=['id'])
    
    print(f"Running matching audit for {df['nm_id'].nunique()} products...")
    data = get_matching_data_from_df(df, args.clusterer)
    
    sim_results = {}
    for nid, d in data.items():
        sim_results[nid] = {
            'aspects': list(d['aspects'].keys()),
            'aspect_keywords': {n: info.keywords for n, info in d['aspects'].items()}
        }
    auto_mapping = _build_auto_mapping(sim_results, df, threshold=args.auto_threshold)
    mapping = {}
    for nid in df['nm_id'].unique():
        if nid in MANUAL_MAPPING: mapping[nid] = MANUAL_MAPPING[nid]
        else: mapping[nid] = auto_mapping.get(nid, {})
            
    run_matching_audit(df, data, mapping, out_dir)
    print(f"Audit finished. Results in {out_dir}")

if __name__ == "__main__":
    main()
