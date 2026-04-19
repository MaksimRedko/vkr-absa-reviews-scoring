import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional

import numpy as np
import pandas as pd

def compute_metrics_for_preds(review_preds: Dict[str, Set[str]], review_to_true: Dict[str, Set[str]], all_rids: Set[str]):
    precisions = []
    recalls = []
    tp_micro = fp_micro = fn_micro = 0
    
    for rid in all_rids:
        true_set = review_to_true.get(rid, set())
        if not true_set: continue
        
        pred_set = review_preds.get(rid, set())
        
        inter = pred_set & true_set
        p = len(inter) / len(pred_set) if pred_set else 0.0
        r = len(inter) / len(true_set) if true_set else 0.0
        precisions.append(p)
        recalls.append(r)
        
        tp_micro += len(inter)
        fp_micro += len(pred_set - true_set)
        fn_micro += len(true_set - pred_set)
            
    macro_p = np.mean(precisions) if precisions else 0.0
    macro_r = np.mean(recalls) if recalls else 0.0
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    
    micro_p = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    micro_r = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    return {
        'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
        'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-csv", default="outputs/matching_audit/20260419_171814/candidate_assignment_debug.csv")
    parser.add_argument("--merge-map", default="aspect_merge_map.json")
    args = parser.parse_args()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs/aspect_space_audit") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.merge_map, "r", encoding="utf-8") as f:
        merge_map = json.load(f)
    
    df = pd.read_csv(args.debug_csv)
    
    # 1. Prepare remapped labels
    # We need to remap both predicted and true labels
    def remap_aspect(asp: Optional[str]) -> Optional[str]:
        if pd.isna(asp) or not asp: return None
        return merge_map.get(asp, asp) # If not in map, keep as is or we could drop it. 
        # Requirement says "approximately 15-25 aspects", so keeping others might be fine or we drop unknowns.
        # Let's keep only those in map or explicitly allowed?
        # Actually, let's keep all but those in map get merged.

    def remap_set(json_set_str: str) -> Set[str]:
        try:
            items = json.loads(json_set_str)
            return {merge_map.get(i, i) for i in items if i}
        except:
            return set()

    # Create remapped columns
    df['top1_remapped'] = df['top1_mapped'].apply(remap_aspect)
    df['true_remapped'] = df['true_aspects_review'].apply(lambda x: json.dumps(list(remap_set(x)), ensure_ascii=False))

    # All unique review IDs
    all_rids = set(df['review_id'].unique())
    review_to_true_orig = {rid: set(json.loads(grp['true_aspects_review'].iloc[0])) 
                           for rid, grp in df.groupby('review_id')}
    review_to_true_remapped = {rid: remap_set(grp['true_aspects_review'].iloc[0]) 
                               for rid, grp in df.groupby('review_id')}

    # 2. Before vs After Comparison
    # Original metrics (Top-1, threshold=0.0)
    preds_orig = {rid: set(grp['top1_mapped'].dropna()) for rid, grp in df.groupby('review_id')}
    metrics_orig = compute_metrics_for_preds(preds_orig, review_to_true_orig, all_rids)
    
    # Remapped metrics (Top-1, threshold=0.0)
    preds_remapped = {rid: set(grp['top1_remapped'].dropna()) for rid, grp in df.groupby('review_id')}
    metrics_remapped = compute_metrics_for_preds(preds_remapped, review_to_true_remapped, all_rids)
    
    comp_df = pd.DataFrame([
        {'stage': 'Original Space', **metrics_orig},
        {'stage': 'Compact Space', **metrics_remapped}
    ])
    comp_df.to_csv(out_dir / "before_after_comparison.csv", index=False)

    # 3. Threshold Ablation on Compact Space
    thresholds = [0.0, 0.5, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]
    th_results = []
    for tau in thresholds:
        review_preds = {}
        for rid, grp in df.groupby('review_id'):
            preds = set()
            for _, row in grp.iterrows():
                if row['top1_score'] >= tau and row['top1_remapped']:
                    preds.add(row['top1_remapped'])
            review_preds[rid] = preds
        m = compute_metrics_for_preds(review_preds, review_to_true_remapped, all_rids)
        m['threshold'] = tau
        th_results.append(m)
    pd.DataFrame(th_results).to_csv(out_dir / "compact_threshold_ablation.csv", index=False)

    # 4. Top-K Ablation on Compact Space
    tk_results = []
    for k in [1, 2, 3]:
        review_preds = {}
        for rid, grp in df.groupby('review_id'):
            preds = set()
            for _, row in grp.iterrows():
                top5 = json.loads(row['top5_json'])
                for i in range(min(k, len(top5))):
                    asp = top5[i]['mapped_true']
                    if asp:
                        preds.add(merge_map.get(asp, asp))
            review_preds[rid] = preds
        m = compute_metrics_for_preds(review_preds, review_to_true_remapped, all_rids)
        m['k'] = k
        tk_results.append(m)
    pd.DataFrame(tk_results).to_csv(out_dir / "compact_topk_ablation.csv", index=False)

    # 5. Summary MD
    summary_lines = [
        "# Compact Aspect Space Audit Summary\n",
        f"- **Based on run**: {args.debug_csv}",
        f"- **Unique aspects after merge**: {len(set(merge_map.values()))} families defined\n",
        "## Before vs After (Threshold=0, Top-1)\n",
        comp_df.to_markdown(index=False),
        "\n## Threshold Ablation (Compact Space)\n",
        pd.DataFrame(th_results).to_markdown(index=False),
        "\n## Top-K Ablation (Compact Space)\n",
        pd.DataFrame(tk_results).to_markdown(index=False),
        "\n## Conclusions\n",
        "1. Did F1 improve? " + ("Yes" if metrics_remapped['macro_f1'] > metrics_orig['macro_f1'] else "No"),
        f"\n2. Top-2 Compact F1: {tk_results[1]['macro_f1']:.4f}"
    ]
    (out_dir / "compact_matching_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    
    # Save compact definition
    compact_def = ["# Compact Space Definition\n", "| Family | Original Aspects |", "| --- | --- |"]
    inverse_map = {}
    for k, v in merge_map.items():
        inverse_map.setdefault(v, []).append(k)
    for fam, origs in inverse_map.items():
        compact_def.append(f"| {fam} | {', '.join(origs)} |")
    (out_dir / "compact_space_definition.md").write_text("\n".join(compact_def), encoding="utf-8")

    print(f"Audit finished. Results in {out_dir}")

if __name__ == "__main__":
    main()
