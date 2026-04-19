"""
ABSA Unified Tool: The single entry point for audit, extraction, and mapping.
Replaces legacy run_*.py scripts.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Project imports
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_pipeline import load_markup, MANUAL_MAPPING, _build_auto_mapping
# We reuse the core evaluation logic we perfected in Iteration 4
import run_grand_final_eval as final_eval

def cmd_extract(args):
    """Command: Extract random reviews with labels for fixtures."""
    from extract_pilot_csv import csv_paths # temporarily use its logic before delete
    
    dfs = []
    for p in args.csv_paths:
        if os.path.exists(p):
            df_item = pd.read_csv(p)
            required = ['id', 'nm_id', 'rating', 'full_text', 'true_labels']
            if all(col in df_item.columns for col in required):
                dfs.append(df_item[required])
    
    if not dfs:
        print("No valid data found.")
        return

    df = pd.concat(dfs).drop_duplicates(subset=['id'])
    df = df[df['true_labels'].notna() & (df['true_labels'] != '{}')]
    if args.min_len:
        df = df[df['full_text'].str.len() > args.min_len]
        
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    sample.to_csv(args.output, index=False, encoding='utf-8')
    print(f"Extracted {len(sample)} rows to {args.output}")

def cmd_audit(args):
    """Command: Run the full grand final audit."""
    print(f"Starting audit for {args.clusterer} clusterer...")
    
    # Load data
    dfs = [load_markup(p) for p in args.csv_paths if os.path.exists(p)]
    df = pd.concat(dfs).drop_duplicates(subset=['id'])
    
    # Load merge map
    with open(args.merge_map, "r", encoding="utf-8") as f:
        merge_map = json.load(f)
        
    # Get raw pipeline results
    print("Running pipeline stages (Extraction -> Scoring -> Clustering -> Sentiment)...")
    raw_data = final_eval.get_raw_pipeline_data(df, args.clusterer)
    
    # Mapping logic
    from eval_pipeline import _build_auto_mapping
    sim_results = {}
    for nid, d in raw_data.items():
        sim_results[nid] = {
            'aspects': list(d['aspects_info'].keys()),
            'aspect_keywords': {n: info.keywords for n, info in d['aspects_info'].items()}
        }
    auto_mapping = _build_auto_mapping(sim_results, df, threshold=args.auto_threshold)
    mapping = {nid: (MANUAL_MAPPING[nid] if nid in MANUAL_MAPPING else auto_mapping.get(nid, {})) 
               for nid in df['nm_id'].unique()}

    # Slices and configs
    manual_nm_ids = set(MANUAL_MAPPING.keys()) & set(df['nm_id'].unique())
    slices = [("Mixed/All", None), ("Manual-Only", manual_nm_ids)]
    configs = [
        ("Original + Top-1", False, 1),
        ("Compact + Top-1", True, 1),
        ("Compact + Top-2", True, 2)
    ]
    
    results_rows = []
    for slice_name, slice_ids in slices:
        print(f"Evaluating slice: {slice_name}")
        for cfg_name, use_compact, top_k in configs:
            m = final_eval.run_eval_for_config(raw_data, df, mapping, merge_map, use_compact, top_k, slice_ids)
            results_rows.append({'Slice': slice_name, 'Config': cfg_name, **m})
        
        # Star Baseline
        b = final_eval.get_star_baseline(df, merge_map, True, slice_ids)
        results_rows.append({'Slice': slice_name, 'Config': "Star Baseline", **b})

    # Save results
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_df = pd.DataFrame(results_rows)
    res_df.to_csv(out_dir / "audit_metrics.csv", index=False)
    
    # Generate MD report
    report = ["# ABSA Audit Report\n", f"Generated: {datetime.now().isoformat()}\n"]
    for slice_name in ["Mixed/All", "Manual-Only"]:
        report.append(f"## Slicing: {slice_name}\n")
        slice_df = res_df[res_df['Slice'] == slice_name].drop(columns=['Slice'])
        report.append(slice_df.to_markdown(index=False))
        report.append("\n")
    
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Audit finished. Results in: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="ABSA Project Unified Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Audit command
    p_audit = subparsers.add_parser("audit", help="Run full pipeline audit")
    p_audit.add_argument("--clusterer", default="aspect", choices=["aspect", "divisive"])
    p_audit.add_argument("--csv-paths", nargs="+", default=[
        "benchmark/eval_datasets/combined_benchmark.csv",
        "parser/razmetka/checked_reviews.csv"
    ])
    p_audit.add_argument("--merge-map", default="aspect_merge_map.json")
    p_audit.add_argument("--auto-threshold", type=float, default=0.3)
    p_audit.add_argument("--output-dir", default="outputs/audit")

    # Extract command
    p_ext = subparsers.add_parser("extract", help="Sample reviews for test data")
    p_ext.add_argument("-n", type=int, default=10)
    p_ext.add_argument("--output", default="sampled_reviews.csv")
    p_ext.add_argument("--min-len", type=int, default=30)
    p_ext.add_argument("--seed", type=int, default=42)
    p_ext.add_argument("--csv-paths", nargs="+", default=[
        "benchmark/eval_datasets/combined_benchmark.csv",
        "parser/razmetka/checked_reviews.csv"
    ])

    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "audit":
        cmd_audit(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
