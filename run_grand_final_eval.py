import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.configs import config
from eval_pipeline import load_markup, MANUAL_MAPPING, _build_auto_mapping
from src.schemas.models import ReviewInput, ScoredCandidate, AspectInfo, SentimentPair

def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def get_raw_pipeline_data(df: pd.DataFrame, clusterer_name: str) -> Dict[int, Dict[str, Any]]:
    from src.factories import build_clustering_stage, build_sentiment_stage
    from src.pipeline import ABSAPipeline
    from src.stages.pairing import extract_all_with_mapping

    encoder = SentenceTransformer(config.models.encoder_path)
    pipeline = ABSAPipeline(encoder=encoder)
    pipeline.clusterer = build_clustering_stage(encoder=encoder, name=clusterer_name)
    pipeline.sentiment_engine = build_sentiment_stage()

    results = {}
    nm_ids = df['nm_id'].unique()

    for nm_id in nm_ids:
        grp = df[df["nm_id"] == nm_id]
        reviews = []
        for _, row in grp.iterrows():
            reviews.append(ReviewInput(
                nm_id=int(row["nm_id"]),
                id=str(row["id"]),
                rating=int(row["rating"]),
                created_date=str(row.get("created_date", "")),
                full_text="" if pd.isna(row.get("full_text")) else str(row["full_text"]),
                pros="" if pd.isna(row.get("pros")) else str(row["pros"]),
                cons="" if pd.isna(row.get("cons")) else str(row["cons"]),
            ))
        
        texts = [r.clean_text for r in reviews]
        review_ids = [r.id for r in reviews]

        all_candidates, sentence_to_review = extract_all_with_mapping(
            pipeline.candidate_extractor, texts, review_ids
        )
        scored_candidates = pipeline.scorer.score_and_select(all_candidates)
        aspects = pipeline.clusterer.cluster(scored_candidates)
        pairing_meta = pipeline.clusterer.get_pairing_metadata()
        
        # Prepare pairs for sentiment
        pairs = []
        for cand in scored_candidates:
            rid = sentence_to_review.get(cand.sentence.strip(), 
                                        sentence_to_review.get(cand.sentence.lower().strip(), "unknown"))
            # Assign a default NLI label (e.g. from its cluster or generic)
            # SentimentEngine v5 uses nli_label for hypothesis template
            # For our audit, we just need the candidate's core sentiment
            nli_lab = pairing_meta.candidate_assignments.get(str(getattr(cand, "candidate_id", "")), "Общее")
            pairs.append(SentimentPair(
                review_id=rid,
                sentence=cand.sentence,
                aspect=nli_lab,
                nli_label=nli_lab
            ))
            
        sentiment_results = pipeline.sentiment_engine.batch_analyze(pairs)
        for cand, res in zip(scored_candidates, sentiment_results):
            cand.sentiment = res.score

        results[int(nm_id)] = {
            'scored_candidates': scored_candidates,
            'anchor_embeddings': pairing_meta.anchor_embeddings,
            'review_ids': review_ids,
            'sentence_to_review': sentence_to_review,
            'aspects_info': aspects
        }
    return results

def compute_all_metrics(preds: Dict[str, Dict[str, float]], true: Dict[str, Dict[str, float]], all_rids: Set[str]) -> Dict[str, Any]:
    precisions = []
    recalls = []
    sent_errors = []
    sent_errors_rounded = []
    
    product_true = {} 
    product_pred = {}
    
    for rid in all_rids:
        t_dict = true.get(rid)
        p_dict = preds.get(rid, {})
        
        if t_dict is None:
            continue
            
        t_set = set(t_dict.keys())
        p_set = set(p_dict.keys())
        
        if not t_set and not p_set:
            continue
        
        # Detection
        inter = t_set & p_set
        precisions.append(len(inter) / len(p_set) if p_set else 0.0)
        recalls.append(len(inter) / len(t_set) if t_set else 0.0)
        
        # Sentiment (only on reviews that HAVE true labels)
        if t_set:
            for asp in inter:
                err = abs(p_dict[asp] - t_dict[asp])
                sent_errors.append(err)
                sent_errors_rounded.append(abs(round(p_dict[asp]) - round(t_dict[asp])))
                
                product_true.setdefault(asp, []).append(t_dict[asp])
                product_pred.setdefault(asp, []).append(p_dict[asp])
            
    macro_p = np.mean(precisions) if precisions else 0.0
    macro_r = np.mean(recalls) if recalls else 0.0
    macro_f1 = 2*macro_p*macro_r/(macro_p+macro_r) if (macro_p+macro_r)>0 else 0.0
    
    prod_maes = []
    prod_maes_n3 = []
    for asp in product_true:
        if asp in product_pred:
            mae = abs(np.mean(product_pred[asp]) - np.mean(product_true[asp]))
            prod_maes.append(mae)
            if len(product_true[asp]) >= 3:
                prod_maes_n3.append(mae)
                
    return {
        'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
        'sent_mae': np.mean(sent_errors) if sent_errors else 0.0,
        'sent_mae_rounded': np.mean(sent_errors_rounded) if sent_errors_rounded else 0.0,
        'prod_mae': np.mean(prod_maes) if prod_maes else 0.0,
        'prod_mae_n3': np.mean(prod_maes_n3) if prod_maes_n3 else 0.0
    }

def run_eval_for_config(
    raw_data: Dict[int, Dict[str, Any]],
    df: pd.DataFrame,
    mapping: Dict[int, Dict[str, Optional[str]]],
    merge_map: Dict[str, str],
    use_compact: bool,
    top_k: int,
    slice_nm_ids: Optional[Set[int]] = None
) -> Dict[str, Any]:
    
    review_to_true = {str(row['id']): row['true_labels_parsed'] for _, row in df.iterrows()}
    if slice_nm_ids:
        all_rids = {str(rid) for rid in df[df['nm_id'].isin(slice_nm_ids)]['id']}
    else:
        all_rids = set(review_to_true.keys())

    review_preds = {}
    
    for nm_id, data in raw_data.items():
        if slice_nm_ids and nm_id not in slice_nm_ids:
            continue
            
        pm = mapping.get(nm_id, {})
        anchor_embeddings = data['anchor_embeddings']
        if not anchor_embeddings: continue
        
        anchor_names = list(anchor_embeddings.keys())
        anchor_matrix = np.stack([anchor_embeddings[n] for n in anchor_names])
        sentence_to_review = data['sentence_to_review']

        for cand in data['scored_candidates']:
            emb = np.asarray(cand.embedding, dtype=np.float64).reshape(1, -1)
            sims = cosine_similarity(emb, anchor_matrix)[0]
            sorted_indices = np.argsort(sims)[::-1]
            
            rid = sentence_to_review.get(cand.sentence.strip(), 
                                        sentence_to_review.get(cand.sentence.lower().strip(), "unknown"))
            if rid not in all_rids: continue
            
            review_preds.setdefault(rid, {})
            
            found_count = 0
            for i in range(len(sorted_indices)):
                anchor = anchor_names[sorted_indices[i]]
                target_aspect = pm.get(anchor)
                
                if not target_aspect: continue
                
                if use_compact:
                    target_aspect = merge_map.get(target_aspect, target_aspect)
                
                review_preds[rid].setdefault(target_aspect, []).append(cand.sentiment)
                found_count += 1
                if found_count >= top_k:
                    break

    final_preds = {}
    for rid, asp_dict in review_preds.items():
        final_preds[rid] = {asp: np.mean(sents) for asp, sents in asp_dict.items()}

    true_remapped = {}
    for rid, labels in review_to_true.items():
        if rid not in all_rids: continue
        if labels is None:
            true_remapped[rid] = {}
            continue
        if use_compact:
            remapped = {}
            for asp, sent in labels.items():
                target = merge_map.get(asp, asp)
                remapped.setdefault(target, []).append(sent)
            true_remapped[rid] = {asp: np.mean(sents) for asp, sents in remapped.items()}
        else:
            true_remapped[rid] = labels

    return compute_all_metrics(final_preds, true_remapped, all_rids)

def get_star_baseline(df: pd.DataFrame, merge_map: Dict[str, str], use_compact: bool, slice_nm_ids: Optional[Set[int]]) -> Dict[str, Any]:
    review_to_true = {str(row['id']): row['true_labels_parsed'] for _, row in df.iterrows()}
    if slice_nm_ids:
        all_rids = {str(rid) for rid in df[df['nm_id'].isin(slice_nm_ids)]['id']}
    else:
        all_rids = set(review_to_true.keys())
        
    preds = {}
    true_remapped = {}
    
    for rid in all_rids:
        row = df[df['id'].astype(str) == rid].iloc[0]
        stars = float(row['rating'])
        t_dict = review_to_true.get(rid, {})
        if not t_dict: continue
        
        if use_compact:
            remapped = {}
            for asp, sent in t_dict.items():
                target = merge_map.get(asp, asp)
                remapped.setdefault(target, []).append(sent)
            t_final = {asp: np.mean(sents) for asp, sents in remapped.items()}
        else:
            t_final = t_dict
            
        true_remapped[rid] = t_final
        preds[rid] = {asp: stars for asp in t_final}
        
    metrics = compute_all_metrics(preds, true_remapped, all_rids)
    metrics['macro_p'] = 1.0
    metrics['macro_r'] = 1.0
    metrics['macro_f1'] = 1.0
    return metrics

def main():
    _ensure_utf8_stdout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs/grand_final_eval") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_paths = [
        "benchmark/eval_datasets/combined_benchmark.csv",
        "parser/razmetka/checked_reviews.csv",
        "parser/reviews_batches/merged_checked_reviews.csv"
    ]
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            dfs.append(load_markup(p))
    df = pd.concat(dfs).drop_duplicates(subset=['id'])
    
    with open("aspect_merge_map.json", "r", encoding="utf-8") as f:
        merge_map = json.load(f)
        
    print(f"Running pipeline for {df['nm_id'].nunique()} products...")
    raw_data = get_raw_pipeline_data(df, "aspect")
    
    from eval_pipeline import _build_auto_mapping
    sim_results = {}
    for nid, d in raw_data.items():
        sim_results[nid] = {
            'aspects': list(d['aspects_info'].keys()),
            'aspect_keywords': {n: info.keywords for n, info in d['aspects_info'].items()}
        }
    auto_mapping = _build_auto_mapping(sim_results, df, threshold=0.3)
    mapping = {nid: (MANUAL_MAPPING[nid] if nid in MANUAL_MAPPING else auto_mapping.get(nid, {})) 
               for nid in df['nm_id'].unique()}

    manual_nm_ids = set(MANUAL_MAPPING.keys()) & set(df['nm_id'].unique())
    slices = [
        ("Mixed/All", None),
        ("Manual-Only", manual_nm_ids)
    ]
    
    configs = [
        ("Original + Top-1", False, 1),
        ("Compact + Top-1", True, 1),
        ("Compact + Top-2", True, 2)
    ]
    
    results_rows = []
    
    for slice_name, slice_ids in slices:
        print(f"Evaluating slice: {slice_name}")
        for cfg_name, use_compact, top_k in configs:
            m = run_eval_for_config(raw_data, df, mapping, merge_map, use_compact, top_k, slice_ids)
            results_rows.append({'Slice': slice_name, 'Config': cfg_name, **m})
        
        b = get_star_baseline(df, merge_map, True, slice_ids)
        results_rows.append({'Slice': slice_name, 'Config': "Star Baseline", **b})

    res_df = pd.DataFrame(results_rows)
    res_df.to_csv(out_dir / "grand_final_metrics.csv", index=False)
    
    summary_md = ["# Grand Final Evaluation Report\n"]
    summary_md.append(f"Timestamp: {ts}\n")
    for slice_name in ["Mixed/All", "Manual-Only"]:
        summary_md.append(f"## Slicing: {slice_name}\n")
        slice_df = res_df[res_df['Slice'] == slice_name].drop(columns=['Slice'])
        summary_md.append(slice_df.to_markdown(index=False))
        summary_md.append("\n")
        
    (out_dir / "final_report.md").write_text("\n".join(summary_md), encoding="utf-8")
    print(f"Final evaluation finished. Results in {out_dir}")

if __name__ == "__main__":
    main()
