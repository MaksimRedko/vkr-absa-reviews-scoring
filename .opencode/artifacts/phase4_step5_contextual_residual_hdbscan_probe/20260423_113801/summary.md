# phase4_step5_contextual_residual_hdbscan_probe

## Setup
- input: repair_v1 residual_clean only
- embedding text: `cluster_text` = candidate context window, not `candidate_lemma`
- residual rows after context filter: 8249
- HDBSCAN min_cluster_size: 10
- HDBSCAN min_samples: 5
- fallback used: false
- clustered share: 0.6065

## Clusters
- total clusters: 2
- top-20 useful_new_aspect: 1
- top-20 duplicate_existing_anchor: 0
- top-20 too_mixed: 1
- top-20 noise_cluster: 0
- top-20 unclear: 0

## Verdict
- contextual residual after HDBSCAN looks like: mostly duplicates / mixed signal / noise
- kill_contextual_hdbscan_branch

## Interpretation
- context embeddings removed the pure one-word clustering failure, but produced one giant mixed cluster: `4993 / 8249` rows.
- only one small cluster looks useful: hospitality reviews with children / age suitability.
- the bottleneck is now embedding separability of short review windows, not exact-anchor leakage.
