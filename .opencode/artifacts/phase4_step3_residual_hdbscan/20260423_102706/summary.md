# phase4_step3_residual_only_hdbscan

## Setup
- routing mode: `domain_priority`
- residual_clean rows: 10253
- HDBSCAN min_cluster_size: 15
- HDBSCAN min_samples: 7
- clustered share: 0.6463

## Clusters
- total clusters: 176
- top-20 useful_new_aspect: 1
- top-20 duplicate_existing_anchor: 18
- top-20 too_mixed: 1
- top-20 noise_cluster: 0
- top-20 unclear: 0

## Verdict
- residual after HDBSCAN looks like: mostly duplicates / mixed signal / noise
- kill_hdbscan_branch
