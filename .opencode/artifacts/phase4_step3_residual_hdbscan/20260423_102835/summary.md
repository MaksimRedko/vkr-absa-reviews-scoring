# phase4_step3_residual_only_hdbscan

## Setup
- routing mode: `domain_priority`
- residual_clean rows: `10253`
- base run: `min_cluster_size=15`, `min_samples=7`, `176` clusters, clustered share `0.6463`
- fallback run used: `min_cluster_size=10`, `min_samples=5`, `285` clusters, clustered share `0.7314`

## Top-20 manual labels
- useful_new_aspect: `1`
- duplicate_existing_anchor: `7`
- noise_cluster: `10`
- too_mixed: `1`
- unclear: `1`

## Reading
- Top clusters are dominated by anchor duplicates (`качество`, `размер`, `внешний вид`, `продавец`, location/infrastructure variants) and by residual garbage (`раз`, `уже`, `после`, `при`, `годы`).
- Only one top-20 cluster looks clearly worth keeping as a new aspect candidate: `горки / аттракционы` in hospitality.
- Residual after HDBSCAN looks more like a source of noise and duplicate anchor mass than a reliable source of new aspects.

## Verdict
- `kill_hdbscan_branch`
