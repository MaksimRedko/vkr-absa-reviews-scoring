# Routing Rules

## Scoring
- candidate unit: old candidates from current extractor, deduplicated by normalized candidate text inside each review
- general layer: anchors from `src/vocabulary/universal_aspects_v1.yaml`
- domain layer: anchors from `src/vocabulary/domain/<category>.yaml`
- score: cosine between candidate lemma embedding and anchor centroid built from canonical name + synonyms
- routing mode: `current`

## Thresholds
- `T_general = 0.880`
- `M_general = 0.040`
- `T_domain = 0.880`
- `M_domain = 0.040`
- `T_general_conflict = 0.830`
- `T_domain_conflict = 0.830`
- `C_overlap = 0.020`
- `weak_score_floor = 0.680`

## Routing
### `general`
- route to `general` if:
  - `best_general_score >= T_general`
  - `best_general_score - second_general_score >= M_general`
  - candidate is not noise
- save:
  - `chosen_layer = general`
  - `chosen_anchor = best_general_anchor`

### `domain`
- route to `domain` if:
  - candidate did not pass `general`
  - `best_domain_score >= T_domain`
  - `best_domain_score - second_domain_score >= M_domain`
  - candidate is not noise
- save:
  - `chosen_layer = domain`
  - `chosen_anchor = best_domain_anchor`

### `overlap`
- route to `overlap` if:
  - `best_general_score >= T_general_conflict`
  - `best_domain_score >= T_domain_conflict`
  - `abs(best_general_score - best_domain_score) <= C_overlap`
- in `current` mode: do not assign anchor automatically
- in `domain_priority` mode: assign `domain`, but only if the domain anchor also passes its main threshold and margin
- send row to `anchor_overlap.csv`

### `noise`
- route to `noise` if any of:
  - too short candidate
  - generic emotion / praise token without object
  - overly generic object word
  - technical garbage token
  - weak score on both layers: `max(general, domain) < weak_score_floor`

### `residual`
- route to `residual` if:
  - not `general`
  - not `domain`
  - not `overlap`
  - not `noise`

## Residual Cleaning For HDBSCAN
- start from `residual` only
- keep only residual candidate lemmas with global frequency `>= 2`
- do not send `noise`, `general`, `domain`, or `overlap` rows to HDBSCAN
