# phase4_step4_residual_repair_diagnostic

## Verdict
- recommendation: `kill_residual_branch`
- HDBSCAN was not run.

## Route Counts
| stage | general | domain | overlap | residual | noise |
|---|---:|---:|---:|---:|---:|
| baseline | 956 | 2362 | 3110 | 17642 | 787 |
| repair_v1 | 1805 | 2395 | 3089 | 16115 | 1453 |

## Residual Quality
| metric | baseline | repair_v1 |
|---|---:|---:|
| residual raw | 17642 | 16115 |
| residual clean | 10253 | 8739 |
| clean unique lemmas | 1593 | 1543 |
| exact anchor leakage clean | 807 | 0 |
| exact anchor leakage clean share | 7.9% | 0.0% |
| single-token clean share | 78.1% | 74.7% |
| bad terms in top-10 | 7 | 0 |

## Residual Sample After
- sample size: 120
- looks_useful: 84
- unclear: 36
- looks_noise: 0

## Interpretation
- exact anchor leakage fixed: true
- bad top terms removed: true
- single-token clean share delta: 3.4%
- repair top-10 residual terms: ребёнок, продавец, горка, аквапарк, деньга, платье, час, таракан, целое, тот
- residual is cleaner, but still mostly context-free one-token object/entity lemmas
- total candidates compared: 24857
