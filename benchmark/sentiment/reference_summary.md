# Sentiment Reference Summary

Final sentiment-only comparison on frozen traced artifacts (`results/20260425_183110_traced`).

| mode | review_mae | review_mae_round | vocab_pair_mae | discovery_pair_mae | evaluable_pair_coverage | kept_after_threshold | runtime_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.9770 | 0.9381 | 0.9089 | 1.0686 | 0.4489 | 5363 | 1145.01 |
| B | 0.8878 | 0.8702 | 0.8466 | 0.9611 | 0.3163 | 2558 | 725.47 |
| C | 0.8934 | 0.8749 | 0.8561 | 0.9262 | 0.3181 | 2606 | 271.05 |
| D | 0.9014 | 0.8834 | 0.8573 | 1.0305 | 0.3483 | 3578 | 975.11 |
| D_weighted | 0.8892 | 0.8707 | 0.8456 | 1.0223 | 0.3483 | 3578 | 1084.28 |

Verdict:
- `D_weighted` beats `D` on `review_mae`, `review_mae_round`, and `vocab_pair_mae` with identical coverage.
- `B` stays the reference mode.
- Reason: `B` still has the best primary `review_mae`, slightly better `review_mae_round`, and much better `discovery_pair_mae` than `D_weighted`.
- Use `D_weighted` only as the higher-coverage fallback among localized-evidence modes.
