# 00 README STATE

## 1. Final state

- `final_main = sentiment mode A / full review baseline`.
- `B`, `C`, `D`, `D_weighted` kept only as sentiment ablation modes, not as final system.
- Manual audit validation is clean: `0/8 failed` (`8/8 PASS`).
- Main frozen package for handoff: `results/final_res_v1`.

## 2. Final architecture

1. Input: `data/dataset_final.csv`.
2. Aspect detection:
   - existing candidate extractor;
   - fixed hybrid vocabulary: core + domain sub-vocabularies;
   - fixed matching: `lexical_only`;
   - no segmentation-first final path;
   - no HDBSCAN/contextual clustering in final main path.
3. Sentiment:
   - final defended mode: full-review `mode A`;
   - localized `B/C/D/D_weighted` are benchmark ablations only.
4. Aggregation:
   - review-level aspect scores;
   - product aggregation downstream from frozen run artifacts.
5. Validation:
   - automatic run metrics from frozen artifacts;
   - separate manual audit layer from repaired SQLite/export.

## 3. Final artifacts

- Main frozen run: `results/final_res_v1`
- Non-final frozen alternative: `results/final_res_v2`
- Honest sentiment ablation comparison: `benchmark/sentiment/mode_abcd_diagnostics/results/20260502_114842`
- Sentiment mode summary: `benchmark/sentiment/reference_summary.md`
- Final manual audit export: `manual_recalc/exports/manual_metrics_20260504_191713`
- Manual audit validation report: `manual_recalc/exports/manual_metrics_20260504_191713/manual_audit_validation_report.md`
- Frozen reusable NLI cache: `cache/nli_global_frozen_fullrun_20260502.sqlite3`

## 4. Final metrics

### Automatic final main (`results/final_res_v1`)

- Detection Track A: precision `0.4767`, recall `0.4198`, F1 `0.4279`
- Review sentiment MAE: `0.8466`
- Review sentiment MAE round: `0.8005`
- Product MAE `n>=3`: `0.7841`

### Manual audit validation

- Validation status: `0/8 failed`, `8/8 PASS`

### Manual detection metrics

- precision strict: `0.5194`
- precision soft: `0.5649`
- recall: `0.6371`
- F1 strict: `0.5723`
- F1 soft: `0.5989`

### Manual sentiment metrics

- MAE: `0.9771`
- MAE round: `0.9496`
- Accuracy@1: `0.6232`
- wrong polarity: `0.1318`

## 5. Sentiment ablations, not final

- `A` is the defended final main mode.
- `B` has the best own-pairs MAE in the honest benchmark: `0.9256`.
- `C` has the best common-pairs MAE: `0.8960`.
- `D_weighted` is better than `D`, but still stays ablation-only.
- Final rule for handoff: these numbers are for comparison; they do not replace `final_main`.

## 6. Negative or partial experiments

- Segmentation branch: no stable gain; candidate baseline stayed reference.
- Localized sentiment experiment: review MAE improved (`0.7116 -> 0.6262`), but product MAE got worse (`0.8920 -> 0.9122`, `0.7528 -> 0.7595`). Verdict: `FAIL`.
- Residual HDBSCAN branch: killed. Top-20 clusters were mostly duplicates/noise; only `1` clearly useful new aspect.
- Contextual residual HDBSCAN: killed. One giant mixed cluster (`4993 / 8249` rows), only `1` useful small cluster.
- Discovery/clustering branch produced partial research signal, but was not accepted as final production path.

## 7. What should not be changed anymore

- Do not replace `final_main` with `B/C/D/D_weighted`.
- Do not reopen HDBSCAN/contextual clustering as the final architecture.
- Do not rewrite sentiment around localized evidence unless it is a separate new experiment branch.
- Do not add category classifier before a new explicit decision.
- Do not touch unit of analysis without isolated A/B.
- Do not rewrite stable pipeline parts just to unify old and new branches.

## 8. What remains before writing the thesis

- Write final VKR tables from frozen artifacts and repaired manual audit export.
- Keep one clear separation in the text:
  - defended final = `A` / full review baseline;
  - `B/C/D/D_weighted` = ablations.
- Add manual audit block with clean validation and final manual metrics.
- Add negative-results block for HDBSCAN, contextual clustering, and localized sentiment.
- Use the deep research report as the source for the analogs review section.

## 9. Key files

- Project roadmap: `NewRoadMap.txt`
- Active plan: `.opencode/plans/current.md`
- Compact state: `.opencode/plans/compact_context.md`
- Runtime config: `run_config.yaml`
- Pipeline entrypoint: `src/pipeline/run_traced_pipeline.py`
- Pipeline orchestrator: `src/pipeline/orchestrator.py`
- Final-freeze helper: `scripts/freeze_final_results.py`
- Manual audit recompute: `scripts/recompute_manual_audit_metrics.py`
- Manual audit UI: `manual_recalc/app.py`
- Vocabulary source log: `docs/vocabulary_reproducibility_log.md`
- Final defense notes: `final_defense_position.md`
- Analogs review source: deep research report for analogs review section; if delivered separately, keep it in the handoff bundle next to this file.
