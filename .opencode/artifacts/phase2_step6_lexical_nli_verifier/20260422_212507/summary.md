# phase2_step6_lexical_only_nli_verifier

Status: **FAIL**

## Baseline reproduction
- baseline macro precision: 0.4806
- baseline macro recall: 0.4130
- baseline macro F1: 0.4251
- reference macro F1: 0.4251
- delta vs reference F1: +0.0000
- reproduction_ok: True

## Head-to-head
- verifier macro precision: 0.4806
- verifier macro recall: 0.4130
- verifier macro F1: 0.4251
- delta precision vs baseline: +0.0000
- delta recall vs baseline: +0.0000
- delta F1 vs baseline: +0.0000

## Threshold
- LOPO-selected threshold (median across folds): 0.0001
- thresholds by fold are saved in `lopo_thresholds.csv`

## Runtime
- latency_sec: 187.70
- NLI calls (candidate-aspect rows): 2215
- unique premise+hypothesis pairs: 2082

## 10 examples: verifier removed obvious FP
_none_

## 10 examples: verifier removed true positive
_none_

## Decision
- FAIL
- lexical_only remains final detection baseline.
- detection should not be tuned further in this branch.
