# Dataset Validation Report

- generated_at: 2026-05-05T08:56:51.046790+00:00
- run_dir: D:\diploma\results\20260502_171530_traced
- TP pairs found: 2776
- rows in calibration dataset: 2776
- matched with NLI probabilities: 2776

## Lost Rows

- no row loss after TP + NLI join

## Duplicates / Missing

- duplicate prediction_id rows: 0
- missing gold_rating: 0
- missing current_final_rating: 0

## Probability Fields

| field | non_null | null |
|---|---:|---:|
| pos_entailment | 2776 | 0 |
| pos_neutral | 2776 | 0 |
| pos_contradiction | 2776 | 0 |
| neg_entailment | 0 | 2776 |
| neg_neutral | 0 | 2776 |
| neg_contradiction | 0 | 2776 |

## NLI Layout

- polarity counts in raw nli_predictions: {'pos': 6224}
- rows with unknown hypothesis polarity: 0
- prediction_ids with multiple NLI rows: 0

## Duplicate Sample

No duplicate prediction_id rows.

