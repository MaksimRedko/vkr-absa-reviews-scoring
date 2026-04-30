# Final End-to-End Pipeline Results

## Aggregate metrics across 16 products

| Metric | Vocab Only | Vocab + Discovery | Star Baseline |
|---|---:|---:|---:|
| Detection Precision | 0.4767 | 0.5698 | N/A |
| Detection Recall | 0.4198 | 0.4545 | N/A |
| Detection F1 | 0.4279 | 0.4847 | N/A |
| Sentiment MAE (review) | 0.8466 | 0.9250 | 0.6398 |
| Sentiment MAE (round) | 0.8005 | 0.8856 | 0.6398 |
| Product MAE (n>=3) | 0.7841 | 0.9140 | 0.5503 |

## Per-category breakdown

| Category | Detection P | Detection R | Review MAE | Product MAE n>=3 |
|---|---:|---:|---:|---:|
| consumables | 0.3987 | 0.2657 | 0.5342 | 0.7234 |
| hospitality | 0.7450 | 0.6839 | 1.0387 | 0.8558 |
| physical_goods | 0.5174 | 0.4084 | 0.8998 | 0.8321 |
| services | 0.6958 | 0.5651 | 0.9454 | 0.9574 |

## Comparison: did discovery help?

Discovery added 151 matched product-aspect rows.
Sentiment quality on discovery aspects: MAE = 1.0686 vs 0.8274 on vocabulary aspects.

## Comparison: did we beat star baseline?

On Sentiment MAE (review-level): no by 0.2853.
On Product MAE: no by 0.3637.

## Sanity check vs old baseline

Old code (commit faad23a):
- Vocab-only sentiment MAE review: 0.7116 (reference)

New code (this run):
- Vocab-only sentiment MAE review: 0.8466
- Difference: 0.8466 - 0.7116 = 0.1350
- Status: regression: inspect before accepting; outside checked range.

## Negation correction stats

Total predictions: 5363
Corrections applied: 31 (0.5780%)
Avg MAE before correction: 0.9943
Avg MAE after correction: 0.9563
Improvement: 0.0379
Inversion rate: 5.6466%
Correction target status: below broad target

Per-category corrections:
- hospitality: 1 corrections
- physical_goods: 28 corrections
- services: 2 corrections

## Negation sanity check

- Vocab-only sentiment MAE: 0.8466 (expected 0.72-0.85)
- Consumables MAE: 0.5342 (expected <0.50)
- Inversion rate: 5.6466% (expected <12%)
- Corrections applied: 31 (target 50-150; hard lower check is >=30)

## Hard cases (10 worst predictions)

| review_id | nm_id | aspect | source | gold | pred | abs_error |
|---|---:|---|---|---:|---:|---:|
| uHKxMyTzD3UaS9xdeJgN | 165234215 | Вес | discovery | 5.0000 | 1.0005 | 3.9995 |
| ym_6bac9620e99ec487_0028 | 1526918294 | Регистрация | discovery | 5.0000 | 1.0015 | 3.9985 |
| ym_dcfc1c3c2862c040_0089 | 1809358565 | Раздевалка | discovery | 1.0000 | 4.9969 | 3.9969 |
| ym_c59de42a475ceefe_0025 | 1504973191 | Еда | discovery | 5.0000 | 1.0059 | 3.9941 |
| CQRlT46poS5ecSeFMEnP | 254445126 | Качество | vocab | 5.0000 | 1.0130 | 3.9870 |
| kgAE6XRsFS4Az0DkwUPz | 165234215 | Комплектация | discovery | 5.0000 | 1.0130 | 3.9870 |
| UOWfXJraKU9GMdxsg6UZ | 9675256 | Фактура | discovery | 5.0000 | 1.0214 | 3.9786 |
| ym_6bac9620e99ec487_0047 | 1526918294 | Цена | vocab | 5.0000 | 1.0219 | 3.9781 |
| JePqX6gyQWKcgndTZFGn | 209269133 | Корпус | discovery | 1.0000 | 4.9732 | 3.9732 |
| JePqX6gyQWKcgndTZFGn | 209269133 | Часы | discovery | 1.0000 | 4.9732 | 3.9732 |
