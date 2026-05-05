# Final End-to-End Pipeline Results

## Aggregate metrics across 16 products

| Metric | Vocab Only | Vocab + Discovery | Star Baseline |
|---|---:|---:|---:|
| Detection Precision | 0.4767 | 0.5698 | N/A |
| Detection Recall | 0.4198 | 0.4545 | N/A |
| Detection F1 | 0.4279 | 0.4847 | N/A |
| Sentiment MAE (review) | 0.9565 | 0.9770 | 0.6398 |
| Sentiment MAE (round) | 0.9103 | 0.9381 | 0.6398 |
| Product MAE (n>=3) | 0.8218 | 0.9343 | 0.5503 |

## Per-category breakdown

| Category | Detection P | Detection R | Review MAE | Product MAE n>=3 |
|---|---:|---:|---:|---:|
| consumables | 0.3987 | 0.2657 | 0.5342 | 0.7234 |
| hospitality | 0.7450 | 0.6839 | 1.0366 | 0.8528 |
| physical_goods | 0.5174 | 0.4084 | 1.0253 | 0.8953 |
| services | 0.6958 | 0.5651 | 0.9454 | 0.9591 |

## Comparison: did discovery help?

Discovery added 151 matched product-aspect rows.
Sentiment quality on discovery aspects: MAE = 1.0686 vs 0.9089 on vocabulary aspects.

## Comparison: did we beat star baseline?

On Sentiment MAE (review-level): no by 0.3372.
On Product MAE: no by 0.3840.

## Sanity check vs old baseline

Old code (commit faad23a):
- Vocab-only sentiment MAE review: 0.7116 (reference)

New code (this run):
- Vocab-only sentiment MAE review: 0.9565
- Difference: 0.9565 - 0.7116 = 0.2449
- Status: regression: inspect before accepting; outside checked range.

## Hard cases (10 worst predictions)

| review_id | nm_id | aspect | source | gold | pred | abs_error |
|---|---:|---|---|---:|---:|---:|
| uHKxMyTzD3UaS9xdeJgN | 165234215 | Вес | discovery | 5.0000 | 1.0005 | 3.9995 |
| ym_6bac9620e99ec487_0028 | 1526918294 | Регистрация | discovery | 5.0000 | 1.0015 | 3.9985 |
| 6JTyUVKALVU7Qrj0hz7W | 619500952 | Запах | vocab | 5.0000 | 1.0017 | 3.9983 |
| ym_dcfc1c3c2862c040_0089 | 1809358565 | Раздевалка | discovery | 1.0000 | 4.9969 | 3.9969 |
| DqXfgYQBeLZkh6kUZ-C3 | 117808756 | Запах | vocab | 5.0000 | 1.0056 | 3.9944 |
| ym_c59de42a475ceefe_0025 | 1504973191 | Еда | discovery | 5.0000 | 1.0059 | 3.9941 |
| 4S4LthtITjFQ5KOMEkJb | 619500952 | Запах | vocab | 5.0000 | 1.0061 | 3.9939 |
| Oj0HyENioo8VFcQJ9q5c | 619500952 | Запах | vocab | 5.0000 | 1.0106 | 3.9894 |
| 44SZXsQ0O5j3d4J0e8iv | 619500952 | Запах | vocab | 5.0000 | 1.0109 | 3.9891 |
| 5jKzT2VTSDnQ8WTMleEQ | 619500952 | Запах | vocab | 5.0000 | 1.0128 | 3.9872 |
