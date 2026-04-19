# Eval audit summary

## Run metadata

- **timestamp**: 20260419_162256
- **csv_path**: benchmark\eval_datasets\combined_benchmark.csv
- **mapping_mode**: mixed
- **mapping_cli**: mixed
- **products_without_manual_mapping**: 8
- **nm_ids_without_manual**: [9675256, 165234215, 209269133, 1504973191, 1526918294, 1645864865, 1733494834, 1809358565]
- **nm_ids_used_auto**: [9675256, 165234215, 209269133, 1504973191, 1526918294, 1645864865, 1733494834, 1809358565]
- **reviews_dropped_due_to_mapping**: 0
- **pred_aspects_unmapped_count**: 0
- **clusterer**: aspect
- **seed**: 42

### Caveat (auto / mixed mapping)

Метрики, зависящие от автоматического pred→true, **не являются primary-интерпретацией** без ручной проверки таблицы маппинга.

## Primary

### Review-level detection (macro over reviews)

- precision: **0.2472**
- recall: **0.2464**
- F1: **0.2307**

### Review-level sentiment (intersection only)

- MAE continuous (macro over reviews): **0.8311**
- MAE rounded (macro over reviews): **0.8088**
- n_matched_pairs: 821

### Product-level (matched subsets)

- MAE matched all: **1.0682290276372652**
- MAE matched n_true≥3: **0.9504030836441071**
- MAE rounded matched all: **1.0698101727623974**
- MAE rounded n_true≥3: **0.9458264876013384**

### Baseline comparison (same metrics)

| metric | model | star | neutral | product_mean |
| --- | --- | --- | --- | --- |
| review_sentiment_mae_continuous_macro | 0.8311 | 0.7465 | 1.7272 | 1.6198 |
| review_sentiment_mae_rounded_macro | 0.8088 | 0.7465 | 1.7272 | 1.6255 |
| product_mae_matched_all | 1.0682290276372652 | 1.0395750801093946 | 1.307967975493046 | 1.3101416384757127 |
| product_mae_matched_n_true_ge_3 | 0.9504030836441071 | 0.9576694821264665 | 1.2700329337384424 | 1.2531469417387382 |

## Secondary / diagnostic

### Mapping metadata

- mapping_mode: mixed
- products_without_manual_mapping: 8
- reviews_dropped_due_to_mapping: 0
- pred_aspects_unmapped_count: 0

### Micro detection

- micro P/R/F1: 0.3148 / 0.2874 / 0.3005

### Coverage

- reviews_with_no_candidates: 0
- reviews_with_candidates_but_no_pred_aspects: 239
- reviews_with_wrong_aspect_only: 242
- reviews_with_at_least_one_correct_aspect: 476

### Legacy eval_pipeline metrics (diagnostic)

- global_mae_raw: 0.855
- global_mae_calibrated: 1.117
- micro_precision / micro_recall: 1.0 / 0.43
- global_mention_recall_review: 0.287

### Files

- `detection_funnel.csv`, `confusion_matrix.csv`, `top_confusions.md`