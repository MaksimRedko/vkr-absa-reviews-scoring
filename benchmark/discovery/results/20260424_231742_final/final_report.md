# Discovery v3 Final Report - All Three Metric Levels

## Aggregate statistics (filtered configuration only)

### Level 1 - Intrinsic metrics (без gold)
- Avg cohesion: 0.6692
- Avg separation: 0.6056
- Avg silhouette: 0.2185 (considering only k>=2 products)
- Avg concentration: 0.6326

### Level 2 - Semantic vs gold (threshold=0.65)
- Avg coverage: 74.7%
- Sensitivity (0.60 / 0.65 / 0.70): 87.4% / 74.7% / 59.5%
- Avg soft purity: 0.1706
- Novel aspects (automatic detection): 40

### Level 3 - Manual evaluation (86 clusters labeled)
- Total valid: 53 (61.6%)
  - valid_known: 21 (24.4%)
  - valid_novel: 32 (37.2%)
- mixed: 22 (25.6%)
- noise: 11 (12.8%)

## Per-category breakdown

| category | n_clusters | valid_rate | novel_rate | noise_rate |
|---|---:|---:|---:|---:|
| consumables | 6 | 50.0% | 33.3% | 16.7% |
| hospitality | 7 | 28.6% | 14.3% | 28.6% |
| physical_goods | 54 | 64.8% | 44.4% | 9.3% |
| services | 19 | 68.4% | 26.3% | 15.8% |

## Per-product breakdown (top-5 best, bottom-3 worst)

### Top-5 best

| nm_id | labeled | valid_rate | valid_novel | noise |
|---:|---:|---:|---:|---:|
| 9675256 | 5 | 80.0% | 3 | 0 |
| 619500952 | 14 | 78.6% | 8 | 0 |
| 1526918294 | 14 | 78.6% | 3 | 2 |
| 117808756 | 4 | 75.0% | 1 | 0 |
| 254445126 | 3 | 66.7% | 1 | 0 |

### Bottom-3 worst

| nm_id | labeled | valid_rate | valid_novel | noise |
|---:|---:|---:|---:|---:|
| 209269133 | 3 | 33.3% | 0 | 0 |
| 1809358565 | 2 | 0.0% | 0 | 1 |
| 1645864865 | 2 | 0.0% | 0 | 1 |

## Key findings
- Метод даёт valid clusters в 61.6% случаев (ручная оценка, n=86)
- Обнаружено 32 новых аспектов, не учтённых в разметке
- Лучшие категории: services, physical_goods
- Слабые категории: hospitality, consumables
