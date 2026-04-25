# A/B: v2 (no filter) vs v3 (filtered)

## Aggregate metrics

| Metric | No Filter | Filtered | Delta |
|---|---:|---:|---:|
| Avg unique phrases | 743.1250 | 611.8125 | -131.3125 |
| Avg n_clusters | 8.1875 | 5.3750 | -2.8125 |
| Avg noise_rate | 0.6125 | 0.5366 | -0.0759 |
| Avg cohesion | 0.6994 | 0.6692 | -0.0302 |
| Avg separation | 0.5817 | 0.6056 | +0.0238 |
| Avg silhouette | 0.2327 | 0.2185 | -0.0143 |
| Coverage@0.65 | 0.6600 | 0.7466 | +0.0866 |
| Avg soft_purity | 0.1499 | 0.1706 | +0.0207 |
| Novel aspects total | 70.0000 | 40.0000 | -30.0000 |

## Per-product winners

| nm_id | winner | wins_filtered | wins_no_filter |
|---:|---|---:|---:|
| 9675256 | filtered | 4 | 3 |
| 15430704 | no_filter | 1 | 4 |
| 54581151 | tie | 3 | 3 |
| 117808756 | filtered | 5 | 1 |
| 165234215 | filtered | 4 | 3 |
| 209269133 | no_filter | 3 | 4 |
| 254445126 | no_filter | 3 | 4 |
| 311233470 | no_filter | 3 | 4 |
| 441378025 | filtered | 4 | 1 |
| 506358703 | no_filter | 2 | 3 |
| 619500952 | filtered | 5 | 2 |
| 1504973191 | filtered | 4 | 3 |
| 1526918294 | no_filter | 2 | 5 |
| 1645864865 | no_filter | 3 | 4 |
| 1733494834 | filtered | 4 | 1 |
| 1809358565 | no_filter | 1 | 6 |

## Verdict

ФИЛЬТР ВРЕДИТ, НЕ использовать
