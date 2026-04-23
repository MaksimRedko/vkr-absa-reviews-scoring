# phase3_step9_segment_to_aspect_semantic_matching

## A. Head-to-head
- baseline precision / recall / F1: 0.4806 / 0.4130 / 0.4251
- semantic precision / recall / F1: 0.1965 / 0.1748 / 0.1766
- delta precision / recall / F1: -0.2841 / -0.2382 / -0.2485
- avg segments per review: 3.2372
- avg predicted aspects per review: 0.6481

## B. Breakdown
### per-category
| group          | baseline_precision | baseline_recall | baseline_f1 | semantic_precision | semantic_recall | semantic_f1 | delta_f1 |
| -------------- | ------------------ | --------------- | ----------- | ------------------ | --------------- | ----------- | -------- |
| consumables    | 0.4088             | 0.2818          | 0.3213      | 0.2477             | 0.2546          | 0.2371      | -0.0842  |
| hospitality    | 0.6013             | 0.5951          | 0.5716      | 0.2157             | 0.1601          | 0.1732      | -0.3984  |
| physical_goods | 0.4858             | 0.4074          | 0.426       | 0.1885             | 0.1687          | 0.171       | -0.255   |
| services       | 0.4248             | 0.4002          | 0.3917      | 0.1771             | 0.1513          | 0.1571      | -0.2346  |

### per-source
| group  | baseline_precision | baseline_recall | baseline_f1 | semantic_precision | semantic_recall | semantic_f1 | delta_f1 |
| ------ | ------------------ | --------------- | ----------- | ------------------ | --------------- | ----------- | -------- |
| wb     | 0.4739             | 0.3879          | 0.4098      | 0.1977             | 0.182           | 0.1812      | -0.2286  |
| yandex | 0.4991             | 0.4822          | 0.4674      | 0.1934             | 0.155           | 0.1639      | -0.3035  |

## C. Diagnostics
- LOPO median threshold: 0.0321

### 10 successful cases
| review_id            | product_id | category    | source | aspect_id               | best_score | segment_text                                                                                                          |
| -------------------- | ---------- | ----------- | ------ | ----------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------- |
| 0KHkCQNmGFMeFrm66UQR | 506358703  | consumables | wb     | palatability_acceptance | 0.0372     | кот ест с удовольствием.                                                                                              |
| 00NhRiMR31BSX73L4WLz | 506358703  | consumables | wb     | palatability_acceptance | 0.0369     | Животные с удовольствием едят                                                                                         |
| TaCunSBY6xFWZDuZDioY | 506358703  | consumables | wb     | palatability_acceptance | 0.0369     | Кот ест с удовольствием!                                                                                              |
| D6H6AJ8ThNcnwrQd8jSa | 506358703  | consumables | wb     | palatability_acceptance | 0.0362     | Не первый раз беру этот корм, кошка с удовольствием ест                                                               |
| Vu38MXwVcl2C6bAL7xbX | 506358703  | consumables | wb     | palatability_acceptance | 0.0362     | Очень нравится нашему коту, ест с удовольствием.                                                                      |
| Ee7liVJY5zlbgxn2TFxs | 506358703  | consumables | wb     | palatability_acceptance | 0.0356     | Кот аллергик и привереда - ест с удовольствием, аллергии нет                                                          |
| pYG71Gk0JMA9xgWv0bXJ | 506358703  | consumables | wb     | palatability_acceptance | 0.0356     | После еды бодр и весел                                                                                                |
| BV6WDSe1BmpVo17cTkod | 506358703  | consumables | wb     | palatability_acceptance | 0.0355     | Моя кошка кушает с удовольствием!                                                                                     |
| PmPKyC25iyuFn38Jw9XJ | 506358703  | consumables | wb     | palatability_acceptance | 0.0353     | Достоинства: Кот ест с удовольствием. Ему очень нравится.                                                             |
| yPqH7sYxF02rLpGzBVEi | 506358703  | consumables | wb     | palatability_acceptance | 0.0353     | Постепенно перевожу своего полного котика на этот корм, ему всё нравится и ест с удовольствием, не выплёвывает зёрна. |

### 10 bad cases
| review_id            | product_id | category       | source | aspect_id       | best_score | segment_text                                                                                |
| -------------------- | ---------- | -------------- | ------ | --------------- | ---------- | ------------------------------------------------------------------------------------------- |
| yaILJIwBwX70cIf6QHvA | 15430704   | consumables    | wb     | value_for_money | 0.0368     | По соотношению цена-качество твердая 5.                                                     |
| gpca64sBGgNwMUDvfCoy | 15430704   | consumables    | wb     | value_for_money | 0.0367     | Деньги списываете до получения товара. Верните деньги.                                      |
| DbZNvcF6hRWlWQJgLoII | 506358703  | consumables    | wb     | recommendation  | 0.0366     | Поэтому всем рекомендую.                                                                    |
| Vfr8aYYBT-IFIyX_plBV | 15430704   | consumables    | wb     | recommendation  | 0.0362     | С удовольствием рекомендую к покупке.                                                       |
| hH4lhz5bKuTwPyqWt8M3 | 619500952  | physical_goods | wb     | ease_of_use     | 0.0359     | Для того, чтобы средство работало, надо постоянно его использовать.                         |
| z7PHKYsBwbGU35ygZ1Kc | 15430704   | consumables    | wb     | value_for_money | 0.0359     | а значит и трудозатрат меньше и стоимость должна быть меньше,                               |
| 9MaGD7ufhUe5ZkcNISwK | 506358703  | consumables    | wb     | value_for_money | 0.0358     | Достоинства: Качество и цена                                                                |
| gargHYzFfDKowBFCUGqc | 311233470  | physical_goods | wb     | value_for_money | 0.0356     | Стыд за такое качество брать такие деньги 🫰                                                 |
| i26HfIoBYE-HFOZ5q9aV | 15430704   | consumables    | wb     | value_for_money | 0.0355     | В нашем городе такая в магазинах дороже. Качество отличное                                  |
| Op3bRogBUYwa0QEfy5Xj | 54581151   | physical_goods | wb     | value_for_money | 0.0353     | Но за такие деньги (1100₽) это лучшая колода в моей коллекции по соотношению цена/качество! |

## D. Decision
- FAIL