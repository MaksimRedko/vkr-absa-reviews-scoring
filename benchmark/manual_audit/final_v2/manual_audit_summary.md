# Manual audit queue summary

- Source run: `D:\diploma\results\final_res_v2`
- NLI rows total: 3446
- Passed relevance filter: 2606

## By aspect_source

| aspect_source | count |
|---|---:|
| vocab | 1393 |
| discovery | 1213 |

## By category

| category | count |
|---|---:|
| physical_goods | 1444 |
| services | 454 |
| hospitality | 418 |
| consumables | 290 |

## Exact auto-hit against gold aspect names

- Exact hits: 674
- Exact hit share among passed rows: 0.2586

## How to use

1. Start with `manual_audit_queue_sample.csv`.
2. Fill `manual_gold_aspect`, `manual_decision`, `manual_sentiment_decision`, `manual_error_type`, `manual_comment`.
3. Then continue with `manual_audit_queue_passed.csv` if time allows.
4. Use `manual_audit_queue_full.csv` only if you also want to inspect rows rejected by relevance filter.

Recommended `manual_decision` values: `TP`, `FP`, `UNCLEAR`, `DUPLICATE`, `OUT_OF_SCOPE`.
Recommended sentiment values: `OK`, `WRONG_POLARITY`, `TOO_HIGH`, `TOO_LOW`, `NOT_EVALUATED`.
