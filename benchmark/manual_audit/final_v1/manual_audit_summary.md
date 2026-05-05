# Manual audit queue summary

- Source run: `D:\diploma\results\20260425_183110_traced`
- NLI rows total: 6211
- Passed relevance filter: 5363

## By aspect_source

| aspect_source | count |
|---|---:|
| discovery | 3715 |
| vocab | 1648 |

## By category

| category | count |
|---|---:|
| physical_goods | 3119 |
| services | 1070 |
| hospitality | 751 |
| consumables | 423 |

## Exact auto-hit against gold aspect names

- Exact hits: 741
- Exact hit share among passed rows: 0.1382

## How to use

1. Start with `manual_audit_queue_sample.csv`.
2. Fill `manual_gold_aspect`, `manual_decision`, `manual_sentiment_decision`, `manual_error_type`, `manual_comment`.
3. Then continue with `manual_audit_queue_passed.csv` if time allows.
4. Use `manual_audit_queue_full.csv` only if you also want to inspect rows rejected by relevance filter.

Recommended `manual_decision` values: `TP`, `FP`, `UNCLEAR`, `DUPLICATE`, `OUT_OF_SCOPE`.
Recommended sentiment values: `OK`, `WRONG_POLARITY`, `TOO_HIGH`, `TOO_LOW`, `NOT_EVALUATED`.
