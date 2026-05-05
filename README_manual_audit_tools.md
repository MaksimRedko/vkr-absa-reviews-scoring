# Full manual audit tools

Я не смог скачать `nli_predictions.parquet` как бинарный файл через GitHub-коннектор: текстовый fetch режет parquet, а raw-download для ветки со слешем не сработал. Поэтому здесь готовый скрипт, который надо положить в репу и запустить локально.

## Что делает скрипт

Берёт:

- `results/20260425_183110_traced/nli_predictions.parquet`
- `results/20260425_183110_traced/clusters_*.json`
- `data/dataset_final.csv`
- `aspect_merge_map.json`

И создаёт:

- `manual_audit_queue_full.csv`
- `manual_audit_queue_passed.csv`
- `manual_audit_queue_sample.csv`
- `manual_audit_summary.md`

## Команда

```bash
python scripts/build_manual_audit_queue.py ^
  --run-dir results/20260425_183110_traced ^
  --dataset data/dataset_final.csv ^
  --merge-map aspect_merge_map.json ^
  --out-dir benchmark/manual_audit/final_v1
```

На Linux/macOS:

```bash
python scripts/build_manual_audit_queue.py \
  --run-dir results/20260425_183110_traced \
  --dataset data/dataset_final.csv \
  --merge-map aspect_merge_map.json \
  --out-dir benchmark/manual_audit/final_v1
```

## Что размечать руками

В первую очередь открывать:

`benchmark/manual_audit/final_v1/manual_audit_queue_sample.csv`

Заполнять колонки:

- `manual_gold_aspect`
- `manual_decision`
- `manual_sentiment_decision`
- `manual_error_type`
- `manual_comment`

Значения:

- `manual_decision`: `TP`, `FP`, `UNCLEAR`, `DUPLICATE`, `OUT_OF_SCOPE`
- `manual_sentiment_decision`: `OK`, `WRONG_POLARITY`, `TOO_HIGH`, `TOO_LOW`, `NOT_EVALUATED`

## Почему основной файл — nli_predictions.parquet

Потому что это фактический список aspect-review пар, по которым система реально запускала тональность. `candidate_matches.parquet` — более ранний слой; он полезен для объяснения происхождения vocab-аспекта, но не равен финальному списку оцениваемых аспектов.
