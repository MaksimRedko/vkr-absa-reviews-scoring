# Cache Workflow

## Что есть сейчас

В проекте теперь 2 независимых слоя кэша:

1. `stage_cache` для `s1-s4`
2. `persistent_nli_cache` для `s5`

Они не меняют алгоритмы.
Они только убирают повторный пересчёт.

## Где настраивается

В `run_config.yaml`:

```yaml
stage_cache:
  enabled: true
  root_dir: cache/pipeline_stages

sentiment:
  persistent_nli_cache_enabled: true
  persistent_nli_cache_path: cache/nli_global.sqlite3
```

## Что кэшируется по стадиям

### `s1` extraction

Ключ:
- хэш `gold_dataset_csv`
- `limit_products`
- `extraction.*`
- хэш кода `src/pipeline/stages/s1_extraction.py`
- хэш кода `scripts/run_phase2_baseline_matching.py`

Артефакт:
- `candidates.parquet`

### `s2` encoding

Ключ:
- хэш `candidates.parquet`
- хэш `core_vocab`
- хэш `domain_vocab_dir`
- имя encoder-модели
- `encoder_batch_size`
- хэш кода `src/pipeline/stages/s2_encoding.py`
- хэш кода `src/discovery/encoder.py`

Артефакты:
- `embeddings_candidates.npy`
- `embedding_index_candidates.csv`
- `embeddings_vocab.npy`
- `embedding_index_vocab.csv`

### `s3` vocab matching

Ключ:
- хэш `candidates.parquet`
- хэши `s2`-артефактов
- `matching.*`
- хэш кода `src/pipeline/stages/s3_vocab_matching.py`
- хэш кода `scripts/run_phase2_baseline_matching.py`

Артефакт:
- `candidate_matches.parquet`

### `s4` discovery binding

Ключ:
- хэш `candidates.parquet`
- хэш `candidate_matches.parquet`
- хэш `discovery_dir`
- имя encoder-модели
- `encoder_batch_size`
- `phrase_to_cluster_threshold`
- хэш кода `src/pipeline/stages/s4_discovery.py`
- хэш кода `src/discovery/encoder.py`
- хэш кода `benchmark/end_to_end/run_final_pipeline.py`

Артефакты:
- `clusters_<nm_id>.json`
- `cluster_centroids_<nm_id>.npy`
- `discovery_candidate_bindings.parquet`
- `aspect_review_assignments.parquet`
- `aspect_review_evidence.parquet`

### `s5` sentiment

Ключ:
- `model_signature`
- `premise`
- `hypothesis`

Хранилище:
- sqlite

Значение:
- raw logits NLI

Артефакт полного run:
- `nli_predictions.parquet`

## Как это работает

### Cold run

Если подходящего ключа нет:
- стадия считается
- её артефакты пишутся в `out_dir`
- копия кладётся в cache root

### Warm run

Если ключ совпал:
- артефакты копируются из cache root в новый `out_dir`
- стадия не пересчитывается

## Где смотреть hit/miss

После traced run:
- `run_summary.json`
- `MANIFEST.json`

Там есть:
- `stage_cache`
- `nli_cache`

Пример:

```json
"stage_cache": {
  "s1": {"hit": true, "fingerprint": "..."},
  "s2": {"hit": true, "fingerprint": "..."},
  "s3": {"hit": true, "fingerprint": "..."},
  "s4": {"hit": true, "fingerprint": "..."}
}
```

## Как запускать

Полный traced run:

```bash
python -m src.pipeline.run_traced_pipeline --config run_config.yaml
```

Smoke:

```bash
python -m src.pipeline.run_traced_pipeline --config run_config.yaml --limit-products 1
```

## Как принудительно сделать cold run

Варианты:

1. удалить `stage_cache.root_dir`
2. удалить `sentiment.persistent_nli_cache_path`
3. указать новый путь для одного из кэшей
4. временно выключить:

```yaml
stage_cache:
  enabled: false

sentiment:
  persistent_nli_cache_enabled: false
```

## Как замораживать рабочий кэш

Если нужен отдельный frozen-кэш под конкретный эксперимент:

1. укажи отдельный sqlite путь в конфиге
2. сделай cold full run
3. сделай второй warm run
4. проверь:
   - `persistent_hits > 0`
   - `misses = 0` на втором run
   - метрики и хэши артефактов совпали
5. после этого не удаляй этот файл

Текущий проверенный frozen NLI cache baseline A:
- `cache/nli_global_frozen_fullrun_20260502.sqlite3`

## Практическое правило

Если меняется только sentiment:
- не надо трогать `s1-s4`
- либо используй уже готовый traced run
- либо используй `stage_cache` + `persistent_nli_cache`

Если меняется только aggregation:
- не надо трогать `s1-s5`

Если меняется extraction / matching / discovery:
- старые stage-cache keys сами перестанут совпадать

## Что не надо делать

- не смешивать несколько разных экспериментов в одном вручную переименованном sqlite без причины
- не подменять артефакты в cache root руками
- не сравнивать A/B sentiment, если у них разный `aspect_review_assignments`

## Коротко

- `stage_cache` ускоряет `s1-s4`
- `persistent_nli_cache` ускоряет `s5`
- traced run остаётся self-contained
- честность A/B не ломается, потому что кэш не меняет состав пар
