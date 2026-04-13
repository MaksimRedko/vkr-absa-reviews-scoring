# Architecture v2 — ABSA Pipeline (дипломный проект)

Файл в репозитории: **`Architecture v2.md`** (пробел в имени; не путать с `Architecture_v2.md`).

Документ описывает **фактическую** архитектуру и поток данных по состоянию кода: оркестратор `ABSAPipeline`, стадии discovery / sentiment / math, опциональные снимки, eval-ветку и точки расширения.

---

## 1. Назначение системы

Конвейер превращает **сырые отзывы** (поля pros / cons / full_text) в:

1. **Аспекты товара** — именованные темы («Логистика», «Качество», …) с наборами ключевых спанов.
2. **Сентимент по парам** (предложение × аспект) — шкала через NLI-модель.
3. **Итоговые метрики по аспекту** — байесовская агрегация с учётом доверия к отзыву и времени.

Это не «семь абстрактных стадий», а цепочка **конкретных типов данных**, **файлов конфигурации**, **опциональных артефактов на диске** и **мест, где логику можно подменить**.

---

## 2. Карта модулей (что за что отвечает)

| Модуль | Роль |
|--------|------|
| `configs/configs.py` | Глобальный `OmegaConf` `config`: пути моделей, пороги discovery / sentiment / math / fraud |
| `src/schemas/models.py` | `ReviewInput` — единая схема отзыва, свойство `clean_text` |
| `src/data/loader.py` | `DataLoader` — чтение SQLite `data/dataset.db`, таблица `reviews` |
| `src/pipeline.py` | `ABSAPipeline` — оркестрация, `_build_sentiment_pairs`, `_build_aggregation_input`, опциональные снимки |
| `src/stages.py` | ABC стадий (`FraudStage`, …) и типовые алиасы `SentimentPair`, `AggregationInput` |
| `src/snapshots.py` | `SnapshotWriter` + `load_*` для записи/чтения промежуточных JSON |
| `src/discovery/candidates.py` | `CandidateExtractor`, `Candidate` |
| `src/discovery/scorer.py` | `KeyBERTScorer`, `ScoredCandidate` |
| `src/discovery/clusterer.py` | `AspectClusterer`, `AspectInfo`, якоря `MACRO_ANCHORS` / `ANTI_ANCHORS` |
| `src/fraud/engine.py` | `AntiFraudEngine` — веса доверия (**свой** экземпляр `SentenceTransformer`, см. §6.1) |
| `src/sentiment/engine.py` | `SentimentEngine` — dual-hypothesis NLI |
| `src/math/engine.py` | `RatingMathEngine` — агрегация, Ledoit-Wolf |
| `eval_pipeline.py` | Оценка против разметки: свой сбор модулей, **без** `RatingMathEngine` и **без** `ABSAPipeline` |
| `scripts/run_pipeline_merged.py` | Полный `ABSAPipeline` по CSV |
| `experiments/experiment_manager.py` | Папки прогонов, `snapshot_writer()` |

### 2.1. Вспомогательные и диагностические скрипты

Не входят в ядро ABSA, но используются для прогонов и анализа:

| Путь | Назначение |
|------|------------|
| `scripts/run_merged_suite.py` | Оркестрация нескольких команд (merged pipeline, eval, diag) подряд |
| `diag_coverage.py`, `diag_loss_funnel.py`, `diag_recall.py` | Аналитика поверх уже посчитанных CSV/JSON eval |
| `manual_eval_holdout.py` | Ручная оценка / precision на holdout |
| `run_eval_config.py` | Запуск eval с JSON-конфигом и логированием в `experiments/runs/...` |
| `experiments/compare_runs.py` | Сравнение результатов прогонов |

При изменении логики пайплайна проверяйте, что эти скрипты по-прежнему указывают на актуальные пути к артефактам.

---

## 3. Конфигурация (`configs/configs.py`)

Все перечисленные параметры читаются модулями напрямую через `from configs.configs import config`.

### 3.1. `config.models`

| Ключ | Назначение |
|------|------------|
| `encoder_path` | Локальный путь к **одной и той же** архитектуре rubert-tiny2. Из этого пути в рантайме создаются **несколько независимых** экземпляров `SentenceTransformer` (см. §6.1 и §11). |
| `nli_path` | Локальный путь к `AutoModelForSequenceClassification` + tokenizer (NLI, `local_files_only=True`) |

**Важно:** в `ABSAPipeline` энкодер `self._encoder` используется для `KeyBERTScorer` и `AspectClusterer`. `AntiFraudEngine` в своём `__init__` создаёт **отдельный** `SentenceTransformer(config.models.encoder_path)` — общая весовая матрица на диске одна, в RAM загружаются два объекта модели (удвоение памяти и времени инициализации относительно одного экземпляра).

### 3.2. `config.discovery`

| Ключ | Где используется |
|------|------------------|
| `ngram_range` | `CandidateExtractor` — диапазон n-грамм (например `[1, 2]`) |
| `min_word_length` | Минимальная длина токена в n-грамме |
| `keybert_top_k` | После косинусного скоринга — сколько кандидатов оставить **до** MMR (на предложение) |
| `cosine_threshold` | Порог сходства span ↔ предложение |
| `mmr_lambda`, `mmr_top_k` | MMR-диверсификация отобранных кандидатов |
| `umap_*`, `hdbscan_min_samples` | Residual-кластеризация в `AspectClusterer` |
| `anchor_similarity_threshold` | Margin в `_name_cluster` для residual (best − second) |
| `anti_anchor_threshold` | Отсев «мусорных» спанов относительно anti-якорей |
| `cluster_merge_threshold` | Слияние близких кластеров в UMAP-пространстве |
| `multi_label_threshold`, `multi_label_max_aspects` | Построение NLI-пар: порог cos(span, якорь) и максимум якорей на один scored-кандидат |

**Фактические значения в репозитории (на момент синхронизации с кодом):** `multi_label_threshold = 0.4`, `multi_label_max_aspects = 3`.

Дополнительно в коде кластеризатора (если **нет** в OmegaConf-объекте) подставляются дефолты:

- `anchor_assign_threshold` → `0.55`
- `anchor_assign_margin` → `0.03`

(через `getattr(config.discovery, "anchor_assign_threshold", 0.55)` и аналог для margin).

### 3.3. `config.sentiment`

| Ключ | Назначение |
|------|------------|
| `hypothesis_template_pos`, `hypothesis_template_neg` | Шаблоны гипотез для NLI; в Python: `template.format(aspect=...)` — плейсхолдер **`{aspect}`** |
| `batch_size` | Размер батча NLI |
| `score_epsilon` | Знаменатель в формуле скора (стабильность) |

**Текущие строки в `configs/configs.py` (вариант B, комментарий в коде):**

- `hypothesis_template_pos`: `"{aspect} — это хорошо"`
- `hypothesis_template_neg`: `"{aspect} — это плохо"`

### 3.4. `config.math`

| Ключ | Назначение |
|------|------------|
| `prior_mean` | Байесовский априор (нейтральная оценка); в репозитории: **`3.0`** |
| `prior_strength_max` | Верхняя граница силы априора `C` в формуле `C = min(median_mentions, prior_strength_max)` |
| `time_decay_days` | Параметр экспоненциального затухания по дате отзыва |
| `variance_penalty` | В текущем конфиге `0.0` — дисперсия не вычитается из score |

**Фактическое значение в репозитории:** `prior_strength_max = 1`. Это **сильно** влияет на степень притягивания оценок к `prior_mean` по сравнению с большим `C_max` (например 3): при `C_max = 1` априор тяжелее при малом эффективном числе наблюдений.

### 3.5. `config.fraud`

| Ключ | Назначение |
|------|------------|
| `length_sigmoid_k`, `length_sigmoid_x0` | Сигмоида от длины отзыва (в словах) |
| `uniqueness_threshold` | Порог косинусной близости для Union-Find «бот-кластеров» |
| `sim_noise_floor` | Ниже этого max-similarity штраф за уникальность не растёт |
| `min_trust_weight` | Нижний клип веса |

### 3.6. `config.ui`

| Ключ | Назначение |
|------|------------|
| `max_aspects_radar` | Максимум аспектов для радара и ползунков в UI (топ по mentions); в репозитории: **`8`** |

Ядро пайплайна (`ABSAPipeline`) эти ключи **не читает** — они для фронтенда/визуализации.

### 3.7. Воспроизводимость и расхождения с черновиками

Документ описывает код **как в репозитории**. 

### Как менять поведение без правки кода алгоритмов

1. Правка `configs/configs.py` (или подгрузка поверх `OmegaConf` в отдельном скрипте — если вы так сделаете).
2. `eval_pipeline.apply_config_overrides(overrides)` — для eval-конфигов JSON (секции должны совпадать с атрибутами `config`).

---

## 4. Сущности данных (типы и поля)

### 4.1. `ReviewInput` (`src/schemas/models.py`)

| Поле | Тип | Назначение |
|------|-----|------------|
| `id` | `str` | Идентификатор отзыва |
| `nm_id` | `int` | Идентификатор товара |
| `rating` | `int` | 1–5 |
| `created_date` | `datetime` | Для временного веса в math |
| `full_text`, `pros`, `cons` | `Optional[str]` | Сырой ввод |

**Производное:** `clean_text` (property) — склейка непустых блоков с префиксами «Достоинства:», «Недостатки:», «Комментарий:».

Отзывы без текста (пустой `clean_text`) отбрасываются при загрузке / в eval / в merged-скрипте.

### 4.2. `Candidate`

| Поле | Назначение |
|------|------------|
| `span` | Текст n-граммы (ключевой фрагмент) |
| `sentence` | Предложение, из которого извлечён span |
| `token_indices` | `(start, end)` индексы токенов в предложении |

**Важно:** после экстракции `token_indices` в downstream почти не используются — трассировка дальше идёт по строкам `span` / `sentence`.

### 4.3. `ScoredCandidate`

| Поле | Назначение |
|------|------------|
| `span`, `sentence` | Как у `Candidate` |
| `score` | Косинус span-эмбеддинга к эмбеддингу предложения (после порога) |
| `embedding` | Вектор span (для кластеризации и для cos к якорям в `_build_sentiment_pairs`) |

### 4.4. `AspectInfo`

| Поле | Назначение |
|------|------------|
| `keywords` | Список спанов, попавших в аспект |
| `centroid_embedding` | Центроид в пространстве эмбеддингов |
| `keyword_weights` | Веса ключевых слов (список; при слияниях дополняется) |
| `nli_label` | Строка для подстановки в шаблон гипотезы NLI; для medoid-кластеров может отличаться от имени аспекта |

### 4.5. `SentimentResult`

| Поле | Назначение |
|------|------------|
| `review_id`, `aspect`, `sentence` | Привязка |
| `score` | Итог [1, 5] |
| `p_ent_pos`, `p_ent_neg` | Вероятности entailment для двух гипотез |
| `confidence` | Вес пары (из soft-anchor: similarity span↔якорь) |

### 4.6. `AggregationResult` / `AspectScore` (`src/math/engine.py`)

На выходе math: по каждому аспекту `score`, `raw_mean`, `controversy`, `mentions`, `effective_mentions`, плюс опционально ковариационная матрица.

### 4.7. `PipelineResult` (`src/pipeline.py`)

| Поле | Содержание |
|------|------------|
| `product_id` | Обычно `nm_id` |
| `reviews_processed`, `processing_time` | Мета |
| `aspects` | Словарь имя аспекта → сериализуемые метрики (из `AspectScore`) |
| `aggregation` | Полный `AggregationResult` |
| `sentiment_details` | Все `SentimentResult` |
| `aspect_keywords` | Копия `keywords` из `AspectInfo` |

---

## 5. Контракты стадий (`src/stages.py`)

Стадии оформлены как ABC; пайплайн хранит ссылки на реализации и вызывает **только** эти методы:

| ABC | Метод | Вход | Выход |
|-----|--------|------|--------|
| `FraudStage` | `calculate_trust_weights` | `List[str]` (тексты) | `List[float]` |
| `ExtractionStage` | `extract` | `str` (один отзыв) | `List[Candidate]` |
| `ScoringStage` | `score_and_select` | `List[Candidate]` | `List[ScoredCandidate]` |
| `ClusteringStage` | `cluster` | `List[ScoredCandidate]` | `Dict[str, AspectInfo]` |
| `SentimentStage` | `batch_analyze` | `List[SentimentPair]` | `List[SentimentResult]` |
| `AggregationStage` | `aggregate` | `List[AggregationInput]` | `AggregationResult` |

Типовые алиасы в том же файле:

- `SentimentPair = Tuple[str, str, str, str, float]` — `(review_id, sentence, aspect_name, nli_label, weight)`.
- `AggregationInput` — dict с ключами `review_id`, `aspects`, `fraud_weight`, `date`.

**Не входит в ABC:** построение `sentence_to_review`, `_build_sentiment_pairs`, `_build_aggregation_input` — это логика **оркестратора** `ABSAPipeline`, чтобы не дублировать контракт «стадии» с чистой математикой.

---

## 6. Поток данных по шагам (детально)

Ниже — порядок в `ABSAPipeline.analyze_reviews_list`. Нумерация «2/7 … 7/7» в логах совпадает с этими шагами; **шаг 1** в логах `analyze_product` — это загрузка из БД **до** вызова `analyze_reviews_list`.

### 6.0. Подготовка входа

**Вход:** `reviews: List[ReviewInput]`, `product_id: int`.

**Опционально:**

1. `save_input_snapshot=True` → `_save_input_snapshot` пишет JSONL (см. раздел 8.1).
2. `snapshot_writer` → `save_reviews` → файл `00_reviews.jsonl` (см. раздел 8.2).

**Производное:** `texts = [r.clean_text for r in reviews]` — дальше по конвейеру идёт только список строк, но **порядок** строго совпадает с `reviews` (индекс `i` → один отзыв).

**Внутренние структуры, создаваемые позже:**

- `sentence_to_review: Dict[str, str]` — ключ: нормализованная строка предложения (`strip` и `lower().strip`), значение: `reviews[i].id`.
- `review_candidates_map: Dict[review_id, List[Candidate]]` — для снимков и отладки.

---

### 6.1. Стадия Anti-Fraud (`FraudStage`)

**Реализация по умолчанию:** `AntiFraudEngine` (`src/fraud/engine.py`).

#### Инициализация (`__init__`)

- Создаётся **`self.model = SentenceTransformer(config.models.encoder_path)`** — это **не** ссылка на `ABSAPipeline._encoder`, а отдельная загрузка тех же весов с диска.
- Считываются пороги из `config.fraud`: `uniqueness_threshold`, `sim_noise_floor`, `min_trust_weight`, `length_sigmoid_k`, `length_sigmoid_x0`.

#### Публичный API

- `calculate_trust_weights(texts) -> List[float]` — обёртка над `analyze()`: возвращает только список весов.
- `analyze(texts) -> List[TrustResult]` — полный разбор с полями `length_weight`, `uniqueness_weight`, `bot_cluster_id`, итоговый `trust_weight`.

**Вход:** `texts` — список `clean_text` в том же порядке, что `reviews`.

#### Внутренняя архитектура (по шагам)

1. **Эмбеддинги отзывов:** `self.model.encode(reviews, show_progress_bar=False)` — один вектор на отзыв.
2. **`_length_weights(reviews)`** — для каждого текста число слов `len(t.split())`, сигмоида  
   `1 / (1 + exp(-k * (words - x0)))` с `k = length_sigmoid_k`, `x0 = length_sigmoid_x0`.
3. **`_uniqueness_weights(embeddings)`** (при `n >= 2`):
   - матрица косинусных сходств `cosine_similarity(embeddings)`, диагональ обнуляется;
   - **уровень 1:** по строке — max сходство с любым другим отзывом; вычитается `sim_noise_floor`, нормируется; `base_uniq = 1 - adjusted²`;
   - **уровень 2:** Union-Find по парам `(i, j)` с `sim_matrix[i,j] >= uniqueness_threshold`; для кластера размера `S` множитель **ln(2)/ln(1+S)** в смысле **натурального логарифма** (основание *e*). В коде: `ln2 = math.log(2)` и `math.log(1 + s)` — в Python `math.log` без второго аргумента это именно ln, не log₂ и не log₁₀ (`src/fraud/engine.py`, блок после Union-Find). Для `S == 1` множитель **1.0** (ветка `if s > 1 else 1.0`).
   - итог: `uniqueness_weight = clip(base_uniq * log_factor, min_trust_weight, 1.0)`; для одиночных кластеров `bot_cluster_id = -1`.
4. **Финальный вес:** по индексу `i`:  
   `trust_weight = clip(length_weights[i] * uniqueness_weights[i], min_trust_weight, 1.0)`.

**Выход:** `trust_weights[i]` соответствует `reviews[i]`.

**Снимок:** `01_fraud.json` — `review_ids`, `trust_weights`, статистика min/max/mean.

**Подмена:** `ABSAPipeline(fraud_stage=MyFraud())` — класс с методом `calculate_trust_weights(self, texts: List[str]) -> List[float]`. Для **одного** экземпляра энкодера на весь пайплайн потребуется либо передавать общий `SentenceTransformer` внутрь кастомной реализации fraud, либо менять конструктор `AntiFraudEngine` в коде.

---

### 6.2. Стадия извлечения кандидатов (`ExtractionStage`)

**Реализация по умолчанию:** `CandidateExtractor` (`src/discovery/candidates.py`).

**Вход:** по одному разу для каждого элемента `texts[i]`.

**Внутри для каждого отзыва:**

1. `_clean` — пробелы после пунктуации, вычищение «лишних» символов, схлопывание пробелов.
2. `_split_sentences` — `re.split(r"[.!?]+", …)`; если пусто — весь текст как одно предложение.
3. Для каждого предложения `_tokenize` — lower, split, очистка токенов.
4. Все n-граммы в диапазоне `config.discovery.ngram_range`:
   - отсев по `min_word_length`;
   - дедуп по строке `span`;
   - фильтр `STOP_TOKENS`;
   - морфология `pymorphy3` (`_pass_morph_filter`, паттерны POS для биграмм / существительные для униграмм).

**Выход:** список `Candidate`; параллельно заполняется `sentence_to_review` для каждого `c.sentence`.

**Снимок:** `02_candidates.json` — `per_review[review_id]` → список `{span, sentence, token_indices}`.

**Подмена:** `ABSAPipeline(extraction_stage=...)`.

---

### 6.3. Стадия скоринга (`ScoringStage`)

**Реализация по умолчанию:** `KeyBERTScorer` с общим энкодером пайплайна.

**Вход:** плоский список всех `Candidate` по всем отзывам.

**Внутри:**

1. Собрать уникальные `sentence` и уникальные `span`, закодировать батчами через `SentenceTransformer`.
2. Сгруппировать кандидатов по `sentence`.
3. Для каждой группы: косинус `span_emb` vs `sentence_emb`; отсечь ниже `cosine_threshold`; отсортировать; взять до `keybert_top_k`.
4. MMR на оставшихся: параметры `mmr_lambda`, `mmr_top_k`.

**Выход:** `List[ScoredCandidate]` (порядок: по предложениям в процессе итерации).

**Снимок:** `03_scored.json` — список кандидатов; эмбеддинги опционально (`save_embeddings` у `SnapshotWriter`).

**Подмена:** `ABSAPipeline(scoring_stage=...)`; для совместимости с кластеризатором новая реализация должна выдавать `ScoredCandidate` с осмысленным `embedding`.

---

### 6.4. Стадия кластеризации (`ClusteringStage`)

**Реализация по умолчанию:** `AspectClusterer` (`src/discovery/clusterer.py`).

**Вход:** `List[ScoredCandidate]`.

**Внутри (логика baseline):**

1. Авто-стоп-слова по частоте однословных span среди кандидатов (`_detect_product_stops`).
2. `_aggregate_spans` — агрегация по строке `span`: `count`, сумма `score`, **первый попавшийся** `embedding` (не усреднение).
3. Эмбеддинги всех span → косинус к матрице **макро-якорей** `MACRO_ANCHORS` (усреднённые эмбеддинги списков слов/фраз на этапе `__init__`).
4. **Pass 1 (anchor assignment):** для каждого span — лучший якорь и margin (лучший − второй); пороги `anchor_assign_threshold`, `anchor_assign_margin`; отсев «junk» через `ANTI_ANCHORS` (`_is_junk_span`).
5. Недостаточно уверенные span → **residual**.
6. **Residual:** UMAP → HDBSCAN → именование кластера `_name_cluster` (либо имя якоря при большом margin, либо medoid-span как имя кластера).
7. Слияние близких residual-кластеров в UMAP-пространстве (`cluster_merge_threshold`).
8. Фильтр по `min_mentions` (аргумент `cluster()`, по умолчанию в вызове из пайплайна — значение по умолчанию метода).
9. **`_apply_nli_labels`:** для кластеров, помеченных как medoid (`aspect_medoid[name]=True`), поле `AspectInfo.nli_label` заменяется на имя ближайшего макро-якоря по cos(centroid, anchor). Для остальных кластеров `nli_label = name` (имя ключа аспекта).

**Выход:** `Dict[str, AspectInfo]` — ключи = имена аспектов (якорь или строка-medoid).

**Диагностика в объекте кластеризатора:** `last_assignment_counts`, `last_residual_medoid_names`, `last_nli_medoid_diagnostics`.

**Снимок:** `04_clusters.json` — сериализация `AspectInfo` (keywords, weights, nli_label, опционально centroid).

**Подмена:** `ABSAPipeline(clustering_stage=...)` — другой алгоритм при том же контракте `cluster(candidates) -> Dict[str, AspectInfo]`.

---

### 6.5. Построение NLI-пар (оркестратор, не ABC)

**Метод:** `_build_sentiment_pairs` в `src/pipeline.py`.

**Вход:**

- `scored_candidates`;
- `aspects: Dict[str, AspectInfo]`;
- `sentence_to_review`.

**Логика:**

1. Матрица якорей берётся из `self.clusterer._anchor_embeddings` (имена = ключи `MACRO_ANCHORS`).
2. Множество «разрешённых» якорей для этого товара: имена из `aspects`, для которых есть вложение в `_anchor_embeddings`, с учётом `info.nli_label`.
3. Для каждого `ScoredCandidate`: cos(`embedding`, каждый якорь); якоря с `sim >= multi_label_threshold` и входящие в `product_anchors`; сортировка; топ `multi_label_max_aspects`.
4. `review_id` из `sentence_to_review.get(cand.sentence.strip())` или `.lower().strip()`, иначе `"unknown"`.
5. Дедуп ключом `(review_id, sentence, aname)`.
6. Кортеж: `(review_id, sentence, aname, aname, sim)` — **оба** текстовых поля аспекта в кортеже равны **`aname`**, то есть имени из списка макро-якорей (`anchor_names`), выбранному по косинусу span↔якорь.

#### 6.5.1. Связь с `AspectInfo.nli_label` (важно для защиты)

- Множество **`product_anchors`** строится из ключей `aspects` и из `info.nli_label`, **если** они входят в `_anchor_embeddings`. Это **фильтр допустимых имён якорей**: в пару не попадёт якорь, который не связан ни с одним кластером товара.
- Сам **выбор** `aname` для пары идёт **только** из цикла по `anchor_names` и матрицы сходств — не подставляется строка `AspectInfo.nli_label` вместо `aname` в кортеже.
- В результате **`_apply_nli_labels` в кластеризаторе не меняет текст гипотезы NLI** в production-ветке: в `SentimentEngine` подстановка в шаблон идёт по полю кортежа (фактически всегда имя якоря из `MACRO_ANCHORS`). Поле `nli_label` в `AspectInfo` остаётся **метаданными кластера** (согласованность имени medoid-кластера с ближайшим якорём) и может использоваться в отчётах/диагностике, но **не** пробрасывается в 4-й элемент пары как отдельная строка.

**Снимок:** `05_sentiment_pairs.json`.

**Риск:** любое расхождение строки предложения с ключами в `sentence_to_review` → потеря привязки к отзыву.

---

### 6.6. Стадия NLI-сентимента (`SentimentStage`)

**Реализация по умолчанию:** `SentimentEngine` (`src/sentiment/engine.py`).

**Вход:** список кортежей (см. выше); длина кортежа 3, 4 или 5 поддерживается в `_process_batch`.

**Внутри:**

1. Для каждой пары: premise = предложение, гипотезы из шаблонов `config.sentiment.hypothesis_template_pos/neg` с подстановкой `aspect` из поля кортежа (для длины 5 — отдельное поле `nli_for_hyp`).
2. Два прохода `AutoModelForSequenceClassification`: pos и neg; берётся вероятность метки `entailment`.
3. `score = 1 + 4 * p_pos / (p_pos + p_neg + epsilon)`, клип [1, 5].
4. `confidence` = вес из кортежа (косинус к якорю).

**Выход:** `List[SentimentResult]`.

**Снимок:** `06_sentiment_results.json`.

**Подмена:** `ABSAPipeline(sentiment_stage=...)`.

---

### 6.7. Построение входа агрегации (оркестратор)

**Метод:** `_build_aggregation_input`.

**Вход:** `reviews`, `sentiment_scores`, `trust_weights`.

**Логика:**

1. Группировка `SentimentResult` по `review_id` и `aspect`; список пар `(score, confidence)`.
2. Взвешенное среднее по аспектам **внутри одного отзыва**.
3. Для каждого `review_id`, который есть в `reviews`: dict  
   `{"review_id", "aspects": {имя: средний_score}, "fraud_weight": trust_weights[idx], "date": created_date}`.

**Снимок:** `07_aggregation_input.json`.

**Важно:** на этом шаге из агрегата **исчезает** привязка к конкретным предложениям — остаются только средние по аспекту на отзыв.

---

### 6.8. Стадия математической агрегации (`AggregationStage`)

**Реализация по умолчанию:** `RatingMathEngine` (`src/math/engine.py`).

**Вход:** `List[AggregationInput]`.

**Внутри (упрощённо):**

1. Для каждого отзыва: вес `w = fraud_weight * time_weight(date)`.
2. Разнесение вкладов по корзинам аспектов.
3. `C = min(median_mentions, prior_strength_max)` — сила априора; **`prior_strength_max` берётся из `config.math`** (в репозитории сейчас **`1`**).
4. По каждому аспекту: взвешенное среднее, взвешенное std как `controversy`, байесовское сглаживание к `prior_mean`.
5. Ledoit-Wolf по матрице «отзыв × аспект» для отзывов с ≥2 аспектами (с fallback).

**Выход:** `AggregationResult`.

**Подмена:** `ABSAPipeline(aggregation_stage=...)`.

---

### 6.9. Финальная сборка `PipelineResult`

Собирается словарь `aspects` для API/JSON (числовые поля из `AspectScore`), плюс полные `aggregation`, `sentiment_details`, `aspect_keywords`.

**Снимок:** `08_pipeline_result.json` (урезанная сериализация без полного дампа ковариации в текущем `save_pipeline_result`).

---

## 7. Управление, изменение, переиспользование

### 7.1. Запуск полного пайплайна (production-путь)

```python
from src.pipeline import ABSAPipeline

p = ABSAPipeline(db_path="data/dataset.db")
r = p.analyze_product(nm_id=154532597, limit=200)
```

Или список отзывов:

```python
r = p.analyze_reviews_list(reviews, product_id=nm_id)
```

### 7.2. Dependency Injection стадий

```python
p = ABSAPipeline(
    fraud_stage=my_fraud,
    extraction_stage=my_extractor,
    scoring_stage=my_scorer,
    clustering_stage=my_clusterer,
    sentiment_stage=my_sentiment,
    aggregation_stage=my_math,
)
```

Требование: реализовать соответствующий ABC из `src/stages.py`. В конструкторе `ABSAPipeline` создаётся **`self._encoder`** и передаётся в дефолтные `KeyBERTScorer` и `AspectClusterer`. **`AntiFraudEngine` по умолчанию энкодер пайплайна не получает** — при подмене только `fraud_stage` можно отдать реализацию, которая переиспользует один `SentenceTransformer` (например, принять модель снаружи в вашем классе).

### 7.3. Снимки по стадиям + replay

**Создание writer:**

```python
from pathlib import Path
from src.snapshots import SnapshotWriter

w = SnapshotWriter(base_dir=Path("experiments/runs/my_run/snapshots"), product_id=nm_id)
```

Или через эксперимент:

```python
exp = ExperimentManager.create("name", cfg_dict)
w = exp.snapshot_writer(nm_id, save_embeddings=True)
```

**Вызов:**

```python
p.analyze_reviews_list(reviews, product_id=nm_id, snapshot_writer=w)
```

**Чтение обратно** (`src/snapshots.py`):

- `load_candidates_snapshot` → продолжить с `score_and_select`;
- `load_scored_snapshot` → продолжить с `cluster`;
- `load_clusters_snapshot` → руками связать с `_build_sentiment_pairs` (нужен ещё `sentence_to_review` и scored — или воспроизвести пары из `05`);
- `load_sentiment_results_snapshot` → `_build_aggregation_input` + `aggregate`.

### 7.4. Отдельный eval-поток (`eval_pipeline.py`)

- Поднимает **свой** `SentenceTransformer`, `CandidateExtractor`, `KeyBERTScorer`, `AspectClusterer`, `AntiFraudEngine`, `SentimentEngine` — **ещё один** экземпляр энкодера отдельно от любого созданного вами `ABSAPipeline` в том же процессе.
- **Не** вызывает `ABSAPipeline` и **не** вызывает `RatingMathEngine`.
- Строит NLI-пары функцией **`_build_pairs(scored, aspects, sentence_to_review, clusterer)`** — по смыслу дублирует **`_build_sentiment_pairs`** в `src/pipeline.py` (те же пороги `multi_label_threshold` / `multi_label_max_aspects`, та же матрица якорей, тот же формат 5-элементного кортежа). Это **нарушение DRY**: при изменении логики пар нужно править **два места**, иначе eval и production разъедутся.
- Выход: структуры для метрик, JSON в `write_prefix` (см. докстринги `eval_pipeline.py`).

Используйте eval для сравнения с разметкой; для финального «рейтинга товара» как в проде — `ABSAPipeline`.

### 7.5. Скрипт merged CSV

`scripts/run_pipeline_merged.py` — загрузка из CSV, `ABSAPipeline.analyze_reviews_list`, запись `pipeline_full_nm*.json`, опционально `--save-input-snapshot`.

---

## 8. Промежуточные файлы на диске

### 8.1. `save_input_snapshot` / `_save_input_snapshot`

- **Когда:** до стадии fraud, если `save_input_snapshot=True`.
- **Путь по умолчанию:** `data/snapshots/input_reviews_nm{product_id}_{UTC}.jsonl` или явный `input_snapshot_path`.
- **Содержимое строки:** поля отзыва + **`clean_text`**.

### 8.2. `SnapshotWriter` (папка `{base_dir}/nm{product_id}/`)

| Файл | Содержание |
|------|------------|
| `00_reviews.jsonl` | Поля ReviewInput **без** `clean_text` |
| `01_fraud.json` | id отзывов, веса, статистика |
| `02_candidates.json` | Кандидаты по `review_id` |
| `03_scored.json` | Отобранные кандидаты, опционально эмбеддинги |
| `04_clusters.json` | Аспекты и keywords |
| `05_sentiment_pairs.json` | Пары до NLI |
| `06_sentiment_results.json` | Результаты NLI |
| `07_aggregation_input.json` | Вход math |
| `08_pipeline_result.json` | Упрощённый итог |

### 8.3. Прочие артефакты

- `scripts/run_pipeline_merged.py` → `pipeline_full_nm*.json`, `pipeline_full_all_nm.json`, `aspects_for_manual_precision.txt`.
- `eval_pipeline` / `run_eval_config.py` → файлы в `experiments/runs/...` (eval-результаты, логи).
- SQLite `data/dataset.db` — только чтение через `DataLoader` (пайплайн БД не пишет).

---

## 9. Выход системы (что считать «результатом»)

| Потребитель | Что читать |
|-------------|------------|
| API / отчёт по товару | `PipelineResult.aspects`, `aspect_keywords`, при необходимости `aggregation` (ковариация, полные `AspectScore`) |
| Объяснимость / отладка | `sentiment_details` (предложение + аспект + score) |
| Воспроизводимость | Снимки `00`–`08` + `meta.json` эксперимента + зафиксированный `config` |
| Сравнение с разметкой | Выходы `eval_pipeline`, не путать с байесовским `score` из math |

---

## 10. Трассировка текста и типичные разрывы lineage

**Сохраняется по цепочке:** символы в `clean_text` → предложение → `span` → `ScoredCandidate` → кластер → NLI-пара (premise = предложение) → `SentimentResult.sentence` → при агрегации **теряется** привязка к предложению (остаётся среднее по отзыву и аспекту).

**Узкие места:**

- `token_indices` не используются после экстракции.
- `sentence_to_review` — строковые ключи; рассинхрон → `review_id == "unknown"`.
- В `_aggregate_spans` для одного и того же `span` эмбеддинг не усредняется — берётся первый.

---

## 11. Зависимости среды

- Локальные веса по путям `config.models.encoder_path` и `config.models.nli_path`.
- **Ресурсы энкодера:** при одном полном прогоне через `ABSAPipeline` в памяти по умолчанию **два** загруженных `SentenceTransformer` с одним и тем же `encoder_path` (пайплайн + fraud). Плюс NLI-модель. При параллельном открытии `eval_pipeline.run_pipeline_for_ids` в том же процессе возможен **третий** экземпляр энкодера.
- Python-пакеты: `sentence_transformers`, `transformers`, `torch`, `sklearn`, `numpy`, `pymorphy3`, `hdbscan`, `umap-learn`, `pydantic`, `omegaconf`, и т.д. (см. `requirements` проекта при наличии).

---

## 12. Версионирование документа

Этот файл описывает **Architecture v2** в терминах модульной декомпозиции стадий (`src/stages.py`), снимков (`src/snapshots.py`) и текущего `ABSAPipeline`. При смене контрактов ABC или порядка вызовов в `analyze_reviews_list` документ следует обновить.

**Синхронизация с аудитом:** разделы 3 (фактические значения `multi_label_threshold`, `prior_strength_max`, точные шаблоны NLI, `config.ui`), 6.1 (внутренняя структура fraud и отдельный энкодер), 6.5.1 (роль `nli_label` vs пары), 7.4 (дублирование `_build_pairs`), 11 (память), 2.1 (diag-скрипты) приведены в соответствие с исходным кодом и замечаниями ревью.
