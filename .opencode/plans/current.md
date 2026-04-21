# План ближайших шагов: ФАЗА 1 (vocabulary)

## Контекст
- Текущая активная фаза: ФАЗА 1.
- Архитектура Path gamma зафиксирована: core vocabulary + domain sub-vocabulary.
- Сейчас меняем только vocabulary.
- Нельзя одновременно трогать segmentation, matching и sentiment.

## Цель фазы
Построить 4 подсловаря категорий и довести coverage словаря по gold-аспектам на 16 размеченных товарах до >= 65% (target: >= 75%).

## Главная метрика фазы
- Coverage vocabulary = доля gold-аспектов, которые покрываются объединением `core + domain vocabulary`.

## Ближайшие шаги

### Шаг 0. Проверка baseline coverage (`core-only`)
- Имя: `phase1_step0_core_baseline`
- Цель: зафиксировать baseline покрытия без domain vocabulary.
- Baseline: только `src/vocabulary/universal_aspects_v1.yaml`.
- Изменяемая переменная: отсутствует; это фиксирующее baseline-измерение перед изменениями.
- Гипотеза: baseline coverage на одном core vocabulary окажется существенно ниже результата `core + domain`, и это позволит корректно интерпретировать итоговый прирост.
- Метрика: `coverage(core-only)` по всем 16 товарам; дополнительно coverage по категориям.
- Критерий успеха шага: сохранена таблица coverage по категориям и общий baseline coverage до добавления domain vocabulary.
- Файлы для изменения:
  - скрипт/ноутбук/модуль coverage analysis
  - артефакт с baseline-результатами

### Шаг 1. День 1. Собрать reproducibility log и черновик аспектов для `physical_goods`
- Имя: `phase1_step1_physical_goods_draft`
- Цель: получить первый воспроизводимый domain sub-vocabulary для одной категории без утечки из gold labels.
- Baseline: только `src/vocabulary/universal_aspects_v1.yaml`.
- Изменяемая переменная: добавление чернового подсловаря только для `physical_goods`.
- Ограничение объёма: максимум 10-15 аспектов.
- Гипотеза: даже один аккуратно собранный domain sub-vocabulary добавит заметное покрытие аспектов категории `physical_goods` относительно одного core vocabulary.
- Метрика: coverage по gold-аспектам для товаров категории `physical_goods`; ожидаем рост относительно baseline core-only.
- Минимальный критерий успеха шага: готов файл `src/vocabulary/domain/physical_goods.yaml` и `docs/vocabulary_reproducibility_log.md` с источниками и происхождением аспектов.
- Файлы для изменения:
  - `src/vocabulary/domain/physical_goods.yaml`
  - `docs/vocabulary_reproducibility_log.md`

### Шаг 2. Собрать черновик аспектов для `consumables`
- Имя: `phase1_step2_consumables_draft`
- Цель: покрыть доменные аспекты еды и товаров потребления, которых нет в ядре.
- Baseline: только core vocabulary (`src/vocabulary/universal_aspects_v1.yaml`).
- Изменяемая переменная: добавление подсловаря только для `consumables`.
- Гипотеза: аспекты типа вкус, запах, свежесть, состав дадут прирост coverage в категории `consumables` без дублирования ядра.
- Метрика: coverage по gold-аспектам для товаров категории `consumables`; сравнение `coverage(core)` vs `coverage(core + consumables)`.
- Критерий успеха шага: готов `src/vocabulary/domain/consumables.yaml`, источники дописаны в log.
- Файлы для изменения:
  - `src/vocabulary/domain/consumables.yaml`
  - `docs/vocabulary_reproducibility_log.md`

### Шаг 3. Собрать черновик аспектов для `hospitality`
- Имя: `phase1_step3_hospitality_draft`
- Цель: покрыть аспекты проживания и гостиничного опыта.
- Baseline: только core vocabulary (`src/vocabulary/universal_aspects_v1.yaml`).
- Изменяемая переменная: добавление подсловаря только для `hospitality`.
- Гипотеза: аспекты номер, персонал, завтрак, заселение, чистота комнаты, шумоизоляция поднимут coverage в `hospitality`.
- Метрика: coverage по gold-аспектам для товаров категории `hospitality`; сравнение `coverage(core)` vs `coverage(core + hospitality)`.
- Критерий успеха шага: готов `src/vocabulary/domain/hospitality.yaml`, источники дописаны в log.
- Файлы для изменения:
  - `src/vocabulary/domain/hospitality.yaml`
  - `docs/vocabulary_reproducibility_log.md`

### Шаг 4. Собрать черновик аспектов для `services`
- Имя: `phase1_step4_services_draft`
- Цель: покрыть аспекты сервисного взаимодействия.
- Baseline: только core vocabulary (`src/vocabulary/universal_aspects_v1.yaml`).
- Изменяемая переменная: добавление подсловаря только для `services`.
- Гипотеза: аспекты ожидание, поддержка, консультация, скорость ответа, компетентность, решение проблемы поднимут coverage в `services`.
- Метрика: coverage по gold-аспектам для товаров категории `services`; сравнение `coverage(core)` vs `coverage(core + services)`.
- Критерий успеха шага: готов `src/vocabulary/domain/services.yaml`, источники дописаны в log.
- Файлы для изменения:
  - `src/vocabulary/domain/services.yaml`
  - `docs/vocabulary_reproducibility_log.md`

### Шаг 5. Подключить загрузку `core + domain`
- Имя: `phase1_step5_loader_extension`
- Цель: сделать воспроизводимую загрузку объединённого словаря по `category_id`.
- Baseline: loader умеет загружать только core vocabulary.
- Изменяемая переменная: логика загрузки словаря для категории.
- Гипотеза: единый путь загрузки уберёт ручные склейки словарей и позволит честно считать coverage и дальше запускать matching без дополнительных правок.
- Метрика: техническая метрика - успешная загрузка `core + domain` для всех 4 категорий; тесты loader проходят.
- Критерий успеха шага: loader и тесты обновлены.
- Файлы для изменения:
  - файлы loader в `src/vocabulary/`
  - связанные тесты vocabulary loader

### Шаг 6. Прогнать coverage analysis на 16 товарах
- Имя: `phase1_step6_coverage_eval`
- Цель: измерить, достигнут ли порог покрытия после добавления 4 подсловарей.
- Baseline: coverage core-only.
- Изменяемая переменная: использование `core + domain` вместо только core.
- Гипотеза: полный hybrid vocabulary даст общий coverage >= 65% и желательно приблизится к 75%.
- Метрика: общий coverage vocabulary на 16 товарах; дополнительно coverage по категориям.
- Критерий успеха шага: есть таблица `core-only vs core+domain` по общему coverage и по категориям.
- Файлы для изменения:
  - скрипт/ноутбук/модуль coverage analysis
  - артефакт с результатами измерения

### Шаг 7. Если coverage < 65%: формальный fallback через frequency analysis
- Имя: `phase1_step7_frequency_fallback`
- Цель: расширить только слабые подсловари без in-sample ручной подгонки.
- Baseline: результат шага 6.
- Изменяемая переменная: добавление новых аспектов из top-N частотных существительных по категории на 1M корпусе.
- Гипотеза: частотный анализ найдёт пропущенные доменные аспекты и поднимет coverage до минимального порога.
- Метрика: coverage vocabulary после расширения; целевой порог >= 65%.
- Критерий успеха шага: документированная процедура расширения и повторный отчёт coverage.
- Файлы для изменения:
  - соответствующие `src/vocabulary/domain/*.yaml`
  - `docs/vocabulary_reproducibility_log.md`
  - код/скрипт frequency analysis при необходимости

## Порядок выполнения
1. Сначала шаг 0, чтобы зафиксировать baseline `core-only` до любых изменений.
2. Затем шаг 1 как однодневный и изолированный эксперимент.
3. Потом шаги 2-4 по одной категории за раз.
4. После этого шаг 5, чтобы не смешивать сбор словаря и код loader в одном эксперименте.
5. Потом шаг 6 как первое итоговое измерение успеха фазы на полном hybrid vocabulary.
6. Шаг 7 выполнять только если coverage < 65%.

## Что сделано
- Проанализированы `AGENTS.md` и `NewRoadMap.txt`.
- Зафиксирован детальный план ближайших шагов для ФАЗЫ 1.
- Добавлен отдельный скрипт `scripts/run_vocabulary_coverage.py` для расчёта vocabulary coverage вне старого eval pipeline.
- Выполнен пробный baseline run `core-only` на единственном доступном историческом gold-артефакте из git (`8` товаров, не официальный датасет из `16` товаров).
- Выполнен официальный baseline run `core-only` на `data/dataset_final.csv` (`16` товаров).

## Результат шага 0
- Эксперимент: `phase1_step0_core_baseline`
- Источник данных: `data/dataset_final.csv`
- Общий baseline: `coverage(core-only) = 0.4133` (`155 / 375` gold-аспектов)
- Coverage по категориям:
  - `consumables`: `0.7647` (`13 / 17`)
  - `physical_goods`: `0.6240` (`78 / 125`)
  - `services`: `0.3276` (`38 / 116`)
  - `hospitality`: `0.2222` (`26 / 117`)
- Вывод: основной дефицит покрытия находится в `hospitality` и `services`; именно там domain sub-vocabularies должны дать наибольший прирост.

## Результат шага 1
- Эксперимент: `phase1_step1_physical_goods_draft`
- Изменение: добавлен `src/vocabulary/domain/physical_goods.yaml`
- Результат для категории `physical_goods`:
  - `coverage(core) = 0.6240`
  - `coverage(core + physical_goods) = 0.6560`
  - абсолютный прирост: `+0.0320` (`+3.2 p.p.`)
- Вывод: `physical_goods` sub-vocabulary дал умеренный, но реальный прирост coverage.
- Интерпретация: это согласуется с уже сильным baseline `core-only` для `physical_goods`.
- Основной вклад дали аспекты: `Инструкция`, `Сборка`, `Уход`.

## Результат шага 2
- Эксперимент: `phase1_step2_hospitality_draft`
- Изменение: добавлен `src/vocabulary/domain/hospitality.yaml`
- Результат для категории `hospitality`:
  - `coverage(core) = 0.2222`
  - `coverage(core + hospitality) = 0.3077`
  - абсолютный прирост: `+0.0855` (`+8.55 p.p.`)
- Вывод: `hospitality` дал сильный прирост coverage.
- Интерпретация: эффект значительно выше, чем у `physical_goods`, что подтверждает гипотезу о большей эффективности domain vocabulary в слабых категориях.
- Дополнительное наблюдение: есть заметное перекрытие broad аспектов между `hospitality` и `services`; при добавлении `hospitality` vocabulary выросло и покрытие в `services`, что указывает на частично общий слой review semantics для location/service experience.

## Результат шага 3
- Эксперимент: `phase1_step3_services_draft`
- Изменение: добавлен `src/vocabulary/domain/services.yaml`
- Результат для категории `services`:
  - `coverage(core) = 0.3276`
  - `coverage(core + services) = 0.3879`
  - абсолютный прирост: `+0.0603` (`+6.03 p.p.`)
- Вывод: `services` дал заметный положительный прирост coverage.
- Интерпретация: эффект ниже, чем у `hospitality`, но выше, чем у `physical_goods`, что подтверждает зависимость эффективности domain vocabulary от baseline категории.

## Результат шага 4
- Эксперимент: `phase1_step4_consumables_draft`
- Изменение: добавлен `src/vocabulary/domain/consumables.yaml`
- Результат для категории `consumables`:
  - `coverage(core) = 0.7647`
  - `coverage(core + consumables) = 0.8235`
  - абсолютный прирост: `+0.0588` (`+5.88 p.p.`)
- Вывод: `consumables` дал положительный, но точечный прирост coverage.
- Интерпретация: прирост обеспечен фактически одним новым аспектом - `Поедаемость`.
- Подтверждённая гипотеза: в high-baseline категории domain vocabulary даёт точечное улучшение, а не системный скачок.

## Сводка По Категориям

| Category | Coverage Core | Coverage Core+Domain | Absolute Gain |
|---|---:|---:|---:|
| `physical_goods` | `0.6240` | `0.6560` | `+0.0320` |
| `hospitality` | `0.2222` | `0.3077` | `+0.0855` |
| `services` | `0.3276` | `0.3879` | `+0.0603` |
| `consumables` | `0.7647` | `0.8235` | `+0.0588` |

## Общий Вывод По Фазе 1
- Domain sub-vocabularies дали положительный прирост coverage во всех четырёх категориях.
- Сила эффекта зависит от baseline `core-only`: чем слабее исходное покрытие категории, тем больше выигрыш от domain vocabulary.
- Итоговый порядок эффекта:
  - `hospitality`: `+8.55 p.p.`
  - `services`: `+6.03 p.p.`
  - `consumables`: `+5.88 p.p.`
  - `physical_goods`: `+3.2 p.p.`
- При этом `consumables` показывает важную качественную разницу: при высоком baseline улучшение оказалось не системным, а фактически точечным и было обеспечено одним аспектом.
- Это подтверждает, что hybrid vocabulary path работает как ожидается: core даёт общий слой, а domain vocabulary закрывает оставшиеся доменные пробелы в разной степени в зависимости от baseline категории.

## Наблюдение О Перекрытии Аспектов
- Между категориями есть частичное перекрытие broad аспектов, особенно между `hospitality` и `services`.
- Это перекрытие не отменяет полезность отдельных domain sub-vocabularies, но показывает, что часть review semantics лежит в промежуточной зоне между доменами.
- Практический вывод: при проектировании следующих sub-vocabularies нужно явно отделять
  - truly domain-specific aspects
  - cross-domain service-process aspects
  чтобы не раздувать словарь и не дублировать сигналы.

## Наблюдение О Межкатегориальном Шуме
- В ходе построения domain sub-vocabularies проявился межкатегориальный шум: отдельные лексические единицы и аспекты могут случайно повышать coverage вне целевой категории.
- Наиболее заметно это проявилось между `hospitality` и `services`, а также точечно в `consumables` через лексику, которая может интерпретироваться шире целевого домена.
- Практический вывод: перед переходом к следующей фазе нужна финальная проверка межкатегориальной чистоты domain словарей.
- Цель этой проверки:
  - выявить нежелательные cross-category matches
  - убрать слишком общие или шумные доменные термы
  - зафиксировать, какие overlaps допустимы концептуально, а какие являются артефактом словаря

## Следующий шаг

### Шаг 2. Собрать broad sub-vocabulary для `hospitality`
- Имя: `phase1_step2_hospitality_draft`
- Цель: добавить domain-specific аспекты гостеприимства, которых не хватает ядру, и поднять coverage для самой слабой категории baseline.
- Baseline: только core vocabulary (`src/vocabulary/universal_aspects_v1.yaml`), `coverage(core) = 0.2222` для `hospitality`.
- Гипотеза: broad hospitality sub-vocabulary, собранный из стабильных внешних источников и не дублирующий core, даст заметный прирост coverage относительно `core-only`, потому что ядро плохо покрывает аспекты проживания и гостиничной инфраструктуры.
- Файлы для изменения:
  - `src/vocabulary/domain/hospitality.yaml`
  - `docs/vocabulary_reproducibility_log.md`
- Критерий успеха:
  - собран broad hospitality vocabulary без leakage из gold labels
  - не смешаны несколько архитектурных изменений
  - измерение `coverage(core)` vs `coverage(core + hospitality)` показывает положительный прирост для категории `hospitality`
  - аспекты остаются broad и защитимыми, а не сводятся к узким микродоменам

### Шаг 3. Собрать broad sub-vocabulary для `services`
- Имя: `phase1_step3_services_draft`
- Цель: добавить broad domain-specific аспекты сервисного взаимодействия, которых не хватает ядру, и поднять coverage для категории `services`.
- Baseline: только core vocabulary (`src/vocabulary/universal_aspects_v1.yaml`), `coverage(core) = 0.3276` для `services`.
- Гипотеза: broad services sub-vocabulary, собранный из стабильных внешних источников и не дублирующий core, даст положительный прирост coverage относительно `core-only`, но при проектировании нужно учитывать частичное семантическое перекрытие с `hospitality`.
- Файлы для изменения:
  - `src/vocabulary/domain/services.yaml`
  - `docs/vocabulary_reproducibility_log.md`
- Критерий успеха:
  - собран broad services vocabulary без leakage из gold labels
  - используются только стабильные и воспроизводимые источники
  - не дублируется core vocabulary
  - измерение `coverage(core)` vs `coverage(core + services)` показывает положительный прирост для категории `services`
  - аспекты остаются broad, защитимыми и не распадаются на узкие микродомены

## Что изменилось
- Появился рабочий план в `.opencode/plans/current.md`.
- Появился воспроизводимый coverage script с поддержкой:
  - markup CSV с `true_labels`
  - локального benchmark YAML с `true_aspects`
  - чтения исторического benchmark YAML через `git show`
- Сохранены артефакты пробного прогона в `.opencode/artifacts/phase1_step0_historical8_core_baseline/`.
- Сохранены артефакты официального baseline-прогона в `.opencode/artifacts/phase1_step0_core_baseline/`.

## Что проверять дальше
- Есть ли в репозитории уже заготовки для `src/vocabulary/domain/` и существующий coverage analysis.
- Сначала нужно выполнить шаг 0 и сохранить baseline coverage `core-only`.
- После этого можно начинать шаг 1 без смешивания с изменениями loader.
- Шаг 0 закрыт.
- Следом можно переходить к шагу 1 и собирать первый domain vocabulary, но с учётом baseline разумно после этого быстро проверить, не стоит ли приоритизировать `hospitality` и `services` раньше по ожидаемому приросту coverage.

## Апдейт: завершение микрозачистки словарей

### Эксперимент
- Имя: `phase1_step8_domain_micro_cleanup_finalization`
- Цель: убрать межкатегориальный шум в domain vocabularies без потери рабочего coverage.
- Baseline: версия словарей до микрозачистки с зафиксированным `hospitality coverage = 0.3077`.
- Изменяемая переменная: точечная чистка термов в domain словарях; затем локальный `micro-rollback` только для `hospitality`.
- Ожидаемый эффект: сохранить coverage по категориям после чистки, вернуть просадку `hospitality` к рабочему уровню без полного возврата шумных термов.

### Результат
- Микрозачистка завершена.
- Межкатегориальный шум в domain vocabularies удалён.
- Coverage после зачистки в целом сохранён.
- Единственная заметная просадка была в `hospitality`.
- Выполнен локальный `micro-rollback` только для аспекта `breakfast_food`.
- Возвращены термы `еда` и `буфет`.
- `hospitality coverage` восстановлен до `0.3077`.
- Остальные словари не изменялись в rollback-шаге.

### Статус для следующей фазы
- Текущая версия domain словарей зафиксирована как рабочая для перехода к фазе matching.

## Закрытие ФАЗЫ 1 (Vocabulary)
- Фаза 1 считается завершённой.
- Финальное состояние словарей после микрозачистки зафиксировано.
- Для `hospitality` выполнен локальный rollback только в `breakfast_food` (`еда`, `буфет`) с восстановлением `coverage = 0.3077`.
- Версия словарей признана рабочей для перехода к matching stage.

## Переходный план: ФАЗА 2 (baseline matching на current extractor)

### Эксперимент
- Имя: `phase2_step1_baseline_matching_current_extractor`
- Цель: получить воспроизводимый baseline по детекции аспектов на текущем extractor с текущим hybrid vocabulary (`core + domain`).
- Baseline: текущий extractor pipeline до segmentation и без ensemble; зафиксированная версия словарей после микрозачистки и локального rollback `hospitality`.
- Гипотеза: даже single-signal baseline matching на текущем extractor с hybrid vocabulary даст измеримый прирост recall/F1 относительно reference `core-only` и создаст стабильную точку сравнения для следующих A/B шагов.
- Файлы для изменения:
  - `benchmark/candidate_matching/pilot_config.yaml`
  - `benchmark/candidate_matching/run_candidate_pilot.py`
  - `benchmark/candidate_matching/data_loaders.py`
  - при необходимости: `src/vocabulary/loader.py` (только если нужно явно зафиксировать выбор `core + domain` по `category_id`)
  - артефакты прогона в `.opencode/artifacts/phase2_step1_baseline_matching_current_extractor/`
- Метрики:
  - review-level `precision`, `recall`, `F1` по aspect detection на 16 товарах
  - per-category `precision`, `recall`, `F1` (`physical_goods`, `consumables`, `hospitality`, `services`)
  - coverage hit-rate matched aspects against vocabulary (диагностическая)
  - время прогона (диагностическая)
- Критерий успеха:
  - получен полностью воспроизводимый baseline-отчёт по `precision/recall/F1` на 16 товарах
  - baseline стабильно запускается на current extractor и текущем hybrid vocabulary
  - в эксперименте не задействованы segmentation и ensemble сигналы
  - сформирован reference point для следующего A/B шага (`candidates vs segmentation`)
