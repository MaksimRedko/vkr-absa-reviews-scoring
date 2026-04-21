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
