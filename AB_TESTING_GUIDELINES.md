# A/B Testing Guidelines

## Purpose

Этот документ фиксирует правила, по которым в проект добавляются новые гипотезы, чтобы:

- сравнение было честным;
- откат был тривиальным;
- `production`, `eval` и `benchmark` собирали один и тот же pipeline.

## Основные правила

1. Новая гипотеза добавляется как отдельная реализация стадии или отдельная запись в factory/registry, а не правкой базового orchestration в `src/pipeline.py`.
2. Если одна и та же логика нужна в нескольких гипотезах, она сначала выносится в helper, а уже потом используется в конкретных реализациях.
3. Все гипотезы одной стадии обязаны соблюдать один и тот же контракт входа/выхода:
   - extraction: `str -> List[Candidate]`
   - scoring: `List[Candidate] -> List[ScoredCandidate]`
   - clustering: `List[ScoredCandidate] -> Dict[str, AspectInfo]`
   - pairing: `PairingContext -> List[SentimentPair]`
   - sentiment: `List[SentimentPair] -> List[SentimentResult]`
   - aggregation: `List[AggregationInput] -> AggregationResult`
4. Дополнительные данные между стадиями передаются только через typed models/dataclasses, а не через tuple разных длин, `getattr` на приватные поля или неформальные словари.
5. Переключение гипотез делается через resolved-config и factory-слой, а не ручными `if` в `eval_pipeline.py`, `benchmark/*` или ad hoc-скриптах.

## Как добавлять новую гипотезу

1. Определить, к какой стадии относится гипотеза.
2. Создать новую реализацию соответствующего интерфейса.
3. Если нужно, добавить helper-функции для общей логики.
4. Зарегистрировать новую реализацию в factory/registry.
5. Добавить конфиг-переключатель или новое имя стратегии.
6. Добавить тесты.

## Обязательные тесты

Для каждой новой гипотезы обязательно:

- smoke-test на контракт стадии;
- integration-test на включение через конфиг/factory;
- если гипотеза меняет сериализуемый формат, тест на snapshot/readback.

## Запрещённые паттерны

- менять основную ветку orchestration ради одной гипотезы;
- читать приватные поля другой стадии вместо typed API;
- дублировать выбор реализации в нескольких entrypoints;
- вводить новый формат данных без обновления общего контракта.

## Рекомендуемый путь запуска экспериментов

- основной entrypoint для A/B: `run_experiment.py`
- конфиг эксперимента должен задавать только `clusterer` и `overrides`
- `eval_pipeline.py` и `benchmark/*` должны использовать те же factory-функции и тот же resolved-config flow
