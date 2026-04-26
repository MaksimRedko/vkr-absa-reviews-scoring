# ABSA Pipeline (Research to Production)

## Цель проекта
Разработка пайплайна аспектно-ориентированного анализа отзывов (ABSA) с математической агрегацией на базе байесовского вывода и портфельной теории (для ВКР). 

## Текущий статус
- `parser/` и `app/` содержат старый MVP-код (линейное взвешивание, жесткий хардкод).
- `src/` содержит скелет новой гибридной архитектуры (в процессе разработки).
- Настройки вынесены в `configs/` через OmegaConf.

## Стек
- NLP: `rubert-tiny2` (NLI), `Natasha` (синтаксис), `HDBSCAN` (кластеризация).
- Configs: `OmegaConf`.
- UI: `Streamlit`.

## Traced pipeline

Запуск полного traced refactor:

```powershell
python -m src.pipeline.run_traced_pipeline --config run_config.yaml
```

Выход пишется в `results/<timestamp>_traced/`.

Оценка прогона:

```powershell
python -m src.evaluation.evaluate_run results/<timestamp>_traced
```

Sanity gate:

```powershell
$env:ABSA_SANITY_RUN_DIR="results/<timestamp>_traced"
pytest tests/integration/test_metrics_consistency.py -q
```

Dashboard:

```powershell
streamlit run dashboard/app.py
```

Графики для ВКР:

```powershell
python -m src.plotting.plot_coverage_breakdown results/<timestamp>_traced
python -m src.plotting.plot_confusion_matrix results/<timestamp>_traced
python -m src.plotting.plot_metrics_comparison results/<timestamp>_traced
python -m src.plotting.plot_negation_impact results/<timestamp>_traced
```
