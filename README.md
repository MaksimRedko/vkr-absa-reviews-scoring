# ABSA Pipeline (Research to Production)

## Цель проекта
Разработка пайплайна аспектно-ориентированного анализа отзывов (ABSA) с математической агрегацией для ВКР.

## Текущий статус
- `parser/` и `app/` содержат старый MVP-код.
- `src/` содержит новый traced pipeline.
- Настройки вынесены в `configs/` через OmegaConf.

## Стек
- NLP: `rubert-tiny2`, `Natasha`, `HDBSCAN`
- Configs: `OmegaConf`
- UI: `Streamlit`

## Traced pipeline

Запуск полного traced refactor:

```powershell
python -m src.pipeline.run_traced_pipeline --config run_config.yaml
```

Точка входа:
- `src/pipeline/run_traced_pipeline.py`
- `src/pipeline/orchestrator.py::run_traced_pipeline()`

Выход пишется в `results/<timestamp>_traced/`.

Stable final result sets поверх frozen traced run:

```powershell
python scripts/freeze_final_results.py
```

Скрипт создаёт:
- `results/final_res_v1` = current baseline / mode A
- `results/final_res_v2` = sentence evidence / mode B
- `benchmark/manual_audit/final_v2` = manual audit queue для `final_res_v2`

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
