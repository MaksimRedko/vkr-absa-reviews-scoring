# Sentiment Postprocess Calibration

Этот эксперимент проверяет только постобработку тональности.
Пайплайн не перезапускается.
Аспекты не пересчитываются.
NLI заново не запускается.
Используются уже сохранённые вероятности модели и ручной аудит.

Главная проверка:
можно ли уменьшить уход оценок к нейтральной зоне за счёт другой формулы или лёгкой калибровочной модели.

## Scope

- `build_dataset.py` собирает датасет только по manual matched TP-парам.
- `run_formula_dryrun.py` считает фиксированные формулы без обучения.
- `run_supervised_calibrator_lopo.py` обучает только лёгкий калибратор поверх сохранённых признаков.
- `report.py` собирает общий итоговый отчёт.

## Important

- `dry-run formulas` = диагностический эксперимент без обучения.
- `supervised calibrator` = обучаемая калибровка, оценивается только через Leave-One-Product-Out по товарам.
- Если в сохранённых артефактах нет нужных `neg_*` вероятностей, соответствующие формулы не дорисовываются, а помечаются как `skipped`.

## Default Data Sources

- traced run: `results/20260502_171530_traced`
- manual audit db: `manual_recalc/data/manual_recalc.sqlite3`
- dataset: `data/dataset_final.csv`

## Run

```bash
python benchmark/sentiment_postprocess_calibration/build_dataset.py
python benchmark/sentiment_postprocess_calibration/run_formula_dryrun.py
python benchmark/sentiment_postprocess_calibration/run_supervised_calibrator_lopo.py
```
