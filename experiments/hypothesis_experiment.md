# Эксперимент: 4 варианта NLI-гипотез

**Текущий зафиксированный baseline** (`configs/configs.py`): NLI remap ON (medoid → `nli_label` через argmax к якорям) + **гипотезы B** (`{aspect} — это хорошо` / `{aspect} — это плохо`). `{aspect}` в гипотезах = `nli_label`.

## Запуск

1. В `configs/configs.py` выставить пару `hypothesis_template_pos` / `hypothesis_template_neg` (см. комментарий в файле).
2. Перезапуск пайплайна и метрик:

```bash
python eval_pipeline.py step12
python eval_pipeline.py step4 --mapping auto
```

Итог: `eval_metrics_auto.json` (или с `--write-prefix`).

## Три числа с каждого прогона

| Вариант | Sentence MAE **raw** | Product MAE **n_true≥3** | Product MAE **all** |
|---------|----------------------|---------------------------|---------------------|
| A | | | |
| B | | | |
| C | | | |
| D | | | |

### Откуда брать в JSON

После `step4` в корне `eval_metrics_auto.json`:

- **Sentence MAE raw** → `global_mae_raw`
- **Product MAE (n≥3)** → `product_ratings.global_product_mae_filtered`
- **Product MAE (all)** → `product_ratings.global_product_mae`

### Тексты гипотез

| | pos | neg |
|---|-----|-----|
| **A** | Автор доволен {aspect} | Автор недоволен {aspect} |
| **B** | {aspect} — это хорошо | {aspect} — это плохо |
| **C** | Автор хвалит {aspect} | Автор критикует {aspect} |
| **D** | {aspect} отличного качества | {aspect} ужасного качества |

`{aspect}` подставляется как **nli_label** (не medoid-имя кластера при необходимости).
