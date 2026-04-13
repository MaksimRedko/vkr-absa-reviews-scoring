Структура:
  logs/01_pipeline_full.log     — полный ABSAPipeline (7 шагов), аспекты+scores
  logs/02_eval_step12.log       — статистика разметки + discovery/sentiment
  logs/03_eval_step4_auto.log   — Recall, MAE, Mention recall (--mapping auto)
  logs/04_diag_loss_funnel.log  — воронка потерь A–E
  pipeline/                     — JSON + aspects_for_manual_precision.txt
  suite_*eval_*                 — файлы метрик с префиксом suite_

Ручная Precision: pipeline/aspects_for_manual_precision.txt
Для каждого аспекта: TP / FP / Borderline.
