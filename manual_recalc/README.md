# Manual Recalc

Отдельный модуль для ручного пересчёта итоговых метрик.

Что делает:
- показывает карточку отзыва: текст, gold, system, evidence;
- хранит черновики и коммиты в SQLite;
- экспортирует ручные `precision/recall/F1` и `MAE`;
- умеет собирать prompt для ИИ по текущей пачке.

Запуск из корня:

```bash
streamlit run manual_recalc/app.py
```

Что лежит внутри:
- `app.py` — Streamlit UI;
- `data_access.py` — сбор review-centric view из traced artifacts;
- `storage.py` — SQLite-слой;
- `metrics.py` — ручной пересчёт detection/sentiment метрик;
- `prompting.py` — prompt + few-shot для batch prefill;
- `config/` — редактируемые few-shot и шаблон prompt.

База по умолчанию:
- `manual_recalc/data/manual_recalc.sqlite3`

Основная логика:
- одна строка `system_decisions` на системный аспект;
- одна строка `gold_decisions` на gold-аспект;
- `review_status` держит `not_started / in_progress / done / needs_review`.

Экспорт:
- `manual_system_decisions.csv`
- `manual_gold_decisions.csv`
- `manual_review_status.csv`
- `manual_metrics.csv`
