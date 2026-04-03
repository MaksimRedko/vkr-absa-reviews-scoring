"""
compare_runs.py — сравнение результатов экспериментов из registry.jsonl.

Примеры:
  python experiments/compare_runs.py                        # все запуски
  python experiments/compare_runs.py --last 5               # последние 5
  python experiments/compare_runs.py --diff A B             # diff двух run_id
  python experiments/compare_runs.py --config baseline      # фильтр по конфигу
  python experiments/compare_runs.py --sort macro_precision # сортировка по метрике
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.experiment_manager import load_registry, ExperimentManager, RUNS_DIR

# ── Форматирование ──────────────────────────────────────────────────────────

_METRIC_COLS = [
    ("macro_precision",          "MacroP"),
    ("macro_recall",             "MacroR"),
    ("micro_precision",          "MicroP"),
    ("micro_recall",             "MicroR"),
    ("global_mae_raw",           "MAE_raw"),
    ("global_mae_calibrated",    "MAE_cal"),
    ("global_mention_recall_review", "MentRec"),
]

_COL_W = 9  # ширина числовой колонки


def _fmt(val: Optional[float], w: int = _COL_W) -> str:
    if val is None:
        return "-".center(w)
    return f"{val:.4f}".center(w)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _print_table(entries: List[Dict[str, Any]], title: str = "") -> None:
    if not entries:
        print("Нет запусков в registry.")
        return

    if title:
        print(f"\n{'─'*80}")
        print(f"  {title}")
        print(f"{'─'*80}")

    id_w    = 30
    conf_w  = 16
    hash_w  = 8
    dirty_w = 5

    # Заголовок
    header = (
        f"{'run_id':<{id_w}}  "
        f"{'config':<{conf_w}}  "
        f"{'git':<{hash_w}}  "
        f"{'dirty':<{dirty_w}}  "
        + "  ".join(label.center(_COL_W) for _, label in _METRIC_COLS)
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    for e in entries:
        dirty_mark = "yes" if e.get("git_dirty") else "no"
        row = (
            f"{_truncate(e.get('run_id', '?'), id_w):<{id_w}}  "
            f"{_truncate(e.get('config_name', '?'), conf_w):<{conf_w}}  "
            f"{e.get('git_hash', '?'):<{hash_w}}  "
            f"{dirty_mark:<{dirty_w}}  "
            + "  ".join(_fmt(e.get(col)) for col, _ in _METRIC_COLS)
        )
        print(row)
    print(sep)
    print(f"  Итого: {len(entries)} запуск(ов)")
    print()


def _print_diff(run_a: str, run_b: str) -> None:
    registry = {e["run_id"]: e for e in load_registry()}

    missing = [r for r in (run_a, run_b) if r not in registry]
    if missing:
        # Попробовать загрузить напрямую из папок
        for run_id in missing:
            try:
                exp = ExperimentManager.load(run_id)
                if not exp.is_complete():
                    print(f"[WARN] {run_id}: метрики не найдены в папке эксперимента")
                    continue
                meta = exp.meta()
                metrics = exp.metrics() or {}
                registry[run_id] = {
                    "run_id": run_id,
                    "config_name": meta.get("config_name", "?"),
                    "git_hash": meta.get("git_hash", "?"),
                    "git_dirty": meta.get("git_dirty", False),
                    **{col: metrics.get(col) for col, _ in _METRIC_COLS},
                }
            except FileNotFoundError:
                print(f"[ERROR] run_id не найден: {run_id}")
                sys.exit(1)

    a, b = registry[run_a], registry[run_b]

    print(f"\n{'─'*60}")
    print(f"  DIFF: {run_a}  vs  {run_b}")
    print(f"{'─'*60}")
    print(f"  {'Метрика':<24}  {'A':>{_COL_W}}  {'B':>{_COL_W}}  {'Δ (B−A)':>{_COL_W}}")
    print(f"{'─'*60}")

    for col, label in _METRIC_COLS:
        va = a.get(col)
        vb = b.get(col)
        if va is None and vb is None:
            delta_s = "-".center(_COL_W)
        elif va is None or vb is None:
            delta_s = "?".center(_COL_W)
        else:
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            delta_s = f"{sign}{delta:.4f}".center(_COL_W)

        print(f"  {label:<24}  {_fmt(va)}  {_fmt(vb)}  {delta_s}")

    print(f"{'─'*60}\n")


def _load_per_product(exp: ExperimentManager) -> Optional[Dict[str, Any]]:
    """Читает per-product метрики из eval_metrics*.json."""
    m = exp.metrics()
    if m is None:
        return None
    return m.get("per_product")


def _print_per_product(run_id: str) -> None:
    try:
        exp = ExperimentManager.load(run_id)
    except FileNotFoundError:
        print(f"[ERROR] run не найден: {run_id}")
        sys.exit(1)

    pp = _load_per_product(exp)
    if pp is None:
        print(f"[WARN] per_product метрики не найдены для {run_id}")
        return

    print(f"\n  Per-product  [{run_id}]\n{'─'*60}")
    print(f"  {'nm_id':<14}  {'Precision':>10}  {'Recall':>8}  {'MAE_raw':>9}  {'MAE_cal':>9}")
    print(f"{'─'*60}")
    for nm_id, vals in sorted(pp.items()):
        p   = vals.get("precision")
        r   = vals.get("recall")
        mr  = vals.get("mae_raw")
        mc  = vals.get("mae_calibrated")
        print(
            f"  {nm_id:<14}  {_fmt(p, 10)}  {_fmt(r, 8)}  {_fmt(mr, 9)}  {_fmt(mc, 9)}"
        )
    print(f"{'─'*60}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение результатов экспериментов из registry.jsonl"
    )
    parser.add_argument(
        "--last", type=int, default=None,
        metavar="N", help="Показать последние N запусков"
    )
    parser.add_argument(
        "--diff", nargs=2, metavar=("A", "B"),
        help="Сравнить два run_id (разница метрик)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Фильтр по config_name (подстрока)"
    )
    parser.add_argument(
        "--sort", type=str, default="timestamp",
        metavar="METRIC",
        help="Сортировать по: timestamp (по умолч.) или имени метрики"
    )
    parser.add_argument(
        "--per-product", type=str, default=None, metavar="RUN_ID",
        help="Показать per-product метрики для конкретного run_id"
    )
    args = parser.parse_args()

    if args.diff:
        _print_diff(args.diff[0], args.diff[1])
        return

    if args.per_product:
        _print_per_product(args.per_product)
        return

    entries = load_registry()
    if not entries:
        print("registry.jsonl пуст — запусков ещё не было.")
        return

    # Фильтр по config_name
    if args.config:
        entries = [e for e in entries if args.config in e.get("config_name", "")]

    # Сортировка
    sort_key = args.sort
    if sort_key == "timestamp":
        pass  # Уже в хронологическом порядке
    else:
        entries.sort(key=lambda e: (e.get(sort_key) is None, e.get(sort_key) or 0))

    # Срез последних N
    if args.last is not None:
        entries = entries[-args.last:]

    title = "Все запуски"
    if args.config:
        title += f" [config={args.config}]"
    if args.last:
        title += f" (последние {args.last})"
    if sort_key != "timestamp":
        title += f" → сортировка по {sort_key}"

    _print_table(entries, title=title)


if __name__ == "__main__":
    main()
