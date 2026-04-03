"""
Полный прогон по трём товарам из merged_checked_reviews.csv:

  1) scripts/run_pipeline_merged.py — ABSAPipeline (аспекты + агрегированные scores, как бек)
  2) eval_pipeline.py step12 — статистика + прогон как в eval (из CSV)
  3) eval_pipeline.py step4 --mapping auto — Recall, MAE, Mention recall
  4) diagnostics/diag_loss_funnel.py — воронка потерь

Логи: experiments/merged_suite_run/logs/*.log
Показывает: длительность шага, оценку оставшегося времени, heartbeat раз в 30s пока subprocess молчит.

Запуск из корня репозитория:
  python scripts/run_merged_suite.py
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_REL = "parser/reviews_batches/merged_checked_reviews.csv"
OUT_DIR = ROOT / "experiments" / "merged_suite_run"
LOG_DIR = OUT_DIR / "logs"


def _fmt_eta(sec: float) -> str:
    sec = max(0, sec)
    if sec < 90:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _tee_run(
    cmd: list[str],
    log_name: str,
    step_i: int,
    total_steps: int,
    prev_durations: list[float],
) -> tuple[int, float]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    t0 = time.perf_counter()

    # Оценка до старта
    eta_line = ""
    if prev_durations:
        avg = sum(prev_durations) / len(prev_durations)
        rem = total_steps - step_i + 1  # текущий + следующие
        eta_line = f" | по среднему шага ~ещё {_fmt_eta(avg * rem)}"
    print(
        f"\n{'='*70}\n"
        f"[Suite] ШАГ {step_i}/{total_steps}  |  старт {datetime.now():%H:%M:%S}"
        f"{eta_line}\n"
        f"$ {' '.join(cmd)}\n"
        f"-> {log_path}\n"
        f"{'='*70}\n",
        flush=True,
    )

    stop_hb = threading.Event()

    def _heartbeat() -> None:
        local_start = time.perf_counter()
        while not stop_hb.wait(30.0):
            elapsed = int(time.perf_counter() - local_start)
            print(
                f"  ⦿ шаг {step_i}/{total_steps} всё ещё работает… {elapsed}s с начала шага",
                flush=True,
            )

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# {datetime.now().isoformat()}\n# {' '.join(cmd)}\n\n")
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        p.wait()
        logf.write(f"\n# exit_code={p.returncode}\n")

    stop_hb.set()
    duration = time.perf_counter() - t0
    print(
        f"\n[Suite] шаг {step_i}/{total_steps} завершён за {_fmt_eta(duration)} "
        f"(код {p.returncode})\n",
        flush=True,
    )
    return p.returncode, duration


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / CSV_REL
    if not csv_path.is_file():
        print(f"Нет файла разметки: {csv_path}")
        return 1

    py = sys.executable
    prefix = str(OUT_DIR / "suite_")

    steps: list[tuple[str, list[str]]] = [
        (
            "01_pipeline_full.log",
            [
                py,
                str(ROOT / "scripts" / "run_pipeline_merged.py"),
                "--csv-path",
                CSV_REL,
                "--out-dir",
                str(OUT_DIR / "pipeline"),
            ],
        ),
        (
            "02_eval_step12.log",
            [
                py,
                str(ROOT / "eval_pipeline.py"),
                "step12",
                "--csv-path",
                CSV_REL,
                "--write-prefix",
                prefix,
            ],
        ),
        (
            "03_eval_step4_auto.log",
            [
                py,
                str(ROOT / "eval_pipeline.py"),
                "step4",
                "--csv-path",
                CSV_REL,
                "--mapping",
                "auto",
                "--write-prefix",
                prefix,
            ],
        ),
        (
            "04_diag_loss_funnel.log",
            [py, str(ROOT / "diagnostics" / "diag_loss_funnel.py"), "--csv-path", CSV_REL],
        ),
    ]

    total_steps = len(steps)
    rc = 0
    prev: list[float] = []
    t_suite = time.perf_counter()

    for i, (log_name, cmd) in enumerate(steps, start=1):
        code, dur = _tee_run(cmd, log_name, i, total_steps, prev)
        prev.append(dur)
        if code != 0:
            rc = code
            print(f"\n[WARNING] шаг {i} завершился с кодом {code}", flush=True)

    total_s = time.perf_counter() - t_suite
    print(f"\n{'='*70}\n[Suite] ВСЕ ШАГИ за {_fmt_eta(total_s)} (всего {total_s:.1f}s)\n")

    readme = OUT_DIR / "README.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "Структура:\n"
            "  logs/01_pipeline_full.log     — полный ABSAPipeline (7 шагов), аспекты+scores\n"
            "  logs/02_eval_step12.log       — статистика разметки + discovery/sentiment\n"
            "  logs/03_eval_step4_auto.log   — Recall, MAE, Mention recall (--mapping auto)\n"
            "  logs/04_diag_loss_funnel.log  — воронка потерь A–E\n"
            "  pipeline/                     — JSON + aspects_for_manual_precision.txt\n"
            "  suite_*eval_*                 — файлы метрик с префиксом suite_\n\n"
            "Ручная Precision: pipeline/aspects_for_manual_precision.txt\n"
            "Для каждого аспекта: TP / FP / Borderline.\n"
        )
    print(f"Готово. Описание: {readme}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
