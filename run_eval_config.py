"""
Запускает eval-пайплайн для заданного конфига, сохраняет все результаты
в изолированную папку эксперимента и регистрирует запуск в registry.jsonl.

Использование:
  python run_eval_config.py --config eval_configs/baseline.json
  python run_eval_config.py --config eval_configs/ab_fix3_a_t001.json --mode step12
  python run_eval_config.py --config eval_configs/baseline.json --mode step4 --mapping auto

Результаты:
  experiments/runs/{timestamp}_{config_name}/
    meta.json                 — git-хэш, timestamp, окружение
    config.json               — исходный eval-конфиг
    resolved_config.json      — конфиг с write_prefix, передаётся в eval_pipeline.py
    eval_results_step1_2.json — вывод step12
    eval_per_review.json      — предсказания per-review
    eval_metrics[_auto].json  — финальные метрики
    logs/
      step12.log
      step4.log
  experiments/registry.jsonl  — история всех запусков
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.experiment_manager import ExperimentManager


def _load_cfg(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_logged(cmd: list[str], log_path: Path) -> int:
    """Запускает subprocess, дублируя stdout/stderr в log_path и на экран."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
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
    return p.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Запуск eval-пайплайна с трекингом эксперимента"
    )
    parser.add_argument("--config", required=True, help="Path to eval config JSON")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "step12", "step4"],
        help="Какие шаги запускать (default: all)",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        choices=["manual", "auto"],
        help="Тип маппинга для step4 (переопределяет конфиг, default: manual)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Конфиг не найден: {cfg_path}")
        sys.exit(1)

    cfg = _load_cfg(cfg_path)
    config_name = cfg_path.stem  # "baseline" из "baseline.json"

    # Определяем mapping: CLI > конфиг > default
    mapping = args.mapping or str(cfg.get("mapping", "manual"))

    # ── Создаём эксперимент ───────────────────────────────────────────────────
    exp = ExperimentManager.create(config_name, cfg)

    # Resolved конфиг: write_prefix указывает на папку эксперимента
    resolved_cfg = dict(cfg)
    resolved_cfg["write_prefix"] = exp.write_prefix
    resolved_cfg_path = exp.run_dir / "resolved_config.json"
    resolved_cfg_path.write_text(
        json.dumps(resolved_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    py = sys.executable
    rc_total = 0

    # ── Step 1-2: discovery + sentiment ──────────────────────────────────────
    if args.mode in ("all", "step12"):
        rc = _run_logged(
            [py, "eval_pipeline.py", "step12", "--config", str(resolved_cfg_path)],
            log_path=exp.logs_dir / "step12.log",
        )
        if rc != 0:
            print(f"[WARN] step12 завершился с кодом {rc}", flush=True)
            rc_total = rc

    # ── Step 4: метрики ───────────────────────────────────────────────────────
    if args.mode in ("all", "step4"):
        rc = _run_logged(
            [
                py, "eval_pipeline.py", "step4",
                "--config", str(resolved_cfg_path),
                "--mapping", mapping,
            ],
            log_path=exp.logs_dir / "step4.log",
        )
        if rc != 0:
            print(f"[WARN] step4 завершился с кодом {rc}", flush=True)
            rc_total = rc

    # ── Регистрация и вывод метрик ────────────────────────────────────────────
    exp.finalize()

    metrics = exp.metrics()
    if metrics:
        summary = {
            "run_id":                  exp.run_id,
            "run_dir":                 str(exp.run_dir),
            "mapping":                 mapping,
            "macro_precision":         metrics.get("macro_precision"),
            "macro_recall":            metrics.get("macro_recall"),
            "micro_precision":         metrics.get("micro_precision"),
            "micro_recall":            metrics.get("micro_recall"),
            "global_mae_raw":          metrics.get("global_mae_raw"),
            "global_mae_calibrated":   metrics.get("global_mae_calibrated"),
            "global_mention_recall_review": metrics.get("global_mention_recall_review"),
        }
        print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"\n[WARN] Метрики не найдены в {exp.run_dir}")

    sys.exit(rc_total)


if __name__ == "__main__":
    main()
