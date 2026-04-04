"""
ExperimentManager — централизованное управление запусками экспериментов.

Каждый эксперимент получает уникальную папку:
  experiments/runs/{YYYYMMDD_HHMMSS}_{config_name}/

Структура папки:
  meta.json                   — git-хэш, timestamp, python/platform, имя конфига
  config.json                 — оригинальный eval-конфиг (из .json)
  resolved_config.json        — конфиг с подставленным write_prefix (передаётся в eval_pipeline.py)
  eval_results_step1_2.json   — вывод step12 (discovery + sentiment)
  eval_per_review.json        — предсказания на уровне отзывов
  eval_metrics[_auto].json    — финальные метрики (manual или auto mapping)
  logs/
    step12.log                — stdout/stderr шага 1-2
    step4.log                 — stdout/stderr шага 4

experiments/registry.jsonl    — append-only лог всех завершённых запусков

Использование:

  exp = ExperimentManager.create("baseline", cfg_dict)
  print(exp.run_id)          # "20260403_143022_baseline"
  print(exp.write_prefix)    # "experiments/runs/20260403_143022_baseline/"
  exp.finalize()             # записывает метрики в registry.jsonl

  # Загрузить существующий:
  exp = ExperimentManager.load("20260403_143022_baseline")
  metrics = exp.metrics()
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
REGISTRY_FILE = ROOT / "experiments" / "registry.jsonl"


# ── Утилиты ──────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        r = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=str(ROOT), capture_output=True, timeout=5,
        )
        return r.returncode != 0
    except Exception:
        return False


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


# ── Основной класс ────────────────────────────────────────────────────────────

class ExperimentManager:
    """Управляет одним запуском эксперимента."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Фабричные методы ──────────────────────────────────────────────────────

    @classmethod
    def create(cls, config_name: str, config_dict: Dict[str, Any]) -> "ExperimentManager":
        """
        Создаёт новую папку эксперимента, сохраняет meta.json и config.json.
        Вызывать один раз перед запуском пайплайна.
        """
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"{timestamp}_{_safe_name(config_name)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        manager = cls(run_dir)

        # Оригинальный конфиг
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        # Метаданные окружения
        meta: Dict[str, Any] = {
            "run_id": run_dir.name,
            "config_name": config_name,
            "timestamp": datetime.now().isoformat(),
            "git_hash": _git_hash(),
            "git_dirty": _git_dirty(),
            "python_version": sys.version,
            "platform": platform.platform(),
        }
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[Experiment] run_id  : {run_dir.name}", flush=True)
        print(f"[Experiment] run_dir : {run_dir}", flush=True)
        return manager

    @classmethod
    def load(cls, run_id_or_path: str) -> "ExperimentManager":
        """
        Загружает существующий эксперимент по run_id (имени папки)
        или по абсолютному пути.
        """
        path = Path(run_id_or_path)
        if not path.is_absolute():
            path = RUNS_DIR / run_id_or_path
        if not path.exists():
            raise FileNotFoundError(f"Run dir not found: {path}")
        return cls(path)

    # ── Свойства ──────────────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def write_prefix(self) -> str:
        """
        Префикс для eval_pipeline.py --write-prefix.
        Оканчивается на '/', поэтому eval_pipeline не добавит лишний '_'.
        """
        return str(self.run_dir) + "/"

    # ── Чтение результатов ────────────────────────────────────────────────────

    def meta(self) -> Dict[str, Any]:
        p = self.run_dir / "meta.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def config(self) -> Dict[str, Any]:
        p = self.run_dir / "config.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def metrics(self) -> Optional[Dict[str, Any]]:
        """
        Читает метрики из папки эксперимента.
        Автоматически определяет имя файла (manual или auto mapping).
        """
        for name in ("eval_metrics.json", "eval_metrics_auto.json"):
            p = self.run_dir / name
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return None

    def step12(self) -> Optional[Dict[str, Any]]:
        p = self.run_dir / "eval_results_step1_2.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def per_review(self) -> Optional[Dict[str, Any]]:
        p = self.run_dir / "eval_per_review.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def is_complete(self) -> bool:
        """True если финальные метрики уже записаны."""
        return self.metrics() is not None

    def snapshot_writer(self, product_id: int, save_embeddings: bool = True):
        """
        Возвращает SnapshotWriter для одного продукта, привязанный к этому эксперименту.

        Пример:
            exp = ExperimentManager.create("baseline", cfg)
            pipeline.analyze_reviews_list(
                reviews, nm_id,
                snapshot_writer=exp.snapshot_writer(nm_id)
            )
        Снепшоты сохранятся в: {run_dir}/snapshots/nm{product_id}/
        """
        from src.snapshots import SnapshotWriter
        return SnapshotWriter(
            base_dir=self.run_dir / "snapshots",
            product_id=product_id,
            save_embeddings=save_embeddings,
        )

    # ── Регистрация результатов ───────────────────────────────────────────────

    def finalize(self) -> None:
        """
        Читает метрики из папки и добавляет одну строку в registry.jsonl.
        Безопасно вызывать повторно — дублирования не будет (проверяет run_id).
        """
        metrics = self.metrics() or {}
        meta = self.meta()

        entry: Dict[str, Any] = {
            "run_id": self.run_id,
            "config_name": meta.get("config_name", "?"),
            "timestamp": meta.get("timestamp", "?"),
            "git_hash": meta.get("git_hash", "?"),
            "git_dirty": meta.get("git_dirty", False),
            "run_dir": str(self.run_dir),
            # Топ-уровневые метрики
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "micro_precision": metrics.get("micro_precision"),
            "micro_recall": metrics.get("micro_recall"),
            "global_mae_raw": metrics.get("global_mae_raw"),
            "global_mae_calibrated": metrics.get("global_mae_calibrated"),
            "global_mention_recall_review": metrics.get("global_mention_recall_review"),
        }

        # Дедупликация: не добавлять если run_id уже в registry
        if REGISTRY_FILE.exists():
            existing = {
                json.loads(line).get("run_id")
                for line in REGISTRY_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            if self.run_id in existing:
                print(f"[Experiment] Already in registry: {self.run_id}", flush=True)
                return

        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[Experiment] Registered: {self.run_id}", flush=True)
        print(f"[Experiment] Registry  : {REGISTRY_FILE}", flush=True)


# ── Утилита для чтения реестра ────────────────────────────────────────────────

def load_registry() -> list[Dict[str, Any]]:
    """Возвращает все записи из registry.jsonl в хронологическом порядке."""
    if not REGISTRY_FILE.exists():
        return []
    entries = []
    for line in REGISTRY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries
