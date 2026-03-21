from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict


def _load_cfg(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to eval config json")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "step12", "step4"],
        help="Which eval steps to run",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path)
    prefix = str(cfg.get("write_prefix", "")).strip()
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    if args.mode in ("all", "step12"):
        _run([sys.executable, "eval_pipeline.py", "step12", "--config", str(cfg_path)])
    if args.mode in ("all", "step4"):
        _run([sys.executable, "eval_pipeline.py", "step4", "--config", str(cfg_path)])

    metrics_path = Path(f"{prefix}eval_metrics.json")
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        print(
            json.dumps(
                {
                    "metrics_file": str(metrics_path),
                    "macro_precision": m.get("macro_precision"),
                    "macro_recall": m.get("macro_recall"),
                    "micro_precision": m.get("micro_precision"),
                    "micro_recall": m.get("micro_recall"),
                    "global_mae_raw": m.get("global_mae_raw"),
                    "global_mae_calibrated": m.get("global_mae_calibrated"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
