"""
Четыре прогона eval: step12 + step4 --mapping manual (гипотезы A–D).
Результаты: hyp_<variant>_eval_results_step1_2.json, hyp_<variant>_eval_per_review.json,
hyp_<variant>_eval_metrics.json
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "configs.py"

VARIANTS = [
    ("A", "Автор доволен {aspect}", "Автор недоволен {aspect}"),
    ("B", "{aspect} — это хорошо", "{aspect} — это плохо"),
    ("C", "Автор хвалит {aspect}", "Автор критикует {aspect}"),
    ("D", "{aspect} отличного качества", "{aspect} ужасного качества"),
]


def patch_hypothesis_templates(pos: str, neg: str) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = re.sub(
        r'"hypothesis_template_pos":\s*"[^"]*"',
        f'"hypothesis_template_pos": "{pos}"',
        text,
        count=1,
    )
    text = re.sub(
        r'"hypothesis_template_neg":\s*"[^"]*"',
        f'"hypothesis_template_neg": "{neg}"',
        text,
        count=1,
    )
    CONFIG_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    py = sys.executable
    eval_py = ROOT / "eval_pipeline.py"
    for tag, pos, neg in VARIANTS:
        print(f"\n{'='*70}\nВариант {tag}\n{'='*70}")
        patch_hypothesis_templates(pos, neg)
        prefix = f"hyp_{tag.lower()}_"
        for step in ("step12", "step4"):
            if step == "step12":
                cmd = [py, str(eval_py), "step12", "--write-prefix", prefix]
            else:
                cmd = [py, str(eval_py), "step4", "--mapping", "manual", "--write-prefix", prefix]
            print("RUN:", " ".join(cmd))
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode != 0:
                print(f"ERROR exit {r.returncode} на {tag} {step}")
                sys.exit(r.returncode)
    # вернуть A в configs
    p0, n0 = VARIANTS[0][1], VARIANTS[0][2]
    patch_hypothesis_templates(p0, n0)
    print("\nconfigs.py: восстановлен вариант A.")


if __name__ == "__main__":
    main()
