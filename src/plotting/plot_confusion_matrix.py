from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.plotting.common import figures_dir


def plot(run_dir: str | Path) -> Path:
    matrix_path = Path(run_dir) / "confusion_matrix_5x5.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path, index_col=0)
    else:
        matrix = pd.DataFrame(0, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
    out = figures_dir(run_dir) / "confusion_matrix_5x5.png"
    ax = matrix.plot(kind="bar", stacked=True, title="5x5 Confusion Matrix")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=160)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    print(plot(parser.parse_args().run_dir))
