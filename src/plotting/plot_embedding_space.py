from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.plotting.common import figures_dir


def plot(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    emb = np.load(run_path / "embeddings_candidates.npy")
    out = figures_dir(run_dir) / "embedding_space.png"
    if len(emb) < 2:
        pd.DataFrame({"x": [], "y": []}).plot.scatter(x="x", y="y").figure.savefig(out, dpi=160)
        return out
    xy = PCA(n_components=2, random_state=42).fit_transform(emb)
    ax = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}).plot.scatter(x="x", y="y", s=4, title="Candidate Embedding Space")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=160)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    print(plot(parser.parse_args().run_dir))
