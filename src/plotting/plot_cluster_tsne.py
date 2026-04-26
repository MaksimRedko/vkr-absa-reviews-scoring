from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.plotting.common import figures_dir


def plot(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    arrays = [np.load(path) for path in run_path.glob("cluster_centroids_*.npy") if np.load(path).size]
    out = figures_dir(run_dir) / "cluster_tsne.png"
    if not arrays:
        pd.DataFrame({"x": [], "y": []}).plot.scatter(x="x", y="y").figure.savefig(out, dpi=160)
        return out
    mat = np.vstack(arrays)
    xy = PCA(n_components=2, random_state=42).fit_transform(mat) if len(mat) > 1 else np.zeros((1, 2))
    ax = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}).plot.scatter(x="x", y="y", title="Discovery Cluster Centroids")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=160)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    print(plot(parser.parse_args().run_dir))
