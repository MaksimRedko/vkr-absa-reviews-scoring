from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

run_dir = Path(st.session_state.get("run_dir", "."))
st.title("Embedding Space")
emb_path = run_dir / "embeddings_candidates.npy"
idx_path = run_dir / "embedding_index_candidates.csv"
if not emb_path.exists() or not idx_path.exists():
    st.info("No traced candidate embeddings.")
    st.stop()
emb = np.load(emb_path)
idx = pd.read_csv(idx_path)
if len(emb) > 1:
    xy = PCA(n_components=2, random_state=42).fit_transform(emb)
    plot_df = idx.copy()
    plot_df["x"] = xy[:, 0]
    plot_df["y"] = xy[:, 1]
    st.scatter_chart(plot_df, x="x", y="y")
else:
    st.info("Not enough embeddings.")
