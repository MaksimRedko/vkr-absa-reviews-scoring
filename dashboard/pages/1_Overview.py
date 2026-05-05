from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics_overall import compute_all_metrics

def _default_run_dir() -> Path:
    if st.session_state.get("run_dir"):
        return Path(st.session_state["run_dir"])
    runs = sorted((ROOT / "results").glob("*_traced"))
    return runs[-1] if runs else ROOT


run_dir = _default_run_dir()
st.title("Overview")
metrics = compute_all_metrics(run_dir)
cols = st.columns(4)
for idx, key in enumerate([
    "detection_precision_track_a",
    "detection_recall_track_a",
    "detection_f1_track_a",
    "sentiment_mae_review",
]):
    cols[idx].metric(key, f"{metrics.get(key, float('nan')):.4f}")
st.metric("product_mae_n3", f"{metrics.get('product_mae_n3', float('nan')):.4f}")
summary = run_dir / "comparison_summary.md"
if summary.exists():
    st.markdown(summary.read_text(encoding="utf-8"))
for name in ["metrics_track_a_vocab_only.csv", "metrics_track_b_vocab_plus_discovery.csv"]:
    path = run_dir / name
    if path.exists():
        st.subheader(name)
        st.dataframe(pd.read_csv(path), use_container_width=True)
