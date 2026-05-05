from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

run_dir = Path(st.session_state.get("run_dir", "."))
st.title("Per Category")
path = run_dir / "metrics_track_a_vocab_only.csv"
if not path.exists():
    st.info("No per-product metrics.")
    st.stop()
df = pd.read_csv(path)
cat = df.groupby("category_id", as_index=False).mean(numeric_only=True)
st.dataframe(cat, use_container_width=True)
if "sentiment_mae_review" in cat.columns:
    st.plotly_chart(px.bar(cat, x="category_id", y="sentiment_mae_review"), use_container_width=True)
