from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

run_dir = Path(st.session_state.get("run_dir", "."))
st.title("Per Product")
path = run_dir / "metrics_track_a_vocab_only.csv"
if path.exists():
    st.dataframe(pd.read_csv(path), use_container_width=True)
else:
    st.info("No per-product metrics.")
