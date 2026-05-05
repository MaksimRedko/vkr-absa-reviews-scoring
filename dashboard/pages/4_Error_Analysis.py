from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]


def _default_run_dir() -> Path:
    if st.session_state.get("run_dir"):
        return Path(st.session_state["run_dir"])
    runs = sorted((ROOT / "results").glob("*_traced"))
    return runs[-1] if runs else ROOT


run_dir = _default_run_dir()
st.title("Error Analysis")
path = run_dir / "hard_cases.csv"
if not path.exists():
    st.info("No hard_cases.csv")
    st.stop()
df = pd.read_csv(path)
st.dataframe(df.head(30), use_container_width=True)
if not df.empty:
    row = st.selectbox("case", df.index, format_func=lambda idx: f"{df.loc[idx, 'review_id']} / {df.loc[idx, 'aspect']}")
    st.write(df.loc[row].to_dict())
