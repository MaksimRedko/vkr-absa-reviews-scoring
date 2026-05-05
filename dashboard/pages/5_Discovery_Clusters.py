from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

run_dir = Path(st.session_state.get("run_dir", "."))
st.title("Discovery Clusters")
files = sorted(run_dir.glob("clusters_*.json"))
if not files:
    st.info("No traced cluster files.")
    st.stop()
path = st.selectbox("product", files, format_func=lambda item: item.stem)
payload = json.loads(path.read_text(encoding="utf-8"))
st.json(payload)
