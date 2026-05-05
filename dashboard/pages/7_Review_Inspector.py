from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui_data.build_review_view import build_review_view

def _default_run_dir() -> Path:
    if st.session_state.get("run_dir"):
        return Path(st.session_state["run_dir"])
    runs = sorted((ROOT / "results").glob("*_traced"))
    return runs[-1] if runs else ROOT


run_dir = _default_run_dir()
st.title("Review Inspector")
query_review_id = st.query_params.get("review_id", "")
review_id = st.text_input("review_id", value=str(query_review_id or ""))
if review_id:
    payload = build_review_view(run_dir, review_id)
    st.json(payload)
