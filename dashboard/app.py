from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def _runs() -> list[Path]:
    candidates = []
    for base in [ROOT / "results", ROOT / "benchmark" / "end_to_end" / "results"]:
        if base.exists():
            candidates.extend([path for path in base.iterdir() if path.is_dir()])
    return sorted(candidates, key=lambda path: path.name, reverse=True)


st.set_page_config(page_title="ABSA Traced Dashboard", layout="wide")
st.sidebar.title("Run")
runs = _runs()
if not runs:
    st.warning("No runs found.")
    st.stop()
selected = st.sidebar.selectbox("run_id", runs, format_func=lambda path: path.name)
st.session_state["run_dir"] = str(selected)
st.title("ABSA Traced Dashboard")
st.caption(str(selected))
st.write("Use the pages in the sidebar.")
