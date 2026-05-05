from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_recalc.data_access import ReviewRecord, default_paths, discover_run_dirs, highlight_text, load_review_records
from manual_recalc.metrics import compute_detection_and_mae, metrics_to_frame
from manual_recalc.prompting import build_batch_prompt
from manual_recalc.storage import (
    build_batch_progress,
    clear_review,
    commit_batch,
    connect,
    dirty_review_ids,
    export_all,
    import_ai_draft,
    load_app_meta,
    load_overview_frames,
    load_review_state,
    now_iso,
    upsert_gold_decision,
    upsert_review_status,
    upsert_system_decision,
)

DECISION_OPTIONS = ["", "TP", "FP", "UNCLEAR", "DUPLICATE", "OUT_OF_SCOPE"]
SENTIMENT_OPTIONS = ["", "OK", "WRONG_POLARITY", "TOO_HIGH", "TOO_LOW", "NOT_EVALUATED"]
GOLD_STATUS_OPTIONS = ["", "FOUND", "FN", "UNCLEAR"]
REVIEW_STATUS_OPTIONS = ["not_started", "in_progress", "done", "needs_review"]

# Streamlit-селектор без key игнорирует index после создания виджета; отдельный ключ фиксирует переключение с первого клика.
BATCH_SELECTOR_KEY = "manual_recalc_batch_selector"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0e1117; }
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 1.4rem;
            max-width: 1680px;
        }
        section[data-testid="stSidebar"] {
            min-width: 272px;
            max-width: 272px;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        h1, h2, h3 { letter-spacing: -0.02em; }
        h1 { font-size: 2rem; margin-bottom: 0.2rem; }
        h2 { font-size: 1.5rem; margin-top: 0.4rem; }
        .mr-toolbar {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin: 0.8rem 0 1rem;
        }
        .mr-review-box {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.9rem;
            line-height: 1.58;
            font-size: 0.98rem;
        }
        .mr-pill {
            display: inline-block;
            padding: 0.14rem 0.52rem;
            margin-right: 0.38rem;
            border-radius: 999px;
            font-size: 0.78rem;
            white-space: nowrap;
        }
        .mr-pill-default {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            color: #d1d5db;
        }
        .mr-pill-accent {
            background: rgba(16,185,129,0.14);
            border: 1px solid rgba(16,185,129,0.35);
            color: #bbf7d0;
        }
        .mr-section-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }
        .mr-card-title {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            flex-wrap: wrap;
            margin-bottom: 0.55rem;
        }
        .mr-card-title strong {
            font-size: 1rem;
            line-height: 1.2;
        }
        .mr-label {
            font-size: 0.77rem;
            color: #9ca3af;
            margin-bottom: 0.18rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
        }
        div[data-testid="stMetric"] label {
            font-size: 0.8rem;
            color: #9ca3af;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            line-height: 1.05;
        }
        .stButton > button {
            border-radius: 10px;
            min-height: 2.35rem;
        }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextArea"] label {
            font-size: 0.82rem;
        }
        .st-emotion-cache-16txtl3,
        .st-emotion-cache-1kyxreq {
            gap: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, accent: bool = False) -> str:
    css = "mr-pill mr-pill-accent" if accent else "mr-pill mr-pill-default"
    return f'<span class="{css}">{escape(text)}</span>'


def section_title(text: str) -> None:
    st.markdown(f'<div class="mr-section-title">{escape(text)}</div>', unsafe_allow_html=True)


def card_title(main: str, *badges: str) -> None:
    st.markdown(
        f'<div class="mr-card-title"><strong>{escape(main)}</strong>{"".join(badges)}</div>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Manual Recalc", layout="wide")
inject_css()
st.title("Manual Recalc")
st.caption("Отдельный модуль для ручного пересчёта detection/sentiment метрик")


@st.cache_data(show_spinner=False)
def cached_reviews(run_dir_str: str, dataset_str: str) -> list[ReviewRecord]:
    return load_review_records(ROOT, Path(run_dir_str), Path(dataset_str))


def clear_imported_state_cache() -> None:
    prefixes = (
        "review_initialized::",
        "sys_decision::",
        "sys_map::",
        "sys_sentiment::",
        "sys_comment::",
        "gold_status::",
        "gold_match::",
        "gold_comment::",
        "review_status::",
    )
    for key in list(st.session_state.keys()):
        if key.startswith(prefixes):
            st.session_state.pop(key, None)


def batchify(review_ids: list[str], batch_size: int) -> list[list[str]]:
    return [review_ids[idx: idx + batch_size] for idx in range(0, len(review_ids), batch_size)]


def get_review_lookup(reviews: list[ReviewRecord]) -> dict[str, ReviewRecord]:
    return {item.review_id: item for item in reviews}


def exact_match_action(review: ReviewRecord) -> None:
    gold_names = {item.aspect for item in review.gold_aspects}
    for system in review.system_aspects:
        if system.aspect_name not in gold_names:
            continue
        st.session_state[f"sys_decision::{system.prediction_id}"] = "TP"
        st.session_state[f"sys_map::{system.prediction_id}"] = system.aspect_name
        st.session_state[f"gold_status::{review.review_id}::{system.aspect_name}"] = "FOUND"
        st.session_state[f"gold_match::{review.review_id}::{system.aspect_name}"] = system.prediction_id


def mark_remaining_system_fp(review: ReviewRecord) -> None:
    for system in review.system_aspects:
        key = f"sys_decision::{system.prediction_id}"
        if st.session_state.get(key):
            continue
        st.session_state[key] = "FP"
        st.session_state[f"sys_map::{system.prediction_id}"] = "NONE"


def mark_remaining_gold_fn(review: ReviewRecord) -> None:
    for gold in review.gold_aspects:
        key = f"gold_status::{review.review_id}::{gold.aspect}"
        if st.session_state.get(key):
            continue
        st.session_state[key] = "FN"


def initialize_review_state(review: ReviewRecord, saved: dict[str, dict], batch_id: str) -> None:
    # Статус и batch_id синхронизируем каждый ран (после commit из БД), иначе persist_review затирает done старыми значениями session_state.
    status_row = saved.get("status") or {}
    st.session_state[f"review_status::{review.review_id}"] = status_row.get("status", "not_started")
    st.session_state[f"review_batch::{review.review_id}"] = batch_id

    if f"review_initialized::{review.review_id}" in st.session_state:
        return
    for system in review.system_aspects:
        saved_row = saved["system"].get(system.prediction_id, {})
        st.session_state[f"sys_decision::{system.prediction_id}"] = saved_row.get("manual_decision", "")
        st.session_state[f"sys_map::{system.prediction_id}"] = saved_row.get("mapped_gold_aspect", "")
        st.session_state[f"sys_sentiment::{system.prediction_id}"] = saved_row.get("manual_sentiment_decision", "")
        st.session_state[f"sys_comment::{system.prediction_id}"] = saved_row.get("comment", "")
    for gold in review.gold_aspects:
        saved_gold = saved["gold"].get(gold.aspect, {})
        st.session_state[f"gold_status::{review.review_id}::{gold.aspect}"] = saved_gold.get("status", "")
        st.session_state[f"gold_match::{review.review_id}::{gold.aspect}"] = saved_gold.get("matched_system_prediction_id", "")
        st.session_state[f"gold_comment::{review.review_id}::{gold.aspect}"] = saved_gold.get("comment", "")
    st.session_state[f"review_initialized::{review.review_id}"] = True


def invalidate_review_init_cache(review_ids: Iterable[str]) -> None:
    for rid in review_ids:
        st.session_state.pop(f"review_initialized::{rid}", None)




def persist_review(conn, review: ReviewRecord, batch_id: str) -> None:
    saved = load_review_state(conn, review.review_id)
    changed = False

    for system in review.system_aspects:
        manual_decision = st.session_state.get(f"sys_decision::{system.prediction_id}", "")
        mapped_gold = st.session_state.get(f"sys_map::{system.prediction_id}", "")
        sentiment_decision = st.session_state.get(f"sys_sentiment::{system.prediction_id}", "")
        comment = st.session_state.get(f"sys_comment::{system.prediction_id}", "")
        has_content = any([manual_decision, mapped_gold, sentiment_decision, comment])
        prev = saved["system"].get(system.prediction_id, {})
        if not has_content and not prev:
            continue

        source = prev.get("source", "human")
        confirmed = bool(prev.get("confirmed_by_human", 0))
        committed = bool(prev.get("committed", 0))
        if source == "ai_prefill_draft" and manual_decision:
            source = "ai_prefill_human_confirmed"
            confirmed = True
        elif manual_decision:
            confirmed = True

        comparable_payload = {
            "manual_decision": manual_decision,
            "mapped_gold_aspect": mapped_gold,
            "manual_sentiment_decision": sentiment_decision,
            "comment": comment,
            "source": source,
            "confirmed_by_human": confirmed,
        }
        comparable_prev = {
            "manual_decision": prev.get("manual_decision", ""),
            "mapped_gold_aspect": prev.get("mapped_gold_aspect", ""),
            "manual_sentiment_decision": prev.get("manual_sentiment_decision", ""),
            "comment": prev.get("comment", ""),
            "source": prev.get("source", "human"),
            "confirmed_by_human": bool(prev.get("confirmed_by_human", 0)),
        }
        if comparable_payload != comparable_prev:
            committed = False

        payload = {
            "prediction_id": system.prediction_id,
            "review_id": review.review_id,
            "system_aspect": system.aspect_name,
            "system_rating": system.final_rating,
            "aspect_source": system.aspect_source,
            "manual_decision": manual_decision,
            "mapped_gold_aspect": mapped_gold,
            "manual_sentiment_decision": sentiment_decision,
            "comment": comment,
            "source": source,
            "confirmed_by_human": confirmed,
            "committed": committed,
            "updated_at": now_iso(),
        }
        if comparable_payload != comparable_prev or not prev:
            upsert_system_decision(conn, payload)
            changed = True

    for gold in review.gold_aspects:
        status = st.session_state.get(f"gold_status::{review.review_id}::{gold.aspect}", "")
        matched = st.session_state.get(f"gold_match::{review.review_id}::{gold.aspect}", "")
        comment = st.session_state.get(f"gold_comment::{review.review_id}::{gold.aspect}", "")
        has_content = any([status, matched, comment])
        prev = saved["gold"].get(gold.aspect, {})
        if not has_content and not prev:
            continue

        committed = bool(prev.get("committed", 0))
        comparable_payload = {
            "status": status,
            "matched_system_prediction_id": matched,
            "comment": comment,
        }
        comparable_prev = {
            "status": prev.get("status", ""),
            "matched_system_prediction_id": prev.get("matched_system_prediction_id", ""),
            "comment": prev.get("comment", ""),
        }
        if comparable_payload != comparable_prev:
            committed = False

        payload = {
            "review_id": review.review_id,
            "gold_aspect": gold.aspect,
            "gold_rating": gold.rating,
            "status": status,
            "matched_system_prediction_id": matched,
            "comment": comment,
            "committed": committed,
            "updated_at": now_iso(),
        }
        if comparable_payload != comparable_prev or not prev:
            upsert_gold_decision(conn, payload)
            changed = True

    review_status = st.session_state.get(f"review_status::{review.review_id}", "not_started")
    if review_status not in REVIEW_STATUS_OPTIONS:
        review_status = "in_progress"
    prev_status = saved.get("status") or {}
    committed = bool(prev_status.get("committed", 0))
    if review_status != prev_status.get("status"):
        committed = False
    if changed:
        committed = False
    if changed or saved.get("status") or review_status != "not_started":
        upsert_review_status(conn, review.review_id, batch_id, review_status, committed=committed)
        conn.commit()


def render_gold_card(review: ReviewRecord, gold) -> None:
    options = [""] + [system.prediction_id for system in review.system_aspects]
    option_labels = {
        "": "—",
        **{system.prediction_id: f"{system.aspect_name} [{system.aspect_source}] · {system.final_rating:.2f}" for system in review.system_aspects},
    }
    with st.container(border=True):
        card_title(gold.aspect, pill(f"{gold.rating:.2f}", accent=True))
        col_a, col_b = st.columns([1, 1.35])
        with col_a:
            st.markdown('<div class="mr-label">Статус</div>', unsafe_allow_html=True)
            st.selectbox(
                f"Статус {gold.aspect}",
                GOLD_STATUS_OPTIONS,
                key=f"gold_status::{review.review_id}::{gold.aspect}",
                label_visibility="collapsed",
            )
        with col_b:
            st.markdown('<div class="mr-label">Matched system</div>', unsafe_allow_html=True)
            st.selectbox(
                f"Matched system {gold.aspect}",
                options,
                format_func=lambda value: option_labels.get(value, value),
                key=f"gold_match::{review.review_id}::{gold.aspect}",
                label_visibility="collapsed",
            )
        st.markdown('<div class="mr-label">Комментарий</div>', unsafe_allow_html=True)
        st.text_input(
            f"Комментарий gold {gold.aspect}",
            key=f"gold_comment::{review.review_id}::{gold.aspect}",
            label_visibility="collapsed",
        )


def render_system_card(review: ReviewRecord, system) -> None:
    badges = [
        pill(system.aspect_source, accent=(system.aspect_source == "discovery")),
        pill(f"{system.final_rating:.2f}", accent=True),
    ]
    if not system.passed_relevance_filter:
        badges.append(pill("filtered"))

    with st.container(border=True):
        card_title(system.aspect_name, *badges)
        col_a, col_b, col_c = st.columns([1, 1.15, 1])
        with col_a:
            st.markdown('<div class="mr-label">Решение</div>', unsafe_allow_html=True)
            st.selectbox(
                f"Decision {system.aspect_name}",
                DECISION_OPTIONS,
                key=f"sys_decision::{system.prediction_id}",
                label_visibility="collapsed",
            )
        with col_b:
            st.markdown('<div class="mr-label">Map to gold</div>', unsafe_allow_html=True)
            st.selectbox(
                f"Map to gold {system.aspect_name}",
                ["", "NONE"] + [item.aspect for item in review.gold_aspects],
                key=f"sys_map::{system.prediction_id}",
                label_visibility="collapsed",
            )
        with col_c:
            st.markdown('<div class="mr-label">Тональность</div>', unsafe_allow_html=True)
            st.selectbox(
                f"Sentiment {system.aspect_name}",
                SENTIMENT_OPTIONS,
                key=f"sys_sentiment::{system.prediction_id}",
                label_visibility="collapsed",
            )
        st.markdown('<div class="mr-label">Комментарий</div>', unsafe_allow_html=True)
        st.text_input(
            f"Комментарий {system.aspect_name}",
            key=f"sys_comment::{system.prediction_id}",
            label_visibility="collapsed",
        )
        with st.expander(f"Показать источник: {system.aspect_name}"):
            if system.evidence_fragments:
                st.markdown(highlight_text(review.full_text, system.evidence_fragments), unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(system.evidence_fragments), use_container_width=True, hide_index=True)
            else:
                st.info("Evidence fragments не найдены.")
            if system.cluster_phrases:
                st.markdown("**Cluster phrases**")
                st.write(", ".join(system.cluster_phrases))
            info_left, info_right = st.columns(2)
            with info_left:
                st.markdown("**Premise**")
                st.write(system.premise_text or "—")
            with info_right:
                st.markdown("**Hypothesis**")
                st.write(system.hypothesis_text or "—")
            st.dataframe(
                pd.DataFrame([{
                    "entailment": system.p_entailment,
                    "neutral": system.p_neutral,
                    "contradiction": system.p_contradiction,
                }]),
                use_container_width=True,
                hide_index=True,
            )


default_run_dir, default_dataset = default_paths(ROOT)
available_runs = discover_run_dirs(ROOT)

with st.sidebar:
    run_dir = st.selectbox(
        "Run dir",
        options=available_runs if available_runs else [default_run_dir],
        format_func=lambda path: str(path.name),
        index=0,
    )
    dataset_path = st.text_input("Dataset CSV", value=str(default_dataset))
    db_path = st.text_input("SQLite DB", value=str(ROOT / "manual_recalc" / "data" / "manual_recalc.sqlite3"))
    batch_size = st.number_input("Batch size", min_value=5, max_value=50, value=25, step=5)
    status_filter = st.selectbox("Фильтр статуса", ["all", "not_started", "in_progress", "done", "needs_review"])
    source_filter = st.selectbox("Фильтр источника", ["all", "any_discovery", "only_vocab", "only_discovery"])
    show_only_fp = st.checkbox("Только отзывы с FP")
    show_only_fn = st.checkbox("Только отзывы с FN")

reviews = cached_reviews(str(run_dir), dataset_path)
review_lookup = get_review_lookup(reviews)
conn = connect(Path(db_path))
system_df, gold_df, status_df = load_overview_frames(conn)

status_map = {}
if not status_df.empty:
    for _, row in status_df.iterrows():
        rid = str(row["review_id"])
        raw_status = row["status"]
        status_map[rid] = str(raw_status) if pd.notna(raw_status) and str(raw_status).strip() else "not_started"

system_by_review = {}
if not system_df.empty:
    for review_id, frame in system_df.groupby("review_id", sort=False):
        system_by_review[str(review_id)] = frame

gold_by_review = {}
if not gold_df.empty:
    for review_id, frame in gold_df.groupby("review_id", sort=False):
        gold_by_review[str(review_id)] = frame

filtered_review_ids: list[str] = []
for review in reviews:
    saved_status = status_map.get(str(review.review_id), "not_started")
    if status_filter != "all" and saved_status != status_filter:
        continue
    if source_filter == "any_discovery" and not any(item.aspect_source == "discovery" for item in review.system_aspects):
        continue
    if source_filter == "only_vocab" and (not review.system_aspects or any(item.aspect_source == "discovery" for item in review.system_aspects)):
        continue
    if source_filter == "only_discovery" and (not review.system_aspects or any(item.aspect_source == "vocab" for item in review.system_aspects)):
        continue
    review_system = system_by_review.get(str(review.review_id), pd.DataFrame())
    review_gold = gold_by_review.get(str(review.review_id), pd.DataFrame())
    if show_only_fp and (review_system.empty or not review_system["manual_decision"].isin(["FP", "DUPLICATE"]).any()):
        continue
    if show_only_fn and (review_gold.empty or not review_gold["status"].isin(["FN"]).any()):
        continue
    filtered_review_ids.append(review.review_id)

repair_batch_ids_raw = load_app_meta(conn, "repair_batch_review_ids", default=[])
repair_batch_ids = [str(review_id) for review_id in repair_batch_ids_raw] if isinstance(repair_batch_ids_raw, list) else []
repair_batch_set = set(repair_batch_ids)
repair_batch_review_ids = [str(review_id) for review_id in filtered_review_ids if str(review_id) in repair_batch_set]
regular_review_ids = [str(review_id) for review_id in filtered_review_ids if str(review_id) not in repair_batch_set]

batch_groups: list[dict[str, list[str] | str]] = []
if repair_batch_review_ids:
    batch_groups.append({"batch_id": "repair_batch", "review_ids": repair_batch_review_ids})
for idx, review_ids in enumerate(batchify(regular_review_ids, int(batch_size))):
    batch_groups.append({"batch_id": f"batch_{idx + 1:03d}", "review_ids": review_ids})
batches = [list(batch["review_ids"]) for batch in batch_groups]

batch_summaries = build_batch_progress(batch_groups, status_df)
batch_labels = [item["label"] for item in batch_summaries]
if BATCH_SELECTOR_KEY not in st.session_state:
    st.session_state[BATCH_SELECTOR_KEY] = 0
if batch_labels:
    max_idx = len(batch_labels) - 1
    st.session_state[BATCH_SELECTOR_KEY] = max(0, min(int(st.session_state[BATCH_SELECTOR_KEY]), max_idx))


def reset_review_pointer() -> None:
    st.session_state["review_pointer"] = 0


with st.sidebar:
    if batch_labels:
        st.selectbox(
            "Пачка",
            options=list(range(len(batch_labels))),
            key=BATCH_SELECTOR_KEY,
            format_func=lambda idx: batch_labels[int(idx)],
            on_change=reset_review_pointer,
        )
        if st.button("Следующий батч", use_container_width=True):
            cur = int(st.session_state[BATCH_SELECTOR_KEY])
            st.session_state[BATCH_SELECTOR_KEY] = (cur + 1) % len(batch_labels)
            st.session_state["review_pointer"] = 0
            st.rerun()
    else:
        st.info("Нет отзывов под текущими фильтрами.")
    ai_json = st.text_area("Импорт JSON от ИИ", height=160)
    if st.button("Применить как черновик", use_container_width=True):
        try:
            summary = import_ai_draft(conn, json.loads(ai_json), review_lookup=review_lookup)
            clear_imported_state_cache()
            msg = f"Импортировано: system={summary['imported_system']}, gold={summary['imported_gold']}, reviews={summary['review_count']}"
            if summary["unresolved_system"]:
                st.warning(msg + f". Не удалось сматчить system rows: {len(summary['unresolved_system'])}")
            else:
                st.success(msg)
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Ошибка импорта: {exc}")

if not batch_groups:
    st.stop()

batch_index = int(st.session_state[BATCH_SELECTOR_KEY])
current_batch_group = batch_groups[batch_index]
current_batch = list(current_batch_group["review_ids"])
current_batch_id = str(current_batch_group["batch_id"])
current_batch_summary = batch_summaries[batch_index]
dirty_ids = dirty_review_ids(conn, current_batch)
batch_prompt = build_batch_prompt(ROOT / "manual_recalc" / "config", [review_lookup[review_id] for review_id in current_batch])

stats_a, stats_b, stats_c, stats_d = st.columns(4)
with stats_a:
    st.metric("Пачка", f"{batch_index + 1} / {len(batches)}")
with stats_b:
    done_global = int((status_df["status"].astype(str) == "done").sum()) if not status_df.empty else 0
    done_in_filter = (
        sum(1 for rid in filtered_review_ids if status_map.get(str(rid), "not_started") == "done")
        if filtered_review_ids
        else 0
    )
    st.metric("Размечено (в фильтре)", f"{done_in_filter} / {len(filtered_review_ids)}")
    st.caption(f"Готово по всей БД относительно набора загрузки: {done_global} / {len(reviews)}")
with stats_c:
    st.metric("Черновики", str(len(dirty_ids)))
with stats_d:
    st.metric("Прогресс пачки", f"{current_batch_summary['done']} / {current_batch_summary['total']}")

st.markdown('<div class="mr-toolbar">', unsafe_allow_html=True)
tool_a, tool_b, tool_c, tool_d = st.columns([1, 1, 1, 1.15])
with tool_a:
    if st.button("Сохранить пачку", use_container_width=True):
        commit_batch(conn, current_batch, current_batch_id)
        invalidate_review_init_cache(current_batch)
        st.success("Пачка сохранена.")
        st.rerun()
with tool_b:
    if st.button("Экспорт CSV и метрик", use_container_width=True):
        export_dir = ROOT / "manual_recalc" / "exports" / datetime.now().strftime("%Y%m%d_%H%M%S")
        export_all(conn, export_dir)
        review_gold_lookup = {
            (review.review_id, item.aspect): item.rating
            for review in reviews
            for item in review.gold_aspects
        }
        metrics_frame = metrics_to_frame(compute_detection_and_mae(*load_overview_frames(conn)[:2], review_gold_lookup))
        metrics_frame.to_csv(export_dir / "manual_metrics.csv", index=False, encoding="utf-8-sig")
        st.success(f"Экспортировано в {export_dir}")
        st.dataframe(metrics_frame, use_container_width=True)
with tool_c:
    if st.button("Показать текущие метрики", use_container_width=True):
        review_gold_lookup = {
            (review.review_id, item.aspect): item.rating
            for review in reviews
            for item in review.gold_aspects
        }
        st.dataframe(
            metrics_to_frame(compute_detection_and_mae(system_df, gold_df, review_gold_lookup)),
            use_container_width=True,
        )
with tool_d:
    st.download_button(
        "Скачать batch prompt",
        data=batch_prompt,
        file_name=f"{current_batch_id}_prompt.txt",
        mime="text/plain",
        use_container_width=True,
    )
st.markdown("</div>", unsafe_allow_html=True)
with st.expander("Batch prompt для ИИ"):
    st.code(batch_prompt)

if "review_pointer" not in st.session_state:
    st.session_state["review_pointer"] = 0
if st.session_state["review_pointer"] >= len(current_batch):
    st.session_state["review_pointer"] = max(0, len(current_batch) - 1)

section_title("Навигация")
nav_left, nav_mid, nav_right = st.columns([0.8, 4.8, 0.8])
with nav_left:
    if st.button("← Назад", disabled=st.session_state["review_pointer"] == 0, use_container_width=True):
        st.session_state["review_pointer"] -= 1
        st.rerun()
with nav_mid:
    safe_ptr = min(int(st.session_state["review_pointer"]), max(0, len(current_batch) - 1))
    review_id = st.selectbox(
        "Отзыв",
        options=current_batch,
        index=safe_ptr,
        format_func=lambda rid: str(rid),
        label_visibility="collapsed",
        key=f"mr_review_select_batch_{batch_index}",
    )
    st.session_state["review_pointer"] = current_batch.index(review_id)
with nav_right:
    if st.button("След. →", disabled=st.session_state["review_pointer"] >= len(current_batch) - 1, use_container_width=True):
        st.session_state["review_pointer"] += 1
        st.rerun()

review = review_lookup[review_id]
saved_state = load_review_state(conn, review.review_id)
initialize_review_state(review, saved_state, current_batch_id)
single_review_prompt = build_batch_prompt(ROOT / "manual_recalc" / "config", [review])

st.subheader("Отзыв")
meta_cols = st.columns(4)
meta_cols[0].metric("review_id", review.review_id)
meta_cols[1].metric("nm_id", str(review.nm_id))
meta_cols[2].metric("category", review.category)
meta_cols[3].metric("stars", f"{review.review_rating:.1f}")
st.markdown(f'<div class="mr-review-box">{escape(review.full_text)}</div>', unsafe_allow_html=True)

quick_a, quick_b, quick_c, quick_d = st.columns(4)
with quick_a:
    if st.button("Сопоставить 1-в-1", use_container_width=True):
        exact_match_action(review)
        st.rerun()
with quick_b:
    if st.button("Оставшиеся system → FP", use_container_width=True):
        mark_remaining_system_fp(review)
        st.rerun()
with quick_c:
    if st.button("Оставшиеся gold → FN", use_container_width=True):
        mark_remaining_gold_fn(review)
        st.rerun()
with quick_d:
    if st.button("Очистить решения", use_container_width=True):
        clear_review(conn, review.review_id)
        for key in list(st.session_state.keys()):
            if review.review_id in key:
                st.session_state.pop(key, None)
        st.rerun()

st.selectbox("Статус отзыва", REVIEW_STATUS_OPTIONS, key=f"review_status::{review.review_id}")
with st.expander("Prompt только для текущего отзыва"):
    st.download_button(
        "Скачать prompt текущего отзыва",
        data=single_review_prompt,
        file_name=f"{review.review_id}_prompt.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.code(single_review_prompt)

left_col, right_col = st.columns(2)
with left_col:
    section_title(f"Gold · {len(review.gold_aspects)}")
    for gold in review.gold_aspects:
        render_gold_card(review, gold)

with right_col:
    section_title(f"System · {len(review.system_aspects)}")
    for system in review.system_aspects:
        render_system_card(review, system)

persist_review(conn, review, current_batch_id)
