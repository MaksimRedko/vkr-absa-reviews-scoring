from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

# ВАЖНО: Streamlit запускает скрипт так, что sys.path[0] = папка скрипта.
# Поэтому используем локальный импорт (из той же папки), без `benchmark.*`.
from data_sources import (  # type: ignore
    build_wb_index,
    build_ym_index,
    get_ym_venue_meta,
    load_wb_index,
    load_wb_reviews,
    load_ym_index,
    load_ym_reviews,
)


BENCH_DIR = Path("benchmark")
BATCHES_DIR = BENCH_DIR / "batches"
LABELS_DIR = BENCH_DIR / "labels"
PROMPT_PATH = BENCH_DIR / "prompts" / "labeling_system_prompt.txt"


DONE_PREFIX = "DONE__"


def ensure_dirs() -> None:
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


def load_system_prompt() -> str:
    if PROMPT_PATH.is_file():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return ""


def _render_rating_hist(df: pd.DataFrame, title: str) -> None:
    if df.empty or "rating" not in df.columns:
        st.info("Нет данных для графика рейтингов.")
        return
    c = df["rating"].value_counts(dropna=False).sort_index().reset_index()
    c.columns = ["rating", "count"]
    st.plotly_chart(px.bar(c, x="rating", y="count", title=title), use_container_width=True)


def _render_len_hist(df: pd.DataFrame, text_col: str, title: str) -> None:
    if df.empty or text_col not in df.columns:
        st.info("Нет данных для графика длины текста.")
        return
    x = df[text_col].fillna("").astype(str).str.len()
    tmp = pd.DataFrame({"text_len": x})
    st.plotly_chart(px.histogram(tmp, x="text_len", nbins=30, title=title), use_container_width=True)


def _compose_wb_text_row(row: pd.Series) -> str:
    parts = [
        str(row.get("full_text", "") or "").strip(),
        str(row.get("pros", "") or "").strip(),
        str(row.get("cons", "") or "").strip(),
    ]
    return " ".join([p for p in parts if p]).strip()


def _style_overview(df: pd.DataFrame) -> Any:
    if df.empty:
        return df

    def entropy_color(v: float) -> str:
        try:
            x = float(v)
        except Exception:
            return ""
        if x < 1.0:
            return "background-color: #4b1f1f; color: white;"
        if x < 1.5:
            return "background-color: #5a4a1a; color: white;"
        return "background-color: #1f4b2e; color: white;"

    styler = df.style
    if "entropy" in df.columns:
        styler = styler.map(entropy_color, subset=["entropy"])
    return styler


def _render_selectable_table(view: pd.DataFrame, id_col: str, state_key: str) -> Any:
    """Кликабельная таблица + хранение выбора в session_state."""
    if view.empty:
        st.dataframe(view, use_container_width=True)
        return None

    # Кликабельная таблица (single-row select)
    event = st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"table_{state_key}",
    )

    selected_id = st.session_state.get(state_key)
    rows = (event or {}).get("selection", {}).get("rows", []) if isinstance(event, dict) else []
    if rows:
        idx = int(rows[0])
        if 0 <= idx < len(view):
            selected_id = view.iloc[idx][id_col]
            st.session_state[state_key] = selected_id

    # Подсвечиваем текущую активную строку в компактной таблице
    if selected_id is not None:
        active = view[view[id_col] == selected_id]
        if not active.empty:
            st.markdown("**Активная строка:**")
            styler = active.style.set_properties(**{"background-color": "#1f3b2d", "color": "white"})
            st.dataframe(styler, use_container_width=True, hide_index=True)

    return selected_id


@dataclass
class BatchMeta:
    meta_path: Path
    batch_prefix: str
    product_label: str
    batch_size: int
    batches: Dict[str, List[Dict[str, Any]]]


def _meta_path_for_prefix(batch_prefix: str) -> Path:
    return BATCHES_DIR / f"{batch_prefix}_meta.json"


def _load_meta(meta_path: Path) -> BatchMeta:
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    return BatchMeta(
        meta_path=meta_path,
        batch_prefix=str(raw.get("id", meta_path.stem.replace("_meta", ""))),
        product_label=str(raw.get("product_label", "")),
        batch_size=int(raw.get("batch_size", 25)),
        batches=dict(raw.get("batches", {})),
    )


def _save_meta(meta: BatchMeta) -> None:
    payload = {
        "id": meta.batch_prefix,
        "product_label": meta.product_label,
        "batch_size": meta.batch_size,
        "batches": meta.batches,
    }
    meta.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_batch_text(product_label: str, reviews: List[Dict[str, Any]]) -> str:
    header = [
        "=== ЗАДАНИЕ ===",
        f"Товар/заведение: {product_label}",
        "",
        "Для каждого отзыва определи аспекты и оценки по шкале 1-5.",
        "Формат ответа — JSON массив (по одному объекту на отзыв, в том же порядке):",
        "",
        "[",
        '  {"Аспект1": оценка, "Аспект2": оценка},',
        '  {"Аспект1": оценка},',
        "  ...",
        "]",
        "",
        "Правила:",
        "- Аспект — конкретное свойство (Еда, Обслуживание, Интерьер, Цена, Чистота и т.д.)",
        "- 5 = явный позитив, 1 = явный негатив, 3 = нейтрально",
        "- Если аспект НЕ упомянут — НЕ включай",
        "- Одно слово на аспект, именительный падеж, с большой буквы",
        "- Если отзыв пустой или бессмысленный — пустой объект {}",
        "",
        "=== ОТЗЫВЫ ===",
        "",
    ]
    parts = list(header)
    for idx, r in enumerate(reviews, start=1):
        parts.append(f"--- Отзыв {idx} (id={r['id']}, rating={r.get('rating', '')}) ---")
        txt = str(r.get("text", "") or "").strip()
        parts.append(txt if txt else "(пустой отзыв)")
        parts.append("")
    parts.append("=== КОНЕЦ БАТЧА ===")
    parts.append("")
    return "\n".join(parts)


def _cut_into_batches(
    batch_prefix: str,
    product_label: str,
    reviews: List[Dict[str, Any]],
    batch_size: int,
) -> Tuple[List[Path], Path]:
    ensure_dirs()

    meta_path = _meta_path_for_prefix(batch_prefix)
    if meta_path.is_file():
        meta = _load_meta(meta_path)
    else:
        meta = BatchMeta(
            meta_path=meta_path,
            batch_prefix=batch_prefix,
            product_label=product_label,
            batch_size=batch_size,
            batches={},
        )

    meta.product_label = product_label
    meta.batch_size = batch_size

    created: List[Path] = []
    total = len(reviews)
    n_batches = int(math.ceil(total / batch_size)) if batch_size else 0

    for bi in range(1, n_batches + 1):
        start = (bi - 1) * batch_size
        end = min(bi * batch_size, total)
        chunk = reviews[start:end]
        batch_name = f"{batch_prefix}_batch_{bi:02d}"
        batch_path = BATCHES_DIR / f"{batch_name}.txt"

        # Идемпотентность: если файл существует, не перезаписываем.
        if not batch_path.is_file():
            batch_path.write_text(_format_batch_text(product_label, chunk), encoding="utf-8")
            created.append(batch_path)

        meta.batches[batch_name] = [
            {"position": i, "review_id": r["id"], "rating": r.get("rating")}
            for i, r in enumerate(chunk)
        ]

    _save_meta(meta)
    return created, meta_path


def _list_batch_files(include_done: bool = False) -> List[Path]:
    ensure_dirs()
    files = sorted(BATCHES_DIR.glob("*.txt"))
    if include_done:
        return files
    return [p for p in files if not p.name.startswith(DONE_PREFIX)]


def _find_meta_for_batch(batch_path: Path) -> Path:
    stem = batch_path.stem
    if stem.startswith(DONE_PREFIX):
        stem = stem[len(DONE_PREFIX) :]
    # prefix = "<batch_prefix>_batch_XX"
    if "_batch_" not in stem:
        raise ValueError(f"Bad batch name: {batch_path.name}")
    batch_prefix = stem.split("_batch_")[0]
    return _meta_path_for_prefix(batch_prefix)


def _validate_and_normalize_llm_json(text: str) -> List[Dict[str, int]]:
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError("Ожидался JSON-массив.")
    out: List[Dict[str, int]] = []
    for obj in raw:
        if obj is None or obj == {}:
            out.append({})
            continue
        if not isinstance(obj, dict):
            out.append({})
            continue
        clean: Dict[str, int] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            try:
                iv = int(v)
            except Exception:
                iv = 3
            iv = max(1, min(5, iv))
            clean[k] = iv
        out.append(clean)
    return out


def _build_stats(labels_map: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    unique_aspects = set()
    aspect_counts: Dict[str, int] = {}
    labeled_reviews = 0
    empty_reviews = 0
    for _, asp_dict in labels_map.items():
        if not asp_dict:
            empty_reviews += 1
            continue
        labeled_reviews += 1
        for asp in asp_dict:
            unique_aspects.add(asp)
            aspect_counts[asp] = aspect_counts.get(asp, 0) + 1
    return {
        "total_reviews": len(labels_map),
        "labeled_reviews": labeled_reviews,
        "empty_reviews": empty_reviews,
        "unique_aspects": sorted(unique_aspects),
        "aspect_counts": dict(sorted(aspect_counts.items(), key=lambda x: x[0])),
    }


def _upsert_parsed_labels(batch_prefix: str, batch_name: str, labels_list: List[Dict[str, int]]) -> Path:
    meta_path = _meta_path_for_prefix(batch_prefix)
    meta = _load_meta(meta_path)

    # batch_name может быть DONE__*
    original_batch_name = batch_name[len(DONE_PREFIX) :] if batch_name.startswith(DONE_PREFIX) else batch_name
    entries = meta.batches.get(original_batch_name) or meta.batches.get(batch_name)
    if not entries:
        raise ValueError(f"Batch {batch_name} not found in meta {meta_path.name}")
    if len(entries) != len(labels_list):
        raise ValueError("Labels length mismatch while building parsed payload.")

    parsed_path = LABELS_DIR / f"{batch_prefix}_parsed.json"
    if parsed_path.is_file():
        payload = json.loads(parsed_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "id": batch_prefix,
            "labels": {},
            "stats": {},
        }

    labels_map: Dict[str, Dict[str, int]] = dict(payload.get("labels", {}))
    for entry, lab in zip(entries, labels_list):
        rid = str(entry["review_id"])
        labels_map[rid] = lab

    payload["labels"] = labels_map
    payload["stats"] = _build_stats(labels_map)
    parsed_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed_path


def _save_llm_answer_and_mark_done(batch_path: Path, llm_json_text: str) -> None:
    ensure_dirs()

    meta_path = _find_meta_for_batch(batch_path)
    meta = _load_meta(meta_path)

    stem = batch_path.stem
    original_stem = stem[len(DONE_PREFIX) :] if stem.startswith(DONE_PREFIX) else stem
    if original_stem not in meta.batches:
        raise ValueError(f"Батч {original_stem} не найден в meta {meta_path.name}")
    expected_n = len(meta.batches[original_stem])

    labels_list = _validate_and_normalize_llm_json(llm_json_text)
    if len(labels_list) != expected_n:
        raise ValueError(f"Размер не совпал: ожидалось {expected_n}, получено {len(labels_list)}.")

    # Сохраняем сырой ответ LLM
    out_labels_path = LABELS_DIR / f"{original_stem}.json"
    out_labels_path.write_text(
        json.dumps(labels_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _upsert_parsed_labels(meta.batch_prefix, original_stem, labels_list)

    # Переименовываем батч и обновляем meta, чтобы новый stem тоже был валиден
    done_name = f"{DONE_PREFIX}{original_stem}.txt"
    done_path = batch_path.parent / done_name
    if not done_path.is_file():
        batch_path.rename(done_path)

    done_stem = f"{DONE_PREFIX}{original_stem}"
    meta.batches[done_stem] = meta.batches.pop(original_stem)
    _save_meta(meta)


def _tab_eda() -> None:
    st.subheader("EDA / отбор товаров и заведений (лениво из `.db` и `.parquet`)")

    source = st.radio("Источник", ["WB (SQLite)", "Yandex Maps (Parquet)"], horizontal=True)

    if source.startswith("WB"):
        with st.expander("Индексы", expanded=False):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                force = st.button("Пересобрать WB index", use_container_width=True)
            with col_b:
                st.caption("WB index кэшируется в `benchmark/eda/cache/wb_products_index.parquet`.")
            if force:
                build_wb_index(force=True)
                st.success("WB index пересобран.")

        idx = load_wb_index(force_rebuild=False)
        min_nonempty = st.slider("Мин. непустых отзывов", 1, 2000, 50)
        view = idx[idx["n_nonempty"] >= min_nonempty].copy()
        preferred_cols = [
            "nm_id",
            "n_reviews",
            "n_nonempty",
            "avg_rating",
            "entropy",
            "skew",
            "non5_pct",
            "rating_1",
            "rating_2",
            "rating_3",
            "rating_4",
            "rating_5",
        ]
        existing_cols = [c for c in preferred_cols if c in view.columns]
        view = view[existing_cols].copy()
        st.caption("entropy: выше лучше; skew: ближе к 0 лучше; non5_pct: выше информативнее.")
        st.dataframe(_style_overview(view), use_container_width=True, hide_index=True)
        nm_id = _render_selectable_table(view, "nm_id", "selected_wb_nm_id")
        if nm_id:
            st.session_state["selected_source"] = "wb"
            st.session_state["selected_id"] = int(nm_id)

            n_load = st.slider("Подгрузить отзывов (для просмотра)", 20, 500, 100)
            reviews = load_wb_reviews(int(nm_id), limit=int(n_load))
            reviews = reviews.copy()
            reviews["text"] = reviews.apply(_compose_wb_text_row, axis=1)
            reviews = reviews[reviews["text"].str.strip() != ""]

            col1, col2 = st.columns(2)
            with col1:
                _render_rating_hist(reviews, "Распределение рейтингов (sample)")
            with col2:
                _render_len_hist(reviews, "text", "Распределение длины текста (sample)")

            for _, r in reviews.iterrows():
                st.markdown("---")
                st.markdown(f"**id:** `{r['id']}` | **rating:** {int(r['rating'])} | **date:** {r['created_date']}")
                st.write(r["text"])

    else:
        with st.expander("Индексы", expanded=False):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                force = st.button("Пересобрать YM index", use_container_width=True)
            with col_b:
                st.caption("YM index кэшируется в `benchmark/eda/cache/yandex_venues_index.parquet`.")
            if force:
                build_ym_index(force=True, batch_size=50_000)
                st.success("YM index пересобран.")

        idx = load_ym_index(force_rebuild=False)
        min_nonempty = st.slider("Мин. непустых отзывов", 1, 5000, 20)
        rubric = st.text_input("Фильтр по рубрике (substring)")
        view = idx[idx["n_nonempty"] >= min_nonempty].copy()
        if rubric.strip():
            view = view[view["rubrics"].fillna("").str.contains(rubric.strip(), case=False, regex=False)]
        preferred_cols = [
            "venue_key",
            "name_ru",
            "rubrics",
            "n_reviews",
            "n_nonempty",
            "avg_rating",
            "entropy",
            "skew",
            "non5_pct",
            "rating_1",
            "rating_2",
            "rating_3",
            "rating_4",
            "rating_5",
        ]
        existing_cols = [c for c in preferred_cols if c in view.columns]
        view = view[existing_cols].copy()
        st.caption("entropy: выше лучше; skew: ближе к 0 лучше; non5_pct: выше информативнее.")
        st.dataframe(_style_overview(view), use_container_width=True, hide_index=True)
        venue_key = _render_selectable_table(view, "venue_key", "selected_ym_venue_key")
        if venue_key:
            st.session_state["selected_source"] = "ym"
            st.session_state["selected_id"] = str(venue_key)

            meta = get_ym_venue_meta(str(venue_key))
            st.markdown(
                f"**Название:** {meta['name_ru']}  \n"
                f"**Рубрики:** {meta['rubrics']}  \n"
                f"**Адрес:** {meta['address']}  \n"
                f"**Непустых:** {int(meta['n_nonempty'])}  \n"
                f"**Средний рейтинг:** {meta['avg_rating']:.2f}"
            )

            n_load = st.slider("Подгрузить отзывов (для просмотра)", 20, 500, 100)
            reviews = load_ym_reviews(str(venue_key), limit=int(n_load))

            col1, col2 = st.columns(2)
            with col1:
                _render_rating_hist(reviews, "Распределение рейтингов (sample)")
            with col2:
                _render_len_hist(reviews, "text", "Распределение длины текста (sample)")

            for _, r in reviews.iterrows():
                st.markdown("---")
                st.markdown(f"**id:** `{r['id']}` | **rating:** {int(r['rating'])}")
                st.write(str(r["text"]).strip())


def _tab_cut_batches() -> None:
    st.subheader("Нарезка батчей для LLM")

    selected_source = st.session_state.get("selected_source")
    selected_id = st.session_state.get("selected_id")

    st.caption("Можно выбрать товар/заведение в EDA-вкладке — здесь он подхватится автоматически.")

    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Источник", ["wb", "ym"], index=0 if selected_source != "ym" else 1)
    with col2:
        default_id = "" if selected_id is None else str(selected_id)
        obj_id = st.text_input("ID (WB nm_id или YM venue_key)", value=default_id)

    product_label = st.text_input("Заголовок (опционально)", value="")
    batch_size = st.slider("batch_size", 10, 50, 25)
    max_reviews = st.slider("Сколько отзывов нарезать", 50, 1000, 200)
    seed = st.number_input("seed (для сэмпла)", value=42, step=1)

    if st.button("Нарезать батчи", type="primary"):
        ensure_dirs()

        if source == "wb":
            nm_id = int(obj_id)
            df = load_wb_reviews(nm_id, limit=None)
            df = df.copy()
            df["text"] = df.apply(_compose_wb_text_row, axis=1)
            df = df[df["text"].str.strip() != ""]
            if df.empty:
                st.error("Нет непустых отзывов для этого nm_id.")
                return
            if len(df) > int(max_reviews):
                df = df.sample(n=int(max_reviews), random_state=int(seed)).copy()
            df = df.sort_values("created_date").reset_index(drop=True)
            reviews = [
                {"id": str(r["id"]), "rating": int(r["rating"]), "text": str(r["text"])}
                for _, r in df.iterrows()
            ]
            prefix = f"wb_{nm_id}"
            label = product_label or f"WB товар nm_id={nm_id}"

        else:
            venue_key = str(obj_id).strip()
            meta = get_ym_venue_meta(venue_key)
            df = load_ym_reviews(venue_key, limit=None)
            if df.empty:
                st.error("Нет непустых отзывов для этого venue_key.")
                return
            if len(df) > int(max_reviews):
                df = df.sample(n=int(max_reviews), random_state=int(seed)).copy()
            df = df.reset_index(drop=True)
            reviews = [
                {"id": str(r["id"]), "rating": int(r["rating"]), "text": str(r["text"])}
                for _, r in df.iterrows()
            ]
            prefix = f"ym_{venue_key}"
            label = product_label or f"{meta['name_ru']} ({meta['rubrics']})"

        created, meta_path = _cut_into_batches(prefix, label, reviews, int(batch_size))
        st.success(f"Готово. Создано новых файлов: {len(created)}. Meta: {meta_path}")
        if created:
            st.write([p.name for p in created][:10])


def _tab_label_batches() -> None:
    st.subheader("Разметка батчей (без проводника)")

    system_prompt = load_system_prompt()
    with st.expander("System prompt (скопируй в LLM один раз)", expanded=False):
        st.code(system_prompt or "(файл не найден)", language="text")

    files = _list_batch_files(include_done=False)
    if not files:
        st.info("Нет необработанных батчей в `benchmark/batches/`.")
        return

    if "batch_index" not in st.session_state:
        st.session_state["batch_index"] = 0
    if st.session_state["batch_index"] >= len(files):
        st.session_state["batch_index"] = 0

    nav_col1, nav_col2, nav_col3 = st.columns([2, 1, 1])
    with nav_col1:
        batch_path = st.selectbox(
            "Батч",
            files,
            index=int(st.session_state["batch_index"]),
            format_func=lambda p: p.name,
        )
        st.session_state["batch_index"] = files.index(batch_path)
    with nav_col2:
        if st.button("Следующий батч", use_container_width=True):
            st.session_state["batch_index"] = (int(st.session_state["batch_index"]) + 1) % len(files)
            st.rerun()
    with nav_col3:
        if st.button("К первому", use_container_width=True):
            st.session_state["batch_index"] = 0
            st.rerun()

    batch_text = batch_path.read_text(encoding="utf-8")
    meta_path = _find_meta_for_batch(batch_path)
    meta = _load_meta(meta_path)
    stem = batch_path.stem
    original_stem = stem[len(DONE_PREFIX) :] if stem.startswith(DONE_PREFIX) else stem
    expected_n = len(meta.batches.get(original_stem, []))

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Батч (копируй и отправляй в LLM)**")
        st.code(batch_text, language="text")

    with col_right:
        st.markdown("**Ответ LLM (вставь JSON-массив)**")
        key = f"llm_answer::{batch_path.name}"
        llm_answer = st.text_area("", value=st.session_state.get(key, ""), height=450, key=key)

        st.caption(f"Ожидается объектов в JSON: **{expected_n}**")

        validate = st.button("Проверить JSON", use_container_width=True)
        if validate:
            try:
                arr = _validate_and_normalize_llm_json(llm_answer)
                if len(arr) != expected_n:
                    st.error(f"Количество объектов не совпадает: {len(arr)} vs {expected_n}")
                else:
                    st.success(f"JSON валиден, объектов: {len(arr)}")
            except Exception as e:
                st.error(f"Ошибка JSON: {e}")

        save = st.button("Сохранить и пометить DONE", type="primary", use_container_width=True)
        if save:
            try:
                _save_llm_answer_and_mark_done(batch_path, llm_answer)
                st.success("Сохранено. Батч помечен DONE.")
                if files:
                    st.session_state["batch_index"] = min(int(st.session_state["batch_index"]), max(len(files) - 2, 0))
                st.rerun()
            except Exception as e:
                st.error(str(e))


def main() -> None:
    ensure_dirs()
    st.set_page_config(page_title="ABSA Benchmark Dashboard", layout="wide")
    st.title("ABSA Benchmark Dashboard")

    tab1, tab2, tab3 = st.tabs(["EDA", "Нарезка батчей", "Разметка батчей"])
    with tab1:
        _tab_eda()
    with tab2:
        _tab_cut_batches()
    with tab3:
        _tab_label_batches()


if __name__ == "__main__":
    main()

