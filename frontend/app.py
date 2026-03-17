"""
Streamlit UI для Aspecta AI — ABSA Pipeline v2.

Улучшения:
  - UI-алерт "Мнения расходятся" для аспектов с высоким controversy
  - Аспекты с 0 упоминаний скрыты
  - Ключевые слова кластера рядом с названием аспекта
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Aspecta AI", layout="wide")
API_URL = "http://127.0.0.1:8000"

CONTROVERSY_THRESHOLD = 1.2

st.title("Aspecta: Умный рейтинг товаров")
st.markdown("Personalized Aspect-Based Sentiment Analysis System")

# --- САЙДБАР ---
with st.sidebar:
    st.header("Настройки анализа")

    try:
        resp = requests.get(f"{API_URL}/products/top", timeout=5)
        if resp.status_code == 200:
            products = resp.json()
            options = {
                f"ID: {p['nm_id']} ({p['review_count']} отзывов)": p["nm_id"]
                for p in products
            }
            selected_label = st.selectbox("Выберите товар:", list(options.keys()))
            selected_nm_id = options[selected_label]
        else:
            st.error("Ошибка API. Бэкенд запущен?")
            selected_nm_id = None
    except Exception:
        st.error("Не удалось соединиться с API. Запустите: uvicorn app.main:app")
        selected_nm_id = None

    limit_reviews = st.slider("Сколько отзывов анализировать?", 10, 500, 100)
    analyze_btn = st.button("Запустить анализ", type="primary")

# --- ЗАПУСК АНАЛИЗА ---
if analyze_btn and selected_nm_id:
    with st.spinner(f"Анализируем отзывы (ID: {selected_nm_id})..."):
        try:
            payload = {"nm_id": selected_nm_id, "limit": limit_reviews}
            response = requests.post(f"{API_URL}/analyze", json=payload, timeout=300)
            if response.status_code == 200:
                st.session_state["result"] = response.json()
            else:
                st.error(f"Ошибка анализа: {response.text}")
        except Exception as e:
            st.error(f"Ошибка соединения: {e}")

# --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ---
if "result" in st.session_state:
    res = st.session_state["result"]
    all_aspects = res.get("aspects", {})
    aspect_keywords = res.get("aspect_keywords", {})

    # Фильтруем аспекты с 0 упоминаний
    aspects = {
        name: data for name, data in all_aspects.items()
        if data.get("mentions", 0) > 0
    }

    if not aspects:
        st.warning("Аспекты не найдены. Попробуйте увеличить количество отзывов.")
    else:
        # Сортируем по mentions (по убыванию)
        aspects_sorted = dict(
            sorted(aspects.items(), key=lambda x: x[1].get("mentions", 0), reverse=True)
        )

        # Топ-8 для радара и ползунков
        MAX_RADAR = 8
        top_aspects = dict(list(aspects_sorted.items())[:MAX_RADAR])
        other_aspects = dict(list(aspects_sorted.items())[MAX_RADAR:])

        # Метаданные
        st.caption(
            f"Товар: {res.get('product_id')} | "
            f"Отзывов: {res.get('reviews_processed')} | "
            f"Время: {res.get('processing_time', 0):.1f}s | "
            f"Аспектов: {len(aspects)} (показано топ-{len(top_aspects)})"
        )

        col1, col2 = st.columns([2, 1])

        # --- Radar Chart (только топ-8) ---
        with col1:
            st.subheader("Лепестковая диаграмма (топ-8)")

            categories = list(top_aspects.keys())
            values = [d["score"] for d in top_aspects.values()]

            # Замыкаем круг
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]

            fig = go.Figure(
                data=[
                    go.Scatterpolar(
                        r=values_closed,
                        theta=categories_closed,
                        fill="toself",
                        name="Аспекты",
                        line_color="deepskyblue",
                    )
                ],
                layout=go.Layout(
                    polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
                    showlegend=False,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Персонализация (только топ-8) ---
        with col2:
            st.subheader("Персонализация")
            st.caption("Настройте важность топ-аспектов:")

            user_weights = {}
            for aspect_name, data in top_aspects.items():
                kw_list = aspect_keywords.get(aspect_name, [])
                kw_display = ", ".join(kw_list[:4]) if kw_list else ""

                label = aspect_name

                user_weights[aspect_name] = st.slider(
                    label,
                    0.0, 1.0, 0.5, step=0.1,
                    help=f"Ключевые слова: {kw_display}" if kw_display else None,
                )

            # Персональный рейтинг (только по топ-8)
            weighted_sum = sum(
                top_aspects[a]["score"] * w for a, w in user_weights.items()
            )
            total_weight = sum(user_weights.values())
            final_rating = weighted_sum / total_weight if total_weight > 0 else 0

            st.divider()
            st.metric(
                label="Ваш персональный рейтинг",
                value=f"{final_rating:.2f} / 5.0",
            )

        # --- Детальная таблица (ВСЕ аспекты) ---
        st.subheader("Детальная статистика (все аспекты)")

        rows = []
        for name, data in aspects_sorted.items():
            kw = aspect_keywords.get(name, [])[:5]
            rows.append({
                "Аспект": name,
                "Рейтинг": data["score"],
                "Сырое среднее": data.get("raw_mean", "-"),
                "Споры": data.get("controversy", 0),
                "Упоминаний": data.get("mentions", 0),
                "Ключевые слова": ", ".join(kw),
            })

        df_stats = pd.DataFrame(rows).set_index("Аспект")
        st.dataframe(
            df_stats.style.highlight_max(
                axis=0, subset=["Рейтинг"], color="lightgreen"
            ).highlight_min(
                axis=0, subset=["Рейтинг"], color="#ffcccc"
            ),
            use_container_width=True,
        )

        # --- Алерты "Мнения расходятся" ---
        controversial = [
            (name, data)
            for name, data in aspects.items()
            if data.get("controversy", 0) >= CONTROVERSY_THRESHOLD
        ]
        if controversial:
            st.subheader("Мнения расходятся")
            for name, data in controversial:
                controversy_val = data["controversy"]
                kw = ", ".join(aspect_keywords.get(name, [])[:5])
                st.warning(
                    f"**{name}** (разброс: {controversy_val:.2f}) — "
                    f"отзывы противоречивы. Ключевые слова: {kw}"
                )
