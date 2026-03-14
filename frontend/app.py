import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Конфигурация страницы
st.set_page_config(page_title="Aspecta AI", layout="wide")
API_URL = "http://127.0.0.1:8000"

st.title("🛍️ Aspecta: Умный рейтинг товаров")
st.markdown("Personzlized Aspect-Based Sentiment Analysis System")

# --- САЙДБАР (Настройки) ---
with st.sidebar:
    st.header("⚙️ Настройки анализа")

    # 1. Загрузка списка товаров
    try:
        resp = requests.get(f"{API_URL}/products/top")
        if resp.status_code == 200:
            products = resp.json()
            options = {f"ID: {p['nm_id']} ({p['review_count']} отзывов)": p['nm_id'] for p in products}
            selected_label = st.selectbox("Выберите товар:", list(options.keys()))
            selected_nm_id = options[selected_label]
        else:
            st.error("Ошибка API. Бэкенд запущен?")
            selected_nm_id = None
    except Exception:
        st.error("Не удалось соединиться с API. Запустите uvicorn.")
        selected_nm_id = None

    limit_reviews = st.slider("Сколько отзывов анализировать?", 10, 500, 50)

    analyze_btn = st.button("🚀 Запустить анализ", type="primary")

# --- ОСНОВНАЯ ЧАСТЬ ---
if analyze_btn and selected_nm_id:
    with st.spinner(f"Анализируем отзывы (ID: {selected_nm_id})... Это может занять время."):
        try:
            # Отправляем запрос на Бэкенд
            payload = {"nm_id": selected_nm_id, "limit": limit_reviews}
            response = requests.post(f"{API_URL}/analyze", json=payload)

            if response.status_code == 200:
                data = response.json()
                st.session_state['result'] = data  # Сохраняем в сессию
            else:
                st.error(f"Ошибка анализа: {response.text}")

        except Exception as e:
            st.error(f"Ошибка соединения: {e}")

# Если есть результаты - показываем
if 'result' in st.session_state:
    res = st.session_state['result']
    aspects = res.get("aspects", {})

    if not aspects:
        st.warning("Аспекты не найдены. Попробуйте увеличить количество отзывов.")
    else:
        # --- БЛОК 1: Визуализация (Radar Chart) ---
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Лепестковая диаграмма качества")

            # Подготовка данных для графика
            categories = list(aspects.keys())
            values = [d['score'] for d in aspects.values()]

            # Замыкаем круг графика
            categories += [categories[0]]
            values += [values[0]]

            fig = go.Figure(
                data=[
                    go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Aspects',
                        line_color='deepskyblue'
                    )
                ],
                layout=go.Layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 5])
                    ),
                    showlegend=False
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- БЛОК 2: Персонализация (Ползунки) ---
        with col2:
            st.subheader("🎚️ Персонализация")
            st.caption("Настройте важность аспектов для себя:")

            user_weights = {}
            for aspect in aspects.keys():
                user_weights[aspect] = st.slider(f"{aspect}", 0.0, 1.0, 0.5, step=0.1)

            # Расчет персонального рейтинга (на лету в UI)
            weighted_sum = sum(aspects[a]['score'] * w for a, w in user_weights.items())
            total_weight = sum(user_weights.values())

            final_rating = weighted_sum / total_weight if total_weight > 0 else 0

            st.divider()
            st.metric(label="⭐ Ваш Персональный Рейтинг", value=f"{final_rating:.2f} / 5.0")

        # --- БЛОК 3: Детализация (Таблица) ---
        st.subheader("📋 Детальная статистика")

        # Превращаем JSON в таблицу
        df_stats = pd.DataFrame.from_dict(aspects, orient='index')
        df_stats = df_stats.rename(columns={
            "score": "Рейтинг",
            "raw_mean": "Сырое среднее",
            "controversy": "Индекс споров",
            "mentions": "Упоминаний"
        })
        st.dataframe(df_stats.style.highlight_max(axis=0, color='lightgreen'))