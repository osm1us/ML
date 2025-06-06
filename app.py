import streamlit as st
import os

import page_dataset_info
import page_visualizations
import page_prediction

st.set_page_config(
    page_title="Анализ и предсказание цен бриллиантов",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

def show_developer_info_page():
    st.title("Информация о разработчике")

    col1, col2 = st.columns([1, 3])

    with col1:
        image_path = os.path.join("assets", "me.jpg")
        try:
            st.image(image_path, caption="Это я", width=200)
        except FileNotFoundError:
            st.warning(f"Фото не найдено по пути: {image_path}. Убедитесь, что папка 'assets' существует в корне проекта и содержит 'me.jpg'.")
        except Exception as e:
            st.error(f"Не удалось загрузить фото: {e}")

    with col2:
        st.markdown("""
        ### Матвеев Дмитрий Андреевич
        **Группа:** ФИТ-231  
        **Тема РГР:** Разработка Web-приложения для инференса моделей ML и анализа данных
        """)
    st.markdown("---")
    st.info("Для начала работы выберите интересующий раздел в меню слева.")

PAGES = {
    "ℹ️ О разработчике": show_developer_info_page,
    "📊 Информация о датасете": page_dataset_info.show_page,
    "📈 Визуализации": page_visualizations.show_page,
    "💎 Прогнозирование цен": page_prediction.show_page
}

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти к разделу:", list(PAGES.keys()))

page_function = PAGES[selection]
page_function()