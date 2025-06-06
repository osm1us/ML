import streamlit as st
from PIL import Image
import os

# Установка настроек страницы
st.set_page_config(page_title="Визуализация данных о бриллиантах", layout="wide")

# Заголовок страницы
st.title("📊 Визуальный анализ датасета бриллиантов")
st.markdown("""
На этой странице представлены визуализации, демонстрирующие взаимосвязи между характеристиками бриллиантов и их ценой.
""")

# Функция для отображения изображения с подписью
def display_image_with_caption(image_path, caption):
    try:
        image = Image.open(f"figures/{image_path}")
        st.image(image, caption=caption, use_column_width=True)
    except Exception as e:
        st.warning(f"Не удалось загрузить изображение {image_path}")

# Создание сетки для изображений
col1, col2 = st.columns(2)

with st.expander("📈 Основные распределения", expanded=True):
    st.subheader("Распределение цен на бриллианты")
    display_image_with_caption("price_distribution.png", "Распределение цен на бриллианты")
    st.markdown("""
    - Большинство бриллиантов в датасете имеют цену до $5000
    - Распределение имеет длинный хвост вправо, что типично для ценовых данных
    - Наблюдаются пики в районе круглых значений цен (например, $1500, $2000)
    """)

with col1:
    st.subheader("Зависимость цены от характеристик")
    
    st.markdown("### Влияние веса на цену")
    display_image_with_caption("price_vs_carat.png", "Зависимость цены от веса (карат)")
    st.markdown("""
    - Четко видна нелинейная зависимость цены от веса
    - Наблюдаются горизонтальные линии, что может указывать на округление цен
    - Разброс цен увеличивается с ростом веса бриллианта
    """)
    
    st.markdown("### Влияние качества огранки")
    display_image_with_caption("cut_vs_price.png", "Влияние качества огранки на цену")
    st.markdown("""
    - Бриллианты с идеальной огранкой (Ideal) в среднем дешевле, чем с премиальной
    - Это объясняется тем, что при лучшей огранке теряется больше веса камня
    - Наибольший разброс цен у бриллиантов с премиальной огранкой
    """)

with col2:
    st.subheader("Анализ категориальных признаков")
    
    st.markdown("### Распределение по цвету и чистоте")
    display_image_with_caption("color_clarity_distribution.png", "Распределение бриллиантов по цвету и чистоте")
    st.markdown("""
    - Наиболее распространены бриллианты с цветами G, H, I
    - Наибольшее количество бриллиантов имеют чистоту SI1, VS2, VS1
    - Бриллианты высшего качества (D, IF) встречаются реже
    """)
    
    st.markdown("### Корреляция признаков")
    display_image_with_caption("correlation_matrix.png", "Корреляционная матрица числовых признаков")
    st.markdown("""
    - Наибольшая корреляция с ценой у веса (carat) и размеров (x, y, z)
    - Сильная корреляция между размерами и весом (мультиколлинеарность)
    - Глубина (depth) имеет слабую корреляцию с ценой
    """)

# Добавляем раздел с метриками моделей, если файлы существуют
metrics_files = [f for f in os.listdir("figures") if "metrics" in f.lower()]
if metrics_files:
    st.markdown("---")
    st.header("Оценка моделей машинного обучения")
    
    for metric_file in metrics_files:
        model_name = metric_file.replace("_metrics_table.png", "").replace("_", " ").title()
        st.subheader(f"Модель: {model_name}")
        display_image_with_caption(metric_file, f"Метрики модели {model_name}")
