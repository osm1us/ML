import streamlit as st
from PIL import Image
import os
from page_prediction import MODEL_FILES

def show_page():
    st.title("📊 Визуальный анализ датасета бриллиантов")
    st.markdown("""
    На этой странице представлены визуализации, демонстрирующие взаимосвязи между характеристиками бриллиантов и их ценой.
    """)

    def display_image_with_caption(image_path, caption):
        try:
            image = Image.open(f"figures/{image_path}")
            st.image(image, caption=caption, use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось загрузить изображение {image_path}")

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

    metrics_files = [f for f in os.listdir("figures") if f.lower().endswith("metrics.png")]
    if metrics_files:
        st.markdown("---")
        st.header("Оценка моделей машинного обучения")
        
        image_prefix_to_display_name = {}
        for display_name_key, model_file_path_val in MODEL_FILES.items():
            base_model_filename = os.path.basename(model_file_path_val).split('.')[0]
            
            simple_name_for_file_calc = display_name_key
            if "(" in simple_name_for_file_calc:
                simple_name_for_file_calc = simple_name_for_file_calc[simple_name_for_file_calc.find("(")+1:simple_name_for_file_calc.find(")")]
            simple_name_for_file_calc = simple_name_for_file_calc.replace(" ", "")

            image_file_prefix_key = ""
            if not simple_name_for_file_calc or simple_name_for_file_calc.lower() == "нейроннаясеть":
                 image_file_prefix_key = base_model_filename.lower()
            else:
                 image_file_prefix_key = simple_name_for_file_calc.lower()
            
            image_prefix_to_display_name[image_file_prefix_key] = display_name_key

        for metric_file in metrics_files:
            try:
                # Extract the model prefix (e.g., "BaggingregressorMetrics.png" -> "baggingregressor")
                current_metric_file_prefix = metric_file.lower().removesuffix("metrics.png")
                
                display_model_name = image_prefix_to_display_name.get(
                    current_metric_file_prefix, 
                    current_metric_file_prefix.replace("_", " ").title() # Fallback if not in map
                )
                
                st.subheader(f"Модель: {display_model_name}")
                display_image_with_caption(metric_file, f"Метрики модели {display_model_name}")
            except Exception as e:
                st.warning(f"Ошибка при обработке файла метрик '{metric_file}': {e}. Отображается имя файла.")
                fallback_name = metric_file.lower().removesuffix("metrics.png").replace("_", " ").title()
                st.subheader(f"Модель: {fallback_name}")
                display_image_with_caption(metric_file, f"Метрики файла {metric_file}")
