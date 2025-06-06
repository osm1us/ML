import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import os

if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

MODEL_FILES = {
    "Нейронная сеть (MLP)": "mlp.h5",
    "XGBoost": "xgboost.pkl",
    "CatBoost": "catboost.pkl",
    "LightGBM": "lightgbm.pkl",
    "Bagging Regressor": "baggingregressor.pkl",
    "Polynomial Regression": "polinomialreg.pkl"
}

@st.cache_resource
def load_preprocessor():
    preprocessor_path = os.path.join('models', 'mlp_preprocessor.pkl')
    loaded_preprocessor = None
    if os.path.exists(preprocessor_path):
        try:
            loaded_preprocessor = joblib.load(preprocessor_path)
            st.sidebar.success("Препроцессор (mlp_preprocessor.pkl) успешно загружен.")
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки препроцессора: {str(e)}")
    else:
        st.sidebar.error(f"Файл препроцессора не найден: {preprocessor_path}")
    return loaded_preprocessor

@st.cache_resource
def load_selected_model(model_filename):
    model_path = os.path.join('models', model_filename)
    loaded_model = None
    if os.path.exists(model_path):
        try:
            if model_filename.endswith('.h5'):
                loaded_model = tf.keras.models.load_model(model_path)
                st.sidebar.info(f"Модель Keras ({model_filename}) успешно загружена.")
            elif model_filename.endswith('.pkl'):
                loaded_model = joblib.load(model_path)
                st.sidebar.info(f"Модель ({model_filename}) успешно загружена.")
            else:
                st.sidebar.error(f"Неизвестный формат файла модели: {model_filename}")
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки модели {model_filename}: {str(e)}")
    else:
        st.sidebar.error(f"Файл модели не найден: {model_path}")
    return loaded_model

def show_page():
    st.title("💎 Предсказание цены бриллианта")

    preprocessor = load_preprocessor()
    if preprocessor is None:
        st.error("Не удалось загрузить препроцессор. Проверьте сообщения в боковой панели.")
        return
    st.session_state.preprocessor = preprocessor

    selected_model_name = st.selectbox(
    "Выберите модель для предсказания:",
    options=list(MODEL_FILES.keys()),
    key="model_selector"
    )

    selected_model_filename = MODEL_FILES[selected_model_name]

    model_session_key = f"model_{selected_model_filename}"

    if model_session_key not in st.session_state or st.session_state[model_session_key] is None:
        with st.spinner(f'Загрузка модели "{selected_model_name}"...'):
            st.session_state[model_session_key] = load_selected_model(selected_model_filename)
    
    model = st.session_state[model_session_key]

    if model is None:
        st.error(f"Не удалось загрузить модель '{selected_model_name}'. Проверьте сообщения в боковой панели.")
        return
    st.sidebar.success(f"Активная модель: {selected_model_name}")

    expected_columns_order = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']

    with st.form("prediction_form"):
        st.subheader("Параметры бриллианта")
        
        col1, col2 = st.columns(2)
        
        with col1:
            carat = st.number_input("Вес (карат)", min_value=0.1, max_value=10.0, value=0.7, step=0.1, key="carat_input")
            cut = st.selectbox("Качество огранки", ["Ideal", "Premium", "Very Good", "Good", "Fair"], index=0, key="cut_input")
            color = st.selectbox("Цвет (от D до J)", ["D", "E", "F", "G", "H", "I", "J"], index=3, key="color_input")
            
        with col2:
            clarity = st.selectbox("Чистота", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"], index=3, key="clarity_input")
            depth = st.number_input("Глубина (%)", min_value=43.0, max_value=79.0, value=61.5, step=0.1, key="depth_input")
            table = st.number_input("Поверхность стола (%)", min_value=43.0, max_value=95.0, value=57.0, step=0.1, key="table_input")
        
        x = st.number_input("Длина (мм)", min_value=0.0, value=5.7, step=0.1, key="x_input")
        y = st.number_input("Ширина (мм)", min_value=0.0, value=5.7, step=0.1, key="y_input")
        z = st.number_input("Высота (мм)", min_value=0.0, value=3.5, step=0.1, key="z_input")
        
        submitted = st.form_submit_button("Предсказать цену")
        
        if submitted:
            try:
                input_data_dict = {
                    'carat': carat,
                    'cut': cut,
                    'color': color,
                    'clarity': clarity,
                    'depth': depth,
                    'table': table,
                    'x': x,
                    'y': y,
                    'z': z
                }
                input_df = pd.DataFrame([input_data_dict])
                
                processed_input = preprocessor.transform(input_df)
                
                prediction_array = model.predict(processed_input)
                if prediction_array.ndim == 2 and prediction_array.shape[0] == 1 and prediction_array.shape[1] == 1:
                    prediction_scalar = float(prediction_array[0, 0])
                elif prediction_array.ndim == 1 and prediction_array.shape[0] == 1:
                    prediction_scalar = float(prediction_array[0])
                else:
                    st.warning(f"Неожиданная форма массива предсказаний: {prediction_array.shape}. Используется первый элемент.")
                    prediction_scalar = float(prediction_array.flatten()[0]) 

                st.success(f"### Предсказанная цена: ${prediction_scalar:,.2f}")
                
            except Exception as e:
                st.error(f"Произошла ошибка при предсказании: {str(e)}")
                st.error("Убедитесь, что все поля заполнены корректно и модель/препроцессор загружены.")
    
    st.markdown("---")
    st.subheader("Или загрузите CSV файл для пакетного предсказания")
    
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"], key="csv_uploader")
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Предпросмотр загруженных данных:")
            st.dataframe(df_upload.head())
            
            missing_cols = [col for col in expected_columns_order if col not in df_upload.columns]
            if missing_cols:
                st.warning(f"В загруженном файле отсутствуют необходимые колонки: {', '.join(missing_cols)}. Пожалуйста, исправьте файл.")
            else:
                if st.button("Сделать пакетное предсказание", key="batch_predict_button"):
                    with st.spinner("Выполняется пакетное предсказание..."):
                        processed_df = preprocessor.transform(df_upload[expected_columns_order])
                        
                        predictions = model.predict(processed_df)
                        
                        result_df = df_upload.copy()
                        result_df['predicted_price'] = predictions.flatten()
                        
                        st.success("Предсказания успешно выполнены!")
                        st.dataframe(result_df)
                        
                        csv_output = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Скачать результаты (.csv)",
                            data=csv_output,
                            file_name="diamond_batch_predictions.csv",
                            mime="text/csv",
                            key="download_csv_button"
                        )
                        
        except Exception as e:
            st.error(f"Ошибка при обработке CSV файла: {str(e)}")
