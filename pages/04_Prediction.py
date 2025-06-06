import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from tensorflow.keras.models import load_model as load_keras_model

# Конфигурация страницы
st.set_page_config(page_title="Прогнозирование цены бриллианта", layout="centered")
st.title("💎 Прогнозирование цены бриллианта")

st.markdown("""
Используйте эту страницу для прогнозирования цены бриллианта на основе его характеристик.
Вы можете загрузить CSV-файл с данными или ввести параметры вручную.
""")

# Константы
MODELS_DIR = "models"

# Описания и ограничения признаков
FEATURES = {
    'carat': {
        'description': 'Вес бриллианта (карат)',
        'min': 0.2,
        'max': 5.01,
        'step': 0.01
    },
    'cut': {
        'description': 'Качество огранки',
        'options': {
            'Fair': 0,
            'Good': 1,
            'Very Good': 2,
            'Premium': 3,
            'Ideal': 4
        }
    },
    'color': {
        'description': 'Цвет (от J до D)',
        'options': {
            'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6
        }
    },
    'clarity': {
        'description': 'Чистота',
        'options': {
            'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7
        }
    },
    'depth': {
        'description': 'Глубина (%)',
        'min': 43, 'max': 79, 'step': 0.1
    },
    'table': {
        'description': 'Ширина верхней грани (мм)',
        'min': 43, 'max': 95, 'step': 0.1
    },
    'x': {
        'description': 'Длина (мм)',
        'min': 3.73, 'max': 10.74, 'step': 0.1
    },
    'y': {
        'description': 'Ширина (мм)',
        'min': 3.68, 'max': 58.9, 'step': 0.1
    },
    'z': {
        'description': 'Высота (мм)',
        'min': 1.07, 'max': 31.8, 'step': 0.1
    }
}

def load_model(model_path):
    """Загрузка модели машинного обучения из файла"""
    if model_path.endswith('.pkl'):
        return joblib.load(model_path)
    elif model_path.endswith('.h5'):
        return load_keras_model(model_path)
    elif model_path.endswith('.json'):
        if 'xgb' in model_path.lower():
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model
        elif 'lgb' in model_path.lower():
            return lgb.Booster(model_file=model_path)
    elif model_path.endswith('.cbm'):
        return CatBoostRegressor().load_model(model_path)
    raise ValueError(f"Неподдерживаемый формат модели: {model_path}")

def preprocess_input(input_data):
    """Предобработка входных данных для предсказания"""
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Приводим числовые колонки к правильному типу
    numeric_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Секция загрузки файла
st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Загрузите CSV файл с данными", 
    type=["csv"],
    help="Файл должен содержать колонки: carat, cut, color, clarity, depth, table, x, y, z"
)

# Выбор модели
st.sidebar.header("Выбор модели")
model_files = []
if os.path.exists(MODELS_DIR):
    model_files = [f for f in os.listdir(MODELS_DIR) 
                  if f.endswith(('.pkl', '.h5', '.json', '.cbm'))]

if not model_files:
    st.error(f"В папке {MODELS_DIR} не найдено моделей")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Выберите модель для предсказания",
    model_files,
    format_func=lambda x: x.split('.')[0].replace('_', ' ').title()
)

# Форма для ввода параметров
st.header("Ввод параметров бриллианта")
input_data = {}
col1, col2 = st.columns(2)

with col1:
    input_data['carat'] = st.number_input(
        FEATURES['carat']['description'],
        min_value=float(FEATURES['carat']['min']),
        max_value=float(FEATURES['carat']['max']),
        value=0.7,
        step=float(FEATURES['carat']['step']),
        help="Вес бриллианта в каратах"
    )
    
    input_data['cut'] = st.selectbox(
        FEATURES['cut']['description'],
        options=list(FEATURES['cut']['options'].keys()),
        index=3
    )
    
    input_data['color'] = st.selectbox(
        FEATURES['color']['description'],
        options=list(FEATURES['color']['options'].keys()),
        index=3
    )

with col2:
    input_data['clarity'] = st.selectbox(
        FEATURES['clarity']['description'],
        options=list(FEATURES['clarity']['options'].keys()),
        index=3
    )
    
    input_data['depth'] = st.number_input(
        FEATURES['depth']['description'],
        min_value=float(FEATURES['depth']['min']),
        max_value=float(FEATURES['depth']['max']),
        value=61.5,
        step=float(FEATURES['depth']['step'])
    )
    
    input_data['table'] = st.number_input(
        FEATURES['table']['description'],
        min_value=float(FEATURES['table']['min']),
        max_value=float(FEATURES['table']['max']),
        value=57.0,
        step=float(FEATURES['table']['step'])
    )

# Размеры бриллианта
st.subheader("Размеры бриллианта (мм)")
dim_col1, dim_col2, dim_col3 = st.columns(3)

with dim_col1:
    input_data['x'] = st.number_input(
        FEATURES['x']['description'],
        min_value=float(FEATURES['x']['min']),
        max_value=float(FEATURES['x']['max']),
        value=5.7,
        step=float(FEATURES['x']['step'])
    )

with dim_col2:
    input_data['y'] = st.number_input(
        FEATURES['y']['description'],
        min_value=float(FEATURES['y']['min']),
        max_value=float(FEATURES['y']['max']),
        value=5.7,
        step=float(FEATURES['y']['step'])
    )

with dim_col3:
    input_data['z'] = st.number_input(
        FEATURES['z']['description'],
        min_value=float(FEATURES['z']['min']),
        max_value=float(FEATURES['z']['max']),
        value=3.5,
        step=float(FEATURES['z']['step'])
    )

# Кнопка предсказания
predict_btn = st.button("Сделать прогноз", type="primary")

if predict_btn or uploaded_file is not None:
    try:
        # Загрузка модели
        model_path = os.path.join(MODELS_DIR, selected_model)
        model = load_model(model_path)
        
        # Подготовка данных
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Загружен файл с {len(df)} записями")
            st.dataframe(df.head())
        else:
            df = preprocess_input(input_data)
        
        # Преобразование категориальных признаков
        df = df.copy()
        df['cut'] = df['cut'].map(FEATURES['cut']['options'])
        df['color'] = df['color'].map(FEATURES['color']['options'])
        df['clarity'] = df['clarity'].map(FEATURES['clarity']['options'])
        
        # Получение предсказаний
        predictions = model.predict(df)
        
        # Отображение результатов
        st.header("Результаты прогнозирования")
        
        if len(predictions) == 1:
            st.metric(
                label="Прогнозируемая цена",
                value=f"${predictions[0]:,.2f}",
                help="Цена в долларах США"
            )
        else:
            results = pd.DataFrame({
                'carat': df['carat'],
                'cut': df['cut'].map({v: k for k, v in FEATURES['cut']['options'].items()}),
                'color': df['color'].map({v: k for k, v in FEATURES['color']['options'].items()}),
                'clarity': df['clarity'].map({v: k for k, v in FEATURES['clarity']['options'].items()}),
                'predicted_price': predictions
            })
            
            st.dataframe(
                results.style.format({
                    'predicted_price': '${:,.2f}',
                    'carat': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Кнопка для скачивания результатов
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Скачать результаты",
                csv,
                "diamond_predictions.csv",
                "text/csv",
                key='download-csv'
            )
    
    except Exception as e:
        st.error(f"Произошла ошибка при прогнозировании: {str(e)}")