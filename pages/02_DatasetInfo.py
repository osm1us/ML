import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Описание данных", layout="wide")
st.title("Анализ датасета бриллиантов")

st.markdown("""
Данные содержат информацию о характеристиках бриллиантов. Цель — прогнозирование цены бриллианта (`price`) 
на основе его физических характеристик. Датасет содержит как числовые, так и категориальные признаки.
""")

try:
    df = pd.read_csv('post_diamonds.csv')
    
    df['table'] = df['table'].astype(int)
    
    dict_cut = {1: 'Fair', 2: 'Good', 3: 'Very Good', 4: 'Premium', 5: 'Ideal'}
    dict_color = {1: 'J', 2: 'I', 3: 'H', 4: 'G', 5: 'F', 6: 'E', 7: 'D'}
    dict_clarity = {1: 'I1', 2: 'SI2', 3: 'SI1', 4: 'VS2', 5: 'VS1', 6: 'VVS2', 7: 'VVS1', 8: 'IF'}
    
    df['cut'] = df['cut'].map(dict_cut)
    df['color'] = df['color'].map(dict_color)
    df['clarity'] = df['clarity'].map(dict_clarity)
    
    st.header("Предпросмотр данных")
    st.dataframe(df.head())
    
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    df = None

st.header("Структура датасета")

data_description = {
    "carat": "Вес бриллианта в каратах (1 карат = 0.2 грамма)",
    "cut": "Качество огранки (Fair, Good, Very Good, Premium, Ideal)",
    "color": "Цвет бриллианта (от J - наихудший до D - лучший)",
    "clarity": "Чистота бриллианта (I1 (наихудший), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (лучший))",
    "depth": "Общая глубина в процентах = z / среднее(x, y) = 2 * z / (x + y)",
    "table": "Ширина верхней грани относительно самой широкой точки (43-95)",
    "price": "Цена в долларах США",
    "x": "Длина в мм",
    "y": "Ширина в мм",
    "z": "Высота в мм"
}

st.subheader("Признаки и их описание:")
for feature, desc in data_description.items():
    st.markdown(f"**{feature}**: {desc}")

st.subheader("Целевая переменная")
st.markdown("`price` — цена бриллианта в долларах США.")

st.subheader("Тип задачи")
st.markdown("Задача регрессии: предсказание цены бриллианта на основе его характеристик.")

st.write(f"Количество признаков: {len(data_description)-1}")

if df is not None:
    st.write(f"Количество записей: {len(df)}")
    st.write(f"Диапазон цен: ${df['price'].min():,} - ${df['price'].max():,}")
    st.write(f"Средний вес: {df['carat'].mean():.2f} карат")

st.markdown("---")

st.header("Предобработка данных (EDA)")
st.markdown("""
**Преобразование категориальных признаков:**

- Признак `cut` (качество огранки) закодирован численно:
  - 1: Fair
  - 2: Good
  - 3: Very Good
  - 4: Premium
  - 5: Ideal

- Признак `color` (цвет) закодирован численно от 1 до 7:
  - 1: J (наименее ценный)
  - 2: I
  - 3: H
  - 4: G
  - 5: F
  - 6: E
  - 7: D (наиболее ценный)

- Признак `clarity` (чистота) закодирован численно от 1 до 8:
  - 1: I1 (наименьшая чистота)
  - 2: SI2
  - 3: SI1
  - 4: VS2
  - 5: VS1
  - 6: VVS2
  - 7: VVS1
  - 8: IF (наивысшая чистота)

**Преобразование типов данных:**
- Столбец `table` преобразован из float64 в int64

**Очистка данных:**
- Удалены дубликаты с помощью `drop_duplicates()`

**Анализ выбросов:**
- Проверены и обработаны выбросы в признаках x, y, z (размеры в мм)
- Проанализированы аномальные значения в цене и весе бриллиантов

**Корреляционный анализ:**
- Наибольшее влияние на цену оказывает вес бриллианта (carat)
- Размеры (x, y, z) сильно коррелируют с весом
- Качество огранки и чистота также оказывают значительное влияние на цену
""")
