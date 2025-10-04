# app/dashboard.py

import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="📊 Dashboard de Entrevistas", layout="wide")
st.title("📈 Análisis de Entrevistas - Chatbot RRHH")

# Ruta donde guardaremos resultados de análisis
RESULTS_FILE = os.path.join("data", "analysis_results.csv")

# Si no existe el archivo, inicializarlo
if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists(RESULTS_FILE):
    df = pd.DataFrame(columns=["documento", "prompt", "resultado"])
    df.to_csv(RESULTS_FILE, index=False)

# Cargar resultados
df = pd.read_csv(RESULTS_FILE)

if df.empty:
    st.warning("Aún no hay análisis de entrevistas cargados.")
    st.stop()

# Mostrar tabla completa
st.subheader("📋 Resultados individuales por entrevista")
st.dataframe(df)

# Agrupación por prompt
st.subheader("📊 Resumen por prompt utilizado")
grouped = df.groupby("prompt").agg(
    cantidad=("documento", "count")
).reset_index()
st.dataframe(grouped)

# Gráfico simple
st.bar_chart(grouped.set_index("prompt")["cantidad"])

# Filtro por documento
st.subheader("🔍 Explorar un documento específico")
doc_choice = st.selectbox("Selecciona un documento:", df["documento"].unique())

filtered = df[df["documento"] == doc_choice]
st.write(filtered)

