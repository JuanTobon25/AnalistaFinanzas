# app/main_interface.py
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="📄 Análisis de entrevistas + Métricas", layout="wide")

import pandas as pd
import mlflow
from app.rag_pipeline import build_chain
from app.docx_analysis import load_interview_docx
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Sidebar: seleccionar vista ---
modo = st.sidebar.radio("Selecciona una vista:", ["📄 Análisis de entrevistas", "📊 Métricas"])

# --- Paths ---
ROOT = Path(__file__).parents[1]
VECTOR_DIR = ROOT / "vectorstore"
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "app" / "prompts"

# --- Cargar vectorstore y chain ---
def cargar_vectordb_chain():
    index_file = VECTOR_DIR / "index.faiss"
    pickle_file = VECTOR_DIR / "index.pkl"

    if not VECTOR_DIR.exists() or not index_file.exists() or not pickle_file.exists():
        st.error("No se encontró un vectorstore válido. Crea el vectorstore primero desde los PDFs en 'data/'.")
        st.stop()
    
    try:
        vectordb = FAISS.load_local(str(VECTOR_DIR), OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        chain = build_chain(vectordb)
        return vectordb, chain
    except Exception as e:
        st.error("Error cargando vectorstore y chain: " + repr(e))
        st.stop()

vectordb, chain = cargar_vectordb_chain()

# --- Vista: análisis de entrevistas ---
if modo == "📄 Análisis de entrevistas":
    st.title("📄 Análisis automático de entrevistas")
    
    uploaded_file = st.file_uploader("Sube un documento Word (.docx) con entrevistas", type=["docx"])
    
    # Listar prompts disponibles
    prompt_files = sorted([p.name for p in PROMPTS_DIR.iterdir() if p.suffix in {".txt", ".md"}]) if PROMPTS_DIR.exists() else []
    prompt_choice = st.selectbox("Selecciona un prompt de análisis", prompt_files)
    
    if uploaded_file and prompt_choice:
        try:
            qa_pairs = load_interview_docx(uploaded_file)
            st.write("Fragmentos detectados (primeros 5):")
            st.write(qa_pairs[:5])
        except Exception as e:
            st.error("No se pudo procesar el documento Word: " + repr(e))
            st.stop()
        
        if st.button("🔍 Analizar entrevista"):
            with st.spinner("Analizando con IA..."):
                try:
                    joined_text = "\n".join(qa_pairs)
                    prompt_path = PROMPTS_DIR / prompt_choice
                    custom_prompt = prompt_path.read_text(encoding="utf-8")
                    
                    # Inputs que espera el chain: 'interview_text' y 'categoria_analisis'
                    chain_input = {
                        "interview_text": joined_text,
                        "categoria_analisis": custom_prompt
                    }
                    
                    analysis = chain.invoke(chain_input)
                    st.markdown("### 📊 Resultado del análisis")
                    st.write(analysis.get("answer", str(analysis)))
                except Exception as e:
                    st.error("Error al ejecutar el análisis con el chain: " + repr(e))

# --- Vista: métricas ---
elif modo == "📊 Métricas":
    st.title("📊 Resultados de Evaluación")

    client = mlflow.tracking.MlflowClient()
    experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

    if not experiments:
        st.warning("No se encontraron experimentos de evaluación.")
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox("Selecciona un experimento:", exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

    if not runs:
        st.warning("No hay ejecuciones registradas.")
        st.stop()

    # Armar dataframe
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        data.append({
            "Prompt": params.get("prompt_version"),
            "Chunk Size": int(params.get("chunk_size", 0)),
            "Correcto (LC)": metrics.get("lc_is_correct", 0)
        })

    df = pd.DataFrame(data)
    st.dataframe(df)

    # Agrupado
    st.subheader("📊 Promedio por configuración")
    grouped = df.groupby(["Prompt", "Chunk Size"]).agg({"Correcto (LC)": "mean"}).reset_index()
    grouped.rename(columns={"Correcto (LC)": "Precisión"}, inplace=True)
    grouped["config"] = grouped["Prompt"] + " | " + grouped["Chunk Size"].astype(str)
    st.bar_chart(grouped.set_index("config")["Precisión"])
