# app/ui_streamlit.py
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.rag_pipeline import build_chain
from app.docx_analysis import load_interview_docx

# --- Usar imports recomendados por LangChain (langchain_community) ---
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="Análisis de Entrevistas - Psy", layout="centered")
st.title("📄 Análisis de entrevistas de manera automática")

# --- Paths ---
ROOT = Path(__file__).parents[1]
VECTOR_DIR = ROOT / "vectorstore"
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "app" / "prompts"

# --- Inicializar vectorstore ---
def crear_vectorstore_desde_pdf(pdf_path: Path, vector_dir: Path, chunk_size=500, chunk_overlap=50):
    st.info(f"Creando vectorstore desde PDF: {pdf_path.name} (esto puede tardar)...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    vector_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(vector_dir))
    st.success("Vectorstore creado y guardado en " + str(vector_dir))
    return vectordb

# --- Selección de PDF ---
if not DATA_DIR.exists():
    st.error("La carpeta 'data/' no existe. Coloca PDFs y reinicia.")
    st.stop()

pdf_files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".pdf"]
if not pdf_files:
    st.error("No se encontró ningún PDF en 'data/'. Agrega al menos uno y recarga.")
    st.stop()

PDF_FILE = pdf_files[0]

# --- Cargar o crear vectorstore ---
vectordb = None
index_file = VECTOR_DIR / "index.faiss"
pickle_file = VECTOR_DIR / "index.pkl"

if not VECTOR_DIR.exists() or not index_file.exists() or not pickle_file.exists():
    try:
        vectordb = crear_vectorstore_desde_pdf(PDF_FILE, VECTOR_DIR)
    except Exception as e:
        st.error("Error creando vectorstore desde PDF: " + repr(e))
        st.stop()
else:
    st.warning(
        "Se detectó un vectorstore existente. Debes confirmar que confías en este vectorstore para cargarlo."
    )
    allow = st.checkbox("Confirmo que confío en este vectorstore y deseo cargarlo")
    if allow:
        try:
            vectordb = FAISS.load_local(str(VECTOR_DIR), OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            st.success("Vectorstore cargado correctamente.")
        except Exception as e:
            st.error("No fue posible cargar el vectorstore existente: " + repr(e))
            st.stop()
    else:
        st.info("Elimina la carpeta 'vectorstore/' si deseas regenerarla desde el PDF.")
        st.stop()

# --- Construir chain ---
try:
    chain = build_chain(vectordb)
except Exception as e:
    st.error("Error al construir la cadena (chain) con el vectorstore: " + repr(e))
    st.stop()

# --- Cargar prompt ---
prompt_files = sorted([p.name for p in PROMPTS_DIR.iterdir() if p.suffix in {".txt", ".md"}]) if PROMPTS_DIR.exists() else []
prompt_choice = st.selectbox("Selecciona un prompt de análisis", prompt_files)

# --- Subida de Word ---
st.header("📄 Subir documento Word con entrevistas")
uploaded_file = st.file_uploader("Sube un documento Word (.docx)", type=["docx"])

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

                # Enviamos al chain solo los inputs que espera: 'interview_text' y 'categoria_analisis'
                chain_input = {
                    "interview_text": joined_text,
                    "categoria_analisis": custom_prompt
                }

                analysis = chain.invoke(chain_input)
                st.markdown("### 📊 Resultado del análisis")
                st.write(analysis.get("answer", str(analysis)))

            except Exception as e:
                st.error("Error al ejecutar el análisis con el chain: " + repr(e))
