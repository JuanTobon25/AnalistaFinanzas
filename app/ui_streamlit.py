# app/ui_streamlit.py
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.rag_pipeline import build_chain
from app.docx_analysis import load_interview_docx

# --- Imports recomendados por LangChain ---
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="Analizador de Entrevistas", layout="centered")
st.title("📄 Analizador de entrevistas - Psicología")

# --- Inicializar session state ---
st.session_state.setdefault("vectorstore_loaded", False)

# --- Paths ---
ROOT = Path(__file__).parents[1]
VECTOR_DIR = ROOT / "vectorstore"
DATA_DIR = ROOT / "data"
PROMPT_DIR = ROOT / "app" / "prompts"

# --- Verificar PDFs ---
if not DATA_DIR.exists():
    st.error("La carpeta 'data/' no existe. Coloca tus PDFs en data/ y recarga.")
    st.stop()

pdf_files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".pdf"]
if not pdf_files:
    st.error("No se encontró ningún PDF en data/. Agrega al menos un PDF.")
    st.stop()

PDF_FILE = pdf_files[0]

# --- Función para crear vectorstore desde PDF ---
def crear_vectorstore_desde_pdf(pdf_path: Path, vector_dir: Path, chunk_size=500, chunk_overlap=50):
    st.info(f"Creando vectorstore desde PDF: {pdf_path.name}...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)

    vector_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(vector_dir))
    st.success(f"Vectorstore creado y guardado en {vector_dir}")
    return vectordb

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
        "Se detectó un vectorstore existente. Cargarlo requiere deserializar un archivo pickle."
    )
    allow = st.checkbox("Confío en este vectorstore y deseo cargarlo")
    if allow:
        try:
            vectordb = FAISS.load_local(
                str(VECTOR_DIR),
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
            st.success("Vectorstore cargado correctamente.")
        except Exception as e:
            st.error("No fue posible cargar el vectorstore: " + repr(e))
            st.stop()
    else:
        st.info("Elimina la carpeta 'vectorstore/' para regenerarlo desde el PDF.")
        st.stop()

# --- Construir chain ---
try:
    chain = build_chain(vectordb)
except Exception as e:
    st.error("Error al construir el chain: " + repr(e))
    st.stop()

# --- Cargar documento Word ---
st.header("📄 Análisis de entrevistas (Word)")
uploaded_file = st.file_uploader("Sube un documento Word con la entrevista", type=["docx"])

# --- Selección de prompt ---
prompt_files = sorted([p.name for p in PROMPT_DIR.iterdir() if p.suffix in {".txt", ".md"}]) if PROMPT_DIR.exists() else []
prompt_choice = st.selectbox("Selecciona un prompt", prompt_files)

if uploaded_file and prompt_choice:
    try:
        qa_pairs = load_interview_docx(uploaded_file)
        st.write("Vista previa de los primeros 5 fragmentos de la entrevista:")
        st.write(qa_pairs[:5])
    except Exception as e:
        st.error("No se pudo procesar el documento Word: " + repr(e))
        st.stop()

    if st.button("🔍 Analizar entrevista"):
        with st.spinner("Analizando con IA..."):
            # Combinamos todo el texto de la entrevista
            interview_text = "\n".join(qa_pairs)
            prompt_path = PROMPT_DIR / prompt_choice
            try:
                custom_prompt = prompt_path.read_text(encoding="utf-8")
            except Exception as e:
                st.error("No se pudo leer el prompt seleccionado: " + repr(e))
                custom_prompt = ""

            # Creamos input único para el chain
            chain_input = {
                "question": f"{custom_prompt}\n\nTranscripción completa de la entrevista:\n{interview_text}",
                "chat_history": []
            }

            try:
                analysis = chain.invoke(chain_input)
                st.markdown("### 📊 Resultado del análisis (JSON)")
                st.json(analysis.get("answer", str(analysis)))
            except Exception as e:
                st.error("Error al ejecutar el análisis con el chain: " + repr(e))
