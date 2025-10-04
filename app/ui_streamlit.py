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

st.set_page_config(page_title="Chatbot GenAI Psy", layout="centered")
st.title("🤖 Asistente de Psicología - Andrés")

# --- Inicializar historial ---
st.session_state.setdefault("chat_history", [])

# --- Paths ---
ROOT = Path(__file__).parents[1]
VECTOR_DIR = ROOT / "vectorstore"
DATA_DIR = ROOT / "data"

# Buscar PDF(s) en data/
if not DATA_DIR.exists():
    st.error("La carpeta 'data/' no existe en el proyecto. Coloca tus PDFs en data/ y reinicia.")
    st.stop()

pdf_files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".pdf"]
if not pdf_files:
    st.error("No se encontró ningún PDF en la carpeta 'data/'. Agrega al menos un PDF y vuelve a intentar.")
    st.stop()

PDF_FILE = pdf_files[0]  # tomamos el primer PDF disponible

# --- Función para crear vectorstore desde PDF ---
def crear_vectorstore_desde_pdf(pdf_path: Path, vector_dir: Path, chunk_size=500, chunk_overlap=50):
    st.info(f"Creando vectorstore desde PDF: {pdf_path.name} (esto puede tardar)...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)

    vector_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(vector_dir))
    st.success("Vectorstore creado y guardado en " + str(vector_dir))
    return vectordb

# --- Cargar o crear vectorstore con check de seguridad para deserialización ---
vectordb = None
index_file = VECTOR_DIR / "index.faiss"
pickle_file = VECTOR_DIR / "index.pkl"  # nombre típico, puede variar

if not VECTOR_DIR.exists() or not index_file.exists() or not pickle_file.exists():
    # No hay vectorstore completo: crear desde PDF
    try:
        vectordb = crear_vectorstore_desde_pdf(PDF_FILE, VECTOR_DIR)
    except Exception as e:
        st.error("Error creando vectorstore desde PDF: " + repr(e))
        st.stop()
else:
    st.warning(
        "Se detectó un vectorstore existente en 'vectorstore/'.\n\n"
        "Cargarlo requiere deserializar un archivo pickle. Esto es seguro solo si confías en "
        "el origen del vectorstore (p. ej. lo generaste tú y nadie lo ha modificado)."
    )

    allow = st.checkbox("Confirmo que confío en este vectorstore y deseo cargarlo (permitir deserialización peligrosa)")
    if allow:
        try:
            # Nota: allow_dangerous_deserialization=True permite cargar .pkl; úsalo solo si confías en los archivos.
            vectordb = FAISS.load_local(str(VECTOR_DIR), OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            st.success("Vectorstore cargado correctamente.")
        except Exception as e:
            st.error("No fue posible cargar el vectorstore existente: " + repr(e))
            st.info("Puedes eliminar la carpeta 'vectorstore/' para forzar la recreación desde los PDFs.")
            st.stop()
    else:
        st.info("Si prefieres no cargar el vectorstore existente, elimina la carpeta 'vectorstore/' y recarga la aplicación para que se regenere a partir del PDF.")
        st.stop()

# --- Construir chain ---
try:
    chain = build_chain(vectordb)
except Exception as e:
    st.error("Error al construir la cadena (chain) con el vectorstore: " + repr(e))
    st.stop()

# --- Modo Chat Normal ---
st.header("💬 Chat en vivo")
question = st.text_input("Escribe tu pregunta:")

if question:
    with st.spinner("Pensando..."):
        try:
            result = chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append((question, result.get("answer", str(result))))
        except Exception as e:
            st.error("Error al invocar el chain: " + repr(e))

if st.session_state.chat_history:
    st.markdown("---")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 Usuario:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")

# --- Análisis de Entrevistas DOCX ---
st.header("📄 Análisis de entrevistas (Word)")

uploaded_file = st.file_uploader("Sube un documento Word con la entrevista", type=["docx"])
prompts_dir = ROOT / "app" / "prompts"
prompt_files = sorted([p.name for p in prompts_dir.iterdir() if p.suffix in {".txt", ".md"}]) if prompts_dir.exists() else []
prompt_choice = st.selectbox("Selecciona un prompt", prompt_files)

if uploaded_file and prompt_choice:
    qa_pairs = load_interview_docx(uploaded_file)
    st.write("Preguntas y respuestas detectadas (primeros 5 fragmentos):")
    st.write(qa_pairs[:5])

    if st.button("🔍 Analizar entrevista"):
        with st.spinner("Analizando con IA..."):
            joined_text = "\n".join(qa_pairs)
            prompt_path = prompts_dir / prompt_choice
            try:
                custom_prompt = prompt_path.read_text(encoding="utf-8")
            except Exception as e:
                st.error("No se pudo leer el prompt seleccionado: " + repr(e))
                custom_prompt = ""

            # Construimos la entrada combinada
            full_input = f"{custom_prompt}\n\nContenido entrevista:\n{joined_text}"

            try:
                analysis = chain.invoke({
                    "question": full_input,
                    "chat_history": []
                })
                st.markdown("### 📊 Resultado del análisis")
                st.write(analysis.get("answer", str(analysis)))
            except Exception as e:
                st.error("Error al ejecutar el análisis con el chain: " + repr(e))
