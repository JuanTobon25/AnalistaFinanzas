# app/ui_streamlit.py
import sys, os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.rag_pipeline import build_chain
from app.docx_analysis import load_interview_docx

# --- imports recomendados ---
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# --- Configuración ---
st.set_page_config(page_title="Chatbot GenAI Psy", layout="centered")
st.title("🤖 Asistente de Psicología - Andrés")
st.session_state.setdefault("chat_history", [])

ROOT = Path(__file__).parents[1]
VECTOR_DIR = ROOT / "vectorstore"
DATA_DIR = ROOT / "data"

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
    st.success(f"Vectorstore creado y guardado: {vector_dir}")
    return vectordb

# --- Cargar o crear vectorstores para todos los PDFs ---
vectordbs = []
pdf_files = [p for p in DATA_DIR.glob("*.pdf")]

if not pdf_files:
    st.error("No se encontró ningún PDF en 'data/'. Coloca PDFs y recarga la app.")
    st.stop()

for pdf_file in pdf_files:
    pdf_vector_dir = VECTOR_DIR / pdf_file.stem
    index_file = pdf_vector_dir / "index.faiss"
    pickle_file = pdf_vector_dir / "index.pkl"
    
    if not pdf_vector_dir.exists() or not index_file.exists() or not pickle_file.exists():
        try:
            vectordb = crear_vectorstore_desde_pdf(pdf_file, pdf_vector_dir)
            vectordbs.append(vectordb)
        except Exception as e:
            st.error(f"Error creando vectorstore desde {pdf_file.name}: {repr(e)}")
            st.stop()
    else:
        st.warning(f"Vectorstore existente detectado para {pdf_file.name}.")
        allow = st.checkbox(f"Confío en el vectorstore {pdf_file.name} y deseo cargarlo")
        if allow:
            try:
                vectordb = FAISS.load_local(str(pdf_vector_dir), OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                vectordbs.append(vectordb)
                st.success(f"Vectorstore cargado: {pdf_file.name}")
            except Exception as e:
                st.error(f"No se pudo cargar {pdf_file.name}: {repr(e)}")
                st.stop()
        else:
            st.info(f"Eliminar la carpeta {pdf_vector_dir} para regenerar el vectorstore desde PDF.")
            st.stop()

# --- Combinar vectorstores si hay más de uno ---
if len(vectordbs) == 1:
    vectordb = vectordbs[0]
else:
    st.info("Combinando múltiples vectorstores en uno solo...")
    vectordb = FAISS.from_existing_indexes(vectordbs)

# --- Construir chain ---
try:
    chain = build_chain(vectordb)
except Exception as e:
    st.error("Error al construir la cadena (chain): " + repr(e))
    st.stop()

# --- Chat RAG en vivo ---
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

# --- Análisis de entrevistas DOCX múltiples ---
st.header("📄 Análisis de entrevistas (Word)")
uploaded_files = st.file_uploader("Sube uno o más documentos Word con entrevistas", type=["docx"], accept_multiple_files=True)
prompts_dir = ROOT / "app" / "prompts"
prompt_files = sorted([p.name for p in prompts_dir.iterdir() if p.suffix in {".txt", ".md"}]) if prompts_dir.exists() else []
prompt_choice = st.selectbox("Selecciona un prompt", prompt_files)

if uploaded_files and prompt_choice:
    all_qa = []
    for uploaded_file in uploaded_files:
        qa_pairs = load_interview_docx(uploaded_file)
        all_qa.extend(qa_pairs)
    
    st.write("Preguntas y respuestas detectadas (primeros 5 fragmentos):")
    st.write(all_qa[:5])

    if st.button("🔍 Analizar todas las entrevistas"):
        with st.spinner("Analizando con IA..."):
            joined_text = "\n".join(all_qa)
            prompt_path = prompts_dir / prompt_choice
            try:
                custom_prompt = prompt_path.read_text(encoding="utf-8")
            except Exception as e:
                st.error("No se pudo leer el prompt: " + repr(e))
                custom_prompt = ""

            full_input = f"{custom_prompt}\n\nContenido de entrevistas:\n{joined_text}"

            try:
                analysis = chain.invoke({
                    "question": full_input,
                    "chat_history": []
                })
                st.markdown("### 📊 Resultado del análisis")
                st.write(analysis.get("answer", str(analysis)))
            except Exception as e:
                st.error("Error al ejecutar el análisis: " + repr(e))
