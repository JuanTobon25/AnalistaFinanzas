# app/ui_streamlit.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.rag_pipeline import build_chain  # importamos solo build_chain
from app.docx_analysis import load_interview_docx  # módulo que debes crear
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

st.set_page_config(page_title="Chatbot GenAI Psy", layout="centered")

st.title("🤖 Asistente de Psicología - Andrés")

# --- Inicializar historial ---
st.session_state.setdefault("chat_history", [])

# --- Configuración de paths ---
VECTOR_DIR = "vectorstore"
DATA_DIR = "data"

# Buscar automáticamente el primer PDF en la carpeta data
pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
if not pdf_files:
    st.error("No se encontró ningún PDF en la carpeta 'data'.")
    st.stop()

PDF_FILE = os.path.join(DATA_DIR, pdf_files[0])

# --- Función para crear vectorstore desde PDF ---
def crear_vectorstore_desde_pdf(pdf_path, vector_dir):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    
    os.makedirs(vector_dir, exist_ok=True)
    vectordb.save_local(vector_dir)
    return vectordb

# --- Cargar o crear vectorstore ---
if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
    st.info(f"Creando vectorstore desde PDF: {pdf_files[0]} ...")
    vectordb = crear_vectorstore_desde_pdf(PDF_FILE, VECTOR_DIR)
else:
    vectordb = FAISS.load_local(VECTOR_DIR, OpenAIEmbeddings(), allow_dangerous_deprecated_kwargs=True)

# --- Construir chain ---
chain = build_chain(vectordb)

# --- Modo Chat Normal ---
st.header("💬 Chat en vivo")
question = st.text_input("Escribe tu pregunta:")

if question:
    with st.spinner("Pensando..."):
        result = chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((question, result["answer"]))

if st.session_state.chat_history:
    st.markdown("---")    
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 Usuario:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")

# --- Análisis de Entrevistas DOCX ---
st.header("📄 Análisis de entrevistas (Word)")

uploaded_file = st.file_uploader("Sube un documento Word con la entrevista", type=["docx"])
prompt_choice = st.selectbox("Selecciona un prompt", os.listdir(os.path.join("app", "prompts")))

if uploaded_file and prompt_choice:
    qa_pairs = load_interview_docx(uploaded_file)
    st.write("Preguntas y respuestas detectadas:")
    st.write(qa_pairs[:5])  # muestra primeras 5

    if st.button("🔍 Analizar entrevista"):
        with st.spinner("Analizando con IA..."):
            joined_text = "\n".join(qa_pairs)
            with open(os.path.join("app", "prompts", prompt_choice), "r", encoding="utf-8") as f:
                custom_prompt = f.read()

            analysis = chain.invoke({
                "question": f"{custom_prompt}\n\nContenido entrevista:\n{joined_text}",
                "chat_history": []
            })

        st.markdown("### 📊 Resultado del análisis")
        st.write(analysis["answer"])
