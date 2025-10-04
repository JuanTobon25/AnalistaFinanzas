# app/ui_streamlit.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.rag_pipeline import load_vectorstore_from_disk, build_chain, crear_vectorstore_desde_pdf
from app.docx_analysis import load_interview_docx

st.set_page_config(page_title="Chatbot GenAI Psy", layout="centered")
st.title("🤖 Asistente de Psicología - Andrés")

# --- Inicializar historial ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Cargar vectorstore ---
VECTOR_DIR = "app/vectorstore"
PDF_FILE = os.path.join("data", "entrevista.pdf")  # PDF por defecto

pdf_files = [PDF_FILE] if os.path.exists(PDF_FILE) else []

if pdf_files:
    st.info(f"Cargando vectorstore desde PDF: {pdf_files[0]} ...")
    vectordb = crear_vectorstore_desde_pdf(PDF_FILE, VECTOR_DIR)
else:
    st.info("Cargando vectorstore existente...")
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    vectordb = FAISS.load_local(VECTOR_DIR, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# --- Construir chain ---
chain = build_chain(vectordb)

# --- Modo Chat Normal ---
st.header("💬 Chat en vivo")
question = st.text_input("Escribe tu pregunta:")

if question:
    with st.spinner("Pensando..."):
        try:
            result = chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append((question, result["answer"]))
        except Exception as e:
            st.error(f"Error al generar respuesta: {e}")

if st.session_state.chat_history:
    st.markdown("---")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 Usuario:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")

# --- Análisis de entrevistas DOCX ---
st.header("📄 Análisis de entrevistas (Word)")
uploaded_file = st.file_uploader("Sube un documento Word con la entrevista", type=["docx"])
prompt_choice = st.selectbox("Selecciona un prompt", os.listdir(os.path.join("app", "prompts")))

if uploaded_file and prompt_choice:
    qa_pairs = load_interview_docx(uploaded_file)
    
    # Vista previa solo primeros 5 fragmentos
    st.write("Primeros 5 fragmentos para vista previa:")
    st.write(qa_pairs[:5])

    if st.button("🔍 Analizar entrevista"):
        with st.spinner("Analizando con IA..."):
            joined_text = "\n".join(qa_pairs)
            with open(os.path.join("app", "prompts", prompt_choice), "r", encoding="utf-8") as f:
                custom_prompt = f.read()
            
            # Input único, sin llaves rígidas
            analysis_input = f"{custom_prompt}\n\nContenido entrevista:\n{joined_text}"

            try:
                analysis = chain.invoke({"question": analysis_input, "chat_history": []})
                st.markdown("### 📊 Resultado del análisis")
                st.write(analysis["answer"])
            except Exception as e:
                st.error(f"Error al ejecutar el análisis con el chain: {e}")

