# app/ui_streamlit.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.rag_pipeline import load_vectorstore_from_disk, build_chain
from app.docx_analysis import load_interview_docx  # <- módulo que debes crear

st.set_page_config(page_title="Chatbot GenAI Psy", layout="centered")

st.title("🤖 Asistente de Psicología - Andrés")

# Inicializar historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cargar vectorstore y chain
vectordb = load_vectorstore_from_disk()
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

# --- Nuevo: Análisis de Entrevistas DOCX ---
st.header("📄 Análisis de entrevistas (Word)")

uploaded_file = st.file_uploader("Sube un documento Word con la entrevista", type=["docx"])
prompt_choice = st.selectbox("Selecciona un prompt", os.listdir(os.path.join("app", "prompts")))

if uploaded_file and prompt_choice:
    qa_pairs = load_interview_docx(uploaded_file)
    st.write("Preguntas y respuestas detectadas:")
    st.write(qa_pairs[:5])  # muestra primeras 5

    if st.button("🔍 Analizar entrevista"):
        with st.spinner("Analizando con IA..."):
            # Usamos el contenido del Word como input al chain
            joined_text = "\n".join(qa_pairs)
            with open(os.path.join("app", "prompts", prompt_choice), "r", encoding="utf-8") as f:
                custom_prompt = f.read()

            analysis = chain.invoke({
                "question": f"{custom_prompt}\n\nContenido entrevista:\n{joined_text}",
                "chat_history": []
            })

        st.markdown("### 📊 Resultado del análisis")
        st.write(analysis["answer"])
