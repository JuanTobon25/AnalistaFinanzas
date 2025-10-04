# app/docx_analysis.py
from docx import Document
import io

def load_interview_docx(file) -> list[str]:
    """
    Carga un archivo DOCX (puede ser un path o un objeto de Streamlit uploader)
    y extrae el contenido como una lista de preguntas/respuestas detectadas.

    Args:
        file (str | UploadedFile): Ruta a un .docx o archivo subido en Streamlit

    Returns:
        list[str]: Lista con cada pregunta/respuesta como string
    """
    # Si el archivo es un UploadedFile de streamlit
    if hasattr(file, "read"):
        file_content = file.read()
        file = io.BytesIO(file_content)

    doc = Document(file)
    qa_pairs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            qa_pairs.append(text)

    return qa_pairs
