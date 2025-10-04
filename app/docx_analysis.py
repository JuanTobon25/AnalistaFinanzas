from docx import Document

def load_interview_docx(path):
    """Extrae preguntas y respuestas de un DOCX de entrevista."""
    doc = Document(path)
    qa_pairs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            qa_pairs.append(text)
    return qa_pairs
