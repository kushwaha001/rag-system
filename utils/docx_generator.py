from docx import Document
import uuid
import os

def generate_docx(paper_data: dict, topic: str, difficulty: str) -> str:
    doc = Document()

    doc.add_heading("QUESTION PAPER", 0)

    doc.add_paragraph(f"Topic: {topic}")
    doc.add_paragraph(f"Difficulty: {difficulty}")
    doc.add_paragraph("\n")

    if "mcq" in paper_data:
        doc.add_heading("Section A: MCQs", level=1)
        for i, q in enumerate(paper_data["mcq"], 1):
            doc.add_paragraph(f"{i}. {q['question']}")
            for opt in q.get("options", []):
                doc.add_paragraph(f"   {opt}")
            doc.add_paragraph(f"Answer: {q.get('answer', '')}\n")

    if "short" in paper_data:
        doc.add_heading("Section B: Short Answer", level=1)
        for i, q in enumerate(paper_data["short"], 1):
            doc.add_paragraph(f"{i}. {q['question']}")
            doc.add_paragraph(f"Answer: {q.get('answer', '')}\n")

    if "long" in paper_data:
        doc.add_heading("Section C: Long Answer", level=1)
        for i, q in enumerate(paper_data["long"], 1):
            doc.add_paragraph(f"{i}. {q['question']}")
            doc.add_paragraph(f"Answer: {q.get('answer', '')}\n")

    filename = f"question_paper_{uuid.uuid4().hex}.docx"
    filepath = os.path.join("/tmp", filename)

    doc.save(filepath)
    return filepath