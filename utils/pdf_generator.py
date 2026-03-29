from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import uuid
import os


def generate_pdf(paper_data: dict, topic: str, difficulty: str) -> str:
    filename = f"question_paper_{uuid.uuid4().hex}.pdf"
    filepath = os.path.join("/tmp", filename)

    doc = SimpleDocTemplate(filepath)
    styles = getSampleStyleSheet()

    elements = []

    # =========================
    # TITLE
    # =========================
    elements.append(Paragraph("<b>QUESTION PAPER</b>", styles["Title"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Topic:</b> {topic}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Difficulty:</b> {difficulty}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # =========================
    # MCQ SECTION
    # =========================
    if paper_data.get("mcq"):
        elements.append(Paragraph("<b>Section A: MCQs</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))

        for i, q in enumerate(paper_data["mcq"], 1):
            elements.append(Paragraph(f"{i}. {q['question']}", styles["Normal"]))
            elements.append(Spacer(1, 5))

            for opt in q.get("options", []):
                elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;{opt}", styles["Normal"]))

            elements.append(Paragraph(f"<b>Answer:</b> {q.get('answer','')}", styles["Normal"]))
            elements.append(Spacer(1, 10))

    # =========================
    # SHORT ANSWER
    # =========================
    if paper_data.get("short"):
        elements.append(Paragraph("<b>Section B: Short Answer</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))

        for i, q in enumerate(paper_data["short"], 1):
            elements.append(Paragraph(f"{i}. {q['question']}", styles["Normal"]))
            elements.append(Paragraph(f"<b>Answer:</b> {q.get('answer','')}", styles["Normal"]))
            elements.append(Spacer(1, 10))

    # =========================
    # LONG ANSWER
    # =========================
    if paper_data.get("long"):
        elements.append(Paragraph("<b>Section C: Long Answer</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))

        for i, q in enumerate(paper_data["long"], 1):
            elements.append(Paragraph(f"{i}. {q['question']}", styles["Normal"]))
            elements.append(Paragraph(f"<b>Answer:</b> {q.get('answer','')}", styles["Normal"]))
            elements.append(Spacer(1, 10))

    # =========================
    # BUILD PDF
    # =========================
    doc.build(elements)

    return filepath