from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import inch
import uuid
import os
from datetime import date


def generate_pdf(paper_data: dict, topic: str, difficulty: str) -> str:
    filename = f"question_paper_{uuid.uuid4().hex}.pdf"
    filepath = os.path.join("/tmp", filename)

    doc = SimpleDocTemplate(
        filepath,
        leftMargin=1.2 * inch,
        rightMargin=1.2 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "PaperSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        spaceAfter=4,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
    )
    meta_style = ParagraphStyle(
        "PaperMeta",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
    )
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    )
    question_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=2,
        leading=16,
    )
    option_style = ParagraphStyle(
        "Option",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=24,
        spaceAfter=2,
    )
    answer_style = ParagraphStyle(
        "Answer",
        parent=styles["Normal"],
        fontSize=11,
        spaceBefore=6,
        spaceAfter=4,
        leading=16,
    )

    today = date.today().strftime("%B %d, %Y")
    total_q = (
        len(paper_data.get("mcq", []))
        + len(paper_data.get("short", []))
        + len(paper_data.get("long", []))
    )

    elements = []

    # ── QUESTION PAPER HEADER ──
    elements.append(Paragraph("<b>QUESTION PAPER</b>", title_style))
    elements.append(Paragraph(f"<b>Subject: {topic}</b>", subtitle_style))
    elements.append(Paragraph(
        f"Difficulty: {difficulty}&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Total Questions: {total_q}&nbsp;&nbsp;|&nbsp;&nbsp;Date: {today}",
        meta_style
    ))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    elements.append(Spacer(1, 14))

    # ── QUESTIONS ──
    qno = 1

    if paper_data.get("mcq"):
        elements.append(Paragraph("Section A — Multiple Choice Questions", section_style))
        for q in paper_data["mcq"]:
            elements.append(Paragraph(f"<b>Q{qno}.</b> {q.get('question', '')}", question_style))
            for key, opt in (q.get("options") or {}).items():
                elements.append(Paragraph(f"{key}) {opt}", option_style))
            elements.append(Spacer(1, 6))
            qno += 1

    if paper_data.get("short"):
        elements.append(Paragraph("Section B — Short Answer Questions", section_style))
        for q in paper_data["short"]:
            elements.append(Paragraph(f"<b>Q{qno}.</b> {q.get('question', '')}", question_style))
            elements.append(Spacer(1, 8))
            qno += 1

    if paper_data.get("long"):
        elements.append(Paragraph("Section C — Long Answer Questions", section_style))
        for q in paper_data["long"]:
            elements.append(Paragraph(f"<b>Q{qno}.</b> {q.get('question', '')}", question_style))
            elements.append(Spacer(1, 10))
            qno += 1

    # ── PAGE BREAK ──
    elements.append(PageBreak())

    # ── ANSWER KEY HEADER ──
    elements.append(Paragraph("<b>ANSWER KEY</b>", title_style))
    elements.append(Paragraph(f"<b>Subject: {topic}</b>", subtitle_style))
    elements.append(Paragraph(f"Difficulty: {difficulty}", meta_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    elements.append(Spacer(1, 14))

    qno = 1

    if paper_data.get("mcq"):
        elements.append(Paragraph("Section A — MCQ Answers", section_style))
        for q in paper_data["mcq"]:
            ans_letter = str(q.get("answer", "—"))
            ans_text = (q.get("options") or {}).get(ans_letter, "")
            ans_display = f"{ans_letter}) {ans_text}" if ans_text else ans_letter
            elements.append(Paragraph(
                f"<b>Q{qno}.</b> {ans_display}",
                answer_style
            ))
            qno += 1
        elements.append(Spacer(1, 10))

    if paper_data.get("short"):
        elements.append(Paragraph("Section B — Short Answer Key", section_style))
        for q in paper_data["short"]:
            elements.append(Paragraph(
                f"<b>Q{qno}.</b> {q.get('answer', '—')}",
                answer_style
            ))
            qno += 1
        elements.append(Spacer(1, 10))

    if paper_data.get("long"):
        elements.append(Paragraph("Section C — Long Answer Key", section_style))
        for q in paper_data["long"]:
            elements.append(Paragraph(
                f"<b>Q{qno}.</b> {q.get('answer', '—')}",
                answer_style
            ))
            qno += 1

    doc.build(elements)
    return filepath
