"""
DOCX Export Service - Generates academic thesis-formatted Word documents.
Matches the format observed in real MEDICINE thesis Results chapters.
"""
import os
from typing import List, Dict, Optional
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)

EXPORT_DIR = "generated_reports"


def _ensure_export_dir():
    os.makedirs(EXPORT_DIR, exist_ok=True)


def _setup_styles(doc: Document):
    """Configure document styles to match academic thesis format."""
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(12)

    pf = style.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)

    # Heading 1 - Chapter heading
    h1 = doc.styles["Heading 1"]
    h1.font.name = "Times New Roman"
    h1.font.size = Pt(16)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0, 0, 0)
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1.paragraph_format.space_before = Pt(24)
    h1.paragraph_format.space_after = Pt(12)

    # Heading 2 - Table/section heading
    h2 = doc.styles["Heading 2"]
    h2.font.name = "Times New Roman"
    h2.font.size = Pt(13)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0, 0, 0)
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(6)


def _add_table(doc: Document, headers: List[str], rows: List[List[str]], title: str = ""):
    """Add a formatted table to the document."""
    if title:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(11)

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    header_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        header_cells[i].text = header
        for paragraph in header_cells[i].paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(11)
                run.font.name = "Times New Roman"

    # Data rows
    for row_data in rows:
        row_cells = table.add_row().cells
        for i, cell_text in enumerate(row_data):
            row_cells[i].text = str(cell_text)
            for paragraph in row_cells[i].paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(11)
                    run.font.name = "Times New Roman"

    # Set column widths proportionally
    total_width = Cm(15)
    col_width = total_width / len(headers)
    for row in table.rows:
        for cell in row.cells:
            cell.width = col_width

    doc.add_paragraph()  # spacing after table


def _add_chart_image(doc: Document, chart_path: str, caption: str = ""):
    """Add a chart image with caption."""
    if not os.path.exists(chart_path):
        logger.warning("Chart image not found", path=chart_path)
        return

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(chart_path, width=Inches(4.5))

    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        run.italic = True
        run.font.size = Pt(10)


def export_report_to_docx(
    report_sections: List[Dict],
    charts: List[Dict],
    title: str,
    methodology_note: str = "",
    summary: str = "",
    submission_id: str = "",
) -> str:
    """
    Export the generated report to a DOCX file.

    Args:
        report_sections: List of section dicts from ReportGeneratorAgent
        charts: List of chart dicts with paths
        title: Thesis title
        methodology_note: Statistical methodology paragraph
        summary: Executive summary
        submission_id: For filename uniqueness

    Returns:
        Path to the generated DOCX file
    """
    _ensure_export_dir()

    doc = Document()
    _setup_styles(doc)

    # Title page content
    doc.add_heading("OBSERVATIONS AND RESULTS", level=1)

    chart_map = {}
    for c in charts:
        var = c.get("variable", "")
        chart_map[var] = c.get("path", "")

    table_counter = 0

    for section in report_sections:
        section_type = section.get("type", "")

        if section_type == "heading":
            doc.add_heading(section["content"], level=1)

        elif section_type == "table_heading":
            table_counter += 1
            doc.add_heading(section["content"], level=2)

        elif section_type == "paragraph":
            p = doc.add_paragraph(section["content"])
            p.paragraph_format.first_line_indent = Cm(1.27)

        elif section_type == "descriptive_note":
            p = doc.add_paragraph()
            run = p.add_run(section["content"])
            run.italic = True
            run.font.size = Pt(11)

        elif section_type == "test_result":
            p = doc.add_paragraph()
            run = p.add_run(section["content"])
            run.bold = True
            run.font.size = Pt(11)

        elif section_type == "table":
            headers = section.get("headers", [])
            rows = section.get("rows", [])
            _add_table(doc, headers, rows)

        elif section_type == "chart":
            var_name = section.get("variable_name", "")
            chart_path = section.get("chart_path", "")
            if chart_path:
                caption = f"Figure: Distribution of {var_name}"
                _add_chart_image(doc, chart_path, caption)

    # Add methodology section if not already in sections
    if methodology_note and not any(
        s.get("content", "").startswith("Statistical analysis") for s in report_sections
    ):
        doc.add_heading("Statistical Analysis", level=1)
        doc.add_paragraph(methodology_note)

    # Summary at end
    if summary:
        doc.add_heading("Summary", level=1)
        doc.add_paragraph(summary)

    # Save
    safe_title = "".join(c if c.isalnum() or c in " _-" else "" for c in title)[:50].strip()
    filename = f"{safe_title}_{submission_id[:8] if submission_id else 'report'}.docx"
    filepath = os.path.join(EXPORT_DIR, filename)
    doc.save(filepath)

    logger.info("DOCX report exported", path=filepath, sections=len(report_sections), charts=len(charts))
    return filepath


def export_report_to_pdf(
    report_sections: List[Dict],
    charts: List[Dict],
    title: str,
    methodology_note: str = "",
    summary: str = "",
    submission_id: str = "",
) -> str:
    """
    Export the generated report to a PDF file.
    Uses reportlab for PDF generation.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, inch
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak
    )
    from reportlab.lib import colors

    _ensure_export_dir()

    safe_title = "".join(c if c.isalnum() or c in " _-" else "" for c in title)[:50].strip()
    filename = f"{safe_title}_{submission_id[:8] if submission_id else 'report'}.pdf"
    filepath = os.path.join(EXPORT_DIR, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=2.5 * cm,
        leftMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ThesisBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=12,
        leading=18,
        alignment=TA_JUSTIFY,
        firstLineIndent=36,
    ))
    styles.add(ParagraphStyle(
        name="ThesisHeading",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="ThesisSubheading",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=13,
        spaceBefore=18,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="TestResult",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=11,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=12,
    ))

    elements = []

    # Title
    elements.append(Paragraph("OBSERVATIONS AND RESULTS", styles["ThesisHeading"]))
    elements.append(Spacer(1, 12))

    for section in report_sections:
        section_type = section.get("type", "")

        if section_type == "heading":
            elements.append(Spacer(1, 18))
            elements.append(Paragraph(section["content"].upper(), styles["ThesisHeading"]))

        elif section_type == "table_heading":
            elements.append(Paragraph(section["content"], styles["ThesisSubheading"]))

        elif section_type == "paragraph":
            elements.append(Paragraph(section["content"], styles["ThesisBody"]))

        elif section_type == "descriptive_note":
            elements.append(Paragraph(f"<i>{section['content']}</i>", styles["ThesisBody"]))

        elif section_type == "test_result":
            elements.append(Paragraph(section["content"], styles["TestResult"]))

        elif section_type == "table":
            headers = section.get("headers", [])
            rows = section.get("rows", [])
            table_data = [headers] + rows

            t = Table(table_data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            elements.append(Spacer(1, 6))
            elements.append(t)
            elements.append(Spacer(1, 12))

        elif section_type == "chart":
            chart_path = section.get("chart_path", "")
            var_name = section.get("variable_name", "")
            if chart_path and os.path.exists(chart_path):
                img = Image(chart_path, width=4.5 * inch, height=3.0 * inch)
                elements.append(Spacer(1, 6))
                elements.append(img)
                elements.append(Paragraph(
                    f"Figure: Distribution of {var_name}", styles["Caption"]
                ))

    # Methodology
    if methodology_note:
        elements.append(Spacer(1, 18))
        elements.append(Paragraph("STATISTICAL ANALYSIS", styles["ThesisHeading"]))
        elements.append(Paragraph(methodology_note, styles["ThesisBody"]))

    # Summary
    if summary:
        elements.append(Spacer(1, 18))
        elements.append(Paragraph("SUMMARY", styles["ThesisHeading"]))
        elements.append(Paragraph(summary, styles["ThesisBody"]))

    doc.build(elements)

    logger.info("PDF report exported", path=filepath, sections=len(report_sections))
    return filepath
