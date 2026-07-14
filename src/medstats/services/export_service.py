"""
DOCX Export Service - Generates academic thesis-formatted Word documents.
Matches the format observed in real MEDICINE thesis Results chapters.
"""
import base64
import json
import os
import re
from typing import List, Dict, Optional
from html import escape
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from src.medstats.services.export_formatters import (
    clean_section_heading,
    should_page_break_before,
)
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)

EXPORT_DIR = "generated_reports"
CLIENT_CHARTS_DIR = os.path.join(EXPORT_DIR, "_client_charts")


def _normalize_var_key(name: str) -> str:
    return re.sub(r"[^\w]+", "_", (name or "").lower()).strip("_")


def materialize_client_charts(chart_images: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Save base64 chart images from the frontend to temp files."""
    if not chart_images:
        return {}

    os.makedirs(CLIENT_CHARTS_DIR, exist_ok=True)
    paths: Dict[str, str] = {}

    for key, data_uri in chart_images.items():
        if not data_uri:
            continue
        payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
        try:
            raw = base64.b64decode(payload)
        except (ValueError, TypeError):
            logger.warning("Invalid chart image payload", key=key)
            continue

        norm_key = _normalize_var_key(key)
        filepath = os.path.join(CLIENT_CHARTS_DIR, f"{norm_key}.png")
        with open(filepath, "wb") as f:
            f.write(raw)
        paths[norm_key] = filepath

    return paths


def resolve_chart_path(
    section: Dict,
    chart_overrides: Optional[Dict[str, str]] = None,
) -> str:
    """Prefer client-rendered chart image, then fall back to stored chart_path."""
    chart_overrides = chart_overrides or {}
    var_name = section.get("variable_name", "")
    norm_key = _normalize_var_key(var_name)

    if norm_key in chart_overrides and os.path.exists(chart_overrides[norm_key]):
        return chart_overrides[norm_key]

    for key, path in chart_overrides.items():
        if norm_key and (norm_key in key or key in norm_key) and os.path.exists(path):
            return path

    chart_path = section.get("chart_path", "")
    if chart_path and os.path.exists(chart_path):
        return chart_path

    return ""


def _ensure_export_dir():
    os.makedirs(EXPORT_DIR, exist_ok=True)


def build_export_document_title(title: str) -> str:
    """Document and file title: `<submission title> - Statistical analysis`."""
    base = (title or "Report").strip()
    suffix = " - Statistical analysis"
    if base.lower().endswith(suffix.lower()):
        return base
    return f"{base}{suffix}"


def build_export_filename(title: str, ext: str) -> str:
    doc_title = build_export_document_title(title)
    safe = "".join(c if c.isalnum() or c in " _-" else "" for c in doc_title)[:120].strip()
    return f"{safe or 'Statistical analysis'}.{ext}"


def _observations_sections(report_sections: List[Dict]) -> List[Dict]:
    """Sections for Observations and Results — excludes embedded methodology block."""
    out: List[Dict] = []
    skip = False
    for section in report_sections:
        if section.get("type") == "heading":
            content = (section.get("content") or "").strip().upper()
            if content in ("STATISTICAL ANALYSIS", "STATISTICAL METHODOLOGY"):
                skip = True
                continue
        if skip:
            continue
        out.append(section)
    return out


def _literature_sections(report_sections: List[Dict]) -> List[Dict]:
    """Extract the Prior Literature Context heading + content from report_sections."""
    out: List[Dict] = []
    in_lit = False
    for section in report_sections:
        if section.get("type") == "heading":
            content = (section.get("content") or "").strip().upper()
            if content == "PRIOR LITERATURE CONTEXT":
                in_lit = True
                out.append(section)
                continue
            elif in_lit:
                break
        if in_lit:
            out.append(section)
    return out


def _resolve_methodology_text(methodology_note: str, report_sections: List[Dict]) -> str:
    if methodology_note and methodology_note.strip():
        return methodology_note.strip()
    for i, section in enumerate(report_sections):
        if section.get("type") != "heading":
            continue
        content = (section.get("content") or "").strip().upper()
        if content not in ("STATISTICAL ANALYSIS", "STATISTICAL METHODOLOGY"):
            continue
        for follow in report_sections[i + 1:]:
            if follow.get("type") == "heading":
                break
            if follow.get("type") == "paragraph" and follow.get("content"):
                return follow["content"].strip()
    return ""


def _format_summary_for_export(summary: str) -> str:
    """Convert structured JSON summary to readable plain text (legacy fallback)."""
    if not summary:
        return ""
    data = _parse_executive_summary(summary)
    if not data:
        return summary
    parts = []
    if data.get("study_description"):
        parts.append(data["study_description"])
    if data.get("key_findings"):
        parts.append("\nKey Findings:")
        for f in data["key_findings"]:
            var = f.get("variable", "")
            result = f.get("result", "")
            parts.append(f"  • {var}: {result}")
    return "\n".join(parts)


def _parse_executive_summary(summary: str) -> Optional[Dict]:
    if not summary:
        return None
    try:
        data = json.loads(summary)
        if isinstance(data, dict) and ("study_description" in data or "key_findings" in data):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _count_variables(report_sections: List[Dict]) -> int:
    return sum(1 for s in report_sections if s.get("type") == "table_heading")


def _append_executive_summary_html(
    body_parts: List[str],
    summary: str,
    report_sections: List[Dict],
) -> None:
    data = _parse_executive_summary(summary)
    if not data:
        formatted = _format_summary_for_export(summary)
        if not formatted:
            return
        body_parts.append('<h1>Executive Summary</h1>')
        for para in formatted.split("\n"):
            if para.strip():
                if para.startswith("Key Findings:"):
                    body_parts.append(f'<h3>{escape(para)}</h3>')
                elif para.strip().startswith("•"):
                    body_parts.append(f'<p class="finding">{escape(para.strip())}</p>')
                else:
                    body_parts.append(f'<p class="body-text">{escape(para)}</p>')
        return

    stats = data.get("stats") or {}
    var_count = _count_variables(report_sections) or stats.get("total_variables", 0)
    sig_count = stats.get("significant", 0)

    body_parts.append('<h1>Executive Summary</h1>')
    body_parts.append(
        f'<p class="summary-meta">{var_count} variables analyzed · '
        f'{sig_count} statistically significant</p>'
    )

    if data.get("study_description"):
        body_parts.append('<h3 class="summary-section">Study Description</h3>')
        body_parts.append(
            f'<p class="summary-body">{escape(data["study_description"])}</p>'
        )

    findings = data.get("key_findings") or []
    if findings:
        body_parts.append('<h3 class="summary-section">Key Findings</h3>')
        body_parts.append('<ul class="findings-list">')
        for f in findings:
            sig_class = " sig" if f.get("significant") else ""
            var = escape(f.get("variable", ""))
            result = escape(f.get("result", ""))
            body_parts.append(
                f'<li class="finding-item{sig_class}">'
                f'<span class="var">{var}</span> — {result}</li>'
            )
        body_parts.append('</ul>')


def _chart_to_base64(chart_path: str) -> str:
    """Embed chart PNG as base64 data URI for HTML preview."""
    if not chart_path or not os.path.exists(chart_path):
        return ""
    with open(chart_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def generate_preview_html(
    report_sections: List[Dict],
    charts: List[Dict],
    title: str,
    methodology_note: str = "",
    summary: str = "",
    chart_overrides: Optional[Dict[str, str]] = None,
) -> str:
    """Generate a self-contained HTML document for export preview."""
    doc_title = build_export_document_title(title)
    body_parts: List[str] = []
    observations = _observations_sections(report_sections)
    methodology_text = _resolve_methodology_text(methodology_note, report_sections)

    body_parts.append(
        '<div class="title-page">'
        f'<h1>{escape(doc_title)}</h1>'
        '</div>'
    )

    formatted_summary = _format_summary_for_export(summary)
    if formatted_summary or _parse_executive_summary(summary):
        _append_executive_summary_html(body_parts, summary, report_sections)

    if observations:
        body_parts.append('<div class="page-break"></div>')
        first_heading = True
        for section in observations:
            section_type = section.get("type", "")
            content = section.get("content") or ""

            if section_type == "heading":
                if should_page_break_before(section_type, content, is_first_heading=first_heading):
                    body_parts.append('<div class="page-break"></div>')
                first_heading = False
                body_parts.append(f'<h1>{escape(content)}</h1>')
            elif section_type == "table_heading":
                body_parts.append('<div class="page-break"></div>')
                heading = escape(clean_section_heading(section))
                body_parts.append(f'<h2>{heading}</h2>')
            elif section_type in ("paragraph", "summary_text"):
                body_parts.append(f'<p class="body-text">{escape(content)}</p>')
            elif section_type == "descriptive_note":
                body_parts.append(f'<p class="note"><em>{escape(content)}</em></p>')
            elif section_type == "test_result":
                sig = "significant" if content and "not significant" not in content.lower() else "not-significant"
                body_parts.append(f'<p class="test-result {sig}">{escape(content)}</p>')
            elif section_type == "table":
                headers = section.get("headers", [])
                rows = section.get("rows", [])
                if headers and rows:
                    body_parts.append('<table><thead><tr>')
                    for h in headers:
                        body_parts.append(f'<th>{escape(str(h))}</th>')
                    body_parts.append('</tr></thead><tbody>')
                    for row in rows:
                        body_parts.append('<tr>')
                        for cell in row:
                            is_total = str(row[0]).upper() == "TOTAL" if row else False
                            cls = ' class="total-row"' if is_total else ""
                            body_parts.append(f'<td{cls}>{escape(str(cell))}</td>')
                        body_parts.append('</tr>')
                    body_parts.append('</tbody></table>')
            elif section_type == "chart":
                var_name = section.get("variable_name", "")
                chart_path = resolve_chart_path(section, chart_overrides)
                if chart_path:
                    data_uri = _chart_to_base64(chart_path)
                    body_parts.append(
                        f'<figure class="chart">'
                        f'<img src="{data_uri}" alt="Chart: {escape(var_name)}" />'
                        f'<figcaption>Figure: Distribution of {escape(var_name)}</figcaption>'
                        f'</figure>'
                    )

    if methodology_text:
        body_parts.append('<div class="page-break"></div>')
        body_parts.append('<h1>Statistical Methodology</h1>')
        body_parts.append(f'<p class="body-text">{escape(methodology_text)}</p>')

    lit_sections = _literature_sections(report_sections)
    if lit_sections:
        body_parts.append('<div class="page-break"></div>')
        for sec in lit_sections:
            stype = sec.get("type", "")
            content = sec.get("content") or ""
            if stype == "heading":
                body_parts.append(f'<h1>{escape(content)}</h1>')
                body_parts.append(
                    '<p class="note"><em>Automated literature matching via PubMed index. '
                    'Verify relevance before citing in your thesis.</em></p>'
                )
            elif stype == "literature_context":
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(tuple("123456789")):
                        body_parts.append(f'<p class="lit-entry">{escape(line)}</p>')
                    else:
                        body_parts.append(f'<p class="body-text">{escape(line)}</p>')

    html_body = "\n".join(body_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape(doc_title)}</title>
<style>
  @page {{ margin: 2.5cm; }}
  body {{
    font-family: "Times New Roman", Times, serif;
    font-size: 12pt;
    line-height: 1.6;
    color: #1a1a1a;
    max-width: 800px;
    margin: 0 auto;
    padding: 40px 48px;
    background: #fff;
  }}
  .title-page {{ text-align: center; margin-bottom: 48px; padding-bottom: 24px; border-bottom: 2px solid #333; }}
  .title-page h1 {{ font-size: 18pt; font-weight: bold; margin-bottom: 8px; }}
  .subtitle {{ font-size: 13pt; color: #555; }}
  h1 {{ font-size: 16pt; font-weight: bold; text-align: center; margin: 32px 0 16px; }}
  h2 {{ font-size: 13pt; font-weight: bold; margin: 24px 0 8px; }}
  h3 {{ font-size: 12pt; font-weight: bold; margin: 16px 0 8px; }}
  .body-text {{ text-align: justify; text-indent: 1.27cm; margin: 8px 0; }}
  .note {{ font-style: italic; font-size: 11pt; margin: 8px 0; color: #444; }}
  .test-result {{ font-weight: bold; font-size: 11pt; margin: 8px 0; padding: 8px 12px; border-left: 3px solid #ccc; }}
  .test-result.significant {{ border-color: #2e8b57; background: #f0faf4; }}
  .finding {{ margin: 4px 0 4px 16px; font-size: 11pt; }}
  .summary-meta {{ text-align: center; font-size: 11pt; color: #555; margin: -8px 0 24px; }}
  .summary-section {{ font-size: 13pt; font-weight: bold; margin: 24px 0 10px; padding-left: 12px; border-left: 3px solid #2c5f8a; text-align: left; }}
  .summary-body {{ text-align: justify; line-height: 1.75; margin: 0 0 8px; text-indent: 0; font-size: 12pt; }}
  ul.findings-list {{ list-style: none; padding: 0; margin: 0 0 16px; }}
  .finding-item {{ margin: 10px 0; padding-left: 22px; position: relative; font-size: 12pt; line-height: 1.6; text-align: left; }}
  .finding-item::before {{ content: "●"; position: absolute; left: 0; color: #9ca3af; font-size: 10pt; top: 2px; }}
  .finding-item.sig::before {{ color: #059669; }}
  .finding-item .var {{ font-weight: bold; color: #111; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 11pt; }}
  th {{ background: #2c5f8a; color: #fff; padding: 8px 12px; text-align: center; font-weight: bold; }}
  td {{ padding: 6px 12px; text-align: center; border: 1px solid #ddd; }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  .total-row {{ font-weight: bold; background: #eee !important; }}
  figure.chart {{ text-align: center; margin: 24px 0; page-break-inside: avoid; }}
  figure.chart img {{ width: 100%; height: auto; max-height: none; }}
  figcaption {{ font-style: italic; font-size: 10pt; color: #555; margin-top: 8px; }}
  .page-break {{ page-break-before: always; break-before: page; margin-top: 0; }}
  .variable-section {{ page-break-before: always; break-before: page; }}
  .variable-section:first-of-type {{ page-break-before: auto; break-before: auto; }}
  .lit-entry {{ margin: 8px 0 8px 16px; font-size: 11pt; line-height: 1.6; text-indent: 0; }}
  .lit-entry + .lit-entry {{ border-top: 1px solid #eee; padding-top: 8px; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""



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
    """Add a chart image with caption, scaled to full content width."""
    if not os.path.exists(chart_path):
        logger.warning("Chart image not found", path=chart_path)
        return

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(chart_path, width=Inches(6.3))

    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        run.italic = True
        run.font.size = Pt(10)


def _pdf_chart_image(chart_path: str, max_width: float, max_height: float):
    """Scale chart to page width while preserving aspect ratio."""
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image

    iw, ih = ImageReader(chart_path).getSize()
    if not iw or not ih:
        return Image(chart_path, width=max_width, height=max_width * 0.6)

    width = max_width
    height = width * (ih / float(iw))
    if height > max_height:
        height = max_height
        width = height * (float(iw) / ih)
    return Image(chart_path, width=width, height=height)


def _add_executive_summary_docx(
    doc: Document,
    summary: str,
    report_sections: List[Dict],
) -> None:
    data = _parse_executive_summary(summary)
    if not data:
        formatted_summary = _format_summary_for_export(summary)
        if not formatted_summary:
            return
        doc.add_heading("Executive Summary", level=1)
        for para in formatted_summary.split("\n"):
            if para.strip():
                if para.startswith("Key Findings:"):
                    doc.add_heading(para.strip(), level=2)
                elif para.strip().startswith("•"):
                    p = doc.add_paragraph(para.strip(), style="List Bullet")
                else:
                    p = doc.add_paragraph(para.strip())
                    p.paragraph_format.first_line_indent = Cm(1.27)
        return

    stats = data.get("stats") or {}
    var_count = _count_variables(report_sections) or stats.get("total_variables", 0)
    sig_count = stats.get("significant", 0)

    doc.add_heading("Executive Summary", level=1)
    meta = doc.add_paragraph(
        f"{var_count} variables analyzed · {sig_count} statistically significant"
    )
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in meta.runs:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    if data.get("study_description"):
        doc.add_heading("Study Description", level=2)
        p = doc.add_paragraph(data["study_description"])
        p.paragraph_format.first_line_indent = Cm(0)
        for run in p.runs:
            run.font.size = Pt(12)

    findings = data.get("key_findings") or []
    if findings:
        doc.add_heading("Key Findings", level=2)
        for f in findings:
            p = doc.add_paragraph(style="List Bullet")
            var_run = p.add_run(f.get("variable", ""))
            var_run.bold = True
            p.add_run(f" — {f.get('result', '')}")
            for run in p.runs:
                run.font.size = Pt(12)


def _add_observations_sections_docx(
    doc: Document,
    report_sections: List[Dict],
    chart_overrides: Optional[Dict[str, str]] = None,
) -> None:
    observations = _observations_sections(report_sections)
    first_heading = True
    for section in observations:
        section_type = section.get("type", "")
        content = section.get("content") or ""

        if should_page_break_before(section_type, content, is_first_heading=first_heading):
            doc.add_page_break()
        if section_type == "heading":
            first_heading = False

        if section_type == "heading":
            doc.add_heading(content, level=1)
        elif section_type == "table_heading":
            doc.add_heading(clean_section_heading(section), level=2)
        elif section_type == "paragraph":
            p = doc.add_paragraph(section["content"])
            p.paragraph_format.first_line_indent = Cm(1.27)
        elif section_type == "summary_text":
            p = doc.add_paragraph(section["content"])
            p.paragraph_format.first_line_indent = Cm(1.27)
            for run in p.runs:
                run.font.size = Pt(12)
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
            chart_path = resolve_chart_path(section, chart_overrides)
            if chart_path:
                caption = f"Figure: Distribution of {var_name}"
                _add_chart_image(doc, chart_path, caption)


def _add_methodology_docx(doc: Document, methodology_text: str) -> None:
    if not methodology_text:
        return
    doc.add_page_break()
    doc.add_heading("Statistical Methodology", level=1)
    p = doc.add_paragraph(methodology_text)
    p.paragraph_format.first_line_indent = Cm(1.27)


def _add_literature_docx(doc: Document, report_sections: List[Dict]) -> None:
    lit_sections = _literature_sections(report_sections)
    if not lit_sections:
        return
    doc.add_page_break()
    for sec in lit_sections:
        stype = sec.get("type", "")
        content = sec.get("content") or ""
        if stype == "heading":
            doc.add_heading("Prior Literature Context", level=1)
            disclaimer = doc.add_paragraph(
                "Automated literature matching via PubMed index. "
                "Verify relevance before citing in your thesis."
            )
            disclaimer.paragraph_format.first_line_indent = Cm(0)
            for run in disclaimer.runs:
                run.italic = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        elif stype == "literature_context":
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                p = doc.add_paragraph()
                if line.startswith(tuple("123456789")):
                    p.paragraph_format.left_indent = Cm(0.63)
                    p.paragraph_format.first_line_indent = Cm(-0.63)
                    for run in p.runs:
                        run.font.size = Pt(11)
                p.add_run(line).font.size = Pt(11)


def export_report_to_docx(
    report_sections: List[Dict],
    charts: List[Dict],
    title: str,
    methodology_note: str = "",
    summary: str = "",
    submission_id: str = "",
    output_dir: Optional[str] = None,
    chart_overrides: Optional[Dict[str, str]] = None,
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

    doc_title = build_export_document_title(title)
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(doc_title)
    title_run.bold = True
    title_run.font.size = Pt(18)
    title_run.font.name = "Times New Roman"
    doc.add_paragraph()

    _add_executive_summary_docx(doc, summary, report_sections)

    if _observations_sections(report_sections):
        doc.add_page_break()
    _add_observations_sections_docx(doc, report_sections, chart_overrides)

    methodology_text = _resolve_methodology_text(methodology_note, report_sections)
    _add_methodology_docx(doc, methodology_text)
    _add_literature_docx(doc, report_sections)

    filename = build_export_filename(title, "docx")
    export_dir = output_dir or EXPORT_DIR
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, filename)
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
    output_dir: Optional[str] = None,
    chart_overrides: Optional[Dict[str, str]] = None,
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

    filename = build_export_filename(title, "pdf")
    export_dir = output_dir or EXPORT_DIR
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, filename)

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
        name="ThesisTitle",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=24,
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
    styles.add(ParagraphStyle(
        name="SummaryMeta",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        alignment=TA_CENTER,
        textColor=colors.Color(0.35, 0.35, 0.35),
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="SummarySection",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=13,
        alignment=TA_JUSTIFY,
        leftIndent=0,
        spaceBefore=16,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SummaryBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=12,
        leading=18,
        alignment=TA_JUSTIFY,
        firstLineIndent=0,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SummaryFinding",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=12,
        leading=17,
        alignment=TA_JUSTIFY,
        leftIndent=18,
        bulletIndent=0,
        spaceAfter=6,
    ))

    content_width = A4[0] - 5 * cm

    def _append_executive_summary_pdf(summary_text: str) -> None:
        data = _parse_executive_summary(summary_text)
        if not data:
            formatted = _format_summary_for_export(summary_text)
            if not formatted:
                return
            elements.append(Paragraph("EXECUTIVE SUMMARY", styles["ThesisHeading"]))
            for para in formatted.split("\n"):
                if para.strip():
                    if para.startswith("Key Findings:"):
                        elements.append(Paragraph(para.strip(), styles["SummarySection"]))
                    elif para.strip().startswith("•"):
                        elements.append(Paragraph(para.strip(), styles["SummaryFinding"]))
                    else:
                        elements.append(Paragraph(para.strip(), styles["SummaryBody"]))
            return

        stats = data.get("stats") or {}
        var_count = _count_variables(report_sections) or stats.get("total_variables", 0)
        sig_count = stats.get("significant", 0)

        elements.append(Paragraph("EXECUTIVE SUMMARY", styles["ThesisHeading"]))
        elements.append(Paragraph(
            f"{var_count} variables analyzed · {sig_count} statistically significant",
            styles["SummaryMeta"],
        ))

        if data.get("study_description"):
            elements.append(Paragraph("Study Description", styles["SummarySection"]))
            elements.append(Paragraph(data["study_description"], styles["SummaryBody"]))

        findings = data.get("key_findings") or []
        if findings:
            elements.append(Paragraph("Key Findings", styles["SummarySection"]))
            for f in findings:
                var = f.get("variable", "").replace("&", "&amp;")
                result = f.get("result", "").replace("&", "&amp;")
                bullet = "●" if f.get("significant") else "○"
                elements.append(Paragraph(
                    f'{bullet} <b>{var}</b> — {result}',
                    styles["SummaryFinding"],
                ))

    elements = []
    doc_title = build_export_document_title(title)
    methodology_text = _resolve_methodology_text(methodology_note, report_sections)
    observations = _observations_sections(report_sections)

    elements.append(Spacer(1, 36))
    elements.append(Paragraph(doc_title, styles["ThesisTitle"]))
    elements.append(Spacer(1, 24))

    _append_executive_summary_pdf(summary)

    if observations:
        elements.append(PageBreak())
        first_heading = True
        for section in observations:
            section_type = section.get("type", "")
            content = section.get("content") or ""

            if should_page_break_before(section_type, content, is_first_heading=first_heading):
                elements.append(PageBreak())
            if section_type == "heading":
                first_heading = False

            if section_type == "heading":
                elements.append(Spacer(1, 18))
                elements.append(Paragraph(content.upper(), styles["ThesisHeading"]))
            elif section_type == "table_heading":
                elements.append(Paragraph(clean_section_heading(section), styles["ThesisSubheading"]))
            elif section_type == "paragraph":
                elements.append(Paragraph(section["content"], styles["ThesisBody"]))
            elif section_type == "summary_text":
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
                var_name = section.get("variable_name", "")
                chart_path = resolve_chart_path(section, chart_overrides)
                if chart_path:
                    img = _pdf_chart_image(chart_path, content_width, 7.5 * inch)
                    elements.append(Spacer(1, 6))
                    elements.append(img)
                    elements.append(Paragraph(
                        f"Figure: Distribution of {var_name}", styles["Caption"]
                    ))

    if methodology_text:
        elements.append(PageBreak())
        elements.append(Paragraph("STATISTICAL METHODOLOGY", styles["ThesisHeading"]))
        elements.append(Paragraph(methodology_text, styles["ThesisBody"]))

    styles.add(ParagraphStyle(
        name="LitEntry",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=16,
        leftIndent=18,
        firstLineIndent=-18,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="LitDisclaimer",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        textColor=colors.Color(0.35, 0.35, 0.35),
        spaceAfter=12,
    ))

    lit_sections = _literature_sections(report_sections)
    if lit_sections:
        elements.append(PageBreak())
        for sec in lit_sections:
            stype = sec.get("type", "")
            content = sec.get("content") or ""
            if stype == "heading":
                elements.append(Paragraph("PRIOR LITERATURE CONTEXT", styles["ThesisHeading"]))
                elements.append(Paragraph(
                    "Automated literature matching via PubMed index. "
                    "Verify relevance before citing in your thesis.",
                    styles["LitDisclaimer"],
                ))
            elif stype == "literature_context":
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        elements.append(Paragraph(line.replace("&", "&amp;"), styles["LitEntry"]))
                        elements.append(Spacer(1, 2))

    doc.build(elements)

    logger.info("PDF report exported", path=filepath, sections=len(report_sections))
    return filepath


def persist_export_to_storage(
    local_path: str,
    submission_id: str,
    fmt: str,
) -> str:
    """Upload exported report to S3 (or return local API path) and return URL."""
    from src.medstats.services.storage_service import get_storage_service

    storage = get_storage_service()
    filename = os.path.basename(local_path)
    key = f"exports/{submission_id}/{filename}"
    content_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if fmt == "docx"
        else "application/pdf"
    )
    return storage.upload_file(local_path, key, content_type=content_type)
