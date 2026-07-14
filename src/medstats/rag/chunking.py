"""
AcademicPDFChunker — hierarchical PDF parsing by heading structure.

Uses pymupdf (fitz) for font-size based heading detection; falls back to
paragraph-based chunking when structure is unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from src.medstats.rag.chunking_utils import (
    detect_structural_type,
    extract_tags,
    split_into_child_chunks,
)
from src.medstats.rag.models import AcademicChunk
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)

CHILD_MAX_CHARS = 1200
PARENT_MAX_CHARS = 4800


def sanitize_text(text: str) -> str:
    """Remove null bytes — PostgreSQL text columns reject \\x00."""
    return text.replace("\x00", "")


class AcademicPDFChunker:
    """Parse PDFs by structural headers rather than arbitrary token lengths."""

    def chunk_pdf(
        self,
        path: str,
        source_name: Optional[str] = None,
        source_type: str = "textbook",
    ) -> List[AcademicChunk]:
        source_name = source_name or Path(path).stem
        try:
            sections = self._extract_sections_pymupdf(path)
        except Exception as e:
            logger.warning("pymupdf extraction failed, using fallback", path=path, error=str(e))
            sections = self._extract_sections_fallback(path)

        chunks: List[AcademicChunk] = []
        for heading_path, body, page_num in sections:
            if not body.strip():
                continue

            heading = heading_path[-1] if heading_path else ""
            structural = detect_structural_type(heading or body[:200])
            parent_text = body[:PARENT_MAX_CHARS]
            child_parts = split_into_child_chunks(body, max_chars=CHILD_MAX_CHARS)

            for child in child_parts:
                tags = extract_tags(child + " " + parent_text)
                chunks.append(
                    AcademicChunk(
                        child_text=child,
                        parent_text=parent_text,
                        heading_path=heading_path,
                        structural_type=structural,
                        source_name=source_name,
                        source_type=source_type,
                        page_number=page_num,
                        tags=tags,
                        section=heading_path[0] if heading_path else "",
                        heading=heading,
                    )
                )

        return chunks

    def chunk_text(
        self,
        text: str,
        source_name: str,
        source_type: str = "textbook",
    ) -> List[AcademicChunk]:
        """Chunk plain text (e.g. from DOCX or pasted content)."""
        text = sanitize_text(text)
        chunks: List[AcademicChunk] = []
        structural = detect_structural_type(text[:500])
        parent_text = text[:PARENT_MAX_CHARS]
        for child in split_into_child_chunks(text, max_chars=CHILD_MAX_CHARS):
            tags = extract_tags(child + " " + parent_text)
            chunks.append(
                AcademicChunk(
                    child_text=child,
                    parent_text=parent_text,
                    heading_path=[],
                    structural_type=structural,
                    source_name=source_name,
                    source_type=source_type,
                    tags=tags,
                )
            )
        return chunks

    def _extract_sections_pymupdf(self, path: str) -> List[Tuple[List[str], str, int]]:
        import fitz  # pymupdf

        doc = fitz.open(path)
        sections: List[Tuple[List[str], str, int]] = []
        heading_path: List[str] = []
        current_body: List[str] = []
        current_page = 1
        body_font_size = 11.0

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if not line_text:
                        continue
                    line_text = sanitize_text(line_text)

                    max_size = max(s.get("size", body_font_size) for s in spans)
                    is_heading = max_size >= body_font_size * 1.25 and len(line_text) < 120

                    if is_heading:
                        if current_body:
                            sections.append(
                                (list(heading_path), "\n".join(current_body), current_page)
                            )
                            current_body = []
                        level = 1 if max_size >= body_font_size * 1.5 else 2
                        if level == 1:
                            heading_path = [line_text]
                        elif heading_path:
                            heading_path = heading_path[:1] + [line_text]
                        else:
                            heading_path = [line_text]
                        current_page = page_num
                    else:
                        current_body.append(line_text)
                        current_page = page_num

        if current_body:
            sections.append((list(heading_path), "\n".join(current_body), current_page))

        doc.close()
        if not sections:
            return self._single_page_fallback(path)
        return sections

    def _extract_sections_fallback(self, path: str) -> List[Tuple[List[str], str, int]]:
        try:
            import fitz
            doc = fitz.open(path)
            text = sanitize_text("\n".join(page.get_text() for page in doc))
            doc.close()
        except Exception:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        return [([], text, 1)]

    def _single_page_fallback(self, path: str) -> List[Tuple[List[str], str, int]]:
        return self._extract_sections_fallback(path)
