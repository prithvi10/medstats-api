"""Tests for academic chunking and tag extraction."""
import pytest

from src.medstats.rag.chunking import AcademicPDFChunker
from src.medstats.rag.chunking_utils import extract_tags, detect_structural_type, split_into_child_chunks
from src.medstats.rag.corpus.knowledge_seed import get_seed_nodes, get_seed_edges


class TestChunkingUtils:
    def test_extract_tags_fisher_and_mann_whitney(self):
        text = "Use Fisher's exact test when chi-square expected cells are below 5. Mann-Whitney U for non-normal data."
        tags = extract_tags(text)
        assert "fisher_exact" in tags
        assert "mann_whitney_u" in tags
        assert "chi_square" in tags

    def test_detect_structural_type_methodology(self):
        assert detect_structural_type("Statistical Analysis Methods") == "methodology"

    def test_split_into_child_chunks(self):
        text = "Para one.\n\nPara two with more content.\n\nPara three."
        chunks = split_into_child_chunks(text, max_chars=30)
        assert len(chunks) >= 2

    def test_split_oversized_pdf_line_block(self):
        # PDFs often use single newlines with no blank lines between paragraphs
        text = "\n".join(f"Line {i} with statistical analysis content." for i in range(200))
        chunks = split_into_child_chunks(text, max_chars=1200)
        assert len(chunks) > 1
        assert all(len(c) <= 1300 for c in chunks)


class TestAcademicPDFChunker:
    def test_chunk_text_produces_parent_child(self):
        chunker = AcademicPDFChunker()
        text = (
            "Statistical Analysis\n\n"
            "The Mann-Whitney U test compares two independent groups when normality is violated. "
            "Fisher's exact test is preferred for small categorical samples."
        )
        chunks = chunker.chunk_text(text, source_name="Test Book", source_type="textbook")
        assert len(chunks) >= 1
        assert chunks[0].child_text
        assert chunks[0].parent_text
        assert "mann_whitney_u" in chunks[0].tags or "fisher_exact" in chunks[0].tags


class TestKnowledgeSeed:
    def test_seed_has_minimum_nodes(self):
        nodes = get_seed_nodes()
        assert len(nodes) >= 30

    def test_seed_has_edges(self):
        edges = get_seed_edges()
        assert len(edges) >= 40

    def test_fisher_alternative_to_chi_square(self):
        edges = get_seed_edges()
        assert any(
            e["from_name"] == "Fisher's exact test"
            and e["to_name"] == "Chi-square test"
            and e["relation"] == "is_alternative_to"
            for e in edges
        )
