"""GraphRAG data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AcademicChunk:
    """A hierarchical chunk from an academic document."""

    child_text: str
    parent_text: str
    heading_path: List[str]
    structural_type: str
    source_name: str
    source_type: str
    page_number: int = 0
    tags: List[str] = field(default_factory=list)
    section: str = ""
    heading: str = ""


@dataclass
class Citation:
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    pmid: Optional[str] = None


@dataclass
class HybridSearchResult:
    doc_id: str
    content: str
    parent_content: str
    source_name: str
    score: float
    source_type: str = "document"
    doc_metadata: Optional[Dict[str, Any]] = None


@dataclass
class RankedResult:
    doc_id: str
    content: str
    parent_content: str
    source_name: str
    rerank_score: float
    source_type: str = "document"
    citation: Optional[Citation] = None


@dataclass
class RAGResult:
    passages: List[str]
    citations: List[Citation]
    ranked_results: List[RankedResult] = field(default_factory=list)


@dataclass
class LiteratureMatch:
    pmid: Optional[str]
    title: str
    authors: List[str]
    year: Optional[int]
    journal: Optional[str]
    abstract_snippet: str
    similarity_score: float
    relevance_note: str = ""
