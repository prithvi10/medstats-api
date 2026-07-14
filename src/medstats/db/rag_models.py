"""SQLAlchemy models for GraphRAG tables (PostgreSQL + pgvector, SQLite dev fallback)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship

from src.shared.config.settings import get_settings
from src.shared.db.session import Base
from src.medstats.db.models import GUID

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # type: ignore

settings = get_settings()
_IS_POSTGRES = "postgresql" in (settings.DATABASE_URL or "")


def _string_array_column(nullable: bool = True):
    if _IS_POSTGRES:
        return Column(ARRAY(String), nullable=nullable)
    return Column(JSON, nullable=nullable)


def _json_column(nullable: bool = True):
    if _IS_POSTGRES:
        return Column(JSONB, nullable=nullable)
    return Column(JSON, nullable=nullable)


def _vector_column(dims: int):
    if _IS_POSTGRES and Vector is not None:
        return Column(Vector(dims), nullable=True)
    return Column(Text, nullable=True)


def serialize_embedding(embedding: Optional[List[float]]) -> Optional[str | List[float]]:
    """Store embeddings as pgvector list (Postgres) or JSON text (SQLite)."""
    if embedding is None:
        return None
    if _IS_POSTGRES:
        return embedding
    return json.dumps(embedding)


class RAGDocument(Base):
    __tablename__ = "rag_documents"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    source_type = Column(String(50), nullable=False, index=True)
    source_name = Column(String(500), nullable=False)
    source_url = Column(String(1000), nullable=True)
    section = Column(String(500), nullable=True)
    heading = Column(String(500), nullable=True)
    heading_path = _string_array_column()
    structural_type = Column(String(50), nullable=True)
    content = Column(Text, nullable=False)
    parent_content = Column(Text, nullable=True)
    embedding = _vector_column(1536)
    doc_metadata = _json_column()
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class KnowledgeNode(Base):
    __tablename__ = "knowledge_nodes"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(50), nullable=False, index=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    aliases = _string_array_column()
    embedding = _vector_column(1536)
    properties = _json_column()

    outgoing_edges = relationship(
        "KnowledgeEdge",
        foreign_keys="KnowledgeEdge.from_node",
        back_populates="source_node",
    )
    incoming_edges = relationship(
        "KnowledgeEdge",
        foreign_keys="KnowledgeEdge.to_node",
        back_populates="target_node",
    )


class KnowledgeEdge(Base):
    __tablename__ = "knowledge_edges"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    from_node = Column(GUID(), ForeignKey("knowledge_nodes.id", ondelete="CASCADE"), nullable=False)
    to_node = Column(GUID(), ForeignKey("knowledge_nodes.id", ondelete="CASCADE"), nullable=False)
    relation = Column(String(100), nullable=False)
    weight = Column(Float, default=1.0)
    properties = _json_column()

    source_node = relationship(
        "KnowledgeNode",
        foreign_keys=[from_node],
        back_populates="outgoing_edges",
    )
    target_node = relationship(
        "KnowledgeNode",
        foreign_keys=[to_node],
        back_populates="incoming_edges",
    )


class LiteratureIndex(Base):
    __tablename__ = "literature_index"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    pmid = Column(String(20), unique=True, nullable=True, index=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    authors = _string_array_column()
    year = Column(Integer, nullable=True)
    journal = Column(String(500), nullable=True)
    mesh_terms = _string_array_column()
    methods_tags = _string_array_column()
    disciplines = _string_array_column()
    embedding = _vector_column(1536)
    indexed_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
