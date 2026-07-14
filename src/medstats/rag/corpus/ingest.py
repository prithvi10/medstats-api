"""Corpus ingestion: PDFs and knowledge graph seed into PostgreSQL."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.db.rag_models import KnowledgeEdge, KnowledgeNode, RAGDocument, serialize_embedding
from src.medstats.rag.chunking import AcademicPDFChunker
from src.medstats.rag.corpus.knowledge_seed import get_seed_edges, get_seed_nodes
from src.medstats.rag.embedder import ChunkEmbedder
from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def seed_knowledge_graph(session: AsyncSession, embedder: Optional[ChunkEmbedder] = None) -> int:
    """Insert knowledge graph nodes and edges from seed data."""
    nodes = get_seed_nodes()
    edges = get_seed_edges()

    name_to_id: dict = {}
    count = 0

    for spec in nodes:
        with session.no_autoflush:
            existing = await session.execute(
                select(KnowledgeNode).where(KnowledgeNode.name == spec["name"])
            )
            row = existing.scalar_one_or_none()
        if row:
            name_to_id[spec["name"]] = row.id
            continue

        node_id = uuid.uuid4()
        embedding = None
        if embedder and settings.OPENAI_API_KEY:
            desc = f"{spec['name']}. {spec.get('description', '')}"
            emb = await embedder.embed_query(desc)
            embedding = emb

        node = KnowledgeNode(
            id=node_id,
            entity_type=spec["entity_type"],
            name=spec["name"],
            description=spec.get("description"),
            aliases=spec.get("aliases"),
            properties=spec.get("properties"),
            embedding=serialize_embedding(emb),
        )
        session.add(node)
        name_to_id[spec["name"]] = node_id
        count += 1

    await session.flush()

    edge_count = 0
    for spec in edges:
        from_id = name_to_id.get(spec["from_name"])
        to_id = name_to_id.get(spec["to_name"])
        if not from_id or not to_id:
            continue
        edge = KnowledgeEdge(
            id=uuid.uuid4(),
            from_node=from_id,
            to_node=to_id,
            relation=spec["relation"],
            weight=spec.get("weight", 1.0),
            properties=spec.get("properties"),
        )
        session.add(edge)
        edge_count += 1

    await session.commit()
    logger.info("Knowledge graph seeded", nodes=count, edges=edge_count)
    return count + edge_count


async def ingest_pdf(
    session: AsyncSession,
    pdf_path: str,
    source_type: str = "textbook",
    source_name: Optional[str] = None,
    embedder: Optional[ChunkEmbedder] = None,
) -> int:
    """Chunk and embed a PDF into rag_documents."""
    chunker = AcademicPDFChunker()
    chunks = chunker.chunk_pdf(pdf_path, source_name=source_name, source_type=source_type)
    if not chunks:
        logger.warning("No chunks extracted", path=pdf_path)
        return 0

    embedder = embedder or ChunkEmbedder()
    texts = [c.child_text for c in chunks]
    embeddings: List[List[float]] = []
    if settings.OPENAI_API_KEY:
        embeddings = await embedder.embed_texts(texts)
    else:
        embeddings = [[] for _ in chunks]

    for i, chunk in enumerate(chunks):
        doc = RAGDocument(
            id=uuid.uuid4(),
            source_type=chunk.source_type,
            source_name=chunk.source_name,
            section=chunk.section,
            heading=chunk.heading,
            heading_path=chunk.heading_path or None,
            structural_type=chunk.structural_type,
            content=chunk.child_text,
            parent_content=chunk.parent_text,
            embedding=serialize_embedding(
                embeddings[i] if i < len(embeddings) and embeddings[i] else None
            ),
            doc_metadata={
                "page_number": chunk.page_number,
                "test_tags": [t for t in chunk.tags if t in {
                    "chi_square", "fisher_exact", "mann_whitney_u", "unpaired_t_test",
                    "paired_t_test", "one_way_anova", "kruskal_wallis", "shapiro_wilk",
                    "z_test_proportion",
                }],
                "design_tags": [t for t in chunk.tags if t in {
                    "single_cohort", "two_group", "multi_group",
                }],
                "concept_tags": chunk.tags,
            },
        )
        session.add(doc)

    await session.commit()
    logger.info("PDF ingested", path=pdf_path, chunks=len(chunks))
    return len(chunks)


async def ingest_text_corpus(
    session: AsyncSession,
    entries: List[dict],
    embedder: Optional[ChunkEmbedder] = None,
) -> int:
    """
    Ingest plain-text corpus entries (for tests / bootstrap without PDFs).

    Each entry: {source_name, source_type, text}
    """
    chunker = AcademicPDFChunker()
    embedder = embedder or ChunkEmbedder()
    total = 0

    for entry in entries:
        chunks = chunker.chunk_text(
            entry["text"],
            source_name=entry["source_name"],
            source_type=entry.get("source_type", "textbook"),
        )
        texts = [c.child_text for c in chunks]
        embeddings = await embedder.embed_texts(texts) if settings.OPENAI_API_KEY else [[]] * len(chunks)

        for i, chunk in enumerate(chunks):
            session.add(
                RAGDocument(
                    id=uuid.uuid4(),
                    source_type=chunk.source_type,
                    source_name=chunk.source_name,
                    content=chunk.child_text,
                    parent_content=chunk.parent_text,
                    heading_path=chunk.heading_path or None,
                    structural_type=chunk.structural_type,
                    embedding=serialize_embedding(
                        embeddings[i] if embeddings[i] else None
                    ),
                    doc_metadata={"concept_tags": chunk.tags, "test_tags": chunk.tags},
                )
            )
            total += 1

    await session.commit()
    return total


def is_rag_available() -> bool:
    """True if database supports vector search (PostgreSQL + pgvector)."""
    url = settings.DATABASE_URL_SYNC or settings.DATABASE_URL
    return "postgresql" in url and settings.RAG_ENABLED
