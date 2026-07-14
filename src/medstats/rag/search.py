"""Hybrid search: dense vector + BM25 + graph expansion with RRF fusion."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.db.rag_models import RAGDocument
from src.medstats.rag.graph import GraphExpander
from src.medstats.rag.models import HybridSearchResult
from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
RRF_K = 60


class HybridSearchEngine:
    """Combines semantic, keyword, and graph-based retrieval."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.graph = GraphExpander(session)

    async def search(
        self,
        query: str,
        query_embedding: List[float],
        tags_filter: Optional[List[str]] = None,
        k_dense: int = 30,
        k_sparse: int = 20,
        candidate_k: Optional[int] = None,
    ) -> List[HybridSearchResult]:
        candidate_k = candidate_k or settings.RAG_CANDIDATE_K

        dense = await self._dense_search(query_embedding, k_dense, tags_filter)
        sparse = await self._bm25_search(query, k_sparse, tags_filter)
        graph_tags = tags_filter or []
        if dense:
            for r in dense[:3]:
                meta = r.doc_metadata or {}
                graph_tags.extend(meta.get("test_tags", []))
                graph_tags.extend(meta.get("concept_tags", [])[:3])

        graph_raw = await self.graph.expand_from_tags(list(set(graph_tags)), limit=15)
        graph = [
            HybridSearchResult(
                doc_id=g["doc_id"],
                content=g["content"],
                parent_content=g["parent_content"],
                source_name=g["source_name"],
                score=g["score"],
                source_type=g.get("source_type", "document"),
                doc_metadata=g.get("doc_metadata"),
            )
            for g in graph_raw
        ]

        fused = self._reciprocal_rank_fusion([dense, sparse, graph])
        return fused[:candidate_k]

    async def _dense_search(
        self,
        embedding: List[float],
        limit: int,
        tags_filter: Optional[List[str]],
    ) -> List[HybridSearchResult]:
        if not embedding or "sqlite" in settings.DATABASE_URL:
            return []

        try:
            emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
            tag_clause = ""
            params: Dict[str, Any] = {"emb": emb_str, "limit": limit}

            if tags_filter:
                tag_clause = """
                AND (
                    doc_metadata->'test_tags' ?| :tags
                    OR doc_metadata->'concept_tags' ?| :tags
                    OR doc_metadata IS NULL
                )
                """
                params["tags"] = tags_filter

            sql = text(f"""
                SELECT id, content, parent_content, source_name, source_type, doc_metadata,
                       1 - (embedding <=> CAST(:emb AS vector)) AS score
                FROM rag_documents
                WHERE embedding IS NOT NULL
                {tag_clause}
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :limit
            """)
            rows = (await self.session.execute(sql, params)).mappings()
        except Exception as e:
            logger.warning("Dense search failed", error=str(e))
            return []

        return [
            HybridSearchResult(
                doc_id=str(r["id"]),
                content=r["content"],
                parent_content=r["parent_content"] or r["content"],
                source_name=r["source_name"],
                score=float(r["score"]),
                source_type=r["source_type"] or "document",
                doc_metadata=r["doc_metadata"],
            )
            for r in rows
        ]

    async def _bm25_search(
        self,
        query: str,
        limit: int,
        tags_filter: Optional[List[str]],
    ) -> List[HybridSearchResult]:
        if "sqlite" in settings.DATABASE_URL:
            return await self._sqlite_keyword_search(query, limit)

        try:
            sql = text("""
                SELECT id, content, parent_content, source_name, source_type, doc_metadata,
                       ts_rank(to_tsvector('english', content), plainto_tsquery('english', :q)) AS score
                FROM rag_documents
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :q)
                ORDER BY score DESC
                LIMIT :limit
            """)
            rows = (await self.session.execute(sql, {"q": query, "limit": limit})).mappings()
        except Exception as e:
            logger.warning("BM25 search failed", error=str(e))
            return []

        return [
            HybridSearchResult(
                doc_id=str(r["id"]),
                content=r["content"],
                parent_content=r["parent_content"] or r["content"],
                source_name=r["source_name"],
                score=float(r["score"]),
                source_type=r["source_type"] or "document",
                doc_metadata=r["doc_metadata"],
            )
            for r in rows
        ]

    async def _sqlite_keyword_search(self, query: str, limit: int) -> List[HybridSearchResult]:
        """Fallback keyword search for SQLite dev."""
        words = [w for w in query.lower().split() if len(w) > 2]
        if not words:
            return []

        results = await self.session.execute(select(RAGDocument).limit(limit * 3))
        docs = results.scalars().all()
        scored: List[HybridSearchResult] = []
        for doc in docs:
            lower = doc.content.lower()
            hits = sum(1 for w in words if w in lower)
            if hits:
                scored.append(
                    HybridSearchResult(
                        doc_id=str(doc.id),
                        content=doc.content,
                        parent_content=doc.parent_content or doc.content,
                        source_name=doc.source_name,
                        score=hits / len(words),
                        source_type=doc.source_type or "document",
                        doc_metadata=doc.doc_metadata,
                    )
                )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[HybridSearchResult]],
    ) -> List[HybridSearchResult]:
        scores: Dict[str, float] = {}
        by_id: Dict[str, HybridSearchResult] = {}

        for results in result_lists:
            for rank, item in enumerate(results):
                scores[item.doc_id] = scores.get(item.doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                if item.doc_id not in by_id or item.score > by_id[item.doc_id].score:
                    by_id[item.doc_id] = item

        sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        fused = []
        for doc_id in sorted_ids:
            item = by_id[doc_id]
            fused.append(
                HybridSearchResult(
                    doc_id=item.doc_id,
                    content=item.content,
                    parent_content=item.parent_content,
                    source_name=item.source_name,
                    score=scores[doc_id],
                    source_type=item.source_type,
                    doc_metadata=item.doc_metadata,
                )
            )
        return fused
