"""Parent-child retrieval orchestrator."""
from __future__ import annotations

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.rag.embedder import ChunkEmbedder
from src.medstats.rag.models import RAGResult, Citation
from src.medstats.rag.reranker import CohereReranker
from src.medstats.rag.search import HybridSearchEngine
from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ParentChildRetriever:
    """
    Search on child chunks for precision; return parent_content to the LLM.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedder: Optional[ChunkEmbedder] = None,
        reranker: Optional[CohereReranker] = None,
    ):
        self.session = session
        self.embedder = embedder or ChunkEmbedder()
        self.reranker = reranker or CohereReranker()
        self.search_engine = HybridSearchEngine(session)

    async def retrieve(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResult:
        top_k = top_k or settings.RAG_TOP_K

        try:
            query_embedding = await self.embedder.embed_query(query)
        except Exception as e:
            logger.warning("Embedding failed, using keyword-only", error=str(e))
            query_embedding = []

        candidates = await self.search_engine.search(
            query=query,
            query_embedding=query_embedding,
            tags_filter=tags,
        )

        ranked = await self.reranker.rerank(query, candidates, top_n=top_k)

        passages = []
        citations: List[Citation] = []
        seen_parents = set()

        for r in ranked:
            parent = r.parent_content or r.content
            if parent in seen_parents:
                continue
            seen_parents.add(parent)

            cite_str = r.source_name
            if r.citation and r.citation.page:
                cite_str += f", p.{r.citation.page}"
            passages.append(f"{parent}\n[{cite_str}]")

            if r.citation:
                citations.append(r.citation)

        return RAGResult(
            passages=passages,
            citations=citations,
            ranked_results=ranked,
        )
