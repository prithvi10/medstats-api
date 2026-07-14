"""Cross-encoder reranking: Cohere API with local BGE fallback."""
from __future__ import annotations

from typing import List, Optional

from src.medstats.rag.models import HybridSearchResult, RankedResult, Citation
from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CohereReranker:
    """Rerank hybrid search candidates to top-k using cross-encoder models."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.COHERE_API_KEY
        self._local_model = None

    async def rerank(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_n: Optional[int] = None,
    ) -> List[RankedResult]:
        top_n = top_n or settings.RAG_TOP_K
        if not candidates:
            return []

        documents = [c.parent_content or c.content for c in candidates]

        if self.api_key:
            try:
                scores = await self._cohere_rerank(query, documents)
            except Exception as e:
                logger.warning("Cohere rerank failed, using local fallback", error=str(e))
                scores = self._local_rerank(query, documents)
        else:
            scores = self._local_rerank(query, documents)

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        results: List[RankedResult] = []
        for idx, score in indexed:
            c = candidates[idx]
            meta = c.doc_metadata or {}
            page = meta.get("page_number")
            results.append(
                RankedResult(
                    doc_id=c.doc_id,
                    content=c.content,
                    parent_content=c.parent_content or c.content,
                    source_name=c.source_name,
                    rerank_score=score,
                    source_type=c.source_type,
                    citation=Citation(
                        source=c.source_name,
                        page=page,
                        section=meta.get("section"),
                    ),
                )
            )
        return results

    async def _cohere_rerank(self, query: str, documents: List[str]) -> List[float]:
        import cohere

        client = cohere.AsyncClient(api_key=self.api_key)
        response = await client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=len(documents),
        )
        scores = [0.0] * len(documents)
        for item in response.results:
            scores[item.index] = item.relevance_score
        return scores

    def _local_rerank(self, query: str, documents: List[str]) -> List[float]:
        """Keyword overlap fallback when no reranker API available."""
        query_words = set(query.lower().split())
        scores: List[float] = []
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scores.append(overlap / max(len(query_words), 1))
        return scores

    def _local_bge_rerank(self, query: str, documents: List[str]) -> List[float]:
        """Optional local cross-encoder via sentence-transformers."""
        try:
            if self._local_model is None:
                from sentence_transformers import CrossEncoder
                self._local_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
            pairs = [[query, d] for d in documents]
            scores = self._local_model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as e:
            logger.warning("BGE reranker unavailable", error=str(e))
            return self._local_rerank(query, documents)
