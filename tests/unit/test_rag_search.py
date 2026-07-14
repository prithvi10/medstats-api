"""Tests for RAG reranker and RRF fusion."""
import pytest

from src.medstats.rag.models import HybridSearchResult
from src.medstats.rag.reranker import CohereReranker
from src.medstats.rag.search import HybridSearchEngine, RRF_K


class TestRRFFusion:
    def test_rrf_combines_lists(self):
        engine = HybridSearchEngine.__new__(HybridSearchEngine)
        list_a = [
            HybridSearchResult("1", "a", "parent a", "src", 0.9),
            HybridSearchResult("2", "b", "parent b", "src", 0.8),
        ]
        list_b = [
            HybridSearchResult("2", "b", "parent b", "src", 0.95),
            HybridSearchResult("3", "c", "parent c", "src", 0.7),
        ]
        fused = engine._reciprocal_rank_fusion([list_a, list_b])
        ids = [f.doc_id for f in fused]
        assert "2" in ids
        assert ids[0] == "2"  # appears in both lists, highest RRF


@pytest.mark.asyncio
async def test_local_rerank_orders_by_overlap():
    reranker = CohereReranker(api_key="")
    candidates = [
        HybridSearchResult("1", "Mann-Whitney U test for non-normal", "full mann text", "Book", 0.5),
        HybridSearchResult("2", "Chi-square for categorical", "full chi text", "Book", 0.5),
    ]
    ranked = await reranker.rerank(
        "Mann-Whitney non-normal distribution",
        candidates,
        top_n=2,
    )
    assert ranked[0].doc_id == "1"
