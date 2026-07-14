"""Live RAG retrieval tests against Postgres corpus (skip if unavailable)."""
from __future__ import annotations

import pytest

from tests.rag_eval.runner import rag_prerequisites_met, run_all_cases

pytestmark = [pytest.mark.integration, pytest.mark.rag]


@pytest.fixture(scope="module")
def rag_ready():
    if not rag_prerequisites_met():
        pytest.skip("PostgreSQL RAG corpus + OPENAI_API_KEY required")
    return True


@pytest.fixture(scope="module")
def corpus(rag_ready):
    from sqlalchemy import create_engine, text

    from src.shared.config.settings import get_settings

    engine = create_engine(get_settings().DATABASE_URL_SYNC)
    with engine.connect() as conn:
        stats = {
            "rag_documents": conn.execute(text("SELECT COUNT(*) FROM rag_documents")).scalar_one(),
            "literature_index": conn.execute(text("SELECT COUNT(*) FROM literature_index")).scalar_one(),
            "knowledge_nodes": conn.execute(text("SELECT COUNT(*) FROM knowledge_nodes")).scalar_one(),
        }
    if stats["rag_documents"] < 10:
        pytest.skip("rag_documents corpus empty — run build_rag_corpus.py")
    if stats["literature_index"] < 10:
        pytest.skip("literature_index empty — run build_rag_corpus.py --pubmed")
    return stats


def test_corpus_minimum_size(corpus):
    assert corpus["rag_documents"] >= 100
    assert corpus["literature_index"] >= 500
    assert corpus["knowledge_nodes"] >= 20


@pytest.mark.asyncio
async def test_all_eval_cases_summary(corpus):
    """Run all masterchart/thesis scenario probes in one event loop."""
    summary = await run_all_cases()
    assert summary["failed"] == 0, _format_failures(summary)


def _format_failures(summary: dict) -> str:
    lines = [f"{summary['passed']}/{summary['total']} cases passed"]
    for r in summary["results"]:
        if not r["passed"]:
            lines.append(
                f"  FAIL {r['case_id']}: "
                f"knowledge={r['knowledge']['passed']} literature={r['literature']['passed']}"
            )
            if not r["knowledge"]["passed"]:
                lines.append(f"    knowledge query: {r['knowledge']['query'][:120]}")
            if not r["literature"]["passed"]:
                lines.append(f"    literature matches: {r['literature']['match_count']}")
    return "\n".join(lines)
