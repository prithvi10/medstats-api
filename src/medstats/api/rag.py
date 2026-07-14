"""RAG explain endpoint for student-facing test rationale."""
from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.agents.rag_knowledge import get_rag_agent
from src.medstats.db.models import AnalysisJob, Submission
from src.shared.db.session import get_db
from src.shared.logging.logger import get_logger
from src.shared.security.auth import get_current_user

router = APIRouter()
logger = get_logger(__name__)

# In-memory rate limit: submission_id -> call count (resets on process restart)
_explain_call_counts: dict[str, int] = {}
MAX_EXPLAIN_CALLS_PER_SUBMISSION = 5

_DISTILL_SYSTEM = (
    "You are a concise statistics tutor. Given raw textbook passages, write a "
    "short, plain-English explanation (2-4 sentences) that directly answers the "
    "student's question. Avoid equations, footnotes, and textbook-specific examples. "
    "Focus only on the reasoning behind the test choice."
)


class RAGExplainRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    submission_id: str
    tags: Optional[List[str]] = None


class RAGCitationResponse(BaseModel):
    source: str
    page: Optional[int] = None
    section: Optional[str] = None


class RAGExplainResponse(BaseModel):
    passages: List[str]
    citations: List[RAGCitationResponse]


async def _distill_passages(query: str, raw_passages: List[str]) -> str:
    """Summarise raw textbook passages into a concise student-friendly answer."""
    try:
        from src.agentic_core.llm.service import LLMService
        from src.shared.config.settings import get_settings
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            return raw_passages[0] if raw_passages else ""
        llm = LLMService()
        combined = "\n\n---\n\n".join(raw_passages[:3])
        prompt = (
            f"Student question: {query}\n\n"
            f"Textbook passages:\n{combined}\n\n"
            "Write a concise 2-4 sentence explanation that directly answers the question."
        )
        return await asyncio.wait_for(
            llm.generate_text(
                prompt=prompt,
                system_prompt=_DISTILL_SYSTEM,
                temperature=0.3,
                max_tokens=200,
            ),
            timeout=20.0,
        )
    except Exception as e:
        logger.warning("Passage distillation failed, falling back to raw", error=str(e))
        return raw_passages[0] if raw_passages else ""


@router.post("/explain", response_model=RAGExplainResponse)
async def explain_test_rationale(
    body: RAGExplainRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Explain why a statistical test applies — grounded in the knowledge base."""
    sub_result = await db.execute(
        select(Submission).where(Submission.id == body.submission_id)
    )
    submission = sub_result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    role = current_user.get("role")
    user_id = current_user.get("sub")
    if role == "student" and str(submission.student_id) != str(user_id):
        raise HTTPException(status_code=403, detail="Not authorized")

    count = _explain_call_counts.get(body.submission_id, 0)
    if count >= MAX_EXPLAIN_CALLS_PER_SUBMISSION:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: max {MAX_EXPLAIN_CALLS_PER_SUBMISSION} explain calls per submission",
        )

    rag_agent = get_rag_agent()
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG knowledge base unavailable")

    # Enrich tags from agent state if available
    tags = list(body.tags or [])
    job_result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.submission_id == body.submission_id)
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )
    job = job_result.scalar_one_or_none()
    if job and job.agent_state:
        import json
        raw = job.agent_state
        state = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(state, dict) and state.get("study_design"):
            if state["study_design"] not in tags:
                tags.append(state["study_design"])

    result = await rag_agent.retrieve(body.query, tags=tags or None, top_k=3)
    _explain_call_counts[body.submission_id] = count + 1

    # Distill raw passages into a concise plain-English answer
    if result.passages:
        summary = await _distill_passages(body.query, result.passages)
        passages = [summary]
    else:
        passages = []

    return RAGExplainResponse(
        passages=passages,
        citations=[
            RAGCitationResponse(
                source=c.source,
                page=c.page,
                section=c.section,
            )
            for c in result.citations
        ],
    )
