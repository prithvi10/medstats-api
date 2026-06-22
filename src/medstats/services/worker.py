"""
Celery worker for async analysis pipeline execution.
"""
from celery import Celery
from src.shared.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "medstats",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="medstats.run_analysis_pipeline")
def run_analysis_pipeline(self, job_id: str):
    """
    Main analysis pipeline task.
    Runs: DocParser -> TestSelector -> StatsEngine -> ReportGenerator
    """
    import asyncio
    asyncio.run(_run_pipeline_async(job_id))


async def _run_pipeline_async(job_id: str):
    """Async pipeline execution."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker
    from src.shared.config.settings import get_settings
    from src.medstats.db.models import AnalysisJob, Submission, SubmissionFile, Report, JobStatus, SubmissionStatus
    from src.medstats.agents.document_parser import DocumentParserAgent
    from src.medstats.agents.test_selector import TestSelectorAgent
    from src.medstats.agents.report_generator import ReportGeneratorAgent
    from src.agentic_core.agents.base_agent import MissionState
    from src.agentic_core.agents.orchestrator import MissionControl
    from src.agentic_core.llm.service import LLMService
    from datetime import datetime, timezone

    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL_SYNC)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        job.current_step = "Initializing pipeline"
        db.commit()

        submission = db.query(Submission).filter(Submission.id == job.submission_id).first()
        files = db.query(SubmissionFile).filter(SubmissionFile.submission_id == submission.id).all()

        llm = LLMService()

        state = MissionState(
            submission_id=str(submission.id),
            job_id=str(job.id),
        )
        state.agent_outputs["files"] = [
            {"path": f.file_url, "type": f.file_type, "name": f.original_name}
            for f in files
        ]
        state.agent_outputs["intent"] = submission.intent_description or ""
        state.agent_outputs["title"] = submission.title

        agents = [
            DocumentParserAgent(services=llm),
            TestSelectorAgent(services=llm),
            ReportGeneratorAgent(services=llm),
        ]

        async def progress_cb(s: MissionState):
            job.progress = s.progress
            job.current_step = s.logs[-1] if s.logs else ""
            db.commit()

        orchestrator = MissionControl(agents=agents, progress_callback=progress_cb)

        async for updated_state in orchestrator.execute(state):
            job.agent_state = updated_state.to_dict()
            db.commit()

        if updated_state.status == "COMPLETED":
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.current_step = "Analysis complete"
            job.test_results = updated_state.agent_outputs.get("test_results")
            job.charts_metadata = updated_state.agent_outputs.get("charts")

            report = Report(
                submission_id=submission.id,
                analysis_job_id=job.id,
                version=1,
                sections=updated_state.agent_outputs.get("report_sections"),
                charts=updated_state.agent_outputs.get("charts"),
                summary=updated_state.agent_outputs.get("summary"),
                methodology_note=updated_state.agent_outputs.get("methodology_note"),
            )
            db.add(report)
            submission.status = SubmissionStatus.REPORT_READY
        else:
            job.status = JobStatus.FAILED
            job.error_log = updated_state.error_message

        job.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_log = str(e)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()
