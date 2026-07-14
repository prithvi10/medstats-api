from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from src.shared.config.settings import get_settings
from src.shared.logging.logger import setup_logging, get_logger
from src.medstats.api.auth import router as auth_router
from src.medstats.api.student import router as student_router
from src.medstats.api.professor import router as professor_router
from src.medstats.api.analysis import router as analysis_router
from src.medstats.api.admin import router as admin_router
from src.medstats.api.notifications import router as notifications_router
from src.medstats.api.rag import router as rag_router

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO")
    logger.info("MedStats API starting", env=settings.ENV)

    # Auto-create tables for local dev (SQLite or Docker PostgreSQL)
    if settings.ENV in ("development", "test") or "sqlite" in settings.DATABASE_URL:
        from src.shared.db.session import init_db
        from src.medstats.db import models  # noqa: F401 — register ORM models
        from src.medstats.db import rag_models  # noqa: F401
        await init_db()
        logger.info("Database tables initialized")

    yield
    logger.info("MedStats API shutting down")


app = FastAPI(
    title="MedStats AI API",
    description="AI-powered statistical analysis for medical thesis",
    version="0.1.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(student_router, prefix="/api/student", tags=["Student"])
app.include_router(professor_router, prefix="/api/professor", tags=["Professor"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])
app.include_router(notifications_router, prefix="/api/notifications", tags=["Notifications"])
app.include_router(rag_router, prefix="/api/rag", tags=["RAG"])

# Serve generated charts as static files
CHARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated_charts")
os.makedirs(CHARTS_DIR, exist_ok=True)
app.mount("/api/charts", StaticFiles(directory=CHARTS_DIR), name="charts")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "medstats-api", "env": settings.ENV}
