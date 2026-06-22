from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.shared.config.settings import get_settings
from src.shared.logging.logger import setup_logging, get_logger
from src.medstats.api.auth import router as auth_router
from src.medstats.api.student import router as student_router
from src.medstats.api.professor import router as professor_router
from src.medstats.api.analysis import router as analysis_router
from src.medstats.api.admin import router as admin_router

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO")
    logger.info("MedStats API starting", env=settings.ENV)
    yield
    logger.info("MedStats API shutting down")


app = FastAPI(
    title="MedStats AI API",
    description="AI-powered statistical analysis for medical thesis",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(student_router, prefix="/api/student", tags=["Student"])
app.include_router(professor_router, prefix="/api/professor", tags=["Professor"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "medstats-api"}
