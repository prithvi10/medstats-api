from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    ENV: str = "development"
    DEBUG: bool = True

    # Default to SQLite for zero-config local dev. Override with PostgreSQL for production.
    DATABASE_URL: str = "sqlite+aiosqlite:///./medstats_dev.db"
    DATABASE_URL_SYNC: str = "sqlite:///./medstats_dev.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.5"
    OPENAI_MODEL_MINI: str = "gpt-5-mini"
    OPENAI_MODEL_REASONING: str = "o3"

    # RAG / GraphRAG
    COHERE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMS: int = 1536
    RAG_TOP_K: int = 5
    RAG_CANDIDATE_K: int = 50
    NCBI_EMAIL: str = "medstats@example.com"
    NCBI_API_KEY: str = ""
    # None = auto (disabled in development/test; enabled in production)
    NCBI_SSL_VERIFY: bool | None = None
    RAG_ENABLED: bool = True

    S3_ENDPOINT_URL: str = ""
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""
    S3_BUCKET_NAME: str = "medstats-files"
    S3_REGION: str = "auto"

    SENTRY_DSN: str = ""

    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # File upload directory (local storage for dev)
    UPLOAD_DIR: str = "/Users/prithviraj/medstats-api/uploads"

    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
