from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    ENV: str = "development"
    DEBUG: bool = True

    DATABASE_URL: str = "postgresql+asyncpg://medstats:medstats@localhost:5433/medstats"
    DATABASE_URL_SYNC: str = "postgresql://medstats:medstats@localhost:5433/medstats"
    REDIS_URL: str = "redis://localhost:6380/0"

    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_MODEL_MINI: str = "gpt-4o-mini"

    S3_ENDPOINT_URL: str = ""
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""
    S3_BUCKET_NAME: str = "medstats-files"
    S3_REGION: str = "auto"

    SENTRY_DSN: str = ""

    CELERY_BROKER_URL: str = "redis://localhost:6380/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6380/0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
