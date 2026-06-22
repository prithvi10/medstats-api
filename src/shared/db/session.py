from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from src.shared.config.settings import get_settings

settings = get_settings()

_db_url = settings.DATABASE_URL

# SQLite fallback for local dev without PostgreSQL
if "sqlite" in _db_url:
    engine = create_async_engine(_db_url, echo=settings.DEBUG)
else:
    engine = create_async_engine(
        _db_url,
        echo=settings.DEBUG,
        pool_size=10,
        max_overflow=20,
    )

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Keep backward compat alias
AsyncSessionLocal = async_session_factory


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create all tables (for SQLite local dev). In production use Alembic."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
