# MedStats AI — Backend API

AI-powered statistical analysis platform for medical thesis work.

## Setup

```bash
# Start services
docker-compose up -d

# Run migrations
alembic upgrade head

# Start API (development)
uvicorn src.main:app --reload --port 8000
```

## Architecture

```
src/
├── agentic_core/    # Domain-agnostic agent framework
├── medstats/        # Medical statistics domain layer
├── shared/          # Cross-cutting concerns (auth, DB, logging)
└── main.py          # FastAPI application entry
```

## Environment

Copy `.env.example` to `.env` and fill in your values.
