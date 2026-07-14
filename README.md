# MedStats AI — Backend API

AI-powered statistical analysis platform for medical thesis work.

## Quick start — full stack with Docker

Requires `medstats-ui` as a sibling directory:

```
parent/
  medstats-api/
  medstats-ui/
```

```bash
cd medstats-api

# Optional: set OpenAI key for LLM features
echo "OPENAI_API_KEY=sk-..." >> .env

# Start everything (PostgreSQL, Redis, API, Celery worker, UI)
chmod +x scripts/docker-up.sh
./scripts/docker-up.sh --build
```

| Service | URL |
|---------|-----|
| **UI** | http://localhost:3000 |
| **API** | http://localhost:8000 |
| **API docs** | http://localhost:8000/docs |
| Postgres | localhost:5433 |
| Redis | localhost:6380 |

Demo logins (auto-seeded): `prof@medstats.io` / `admin123`, `student@medstats.io` / `admin123`

```bash
# Stop
docker compose down

# View logs
docker compose logs -f api ui

# Run tests in Docker
docker compose --profile test run --rm test
```

The UI container proxies `/api` to the API container. No separate frontend setup needed.

## Local development (without Docker UI)

```bash
# Infra only
docker compose up -d db redis

# API on host
./scripts/run_local.sh

# UI on host (separate terminal)
cd ../medstats-ui && npm run dev
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
