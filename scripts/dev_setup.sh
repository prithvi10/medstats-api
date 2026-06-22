#!/bin/bash
# Quick setup for local development.
# Starts PostgreSQL and Redis via Docker, then runs the API.
#
# Usage: ./scripts/dev_setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== MedStats Dev Setup ==="
echo ""

# 1. Create venv if missing
if [ ! -d ".venv" ]; then
    echo "1. Creating virtual environment..."
    python3 -m venv .venv
else
    echo "1. Virtual environment exists."
fi
source .venv/bin/activate

# 2. Install dependencies
echo "2. Installing dependencies..."
pip install -r requirements.txt -q
pip install -r requirements-dev.txt -q
echo "   Done."

# 3. Create .env if missing
if [ ! -f ".env" ]; then
    echo "3. Creating .env file..."
    cat > .env << 'EOF'
ENV=development
DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=postgresql+asyncpg://medstats:medstats@localhost:5432/medstats_dev
SYNC_DATABASE_URL=postgresql://medstats:medstats@localhost:5432/medstats_dev
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=sk-your-key-here
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
EOF
    echo "   Created .env (edit OPENAI_API_KEY for LLM features)"
else
    echo "3. .env already exists."
fi

# 4. Start infra via Docker
echo "4. Starting PostgreSQL and Redis..."
docker compose up -d db redis 2>/dev/null || {
    echo "   Docker Compose not available. Starting containers manually..."
    docker run -d --name medstats-db \
        -e POSTGRES_USER=medstats \
        -e POSTGRES_PASSWORD=medstats \
        -e POSTGRES_DB=medstats_dev \
        -p 5432:5432 \
        postgres:16-alpine 2>/dev/null || echo "   PostgreSQL already running or Docker not available"

    docker run -d --name medstats-redis \
        -p 6379:6379 \
        redis:7-alpine 2>/dev/null || echo "   Redis already running or Docker not available"
}

echo "   Waiting for PostgreSQL..."
sleep 3

# 5. Run migrations
echo "5. Running database migrations..."
alembic upgrade head 2>/dev/null || echo "   Warning: Alembic migrations skipped"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Available commands:"
echo "  ./scripts/run_local.sh              - Start the API server"
echo "  ./scripts/run_tests.sh              - Run all tests"
echo "  ./scripts/run_tests.sh unit         - Run unit tests only"
echo "  ./scripts/run_tests.sh integration  - Run integration tests only"
echo "  python scripts/test_pipeline_direct.py  - Run pipeline without server"
echo "  python scripts/test_api.py          - Test API endpoints (server must be running)"
echo ""
