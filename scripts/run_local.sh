#!/bin/bash
# Start the MedStats API locally (without Docker).
# Requires: PostgreSQL running on localhost:5432 with database "medstats_dev"
#
# Usage: ./scripts/run_local.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit .env with your local settings, then re-run."
    exit 1
fi

export $(grep -v '^#' .env | xargs)

echo "=== MedStats API Local Runner ==="
echo "Database: ${DATABASE_URL:-not set}"
echo "Port: 8000"
echo ""

# Run migrations
echo "Running database migrations..."
alembic upgrade head 2>/dev/null || echo "Warning: migrations skipped (DB may not be ready)"

echo ""
echo "Starting API server on http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""

uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
