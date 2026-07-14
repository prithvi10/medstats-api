#!/bin/bash
# Start the full MedStats stack (API + UI + PostgreSQL + Redis) for local testing.
#
# Prerequisites:
#   - Docker Desktop / Docker Engine running
#   - medstats-ui cloned as a sibling of medstats-api:
#       ~/medstats-api
#       ~/medstats-ui
#
# Usage:
#   ./scripts/docker-up.sh          # start in background
#   ./scripts/docker-up.sh --build  # rebuild images first
#   ./scripts/docker-up.sh --logs     # follow logs after start

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UI_DIR="$(cd "$PROJECT_DIR/../medstats-ui" 2>/dev/null && pwd || true)"

cd "$PROJECT_DIR"

if [ -z "$UI_DIR" ] || [ ! -f "$UI_DIR/package.json" ]; then
  echo "ERROR: medstats-ui not found at ../medstats-ui"
  echo "Clone it next to medstats-api:"
  echo "  parent/"
  echo "    medstats-api/"
  echo "    medstats-ui/"
  exit 1
fi

BUILD_FLAG=""
LOGS=false
for arg in "$@"; do
  case "$arg" in
    --build) BUILD_FLAG="--build" ;;
    --logs) LOGS=true ;;
  esac
done

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | grep OPENAI_API_KEY | xargs) 2>/dev/null || true
fi

echo "=== MedStats Full Stack (Docker) ==="
echo ""
docker compose up -d $BUILD_FLAG db redis api worker ui

echo ""
echo "Services:"
echo "  UI:        http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "  API docs:  http://localhost:8000/docs"
echo "  Postgres:  localhost:5433 (user/pass/db: medstats)"
echo "  Redis:     localhost:6380"
echo ""
echo "Demo accounts (seeded on first API start):"
echo "  Professor: prof@medstats.io / admin123"
echo "  Student:   student@medstats.io / admin123"
echo ""
echo "Stop:  docker compose down"
echo "Logs:  docker compose logs -f api ui"

if [ "$LOGS" = true ]; then
  docker compose logs -f api ui
fi
