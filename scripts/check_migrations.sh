#!/usr/bin/env bash
# Apply Alembic migrations on a throwaway Postgres container and check for drift.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$API_DIR"

if [ -f "$API_DIR/.venv/bin/python" ]; then
  PYTHON="$API_DIR/.venv/bin/python"
else
  PYTHON=python3
fi

CONTAINER="medstats-migration-check-$$"
NETWORK="medstats-mig-net-$$"
PG_PORT=55432

cleanup() {
  docker rm -f "$CONTAINER" 2>/dev/null || true
  docker network rm "$NETWORK" 2>/dev/null || true
}
trap cleanup EXIT

docker network create "$NETWORK" >/dev/null
docker run -d --name "$CONTAINER" --network "$NETWORK" \
  -e POSTGRES_USER=ci -e POSTGRES_PASSWORD=ci -e POSTGRES_DB=ci_test \
  -p "${PG_PORT}:5432" \
  pgvector/pgvector:pg16 >/dev/null

echo "Waiting for Postgres..."
for i in $(seq 1 30); do
  if docker exec "$CONTAINER" pg_isready -U ci -q 2>/dev/null; then
    # Also verify the host-side port is actually open before proceeding
    if (echo >/dev/tcp/localhost/${PG_PORT}) 2>/dev/null; then
      break
    fi
  fi
  sleep 1
done

# One extra second to avoid rare race between port open and pg accepting conns
sleep 1

docker exec "$CONTAINER" psql -U ci -d ci_test -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

export DATABASE_URL="postgresql+asyncpg://ci:ci@localhost:${PG_PORT}/ci_test"
export DATABASE_URL_SYNC="postgresql://ci:ci@localhost:${PG_PORT}/ci_test"
export JWT_SECRET=ci-secret
export ENV=test

$PYTHON -m alembic upgrade head
$PYTHON -m alembic check

echo "Migration check passed"
