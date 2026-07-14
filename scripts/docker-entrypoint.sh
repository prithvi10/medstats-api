#!/bin/bash
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Ensuring database tables exist..."
python -c "
import asyncio
from src.medstats.db import models  # noqa: F401
from src.shared.db.session import init_db
asyncio.run(init_db())
"

if [ "${ENV:-development}" != "production" ]; then
  echo "Seeding demo users (if needed)..."
  python scripts/seed_demo_users.py || true
fi

exec "$@"
