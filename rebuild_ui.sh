#!/bin/bash
set -e

echo "=== Forcing UI rebuild ==="

# Stop UI container
docker compose -f docker-compose.beta.yml stop ui

# Remove UI container and image
docker compose -f docker-compose.beta.yml rm -f ui
docker rmi medstats-api-ui 2>/dev/null || true

# Rebuild UI with no cache
docker compose -f docker-compose.beta.yml build --no-cache ui

# Start everything
docker compose -f docker-compose.beta.yml up -d

echo "=== UI rebuilt and restarted ==="
echo "Check: curl -I http://localhost:3000/"
