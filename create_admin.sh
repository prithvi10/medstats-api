#!/bin/bash
# Quick script to create admin user in running container

set -e

echo "Creating admin user in MedStats..."

# Check if using Colima
if [ -S "$HOME/.colima/default/docker.sock" ]; then
    export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"
fi

# Run seed script in API container
docker compose -f docker-compose.beta.yml exec api python scripts/seed_admin.py

echo ""
echo "✓ Done! You can now login at http://localhost:3000/admin"
echo ""
echo "Credentials:"
echo "  Email:    admin@medstats.io"
echo "  Password: MedstatsAdmin@10"
