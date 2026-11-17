#!/usr/bin/env bash
# Complete deploy wrapper - ensures Docker is running, then deploys
#
# Usage:
#   ./scripts/deploy.sh              # standard deploy
#   ./scripts/deploy.sh --full       # include scenario regressions
#   ./scripts/deploy.sh --no-gate    # skip pre-deploy checks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== MedStats Complete Deploy ===${NC}"
echo ""

# Check if Colima is installed
if ! command -v colima &> /dev/null; then
    echo -e "${RED}✗ Colima not found. Install with: brew install colima${NC}"
    exit 1
fi

# Check Colima status first
COLIMA_STATUS=$(colima status 2>&1 || echo "stopped")
if [[ "$COLIMA_STATUS" != *"Running"* ]]; then
    echo -e "${YELLOW}⚠ Colima is not running. Starting...${NC}"
    colima start --cpu 4 --memory 8 --disk 60

    # Give it extra time to fully start
    echo "Waiting for Colima to fully initialize..."
    sleep 5
else
    echo -e "${GREEN}✓ Colima is running${NC}"
fi

# Wait for Docker socket to be available
echo "Checking Docker socket..."
for i in {1..60}; do
    if [ -S "$HOME/.colima/default/docker.sock" ]; then
        echo -e "${GREEN}✓ Docker socket found${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${RED}✗ Docker socket not available after 2 minutes${NC}"
        echo "Try manually: colima start"
        exit 1
    fi
    printf "."
    sleep 2
done

# Set Docker context to Colima
export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"

# Verify Docker is responsive
echo "Testing Docker connectivity..."
for i in {1..30}; do
    if docker info &> /dev/null; then
        echo -e "${GREEN}✓ Docker is ready and responsive${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Docker not responding${NC}"
        echo "Debug info:"
        echo "  Socket: $DOCKER_HOST"
        echo "  Colima status:"
        colima status
        exit 1
    fi
    printf "."
    sleep 2
done

echo ""
echo -e "${GREEN}✓ Environment ready. Starting deployment...${NC}"
echo ""

# Run the actual deploy script
cd "$API_DIR"
exec "$SCRIPT_DIR/deploy-local.sh" "$@"
