#!/usr/bin/env bash
# Deploy local beta: run quality gates → build → start stack → wait for full readiness → smoke test.
#
# Usage:
#   ./scripts/beta_deploy.sh              # standard deploy
#   ./scripts/beta_deploy.sh --full       # include scenario regressions
#   ./scripts/beta_deploy.sh --no-gate    # skip pre-deploy checks (emergency only)
#   ./scripts/beta_deploy.sh --no-wait    # skip /readyz polling (just /health liveness)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$API_DIR"

FULL=false
NO_GATE=false
NO_WAIT=false
for arg in "$@"; do
  case "$arg" in
    --full)    FULL=true ;;
    --no-gate) NO_GATE=true ;;
    --no-wait) NO_WAIT=true ;;
  esac
done

if [ ! -f ".env.beta" ]; then
  echo "ERROR: .env.beta not found. Copy from template:"
  echo "  cp .env.beta.example .env.beta"
  echo "  # then fill in POSTGRES_PASSWORD, JWT_SECRET, SMTP_*, OPENAI_API_KEY"
  exit 1
fi

BETA_HTTP_PORT=$(grep -E '^BETA_HTTP_PORT=' .env.beta 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" || echo "3000")
export BETA_HTTP_PORT="${BETA_HTTP_PORT:-3000}"
BASE_URL="http://localhost:${BETA_HTTP_PORT}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== MedStats Beta Deploy ==="

if [ "$NO_GATE" = false ]; then
  GATE_ARGS=""
  [ "$FULL" = true ] && GATE_ARGS="--full"
  "$SCRIPT_DIR/pre_deploy_check.sh" $GATE_ARGS
else
  echo -e "${YELLOW}WARNING: Skipping pre-deploy gate (--no-gate)${NC}"
fi

echo ""
echo "Building and starting beta stack..."
docker compose -f docker-compose.beta.yml --env-file .env.beta up -d --build --force-recreate

# Restart nginx so it re-resolves upstream hostnames after container rebuilds
docker compose -f docker-compose.beta.yml restart nginx >/dev/null 2>&1 || true

# ── Phase 1: liveness (API process up) ──────────────────────────────────────
echo ""
echo "Phase 1/2  Waiting for API liveness (/health)..."
LIVENESS_TRIES=0
until curl -sf "${BASE_URL}/health" >/dev/null 2>&1; do
  LIVENESS_TRIES=$((LIVENESS_TRIES+1))
  if [ "$LIVENESS_TRIES" -ge 60 ]; then
    echo -e "${RED}✗ API did not become live after 120s — check logs:${NC}"
    echo "  docker compose -f docker-compose.beta.yml logs api"
    exit 1
  fi
  printf "."
  sleep 2
done
echo ""
echo -e "${GREEN}✓ API is live${NC}"

if [ "$NO_WAIT" = true ]; then
  echo -e "${YELLOW}–  Skipping readiness wait (--no-wait)${NC}"
else
  # ── Phase 2: readiness (corpus indexed, users seeded, Redis live) ──────────
  echo ""
  echo "Phase 2/2  Waiting for full readiness (/readyz) — corpus indexing in progress..."
  echo "           (This may take 2-5 minutes while PubMed abstracts are indexed.)"
  echo ""

  READY_TRIES=0
  MAX_READY_TRIES=150   # 150 × 4s = 10 minutes max
  LAST_STATUS=""
  while true; do
    READY_TRIES=$((READY_TRIES+1))
    if [ "$READY_TRIES" -ge "$MAX_READY_TRIES" ]; then
      echo ""
      echo -e "${RED}✗ Stack not fully ready after 10 minutes.${NC}"
      echo "  Check API logs: docker compose -f docker-compose.beta.yml logs api worker"
      echo "  Current /readyz:"
      curl -s "${BASE_URL}/readyz" | python3 -m json.tool 2>/dev/null || true
      echo ""
      echo -e "${YELLOW}Continuing anyway — some features may be degraded.${NC}"
      break
    fi

    RESP=$(curl -sf "${BASE_URL}/readyz" 2>/dev/null || echo '{}')
    READY=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ready','false'))" 2>/dev/null || echo "false")

    # Build a one-line status string
    RAG=$(echo "$RESP"   | python3 -c "import sys,json; d=json.load(sys.stdin); c=d.get('corpus',{}); print(f\"rag={c.get('rag_docs',0)}, lit={c.get('literature_docs',0)}, nodes={c.get('knowledge_nodes',0)}\")" 2>/dev/null || echo "")
    DB=$(echo "$RESP"    | python3 -c "import sys,json; d=json.load(sys.stdin); print('db=✓' if d.get('db',{}).get('ok') else 'db=✗')" 2>/dev/null || echo "")
    REDIS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('redis=✓' if d.get('redis',{}).get('ok') else 'redis=✗')" 2>/dev/null || echo "")
    USERS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('users=✓' if d.get('users_seeded') else 'users=✗')" 2>/dev/null || echo "")
    STATUS="  [${DB} ${REDIS} ${USERS} ${RAG}]"

    if [ "$STATUS" != "$LAST_STATUS" ]; then
      echo -e "${YELLOW}${STATUS}${NC}"
      LAST_STATUS="$STATUS"
    else
      printf "."
    fi

    if [ "$READY" = "True" ] || [ "$READY" = "true" ]; then
      echo ""
      echo -e "${GREEN}✓ Stack is fully ready${NC}"
      break
    fi

    sleep 4
  done
fi

# ── Smoke tests ────────────────────────────────────────────────────────────
echo ""
BETA_URL="${BASE_URL}" "$SCRIPT_DIR/smoke_test.sh"

# ── Seed-and-test: upload reference masterchart, run full pipeline, then test ─
# This single script replaces the old piecemeal RAG + professor-review checks.
# It seeds a real submission, runs all post-deploy verifications against it,
# then cleans up the seeded data.
echo ""
echo -e "${YELLOW}▶ Post-deploy integration tests (seed-and-test)${NC}"
BASE_URL="${BASE_URL}" "$SCRIPT_DIR/seed_and_test.sh" \
  && echo -e "${GREEN}✓ All seed-and-test checks passed${NC}" \
  || echo -e "${YELLOW}⚠ seed-and-test had failures — see output above${NC}"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} Beta is live at: ${BASE_URL}${NC}"
echo -e "${GREEN}============================================${NC}"
echo "  Logs:  docker compose -f docker-compose.beta.yml logs -f api worker"
echo "  Watch: ./scripts/watch_stack.sh --daemon"

