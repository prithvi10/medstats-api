#!/usr/bin/env bash
# Post-deploy smoke tests — run after beta stack is up.
#
# Usage:
#   ./scripts/smoke_test.sh
#   BETA_URL=http://localhost:3000 ./scripts/smoke_test.sh

set -euo pipefail

BETA_URL="${BETA_URL:-http://localhost:3000}"
API_URL="${API_URL:-$BETA_URL}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
  local name="$1"
  local url="$2"
  local expected="${3:-ok}"
  echo -n "  $name ... "
  body=$(curl -sf "$url" 2>/dev/null) || { echo -e "${RED}FAIL (unreachable)${NC}"; return 1; }
  if echo "$body" | grep -q "$expected"; then
    echo -e "${GREEN}OK${NC}"
  else
    echo -e "${RED}FAIL (unexpected response)${NC}"
    echo "    $body"
    return 1
  fi
}

echo "Smoke tests against $BETA_URL"
FAILED=0

check "API health" "$API_URL/health" "ok" || FAILED=1
check "UI reachable" "$BETA_URL/" "<!DOCTYPE html>" || FAILED=1
check "API docs" "$API_URL/api/docs" "swagger" || FAILED=1

# Docker container health
if command -v docker &>/dev/null; then
  echo -n "  Docker services ... "
  UNHEALTHY=$(docker compose -f docker-compose.beta.yml ps --format json 2>/dev/null \
    | python3 -c "
import sys, json
bad = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        s = json.loads(line)
        if s.get('Health') and s['Health'] != 'healthy':
            bad.append(s.get('Service','?'))
    except: pass
print(','.join(bad))
" 2>/dev/null || echo "")
  if [ -n "$UNHEALTHY" ]; then
    echo -e "${RED}FAIL (unhealthy: $UNHEALTHY)${NC}"
    FAILED=1
  else
    echo -e "${GREEN}OK${NC}"
  fi
fi

if [ "$FAILED" -ne 0 ]; then
  echo -e "\n${RED}Smoke tests FAILED${NC}"
  exit 1
fi
echo -e "\n${GREEN}All smoke tests passed${NC}"
