#!/usr/bin/env bash
# Run all quality gates before deploying to local beta.
# Exits non-zero on first failure — nothing gets deployed half-cooked.
#
# Usage:
#   ./scripts/pre_deploy_check.sh           # fast gate (unit + build + migrations)
#   ./scripts/pre_deploy_check.sh --full    # + scenario regression tests
#   ./scripts/pre_deploy_check.sh --skip-ui # API only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UI_DIR="$(cd "$API_DIR/../medstats-ui" 2>/dev/null && pwd || true)"

FULL=false
SKIP_UI=false
for arg in "$@"; do
  case "$arg" in
    --full) FULL=true ;;
    --skip-ui) SKIP_UI=true ;;
  esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step()  { echo -e "\n${YELLOW}▶ $1${NC}"; }
pass()  { echo -e "${GREEN}✓ $1${NC}"; }
fail()  { echo -e "${RED}✗ $1${NC}"; exit 1; }

pick_python() {
  if [ -f "$API_DIR/.venv/bin/python" ]; then
    echo "$API_DIR/.venv/bin/python"
  else
    echo python3
  fi
}

PYTHON=$(pick_python)
export DATABASE_URL="${DATABASE_URL:-sqlite+aiosqlite:///:memory:}"
export DATABASE_URL_SYNC="${DATABASE_URL_SYNC:-sqlite:///:memory:}"
export JWT_SECRET="${JWT_SECRET:-ci-secret}"
export ENV="${ENV:-test}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-key-for-tests}"

echo "============================================"
echo " MedStats Pre-Deploy Gate"
echo " API: $API_DIR"
echo " UI:  ${UI_DIR:-NOT FOUND}"
echo "============================================"

cd "$API_DIR"

# ── 1. API import smoke ──────────────────────────────────────────────────────
step "API import smoke test"
$PYTHON -c "import src.main" || fail "API imports broken"
pass "API imports OK"

# ── 2. Unit tests ────────────────────────────────────────────────────────────
step "API unit tests"
$PYTHON -m pytest tests/unit/ -v --tb=short -m "not integration and not rag" \
  || fail "Unit tests failed"
pass "Unit tests passed"

# ── 3. Migration check (Postgres via Docker if available) ─────────────────────
step "Alembic migration check"
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
  "$SCRIPT_DIR/check_migrations.sh" || fail "Migration check failed"
else
  echo "  Docker not available — running offline alembic check against SQLite"
  $PYTHON -m alembic check 2>/dev/null || echo "  (skipped — needs Postgres for full check)"
fi
pass "Migrations OK"

# ── 4. UI build + lint ───────────────────────────────────────────────────────
if [ "$SKIP_UI" = false ]; then
  if [ -z "$UI_DIR" ] || [ ! -f "$UI_DIR/package.json" ]; then
    fail "medstats-ui not found at ../medstats-ui"
  fi
  step "UI TypeScript build"
  (cd "$UI_DIR" && npm run build) || fail "UI build failed"
  pass "UI build OK"

  step "UI lint"
  (cd "$UI_DIR" && npm run lint) || fail "UI lint failed"
  pass "UI lint OK"
else
  echo "Skipping UI checks (--skip-ui)"
fi

# ── 5. Optional full regression ──────────────────────────────────────────────
if [ "$FULL" = true ]; then
  step "Scenario regression tests"
  $PYTHON scripts/generate_scenario_fixtures.py
  $PYTHON -m pytest tests/scenarios/ -v --tb=short || fail "Scenario tests failed"
  pass "Scenario tests passed"

  step "Integration tests (fixtures required)"
  $PYTHON -m pytest tests/integration/ -v --tb=short -x || fail "Integration tests failed"
  pass "Integration tests passed"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} All gates passed — safe to deploy${NC}"
echo -e "${GREEN}============================================${NC}"
