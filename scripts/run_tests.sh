#!/bin/bash
# Run all tests (unit + integration) with coverage report.
#
# Usage: ./scripts/run_tests.sh [unit|integration|all]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"
source .venv/bin/activate

TEST_SCOPE="${1:-all}"

echo "=== MedStats Test Runner ==="
echo "Scope: $TEST_SCOPE"
echo ""

case "$TEST_SCOPE" in
    unit)
        python -m pytest tests/unit/ -v --cov=src --cov-report=term-missing
        ;;
    integration)
        python -m pytest tests/integration/ -v -s --cov=src --cov-report=term-missing
        ;;
    scenarios)
        python scripts/generate_scenario_fixtures.py
        python -m pytest tests/scenarios/ -v -s
        ;;
    all)
        python scripts/generate_scenario_fixtures.py
        python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
        echo ""
        echo "HTML coverage report: htmlcov/index.html"
        ;;
    *)
        echo "Usage: $0 [unit|integration|scenarios|all]"
        exit 1
        ;;
esac
