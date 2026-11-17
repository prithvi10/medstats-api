#!/usr/bin/env bash
# Single command to deploy MedStats locally (beta).
#
# First time only:
#   cp .env.beta.example .env.beta
#   # edit .env.beta — POSTGRES_PASSWORD, JWT_SECRET, OPENAI_API_KEY, SMTP_*
#
# Deploy:
#   ./scripts/deploy-local.sh
#
# Options passed through:
#   ./scripts/deploy-local.sh --full      # include scenario regressions
#   ./scripts/deploy-local.sh --no-gate   # emergency skip (not recommended)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/beta_deploy.sh" "$@"
