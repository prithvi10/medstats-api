#!/usr/bin/env bash
# Monitor beta stack health and error logs; alert on failures.
#
# Usage:
#   ./scripts/watch_stack.sh              # single check
#   ./scripts/watch_stack.sh --daemon   # loop every 5 min (for cron/launchd)
#   ./scripts/watch_stack.sh --since 10m  # look back 10 min in logs
#
# Cron example (every 5 min):
#   */5 * * * * cd /path/to/medstats-api && ./scripts/watch_stack.sh >> logs/watch.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$API_DIR"

COMPOSE_FILE="docker-compose.beta.yml"
ENV_FILE=".env.beta"
BETA_URL="${BETA_URL:-http://localhost:3000}"
SINCE="${SINCE:-5m}"
DAEMON=false
INTERVAL=300

for arg in "$@"; do
  case "$arg" in
    --daemon) DAEMON=true ;;
    --since=*) SINCE="${arg#*=}" ;;
    --since) shift; SINCE="${1:-5m}" ;;
    --interval=*) INTERVAL="${arg#*=}" ;;
  esac
done

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

pick_python() {
  if [ -f "$API_DIR/.venv/bin/python" ]; then
    echo "$API_DIR/.venv/bin/python"
  else
    echo python3
  fi
}
PYTHON=$(pick_python)

alert() {
  local subject="$1"
  local body="$2"
  echo "[$(date -Iseconds)] ALERT: $subject"
  echo "$body"
  $PYTHON "$SCRIPT_DIR/alert_notify.py" --subject "$subject" --body "$body" 2>/dev/null \
    || echo "  (alert delivery failed — check SMTP/ALERT_EMAIL_TO)"
}

run_check() {
  local issues=()

  # 1. API health
  if ! curl -sf "${BETA_URL}/health" >/dev/null 2>&1; then
    issues+=("API health check failed at ${BETA_URL}/health")
  fi

  # 2. Docker container status
  if command -v docker &>/dev/null; then
    for svc in api worker db redis nginx; do
      status=$(docker compose -f "$COMPOSE_FILE" ps "$svc" --format "{{.Status}}" 2>/dev/null || echo "missing")
      if echo "$status" | grep -qiE "exit|unhealthy|restarting"; then
        issues+=("Container $svc: $status")
      fi
    done
  fi

  # 3. Error patterns in recent logs
  if command -v docker &>/dev/null; then
    LOG_HITS=$(docker compose -f "$COMPOSE_FILE" logs --since="$SINCE" api worker 2>/dev/null \
      | grep -iE "error|critical|traceback|exception" \
      | grep -viE "error_log|no error|0 error" \
      | tail -20 || true)
    if [ -n "$LOG_HITS" ]; then
      issues+=("Recent errors in api/worker logs (last $SINCE):\n$LOG_HITS")
    fi
  fi

  # 4. Stuck analysis jobs (optional DB check via API admin health — skip if no auth)
  # Left for future: query failed job count via admin token

  if [ ${#issues[@]} -gt 0 ]; then
    body=$(printf '%s\n\n' "${issues[@]}")
    alert "Beta stack issue detected" "$body"
    return 1
  fi

  echo "[$(date -Iseconds)] OK — all checks passed"
  return 0
}

if [ "$DAEMON" = true ]; then
  echo "Watching beta stack every ${INTERVAL}s (Ctrl+C to stop)"
  while true; do
    run_check || true
    sleep "$INTERVAL"
  done
else
  run_check
fi
