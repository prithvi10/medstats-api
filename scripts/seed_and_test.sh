#!/usr/bin/env bash
# seed_and_test.sh
#
# Self-contained post-deploy integration test that:
#   1. Uploads the reference masterchart (13_medicine5_reference) as a student
#   2. Triggers the full analysis pipeline and waits for completion
#   3. Runs all post-deploy checks against the resulting report:
#      a. RAG /explain smoke test (Why this test?)
#      b. Related literature RAG smoke test
#      c. Professor review end-to-end test (submit → review → revision → LLM judge)
#   4. Cleans up the seeded submission from the DB
#
# Usage:
#   ./scripts/seed_and_test.sh
#   BASE_URL=http://localhost:3000 ./scripts/seed_and_test.sh
#   SKIP_CLEANUP=1 ./scripts/seed_and_test.sh    # keep seeded data for manual inspection
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ─── Config ────────────────────────────────────────────────────────────────
BASE_URL="${BASE_URL:-http://localhost:3000}"
API_URL="$BASE_URL"
STUDENT_EMAIL="${STUDENT_EMAIL:-student@medstats.io}"
STUDENT_PASS="${STUDENT_PASS:-admin123}"
PROFESSOR_EMAIL="${PROFESSOR_EMAIL:-prof@medstats.io}"
PROFESSOR_PASS="${PROFESSOR_PASS:-admin123}"
ANALYSIS_TIMEOUT="${ANALYSIS_TIMEOUT:-600}"    # seconds to wait for analysis
REVISION_TIMEOUT="${REVISION_TIMEOUT:-600}"    # seconds to wait for revision
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
POLL_INTERVAL=6

# The reference scenario: medicine5 has xlsx + intent.docx + clear fracture variables
FIXTURE_DIR="tests/fixtures/scenarios/13_medicine5_reference"
FIXTURE_XLSX="$FIXTURE_DIR/medicine5_data.xlsx"
SUBMISSION_TITLE="[AUTOTEST] Medicine5 Reference — $(date +%Y%m%dT%H%M%S)"

# ─── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'

log()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
pass()   { echo -e "${GREEN}[PASS]${NC}  $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail_clean() {
  echo -e "${RED}[FAIL]${NC}  $*"
  _cleanup
  exit 1
}

# Track created resource IDs so we can clean them up
SEEDED_SUBMISSION_ID=""
SEEDED_JOB_ID=""

# ─── Cleanup ────────────────────────────────────────────────────────────────
_cleanup() {
  if [ "$SKIP_CLEANUP" = "1" ] || [ "$SKIP_CLEANUP" = "true" ]; then
    warn "SKIP_CLEANUP set — leaving seeded submission $SEEDED_SUBMISSION_ID in DB"
    return
  fi
  if [ -z "$SEEDED_SUBMISSION_ID" ]; then return; fi

  log "Cleaning up seeded submission $SEEDED_SUBMISSION_ID..."
  # Use docker exec into the API container to delete directly (bypasses auth concerns)
  API_CONTAINER=$(docker ps --format "{{.Names}}" 2>/dev/null \
    | grep -E "medstats.*-api-[0-9]" | grep -v worker | head -1 || echo "")
  if [ -n "$API_CONTAINER" ]; then
    cat << 'PYEOF' | docker exec -i -w /app -e "SUB_ID=$SEEDED_SUBMISSION_ID" "$API_CONTAINER" python3 - 2>/dev/null && \
      log "Seeded submission deleted" || warn "Could not delete seeded submission (manual cleanup may be needed)"
import os, asyncio, logging
logging.disable(logging.CRITICAL)
from sqlalchemy import delete, select
from src.medstats.db.models import (
    Submission, SubmissionFile, AnalysisJob, Report,
    ProfessorReview, AgentCorrection, Notification,
)
from src.shared.db.session import async_session_factory
from uuid import UUID

sub_id = UUID(os.environ["SUB_ID"])

async def cleanup():
    async with async_session_factory() as session:
        async with session.begin():
            # Get all job/report IDs
            job_ids = (await session.execute(
                select(AnalysisJob.id).where(AnalysisJob.submission_id == sub_id)
            )).scalars().all()
            report_ids = []
            if job_ids:
                report_ids = (await session.execute(
                    select(Report.id).where(Report.job_id.in_(job_ids))
                )).scalars().all()

            # Delete agent_corrections (FK → professor_reviews → reports)
            if report_ids:
                review_ids = (await session.execute(
                    select(ProfessorReview.id).where(ProfessorReview.report_id.in_(report_ids))
                )).scalars().all()
                if review_ids:
                    await session.execute(
                        delete(AgentCorrection).where(AgentCorrection.review_id.in_(review_ids))
                    )
                await session.execute(
                    delete(ProfessorReview).where(ProfessorReview.report_id.in_(report_ids))
                )
                await session.execute(
                    delete(Report).where(Report.id.in_(report_ids))
                )

            # Delete notifications
            await session.execute(delete(Notification).where(Notification.submission_id == sub_id))
            # Delete jobs
            await session.execute(delete(AnalysisJob).where(AnalysisJob.submission_id == sub_id))
            # Delete submission files
            await session.execute(delete(SubmissionFile).where(SubmissionFile.submission_id == sub_id))
            # Delete submission
            await session.execute(delete(Submission).where(Submission.id == sub_id))

try:
    from src.shared.db.session import engine
    engine.sync_engine.dispose()
except Exception:
    pass
asyncio.run(cleanup())
print("ok")
PYEOF
  else
    warn "No API container found — seeded submission $SEEDED_SUBMISSION_ID was not deleted"
  fi
}

# Always clean up on exit (unless SKIP_CLEANUP)
trap '_cleanup' EXIT

# ─── Helpers ────────────────────────────────────────────────────────────────
api_get() {
  local token="$1"; local path="$2"
  curl -sf -H "Authorization: Bearer $token" "$API_URL$path"
}

api_post() {
  local token="$1"; local path="$2"; local body="$3"
  curl -sf -X POST -H "Authorization: Bearer $token" \
    -H "Content-Type: application/json" -d "$body" "$API_URL$path"
}

login() {
  local email="$1"; local pass="$2"
  local resp
  resp=$(curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"email\":\"$email\",\"password\":\"$pass\"}" \
    "$API_URL/api/auth/login")
  echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null \
    || { echo "  Login response: $resp" >&2; return 1; }
}

jq_str() {
  # jq_str <json_string> <python_key_expr>
  echo "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d$2)" 2>/dev/null || echo ""
}

poll_job() {
  # poll_job <token> <submission_id> <job_id> <timeout> <poll_label>
  local token="$1" sub_id="$2" job_id="$3" timeout_s="$4" label="${5:-job}"
  local deadline=$(( $(date +%s) + timeout_s ))
  local status="queued"

  while true; do
    if (( $(date +%s) >= deadline )); then
      echo ""
      fail_clean "$label timed out after ${timeout_s}s. Last status: $status"
    fi
    local resp
    resp=$(api_get "$token" "/api/analysis/submissions/$sub_id/job" 2>/dev/null) || {
      sleep "$POLL_INTERVAL"; continue
    }
    status=$(jq_str "$resp" "['status']")
    local pct
    pct=$(jq_str "$resp" ".get('progress',0)")
    echo -ne "\r  $label: $status ($pct%)    "

    case "$status" in
      completed) echo ""; return 0 ;;
      failed)
        echo ""
        local err; err=$(jq_str "$resp" ".get('error_log','')")
        fail_clean "$label FAILED: $err" ;;
    esac
    sleep "$POLL_INTERVAL"
  done
}

# ─── Banner ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  MedStats Seed-and-Test (full pipeline integration)${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ─── 0. Pre-flight ───────────────────────────────────────────────────────────
log "Checking stack health..."
HEALTH=$(curl -sf "$API_URL/health" 2>/dev/null || echo "")
if ! echo "$HEALTH" | grep -q "ok"; then
  fail_clean "Stack is not healthy (GET /health). Run beta_deploy.sh first."
fi
pass "Stack is healthy"

if [ ! -f "$FIXTURE_XLSX" ]; then
  fail_clean "Fixture not found: $FIXTURE_XLSX"
fi
pass "Fixture found: $FIXTURE_XLSX"

# ─── 1. Authenticate ─────────────────────────────────────────────────────────
log "Authenticating as student ($STUDENT_EMAIL)..."
STUDENT_TOKEN=$(login "$STUDENT_EMAIL" "$STUDENT_PASS") \
  || fail_clean "Student login failed. Is SEED_DEMO_USERS=true?"
pass "Student authenticated"

log "Authenticating as professor ($PROFESSOR_EMAIL)..."
PROF_TOKEN=$(login "$PROFESSOR_EMAIL" "$PROFESSOR_PASS") \
  || fail_clean "Professor login failed"
pass "Professor authenticated"

# ─── 2. Upload fixture submission ────────────────────────────────────────────
log "Uploading reference masterchart fixture..."
INTENT="Prospective cohort study of paediatric long-bone fracture management. Variables include bone involved, fracture type, mechanism of injury, side, sex, and post-operative outcomes."
UPLOAD_RESP=$(curl -sf -X POST \
  -H "Authorization: Bearer $STUDENT_TOKEN" \
  -F "title=$SUBMISSION_TITLE" \
  -F "intent_description=$INTENT" \
  -F "files=@$FIXTURE_XLSX" \
  "$API_URL/api/student/submissions") \
  || fail_clean "Failed to upload submission"

SEEDED_SUBMISSION_ID=$(jq_str "$UPLOAD_RESP" "['id']")
if [ -z "$SEEDED_SUBMISSION_ID" ]; then
  fail_clean "Upload response missing 'id'. Response: $UPLOAD_RESP"
fi
pass "Submission created: $SEEDED_SUBMISSION_ID"

# ─── 3. Trigger analysis pipeline ─────────────────────────────────────────────
log "Triggering analysis pipeline..."
RUN_RESP=$(api_post "$STUDENT_TOKEN" "/api/analysis/submissions/$SEEDED_SUBMISSION_ID/run" '{}') \
  || fail_clean "Failed to trigger analysis"
SEEDED_JOB_ID=$(jq_str "$RUN_RESP" "['job_id']")
pass "Analysis job queued: $SEEDED_JOB_ID"

# ─── 4. Wait for analysis to complete ─────────────────────────────────────────
log "Waiting for analysis pipeline to complete (timeout: ${ANALYSIS_TIMEOUT}s)..."
poll_job "$STUDENT_TOKEN" "$SEEDED_SUBMISSION_ID" "$SEEDED_JOB_ID" "$ANALYSIS_TIMEOUT" "Analysis"
pass "Analysis pipeline completed"

# Verify the report exists
REPORT=$(api_get "$STUDENT_TOKEN" "/api/student/submissions/$SEEDED_SUBMISSION_ID/report") \
  || fail_clean "Report endpoint returned error after analysis completion"
SECTION_COUNT=$(jq_str "$REPORT" " .get('sections',[])" | python3 -c "import sys; print(len(eval(sys.stdin.read())))" 2>/dev/null || echo "0")
pass "Report has $SECTION_COUNT sections"

echo ""
echo -e "${BOLD}━━━  Phase 1 complete: submission seeded and analysis done  ━━━${NC}"
echo ""

# ─── 5. RAG /explain smoke test ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}▶ Phase 2a: RAG /explain (Why this test?) checks${NC}"
BASE_URL="$BASE_URL" SUBMISSION_ID_OVERRIDE="$SEEDED_SUBMISSION_ID" \
  "$SCRIPT_DIR/test_rag_explain.sh" \
  && pass "RAG /explain checks passed" \
  || warn "RAG /explain checks had failures"

# ─── 6. Related literature RAG smoke test ─────────────────────────────────────
echo ""
echo -e "${BOLD}▶ Phase 2b: Related Literature RAG checks${NC}"
BASE_URL="$BASE_URL" "$SCRIPT_DIR/test_literature_rag.sh" \
  && pass "Literature RAG checks passed" \
  || warn "Literature RAG checks had failures"

# ─── 7. Professor review end-to-end ───────────────────────────────────────────
echo ""
echo -e "${BOLD}▶ Phase 2c: Professor Review end-to-end test${NC}"
BASE_URL="$BASE_URL" \
  REVISION_TIMEOUT="$REVISION_TIMEOUT" \
  "$SCRIPT_DIR/test_professor_review.sh" --submission-id "$SEEDED_SUBMISSION_ID" \
  && pass "Professor review test passed" \
  || warn "Professor review test had failures"

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Seed-and-test complete.${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Cleanup is handled by the EXIT trap
