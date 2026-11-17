# MedStats Reference Suite — Milestones (Phase 0 → Production)

Track progress against the 89 historical thesis input/output pairs under `data/Thesis Project/data/`.

## Success metrics

| Metric | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Prod |
|---|---|---|---|---|---|---|
| Pipeline completes (no crash) | 89/89 | 89/89 | 89/89 | 89/89 | 89/89 | 89/89 |
| Tier 1 p-value match | baseline | ↑ | 40–60% | 60–75% | 75%+ | 75%+ |
| Tier 2 style score ≥ 0.75 | baseline | — | — | — | 50%+ | 75%+ |
| New masterchart (no reference) | — | — | — | retrieval works | styled output | reviewed |

## Phase 0 — Stabilize regression harness (Week 1)

**Goal:** Clean signal; zero hard FAILED from infra bugs.

- [x] Masterchart header detection + generic sheet handling (`masterchart_normalizer.py`)
- [x] Safe numeric coercion in TestSelector + StatsEngine
- [ ] Robust p-value / mean extraction (malformed reference typos: `0..3369`, `0.0.000`)
- [ ] Dedupe columns before all `to_json()` / side-by-side DOCX table merges
- [ ] `.xls` masterchart support (`xlrd`)
- [ ] Integration test: evaluator never crashes on reference text
- [ ] CI gate: `pipeline status != FAILED` for all 89 cases

**Verify:** `MEDSTATS_REFERENCE_SUITE=all pytest tests/integration/test_reference_report_suite.py -q`  
**Target:** 0 FAILED, 89 xfailed (Tier 1) acceptable.

---

## Phase 1 — Parser completeness (Week 2)

**Goal:** Correct dataframe for every masterchart layout.

- [ ] Multi-input stitch: `Input Group A.xlsx` + `Input Group B.xlsx` → unified table + `__group__`
- [ ] Group labels from filenames / title rows (not `Sheet1`/`Sheet2`)
- [ ] Transposed / wide layouts → standard columnar form
- [ ] Multi-output concat for seed/eval/regression
- [ ] Age strings: `19y`, `50Y`, `60YR` → numeric
- [ ] Golden unit tests per layout class

**Verify:** Re-run full suite; Tier 1 `generated=0 p-values` cases drop sharply.

---

## Phase 2 — Statistical alignment (Weeks 3–5)

**Goal:** Match father's numbers on historical cases.

- [ ] Per-thesis expected manifest: `{variable, test, statistic, p_value}` from reference output
- [ ] Variable selection: test only clinically relevant columns (not every parsed column)
- [ ] Test parity: Fisher vs chi-square, paired vs unpaired, exact tests for small n
- [ ] Study design from content (title rows, group columns) not sheet count
- [ ] Tier 1 comparison keyed by variable (not global p-value list order)

**Verify:** Tier 1 pass rate ≥ 40% on 89 cases, trending up per commit.

---

## Phase 3 — Per-thesis knowledge ingest (Week 5–6)

**Goal:** Professor-like guidance on **new** mastercharts without overriding StatsEngine.

- [ ] `scripts/ingest_thesis_pairs.py` — 89 separate LLM calls, full input + full output (no truncation)
- [ ] `reference_case` knowledge node per thesis + embedding
- [ ] Aggregated `ProfessorStyleProfile` (formatting only, not test-selection rules)
- [ ] Re-seed `reference_reports.pair_analysis` JSON

**Verify:** 89 nodes in DB; spot-check retrieval for 5 new-style queries.

---

## Phase 4 — Report generator + style (Week 6–7)

**Goal:** Tier 2 passes when Tier 1 passes.

- [ ] Load `ProfessorStyleProfile` in ReportGenerator at runtime
- [ ] Professor narrative templates before LLM fallback
- [ ] SYSTAT boilerplate + p-value format from profile
- [ ] Retrieve top-3 similar `reference_case` nodes for narrative prompts
- [ ] Table layout: title above, interpretation below

**Verify:** Tier 2 ≥ 0.75 on cases where Tier 1 already passes.

---

## Phase 5 — Production regression + CI (Week 7+)

**Goal:** Ship with confidence.

- [ ] Split CI jobs: unit → integration (3 cases) → nightly full 89
- [ ] `pipeline_evaluation_runs` trend dashboard
- [ ] PR gate: 0 pipeline failures; Tier 1 pass rate must not regress
- [ ] Optional `MEDSTATS_USE_LLM_JUDGE=1` Tier 2 on nightly
- [ ] Document deploy checklist: migrate DB, seed references, ingest knowledge, run suite

**Prod-ready definition:** 89/89 pipeline complete, ≥75% Tier 1 match, ≥75% Tier 2 style on matched cases, knowledge retrieval wired for new uploads.

---

## Commands

```bash
cd ~/medstats-api && source .venv/bin/activate

# Quick (3 cases)
pytest tests/integration/test_reference_report_suite.py -q

# Full historical suite (~10 min)
MEDSTATS_REFERENCE_SUITE=all pytest tests/integration/test_reference_report_suite.py -q

# Unit tests (parser + evaluator)
pytest tests/unit/test_masterchart_normalizer.py tests/unit/test_report_extractor.py tests/unit/test_research_evaluator.py -q
```

## Current baseline (after parser header fix)

| Outcome | Count |
|---|---|
| FAILED | 3 → target 0 |
| PASSED | 3 (unit smoke + tier1 tests) |
| XFAILED | 86 (Tier 1 numeric mismatch) |
