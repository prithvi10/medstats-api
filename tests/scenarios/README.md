# Student scenario tests

Local scenario tests validate the full statistical pipeline using dummy datasets.

## Quick start

```bash
cd medstats-api
source .venv/bin/activate

# 1. Generate dummy Excel + intent DOCX fixtures (one-time, or after fixture changes)
python scripts/generate_scenario_fixtures.py

# 2. Run all scenarios with readable output
python scripts/run_scenario_tests.py

# 3. Run pytest suite
./scripts/run_tests.sh scenarios
```

## What is covered

| Scenario | Study design | Statistical test |
|----------|--------------|------------------|
| 01 | Single cohort | Z-test (binary) |
| 02 | Single cohort | Z-test (multi-category) |
| 03 | Single cohort | Shapiro-Wilk + histogram |
| 04 | Single cohort | Z-test (ordered bands → vertical bar) |
| 05 | Two-group | Unpaired t-test |
| 06 | Two-group | Mann-Whitney U |
| 07 | Two-group | Chi-square |
| 08 | Two-group | Fisher's exact |
| 09 | Multi-group | One-way ANOVA |
| 10 | Multi-group | Kruskal-Wallis |
| 11 | Single cohort | Complex chart matrix |
| 12 | Single cohort | Master chart DOCX ingestion |
| 13 | Single cohort | Reference medicine5 dataset |
| 14 | Two-group | Complex multi-variable analysis |

Each scenario validates:
- Correct study design detection
- Appropriate test selection
- Report structure (`Table No.X`, summary, table, test result, chart)
- Chart files generated on disk
- Executive summary JSON structure

## Run one scenario

```bash
python scripts/run_scenario_tests.py 05_two_group_ttest
```

## Fixtures location

`tests/fixtures/scenarios/<scenario_id>/`
- `data.xlsx` — dummy patient data
- `intent.docx` — study intent / description
