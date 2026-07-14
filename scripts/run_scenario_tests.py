#!/usr/bin/env python3
"""
Student-friendly local scenario test runner (no API server required).

Usage:
    # Generate dummy datasets first (one-time)
    python scripts/generate_scenario_fixtures.py

    # Run all scenarios with readable output
    python scripts/run_scenario_tests.py

    # Run one scenario
    python scripts/run_scenario_tests.py 05_two_group_ttest

    # Run pytest suite
    python scripts/run_scenario_tests.py --pytest
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tests.scenarios.definitions import SCENARIO_BY_ID, STUDENT_SCENARIOS  # noqa: E402
from tests.scenarios.pipeline_runner import run_scenario_pipeline, summarize_scenario_result  # noqa: E402
from tests.scenarios.report_validators import (  # noqa: E402
    assert_charts_exist,
    assert_chart_suffixes,
    assert_expected_tests,
    assert_report_structure,
    collect_test_types,
    collect_variables_with_tests,
)


async def run_one(scenario_id: str, chart_dir: str) -> int:
    scenario = SCENARIO_BY_ID.get(scenario_id)
    if not scenario:
        print(f"Unknown scenario: {scenario_id}")
        return 1

    if not scenario.data_path().exists():
        print("Fixtures missing. Run: python scripts/generate_scenario_fixtures.py")
        return 1

    print("=" * 72)
    print(f"  {scenario.title}")
    print(f"  ID: {scenario.id}")
    print("=" * 72)

    state, meta = await run_scenario_pipeline(scenario, chart_dir=chart_dir)

    for line in summarize_scenario_result(scenario, state, meta):
        print(f"  {line}")

    if state.status != "COMPLETED":
        print("\n  FAILED")
        return 1

    try:
        assert meta["study_design"] == scenario.study_design
        assert_report_structure(meta["sections"], min_variables=scenario.min_variables)
        assert_charts_exist(meta["charts"])
        if scenario.expected_tests:
            assert_expected_tests(meta["test_results"], scenario.expected_tests)
        if scenario.expected_chart_suffixes:
            assert_chart_suffixes([c["path"] for c in meta["charts"]], scenario.expected_chart_suffixes)
    except AssertionError as exc:
        print(f"\n  VALIDATION FAILED: {exc}")
        return 1

    tests = collect_variables_with_tests(meta["test_results"])
    print("\n  Tests selected:")
    for var, test_type in sorted(tests.items()):
        print(f"    - {var}: {test_type}")

    print("\n  Charts:")
    for chart in meta["charts"]:
        print(f"    - {Path(chart['path']).name}")

    print("\n  PASSED")
    return 0


async def run_all(chart_base: Path) -> int:
    failed = 0
    all_types = set()

    for scenario in STUDENT_SCENARIOS:
        chart_dir = str(chart_base / scenario.id)
        Path(chart_dir).mkdir(parents=True, exist_ok=True)
        code = await run_one(scenario.id, chart_dir)
        if code != 0:
            failed += 1
        else:
            _, meta = await run_scenario_pipeline(scenario, chart_dir=chart_dir)
            all_types |= collect_test_types(meta["test_results"])
        print()

    print("=" * 72)
    print(f"  SUMMARY: {len(STUDENT_SCENARIOS) - failed}/{len(STUDENT_SCENARIOS)} passed")
    print(f"  Test types covered: {', '.join(sorted(all_types))}")
    print("=" * 72)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MedStats student scenario tests locally")
    parser.add_argument("scenario_id", nargs="?", help="Scenario id (default: run all)")
    parser.add_argument("--pytest", action="store_true", help="Run pytest suite instead")
    args = parser.parse_args()

    if args.pytest:
        import pytest
        return pytest.main(["tests/scenarios/", "-v"])

    chart_base = Path(tempfile.mkdtemp(prefix="medstats_scenarios_"))

    if args.scenario_id:
        return asyncio.run(run_one(args.scenario_id, str(chart_base / args.scenario_id)))

    return asyncio.run(run_all(chart_base))


if __name__ == "__main__":
    raise SystemExit(main())
