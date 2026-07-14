#!/usr/bin/env python3
"""
Generate dummy Excel/DOCX fixtures for student scenario tests.

Usage:
    python scripts/generate_scenario_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tests.scenarios.definitions import SCENARIOS_ROOT, STUDENT_SCENARIOS  # noqa: E402

RNG = np.random.default_rng(42)


def write_intent_docx(path: Path, text: str, title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    if title:
        doc.add_heading(title, level=1)
    doc.add_paragraph(text)
    doc.save(path)


def write_excel(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


def _base_cols(n: int) -> dict:
    return {
        "NO": list(range(1, n + 1)),
        "NAME": [f"Patient_{i:03d}" for i in range(1, n + 1)],
    }


def scenario_01(n: int = 60) -> pd.DataFrame:
    sex = RNG.choice(["Male", "Female"], size=n, p=[0.82, 0.18])
    return pd.DataFrame({**_base_cols(n), "SEX": sex, "AGE": RNG.integers(6, 16, size=n)})


def scenario_02(n: int = 50) -> pd.DataFrame:
    mech = RNG.choice(
        ["RTA", "Simple Fall", "Fall From Height", "Sports Related"],
        size=n,
        p=[0.34, 0.32, 0.24, 0.10],
    )
    return pd.DataFrame({**_base_cols(n), "MECHANISM OF INJURY": mech, "AGE": RNG.integers(6, 16, size=n)})


def scenario_03(n: int = 50) -> pd.DataFrame:
    age = np.clip(RNG.normal(10.5, 2.2, size=n).round(1), 6, 15)
    return pd.DataFrame({**_base_cols(n), "AGE": age, "SEX": RNG.choice(["Male", "Female"], size=n)})


def scenario_04(n: int = 50) -> pd.DataFrame:
    bins = ["6-7", "8-9", "10-11", "12-13", "14-15"]
    groups = RNG.choice(bins, size=n, p=[0.16, 0.12, 0.40, 0.20, 0.12])
    return pd.DataFrame({**_base_cols(n), "AGE BAND": groups, "SEX": RNG.choice(["Male", "Female"], size=n)})


def scenario_05(n: int = 50) -> pd.DataFrame:
    group = RNG.choice(["Control", "Intervention"], size=n)
    recovery = np.where(
        group == "Control",
        RNG.normal(30, 4, size=n),
        RNG.normal(22, 3.5, size=n),
    ).round(1)
    return pd.DataFrame({
        **_base_cols(n),
        "GROUP": group,
        "RECOVERY DAYS": np.clip(recovery, 10, 45),
        "SEX": RNG.choice(["Male", "Female"], size=n),
    })


def scenario_06(n: int = 50) -> pd.DataFrame:
    group = RNG.choice(["Control", "Intervention"], size=n)
    stay = np.where(
        group == "Control",
        RNG.exponential(8, size=n) + 2,
        RNG.exponential(5, size=n) + 1,
    ).round(1)
    return pd.DataFrame({
        **_base_cols(n),
        "GROUP": group,
        "HOSPITAL STAY": np.clip(stay, 1, 40),
        "SEX": RNG.choice(["Male", "Female"], size=n),
    })


def scenario_07(n: int = 50) -> pd.DataFrame:
    group = RNG.choice(["Control", "Treatment"], size=n)
    recovered = np.where(
        group == "Control",
        RNG.choice(["Yes", "No"], size=n, p=[0.55, 0.45]),
        RNG.choice(["Yes", "No"], size=n, p=[0.85, 0.15]),
    )
    return pd.DataFrame({**_base_cols(n), "GROUP": group, "RECOVERED": recovered})


def scenario_08(n: int = 20) -> pd.DataFrame:
    group = ["Control"] * 10 + ["Treatment"] * 10
    complication = (
        ["No"] * 8 + ["Yes"] * 2 + ["No"] * 4 + ["Yes"] * 6
    )
    return pd.DataFrame({**_base_cols(n), "GROUP": group, "COMPLICATION": complication})


def scenario_09(n: int = 60) -> pd.DataFrame:
    treatment = RNG.choice(["Drug A", "Drug B", "Drug C"], size=n)
    pain = np.where(
        treatment == "Drug A",
        RNG.normal(7.5, 1.2, size=n),
        np.where(treatment == "Drug B", RNG.normal(5.0, 1.0, size=n), RNG.normal(3.2, 0.9, size=n)),
    ).round(1)
    return pd.DataFrame({
        **_base_cols(n),
        "TREATMENT": treatment,
        "PAIN SCORE": np.clip(pain, 0, 10),
        "SEX": RNG.choice(["Male", "Female"], size=n),
    })


def scenario_10(n: int = 60) -> pd.DataFrame:
    treatment = RNG.choice(["Arm 1", "Arm 2", "Arm 3"], size=n)
    score = np.where(
        treatment == "Arm 1",
        RNG.exponential(4, size=n),
        np.where(treatment == "Arm 2", RNG.exponential(6, size=n), RNG.exponential(9, size=n)),
    ).round(1)
    return pd.DataFrame({
        **_base_cols(n),
        "TREATMENT": treatment,
        "COMPLICATION SCORE": np.clip(score, 0, 30),
    })


def scenario_11(n: int = 55) -> pd.DataFrame:
    return pd.DataFrame({
        **_base_cols(n),
        "SEX": RNG.choice(["Male", "Female"], size=n, p=[0.7, 0.3]),
        "INJURY TYPE": RNG.choice(["Type A", "Type B", "Type C", "Type D"], size=n),
        "MECHANISM": RNG.choice(
            [f"Cause {i}" for i in range(1, 10)],
            size=n,
        ),
        "AGE BAND": RNG.choice(["6-8", "9-11", "12-14", "15-17"], size=n),
        "AGE": np.clip(RNG.normal(11, 2.5, size=n).round(1), 6, 17),
    })


def scenario_12_masterchart(path: Path, n: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    doc.add_heading("Master Chart - Fracture Cohort", level=1)
    table = doc.add_table(rows=1, cols=4)
    hdr = table.rows[0].cells
    hdr[0].text = "NO"
    hdr[1].text = "AGE"
    hdr[2].text = "SEX"
    hdr[3].text = "SIDE"
    sex = RNG.choice(["Male", "Female"], size=n, p=[0.75, 0.25])
    side = RNG.choice(["Left", "Right"], size=n)
    for i in range(n):
        row = table.add_row().cells
        row[0].text = str(i + 1)
        row[1].text = str(int(RNG.integers(6, 16)))
        row[2].text = sex[i]
        row[3].text = side[i]
    doc.save(path)


def scenario_14(n: int = 60) -> pd.DataFrame:
    group = RNG.choice(["Control", "Treatment"], size=n)
    pain = np.where(group == "Control", RNG.normal(6.5, 1.5, size=n), RNG.normal(4.0, 1.2, size=n))
    recovered = np.where(
        group == "Control",
        RNG.choice(["Yes", "No"], size=n, p=[0.5, 0.5]),
        RNG.choice(["Yes", "No"], size=n, p=[0.8, 0.2]),
    )
    complication = np.where(
        group == "Control",
        RNG.choice(["Yes", "No"], size=n, p=[0.35, 0.65]),
        RNG.choice(["Yes", "No"], size=n, p=[0.15, 0.85]),
    )
    return pd.DataFrame({
        **_base_cols(n),
        "GROUP": group,
        "AGE": np.clip(RNG.normal(10, 2, size=n).round(1), 6, 15),
        "SEX": RNG.choice(["Male", "Female"], size=n),
        "PAIN SCORE": np.clip(pain.round(1), 0, 10),
        "RECOVERED": recovered,
        "COMPLICATION": complication,
        "FOLLOW UP WEEKS": np.clip(RNG.normal(8, 2, size=n).round(1), 2, 16),
    })


BUILDERS = {
    "01_single_cohort_binary": scenario_01,
    "02_single_cohort_multicategory": scenario_02,
    "03_single_cohort_numerical": scenario_03,
    "04_single_cohort_ordered_ranges": scenario_04,
    "05_two_group_ttest": scenario_05,
    "06_two_group_mann_whitney": scenario_06,
    "07_two_group_chi_square": scenario_07,
    "08_two_group_fisher_exact": scenario_08,
    "09_multi_group_anova": scenario_09,
    "10_multi_group_kruskal": scenario_10,
    "11_complex_charts": scenario_11,
    "14_complex_analysis": scenario_14,
}


def copy_medicine5_reference() -> None:
    src_dir = ROOT / "tests" / "fixtures"
    dest = SCENARIOS_ROOT / "13_medicine5_reference"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("medicine5_data.xlsx", "DISCUSSION.docx"):
        src = src_dir / name
        dst = dest / ("discussion.docx" if name == "DISCUSSION.docx" else name)
        if src.exists():
            dst.write_bytes(src.read_bytes())


def main() -> int:
    SCENARIOS_ROOT.mkdir(parents=True, exist_ok=True)

    for scenario in STUDENT_SCENARIOS:
        if scenario.id == "13_medicine5_reference":
            copy_medicine5_reference()
            write_intent_docx(
                scenario.intent_path(),
                scenario.intent,
                scenario.title,
            )
            print(f"  OK  {scenario.id} (copied reference fixtures)")
            continue

        if scenario.id == "12_masterchart_docx":
            scenario_12_masterchart(scenario.data_path())
            write_intent_docx(scenario.intent_path(), scenario.intent, scenario.title)
            print(f"  OK  {scenario.id} (masterchart.docx)")
            continue

        builder = BUILDERS.get(scenario.folder)
        if not builder:
            print(f"  SKIP {scenario.id} — no builder")
            continue

        df = builder()
        write_excel(scenario.data_path(), df)
        write_intent_docx(scenario.intent_path(), scenario.intent, scenario.title)
        print(f"  OK  {scenario.id} ({len(df)} rows, {len(df.columns)} cols)")

    print(f"\nFixtures written to {SCENARIOS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
