"""Unit tests for chart generator."""
import pytest
import os
from src.medstats.core.chart_generator import (
    generate_pie_chart,
    generate_bar_chart,
    generate_grouped_bar_chart,
    generate_line_chart,
)


class TestPieChart:
    def test_generates_file(self, tmp_path):
        path = generate_pie_chart(
            labels=["RTA", "Fall", "Sports"],
            values=[21, 14, 5],
            title="Mechanism of Injury",
            variable_name="mechanism",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)
        assert path.endswith(".png")
        assert os.path.getsize(path) > 1000

    def test_single_category(self, tmp_path):
        path = generate_pie_chart(
            labels=["Full ROM"],
            values=[47],
            title="Range of Motion",
            variable_name="rom",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)


class TestBarChart:
    def test_generates_file(self, tmp_path):
        path = generate_bar_chart(
            categories=["5-7", "7-9", "9-11", "11-13", "13-15"],
            values=[8, 6, 20, 10, 6],
            title="Age Distribution",
            variable_name="age_dist",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_with_labels(self, tmp_path):
        path = generate_bar_chart(
            categories=["Femur", "Tibia", "Humerus", "Radius-Ulna"],
            values=[23, 8, 5, 14],
            title="Bone Involved",
            variable_name="bone",
            xlabel="Bone",
            ylabel="Number of patients",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)


class TestGroupedBarChart:
    def test_generates_file(self, tmp_path):
        path = generate_grouped_bar_chart(
            categories=["HR", "SBP", "DBP", "MAP"],
            group1_values=[76.5, 118.2, 72.4, 87.6],
            group2_values=[78.1, 120.1, 74.2, 89.5],
            group1_label="Group L",
            group2_label="Group B",
            title="Vital Signs Comparison",
            variable_name="vitals_comparison",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestLineChart:
    def test_generates_file(self, tmp_path):
        path = generate_line_chart(
            x_values=["Pre-op", "0min", "5min", "10min", "15min", "30min"],
            y_series={
                "Group L": [76.5, 74.2, 72.1, 70.3, 71.5, 73.8],
                "Group B": [78.1, 76.8, 75.2, 74.0, 74.5, 76.2],
            },
            title="Heart Rate Over Time",
            variable_name="hr_timeline",
            xlabel="Time Point",
            ylabel="Heart Rate (bpm)",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)
        assert path.endswith(".png")
