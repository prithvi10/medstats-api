"""
Chart Generator - Creates publication-ready charts for statistical reports.
Generates matplotlib figures and saves as PNG images.
"""
import os
import uuid
from typing import Dict, List, Optional, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHART_DIR = "generated_charts"


def ensure_chart_dir():
    os.makedirs(CHART_DIR, exist_ok=True)


def generate_pie_chart(
    labels: List[str],
    values: List[float],
    title: str,
    variable_name: str,
) -> str:
    """Generate a pie chart and return the file path."""
    ensure_chart_dir()
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()

    filename = f"{variable_name}_{uuid.uuid4().hex[:8]}_pie.png"
    filepath = os.path.join(CHART_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def generate_bar_chart(
    categories: List[str],
    values: List[float],
    title: str,
    variable_name: str,
    xlabel: str = "",
    ylabel: str = "Count",
) -> str:
    """Generate a bar chart and return the file path."""
    ensure_chart_dir()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(categories)))
    bars = ax.bar(categories, values, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(val)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    filename = f"{variable_name}_{uuid.uuid4().hex[:8]}_bar.png"
    filepath = os.path.join(CHART_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def generate_grouped_bar_chart(
    categories: List[str],
    group1_values: List[float],
    group2_values: List[float],
    group1_label: str,
    group2_label: str,
    title: str,
    variable_name: str,
) -> str:
    """Generate a grouped bar chart comparing two groups."""
    ensure_chart_dir()
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, group1_values, width, label=group1_label, color="#4C8BF5")
    bars2 = ax.bar(x + width / 2, group2_values, width, label=group2_label, color="#F5A623")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    filename = f"{variable_name}_{uuid.uuid4().hex[:8]}_grouped_bar.png"
    filepath = os.path.join(CHART_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def generate_line_chart(
    x_values: List[Any],
    y_series: Dict[str, List[float]],
    title: str,
    variable_name: str,
    xlabel: str = "Time",
    ylabel: str = "Value",
) -> str:
    """Generate a multi-line chart (e.g., vitals over time)."""
    ensure_chart_dir()
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, values in y_series.items():
        ax.plot(x_values[: len(values)], values, marker="o", markersize=3, label=label)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()

    filename = f"{variable_name}_{uuid.uuid4().hex[:8]}_line.png"
    filepath = os.path.join(CHART_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath
