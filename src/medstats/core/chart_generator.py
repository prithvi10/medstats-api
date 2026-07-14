"""
Chart Generator - Creates publication-quality, investor-ready charts.
High DPI, clean typography, professional color palettes.
Supports: 3D-style pie, donut, bar (horizontal/vertical), stacked bar, histogram, lollipop, grouped bar, line.
"""
import os
import re
import uuid
from typing import Dict, List, Optional, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.15)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "generated_charts")
CHART_DIR = os.path.abspath(CHART_DIR)

COLORS_PRIMARY = ["#2C5F8A", "#4A90D9", "#7AB5E8", "#2E8B57", "#F5A623", "#E74C3C", "#9B59B6", "#1ABC9C"]
COLORS_PIE = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000", "#5B9BD5", "#264478", "#9B59B6", "#E74C3C"]

# Typography scale for export-quality readability
FONT_TITLE = 18
FONT_LABEL = 14
FONT_TICK = 13
FONT_VALUE = 13
FONT_PIE_PCT = 15


def ensure_chart_dir():
    os.makedirs(CHART_DIR, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r'[^\w\-]', '_', name.lower())
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe


def _darken_hex(hex_color: str, factor: float = 0.55) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def _apply_style(ax, title: str):
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=22, color="#1a1a1a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK, colors="#333333")


def _save_figure(fig, variable_name: str, chart_type: str, output_dir: Optional[str] = None) -> str:
    chart_dir = output_dir or CHART_DIR
    os.makedirs(chart_dir, exist_ok=True)
    filename = f"{_sanitize_filename(variable_name)}_{uuid.uuid4().hex[:8]}_{chart_type}.png"
    filepath = os.path.join(chart_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.4)
    plt.close(fig)
    return filepath


def _truncate_label(label: str, max_len: int = 30) -> str:
    return label[:max_len] + "..." if len(label) > max_len else label


# ============================================================================
# 3D-STYLE PIE CHART
# ============================================================================

def generate_pie_chart(
    labels: List[str],
    values: List[float],
    title: str,
    variable_name: str,
    output_dir: Optional[str] = None,
) -> str:
    """Generate a 3D-style pie chart with exploded slices and external labels."""
    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")

    total = sum(values) if sum(values) > 0 else 1
    colors = COLORS_PIE[: len(labels)]
    explode = [0.07] * len(labels)
    display_labels = [_truncate_label(l) for l in labels]

    # Depth layer (offset shadow wedges for 3D effect)
    ax.pie(
        values,
        explode=explode,
        colors=[_darken_hex(c) for c in colors],
        startangle=90,
        radius=0.98,
        center=(-0.03, -0.05),
        wedgeprops={"linewidth": 0, "edgecolor": "none"},
    )

    wedges, _, autotexts = ax.pie(
        values,
        explode=explode,
        colors=colors,
        autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        pctdistance=0.72,
        startangle=90,
        shadow=False,
        wedgeprops={"linewidth": 2.5, "edgecolor": "white", "antialiased": True},
        textprops={"fontsize": FONT_PIE_PCT, "fontweight": "bold", "color": "white"},
    )

    for autotext in autotexts:
        autotext.set_path_effects([])

    # External labels with leader lines (like reference 3D pie)
    for i, wedge in enumerate(wedges):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        pct = values[i] / total * 100
        ha = "left" if x >= 0 else "right"
        tx = 1.45 * (1 if x >= 0 else -1)
        ty = 1.25 * y
        ax.annotate(
            f"{display_labels[i]}\n{pct:.0f}%  (n={int(values[i])})",
            xy=(x * 0.82, y * 0.82),
            xytext=(tx, ty),
            fontsize=FONT_LABEL,
            fontweight="bold",
            color=colors[i],
            ha=ha,
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color=colors[i],
                lw=1.5,
                connectionstyle="arc3,rad=0.15",
            ),
        )

    ax.set_title(title, fontsize=FONT_TITLE + 2, fontweight="bold", pad=24, color="#1a1a1a")
    plt.tight_layout()
    return _save_figure(fig, variable_name, "pie", output_dir)


def generate_donut_chart(
    labels: List[str],
    values: List[float],
    title: str,
    variable_name: str,
    center_text: str = "",
    output_dir: Optional[str] = None,
) -> str:
    """Generate a styled donut chart with center statistic."""
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    ax.set_aspect("equal")
    colors = COLORS_PIE[: len(labels)]

    wedges, _, autotexts = ax.pie(
        values,
        autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
        colors=colors,
        startangle=90,
        textprops={"fontsize": FONT_PIE_PCT, "color": "white", "fontweight": "bold"},
        pctdistance=0.78,
        wedgeprops={"width": 0.42, "linewidth": 2.5, "edgecolor": "white"},
    )

    centre = plt.Circle((0, 0), 0.52, fc="white", ec="#dddddd", linewidth=1)
    ax.add_patch(centre)

    if center_text:
        ax.text(0, 0, center_text, ha="center", va="center", fontsize=16, fontweight="bold", color="#1a1a1a")

    display_labels = [_truncate_label(l) for l in labels]
    legend_labels = [f"{l} (n={int(v)})" for l, v in zip(display_labels, values)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=FONT_LABEL, frameon=False)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=18, color="#1a1a1a")
    plt.tight_layout()
    return _save_figure(fig, variable_name, "donut", output_dir)


# ============================================================================
# BAR CHARTS (seaborn-enhanced)
# ============================================================================

def generate_bar_chart(
    categories: List[str],
    values: List[float],
    title: str,
    variable_name: str,
    xlabel: str = "",
    ylabel: str = "Count",
    output_dir: Optional[str] = None,
) -> str:
    """Generate a horizontal bar chart with large readable text."""
    n = len(categories)
    fig_height = max(5.5, min(n * 0.72 + 2.5, 14))
    fig, ax = plt.subplots(figsize=(12, fig_height), facecolor="white")

    display_cats = [_truncate_label(c, 40) for c in categories]
    colors = [COLORS_PRIMARY[i % len(COLORS_PRIMARY)] for i in range(n)]

    if HAS_SEABORN:
        import pandas as pd
        df = pd.DataFrame({"category": display_cats, "count": values})
        sns.barplot(
            data=df, y="category", x="count", hue="category", palette=colors,
            orient="h", ax=ax, order=display_cats, legend=False,
            edgecolor="white", linewidth=1.2,
        )
    else:
        y_pos = np.arange(n)
        ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.68)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_cats)

    max_val = max(values) if values else 1
    total = sum(values) if sum(values) > 0 else 1
    for bar in ax.patches:
        width = bar.get_width()
        if width <= 0:
            continue
        pct = width / total * 100
        ax.text(
            width + max_val * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"  {int(width)} ({pct:.1f}%)",
            ha="left", va="center", fontsize=FONT_VALUE, fontweight="medium", color="#333333",
        )

    ax.set_xlabel(ylabel, fontsize=FONT_LABEL + 1, color="#333333", labelpad=10)
    ax.set_ylabel("")
    ax.set_xlim(0, max_val * 1.38)
    if not HAS_SEABORN:
        ax.invert_yaxis()

    ax.tick_params(axis="y", labelsize=FONT_LABEL)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    _apply_style(ax, title)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.grid(axis="y", visible=False)

    plt.tight_layout()
    return _save_figure(fig, variable_name, "bar", output_dir)


def generate_vertical_bar_chart(
    categories: List[str],
    values: List[float],
    title: str,
    variable_name: str,
    ylabel: str = "Number of patients",
    output_dir: Optional[str] = None,
) -> str:
    """Generate a vertical bar chart for ordered/range data."""
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    display_cats = [_truncate_label(c, 22) for c in categories]
    total = sum(values) if sum(values) > 0 else 1
    max_v = max(values) if values else 1
    colors = [COLORS_PRIMARY[i % len(COLORS_PRIMARY)] for i in range(len(categories))]

    if HAS_SEABORN:
        import pandas as pd
        df = pd.DataFrame({"category": display_cats, "count": values})
        sns.barplot(
            data=df, x="category", y="count", hue="category", palette=colors,
            ax=ax, legend=False, edgecolor="white", linewidth=1.2,
        )
    else:
        x_pos = np.arange(len(categories))
        ax.bar(x_pos, values, color=colors, edgecolor="white", width=0.65)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_cats, rotation=28, ha="right")

    for i, val in enumerate(values):
        pct = val / total * 100
        ax.text(
            i, val + max_v * 0.03,
            f"{int(val)}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=FONT_VALUE, fontweight="medium", color="#333333",
        )

    ax.set_ylabel(ylabel, fontsize=FONT_LABEL + 1, color="#333333")
    ax.set_xlabel("")
    ax.set_ylim(0, max_v * 1.28)
    ax.tick_params(axis="x", labelsize=FONT_LABEL, rotation=28)
    ax.tick_params(axis="y", labelsize=FONT_TICK)

    _apply_style(ax, title)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)

    plt.tight_layout()
    return _save_figure(fig, variable_name, "vbar", output_dir)


def generate_lollipop_chart(
    categories: List[str],
    values: List[float],
    title: str,
    variable_name: str,
    ylabel: str = "Count",
    output_dir: Optional[str] = None,
) -> str:
    """Generate a lollipop chart with large labels."""
    n = len(categories)
    fig_height = max(5.5, min(n * 0.68 + 2.5, 14))
    fig, ax = plt.subplots(figsize=(12, fig_height), facecolor="white")

    display_cats = [_truncate_label(c, 40) for c in categories]
    sorted_idx = np.argsort(values)
    sorted_vals = [values[i] for i in sorted_idx]
    sorted_cats = [display_cats[i] for i in sorted_idx]
    y_pos = np.arange(n)

    ax.hlines(y=y_pos, xmin=0, xmax=sorted_vals, color=COLORS_PRIMARY[0], alpha=0.55, linewidth=3)
    ax.scatter(sorted_vals, y_pos, color=COLORS_PRIMARY[1], s=140, zorder=3,
               edgecolors="white", linewidth=2)

    total = sum(values) if sum(values) > 0 else 1
    max_val = max(values) if values else 1
    for val, y in zip(sorted_vals, y_pos):
        pct = val / total * 100
        ax.text(val + max_val * 0.03, y, f"{int(val)} ({pct:.1f}%)",
                va="center", fontsize=FONT_VALUE, fontweight="medium", color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_cats, fontsize=FONT_LABEL)
    ax.set_xlabel(ylabel, fontsize=FONT_LABEL + 1, color="#333333")
    ax.set_xlim(0, max_val * 1.38)

    _apply_style(ax, title)
    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)

    plt.tight_layout()
    return _save_figure(fig, variable_name, "lollipop", output_dir)


def generate_stacked_bar_chart(
    categories: List[str],
    data_dict: Dict[str, List[float]],
    title: str,
    variable_name: str,
    ylabel: str = "Count",
    output_dir: Optional[str] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    x_pos = np.arange(len(categories))
    bottom = np.zeros(len(categories))

    for i, (label, vals) in enumerate(data_dict.items()):
        color = COLORS_PRIMARY[i % len(COLORS_PRIMARY)]
        ax.bar(x_pos, vals, bottom=bottom, label=label, color=color, edgecolor="white", width=0.6)
        bottom += np.array(vals)

    display_cats = [_truncate_label(c, 20) for c in categories]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_cats, rotation=25, ha="right", fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL + 1)
    ax.legend(fontsize=FONT_TICK, frameon=False, loc="upper right")
    _apply_style(ax, title)
    plt.tight_layout()
    return _save_figure(fig, variable_name, "stacked", output_dir)


def generate_histogram(
    values: List[float],
    title: str,
    variable_name: str,
    xlabel: str = "",
    ylabel: str = "Frequency",
    bins: int = 10,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="white")

    if HAS_SEABORN:
        sns.histplot(values, bins=bins, color=COLORS_PRIMARY[0], edgecolor="white",
                     linewidth=1.2, alpha=0.88, ax=ax)
    else:
        ax.hist(values, bins=bins, color=COLORS_PRIMARY[0], edgecolor="white", linewidth=1, alpha=0.85)

    if mean is not None:
        ax.axvline(mean, color="#E74C3C", linewidth=2.5, linestyle="--", label=f"Mean = {mean:.2f}")
        if std is not None:
            ax.axvspan(mean - std, mean + std, alpha=0.12, color="#E74C3C", label=f"±1 SD ({std:.2f})")
        ax.legend(fontsize=FONT_TICK, frameon=False, loc="upper right")

    ax.set_xlabel(xlabel or variable_name.replace("_", " ").title(), fontsize=FONT_LABEL + 1)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL + 1)
    ax.tick_params(labelsize=FONT_TICK)
    _apply_style(ax, title)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    return _save_figure(fig, variable_name, "hist", output_dir)


def generate_grouped_bar_chart(
    categories: List[str],
    group1_values: List[float],
    group2_values: List[float],
    group1_label: str,
    group2_label: str,
    title: str,
    variable_name: str,
    output_dir: Optional[str] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, group1_values, width, label=group1_label,
                   color=COLORS_PRIMARY[0], edgecolor="white")
    bars2 = ax.bar(x + width / 2, group2_values, width, label=group2_label,
                   color=COLORS_PRIMARY[1], edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{int(h)}",
                        ha="center", va="bottom", fontsize=FONT_VALUE, color="#333333")

    display_cats = [_truncate_label(c, 20) for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(display_cats, rotation=28, ha="right", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_TICK, frameon=False)
    _apply_style(ax, title)
    plt.tight_layout()
    return _save_figure(fig, variable_name, "grouped_bar", output_dir)


def generate_line_chart(
    x_values: List[Any],
    y_series: Dict[str, List[float]],
    title: str,
    variable_name: str,
    xlabel: str = "Time",
    ylabel: str = "Value",
    output_dir: Optional[str] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")

    for i, (label, values) in enumerate(y_series.items()):
        color = COLORS_PRIMARY[i % len(COLORS_PRIMARY)]
        ax.plot(x_values[:len(values)], values, marker="o", markersize=7,
                label=label, color=color, linewidth=2.8)

    ax.set_xlabel(xlabel, fontsize=FONT_LABEL + 1)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL + 1)
    ax.legend(fontsize=FONT_TICK, frameon=False)
    ax.tick_params(labelsize=FONT_TICK)
    _apply_style(ax, title)
    plt.xticks(rotation=35, fontsize=FONT_TICK)
    plt.tight_layout()
    return _save_figure(fig, variable_name, "line", output_dir)


def select_and_generate_chart(
    variable_name: str,
    freq_table: Optional[dict],
    descriptive: Optional[dict],
    data_type: str = "categorical",
    raw_values: Optional[List[float]] = None,
    sample_size: int = 0,
    output_dir: Optional[str] = None,
) -> str:
    """Intelligently select the best chart type based on data characteristics."""
    safe_name = variable_name.replace(" ", "_").replace("/", "_").lower()
    title = f"Distribution of {variable_name.replace('_', ' ').title()}"

    if data_type == "continuous" and descriptive and descriptive.get("mean") is not None:
        if raw_values and len(raw_values) >= 10:
            return generate_histogram(
                values=raw_values, title=title, variable_name=safe_name,
                mean=descriptive.get("mean"), std=descriptive.get("std"),
                output_dir=output_dir,
            )

    if not freq_table or not freq_table.get("rows"):
        return ""

    rows = [r for r in freq_table["rows"] if r.get("category") != "TOTAL"]
    if not rows:
        return ""

    from src.medstats.agents.report_generator.formatters import group_top_categories
    if len(rows) > 12:
        rows = group_top_categories(rows, top_n=10)

    labels = [r["category"] for r in rows]
    values = [r["count"] for r in rows]
    total = sum(values)

    try:
        is_ordered = _looks_like_ordered_range(labels)
        n_cats = len(labels)

        if n_cats == 2:
            return generate_donut_chart(
                labels=labels, values=values, title=title, variable_name=safe_name,
                center_text=f"N={total}", output_dir=output_dir,
            )
        elif n_cats <= 5 and not is_ordered:
            return generate_pie_chart(
                labels=labels, values=values, title=title,
                variable_name=safe_name, output_dir=output_dir,
            )
        elif is_ordered:
            return generate_vertical_bar_chart(
                categories=labels, values=values, title=title,
                variable_name=safe_name, output_dir=output_dir,
            )
        elif n_cats <= 8:
            return generate_lollipop_chart(
                categories=labels, values=values, title=title,
                variable_name=safe_name, output_dir=output_dir,
            )
        else:
            return generate_bar_chart(
                categories=labels, values=values, title=title,
                variable_name=safe_name, ylabel="Number of patients",
                output_dir=output_dir,
            )
    except (OSError, PermissionError, ValueError):
        return ""


def _looks_like_ordered_range(labels: List[str]) -> bool:
    range_patterns = [
        r'^\d+\s*[-–]\s*\d+',
        r'^[<>≤≥]',
        r'^\d+\+',
        r'weeks?|days?|mins?|years?|months?|hrs?',
    ]
    matches = sum(
        1 for label in labels
        if any(re.search(pat, label, re.IGNORECASE) for pat in range_patterns)
    )
    return matches >= len(labels) * 0.5
