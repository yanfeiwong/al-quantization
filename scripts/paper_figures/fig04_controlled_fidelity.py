"""Generate publication Figure 4: controlled quantization fidelity.

The three panels are sourced from executed cells F7, A1, and D4 of
``theory_and_ablation_final.ipynb``. Stable PDF, SVG, and PNG assets are
written under ``figures/``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from style import (
    ACCENT,
    ACCENT_DARK,
    FAINT,
    INK,
    LINE,
    MUTED,
    WHITE,
    configure_matplotlib,
    save_public_figure,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Manual tuning controls
# =============================================================================

FIGURE_SIZE_IN = (7.05, 3.62)
GRID_LEFT = 0.105
GRID_RIGHT = 0.985
GRID_BOTTOM = 0.125
GRID_TOP = 0.895
GRID_WIDTH_RATIOS = (1.48, 1.0)
GRID_HEIGHT_RATIOS = (1.0, 1.0)
GRID_WSPACE = 0.42
GRID_HSPACE = 0.60

TITLE_SIZE = 7.2
PANEL_LABEL_SIZE = 7.8
AXIS_LABEL_SIZE = 5.7
TICK_LABEL_SIZE = 5.25
LEGEND_SIZE = 5.0
NOTE_SIZE = 4.8
ANNOTATION_SIZE = 4.8

AL8_COLOR = ACCENT
AL16_COLOR = ACCENT_DARK
DYN8_COLOR = "#7E8999"
FIXED_DARK = "#626D7C"
FIXED_LIGHT = "#A6AFBC"
ROW_BAND = "#F6F8FB"

SENSITIVITY_X_MIN = 3e-4
SENSITIVITY_X_MAX = 165.0
CENSORED_X = 105.0
LOW_CENSORED_X = 5e-4

DATA_DIR = Path(__file__).resolve().parent / "data"
SENSITIVITY_CSV = DATA_DIR / "fig04_state_sensitivity.csv"
FIXED_RANGE_CSV = DATA_DIR / "fig04_fixed_range.csv"
PARETO_CSV = DATA_DIR / "fig04_pareto.csv"
OUTPUT_STEM = "fig04_controlled_fidelity"


# =============================================================================
# Data loading
# =============================================================================

def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def style_axis(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(LINE)
        ax.spines[spine].set_linewidth(0.65)
    ax.tick_params(colors=MUTED, labelsize=TICK_LABEL_SIZE, width=0.6)
    ax.set_axisbelow(True)


def panel_title(ax, label: str, title: str) -> None:
    ax.text(
        -0.08,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    ax.text(
        0.0,
        1.10,
        title,
        transform=ax.transAxes,
        fontsize=TITLE_SIZE,
        fontweight="semibold",
        color=INK,
        va="bottom",
    )


# =============================================================================
# Panel a: trace-calibrated state sensitivity
# =============================================================================

def draw_sensitivity(ax, rows: list[dict[str, str]]) -> None:
    categories = [
        ("V_r", "Attention"),
        ("V_r", "MLP"),
        ("V_r", "Embedding"),
        ("C_r", "Attention"),
        ("C_r", "MLP"),
        ("C_r", "LM head"),
        ("C_r", "Embedding"),
    ]
    labels = [
        r"$V_r$  Attention",
        r"$V_r$  MLP",
        r"$V_r$  Embedding",
        r"$C_r$  Attention",
        r"$C_r$  MLP",
        r"$C_r$  LM head",
        r"$C_r$  Embedding",
    ]
    base = np.arange(len(categories), dtype=float)
    offsets = {"AL8": -0.17, "AL16": 0.0, "Dyn8": 0.17}
    colors = {"AL8": AL8_COLOR, "AL16": AL16_COLOR, "Dyn8": DYN8_COLOR}
    markers = {"AL8": "o", "AL16": "D", "Dyn8": "s"}

    lookup = {
        (row["family"], row["state"], row["method"]): row
        for row in rows
    }

    for index, y in enumerate(base):
        if index % 2 == 0:
            ax.axhspan(y - 0.44, y + 0.44, color=ROW_BAND, zorder=0)
    ax.axhline(2.5, color=LINE, linewidth=0.8, zorder=1)

    for method in ("AL8", "AL16", "Dyn8"):
        xs = []
        ys = []
        censored_high = []
        censored_low = []
        false_zeros = []
        for index, key in enumerate(categories):
            row = lookup[(key[0], key[1], method)]
            is_high = bool(int(row["censored_high"]))
            is_low = bool(int(row["censored_low"]))
            value = (
                CENSORED_X if is_high
                else LOW_CENSORED_X if is_low
                else float(row["update_error_pct"])
            )
            xs.append(value)
            ys.append(base[index] + offsets[method])
            censored_high.append(is_high)
            censored_low.append(is_low)
            false_zeros.append(int(row["false_zeros"]))

        normal = np.asarray([
            not high and not low
            for high, low in zip(censored_high, censored_low)
        ])
        ax.scatter(
            np.asarray(xs)[normal],
            np.asarray(ys)[normal],
            s=18 if method != "AL16" else 15,
            marker=markers[method],
            facecolors=(WHITE if method == "AL16" else colors[method]),
            edgecolors=colors[method],
            linewidths=0.75,
            zorder=4,
        )
        for x, y, is_high, is_low, false_zero in zip(
            xs, ys, censored_high, censored_low, false_zeros
        ):
            if not is_high and not is_low:
                continue
            ax.scatter(
                [x],
                [y],
                s=24,
                marker=">" if is_high else "<",
                facecolors=(WHITE if method == "AL16" else colors[method]),
                edgecolors=colors[method],
                linewidths=0.85,
                zorder=5,
            )
            if is_high:
                ax.annotate(
                    f">100; {false_zero / 1000:.1f}K false zeros",
                    (x, y),
                    xytext=(-3, 4),
                    textcoords="offset points",
                    ha="right",
                    fontsize=ANNOTATION_SIZE,
                    color=DYN8_COLOR,
                )
            else:
                ax.annotate(
                    "<0.001",
                    (x, y),
                    xytext=(4, -7),
                    textcoords="offset points",
                    ha="left",
                    fontsize=ANNOTATION_SIZE,
                    color=AL16_COLOR,
                )

    ax.set_xscale("log")
    ax.set_xlim(SENSITIVITY_X_MIN, SENSITIVITY_X_MAX)
    ax.set_xticks(
        [0.001, 0.01, 0.1, 1, 10, 100],
        ["0.001", "0.01", "0.1", "1", "10", "100"],
    )
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_ylim(len(categories) - 0.45, -0.55)
    ax.set_yticks(base, labels)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", color=LINE, linewidth=0.5, alpha=0.78)
    ax.set_xlabel("Single-step update error (%)", fontsize=AXIS_LABEL_SIZE, color=INK)
    style_axis(ax)
    panel_title(ax, "a", "Trace-calibrated state sensitivity")

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor=AL8_COLOR, markeredgecolor=AL8_COLOR, label="AL8"),
        Line2D([], [], marker="D", linestyle="none", markersize=3.6,
               markerfacecolor=WHITE, markeredgecolor=AL16_COLOR, label="AL16"),
        Line2D([], [], marker="s", linestyle="none", markersize=4.0,
               markerfacecolor=DYN8_COLOR, markeredgecolor=DYN8_COLOR, label="Dyn8"),
    ]
    ax.legend(
        handles=handles,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        frameon=False,
        fontsize=LEGEND_SIZE,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0,
    )


# =============================================================================
# Panel b: adaptive versus fixed range
# =============================================================================

def draw_fixed_range(ax, rows: list[dict[str, str]]) -> None:
    regimes = ["Narrow", "Medium", "Wide"]
    methods = ["Adaptive AL8", "Fixed -126", "Fixed -40"]
    colors = {
        "Adaptive AL8": AL8_COLOR,
        "Fixed -126": FIXED_DARK,
        "Fixed -40": FIXED_LIGHT,
    }
    markers = {"Adaptive AL8": "o", "Fixed -126": "^", "Fixed -40": "s"}
    lookup = {(row["regime"], row["method"]): float(row["update_error_pct"])
              for row in rows}
    x = np.arange(len(regimes), dtype=float)

    for method in methods:
        y = [lookup[(regime, method)] for regime in regimes]
        ax.plot(
            x,
            y,
            color=colors[method],
            linewidth=1.05,
            marker=markers[method],
            markersize=3.6,
            markerfacecolor=(WHITE if method != "Adaptive AL8" else colors[method]),
            markeredgewidth=0.75,
            label=method,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_ylim(0.025, 80)
    ax.set_yticks([0.03, 0.1, 1, 10], ["0.03", "0.1", "1", "10"])
    ax.tick_params(axis="y", which="minor", left=False)
    ax.set_xticks(x, regimes)
    ax.set_ylabel("Update error (%)", fontsize=AXIS_LABEL_SIZE, color=INK)
    ax.grid(axis="y", color=LINE, linewidth=0.5, alpha=0.78)
    style_axis(ax)
    panel_title(ax, "b", "No fixed floor fits every regime")
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.5,
        borderaxespad=0.2,
        labelspacing=0.25,
    )


# =============================================================================
# Panel c: memory-fidelity Pareto
# =============================================================================

def draw_pareto(ax, rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    for method, color, marker in (
        ("AL8", AL8_COLOR, "o"),
        ("Dyn8", DYN8_COLOR, "s"),
    ):
        selected = sorted(grouped[method], key=lambda row: float(row["bits_per_element"]))
        x = [float(row["bits_per_element"]) for row in selected]
        y = [float(row["drift_pct"]) for row in selected]
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.05,
            marker=marker,
            markersize=3.5,
            label=method,
            zorder=3,
        )
        annotation_offsets = {
            ("AL8", 2048): (5, 6),
            ("AL8", 256): (5, 4),
            ("Dyn8", 2048): (7, 3),
            ("Dyn8", 256): (7, -8),
        }
        for row in selected:
            if int(row["block_size"]) not in (256, 2048):
                continue
            offset = annotation_offsets[(method, int(row["block_size"]))]
            ax.annotate(
                f"B={row['block_size']}",
                (float(row["bits_per_element"]), float(row["drift_pct"])),
                xytext=offset,
                textcoords="offset points",
                fontsize=ANNOTATION_SIZE,
                color=color,
                va="bottom" if offset[1] >= 0 else "top",
            )

    ax.set_yscale("log")
    ax.set_xlim(7.995, 8.52)
    ax.set_ylim(0.4, 22)
    ax.set_xticks([8.0, 8.125, 8.25, 8.5], ["8.00", "8.125", "8.25", "8.50"])
    ax.set_yticks([0.5, 1, 5, 10, 20], ["0.5", "1", "5", "10", "20"])
    ax.tick_params(axis="y", which="minor", left=False)
    ax.set_xlabel("Stored bits / element", fontsize=AXIS_LABEL_SIZE, color=INK)
    ax.set_ylabel("5K V drift (%)", fontsize=AXIS_LABEL_SIZE, color=INK)
    ax.grid(axis="y", color=LINE, linewidth=0.5, alpha=0.78)
    style_axis(ax)
    panel_title(ax, "c", "AL stays on the low-drift frontier")
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=LEGEND_SIZE,
        borderaxespad=0.2,
        labelspacing=0.25,
    )


# =============================================================================
# Figure assembly
# =============================================================================

def build() -> dict[str, Path]:
    configure_matplotlib()
    fig = plt.figure(figsize=FIGURE_SIZE_IN, facecolor=WHITE)
    grid = fig.add_gridspec(
        2,
        2,
        left=GRID_LEFT,
        right=GRID_RIGHT,
        bottom=GRID_BOTTOM,
        top=GRID_TOP,
        width_ratios=GRID_WIDTH_RATIOS,
        height_ratios=GRID_HEIGHT_RATIOS,
        wspace=GRID_WSPACE,
        hspace=GRID_HSPACE,
    )

    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    draw_sensitivity(ax_a, load_csv(SENSITIVITY_CSV))
    draw_fixed_range(ax_b, load_csv(FIXED_RANGE_CSV))
    draw_pareto(ax_c, load_csv(PARETO_CSV))

    fig.text(
        GRID_RIGHT,
        0.025,
        "Lower is better",
        fontsize=NOTE_SIZE,
        color=MUTED,
        ha="right",
        va="bottom",
    )

    return save_public_figure(fig, OUTPUT_STEM)


def main() -> None:
    for path in build().values():
        print(path)


if __name__ == "__main__":
    main()
