"""Generate publication Figure 5: 20K-step language-model trajectories.

The curves are parsed from ``reports_md/tb_analysis_report.md``, which was
generated from the TensorBoard event files.  Stable PDF, SVG, and PNG assets
are written under ``figures/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from style import (
    ACCENT,
    INK,
    LINE,
    MUTED,
    WHITE,
    configure_matplotlib,
    save_public_figure,
)

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Manual tuning controls
# =============================================================================

FIGURE_SIZE_IN = (7.05, 3.55)
GRID_LEFT = 0.075
GRID_RIGHT = 0.975
GRID_BOTTOM = 0.145
GRID_TOP = 0.905
GRID_WSPACE = 0.27
GRID_HSPACE = 0.54

TITLE_SIZE = 7.3
PANEL_LABEL_SIZE = 7.8
AXIS_LABEL_SIZE = 5.8
TICK_LABEL_SIZE = 5.2
END_LABEL_SIZE = 5.0
FOOTER_SIZE = 4.8

LINE_WIDTH = 1.00
OURS_LINE_WIDTH = 1.16
MARKER_SIZE = 2.15
MARK_EVERY = 2
START_INDEX = 1  # omit the first evaluation at step 1K
LABEL_X = 20_900
X_MAX = 28_000  # reserve room for direct endpoint labels

REPORT_PATH = Path(__file__).resolve().parents[2] / "reports_md" / "tb_analysis_report.md"
OUTPUT_STEM = "fig05_training_curves"

REFERENCE_COLOR = "#333944"
BASELINE_COLOR = "#D98E19"
AL16_COLOR = "#008F72"
SUPPORT_COLOR = "#56A3D9"
G0_COLOR = "#7E9FD3"


@dataclass(frozen=True)
class Curve:
    run_id: str
    label: str
    color: str
    linestyle: str = "-"
    marker: str = "o"
    label_y: float = 0.0
    linewidth: float = LINE_WIDTH


PANELS = [
    (
        "a",
        "AdamW",
        (66, 305),
        [
            Curve(
                "G0_adamw_torch", "FP32", REFERENCE_COLOR,
                linestyle="--", label_y=85.0,
            ),
            Curve(
                "G0_adamw_8bit_bnb_d8_bnb_u8_vblk256",
                "Dyn8",
                BASELINE_COLOR,
                marker="s",
                label_y=135.0,
            ),
            Curve(
                "G0_adamw_ours_uf8_al8_vblk2048",
                "Ours AL8",
                ACCENT,
                label_y=110.0,
                linewidth=OURS_LINE_WIDTH,
            ),
        ],
    ),
    (
        "b",
        "CAME",
        (80, 285),
        [
            Curve(
                "G0_came_torch", "FP32", REFERENCE_COLOR,
                linestyle="--", label_y=108.0,
            ),
            Curve(
                "G0_came_ours_uf8_al8_vblk2048",
                "AL8",
                ACCENT,
                label_y=148.0,
                linewidth=OURS_LINE_WIDTH,
            ),
            Curve(
                "G0_came_ours_uf8_al8_vblk2048_c_fp32",
                "AL8 + FP32 C",
                SUPPORT_COLOR,
                linestyle="--",
                marker="s",
                label_y=128.0,
            ),
            Curve(
                "G0_came_ours_uf8_al16_vblk2048",
                "AL16",
                AL16_COLOR,
                marker="D",
                label_y=90.0,
                linewidth=OURS_LINE_WIDTH,
            ),
        ],
    ),
    (
        "c",
        "Adafactor",
        (70, 375),
        [
            Curve(
                "G0_adafactor_hf", "HF FP32", REFERENCE_COLOR,
                linestyle="--", label_y=90.0,
            ),
            Curve(
                "G0_adafactor_ours_al8_vblk256",
                "G0 AL8",
                G0_COLOR,
                linestyle=":",
                label_y=140.0,
            ),
            Curve(
                "G1_adafactor_ours_al8_vblk256",
                "G1 AL8",
                ACCENT,
                label_y=115.0,
                linewidth=OURS_LINE_WIDTH,
            ),
        ],
    ),
    (
        "d",
        "APOLLO",
        (67, 355),
        [
            Curve(
                "G0_apollo_torch", "FP32", REFERENCE_COLOR,
                linestyle="--", label_y=90.0,
            ),
            Curve(
                "G0_apollo_ours_uf8_al8_vblk2048",
                "Ours AL8",
                ACCENT,
                label_y=115.0,
                linewidth=OURS_LINE_WIDTH,
            ),
        ],
    ),
]


# =============================================================================
# Data loading
# =============================================================================

def load_eval_ppl(report: str, run_id: str) -> np.ndarray:
    pattern = re.compile(
        rf"^\*\*{re.escape(run_id)}\s+\([^\n]+\)\*\*\s*$"
        rf".*?^- Eval PPL:\s*(.+?)\s*$",
        flags=re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(report)
    if not match:
        raise ValueError(f"Run not found in TensorBoard report: {run_id}")
    values = np.asarray(
        [float(item.strip()) for item in match.group(1).split("➔")],
        dtype=float,
    )
    if values.size != 20:
        raise ValueError(f"Expected 20 evaluation points for {run_id}, got {values.size}")
    return values


def style_axis(ax, *, row: int, col: int) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(LINE)
        ax.spines[spine].set_linewidth(0.65)
    ax.tick_params(colors=MUTED, labelsize=TICK_LABEL_SIZE, width=0.6)
    ax.grid(axis="y", color=LINE, linewidth=0.48, alpha=0.78)
    ax.set_axisbelow(True)
    if row == 1:
        ax.set_xlabel("Training step", fontsize=AXIS_LABEL_SIZE, color=INK)
    if col == 0:
        ax.set_ylabel("Validation perplexity", fontsize=AXIS_LABEL_SIZE, color=INK)


def panel_title(ax, label: str, title: str) -> None:
    ax.text(
        -0.075,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    ax.text(
        0.02,
        1.08,
        title,
        transform=ax.transAxes,
        fontsize=TITLE_SIZE,
        fontweight="semibold",
        color=INK,
        va="bottom",
    )


def build():
    configure_matplotlib()
    report = REPORT_PATH.read_text(encoding="utf-8")
    steps = np.arange(1, 21, dtype=float) * 1000

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_IN, facecolor=WHITE)
    fig.subplots_adjust(
        left=GRID_LEFT,
        right=GRID_RIGHT,
        bottom=GRID_BOTTOM,
        top=GRID_TOP,
        wspace=GRID_WSPACE,
        hspace=GRID_HSPACE,
    )

    for panel_index, (panel_label, title, ylim, curves) in enumerate(PANELS):
        row, col = divmod(panel_index, 2)
        ax = axes[row, col]
        panel_title(ax, panel_label, title)

        for curve in curves:
            values = load_eval_ppl(report, curve.run_id)
            shown_steps = steps[START_INDEX:]
            shown_values = values[START_INDEX:]
            ax.plot(
                shown_steps,
                shown_values,
                color=curve.color,
                linewidth=curve.linewidth,
                linestyle=curve.linestyle,
                marker=curve.marker,
                markersize=MARKER_SIZE,
                markevery=MARK_EVERY,
                markerfacecolor=(WHITE if curve.marker == "D" else curve.color),
                markeredgecolor=curve.color,
                markeredgewidth=0.6,
                zorder=3,
            )
            final = float(values[-1])
            ax.plot(
                [steps[-1], LABEL_X - 170],
                [final, curve.label_y],
                color=curve.color,
                linewidth=0.55,
                alpha=0.78,
                clip_on=False,
                zorder=2,
            )
            ax.text(
                LABEL_X,
                curve.label_y,
                f"{curve.label}  {final:.2f}",
                fontsize=END_LABEL_SIZE,
                color=curve.color,
                ha="left",
                va="center",
                clip_on=False,
                bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.25, "alpha": 0.92},
            )

        ax.set_xlim(0, X_MAX)
        ax.set_ylim(*ylim)
        ax.set_xticks(
            [0, 5000, 10000, 15000, 20000],
            ["0", "5K", "10K", "15K", "20K"],
        )
        style_axis(ax, row=row, col=col)

    fig.text(
        GRID_LEFT,
        0.035,
        "TinyLlama-1.1B / WikiText-103 / 20K steps / seed 921",
        fontsize=FOOTER_SIZE,
        color=MUTED,
        ha="left",
    )
    fig.text(
        GRID_RIGHT,
        0.035,
        "Single runs; plotted evaluations begin at step 2K",
        fontsize=FOOTER_SIZE,
        color=MUTED,
        ha="right",
    )
    return save_public_figure(fig, OUTPUT_STEM)


if __name__ == "__main__":
    for generated_path in build().values():
        print(generated_path)
