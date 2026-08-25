"""Generate publication Figure 3: optimizer-state heterogeneity.

The statistical unit in panel (a) is a parameter-state tensor at one training
snapshot. Attention and MLP therefore support box + raw-point distributions,
whereas Embedding and LM head are singleton tensor observations and are drawn
as diamonds. The stored traces do not contain individual block samples.

Running this file writes stable PDF, SVG, and PNG assets under ``figures/``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from style import (
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

# Canvas and panel layout ------------------------------------------------------
FIGURE_SIZE_IN = (7.05, 4.35)
GRID_LEFT = 0.135
GRID_RIGHT = 0.985
GRID_BOTTOM = 0.12
GRID_TOP = 0.84
GRID_HEIGHT_RATIOS = (1.55, 0.62)
GRID_HSPACE = 0.62

# Shared linear range axis -----------------------------------------------------
RANGE_X_MIN = 0.0
RANGE_X_MAX = 90.0
RANGE_TICK_STEP = 15.0
TRACE_STEP = 10_000

# Positions are offsets within one state-path row.
TOPOLOGY_OFFSETS = {
    "emb": 0.27,
    "attn": -0.27,
    "mlp": 0.0,
    "lm_head": 0.27,
}
STATE_ROW_BAND_HALF_HEIGHT = 0.42

# Attention / MLP boxes and raw points ----------------------------------------
BOX_WIDTH = 0.25
BOX_FILL_ALPHA = 0.34
BOX_EDGE_WIDTH = 0.72
BOX_MEDIAN_WIDTH = 0.86
BOX_WHISKER_WIDTH = 0.62

RAW_POINT_SIZE = 3.8       # scatter marker area in pt^2, not diameter
RAW_POINT_ALPHA = 0.28
RAW_POINT_JITTER = 0.07    # y-axis row units
RANDOM_JITTER_SEED = 921

# Embedding / LM-head singleton observations ---------------------------------
# These are also scatter marker areas in pt^2. Adjust independently.
EMBEDDING_MARKER_SIZE = 5.0
LM_HEAD_MARKER_SIZE = 5.0
SINGLETON_EDGE_WIDTH = 0.70

# Restrained grayscale + one-blue-accent visual system ------------------------
EMBEDDING_COLOR = "#535D6D"
ATTENTION_COLOR = "#2D5FC4"
MLP_COLOR = "#8996A8"
LM_HEAD_COLOR = "#657083"
ALTERNATE_ROW_COLOR = "#F6F8FB"
ZERO_CURVE_COLOR = "#356AE6"
ZERO_LABEL_COLOR = "#2855BC"

# Text and legend --------------------------------------------------------------
PANEL_LABEL_SIZE = 8.2
PANEL_TITLE_SIZE = 7.8
AXIS_LABEL_SIZE = 6.0
TICK_LABEL_SIZE = 5.8
LEGEND_SIZE = 5.55
NOTE_SIZE = 5.15
ANNOTATION_SIZE = 5.3

PANEL_A_TITLE = "Optimizer states occupy distinct numerical regimes"
PANEL_B_TITLE = "Exact zeros persist in the embedding second moment"

# Figure-level text positions -------------------------------------------------
PANEL_LABEL_X = 0.115
PANEL_TITLE_X = 0.14
PANEL_A_TITLE_Y = 0.952
PANEL_B_TITLE_Y = 0.355
TOP_ENCODING_NOTE_Y = 0.912
FOOTER_Y = 0.028


# =============================================================================
# Data and state-path definitions
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
TRACE_CSV = DATA_DIR / "fig03_trace_components.csv"
OUTPUT_STEM = "fig03_state_heterogeneity"

ADAMW = "G0_adamw_torch_x1.0_bs4_seq512"
ADAFACTOR = "G0_adafactor_hf_x1.0_bs4_seq512"
CAME = "G0_came_torch_x0.1_bs4_seq512"

STATE_PATHS = [
    {
        "label": r"AdamW  dense $V$",
        "optimizer": ADAMW,
        "family": "second_moment",
        "component": "full",
    },
    {
        "label": r"Adafactor  $V_r$",
        "optimizer": ADAFACTOR,
        "family": "second_moment",
        "component": "row",
    },
    {
        "label": r"Adafactor  $V_c$",
        "optimizer": ADAFACTOR,
        "family": "second_moment",
        "component": "col",
    },
    {
        "label": r"CAME  $V_r$",
        "optimizer": CAME,
        "family": "second_moment",
        "component": "row",
    },
    {
        "label": r"CAME  $V_c$",
        "optimizer": CAME,
        "family": "second_moment",
        "component": "col",
    },
    {
        "label": r"CAME  $C_r$",
        "optimizer": CAME,
        "family": "confidence",
        "component": "row",
    },
    {
        "label": r"CAME  $C_c$",
        "optimizer": CAME,
        "family": "confidence",
        "component": "col",
    },
]

TOPOLOGY_COLORS = {
    "emb": EMBEDDING_COLOR,
    "attn": ATTENTION_COLOR,
    "mlp": MLP_COLOR,
    "lm_head": LM_HEAD_COLOR,
}


# =============================================================================
# Data loading
# =============================================================================

def load_rows() -> list[dict]:
    rows: list[dict] = []
    with TRACE_CSV.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "step": int(raw["step"]),
                    "zero_fraction": float(raw["zero_fraction"]),
                    "block_logrange_median": float(raw["block_logrange_median"]),
                }
            )
    return rows


def values_for(rows: list[dict], state_path: dict, tensor_type: str) -> np.ndarray:
    return np.asarray(
        [
            row["block_logrange_median"]
            for row in rows
            if row["optimizer"] == state_path["optimizer"]
            and row["state_family"] == state_path["family"]
            and row["state_component"] == state_path["component"]
            and row["tensor_type"] == tensor_type
            and row["step"] == TRACE_STEP
        ],
        dtype=float,
    )


def embedding_zero_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["optimizer"] == ADAMW
        and row["state_family"] == "second_moment"
        and row["state_component"] == "full"
        and row["tensor_type"] == "emb"
    ]
    selected.sort(key=lambda row: row["step"])
    return (
        np.asarray([row["step"] for row in selected], dtype=float),
        np.asarray([100.0 * row["zero_fraction"] for row in selected], dtype=float),
    )


# =============================================================================
# Drawing helpers
# =============================================================================

def style_axis(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(LINE)
        ax.spines[spine].set_linewidth(0.65)
    ax.tick_params(colors=MUTED, labelsize=TICK_LABEL_SIZE, width=0.6)


def draw_state_ranges(ax, rows: list[dict]) -> None:
    base_positions = np.arange(1, len(STATE_PATHS) + 1, dtype=float)
    rng = np.random.default_rng(RANDOM_JITTER_SEED)

    for index, base in enumerate(base_positions):
        if index % 2 == 0:
            ax.axhspan(
                base - STATE_ROW_BAND_HALF_HEIGHT,
                base + STATE_ROW_BAND_HALF_HEIGHT,
                color=ALTERNATE_ROW_COLOR,
                zorder=0,
            )

    for tensor_type in ("attn", "mlp"):
        positions = base_positions + TOPOLOGY_OFFSETS[tensor_type]
        distributions = [
            values_for(rows, state_path, tensor_type)
            for state_path in STATE_PATHS
        ]
        boxes = ax.boxplot(
            distributions,
            vert=False,
            positions=positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": BOX_MEDIAN_WIDTH},
            whiskerprops={"color": MUTED, "linewidth": BOX_WHISKER_WIDTH},
            capprops={"color": MUTED, "linewidth": BOX_WHISKER_WIDTH},
            boxprops={
                "edgecolor": TOPOLOGY_COLORS[tensor_type],
                "linewidth": BOX_EDGE_WIDTH,
            },
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(TOPOLOGY_COLORS[tensor_type])
            patch.set_alpha(BOX_FILL_ALPHA)
        for position, values in zip(positions, distributions):
            jitter = rng.uniform(-RAW_POINT_JITTER, RAW_POINT_JITTER, values.size)
            ax.scatter(
                values,
                position + jitter,
                s=RAW_POINT_SIZE,
                color=TOPOLOGY_COLORS[tensor_type],
                alpha=RAW_POINT_ALPHA,
                linewidths=0,
                zorder=2,
            )

    singleton_values: dict[str, np.ndarray] = {}
    for tensor_type in ("emb", "lm_head"):
        values = np.asarray(
            [
                float(values_for(rows, state_path, tensor_type)[0])
                for state_path in STATE_PATHS
            ],
            dtype=float,
        )
        singleton_values[tensor_type] = values
        marker_size = (
            EMBEDDING_MARKER_SIZE
            if tensor_type == "emb"
            else LM_HEAD_MARKER_SIZE
        )
        marker_kwargs = {
            "facecolors": (
                TOPOLOGY_COLORS[tensor_type] if tensor_type == "emb" else WHITE
            ),
            "edgecolors": TOPOLOGY_COLORS[tensor_type],
            "linewidths": SINGLETON_EDGE_WIDTH,
        }
        ax.scatter(
            values,
            base_positions + TOPOLOGY_OFFSETS[tensor_type],
            s=marker_size,
            marker="D",
            zorder=4,
            **marker_kwargs,
        )

    embedding_values = singleton_values["emb"]
    maximum = int(np.argmax(embedding_values))
    minimum = int(np.argmin(embedding_values))
    ax.annotate(
        f"{embedding_values[maximum]:.1f}",
        (
            embedding_values[maximum],
            base_positions[maximum] + TOPOLOGY_OFFSETS["emb"],
        ),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        fontsize=ANNOTATION_SIZE,
        color=ATTENTION_COLOR,
    )
    ax.annotate(
        f"{embedding_values[minimum]:.2f}",
        (
            embedding_values[minimum],
            base_positions[minimum] + TOPOLOGY_OFFSETS["emb"],
        ),
        xytext=(11, -4),
        textcoords="offset points",
        ha="left",
        fontsize=ANNOTATION_SIZE,
        color=ATTENTION_COLOR,
    )

    ax.set_xlim(RANGE_X_MIN, RANGE_X_MAX)
    ax.set_xticks(
        np.arange(
            RANGE_X_MIN,
            RANGE_X_MAX + 0.5 * RANGE_TICK_STEP,
            RANGE_TICK_STEP,
        )
    )
    ax.set_ylim(len(STATE_PATHS) + 0.75, 0.5)
    ax.set_yticks(
        base_positions,
        [state_path["label"] for state_path in STATE_PATHS],
    )
    ax.tick_params(axis="y", length=0, labelsize=TICK_LABEL_SIZE, pad=5)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="x", color=LINE, linewidth=0.5, alpha=0.72)
    ax.set_axisbelow(True)
    ax.set_xlabel(
        r"Tensor-wise median block $\log_2$ range (bits)",
        fontsize=AXIS_LABEL_SIZE,
        color=INK,
    )
    style_axis(ax)

    handles = [
        Line2D(
            [], [], marker="s", linestyle="none", markersize=5.2,
            markerfacecolor=ATTENTION_COLOR,
            markeredgecolor=ATTENTION_COLOR,
            alpha=0.55,
            label="Attention",
        ),
        Line2D(
            [], [], marker="s", linestyle="none", markersize=5.2,
            markerfacecolor=MLP_COLOR,
            markeredgecolor=MLP_COLOR,
            alpha=0.65,
            label="MLP",
        ),
        Line2D(
            [], [], marker="D", linestyle="none",
            markersize=np.sqrt(EMBEDDING_MARKER_SIZE),
            markerfacecolor=EMBEDDING_COLOR,
            markeredgecolor=EMBEDDING_COLOR,
            markeredgewidth=SINGLETON_EDGE_WIDTH,
            label="Embedding",
        ),
        Line2D(
            [], [], marker="D", linestyle="none",
            markersize=np.sqrt(LM_HEAD_MARKER_SIZE),
            markerfacecolor=WHITE,
            markeredgecolor=LM_HEAD_COLOR,
            markeredgewidth=SINGLETON_EDGE_WIDTH,
            label="LM head",
        ),
    ]
    ax.legend(
        handles=handles,
        ncol=4,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        frameon=False,
        fontsize=LEGEND_SIZE,
        handletextpad=0.35,
        columnspacing=1.05,
        borderaxespad=0,
    )


def draw_zero_trajectory(ax, rows: list[dict]) -> None:
    steps, zeros = embedding_zero_series(rows)
    ax.plot(steps, zeros, color=ZERO_CURVE_COLOR, linewidth=1.55, zorder=2)
    ax.scatter(
        steps,
        zeros,
        s=15,
        color=ZERO_CURVE_COLOR,
        edgecolors=WHITE,
        linewidths=0.35,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlim(42, 12_500)
    ax.set_ylim(-4, 100)
    ax.set_xticks(
        [50, 100, 500, 1000, 2000, 10_000],
        ["50", "100", "500", "1K", "2K", "10K"],
    )
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Training step", fontsize=AXIS_LABEL_SIZE, color=INK)
    ax.set_ylabel("Exact zeros (%)", fontsize=AXIS_LABEL_SIZE, color=INK)
    ax.grid(axis="y", color=LINE, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    style_axis(ax)

    ax.annotate(
        f"{zeros[0]:.1f}%",
        (steps[0], zeros[0]),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE,
        color=ZERO_LABEL_COLOR,
    )
    ax.annotate(
        f"{zeros[-1]:.1f}%",
        (steps[-1], zeros[-1]),
        xytext=(-4, 12),
        textcoords="offset points",
        ha="right",
        fontsize=ANNOTATION_SIZE,
        color=ZERO_LABEL_COLOR,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.5, "alpha": 0.88},
    )
    ax.text(
        0.04,
        0.16,
        "Other traced tensor types: 0%",
        transform=ax.transAxes,
        fontsize=NOTE_SIZE,
        color=MUTED,
        va="top",
    )


# =============================================================================
# Figure assembly
# =============================================================================

def build() -> dict[str, Path]:
    rows = load_rows()
    configure_matplotlib()
    fig = plt.figure(figsize=FIGURE_SIZE_IN, facecolor=WHITE)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=GRID_HEIGHT_RATIOS,
        left=GRID_LEFT,
        right=GRID_RIGHT,
        bottom=GRID_BOTTOM,
        top=GRID_TOP,
        hspace=GRID_HSPACE,
    )
    ax_ranges = fig.add_subplot(outer[0, 0])
    ax_zeros = fig.add_subplot(outer[1, 0])

    draw_state_ranges(ax_ranges, rows)
    draw_zero_trajectory(ax_zeros, rows)

    fig.text(
        PANEL_LABEL_X,
        PANEL_A_TITLE_Y,
        "a",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        PANEL_TITLE_X,
        PANEL_A_TITLE_Y,
        PANEL_A_TITLE,
        fontsize=PANEL_TITLE_SIZE,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    fig.text(
        GRID_RIGHT,
        TOP_ENCODING_NOTE_Y,
        "boxes + raw points: multiple tensors    diamonds: one tensor",
        fontsize=NOTE_SIZE,
        color=MUTED,
        ha="right",
        va="top",
    )
    fig.text(
        PANEL_LABEL_X,
        PANEL_B_TITLE_Y,
        "b",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        PANEL_TITLE_X,
        PANEL_B_TITLE_Y,
        PANEL_B_TITLE,
        fontsize=PANEL_TITLE_SIZE,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    fig.text(
        GRID_LEFT,
        FOOTER_Y,
        "Selected 2D state paths; each range observation is one parameter-state tensor at step 10K",
        ha="left",
        va="bottom",
        fontsize=NOTE_SIZE,
        color=MUTED,
    )
    fig.text(
        GRID_RIGHT,
        FOOTER_Y,
        "TinyLlama-1.1B / WikiText-103 / seed 921",
        ha="right",
        va="bottom",
        fontsize=NOTE_SIZE,
        color=MUTED,
    )
    return save_public_figure(fig, OUTPUT_STEM)


def main() -> None:
    for output_path in build().values():
        print(output_path)


if __name__ == "__main__":
    main()
