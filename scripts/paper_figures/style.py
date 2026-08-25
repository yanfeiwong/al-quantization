"""Shared visual system and output paths for paper figures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adafactor8bit-mpl")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PAPER_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PAPER_ROOT / "figures"
REVIEW_DIR = PAPER_ROOT / "tmp" / "figure_reviews"

DOUBLE_COLUMN_WIDTH_IN = 7.05

INK = "#252A33"
MUTED = "#5F6978"
FAINT = "#93A0B2"
LINE = "#D4DAE3"
PANEL = "#F8F9FB"
ACCENT = "#356AE6"
ACCENT_DARK = "#2855BC"
ACCENT_SOFT = "#EAF0FE"
WHITE = "#FFFFFF"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def make_canvas(height_in: float, *, ymax: float = 46.0):
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH_IN, height_in),
        facecolor=WHITE,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    ax.set_facecolor(WHITE)
    return fig, ax


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str = WHITE,
    edgecolor: str = LINE,
    linewidth: float = 0.75,
    radius: float = 0.75,
    linestyle: str | tuple = "-",
    zorder: int = 1,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_text(
    ax,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 7.0,
    color: str = INK,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    rotation: float = 0,
    zorder: int = 5,
):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=zorder,
    )


def save_public_figure(fig, stem: str) -> dict[str, Path]:
    """Save stable PDF, SVG, and PNG assets together under ``figures/``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": FIGURES_DIR / f"{stem}.pdf",
        "svg": FIGURES_DIR / f"{stem}.svg",
        "png": FIGURES_DIR / f"{stem}.png",
    }
    fig.savefig(paths["pdf"], facecolor=WHITE)
    fig.savefig(paths["svg"], facecolor=WHITE)
    fig.savefig(paths["png"], facecolor=WHITE, dpi=240)
    plt.close(fig)
    return paths


def save_review_png(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=WHITE, dpi=240)
    plt.close(fig)
    return path
