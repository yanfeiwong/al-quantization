"""Generate Figure 2: Adaptive Log-Space encoding intuition."""

from __future__ import annotations

import argparse

import numpy as np

from style import (
    ACCENT,
    ACCENT_DARK,
    ACCENT_SOFT,
    INK,
    LINE,
    MUTED,
    PANEL,
    REVIEW_DIR,
    WHITE,
    add_text,
    configure_matplotlib,
    make_canvas,
    rounded_box,
    save_public_figure,
    save_review_png,
)


def draw(display_levels: int):
    if display_levels < 2:
        raise ValueError("display_levels must be at least 2")

    fig, ax = make_canvas(2.58, ymax=43)
    rounded_box(
        ax,
        3.2,
        15.2,
        11.0,
        17.4,
        facecolor=PANEL,
        edgecolor=LINE,
        linewidth=0.65,
        radius=0.95,
    )
    add_text(ax, 8.7, 29.5, "exact zero", size=6.7, weight="semibold", color=MUTED)
    add_text(ax, 8.7, 23.8, "0", size=15.0, color=ACCENT, weight="bold")
    add_text(ax, 8.7, 18.8, r"$q_i=0$", size=7.4, color=ACCENT_DARK, weight="bold")

    axis_left, axis_right, axis_y = 21.0, 93.0, 24.8
    ax.plot([axis_left, axis_right], [axis_y, axis_y], color=INK, linewidth=0.72)
    ax.plot([axis_left, axis_left], [axis_y - 0.65, axis_y + 0.65], color=INK, linewidth=0.62)
    ax.plot([axis_right, axis_right], [axis_y - 0.65, axis_y + 0.65], color=INK, linewidth=0.62)

    # Three illustrative octaves keep the schematic legible; actual block
    # intervals are data-dependent and may be much wider.
    log_values = np.linspace(0.0, 3.0, display_levels)
    values = 2.0**log_values
    positions = axis_left + (values - values.min()) / (values.max() - values.min()) * (
        axis_right - axis_left
    )
    marker_area = max(3.0, min(11.0, 460.0 / display_levels))
    ax.scatter(
        positions,
        np.full_like(positions, axis_y),
        s=marker_area,
        color=ACCENT,
        edgecolors=WHITE,
        linewidths=0.2,
        zorder=4,
    )

    add_text(ax, axis_left, 28.0, r"$2^{\ell_{\min}}$", size=6.2, color=MUTED)
    add_text(ax, axis_right, 28.0, r"$2^{\ell_{\max}}$", size=6.2, color=MUTED)
    add_text(
        ax,
        (axis_left + axis_right) / 2,
        34.2,
        "nonzero levels in value space",
        size=8.4,
        weight="semibold",
    )
    add_text(
        ax,
        (axis_left + axis_right) / 2,
        20.6,
        r"uniform code steps in $\log_2 x$  $\Rightarrow$  nonuniform levels in value space",
        size=6.6,
        color=MUTED,
    )

    rounded_box(
        ax,
        26.0,
        10.0,
        57.0,
        5.2,
        facecolor=ACCENT_SOFT,
        edgecolor=ACCENT,
        linewidth=0.75,
        radius=0.85,
    )
    add_text(
        ax,
        54.5,
        12.6,
        r"$q_i=0$ ⇔ $x_i=0$",
        size=9.2,
        color=ACCENT,
        weight="bold",
    )
    add_text(
        ax,
        54.5,
        6.8,
        "AL8: 255 nonzero codes   •   AL16: 65,535 nonzero codes",
        size=6.3,
        color=MUTED,
    )
    add_text(
        ax,
        54.5,
        3.7,
        r"each block adapts $[\ell_{\min},\ell_{\max}]$ to its observed nonzero range",
        size=6.35,
        color=MUTED,
        weight="semibold",
    )
    return fig


def build(display_levels: int = 64):
    configure_matplotlib()
    return save_public_figure(draw(display_levels), "fig02_al_encoding")


def build_density_reviews() -> list:
    configure_matplotlib()
    paths = []
    for count in (19, 64, 255):
        path = REVIEW_DIR / "fig02_density" / f"fig02_levels_{count}.png"
        paths.append(save_review_png(draw(count), path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=int,
        default=64,
        help="Number of illustrative markers; this does not change AL code counts.",
    )
    parser.add_argument(
        "--review-densities",
        action="store_true",
        help="Also render 19/64/255-marker PNG comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in build(args.levels).values():
        print(path)
    if args.review_densities:
        for path in build_density_reviews():
            print(path)


if __name__ == "__main__":
    main()
