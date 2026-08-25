"""Generate Figure 1: optimizer-state topology landscape."""

from __future__ import annotations

from style import (
    ACCENT,
    ACCENT_SOFT,
    FAINT,
    INK,
    LINE,
    MUTED,
    PANEL,
    WHITE,
    add_text,
    configure_matplotlib,
    make_canvas,
    rounded_box,
    save_public_figure,
)

from matplotlib.patches import FancyArrowPatch


def state_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    rotation: float = 0,
    size: float = 8.2,
):
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        facecolor=ACCENT_SOFT,
        edgecolor=ACCENT,
        linewidth=0.8,
        radius=min(0.78, h / 3),
        zorder=2,
    )
    add_text(
        ax,
        x + w / 2,
        y + h / 2,
        label,
        size=size,
        color=ACCENT,
        weight="bold",
        rotation=rotation,
    )


def panel(ax, x: float, title: str, subtitle: str):
    y, w, h = 8.4, 21.2, 33.7
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        facecolor=PANEL,
        edgecolor=LINE,
        linewidth=0.65,
        radius=1.1,
    )
    add_text(ax, x + w / 2, 39.4, title, size=9.0, weight="bold")
    add_text(ax, x + w / 2, 36.9, subtitle, size=6.3, color=MUTED)
    return x, y, w, h


def build():
    configure_matplotlib()
    fig, ax = make_canvas(2.68)

    x0, gap = 2.2, 1.45
    xs = [x0 + i * (21.2 + gap) for i in range(4)]

    x, _, w, _ = panel(ax, xs[0], "AdamW", "dense")
    state_box(ax, x + 3.25, 26.0, 14.7, 7.1, r"$M_t$", size=10.0)
    state_box(ax, x + 3.25, 15.8, 14.7, 7.1, r"$V_t$", size=10.0)
    add_text(ax, x + w / 2, 12.3, "parameter-shaped states", size=6.05, color=MUTED)

    x, _, w, _ = panel(ax, xs[1], "Adafactor", "factored")
    matrix_x, matrix_y, matrix_w, matrix_h = x + 4.55, 20.25, 11.65, 11.55
    rounded_box(
        ax,
        matrix_x,
        matrix_y,
        matrix_w,
        matrix_h,
        facecolor=WHITE,
        edgecolor=FAINT,
        linewidth=0.65,
        radius=0.55,
        linestyle=(0, (3, 3)),
    )
    add_text(
        ax,
        matrix_x + matrix_w / 2,
        matrix_y + matrix_h / 2,
        r"reconstructed $\widehat{V}_t$",
        size=6.0,
        color=MUTED,
    )
    state_box(ax, matrix_x, 16.05, matrix_w, 2.35, r"$V_{r,t}$", size=6.7)
    state_box(
        ax,
        matrix_x + matrix_w + 1.25,
        matrix_y,
        2.05,
        matrix_h,
        r"$V_{c,t}$",
        rotation=90,
        size=6.7,
    )
    add_text(ax, x + w / 2, 12.3, "row / column marginals", size=6.05, color=MUTED)

    x, _, w, _ = panel(ax, xs[2], "CAME", "confidence + factored")
    state_box(ax, x + 3.3, 27.3, 14.6, 5.9, r"$M_t$", size=9.2)
    strip_x, strip_w, strip_h = x + 5.15, 10.9, 2.15
    for y, label_value in zip(
        [23.1, 19.8, 16.5, 13.2],
        [r"$V_{r,t}$", r"$V_{c,t}$", r"$C_{r,t}$", r"$C_{c,t}$"],
    ):
        state_box(ax, strip_x, y, strip_w, strip_h, label_value, size=6.4)
    add_text(ax, x + w / 2, 10.8, r"dense $M$ + factored $V$ and $C$", size=5.9, color=MUTED)

    x, _, w, _ = panel(ax, xs[3], "APOLLO", "projected")
    add_text(ax, x + 4.8, 32.6, r"$G_t$", size=7.3, weight="bold")
    arrow = FancyArrowPatch(
        (x + 6.4, 32.6),
        (x + 13.1, 32.6),
        arrowstyle="-|>",
        mutation_scale=6.5,
        linewidth=0.7,
        color=MUTED,
        zorder=3,
    )
    ax.add_patch(arrow)
    add_text(ax, x + 9.75, 34.2, r"projection $P_t$", size=5.55, color=MUTED)
    add_text(ax, x + 14.65, 32.6, r"$R_t$", size=7.0, weight="bold")
    state_box(ax, x + 5.45, 23.8, 10.3, 5.5, r"$M_t^{R}$", size=8.0)
    state_box(ax, x + 5.45, 16.0, 10.3, 5.5, r"$V_t^{R}$", size=8.0)
    add_text(
        ax,
        x + w / 2,
        12.9,
        r"projected moments $\rightarrow$ scale",
        size=5.75,
        color=MUTED,
    )
    add_text(
        ax,
        x + w / 2,
        11.15,
        r"on original $G_t$",
        size=5.75,
        color=MUTED,
    )

    ax.plot([9, 91], [5.3, 5.3], color=LINE, linewidth=0.6)
    add_text(
        ax,
        50,
        2.75,
        "representation  ×  topology  ×  update semantics",
        size=7.0,
        color=MUTED,
        weight="semibold",
    )
    return save_public_figure(fig, "fig01_state_landscape")


def main() -> None:
    for path in build().values():
        print(path)


if __name__ == "__main__":
    main()
