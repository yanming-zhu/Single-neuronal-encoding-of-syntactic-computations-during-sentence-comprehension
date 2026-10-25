#!/usr/bin/env python3
"""Draw Fig. 1a/b schematic for the Mandarin active/passive Neuropixels study."""

from __future__ import annotations

import os
from pathlib import Path as FilePath

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = FilePath(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports/gz002_gz003_manuscript/figures"


INK = "#222222"
MUTED = "#656565"
BLUE = "#386cb0"
RED = "#c85245"
GREEN = "#5f8d4e"
GOLD = "#c49345"
PURPLE = "#7c65a6"
PANEL_EDGE = "#d9dce1"


def configure_fonts() -> None:
    for path in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Users/yanming/Library/Fonts/simhei.ttf",
    ]:
        if FilePath(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.facecolor"] = "white"


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_frame(ax: plt.Axes, title: str, letter: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.025),
            0.970,
            0.950,
            boxstyle="round,pad=0.008,rounding_size=0.020",
            fc="white",
            ec=PANEL_EDGE,
            lw=1.0,
        )
    )
    ax.text(0.040, 0.945, letter, fontsize=18, weight="bold", va="top", ha="left", color=INK)
    ax.text(0.095, 0.945, title, fontsize=13.5, weight="bold", va="top", ha="left", color=INK)


def label(ax: plt.Axes, x: float, y: float, text: str, size: float = 8.0, color: str = INK, **kwargs) -> None:
    ax.text(x, y, text, fontsize=size, color=color, linespacing=1.15, **kwargs)


def rounded_label(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str = "#f5f6f7",
    edge: str = "#c6cbd2",
    size: float = 7.5,
    color: str = INK,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.020",
            fc=face,
            ec=edge,
            lw=0.9,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, fontsize=size, color=color, va="center", ha="center", linespacing=1.05)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED, lw: float = 1.2) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw))


def curve(ax: plt.Axes, points: list[tuple[float, float]], color: str, lw: float = 1.0, alpha: float = 1.0) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle="round")


def draw_lateral_brain(ax: plt.Axes, cx: float, cy: float, s: float, face: str = "#ead0aa") -> None:
    verts = [
        (cx - 0.30 * s, cy - 0.05 * s),
        (cx - 0.35 * s, cy + 0.13 * s),
        (cx - 0.23 * s, cy + 0.25 * s),
        (cx - 0.03 * s, cy + 0.28 * s),
        (cx + 0.17 * s, cy + 0.23 * s),
        (cx + 0.31 * s, cy + 0.10 * s),
        (cx + 0.30 * s, cy - 0.08 * s),
        (cx + 0.18 * s, cy - 0.20 * s),
        (cx - 0.04 * s, cy - 0.22 * s),
        (cx - 0.24 * s, cy - 0.16 * s),
        (cx - 0.30 * s, cy - 0.05 * s),
    ]
    ax.add_patch(Polygon(verts, closed=True, fc=face, ec="#8e785b", lw=1.1, joinstyle="round"))
    sulci = [
        [(-0.22, 0.12), (-0.09, 0.18), (0.04, 0.17), (0.18, 0.11), (0.25, 0.02)],
        [(-0.25, 0.01), (-0.11, 0.06), (0.07, 0.04), (0.22, -0.03)],
        [(-0.20, -0.10), (-0.03, -0.05), (0.16, -0.10)],
        [(-0.06, 0.23), (-0.04, 0.10), (-0.03, -0.08)],
        [(0.10, 0.20), (0.08, 0.07), (0.06, -0.12)],
    ]
    for pts in sulci:
        curve(ax, [(cx + px * s, cy + py * s) for px, py in pts], "#9a7d5b", lw=0.8, alpha=0.75)


def draw_head_and_probe(ax: plt.Axes) -> None:
    # Stylized lateral head, craniotomy, and probe. No patient-identifying imagery is used.
    head = [
        (0.075, 0.560),
        (0.090, 0.705),
        (0.215, 0.815),
        (0.350, 0.800),
        (0.420, 0.692),
        (0.435, 0.590),
        (0.400, 0.492),
        (0.302, 0.452),
        (0.210, 0.460),
        (0.126, 0.505),
    ]
    ax.add_patch(Polygon(head, closed=True, fc="#f2d2b7", ec="#9a7256", lw=1.2, joinstyle="round"))
    ax.add_patch(Ellipse((0.136, 0.605), 0.040, 0.072, angle=8, fc="#e7b996", ec="#9a7256", lw=0.9))
    ax.add_patch(Rectangle((0.210, 0.466), 0.090, 0.055, angle=-12, fc="#d9b48d", ec="#9a7256", lw=0.9, alpha=0.75))

    # Exposed cortical surface.
    draw_lateral_brain(ax, 0.295, 0.612, 0.48, face="#e8c99c")
    ax.add_patch(Ellipse((0.298, 0.605), 0.210, 0.115, angle=-12, fc="none", ec="#b85c38", lw=1.3))
    ax.scatter([0.302], [0.611], s=48, c=BLUE, edgecolors="white", lw=0.6, zorder=7)
    label(ax, 0.236, 0.392, "left temporal target\n(STG/MTG region)", size=7.5, color=INK, ha="left", va="top")
    ax.plot([0.306, 0.262], [0.602, 0.408], color="#777", lw=0.7)

    # Neuropixels probe shank and headstage.
    base = np.array([0.470, 0.785])
    tip = np.array([0.327, 0.608])
    direction = tip - base
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    width = 0.014
    shank = np.vstack([base + normal * width, base - normal * width, tip - normal * 0.007, tip + normal * 0.007])
    ax.add_patch(Polygon(shank, closed=True, fc="#2e4057", ec="#192435", lw=0.8, zorder=6))
    for t in np.linspace(0.20, 0.86, 9):
        p = base * (1 - t) + tip * t
        ax.add_patch(Circle(tuple(p), 0.005, fc="#d9edf7", ec="none", zorder=7, alpha=0.95))
    connector = np.vstack(
        [
            base + normal * 0.045 - direction * 0.010,
            base - normal * 0.045 - direction * 0.010,
            base - normal * 0.040 + direction * 0.030,
            base + normal * 0.040 + direction * 0.030,
        ]
    )
    ax.add_patch(Polygon(connector, closed=True, fc="#c9ced6", ec="#6f7680", lw=0.8, zorder=8))
    ax.plot([0.487, 0.590, 0.642], [0.795, 0.840, 0.730], color="#5c6470", lw=1.4)

    rounded_label(ax, 0.424, 0.835, 0.180, 0.055, "acute Neuropixels\ninsertion", face="#eef4f9", edge="#b9c9d8", size=7.2)


def draw_acquisition_box(ax: plt.Axes) -> None:
    ax.add_patch(FancyBboxPatch((0.620, 0.515), 0.295, 0.235, boxstyle="round,pad=0.014,rounding_size=0.020", fc="#f8f8f8", ec="#bcc2ca", lw=1.0))
    label(ax, 0.645, 0.714, "recording output", size=8.6, weight="bold", ha="left", va="top")

    # AP spikes.
    ax.plot([0.653, 0.770], [0.657, 0.657], color="#bdbdbd", lw=0.6)
    t = np.linspace(0, 1, 160)
    wave = -np.exp(-((t - 0.38) / 0.045) ** 2) + 0.45 * np.exp(-((t - 0.47) / 0.065) ** 2)
    ax.plot(0.653 + t * 0.115, 0.657 + wave * 0.055, color=BLUE, lw=1.2)
    label(ax, 0.790, 0.662, "AP band\nsingle units", size=7.2, ha="left", va="center")

    # LF/CSD heat map drawn as vector tiles, so the SVG/PDF stays fully editable.
    heat_x, heat_y, heat_w, heat_h = 0.656, 0.545, 0.109, 0.070
    n_cols, n_rows = 5, 4
    tile_vals = np.array(
        [
            [-0.8, 0.2, 0.9, 0.1, -0.7],
            [0.8, -0.4, -0.9, 0.3, 0.7],
            [-0.7, 0.4, 0.8, -0.3, -0.8],
            [0.9, 0.2, -0.7, 0.5, 0.8],
        ]
    )
    cmap = plt.get_cmap("RdBu_r")
    for r in range(n_rows):
        for c in range(n_cols):
            val = float(tile_vals[r, c])
            color = cmap((val + 1) / 2)
            ax.add_patch(
                Rectangle(
                    (heat_x + c * heat_w / n_cols, heat_y + (n_rows - 1 - r) * heat_h / n_rows),
                    heat_w / n_cols,
                    heat_h / n_rows,
                    fc=color,
                    ec="white",
                    lw=0.35,
                    zorder=4,
                )
            )
    ax.add_patch(Rectangle((heat_x, heat_y), heat_w, heat_h, fc="none", ec="#777", lw=0.5, zorder=5))
    label(ax, 0.790, 0.580, "LF band\nLFP/CSD", size=7.2, ha="left", va="center")

    rounded_label(ax, 0.650, 0.458, 0.210, 0.050, "SpikeGLX / Kilosort\ntrial alignment", face="#f3f1f7", edge="#cbc3dd", size=7.0)
    arrow(ax, (0.762, 0.515), (0.762, 0.509), lw=1.0)


def draw_mini_target_map(ax: plt.Axes) -> None:
    ax.add_patch(FancyBboxPatch((0.086, 0.120), 0.360, 0.180, boxstyle="round,pad=0.014,rounding_size=0.018", fc="#fbfbfb", ec="#cbd0d7", lw=0.9))
    draw_lateral_brain(ax, 0.205, 0.197, 0.32, face="#e9ecef")
    colors = [BLUE, GOLD, GREEN, RED, PURPLE]
    pts = [(0.236, 0.196), (0.254, 0.190), (0.270, 0.194), (0.241, 0.178), (0.283, 0.166)]
    for (x, y), c in zip(pts, colors):
        ax.scatter([x], [y], s=28, c=c, edgecolors="white", lw=0.5, zorder=6)
    label(ax, 0.315, 0.236, "entry locations\nprojected near\nleft posterior\nsuperior temporal\ncortex", size=6.6, ha="left", va="top", color=MUTED)
    label(ax, 0.102, 0.282, "cohort localization", size=7.4, weight="bold", ha="left", va="top")


def draw_panel_a(ax: plt.Axes) -> None:
    panel_frame(ax, "Recording mode", "a")
    rounded_label(ax, 0.100, 0.828, 0.170, 0.055, "awake clinical\nprocedure", face="#eef4f9", edge="#b9c9d8", size=7.2)
    draw_head_and_probe(ax)
    draw_acquisition_box(ax)
    draw_mini_target_map(ax)
    arrow(ax, (0.595, 0.727), (0.635, 0.710), lw=1.1)
    label(
        ax,
        0.083,
        0.060,
        "Conceptual layout based on human intraoperative Neuropixels studies;\nno identifying surgical photograph is used.",
        size=6.8,
        color=MUTED,
        ha="left",
        va="bottom",
    )


def token_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    text: str,
    face: str = "#f8f8f8",
    edge: str = "#c8ccd2",
    color: str = INK,
    weight: str = "normal",
) -> None:
    h = 0.056
    ax.add_patch(Rectangle((x, y), w, h, fc=face, ec=edge, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, fontsize=9.2, ha="center", va="center", color=color, weight=weight)


def draw_condition_rows(ax: plt.Axes) -> None:
    label(ax, 0.088, 0.808, "matched Mandarin sentence pairs", size=8.6, weight="bold", ha="left", va="top")
    y1, y2 = 0.711, 0.625
    widths = [0.112, 0.064, 0.112, 0.134]
    x0 = 0.150
    labels_top = ["虫子", "把", "树叶", "啃光了"]
    labels_bottom = ["树叶", "被", "虫子", "啃光了"]
    active_colors = ["#fafafa", "#e8f0fb", "#fafafa", "#fafafa"]
    passive_colors = ["#fafafa", "#f9e8e5", "#fafafa", "#fafafa"]
    x = x0
    for w, txt, fc in zip(widths, labels_top, active_colors):
        token_box(ax, x, y1, w, txt, face=fc, color=BLUE if txt == "把" else INK, weight="bold" if txt == "把" else "normal")
        x += w
    x = x0
    for w, txt, fc in zip(widths, labels_bottom, passive_colors):
        token_box(ax, x, y2, w, txt, face=fc, color=RED if txt == "被" else INK, weight="bold" if txt == "被" else "normal")
        x += w
    label(ax, 0.075, y1 + 0.028, "active", size=7.5, color=BLUE, ha="left", va="center", weight="bold")
    label(ax, 0.075, y2 + 0.028, "passive", size=7.5, color=RED, ha="left", va="center", weight="bold")
    marker_x = x0 + widths[0]
    ax.plot([marker_x, marker_x], [0.592, 0.784], color="#545454", lw=0.9, ls="--")
    label(ax, marker_x + 0.014, 0.598, "function-word onset (t = 0)", size=7.0, color=MUTED, ha="left", va="bottom")

    rounded_label(ax, 0.610, 0.710, 0.120, 0.047, "把 / 将 / 给", face="#e8f0fb", edge="#c0d1ea", size=7.3, color=BLUE)
    rounded_label(ax, 0.610, 0.635, 0.120, 0.047, "被 / 让 / 叫", face="#f9e8e5", edge="#e5beb9", size=7.3, color=RED)
    rounded_label(ax, 0.758, 0.676, 0.140, 0.050, "10 active +\n10 passive / run", face="#f6f6f6", edge="#c7ccd3", size=6.8, color=INK)


def draw_timeline(ax: plt.Axes) -> None:
    label(ax, 0.088, 0.498, "trial timing and alignment", size=8.6, weight="bold", ha="left", va="top")
    x0, x1, y = 0.115, 0.900, 0.410
    ax.plot([x0, x1], [y, y], color=INK, lw=1.3)
    ticks = [
        (0.155, "fixation", "#555", "-"),
        (0.265, "speech onset", "#555", "-"),
        (0.485, "function word\nanalysis t = 0", RED, "--"),
        (0.750, "sentence\nend", "#555", "-"),
        (0.882, "catch press\nif nonsensical", GREEN, "-"),
    ]
    for tx, txt, col, style in ticks:
        ax.plot([tx, tx], [y - 0.046, y + 0.046], color=col, lw=1.2, ls=style)
        label(ax, tx, y - 0.070, txt, size=6.8, color=col, ha="center", va="top")
    # Audio waveform.
    t = np.linspace(0, 1, 450)
    env = np.sin(t * 20 * np.pi) * np.exp(-((t - 0.45) / 0.36) ** 2)
    ax.fill_between(0.265 + t * 0.485, y + env * 0.047, y - env * 0.047, color="#b9c6d9", alpha=0.75, lw=0)
    ax.plot([0.265, 0.750], [y, y], color="#6d7890", lw=0.7)
    ax.text(0.485, y + 0.085, "target onset", fontsize=7.2, color=RED, ha="center", va="bottom", weight="bold")
    arrow(ax, (0.485, y + 0.079), (0.485, y + 0.049), color=RED, lw=0.9)


def draw_task_stream(ax: plt.Axes) -> None:
    label(ax, 0.088, 0.260, "randomized auditory task stream", size=8.6, weight="bold", ha="left", va="top")
    x = 0.098
    y = 0.160
    blocks = [
        ("active/\npassive", BLUE, 0.102),
        ("synonym\ncontrols", GOLD, 0.112),
        ("scrambled\ncatch", GREEN, 0.108),
        ("active/\npassive", BLUE, 0.102),
        ("synonym\ncontrols", GOLD, 0.112),
        ("...", "#9aa0a8", 0.064),
    ]
    for txt, col, w in blocks:
        ax.add_patch(Rectangle((x, y), w, 0.066, fc=col, ec="white", lw=1.0, alpha=0.88))
        ax.text(x + w / 2, y + 0.033, txt, fontsize=6.5, ha="center", va="center", color="white", weight="bold", linespacing=0.95)
        x += w + 0.012
    label(
        ax,
        0.104,
        0.092,
        "Analyses in this work focus on active/passive sentence pairs;\nfiller/control and scrambled trials remain interleaved in the run.",
        size=6.9,
        color=MUTED,
        ha="left",
        va="top",
    )


def draw_speaker(ax: plt.Axes) -> None:
    ax.add_patch(Rectangle((0.798, 0.540), 0.032, 0.046, fc="#3b4654", ec="none"))
    ax.add_patch(Polygon([(0.830, 0.545), (0.868, 0.525), (0.868, 0.601), (0.830, 0.581)], closed=True, fc="#3b4654", ec="none"))
    for r in [0.035, 0.060, 0.085]:
        ax.add_patch(Ellipse((0.870, 0.563), r, r * 1.28, fc="none", ec="#7a8492", lw=0.9))


def draw_panel_b(ax: plt.Axes) -> None:
    panel_frame(ax, "Mandarin active/passive sentence task", "b")
    draw_condition_rows(ax)
    draw_speaker(ax)
    draw_timeline(ax)
    draw_task_stream(ax)


def make_full_figure() -> tuple[FilePath, FilePath, FilePath]:
    configure_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.0, 6.7), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, left=0.025, right=0.985, bottom=0.040, top=0.965, width_ratios=[1.02, 1.0], wspace=0.035)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    clean_axis(ax_a)
    clean_axis(ax_b)
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    png = OUT_DIR / "fig01_recording_task_schematic.png"
    pdf = OUT_DIR / "fig01_recording_task_schematic.pdf"
    svg = OUT_DIR / "fig01_recording_task_schematic.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png, pdf, svg


def make_single_panel(panel: str) -> tuple[FilePath, FilePath, FilePath]:
    configure_fonts()
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    clean_axis(ax)
    if panel == "a":
        draw_panel_a(ax)
        stem = "fig01a_recording_mode"
    elif panel == "b":
        draw_panel_b(ax)
        stem = "fig01b_task_schematic"
    else:
        raise ValueError(panel)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    svg = OUT_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png, pdf, svg


def main() -> None:
    full = make_full_figure()
    panel_a = make_single_panel("a")
    panel_b = make_single_panel("b")
    print(
        {
            "full_png": str(full[0]),
            "full_pdf": str(full[1]),
            "full_svg": str(full[2]),
            "panel_a_png": str(panel_a[0]),
            "panel_a_pdf": str(panel_a[1]),
            "panel_a_svg": str(panel_a[2]),
            "panel_b_png": str(panel_b[0]),
            "panel_b_pdf": str(panel_b[1]),
            "panel_b_svg": str(panel_b[2]),
        }
    )


if __name__ == "__main__":
    main()
