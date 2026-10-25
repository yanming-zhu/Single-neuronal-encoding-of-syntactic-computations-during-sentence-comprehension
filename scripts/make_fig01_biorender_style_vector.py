#!/usr/bin/env python3
"""Vector remake of the BioRender-style Fig. 1 draft."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports/gz002_gz003_manuscript/figures"

INK = "#1f1f1f"
MUTED = "#5f646b"
PANEL_EDGE = "#bcc3cc"
ACTIVE = "#386CB0"
PASSIVE = "#C85245"
CATCH = "#5F8D4E"
CONTROL = "#C49345"


def configure_fonts() -> None:
    for path in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Users/yanming/Library/Fonts/simhei.ttf",
    ]:
        if Path(path).exists():
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


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.010, 0.020),
            0.980,
            0.960,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            fc="white",
            ec=PANEL_EDGE,
            lw=1.0,
        )
    )
    ax.text(0.045, 0.940, letter, fontsize=21, weight="bold", ha="left", va="top", color="black")
    ax.text(0.116, 0.936, title, fontsize=12.8, ha="left", va="top", color="black")


def text(ax: plt.Axes, x: float, y: float, s: str, size: float = 8.0, color: str = INK, **kwargs) -> None:
    ax.text(x, y, s, fontsize=size, color=color, linespacing=1.12, **kwargs)


def path_patch(points, codes, **kwargs) -> PathPatch:
    return PathPatch(MplPath(points, codes), **kwargs)


def curve(ax: plt.Axes, pts: list[tuple[float, float]], color: str, lw: float = 1.0, alpha: float = 1.0) -> None:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle="round")


def arrow(ax: plt.Axes, start, end, color: str = MUTED, lw: float = 1.0, ms: float = 12) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw))


def rounded_box(ax: plt.Axes, x: float, y: float, w: float, h: float, fc="#fbfbfb", ec=PANEL_EDGE, lw=1.0) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            fc=fc,
            ec=ec,
            lw=lw,
        )
    )


def draw_head_profile(ax: plt.Axes) -> None:
    # Soft human lateral head silhouette, rebuilt with editable paths.
    pts = [
        (0.094, 0.530),
        (0.076, 0.574),
        (0.091, 0.626),
        (0.137, 0.686),
        (0.214, 0.731),
        (0.320, 0.747),
        (0.439, 0.709),
        (0.482, 0.620),
        (0.466, 0.523),
        (0.413, 0.446),
        (0.331, 0.404),
        (0.245, 0.402),
        (0.193, 0.426),
        (0.156, 0.454),
        (0.114, 0.459),
        (0.092, 0.473),
        (0.113, 0.488),
        (0.092, 0.501),
        (0.102, 0.515),
        (0.094, 0.530),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * (len(pts) - 2) + [MplPath.CLOSEPOLY]
    ax.add_patch(path_patch(pts, codes, fc="#f7dec6", ec="#9d8065", lw=0.85, alpha=0.88))
    # Face details.
    curve(ax, [(0.100, 0.522), (0.074, 0.513), (0.093, 0.501)], "#9d8065", 0.65, 0.8)
    curve(ax, [(0.084, 0.575), (0.107, 0.579), (0.115, 0.568)], "#8f7968", 0.65, 0.8)
    curve(ax, [(0.092, 0.471), (0.125, 0.468), (0.143, 0.482)], "#9d8065", 0.55, 0.65)
    curve(ax, [(0.151, 0.545), (0.125, 0.539), (0.111, 0.525)], "#9d8065", 0.50, 0.60)


def brain_outline(ax: plt.Axes, cx: float, cy: float, s: float, fc="#efc5c1", ec="#9e6f6a", alpha: float = 0.95) -> None:
    verts = [
        (cx - 0.245 * s, cy - 0.020 * s),
        (cx - 0.290 * s, cy + 0.075 * s),
        (cx - 0.218 * s, cy + 0.175 * s),
        (cx - 0.105 * s, cy + 0.202 * s),
        (cx + 0.010 * s, cy + 0.235 * s),
        (cx + 0.170 * s, cy + 0.214 * s),
        (cx + 0.242 * s, cy + 0.112 * s),
        (cx + 0.302 * s, cy + 0.014 * s),
        (cx + 0.216 * s, cy - 0.102 * s),
        (cx + 0.085 * s, cy - 0.185 * s),
        (cx - 0.055 * s, cy - 0.175 * s),
        (cx - 0.194 * s, cy - 0.140 * s),
        (cx - 0.260 * s, cy - 0.060 * s),
        (cx - 0.245 * s, cy - 0.020 * s),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    ax.add_patch(path_patch(verts, codes, fc=fc, ec=ec, lw=0.75, alpha=alpha))
    # Subtle overlapping lobes soften the outline and make it closer to BioRender's brain icons.
    lobe_specs = [
        (-0.155, 0.090, 0.165, 0.105, 12),
        (-0.020, 0.115, 0.175, 0.108, -8),
        (0.122, 0.078, 0.168, 0.102, -18),
        (-0.095, -0.020, 0.178, 0.106, -12),
        (0.060, -0.030, 0.190, 0.112, 4),
        (0.160, -0.080, 0.120, 0.075, -14),
    ]
    for dx, dy, ww, hh, ang in lobe_specs:
        ax.add_patch(Ellipse((cx + dx * s, cy + dy * s), ww * s, hh * s, angle=ang, fc=fc, ec="none", alpha=alpha * 0.25))
    gyri = [
        [(-0.20, 0.06), (-0.16, 0.12), (-0.10, 0.10), (-0.07, 0.16), (-0.01, 0.12)],
        [(-0.14, 0.02), (-0.07, 0.05), (0.00, 0.02), (0.08, 0.07), (0.17, 0.02)],
        [(-0.18, -0.04), (-0.10, -0.02), (-0.02, -0.06), (0.08, -0.03), (0.18, -0.08)],
        [(-0.05, 0.18), (-0.03, 0.09), (-0.06, 0.01), (-0.04, -0.10)],
        [(0.08, 0.15), (0.09, 0.07), (0.06, -0.02), (0.08, -0.12)],
        [(0.16, 0.11), (0.13, 0.03), (0.18, -0.02)],
        [(-0.21, 0.11), (-0.15, 0.17), (-0.08, 0.15), (-0.02, 0.20)],
    ]
    for g in gyri:
        curve(ax, [(cx + x * s, cy + y * s) for x, y in g], "#b9827e", lw=0.55, alpha=0.70)


def draw_brain_and_probe(ax: plt.Axes) -> None:
    draw_head_profile(ax)
    brain_outline(ax, 0.302, 0.635, 0.74)
    # Cerebellum / brainstem.
    ax.add_patch(Ellipse((0.405, 0.535), 0.138, 0.076, angle=-12, fc="#ddb0a8", ec="#9e6f6a", lw=0.55, alpha=0.95))
    for i in range(5):
        curve(ax, [(0.360 + i * 0.018, 0.555), (0.392 + i * 0.010, 0.535), (0.426 + i * 0.004, 0.548)], "#a67873", 0.45, 0.55)
    ax.add_patch(Polygon([(0.386, 0.500), (0.423, 0.505), (0.428, 0.362), (0.390, 0.362)], closed=True, fc="#e1b5a8", ec="none", alpha=0.45))
    # Highlight temporal target.
    ax.add_patch(Ellipse((0.344, 0.566), 0.205, 0.083, angle=-10, fc="#8fb4e6", ec=ACTIVE, lw=0.95, alpha=0.52))
    text(ax, 0.158, 0.421, "left temporal target\n(STG/MTG region)", size=7.8, ha="left", va="top")
    arrow(ax, (0.252, 0.453), (0.305, 0.538), color="#333", lw=0.8, ms=9)

    # Neuropixels-like probe.
    base = np.array([0.490, 0.805])
    tip = np.array([0.362, 0.548])
    direction = (tip - base) / np.linalg.norm(tip - base)
    normal = np.array([-direction[1], direction[0]])
    shank_w = 0.010
    shank = np.vstack([base + normal * shank_w, base - normal * shank_w, tip - normal * 0.004, tip + normal * 0.004])
    ax.add_patch(Polygon(shank, closed=True, fc="#203040", ec="#0f1822", lw=0.65, zorder=10))
    for t in np.linspace(0.12, 0.87, 11):
        p = base * (1 - t) + tip * t
        ax.add_patch(Circle(tuple(p), 0.0042, fc="#a9d4ec", ec="none", zorder=11))
    conn_center = base - direction * 0.022
    connector = np.vstack(
        [
            conn_center + normal * 0.035 - direction * 0.018,
            conn_center - normal * 0.035 - direction * 0.018,
            conn_center - normal * 0.030 + direction * 0.027,
            conn_center + normal * 0.030 + direction * 0.027,
        ]
    )
    ax.add_patch(Polygon(connector, closed=True, fc="#cbd0d8", ec="#667180", lw=0.75, zorder=12))
    # Cable to output box.
    cable = [(0.500, 0.825), (0.548, 0.878), (0.635, 0.865), (0.680, 0.790)]
    curve(ax, cable, "#293442", lw=1.4)
    arrow(ax, (0.680, 0.790), (0.695, 0.730), color="#293442", lw=1.3, ms=12)


def draw_vector_heat(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    vals = np.array(
        [
            [-0.8, 0.1, 0.9, 0.2, -0.7],
            [0.8, -0.3, -0.9, 0.2, 0.7],
            [-0.6, 0.2, 0.8, -0.2, -0.8],
            [0.9, 0.1, -0.7, 0.4, 0.8],
        ]
    )
    cmap = plt.get_cmap("RdBu_r")
    nr, nc = vals.shape
    for r in range(nr):
        for c in range(nc):
            ax.add_patch(
                Rectangle(
                    (x + c * w / nc, y + (nr - 1 - r) * h / nr),
                    w / nc,
                    h / nr,
                    fc=cmap((float(vals[r, c]) + 1) / 2),
                    ec="white",
                    lw=0.25,
                )
            )
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#777", lw=0.4))


def draw_recording_box(ax: plt.Axes) -> None:
    rounded_box(ax, 0.625, 0.575, 0.260, 0.240, fc="#fbfbfb", ec="#aab2bd", lw=0.95)
    text(ax, 0.660, 0.790, "recording output", size=8.0, ha="left", va="top")
    # Spike waveform.
    ax.plot([0.650, 0.775], [0.724, 0.724], color=ACTIVE, lw=1.25)
    t = np.linspace(0, 1, 180)
    spike = -1.45 * np.exp(-((t - 0.38) / 0.052) ** 2) + 0.82 * np.exp(-((t - 0.50) / 0.070) ** 2)
    ax.plot(0.650 + t * 0.125, 0.724 + spike * 0.036, color=ACTIVE, lw=1.10)
    text(ax, 0.800, 0.723, "AP band /\nsingle units", size=6.8, ha="left", va="center")
    draw_vector_heat(ax, 0.650, 0.594, 0.125, 0.080)
    text(ax, 0.800, 0.634, "LF band /\nLFP-CSD", size=6.8, ha="left", va="center")
    arrow(ax, (0.760, 0.574), (0.760, 0.522), color="#4a4a4a", lw=1.0, ms=10)
    rounded_box(ax, 0.636, 0.472, 0.246, 0.060, fc="#f5f1fa", ec="#c3b7dc", lw=0.9)
    text(ax, 0.759, 0.504, "SpikeGLX / Kilosort\ntrial alignment", size=7.1, ha="center", va="center")


def draw_cohort_inset(ax: plt.Axes) -> None:
    rounded_box(ax, 0.085, 0.118, 0.650, 0.235, fc="white", ec="#aeb7c2", lw=0.85)
    text(ax, 0.120, 0.319, "cohort entry locations", size=8.2, ha="left", va="top")
    brain_outline(ax, 0.235, 0.205, 0.46, fc="#edf0f3", ec="#9a9a9a", alpha=1)
    ax.add_patch(Ellipse((0.280, 0.176), 0.070, 0.036, angle=-10, fc="#d8dde3", ec="#aaa", lw=0.4))
    pts = [(0.272, 0.205), (0.296, 0.198), (0.313, 0.213), (0.282, 0.185), (0.348, 0.176)]
    cols = [ACTIVE, "#e08c26", "#4f9343", "#d84c42", "#8b62b5"]
    labs = ["gz001", "gz002", "gz003", "gz004", "gz005"]
    for (x, y), c in zip(pts, cols):
        ax.scatter([x], [y], s=28, c=c, edgecolors="white", lw=0.5, zorder=8)
    text(ax, 0.420, 0.258, "entry locations\nprojected near\nleft posterior\nsuperior temporal\ncortex", size=6.8, color=MUTED, ha="left", va="top")
    for i, (lab, c) in enumerate(zip(labs, cols)):
        yy = 0.270 - i * 0.032
        ax.scatter([0.625], [yy], s=28, c=c, edgecolors="#555", lw=0.4)
        text(ax, 0.648, yy, lab, size=7.0, ha="left", va="center")


def draw_panel_a(ax: plt.Axes) -> None:
    panel(ax, "a", "Recording mode")
    draw_brain_and_probe(ax)
    draw_recording_box(ax)
    draw_cohort_inset(ax)
    text(
        ax,
        0.090,
        0.083,
        "Conceptual layout based on human intraoperative Neuropixels studies;\nno identifying surgical photograph is used.",
        size=6.7,
        color=MUTED,
        ha="left",
        va="top",
    )


def token(ax: plt.Axes, x: float, y: float, w: float, s: str, fc: str, color: str = INK, weight="normal") -> None:
    h = 0.060
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="#c8ced6", lw=0.75))
    text(ax, x + w / 2, y + h / 2, s, size=8.5, color=color, ha="center", va="center", weight=weight)


def draw_sentence_task(ax: plt.Axes) -> None:
    text(ax, 0.066, 0.846, "matched Mandarin sentence pairs", size=7.8, ha="left", va="top")
    x0 = 0.120
    widths = [0.110, 0.070, 0.110, 0.145]
    y_a, y_p = 0.734, 0.642
    x = x0
    for w, s, fc, col, wt in zip(widths, ["虫子", "把", "树叶", "啃光了"], ["#fbfbfb", "#e8f0fb", "#fbfbfb", "#fbfbfb"], [INK, ACTIVE, INK, INK], ["normal", "bold", "normal", "normal"]):
        token(ax, x, y_a, w, s, fc, col, wt)
        x += w
    x = x0
    for w, s, fc, col, wt in zip(widths, ["树叶", "被", "虫子", "啃光了"], ["#fbfbfb", "#f9e9e7", "#fbfbfb", "#fbfbfb"], [INK, PASSIVE, INK, INK], ["normal", "bold", "normal", "normal"]):
        token(ax, x, y_p, w, s, fc, col, wt)
        x += w
    text(ax, 0.050, y_a + 0.030, "active", size=7.4, color=ACTIVE, ha="left", va="center")
    text(ax, 0.050, y_p + 0.030, "passive", size=7.4, color=PASSIVE, ha="left", va="center")
    onset_x = x0 + widths[0]
    ax.plot([onset_x, onset_x], [0.606, 0.804], color="#3f3f3f", lw=0.9, ls="--")
    text(ax, onset_x + 0.012, 0.618, "function-word onset (t = 0)", size=6.8, color=MUTED, ha="left", va="center")
    rounded_box(ax, 0.600, 0.728, 0.145, 0.064, fc="#e8f0fb", ec="#bbcee9", lw=0.85)
    text(ax, 0.672, 0.760, "把 / 将 / 给", size=7.1, color=ACTIVE, ha="center", va="center")
    rounded_box(ax, 0.600, 0.636, 0.145, 0.064, fc="#f9e8e5", ec="#e8c2bd", lw=0.85)
    text(ax, 0.672, 0.668, "被 / 让 / 叫", size=7.1, color=PASSIVE, ha="center", va="center")
    rounded_box(ax, 0.770, 0.675, 0.170, 0.070, fc="#fbfbfb", ec="#c4cbd5", lw=0.85)
    text(ax, 0.855, 0.710, "10 active +\n10 passive / run", size=7.2, ha="center", va="center")


def draw_timeline(ax: plt.Axes) -> None:
    text(ax, 0.070, 0.520, "trial timing and alignment", size=8.0, ha="left", va="top")
    y = 0.414
    ax.plot([0.072, 0.925], [y, y], color="#111", lw=1.2)
    ticks = [
        (0.112, "fixation", "#444"),
        (0.230, "speech onset", "#444"),
        (0.470, "function-word onset /\nanalysis t = 0", PASSIVE),
        (0.778, "sentence end", "#444"),
        (0.908, "catch press\nif nonsensical", CATCH),
    ]
    for x, lab, col in ticks:
        ax.plot([x, x], [y - 0.049, y + 0.049], color=col, lw=1.0, ls="--" if col == PASSIVE else "-")
        text(ax, x, y - 0.073, lab, size=6.5, color=col, ha="center", va="top")
    t = np.linspace(0, 1, 500)
    env = np.sin(t * 28 * np.pi) * np.exp(-((t - 0.48) / 0.31) ** 2)
    ax.fill_between(0.230 + t * 0.500, y + env * 0.044, y - env * 0.044, color="#b9c8dc", alpha=0.85, lw=0)


def draw_task_stream(ax: plt.Axes) -> None:
    text(ax, 0.070, 0.243, "randomized task stream", size=8.0, ha="left", va="top")
    blocks = [
        ("active/\npassive", ACTIVE, 0.112),
        ("synonym\ncontrols", CONTROL, 0.126),
        ("scrambled\ncatch", CATCH, 0.120),
        ("active/\npassive", ACTIVE, 0.112),
        ("synonym\ncontrols", CONTROL, 0.126),
        ("...", "#9fa6af", 0.090),
    ]
    x = 0.074
    y = 0.130
    for lab, col, w in blocks:
        ax.add_patch(Rectangle((x, y), w, 0.073, fc=col, ec="white", lw=0.8, alpha=0.95))
        text(ax, x + w / 2, y + 0.0365, lab, size=6.4, color="white", ha="center", va="center", weight="bold")
        x += w + 0.014
    text(
        ax,
        0.085,
        0.073,
        "Analyses in this work focus on active/passive sentence pairs;\nfiller/control and scrambled trials remain interleaved in the run.",
        size=6.5,
        color=MUTED,
        ha="left",
        va="top",
    )


def draw_panel_b(ax: plt.Axes) -> None:
    panel(ax, "b", "Mandarin active/passive sentence task")
    draw_sentence_task(ax)
    draw_timeline(ax)
    draw_task_stream(ax)


def main() -> None:
    configure_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.8, 7.2))
    gs = fig.add_gridspec(1, 2, left=0.015, right=0.990, bottom=0.035, top=0.975, wspace=0.025)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    clean_axis(ax_a)
    clean_axis(ax_b)
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    outputs = {
        "svg": OUT_DIR / "fig01_biorender_style_vector.svg",
        "pdf": OUT_DIR / "fig01_biorender_style_vector.pdf",
        "png": OUT_DIR / "fig01_biorender_style_vector_preview.png",
    }
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print({k: str(v) for k, v in outputs.items()})


if __name__ == "__main__":
    main()
