from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = (
    ROOT
    / "reports"
    / "nature_science_draft"
    / "assembled_figures_20260805"
    / "subplots"
)
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#0F7F78"
TEAL_LIGHT = "#E6F2F0"
CORAL = "#D95F4C"
CORAL_LIGHT = "#FBEAE5"
BLUE = "#2F78B7"
ORANGE = "#D96A0B"
INK = "#17312F"
GREY = "#777777"
BORDER = "#D5DDDA"

CN = FontProperties(family="Hiragino Sans GB")
CN_BOLD = FontProperties(family="Hiragino Sans GB", weight="bold")


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def token_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = BORDER,
    textcolor: str = INK,
) -> None:
    height = 0.15
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=16,
        color=textcolor,
        fontproperties=CN_BOLD,
        zorder=4,
    )


def role_label(
    ax: plt.Axes,
    x: float,
    y: float,
    noun_id: str,
    role_cn: str,
    role_en: str,
    color: str,
) -> None:
    ax.text(
        x,
        y,
        f"{noun_id}  {role_cn} {role_en}",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=color,
        fontproperties=CN_BOLD,
    )


def level_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    title: str,
    body: str,
    *,
    edgecolor: str,
    facecolor: str = "white",
    titlecolor: str | None = None,
    linewidth: float = 1.0,
) -> None:
    height = 0.13
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.022",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y + 0.025,
        title,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=titlecolor or edgecolor,
        zorder=3,
    )
    ax.text(
        x,
        y - 0.027,
        body,
        ha="center",
        va="center",
        fontsize=5.8,
        color=INK,
        zorder=3,
    )


def make_subplot() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.012, 0.975, "a", fontsize=12, fontweight="bold", ha="left", va="top")
    ax.text(
        0.070,
        0.975,
        "Syntactic role binding in matched role-reversal sentences",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
    )

    level_y = 0.82
    level_x = [0.13, 0.38, 0.63, 0.87]
    level_width = [0.20, 0.22, 0.22, 0.19]
    level_box(
        ax,
        level_x[0],
        level_y,
        level_width[0],
        "Speech sequence",
        "phonology · word onsets",
        edgecolor=BLUE,
        facecolor="#F3F7FB",
    )
    level_box(
        ax,
        level_x[1],
        level_y,
        level_width[1],
        "Constructional cues",
        "word order · BA/BEI",
        edgecolor="#D89A21",
        facecolor="#FFF8E8",
    )
    level_box(
        ax,
        level_x[2],
        level_y,
        level_width[2],
        "Syntactic role binding",
        "N1/N2 → agent/patient",
        edgecolor=TEAL,
        facecolor=TEAL_LIGHT,
        linewidth=1.8,
    )
    level_box(
        ax,
        level_x[3],
        level_y,
        level_width[3],
        "Event model",
        "who did what to whom",
        edgecolor="#8C79BE",
        facecolor="#F7F4FC",
    )
    for left, right, left_w, right_w in zip(
        level_x[:-1],
        level_x[1:],
        level_width[:-1],
        level_width[1:],
    ):
        ax.add_patch(
            FancyArrowPatch(
                (left + left_w / 2 + 0.01, level_y),
                (right - right_w / 2 - 0.01, level_y),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.9,
                color="#9AA3A0",
                zorder=1,
            )
        )
    ax.text(
        level_x[2],
        level_y + 0.092,
        "CURRENT STUDY LEVEL",
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color=TEAL,
    )
    ax.plot([0.04, 0.96], [0.695, 0.695], color="#E5E8E7", lw=0.8)
    ax.text(
        0.055,
        0.655,
        "Construction templates",
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=GREY,
    )
    ax.text(0.055, 0.565, "Active", color=BLUE, fontsize=8.2, fontweight="bold", va="center")
    ax.text(
        0.055,
        0.505,
        "N1   ba/gei/jiang   N2   V",
        family="DejaVu Sans Mono",
        fontsize=7.5,
        ha="left",
        va="center",
        bbox=dict(boxstyle="square,pad=0.10", fc="#F8DFDF", ec="none"),
    )
    ax.text(
        0.055,
        0.445,
        "N1 agent-like; N2 patient-like",
        fontsize=6.5,
        color=GREY,
        ha="left",
        va="center",
    )
    ax.text(0.055, 0.345, "Passive", color=CORAL, fontsize=8.2, fontweight="bold", va="center")
    ax.text(
        0.055,
        0.285,
        "N2   bei/rang/jiao   N1   V",
        family="DejaVu Sans Mono",
        fontsize=7.5,
        ha="left",
        va="center",
        bbox=dict(boxstyle="square,pad=0.10", fc="#F8DFDF", ec="none"),
    )
    ax.text(0.055, 0.225, "roles reversed", fontsize=6.5, color=GREY, ha="left", va="center")
    ax.text(
        0.055,
        0.115,
        "Construction and function-word\nidentity covary.",
        fontsize=6.4,
        color=ORANGE,
        ha="left",
        va="center",
    )
    ax.plot([0.315, 0.315], [0.075, 0.655], color="#E5E8E7", lw=0.8)
    ax.text(
        0.345,
        0.655,
        "Concrete role-reversal example",
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=GREY,
    )

    active_y = 0.48
    passive_y = 0.17
    noun_left = 0.435
    marker_x = 0.600
    noun_right = 0.765
    verb_x = 0.915

    ax.text(
        0.365,
        active_y,
        "active",
        color=BLUE,
        fontsize=7.5,
        fontweight="bold",
        ha="right",
        va="center",
    )
    ax.text(
        0.365,
        passive_y,
        "passive",
        color=ORANGE,
        fontsize=7.5,
        fontweight="bold",
        ha="right",
        va="center",
    )

    role_label(ax, noun_left, active_y + 0.11, "N1", "施事", "agent", TEAL)
    ax.text(marker_x, active_y + 0.11, "marker", ha="center", va="bottom", fontsize=7.2, color=GREY)
    role_label(ax, noun_right, active_y + 0.11, "N2", "受事", "patient", CORAL)
    ax.text(verb_x, active_y + 0.11, "verb/result", ha="center", va="bottom", fontsize=7.2, color=GREY)

    role_label(ax, noun_left, passive_y + 0.11, "N2", "受事", "patient", CORAL)
    ax.text(marker_x, passive_y + 0.11, "marker", ha="center", va="bottom", fontsize=7.2, color=GREY)
    role_label(ax, noun_right, passive_y + 0.11, "N1", "施事", "agent", TEAL)
    ax.text(verb_x, passive_y + 0.11, "verb/result", ha="center", va="bottom", fontsize=7.2, color=GREY)

    token_box(ax, noun_left, active_y, 0.12, "张三", facecolor=TEAL_LIGHT, edgecolor=TEAL)
    token_box(ax, marker_x, active_y, 0.075, "把", facecolor=TEAL, edgecolor=TEAL, textcolor="white")
    token_box(ax, noun_right, active_y, 0.12, "李四", facecolor=CORAL_LIGHT, edgecolor=CORAL)
    token_box(ax, verb_x, active_y, 0.105, "推倒")

    token_box(ax, noun_left, passive_y, 0.12, "李四", facecolor=CORAL_LIGHT, edgecolor=CORAL)
    token_box(ax, marker_x, passive_y, 0.075, "被", facecolor=CORAL, edgecolor=CORAL, textcolor="white")
    token_box(ax, noun_right, passive_y, 0.12, "张三", facecolor=TEAL_LIGHT, edgecolor=TEAL)
    token_box(ax, verb_x, passive_y, 0.105, "推倒")

    arrow_n1 = FancyArrowPatch(
        (noun_left, active_y - 0.085),
        (noun_right, passive_y + 0.085),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=2.0,
        color=TEAL,
        connectionstyle="arc3,rad=0.0",
        zorder=2,
    )
    arrow_n2 = FancyArrowPatch(
        (noun_right, active_y - 0.085),
        (noun_left, passive_y + 0.085),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=2.0,
        color=CORAL,
        connectionstyle="arc3,rad=0.0",
        zorder=2,
    )
    ax.add_patch(arrow_n1)
    ax.add_patch(arrow_n2)

    ax.text(
        0.685,
        0.008,
        "N1 and N2 exchange serial position while retaining agent/patient roles",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=GREY,
    )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.98)
    png = OUT / "fig1a_role_reversal_example.png"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


if __name__ == "__main__":
    print(make_subplot())
