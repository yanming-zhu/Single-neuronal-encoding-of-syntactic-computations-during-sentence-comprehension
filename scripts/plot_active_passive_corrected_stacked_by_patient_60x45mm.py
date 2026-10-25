#!/usr/bin/env python3
"""Export the corrected active/passive patient stack at 60 x 45 mm, 8 pt."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = ROOT / "outputs/active_passive_corrected_timepoint_screen_20260722/unit_crosswalk.csv"
OUTPUT_DIR = (
    ROOT
    / "reports"
    / "nature_science_draft"
    / "assembled_figures_20260805"
    / "subplots"
)
SOURCE_DIR = OUTPUT_DIR / "source_data"
OUTPUT_STEM = OUTPUT_DIR / "corrected_unit_census_stacked_by_patient_60x45mm"
SOURCE_OUTPUT = SOURCE_DIR / "corrected_unit_census_stacked_by_patient_60x45mm.csv"

WIDTH_MM = 60.0
HEIGHT_MM = 45.0
TEXT_PT = 8.0

PATIENTS = ("gz001", "gz002", "gz003", "gz004", "gz005")
CENTERS = dict(zip(PATIENTS, (20.0, 15.0, 10.0, 5.0, 0.0)))
COLORS = {
    "gz001": "#0072B2",
    "gz002": "#E69F00",
    "gz003": "#009E73",
    "gz004": "#CC79A7",
    "gz005": "#D55E00",
}
INK = "#17322F"
GRID = "#DCE4E1"
POSITIVE_BG = "#F1F7F5"
NEGATIVE_BG = "#FBF5EE"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": TEXT_PT,
            "axes.labelsize": TEXT_PT,
            "axes.titlesize": TEXT_PT,
            "xtick.labelsize": TEXT_PT,
            "ytick.labelsize": TEXT_PT,
            "legend.fontsize": TEXT_PT,
            "figure.titlesize": TEXT_PT,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def load_and_validate() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    required = {
        "subject",
        "run",
        "unit_id",
        "ks_label",
        "corrected_window_cohens_d",
        "corrected_contiguous_gt5ms",
        "corrected_direction",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(data) != 1782:
        raise ValueError(f"Expected 1,782 units, found {len(data)}")
    if tuple(sorted(data["subject"].unique())) != PATIENTS:
        raise ValueError("Patient set does not match gz001-gz005")
    if int(data["corrected_contiguous_gt5ms"].astype(bool).sum()) != 172:
        raise ValueError("Corrected screen-positive total is not 172")
    return data


def main() -> None:
    data = load_and_validate()
    configure_style()
    fig, ax = plt.subplots(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
    )

    source_rows: list[pd.DataFrame] = []
    for patient in PATIENTS:
        frame = data.loc[data["subject"].eq(patient)].copy()
        frame["plot_d"] = (
            pd.to_numeric(frame["corrected_window_cohens_d"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        frame = frame.sort_values(["plot_d", "unit_id"]).reset_index(drop=True)
        frame["rank_percentile"] = np.linspace(0.0, 100.0, len(frame))
        significant = frame["corrected_contiguous_gt5ms"].astype(bool).to_numpy()
        x = frame["rank_percentile"].to_numpy(float)
        d = frame["plot_d"].to_numpy(float)
        center = CENTERS[patient]
        y = center + d
        color = COLORS[patient]

        ax.axhspan(center, center + 2.2, color=POSITIVE_BG, zorder=0)
        ax.axhspan(center - 2.2, center, color=NEGATIVE_BG, zorder=0)
        ax.axhline(center, color=INK, lw=0.42, zorder=1)
        ax.plot(x, y, color=color, lw=0.65, alpha=0.82, zorder=2)
        ax.scatter(
            x,
            y,
            s=1.7,
            color=color,
            alpha=0.72,
            linewidth=0,
            zorder=3,
            rasterized=True,
        )
        ax.scatter(
            x[significant],
            y[significant],
            s=12,
            facecolors="white",
            edgecolors="black",
            linewidth=0.45,
            zorder=4,
        )
        ax.scatter(
            x[significant],
            y[significant],
            s=2.1,
            color=color,
            linewidth=0,
            zorder=5,
        )

        selected = frame.loc[significant]
        directions = selected["corrected_direction"].value_counts()
        n_active = int(directions.get("active_gt_passive", 0))
        n_passive = int(directions.get("passive_gt_active", 0))
        n_mixed = int(directions.get("mixed", 0))
        label_y = center + 1.72
        text_box = dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.15)
        ax.text(
            0.5,
            label_y,
            patient,
            ha="left",
            va="center",
            fontsize=TEXT_PT,
            fontweight="bold",
            color=color,
            bbox=text_box,
            zorder=7,
        )
        ax.text(
            99.5,
            label_y,
            f"{len(frame)} | {significant.sum()} ({n_active}/{n_passive}/{n_mixed})",
            ha="right",
            va="center",
            fontsize=TEXT_PT,
            fontweight="bold",
            color=INK,
            bbox=text_box,
            zorder=7,
        )

        frame["display_patient"] = patient
        frame["stack_center"] = center
        frame["stacked_y"] = y
        frame["screen_positive"] = significant
        source_rows.append(
            frame[
                [
                    "display_patient",
                    "run",
                    "unit_id",
                    "ks_label",
                    "rank_percentile",
                    "plot_d",
                    "stack_center",
                    "stacked_y",
                    "screen_positive",
                    "corrected_direction",
                ]
            ]
        )

    ax.set_xlim(-1.5, 101.5)
    ax.set_ylim(-2.25, 22.25)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2.0, width=0.55, pad=1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.55)

    # One shared d scale is sufficient because every patient band uses the
    # identical [-2.2, 2.2] range.
    top_center = CENTERS["gz001"]
    for value, label in ((2, "+2"), (0, "0"), (-2, "-2")):
        ax.text(
            -2.3,
            top_center + value,
            label,
            ha="right",
            va="center",
            fontsize=TEXT_PT,
            clip_on=False,
        )

    fig.suptitle(
        "Corrected-onset effects by patient",
        x=0.56,
        y=0.985,
        fontsize=TEXT_PT,
        fontweight="bold",
    )
    fig.text(
        0.13,
        0.900,
        "n|sig (A/P/M)",
        ha="left",
        va="center",
        fontsize=TEXT_PT,
    )
    fig.text(
        0.985,
        0.900,
        "rings: p<.025, 6+ ms",
        ha="right",
        va="center",
        fontsize=TEXT_PT,
    )
    fig.text(
        0.025,
        0.505,
        "Cohen's d (+ active; - passive)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=TEXT_PT,
    )
    fig.text(
        0.57,
        0.040,
        "Within-patient unit rank (%)",
        ha="center",
        va="bottom",
        fontsize=TEXT_PT,
    )
    fig.subplots_adjust(
        left=0.145,
        right=0.985,
        top=0.840,
        bottom=0.155,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUTPUT_STEM}.png", dpi=600, facecolor="white")
    fig.savefig(f"{OUTPUT_STEM}.pdf", facecolor="white")
    fig.savefig(f"{OUTPUT_STEM}.svg", facecolor="white")
    plt.close(fig)

    pd.concat(source_rows, ignore_index=True).to_csv(SOURCE_OUTPUT, index=False)
    print(f"{OUTPUT_STEM}.png")
    print(f"{OUTPUT_STEM}.pdf")
    print(f"{OUTPUT_STEM}.svg")


if __name__ == "__main__":
    main()
