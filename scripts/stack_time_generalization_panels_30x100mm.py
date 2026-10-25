#!/usr/bin/env python3
"""Stack the time-resolved decoder and cross-time matrix at 30 x 100 mm."""

from __future__ import annotations

import json
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
from matplotlib.patches import Rectangle


INPUT_DIR = (
    ROOT
    / "neuropixel2025"
    / "data"
    / "combined_active_passive_time_generalization"
)
OUTPUT_DIR = (
    ROOT
    / "reports"
    / "nature_science_draft"
    / "assembled_figures_20260805"
    / "subplots"
)
SOURCE_DIR = OUTPUT_DIR / "source_data"
STEM = "time_generalization_d_e_stacked_30x100mm"

WIDTH_MM = 30.0
HEIGHT_MM = 100.0
TEXT_PT = 8.0
INK = "#17322F"
BLUE = "#357ABD"
LIGHT_GREY = "#D8DDE3"
MID_GREY = "#6E7175"


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
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def as_text(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def main() -> None:
    matrix_path = INPUT_DIR / "active_passive_time_generalization_matrices.npz"
    summary_path = INPUT_DIR / "active_passive_time_generalization_summary.json"
    run_summary_path = INPUT_DIR / "active_passive_time_generalization_run_summary.csv"

    payload = np.load(matrix_path, allow_pickle=True)
    summary = json.loads(summary_path.read_text())
    run_summary = pd.read_csv(run_summary_path)

    centers = np.asarray(payload["centers_sec"], dtype=float)
    auc_matrices = np.asarray(payload["auc_matrices"], dtype=float)
    diag_auc = np.asarray(payload["diag_auc"], dtype=float)
    subjects = as_text(payload["subjects"])
    runs = as_text(payload["runs"])

    if auc_matrices.shape != (7, 51, 51) or diag_auc.shape != (7, 51):
        raise ValueError(
            "Expected 7 runs, 51 diagonal time points, and 51 x 51 matrices"
        )
    if not np.allclose(np.diagonal(auc_matrices, axis1=1, axis2=2), diag_auc, equal_nan=True):
        raise ValueError("Stored diagonal AUC does not match the matrix diagonal")

    diagonal_median = np.nanmedian(diag_auc, axis=0)
    matrix_median = np.nanmedian(auc_matrices, axis=0)
    peak = summary["median_diagonal_peak"]

    configure_style()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))

    ax_diag = fig.add_axes([0.28, 0.59, 0.64, 0.29])
    ax_matrix = fig.add_axes([0.28, 0.18, 0.64, 0.21])
    ax_cbar = fig.add_axes([0.30, 0.055, 0.60, 0.018])

    for row in diag_auc:
        ax_diag.plot(centers, row, color=LIGHT_GREY, lw=0.55, alpha=0.80, zorder=1)
    ax_diag.axvspan(0.0, 0.8, color="#BFD3EA", alpha=0.25, lw=0, zorder=0)
    ax_diag.plot(centers, diagonal_median, color=BLUE, lw=1.5, zorder=3)
    ax_diag.axhline(0.5, color=MID_GREY, ls="--", lw=0.65, zorder=2)
    ax_diag.add_patch(
        Rectangle(
            (0.20, 0.27),
            0.20,
            0.66,
            facecolor="#F4A261",
            edgecolor="#D95F4C",
            linewidth=1.0,
            alpha=0.22,
            zorder=5,
        )
    )
    ax_diag.set_xlim(-0.75, 1.75)
    ax_diag.set_ylim(0.25, 0.95)
    ax_diag.set_xticks([-0.5, 0.5, 1.5])
    ax_diag.set_yticks([0.3, 0.5, 0.7, 0.9])
    ax_diag.set_xlabel("Time (s)", labelpad=2.0)
    ax_diag.spines["top"].set_visible(False)
    ax_diag.spines["right"].set_visible(False)
    ax_diag.text(
        0.03,
        0.97,
        f"peak {peak['median_auc']:.2f}\nat {peak['time_sec']:.2f} s",
        transform=ax_diag.transAxes,
        ha="left",
        va="top",
        fontsize=TEXT_PT,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.4),
        zorder=4,
    )
    ax_diag.text(
        0.97,
        0.04,
        "AUC",
        transform=ax_diag.transAxes,
        ha="right",
        va="bottom",
        fontsize=TEXT_PT,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.25),
        zorder=4,
    )

    extent = [centers[0], centers[-1], centers[-1], centers[0]]
    image = ax_matrix.imshow(
        matrix_median,
        extent=extent,
        cmap="viridis",
        vmin=0.35,
        vmax=0.75,
        interpolation="nearest",
        aspect="auto",
        rasterized=True,
    )
    ax_matrix.plot(
        [centers[0], centers[-1]],
        [centers[0], centers[-1]],
        color="white",
        lw=0.65,
        alpha=0.9,
    )
    ax_matrix.add_patch(
        Rectangle(
            (0.20, 0.20),
            0.20,
            0.20,
            facecolor="#F4A261",
            edgecolor="#D95F4C",
            linewidth=1.0,
            alpha=0.45,
            zorder=5,
        )
    )
    ax_matrix.set_xticks([-0.5, 0.5, 1.5])
    ax_matrix.set_yticks([-0.5, 0.5, 1.5])
    ax_matrix.set_xlabel("Test (s)", labelpad=2.0)
    ax_matrix.text(
        0.03,
        0.97,
        "Train (s)",
        transform=ax_matrix.transAxes,
        ha="left",
        va="top",
        fontsize=TEXT_PT,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.25),
    )
    for spine in ax_matrix.spines.values():
        spine.set_linewidth(0.6)

    colorbar = fig.colorbar(
        image,
        cax=ax_cbar,
        orientation="horizontal",
        ticks=[0.35, 0.55, 0.75],
    )
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(length=2.0, width=0.5, pad=1.0)
    colorbar.ax.set_title("AUC", fontsize=TEXT_PT, pad=1.0)

    fig.text(0.035, 0.975, "d", ha="left", va="top", fontsize=TEXT_PT, fontweight="bold")
    fig.text(
        0.25,
        0.975,
        "Transient top-5\ndecoder",
        ha="left",
        va="top",
        fontsize=TEXT_PT,
        fontweight="bold",
        linespacing=1.05,
    )
    fig.text(0.035, 0.485, "e", ha="left", va="top", fontsize=TEXT_PT, fontweight="bold")
    fig.text(
        0.25,
        0.485,
        "Weak\ncross-time\nstability",
        ha="left",
        va="top",
        fontsize=TEXT_PT,
        fontweight="bold",
        linespacing=1.05,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / STEM
    fig.savefig(f"{output_stem}.png", dpi=600, facecolor="white")
    fig.savefig(f"{output_stem}.pdf", facecolor="white")
    fig.savefig(f"{output_stem}.svg", facecolor="white")
    plt.close(fig)

    diagonal_source = pd.DataFrame({"time_sec": centers, "median_auc": diagonal_median})
    for subject, run, values in zip(subjects, runs, diag_auc):
        diagonal_source[f"{subject}__{run}"] = values
    diagonal_source.to_csv(SOURCE_DIR / f"{STEM}_diagonal_auc.csv", index=False)

    train_time, test_time = np.meshgrid(centers, centers, indexing="ij")
    pd.DataFrame(
        {
            "train_time_sec": train_time.ravel(),
            "test_time_sec": test_time.ravel(),
            "median_auc": matrix_median.ravel(),
        }
    ).to_csv(SOURCE_DIR / f"{STEM}_median_matrix_long.csv", index=False)

    metadata = {
        "figure_size_mm": [WIDTH_MM, HEIGHT_MM],
        "text_size_pt": TEXT_PT,
        "n_runs": int(diag_auc.shape[0]),
        "n_time_points": int(centers.size),
        "analysis_mode": str(run_summary["analysis_mode"].iloc[0]),
        "unit_set": str(run_summary["unit_set"].iloc[0]),
        "model": str(run_summary["model"].iloc[0]),
        "logistic_c": float(run_summary["logistic_c"].iloc[0]),
        "cv_repeats": int(run_summary["cv_repeats"].iloc[0]),
        "median_diagonal_peak": peak,
        "descriptive_peak_highlight_window_sec": [0.20, 0.40],
        "cross_time_highlight_train_test_window_sec": [0.20, 0.40],
        "primary_window_summary": summary["primary_window_summary"],
    }
    (SOURCE_DIR / f"{STEM}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"{output_stem}.png")
    print(f"{output_stem}.pdf")
    print(f"{output_stem}.svg")


if __name__ == "__main__":
    main()
