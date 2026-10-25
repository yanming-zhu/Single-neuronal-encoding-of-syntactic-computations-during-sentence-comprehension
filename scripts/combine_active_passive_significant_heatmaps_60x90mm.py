#!/usr/bin/env python3
"""Combine corrected A>P and A<P unit heatmaps, all text 8 pt."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from plot_active_passive_significant_heatmaps_by_direction_20260723 import (
    DIRECTIONS,
    PATIENTS,
    PATIENT_COLORS,
    SCREEN_PATH,
    contiguous_blocks,
    filter_horizontal_noise_stripes,
    load_direction,
)


OUTPUT_DIR = (
    ROOT
    / "reports"
    / "nature_science_draft"
    / "assembled_figures_20260805"
    / "subplots"
)
SOURCE_DIR = OUTPUT_DIR / "source_data"
WIDTH_MM = 75.0
HEIGHT_MM = 90.0
STEM = "corrected_significant_unit_heatmaps_both_directions_75x90mm"
TEXT_PT = 8.0
INK = "#17322F"


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
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def add_boundaries(ax: plt.Axes, metadata: pd.DataFrame, *, strip: bool = False) -> None:
    run_blocks = contiguous_blocks(metadata["run"].to_numpy())
    patient_blocks = contiguous_blocks(metadata["patient"].to_numpy())
    for _, _, stop in run_blocks[:-1]:
        ax.axhline(stop, color="white", lw=0.35 if not strip else 0.3, alpha=0.9, zorder=5)
    for _, _, stop in patient_blocks[:-1]:
        ax.axhline(stop, color="white", lw=0.9 if not strip else 0.6, zorder=6)
        ax.axhline(stop, color=INK, lw=0.25, zorder=7)


def add_time_markers(ax: plt.Axes) -> None:
    for value, style in ((0.0, "--"), (0.8, ":")):
        ax.axvline(value, color="white", lw=0.75, ls=style, alpha=0.96, zorder=6)
        ax.axvline(value, color=INK, lw=0.25, ls=style, alpha=0.96, zorder=7)


def draw_direction(
    fig: plt.Figure,
    subplot_spec,
    payload,
    *,
    rate_vmax: float,
    difference_vmax: float,
) -> None:
    n_units = len(payload.metadata)
    patient_codes = payload.metadata["patient_index"].to_numpy(int)[:, None]
    inner = subplot_spec.subgridspec(
        1,
        6,
        width_ratios=[0.09, 1.0, 1.0, 0.11, 1.0, 0.11],
        wspace=0.18,
    )
    ax_strip = fig.add_subplot(inner[0, 0])
    ax_active = fig.add_subplot(inner[0, 1], sharey=ax_strip)
    ax_passive = fig.add_subplot(inner[0, 2], sharex=ax_active, sharey=ax_strip)
    ax_rate_cbar = fig.add_subplot(inner[0, 3])
    ax_difference = fig.add_subplot(inner[0, 4], sharex=ax_active, sharey=ax_strip)
    ax_difference_cbar = fig.add_subplot(inner[0, 5])

    patient_cmap = ListedColormap([PATIENT_COLORS[patient] for patient in PATIENTS])
    ax_strip.imshow(
        patient_codes,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[0, 1, n_units, 0],
        cmap=patient_cmap,
        vmin=-0.5,
        vmax=len(PATIENTS) - 0.5,
        rasterized=True,
    )
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    ax_strip.set_title("Pt", fontsize=TEXT_PT, pad=1.5)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)
    add_boundaries(ax_strip, payload.metadata, strip=True)

    extent = [float(payload.time_sec[0]), float(payload.time_sec[-1]), n_units, 0]
    active_image = ax_active.imshow(
        payload.active_hz,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=rate_vmax,
        rasterized=True,
    )
    ax_passive.imshow(
        payload.passive_hz,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=rate_vmax,
        rasterized=True,
    )
    difference_image = ax_difference.imshow(
        payload.difference_hz,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        cmap="RdBu_r",
        vmin=-difference_vmax,
        vmax=difference_vmax,
        rasterized=True,
    )

    for ax, title in zip((ax_active, ax_passive, ax_difference), ("Active", "Passive", "A-P")):
        ax.set_title(title, fontsize=TEXT_PT, fontweight="bold", pad=1.5)
        ax.set_xlim(float(payload.time_sec[0]), float(payload.time_sec[-1]))
        ax.set_ylim(n_units, 0)
        ax.set_xticks([-1.0, 0.0, 2.0] if ax is ax_difference else [])
        ax.tick_params(axis="x", length=1.8, width=0.5, pad=1.0)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_boundaries(ax, payload.metadata)
        add_time_markers(ax)

    ax_active.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_passive.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    rate_cbar = fig.colorbar(active_image, cax=ax_rate_cbar, ticks=[])
    rate_cbar.ax.set_title("Hz", fontsize=TEXT_PT, pad=1.5)
    rate_cbar.outline.set_linewidth(0.45)
    rate_cbar.ax.text(
        0.5,
        0.5,
        f"0-{rate_vmax:.0f}",
        transform=rate_cbar.ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=TEXT_PT,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.15),
    )

    diff_tick = float(round(difference_vmax))
    difference_cbar = fig.colorbar(
        difference_image,
        cax=ax_difference_cbar,
        ticks=[],
    )
    difference_cbar.ax.set_title("Hz", fontsize=TEXT_PT, pad=1.5)
    difference_cbar.outline.set_linewidth(0.45)
    difference_cbar.ax.text(
        0.5,
        0.5,
        f"-{diff_tick:.0f} to +{diff_tick:.0f}",
        transform=difference_cbar.ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=TEXT_PT,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.15),
    )


def main(
    *,
    width_mm: float = WIDTH_MM,
    height_mm: float = HEIGHT_MM,
    stem: str = STEM,
) -> None:
    output_stem = OUTPUT_DIR / stem
    order_output = SOURCE_DIR / f"{stem}_unit_order.csv"
    removed_output = SOURCE_DIR / f"{stem}_display_qc_removed.csv"
    matrix_output = SOURCE_DIR / f"{stem}_matrices.npz"

    screen = pd.read_csv(SCREEN_PATH)
    if int(screen["corrected_contiguous_gt5ms"].astype(bool).sum()) != 172:
        raise ValueError("Expected 172 corrected screen-positive units")

    payloads = {}
    removed_parts = []
    for direction in DIRECTIONS:
        filtered, removed = filter_horizontal_noise_stripes(load_direction(screen, direction))
        payloads[direction] = filtered
        removed_parts.append(removed)
    if len(payloads["active_gt_passive"].metadata) != 71:
        raise ValueError("Expected 71 displayed Active > Passive units")
    if len(payloads["passive_gt_active"].metadata) != 78:
        raise ValueError("Expected 78 displayed Active < Passive units")

    rate_values = np.concatenate(
        [
            matrix.ravel()
            for payload in payloads.values()
            for matrix in (payload.active_hz, payload.passive_hz)
        ]
    )
    difference_values = np.concatenate(
        [payload.difference_hz.ravel() for payload in payloads.values()]
    )
    rate_vmax = max(1.0, float(np.nanpercentile(rate_values, 98)))
    difference_vmax = max(1.0, float(np.nanpercentile(np.abs(difference_values), 98)))

    configure_style()
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    outer = fig.add_gridspec(
        2,
        1,
        left=0.055,
        right=0.985,
        bottom=0.10,
        top=0.88,
        hspace=0.69,
    )
    draw_direction(
        fig,
        outer[0, 0],
        payloads["active_gt_passive"],
        rate_vmax=rate_vmax,
        difference_vmax=difference_vmax,
    )
    draw_direction(
        fig,
        outer[1, 0],
        payloads["passive_gt_active"],
        rate_vmax=rate_vmax,
        difference_vmax=difference_vmax,
    )

    top_counts = payloads["active_gt_passive"].metadata.groupby("patient", sort=False).size()
    bottom_counts = payloads["passive_gt_active"].metadata.groupby("patient", sort=False).size()
    fig.text(
        0.52,
        0.985,
        "Active > passive (n=71)",
        ha="center",
        va="top",
        fontsize=TEXT_PT,
        fontweight="bold",
    )
    fig.text(
        0.52,
        0.952,
        "gz001-gz005: " + "/".join(str(int(top_counts.get(patient, 0))) for patient in PATIENTS),
        ha="center",
        va="top",
        fontsize=TEXT_PT,
    )
    fig.text(
        0.52,
        0.505,
        "Active < passive (n=78)",
        ha="center",
        va="top",
        fontsize=TEXT_PT,
        fontweight="bold",
    )
    fig.text(
        0.52,
        0.472,
        "gz001-gz005: " + "/".join(str(int(bottom_counts.get(patient, 0))) for patient in PATIENTS),
        ha="center",
        va="top",
        fontsize=TEXT_PT,
    )
    fig.text(
        0.55,
        0.535,
        "Time from corrected function-word onset (s)",
        ha="center",
        va="center",
        fontsize=TEXT_PT,
    )
    fig.text(
        0.55,
        0.025,
        "Time from corrected function-word onset (s)",
        ha="center",
        va="bottom",
        fontsize=TEXT_PT,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_stem}.png", dpi=600, facecolor="white")
    fig.savefig(f"{output_stem}.pdf", facecolor="white")
    fig.savefig(f"{output_stem}.svg", facecolor="white")
    plt.close(fig)

    order = pd.concat([payload.metadata for payload in payloads.values()], ignore_index=True)
    order["rate_vmax_hz_shared"] = rate_vmax
    order["difference_abs_vmax_hz_shared"] = difference_vmax
    order.to_csv(order_output, index=False)
    pd.concat(removed_parts, ignore_index=True).to_csv(removed_output, index=False)
    np.savez_compressed(
        matrix_output,
        time_sec=payloads["active_gt_passive"].time_sec,
        active_gt_passive_active_hz=payloads["active_gt_passive"].active_hz,
        active_gt_passive_passive_hz=payloads["active_gt_passive"].passive_hz,
        active_gt_passive_difference_hz=payloads["active_gt_passive"].difference_hz,
        passive_gt_active_active_hz=payloads["passive_gt_active"].active_hz,
        passive_gt_active_passive_hz=payloads["passive_gt_active"].passive_hz,
        passive_gt_active_difference_hz=payloads["passive_gt_active"].difference_hz,
    )
    print(f"{output_stem}.png")
    print(f"{output_stem}.pdf")
    print(f"{output_stem}.svg")


if __name__ == "__main__":
    main()
