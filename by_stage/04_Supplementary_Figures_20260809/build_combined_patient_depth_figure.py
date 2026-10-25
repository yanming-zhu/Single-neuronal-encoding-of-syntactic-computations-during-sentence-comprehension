#!/usr/bin/env python3
"""Assemble the five patient-resolved depth analyses into one figure."""

from __future__ import annotations

import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image


SOURCE_DATA_DIR = Path(os.environ.get("TAN_SUPPLEMENTARY_SOURCE_DATA_DIR", HERE / "source_data")).expanduser().resolve()
SOURCE_DIR = SOURCE_DATA_DIR / "patient_depth_activation"
OUT_DIR = HERE / "new_figures" / "combined_patient_depth"
SUBJECTS = ("gz001", "gz002", "gz003", "gz004", "gz005")
PANEL_LETTERS = tuple("abcde")

BLUE = "#2B6CB0"
CORAL = "#C94F3D"
GRAY = "#C8CECC"
GRID = "#DDE3E1"
INK = "#16312C"
MUTED = "#5E6A66"
PALE = "#EFF5F3"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def point_sizes(df: pd.DataFrame) -> np.ndarray:
    spike_column = "n_spikes" if "n_spikes" in df.columns else "n_spikes_position"
    spikes = pd.to_numeric(df[spike_column], errors="coerce").fillna(0).to_numpy()
    return np.clip(10 + np.sqrt(spikes) * 0.55, 11, 58)


def short_run_label(run_name: str) -> str:
    match = re.search(r"run(\d+)$", str(run_name))
    return f"r{match.group(1)}" if match else str(run_name)


def overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def add_candidate_labels(ax: plt.Axes, df: pd.DataFrame, include_run: bool) -> None:
    """Place nominal-candidate labels with deterministic collision minimization."""
    occupied: list[tuple[float, float, float, float]] = []
    x_offsets = (0.020, 0.055, 0.095, 0.135, -0.020, -0.055, -0.095, -0.135)
    y_offsets = (0.000, 0.045, -0.045, 0.090, -0.090, 0.135, -0.135, 0.180, -0.180)
    offsets = [(dx, dy) for dy in y_offsets for dx in x_offsets]
    to_axes = ax.transAxes.inverted().transform
    to_display = ax.transData.transform
    fontsize = 5.0 if include_run else 5.35

    # Treat candidate markers as occupied regions so labels are displaced from
    # neighbouring points as well as from other labels.
    for row in df.itertuples(index=False):
        px, py = to_axes(to_display((row.abs_active_passive_diff_hz, row.depth_um_median)))
        occupied.append((px - 0.010, py - 0.014, px + 0.010, py + 0.014))

    ordered = df.sort_values(
        ["abs_active_passive_diff_hz", "depth_um_median"],
        ascending=[False, True],
    )
    for row in ordered.itertuples(index=False):
        px, py = to_axes(to_display((row.abs_active_passive_diff_hz, row.depth_um_median)))
        label = (
            f"{short_run_label(row.run)}·u{int(row.unit_id)}"
            if include_run
            else f"u{int(row.unit_id)}"
        )
        text_w = 0.006 + 0.0128 * len(label)
        half_h = 0.026
        best: tuple[float, float] | None = None
        best_box: tuple[float, float, float, float] | None = None
        best_score = float("inf")
        for dx, dy in offsets:
            cx, cy = px + dx, py + dy
            box = (
                (cx, cy - half_h, cx + text_w, cy + half_h)
                if dx >= 0
                else (cx - text_w, cy - half_h, cx, cy + half_h)
            )
            edge_penalty = (
                max(0.0, -box[0]) + max(0.0, box[2] - 1.0)
                + max(0.0, -box[1]) + max(0.0, box[3] - 1.0)
            )
            collision = sum(overlap_area(box, old) for old in occupied)
            score = collision * 3000.0 + edge_penalty * 50.0 + np.hypot(dx, dy)
            if score < best_score:
                best_score = score
                best = (dx, dy)
                best_box = box
        assert best is not None and best_box is not None
        occupied.append(best_box)
        dx, dy = best
        ax.annotate(
            label,
            xy=(row.abs_active_passive_diff_hz, row.depth_um_median),
            xycoords="data",
            xytext=(px + dx, py + dy),
            textcoords=ax.transAxes,
            fontsize=fontsize,
            color="#222222",
            ha="left" if dx >= 0 else "right",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "#7A8581",
                "linewidth": 0.35,
                "shrinkA": 0.5,
                "shrinkB": 1.5,
            },
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.08},
            zorder=5,
        )


def load_subject(subject: str) -> pd.DataFrame:
    csv_path = SOURCE_DIR / f"{subject}_unit_depth_activation_metrics.csv"
    df = pd.read_csv(csv_path).copy()
    required = {
        "run",
        "unit_id",
        "ks_label",
        "depth_um_median",
        "active_minus_passive_bc_hz",
        "abs_active_passive_diff_hz",
        "significant_uncorrected_p05",
        "significant_run_fdr05",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    if not {"n_spikes", "n_spikes_position"}.intersection(df.columns):
        raise ValueError(f"{csv_path} is missing a spike-count column")
    for column in ("depth_um_median", "active_minus_passive_bc_hz", "abs_active_passive_diff_hz"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return (
        df.dropna(subset=["depth_um_median", "active_minus_passive_bc_hz", "abs_active_passive_diff_hz"])
        .sort_values(["depth_um_median", "abs_active_passive_diff_hz"])
        .reset_index(drop=True)
    )


def add_patient_panel(fig: plt.Figure, slot, subject: str, letter: str, left_column: bool) -> dict[str, int]:
    df = load_subject(subject)
    sig = bool_series(df["significant_uncorrected_p05"]).to_numpy()
    fdr = bool_series(df["significant_run_fdr05"]).to_numpy()
    good = (df["ks_label"].astype(str) == "good").to_numpy()
    active = df["active_minus_passive_bc_hz"].to_numpy() > 0
    colors = np.where(active, BLUE, CORAL)
    sizes = point_sizes(df)

    x_max = max(1.0, float(np.nanmax(df["abs_active_passive_diff_hz"].to_numpy())))
    bins = np.linspace(0, x_max * 1.05, 24)
    inner = slot.subgridspec(2, 1, height_ratios=(1.0, 4.15), hspace=0.055)
    ax_hist = fig.add_subplot(inner[0, 0])
    ax = fig.add_subplot(inner[1, 0], sharex=ax_hist)

    ax_hist.hist(df.loc[~sig, "abs_active_passive_diff_hz"], bins=bins, color="#E3E7E5", edgecolor="white", linewidth=0.25)
    ax_hist.hist(df.loc[sig & active, "abs_active_passive_diff_hz"], bins=bins, color=BLUE, alpha=0.88, edgecolor="white", linewidth=0.25)
    ax_hist.hist(df.loc[sig & ~active, "abs_active_passive_diff_hz"], bins=bins, color=CORAL, alpha=0.88, edgecolor="white", linewidth=0.25)
    ax_hist.tick_params(axis="x", labelbottom=False, length=2.5, width=0.55)
    ax_hist.tick_params(axis="y", labelsize=5.2, length=2.5, width=0.55)
    ax_hist.set_ylabel("Entries" if left_column else "", fontsize=5.6, labelpad=3)

    ax.scatter(
        df.loc[~sig, "abs_active_passive_diff_hz"],
        df.loc[~sig, "depth_um_median"],
        s=sizes[~sig] * 0.30,
        c=GRAY,
        alpha=0.38,
        linewidths=0,
        rasterized=False,
        zorder=1,
    )
    ax.scatter(
        df.loc[sig, "abs_active_passive_diff_hz"],
        df.loc[sig, "depth_um_median"],
        s=sizes[sig] * 0.70,
        c=colors[sig],
        edgecolors=np.where(good[sig], "black", "#707875"),
        linewidths=np.where(good[sig], 0.58, 0.25),
        alpha=0.96,
        rasterized=False,
        zorder=3,
    )
    ax.set_xlim(0, x_max * 1.18)
    ax.invert_yaxis()
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.tick_params(labelsize=5.35, length=2.7, width=0.55)
    ax.set_xlabel("|Active − passive response| (Hz)", fontsize=5.8, labelpad=3)
    ax.set_ylabel("Probe depth (µm)" if left_column else "", fontsize=5.8, labelpad=3)

    n_entries = int(len(df))
    n_runs = int(df["run"].nunique())
    n_nominal = int(sig.sum())
    n_fdr = int(fdr.sum())
    run_word = "run" if n_runs == 1 else "runs"
    ax_hist.text(
        -0.17,
        1.30,
        letter,
        transform=ax_hist.transAxes,
        fontsize=8.6,
        fontweight="bold",
        color="#111111",
        ha="left",
        va="top",
    )
    ax_hist.text(
        -0.035,
        1.30,
        subject,
        transform=ax_hist.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    ax_hist.text(
        1.00,
        1.30,
        f"{n_entries} entries · {n_runs} {run_word} · {n_nominal} nominal",
        transform=ax_hist.transAxes,
        fontsize=5.35,
        color=MUTED,
        ha="right",
        va="top",
    )
    fig.canvas.draw()
    add_candidate_labels(ax, df.loc[sig], include_run=n_runs > 1)
    return {
        "entries": n_entries,
        "runs": n_runs,
        "nominal": n_nominal,
        "fdr": n_fdr,
    }


def add_shared_key(ax: plt.Axes) -> None:
    ax.set_axis_off()
    card = FancyBboxPatch(
        (0.04, 0.05),
        0.92,
        0.90,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=PALE,
        edgecolor="#D4DFDB",
        linewidth=0.7,
    )
    ax.add_patch(card)
    ax.text(0.10, 0.88, "Shared key", transform=ax.transAxes, fontsize=7.8, fontweight="bold", color=INK, va="top")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=BLUE, markeredgecolor="#707875", markeredgewidth=0.35, markersize=5.2, label="Nominal P < 0.05, active > passive"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CORAL, markeredgecolor="#707875", markeredgewidth=0.35, markersize=5.2, label="Nominal P < 0.05, passive > active"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=GRAY, markeredgecolor="none", markersize=5.2, label="Not nominal P < 0.05"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.7, markersize=5.2, label="Black edge: Kilosort good label"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.79),
        fontsize=5.7,
        handletextpad=0.7,
        labelspacing=0.55,
        borderaxespad=0,
    )
    ax.text(
        0.10,
        0.48,
        "Point size scales with spike count.\n"
        "Response = mean 0–0.8 s minus −1–0 s baseline.\n"
        "Two-sided Mann–Whitney U test; nominal P < 0.05.\n"
        "No entry survived within-run BH-FDR at q < 0.05.",
        transform=ax.transAxes,
        fontsize=5.65,
        color="#303936",
        va="top",
        linespacing=1.45,
    )
    ax.text(
        0.10,
        0.20,
        "Horizontal scales are patient-specific. Probe depth is an\n"
        "unregistered recording coordinate, not a cortical-layer label.",
        transform=ax.transAxes,
        fontsize=5.65,
        color=MUTED,
        style="italic",
        va="top",
        linespacing=1.4,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width_in = 183 / 25.4
    height_in = 203 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    outer = fig.add_gridspec(
        3,
        2,
        left=0.085,
        right=0.985,
        bottom=0.055,
        top=0.970,
        wspace=0.28,
        hspace=0.42,
    )
    slots = (outer[0, 0], outer[0, 1], outer[1, 0], outer[1, 1], outer[2, 0])
    summaries = []
    for idx, (subject, letter, slot) in enumerate(zip(SUBJECTS, PANEL_LETTERS, slots)):
        summaries.append(add_patient_panel(fig, slot, subject, letter, left_column=idx % 2 == 0))
    add_shared_key(fig.add_subplot(outer[2, 1]))

    stem = OUT_DIR / "Supplementary_Figure_patient_resolved_probe_depth"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    tiff_path = stem.with_suffix(".tiff")
    fig.savefig(tiff_path, dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # Nature-family production systems are most predictable with an opaque RGB
    # TIFF rather than Matplotlib's default RGBA TIFF.
    with Image.open(tiff_path) as image:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        rgb.save(tiff_path, compression="tiff_lzw", dpi=(600, 600))

    print(
        {
            "output_stem": str(stem),
            "final_size_mm": [183, 203],
            "subjects": dict(zip(SUBJECTS, summaries)),
        }
    )


if __name__ == "__main__":
    main()
