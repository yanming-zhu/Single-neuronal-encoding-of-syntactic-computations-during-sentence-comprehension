#!/usr/bin/env python3
"""Build patient-resolved unit/run response-by-depth supplementary panels."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
SOURCE_DATA_DIR = Path(os.environ.get("TAN_SUPPLEMENTARY_SOURCE_DATA_DIR", HERE / "source_data")).expanduser().resolve()
SOURCE_DIR = SOURCE_DATA_DIR / "patient_depth_activation"
PNG_DIR = HERE / "assets" / "patient_depth_activation"
EXPORT_DIR = HERE / "new_figures" / "patient_depth_activation"
SUBJECTS = ("gz001", "gz002", "gz003", "gz004", "gz005")

BLUE = "#2B6CB0"
CORAL = "#C94F3D"
GRAY = "#D0D0D0"
INK = "#16312C"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
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
    n_spikes = pd.to_numeric(df[spike_column], errors="coerce").fillna(0).to_numpy()
    return np.clip(18 + np.sqrt(n_spikes) * 1.1, 20, 130)


def short_run_label(run_name: str) -> str:
    match = re.search(r"run(\d+)$", str(run_name))
    return f"r{match.group(1)}" if match else str(run_name)


def add_nonoverlapping_labels(ax: plt.Axes, df: pd.DataFrame, include_run: bool) -> None:
    """Place candidate unit IDs with deterministic local collision avoidance."""
    occupied: list[tuple[float, float, float, float]] = []
    near_x = 0.020 if include_run else 0.010
    offsets = [
        (near_x, 0.000),
        (near_x, 0.030),
        (near_x, -0.030),
        (0.030, 0.048),
        (0.030, -0.048),
        (-near_x, 0.000),
        (-near_x, 0.030),
        (-near_x, -0.030),
        (-0.034, 0.050),
        (-0.034, -0.050),
    ]
    to_axes = ax.transAxes.inverted().transform
    to_display = ax.transData.transform

    for row in df.sort_values(["depth_um_median", "abs_active_passive_diff_hz"]).itertuples(index=False):
        px, py = to_axes(to_display((row.abs_active_passive_diff_hz, row.depth_um_median)))
        label = (
            f"{short_run_label(row.run)}·u{int(row.unit_id)}"
            if include_run
            else f"u{int(row.unit_id)}"
        )
        half_w = 0.012 + 0.0055 * len(label)
        half_h = 0.017
        chosen = offsets[0]
        for dx, dy in offsets:
            cx, cy = px + dx, py + dy
            box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
            inside = box[0] >= 0 and box[2] <= 1 and box[1] >= 0 and box[3] <= 1
            collision = any(
                not (box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3])
                for old in occupied
            )
            if inside and not collision:
                chosen = (dx, dy)
                occupied.append(box)
                break
        dx, dy = chosen
        leader = include_run or float(np.hypot(dx, dy)) > 0.020
        ax.annotate(
            label,
            xy=(row.abs_active_passive_diff_hz, row.depth_um_median),
            xycoords="data",
            xytext=(px + dx, py + dy),
            textcoords=ax.transAxes,
            fontsize=6.2 if include_run else 6.8,
            color="#222222",
            ha="left" if dx >= 0 else "right",
            va="center",
            arrowprops=(
                {"arrowstyle": "-", "color": "#777777", "linewidth": 0.45, "shrinkA": 1, "shrinkB": 2}
                if leader
                else None
            ),
            zorder=5,
        )


def build_subject(subject: str) -> dict[str, object]:
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
    df = df.dropna(subset=["depth_um_median", "active_minus_passive_bc_hz", "abs_active_passive_diff_hz"])
    df = df.sort_values(["depth_um_median", "abs_active_passive_diff_hz"]).reset_index(drop=True)

    sig = bool_series(df["significant_uncorrected_p05"]).to_numpy()
    fdr = bool_series(df["significant_run_fdr05"]).to_numpy()
    good = (df["ks_label"].astype(str) == "good").to_numpy()
    active = df["active_minus_passive_bc_hz"].to_numpy() > 0
    colors = np.where(active, BLUE, CORAL)
    sizes = point_sizes(df)

    x_max = max(1.0, float(np.nanmax(df["abs_active_passive_diff_hz"].to_numpy())))
    bins = np.linspace(0, x_max * 1.05, 28)

    fig = plt.figure(figsize=(8.0, 5.8), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 5.2), hspace=0.08)
    ax_hist = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[1, 0], sharex=ax_hist)

    ax_hist.hist(df.loc[~sig, "abs_active_passive_diff_hz"], bins=bins, color="#E2E2E2", edgecolor="white")
    ax_hist.hist(df.loc[sig & active, "abs_active_passive_diff_hz"], bins=bins, color=BLUE, alpha=0.82, edgecolor="white")
    ax_hist.hist(df.loc[sig & ~active, "abs_active_passive_diff_hz"], bins=bins, color=CORAL, alpha=0.82, edgecolor="white")
    ax_hist.set_ylabel("Entries", fontsize=9)
    ax_hist.tick_params(axis="x", labelbottom=False)
    ax_hist.tick_params(labelsize=8)

    ax_scatter.scatter(
        df.loc[~sig, "abs_active_passive_diff_hz"],
        df.loc[~sig, "depth_um_median"],
        s=sizes[~sig] * 0.42,
        c=GRAY,
        alpha=0.42,
        linewidths=0,
        zorder=1,
    )
    ax_scatter.scatter(
        df.loc[sig, "abs_active_passive_diff_hz"],
        df.loc[sig, "depth_um_median"],
        s=sizes[sig],
        c=colors[sig],
        edgecolors=np.where(good[sig], "black", "#666666"),
        linewidths=np.where(good[sig], 0.8, 0.3),
        alpha=0.96,
        zorder=3,
    )

    ax_scatter.set_xlim(0, x_max * 1.16)
    ax_scatter.set_xlabel("|Active - passive response| (Hz, baseline-corrected)", fontsize=10)
    ax_scatter.set_ylabel("Estimated depth on Neuropixels probe (µm)", fontsize=10)
    ax_scatter.invert_yaxis()
    ax_scatter.grid(axis="y", color="0.88", linewidth=0.7)
    ax_scatter.tick_params(labelsize=8)
    add_nonoverlapping_labels(ax_scatter, df.loc[sig], include_run=int(df["run"].nunique()) > 1)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markeredgecolor="black", label="nominal P < 0.05, active > passive", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CORAL, markeredgecolor="black", label="nominal P < 0.05, passive > active", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GRAY, label="not nominal P < 0.05", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", label="black edge: Kilosort good", markersize=7),
    ]
    ax_scatter.legend(handles=legend_items, loc="lower right", fontsize=7.2, handletextpad=0.6, labelspacing=0.3)

    n_entries = int(len(df))
    n_runs = int(df["run"].nunique())
    n_nominal = int(sig.sum())
    n_fdr = int(fdr.sum())
    fig.suptitle(f"{subject}: unit/run responses across probe depth", fontsize=14, fontweight="bold", color=INK, y=0.975)
    fig.text(
        0.10,
        0.024,
        f"Each point is one Kilosort unit/run entry (n = {n_entries}; runs = {n_runs}). "
        f"Response: mean 0-0.8 s minus -1 to 0 s baseline.\n"
        f"{n_nominal} entries nominal P < 0.05; "
        f"{n_fdr} entries within-run BH-FDR q < 0.05.",
        fontsize=7.2,
        color="#303A37",
        linespacing=1.25,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.165, top=0.90)

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PNG_DIR / f"{subject}_depth_activation_left_map.png"
    export_base = EXPORT_DIR / f"{subject}_unit_run_response_by_depth"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(export_base.with_suffix(".svg"), facecolor="white")
    fig.savefig(export_base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(export_base.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)

    return {
        "subject": subject,
        "source_data": str(csv_path),
        "png": str(png_path),
        "svg": str(export_base.with_suffix(".svg")),
        "pdf": str(export_base.with_suffix(".pdf")),
        "tiff": str(export_base.with_suffix(".tiff")),
        "n_unit_run_entries": n_entries,
        "n_runs": n_runs,
        "n_nominal_p05": n_nominal,
        "n_within_run_fdr05": n_fdr,
    }


def main() -> None:
    summary = [build_subject(subject) for subject in SUBJECTS]
    metadata_path = EXPORT_DIR / "patient_depth_supplement_metadata.json"
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(metadata_path), "subjects": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
