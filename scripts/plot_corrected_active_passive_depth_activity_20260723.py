#!/usr/bin/env python3
"""Depth-versus-activity plots using the corrected active/passive screen."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCREEN_PATH = ROOT / "outputs/active_passive_corrected_timepoint_screen_20260722/unit_crosswalk.csv"
OUTPUT = ROOT / "reports/active_passive_depth_corrected_screen_20260723"
FIGURES = OUTPUT / "figures"
SOURCE = OUTPUT / "source_data"

PATIENTS = ("gz001", "gz002", "gz003", "gz004", "gz005")
RUN_ORDER = (
    "gz001t_g0_actpas_run1",
    "gz002_g0_actpas_run2",
    "gz003t2_g0_actpas_run1",
    "gz003t2_g0_actpas_run4",
    "gz004t1_g0_actpas_run0",
    "gz004t1_g3_actpas_run3",
    "gz005t0_g0_actpas_run0",
    "gz005t0_g3_actpas_run3",
)
RUN_SHORT = {
    "gz001t_g0_actpas_run1": "run1",
    "gz002_g0_actpas_run2": "run2",
    "gz003t2_g0_actpas_run1": "run1",
    "gz003t2_g0_actpas_run4": "run4",
    "gz004t1_g0_actpas_run0": "run0",
    "gz004t1_g3_actpas_run3": "run3",
    "gz005t0_g0_actpas_run0": "run0",
    "gz005t0_g3_actpas_run3": "run3",
}
RUN_MARKERS = ("o", "D", "s")

ACTIVE = "#2E86C1"
PASSIVE = "#D55E00"
BACKGROUND = "#C8CCCE"
INK = "#17322F"
GRAY = "#6D7775"
LIGHT = "#DCE4E1"
PAPER = "#FBFAF6"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def run_number(run: str) -> str:
    match = re.search(r"run(\d+)$", str(run))
    return match.group(1) if match else str(run)


def load_table() -> pd.DataFrame:
    screen = pd.read_csv(SCREEN_PATH)
    required = {
        "subject",
        "run",
        "unit_id",
        "ks_label",
        "corrected_contiguous_gt5ms",
        "corrected_direction",
        "corrected_window_active_minus_passive_bc_hz",
        "corrected_longest_segment_ms",
    }
    missing = sorted(required - set(screen.columns))
    if missing:
        raise ValueError(f"{SCREEN_PATH} lacks {missing}")
    depth_parts = []
    for patient in PATIENTS:
        path = (
            ROOT
            / f"neuropixel2025/data/{patient}/processed/ActPasSynSentence"
            / "event_related_active_passive/depth_activation"
            / f"{patient}_unit_depth_activation_metrics.csv"
        )
        depth = pd.read_csv(path)[["run", "unit_id", "depth_um_median"]].copy()
        depth["subject"] = patient
        depth_parts.append(depth)
    depths = pd.concat(depth_parts, ignore_index=True)
    table = screen.merge(
        depths,
        on=["subject", "run", "unit_id"],
        how="left",
        validate="one_to_one",
    )
    if len(table) != 1782 or table["depth_um_median"].isna().any():
        raise ValueError("Depth merge did not retain all 1,782 registered unit/run entries")
    table["is_significant"] = table["corrected_contiguous_gt5ms"].astype(bool)
    table["activity_abs_hz"] = table["corrected_window_active_minus_passive_bc_hz"].abs()
    table["direction_label"] = table["corrected_direction"].map(
        {
            "active_gt_passive": "Active > Passive",
            "passive_gt_active": "Active < Passive",
            "not_significant": "Not screen-positive",
        }
    )
    run_index = {run: index for index, run in enumerate(RUN_ORDER)}
    table["run_order"] = table["run"].map(run_index)
    if table["run_order"].isna().any():
        raise ValueError("Unexpected run in merged table")
    table = table.sort_values(["run_order", "depth_um_median", "unit_id"]).reset_index(drop=True)
    table.to_csv(SOURCE / "corrected_active_passive_depth_activity_units.csv", index=False)
    return table


def nice_xmax(value: float) -> float:
    if value <= 10:
        step = 2.0
    elif value <= 40:
        step = 5.0
    else:
        step = 10.0
    return max(step, math.ceil(value / step) * step)


def marker_map(frame: pd.DataFrame) -> dict[str, str]:
    runs = [run for run in RUN_ORDER if run in set(frame["run"].astype(str))]
    return {run: RUN_MARKERS[index] for index, run in enumerate(runs)}


def patient_counts(frame: pd.DataFrame) -> tuple[int, int, int]:
    selected = frame.loc[frame["is_significant"]]
    return (
        len(selected),
        int(selected["corrected_direction"].eq("active_gt_passive").sum()),
        int(selected["corrected_direction"].eq("passive_gt_active").sum()),
    )


def label_positions(depths: np.ndarray, max_depth: float) -> np.ndarray:
    values = np.asarray(depths, dtype=float)
    if len(values) <= 1:
        return values.copy()
    order = np.argsort(values)
    sorted_values = values[order]
    minimum_separation = max(42.0, min(85.0, max_depth / (len(values) + 8)))
    positions = sorted_values.copy()
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + minimum_separation)
    if positions[-1] > max_depth:
        positions[-1] = max_depth
        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(positions[index], positions[index + 1] - minimum_separation)
    if positions[0] < 0:
        positions -= positions[0]
    result = np.empty_like(positions)
    result[order] = positions
    return result


def direction_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=ACTIVE, markeredgecolor="none", markersize=7, label="Corrected screen: Active > Passive"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PASSIVE, markeredgecolor="none", markersize=7, label="Corrected screen: Active < Passive"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=BACKGROUND, markeredgecolor="none", alpha=0.55, markersize=6, label="Not screen-positive"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black", markersize=7, label="Black edge = KS good"),
    ]


def session_handles(frame: pd.DataFrame, markers: dict[str, str]) -> list[Line2D]:
    handles = []
    for run in [run for run in RUN_ORDER if run in markers]:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=markers[run],
                linestyle="none",
                markerfacecolor="#FFFFFF",
                markeredgecolor=GRAY,
                markersize=6.5,
                label=RUN_SHORT[run],
            )
        )
    return handles


def scatter_patient(
    ax: plt.Axes,
    frame: pd.DataFrame,
    markers: dict[str, str],
    *,
    x_cap: float | None = None,
    label_units: bool = False,
) -> int:
    clipped_count = 0
    for run, marker in markers.items():
        run_frame = frame.loc[frame["run"].eq(run)]
        x = run_frame["activity_abs_hz"].to_numpy(float)
        if x_cap is not None:
            clipped = x > x_cap
            clipped_count += int(clipped.sum())
            x = np.minimum(x, x_cap)
        else:
            clipped = np.zeros(len(x), dtype=bool)
        normal = ~clipped
        ax.scatter(
            x[normal],
            run_frame.loc[normal, "depth_um_median"],
            s=22,
            marker=marker,
            color=BACKGROUND,
            alpha=0.34,
            edgecolors="none",
            zorder=1,
        )
        if clipped.any():
            ax.scatter(
                x[clipped],
                run_frame.loc[clipped, "depth_um_median"],
                s=24,
                marker=">",
                color=BACKGROUND,
                alpha=0.55,
                edgecolors="none",
                zorder=2,
            )

    significant = frame.loc[frame["is_significant"]].copy()
    for run, marker in markers.items():
        run_sig = significant.loc[significant["run"].eq(run)]
        for direction, color in (("active_gt_passive", ACTIVE), ("passive_gt_active", PASSIVE)):
            direction_rows = run_sig.loc[run_sig["corrected_direction"].eq(direction)]
            if direction_rows.empty:
                continue
            good = direction_rows["ks_label"].eq("good").to_numpy(bool)
            x = direction_rows["activity_abs_hz"].to_numpy(float)
            if x_cap is not None:
                x = np.minimum(x, x_cap)
            ax.scatter(
                x[~good],
                direction_rows.loc[~good, "depth_um_median"],
                s=45,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.45,
                alpha=0.94,
                zorder=4,
            )
            ax.scatter(
                x[good],
                direction_rows.loc[good, "depth_um_median"],
                s=54,
                marker=marker,
                color=color,
                edgecolors="black",
                linewidths=1.05,
                alpha=0.98,
                zorder=5,
            )

    if label_units and not significant.empty:
        max_depth = float(frame["depth_um_median"].max()) * 1.025
        label_y = label_positions(significant["depth_um_median"].to_numpy(float), max_depth)
        duplicated = set(
            significant.loc[significant.duplicated("unit_id", keep=False), "unit_id"].astype(int)
        )
        x_range = float(frame["activity_abs_hz"].max())
        offset = max(0.12, x_range * 0.010)
        font_size = 6.1 if len(significant) > 55 else 6.8 if len(significant) > 35 else 7.5
        for row, y_text in zip(significant.itertuples(index=False), label_y):
            label = str(int(row.unit_id))
            if int(row.unit_id) in duplicated:
                label = f"r{run_number(str(row.run))}:{label}"
            x_point = float(row.activity_abs_hz)
            ax.plot(
                [x_point, x_point + offset * 0.72],
                [float(row.depth_um_median), y_text],
                color=GRAY,
                linewidth=0.35,
                alpha=0.60,
                zorder=5,
            )
            ax.text(
                x_point + offset,
                y_text,
                label,
                va="center",
                ha="left",
                fontsize=font_size,
                color=INK,
                zorder=6,
            )
    return clipped_count


def plot_individual(patient: str, frame: pd.DataFrame) -> Path:
    markers = marker_map(frame)
    total_sig, active_sig, passive_sig = patient_counts(frame)
    x_max = nice_xmax(float(frame["activity_abs_hz"].max()) * 1.05)
    max_depth = nice_xmax(float(frame["depth_um_median"].max()) * 1.02)

    fig = plt.figure(figsize=(9.7, 8.6), facecolor=PAPER)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 5.9], hspace=0.08, left=0.12, right=0.96, top=0.86, bottom=0.16)
    ax_hist = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_hist)
    bins = np.linspace(0, x_max, 18)
    ax_hist.hist(frame["activity_abs_hz"], bins=bins, color=BACKGROUND, alpha=0.55, edgecolor="white", label="All units")
    for direction, color in (("active_gt_passive", ACTIVE), ("passive_gt_active", PASSIVE)):
        values = frame.loc[frame["is_significant"] & frame["corrected_direction"].eq(direction), "activity_abs_hz"]
        ax_hist.hist(values, bins=bins, histtype="step", color=color, linewidth=1.7)
    ax_hist.set_ylabel("Units")
    ax_hist.tick_params(axis="x", labelbottom=False, length=0)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.grid(axis="y", color=LIGHT, linewidth=0.6)

    scatter_patient(ax, frame, markers, label_units=True)
    ax.set_xlim(0, x_max)
    ax.set_ylim(max_depth, 0)
    ax.set_xlabel("|Active − passive response| (Hz, baseline-corrected)")
    ax.set_ylabel("Estimated depth on Neuropixels probe (µm)")
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    title = f"{patient} ActPasSynSentence: depth and corrected active/passive activity"
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.965)
    run_text = " · ".join(f"{markers[run]} {RUN_SHORT[run]}" for run in markers)
    fig.text(
        0.12,
        0.895,
        f"n={len(frame)} unit/run entries · corrected screen={total_sig} (Active>Passive {active_sig}; Active<Passive {passive_sig}) · sessions: {run_text}",
        fontsize=10,
        color=GRAY,
    )

    handles = direction_handles() + session_handles(frame, markers)
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.2, ncol=1)
    fig.text(
        0.5,
        0.045,
        "Screen-positive = corrected function-word onset, Gaussian SDF σ=25 ms, pointwise p<0.025 for ≥6 consecutive 1-ms bins in 0–0.8 s. "
        "Colors use the direction within qualifying bins; unit IDs are annotated; no stripe-QC rows are removed.",
        ha="center",
        fontsize=8.7,
        color=GRAY,
    )
    path = FIGURES / f"Fig_{patient}_corrected_active_passive_depth_activity.png"
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def plot_stacked(table: pd.DataFrame) -> tuple[Path, float, dict[str, int]]:
    significant_max = float(table.loc[table["is_significant"], "activity_abs_hz"].max())
    x_cap = max(20.0, nice_xmax(significant_max * 1.05))
    fig, axes = plt.subplots(
        len(PATIENTS),
        1,
        figsize=(9.8, 12.4),
        sharex=True,
        facecolor=PAPER,
        gridspec_kw={"hspace": 0.16, "left": 0.13, "right": 0.965, "top": 0.89, "bottom": 0.12},
    )
    clipped_counts: dict[str, int] = {}
    for index, (ax, patient) in enumerate(zip(axes, PATIENTS)):
        frame = table.loc[table["subject"].eq(patient)]
        markers = marker_map(frame)
        total_sig, active_sig, passive_sig = patient_counts(frame)
        clipped_counts[patient] = scatter_patient(ax, frame, markers, x_cap=x_cap, label_units=False)
        max_depth = nice_xmax(float(frame["depth_um_median"].max()) * 1.02)
        ax.set_ylim(max_depth, 0)
        ax.set_xlim(0, x_cap)
        ax.grid(axis="y", color=LIGHT, linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        session_text = " / ".join(
            f"{markers[run]} {RUN_SHORT[run]}" for run in markers
        )
        ax.set_title(
            f"{patient}   n={len(frame)} · significant={total_sig} (A>P {active_sig}; A<P {passive_sig}) · {session_text}",
            loc="left",
            fontsize=10.8,
            fontweight="bold",
            pad=4,
        )
        ax.set_ylabel("Depth\n(µm)")
        if index < len(PATIENTS) - 1:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        if clipped_counts[patient]:
            ax.text(
                0.995,
                0.04,
                f"{clipped_counts[patient]} gray units >{x_cap:g} Hz clipped",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=GRAY,
            )
    axes[-1].set_xlabel("|Active − passive response| (Hz, baseline-corrected)")

    fig.suptitle(
        "Corrected active/passive significant units across five patients",
        fontsize=18.5,
        fontweight="bold",
        y=0.972,
    )
    fig.legend(
        handles=direction_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        fontsize=8.8,
    )
    fig.text(
        0.5,
        0.03,
        "Corrected screen: pointwise p<0.025 for ≥6 consecutive ms in 0–0.8 s. Marker shapes distinguish sessions within a patient. "
        f"Shared x display is clipped at {x_cap:g} Hz; every significant unit is within range. Only the bottom shared x-axis is shown.",
        ha="center",
        color=GRAY,
        fontsize=8.6,
    )
    path = FIGURES / "Fig_all_patients_corrected_active_passive_depth_activity_stacked.png"
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path, x_cap, clipped_counts


def main() -> None:
    configure()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    table = load_table()
    if int(table["is_significant"].sum()) != 172:
        raise ValueError("Expected 172 corrected screen-positive units")
    individual = {}
    for patient in PATIENTS:
        path = plot_individual(patient, table.loc[table["subject"].eq(patient)].copy())
        individual[patient] = str(path)
    stacked, x_cap, clipped = plot_stacked(table)
    summary = {
        "method": {
            "activity_metric": "absolute 0-0.8 s active-minus-passive response after -1-0 s baseline correction",
            "significance": "corrected function-word onset; Gaussian SDF sigma=25 ms; pointwise p<0.025 for >=6 consecutive 1-ms bins",
            "depth": "median estimated depth on Neuropixels probe in micrometres",
            "unit_of_observation": "registered unit/run entry",
        },
        "counts": {
            patient: {
                "n_units": int(len(frame)),
                "n_significant": patient_counts(frame)[0],
                "active_gt_passive": patient_counts(frame)[1],
                "passive_gt_active": patient_counts(frame)[2],
            }
            for patient, frame in table.groupby("subject", sort=False)
        },
        "stacked_display": {
            "x_cap_hz": x_cap,
            "clipped_nonsignificant_by_patient": clipped,
            "all_significant_within_x_cap": bool(
                table.loc[table["is_significant"], "activity_abs_hz"].le(x_cap).all()
            ),
        },
        "outputs": {
            "individual_figures": individual,
            "stacked_figure": str(stacked),
            "source_data": str(SOURCE / "corrected_active_passive_depth_activity_units.csv"),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
