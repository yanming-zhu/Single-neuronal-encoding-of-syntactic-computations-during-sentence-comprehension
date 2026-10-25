#!/usr/bin/env python3
"""Build 60 x 40 mm Nature-style Fig. 1b from corrected-onset source data.

The compact 2 x 2 panel shows two active-dominant and two passive-dominant
representative units. Qualifying intervals use pointwise one-way ANOVA
p < 0.05 / 2 and at least six consecutive 1-ms bins, without gap filling.
"""

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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analyze_active_passive_corrected_timepoint_screen_20260722 import (
    ALPHA,
    MIN_CONSECUTIVE_BINS,
    _task_dir,
    extract_smoothed_epochs,
    load_trials,
)


OUT = (
    ROOT
    / "reports"
    / "nature_science_draft"
    / "assembled_figures_20260805"
    / "subplots"
)
SOURCE_OUT = OUT / "source_data"
SEGMENT_PATH = (
    ROOT
    / "outputs"
    / "active_passive_corrected_timepoint_screen_20260722"
    / "qualifying_segments.csv"
)

STEM = "fig1b_function_word_aligned_4examples_60x40mm"
ACTIVE = "#2F78B7"
PASSIVE = "#D95F4C"
TEAL_LIGHT = "#DDF0EC"
INK = "#17312F"
GREY = "#707876"
GRID = "#E4E9E7"

FIGURE_WIDTH_MM = 60.0
FIGURE_HEIGHT_MM = 40.0
PLOT_START_MS = -150
PLOT_STOP_MS = 750

# Column 1: active-dominant examples. Column 2: passive-dominant examples.
# The first three units are KS-good; unit 37 is the strongest source-defined
# passive-dominant MUA example in the corrected-onset representative table.
REPRESENTATIVES = (
    {
        "row": 0,
        "column": 0,
        "direction": "active_gt_passive",
        "direction_label": "Active > passive",
        "run": "gz005t0_g3_actpas_run3",
        "run_label": "gz005 r3",
        "unit_id": 142,
    },
    {
        "row": 1,
        "column": 0,
        "direction": "active_gt_passive",
        "direction_label": "Active > passive",
        "run": "gz002_g0_actpas_run2",
        "run_label": "gz002 r2",
        "unit_id": 106,
    },
    {
        "row": 0,
        "column": 1,
        "direction": "passive_gt_active",
        "direction_label": "Active < passive",
        "run": "gz003t2_g0_actpas_run4",
        "run_label": "gz003 r4",
        "unit_id": 116,
    },
    {
        "row": 1,
        "column": 1,
        "direction": "passive_gt_active",
        "direction_label": "Active < passive",
        "run": "gz001t_g0_actpas_run1",
        "run_label": "gz001 r1",
        "unit_id": 37,
    },
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 5.0,
            "axes.labelsize": 5.0,
            "axes.titlesize": 4.6,
            "xtick.labelsize": 4.2,
            "ytick.labelsize": 4.2,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def load_unit_source(run: str, unit_id: int) -> tuple[pd.DataFrame, str, np.ndarray, np.ndarray]:
    subject = run[:5]
    trials = load_trials(subject, run)
    archive_path = _task_dir(subject) / run / "firing_rate_1000Hz/firing_rate_1000Hz_all_clusters.npz"
    archive = np.load(archive_path, allow_pickle=True)
    unit_ids = archive["unit_ids"].astype(int)
    matched = np.flatnonzero(unit_ids == int(unit_id))
    if len(matched) != 1:
        raise ValueError(f"{run}: unit {unit_id} is not uniquely present")
    unit_index = int(matched[0])
    ks_label = str(archive["ks_labels"].astype(str)[unit_index])
    epochs = extract_smoothed_epochs(
        archive["firing_rate_hz"].astype(np.float32, copy=False),
        trials["corrected_event_bin"].to_numpy(int),
    )[:, unit_index, :]
    time_ms = np.arange(-1000, 800, dtype=int)
    return trials, ks_label, time_ms, epochs


def select_segments(
    segments: pd.DataFrame,
    *,
    run: str,
    unit_id: int,
    direction: str,
) -> pd.DataFrame:
    chosen = segments.loc[
        segments["alignment"].eq("corrected")
        & segments["run"].eq(run)
        & segments["unit_id"].eq(unit_id)
        & segments["direction"].eq(direction)
        & segments["start_ms"].lt(PLOT_STOP_MS)
        & segments["stop_ms_exclusive"].gt(0)
    ].copy()
    chosen["plot_start_ms"] = chosen["start_ms"].clip(lower=0)
    chosen["plot_stop_ms"] = chosen["stop_ms_exclusive"].clip(upper=PLOT_STOP_MS)
    chosen["plot_duration_ms"] = chosen["plot_stop_ms"] - chosen["plot_start_ms"]
    chosen = chosen.loc[chosen["plot_duration_ms"].ge(MIN_CONSECUTIVE_BINS)]
    if chosen.empty:
        raise ValueError(f"{run} unit {unit_id}: no qualifying corrected segment")
    return chosen.sort_values("plot_start_ms").reset_index(drop=True)


def segment_label(segments: pd.DataFrame) -> str:
    return ", ".join(
        f"{int(row.plot_start_ms)}-{int(row.plot_stop_ms)} ms"
        for row in segments.itertuples()
    )


def significant_mask(time_ms: np.ndarray, segments: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(time_ms.shape, dtype=bool)
    for row in segments.itertuples():
        mask |= (time_ms >= int(row.plot_start_ms)) & (time_ms < int(row.plot_stop_ms))
    return mask


def build_figure() -> tuple[Path, Path, Path]:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    all_segments = pd.read_csv(SEGMENT_PATH)

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.135,
        right=0.985,
        bottom=0.165,
        top=0.800,
        hspace=0.55,
        wspace=0.30,
    )

    fig.text(0.018, 0.982, "b", fontsize=8.0, fontweight="bold", ha="left", va="top")
    fig.text(
        0.290,
        0.846,
        "Active > passive",
        fontsize=5.4,
        fontweight="bold",
        color=ACTIVE,
        ha="center",
        va="bottom",
    )
    fig.text(
        0.785,
        0.846,
        "Active < passive",
        fontsize=5.4,
        fontweight="bold",
        color=PASSIVE,
        ha="center",
        va="bottom",
    )
    legend_handles = [
        Line2D([0], [0], color=ACTIVE, lw=1.2, label="Active"),
        Line2D([0], [0], color=PASSIVE, lw=1.2, label="Passive"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.985),
        ncol=2,
        fontsize=4.6,
        handlelength=1.7,
        columnspacing=1.4,
        borderaxespad=0,
    )
    fig.text(
        0.025,
        0.480,
        "Firing rate (Hz)",
        fontsize=4.8,
        rotation=90,
        ha="center",
        va="center",
    )
    fig.text(
        0.555,
        0.038,
        "Time from corrected function-word onset (s)",
        fontsize=4.7,
        ha="center",
        va="bottom",
    )

    trace_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    statistics_rows: list[dict[str, object]] = []

    for representative in REPRESENTATIVES:
        row = int(representative["row"])
        column = int(representative["column"])
        run = str(representative["run"])
        run_label = str(representative["run_label"])
        unit_id = int(representative["unit_id"])
        direction = str(representative["direction"])
        direction_label = str(representative["direction_label"])
        trials, ks_label, time_ms, epochs = load_unit_source(run, unit_id)
        selected_segments = select_segments(
            all_segments,
            run=run,
            unit_id=unit_id,
            direction=direction,
        )
        interval_label = segment_label(selected_segments)
        interval_mask = significant_mask(time_ms, selected_segments)
        plot_mask = (time_ms >= PLOT_START_MS) & (time_ms <= PLOT_STOP_MS)
        conditions = trials["condition"].astype(str).to_numpy()
        active = epochs[conditions == "active"]
        passive = epochs[conditions == "passive"]
        active_mean = active.mean(axis=0)
        passive_mean = passive.mean(axis=0)
        active_sem = active.std(axis=0, ddof=1) / np.sqrt(active.shape[0])
        passive_sem = passive.std(axis=0, ddof=1) / np.sqrt(passive.shape[0])
        trial_means = epochs[:, interval_mask].mean(axis=1)
        signed_delta = float(active_mean[interval_mask].mean() - passive_mean[interval_mask].mean())

        expected_positive = direction == "active_gt_passive"
        if (signed_delta > 0) != expected_positive:
            raise ValueError(f"{run} unit {unit_id}: effect direction mismatch")

        ax = fig.add_subplot(grid[row, column])
        for segment in selected_segments.itertuples():
            ax.axvspan(
                int(segment.plot_start_ms) / 1000.0,
                int(segment.plot_stop_ms) / 1000.0,
                color=TEAL_LIGHT,
                lw=0,
                zorder=0,
            )
        ax.axvline(0, color=INK, lw=0.55, ls="--", zorder=1)
        time_s = time_ms[plot_mask] / 1000.0
        ax.plot(time_s, active_mean[plot_mask], color=ACTIVE, lw=0.95, zorder=3)
        ax.fill_between(
            time_s,
            np.maximum(0, active_mean[plot_mask] - active_sem[plot_mask]),
            active_mean[plot_mask] + active_sem[plot_mask],
            color=ACTIVE,
            alpha=0.16,
            lw=0,
            zorder=2,
        )
        ax.plot(time_s, passive_mean[plot_mask], color=PASSIVE, lw=0.95, zorder=3)
        ax.fill_between(
            time_s,
            np.maximum(0, passive_mean[plot_mask] - passive_sem[plot_mask]),
            passive_mean[plot_mask] + passive_sem[plot_mask],
            color=PASSIVE,
            alpha=0.16,
            lw=0,
            zorder=2,
        )
        ax.set_xlim(PLOT_START_MS / 1000.0, PLOT_STOP_MS / 1000.0)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0.0, 0.4, 0.7])
        ax.grid(axis="y", color=GRID, lw=0.38)
        ax.tick_params(axis="both", length=1.8, width=0.5, pad=1.2)
        ax.set_title(
            f"{run_label} · u{unit_id} · Δsig {signed_delta:+.1f} Hz",
            loc="left",
            color=INK,
            fontweight="bold",
            pad=2.0,
        )
        if row == 0:
            ax.tick_params(axis="x", labelbottom=False)

        for condition, mean_values, sem_values in (
            ("active", active_mean, active_sem),
            ("passive", passive_mean, passive_sem),
        ):
            for index in np.flatnonzero(plot_mask):
                trace_rows.append(
                    {
                        "panel_direction": direction_label,
                        "subject": run[:5],
                        "run": run,
                        "unit_id": unit_id,
                        "ks_label": ks_label,
                        "condition": condition,
                        "time_ms": int(time_ms[index]),
                        "mean_firing_rate_hz": float(mean_values[index]),
                        "sem_firing_rate_hz": float(sem_values[index]),
                        "in_qualifying_interval": bool(interval_mask[index]),
                    }
                )
        for trial_position, trial in trials.reset_index(drop=True).iterrows():
            trial_rows.append(
                {
                    "panel_direction": direction_label,
                    "subject": run[:5],
                    "run": run,
                    "unit_id": unit_id,
                    "ks_label": ks_label,
                    "trial_index": int(trial["trial_index"]),
                    "condition": str(trial["condition"]),
                    "function_word": str(trial["target_word"]),
                    "qualifying_intervals": interval_label,
                    "interval_mean_firing_rate_hz": float(trial_means[trial_position]),
                }
            )
        statistics_rows.append(
            {
                "panel_direction": direction_label,
                "subject": run[:5],
                "run": run,
                "unit_id": unit_id,
                "ks_label": ks_label,
                "n_active_trials": int(active.shape[0]),
                "n_passive_trials": int(passive.shape[0]),
                "test": "two-sided one-way ANOVA at each 1-ms bin",
                "pointwise_alpha": float(ALPHA),
                "minimum_consecutive_bins": int(MIN_CONSECUTIVE_BINS),
                "qualifying_intervals": interval_label,
                "qualifying_duration_ms": int(interval_mask.sum()),
                "active_minus_passive_hz_in_qualifying_intervals": signed_delta,
                "minimum_pointwise_p_in_qualifying_intervals": float(selected_segments["min_p"].min()),
                "selection_note": "source-defined representative unit; descriptive example",
            }
        )

    png_path = OUT / f"{STEM}.png"
    pdf_path = OUT / f"{STEM}.pdf"
    svg_path = OUT / f"{STEM}.svg"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    pd.DataFrame(trace_rows).to_csv(SOURCE_OUT / f"{STEM}_traces.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(SOURCE_OUT / f"{STEM}_trials.csv", index=False)
    pd.DataFrame(statistics_rows).to_csv(SOURCE_OUT / f"{STEM}_statistics.csv", index=False)
    return png_path, pdf_path, svg_path


if __name__ == "__main__":
    for output_path in build_figure():
        print(output_path)
