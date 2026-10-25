#!/usr/bin/env python3
"""Baseline-corrected evoked timing with a matched-by-run comparison.

Analysis contract
-----------------
* all registered Kilosort good + MUA units are reconsidered;
* corrected function-word onset is used;
* each trial/unit SDF is centred by its own -1000 to 0 ms mean;
* pointwise active/passive tests cover -500 to 800 ms;
* p < 0.05 / 2 for at least six consecutive 1-ms bins;
* any significant segment beginning before and continuing after t=0 is
  classified as pre-existing and is not used as an evoked segment;
* each run contributes the median evoked latency in each preference direction;
* the paired directional inference is an exact Wilcoxon signed-rank test.

Runs are nested within patients, so the matched-run result remains exploratory.
"""

from __future__ import annotations

import gc
import json
import os
import warnings
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
from scipy import stats

from analyze_active_passive_corrected_timepoint_screen_20260722 import (
    ALPHA,
    BASELINE_START_MS,
    DATA_ROOT,
    MIN_CONSECUTIVE_BINS,
    RESPONSE_STOP_MS,
    RUN_ORDER,
    _task_dir,
    contiguous_segments,
    extract_smoothed_epochs,
    load_trials,
)


OUT_ROOT = (
    ROOT
    / "reports"
    / "nature_science_draft"
    / "assembled_figures_20260805"
)
OUT = OUT_ROOT / "subplots"
SOURCE = (
    OUT_ROOT
    / "source_data"
    / "fig3i_function_word_baseline_corrected_matched_run_60x40mm"
)
STEM = OUT / "fig3i_function_word_baseline_corrected_matched_run_60x40mm"

WIDTH_MM = 60.0
HEIGHT_MM = 40.0
PT = 8.0
PRETEST_START_MS = -500
ZERO_INDEX = -PRETEST_START_MS
ACTIVE = "#357ABD"
PASSIVE = "#D95F4C"
GRAY = "#A9AFB5"
INK = "#202426"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": PT,
            "axes.labelsize": PT,
            "axes.titlesize": PT,
            "xtick.labelsize": PT,
            "ytick.labelsize": PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def analyze_run(subject: str, run: str) -> pd.DataFrame:
    trials = load_trials(subject, run)
    fr_path = (
        _task_dir(subject)
        / run
        / "firing_rate_1000Hz"
        / "firing_rate_1000Hz_all_clusters.npz"
    )
    archive = np.load(fr_path, allow_pickle=True)
    firing_rate = archive["firing_rate_hz"].astype(np.float32, copy=False)
    unit_ids = archive["unit_ids"].astype(int)
    ks_labels = archive["ks_labels"].astype(str)
    epochs = extract_smoothed_epochs(
        firing_rate, trials["corrected_event_bin"].to_numpy(int)
    )
    del firing_rate, archive

    baseline_bins = -BASELINE_START_MS
    trial_baseline = epochs[:, :, :baseline_bins].mean(axis=2, keepdims=True)
    baseline_corrected = epochs - trial_baseline
    analysis_start_index = PRETEST_START_MS - BASELINE_START_MS
    analysis = baseline_corrected[:, :, analysis_start_index:]
    if analysis.shape[2] != RESPONSE_STOP_MS - PRETEST_START_MS:
        raise ValueError(f"{run}: unexpected analysis-window length {analysis.shape[2]}")

    conditions = trials["condition"].astype(str).to_numpy()
    active_mask = conditions == "active"
    passive_mask = conditions == "passive"
    active = analysis[active_mask]
    passive = analysis[passive_mask]
    if active.shape[0] != 10 or passive.shape[0] != 10:
        raise ValueError(f"{run}: expected 10 trials per condition")

    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            anova = stats.f_oneway(active, passive, axis=0)
    p_values = np.asarray(anova.pvalue, dtype=float)
    p_values[~np.isfinite(p_values)] = 1.0
    active_mean = active.mean(axis=0)
    passive_mean = passive.mean(axis=0)
    difference = active_mean - passive_mean
    passing = p_values < ALPHA

    raw_active_baseline = trial_baseline[active_mask, :, 0].mean(axis=0)
    raw_passive_baseline = trial_baseline[passive_mask, :, 0].mean(axis=0)
    response_start = -PRETEST_START_MS
    bc_response_active = active[:, :, response_start:].mean(axis=(0, 2))
    bc_response_passive = passive[:, :, response_start:].mean(axis=(0, 2))

    rows: list[dict[str, object]] = []
    for unit_index, unit_id in enumerate(unit_ids):
        all_segments = contiguous_segments(
            passing[unit_index], min_bins=MIN_CONSECUTIVE_BINS
        )
        crossing_segments = tuple(
            segment
            for segment in all_segments
            if segment.start_ms < ZERO_INDEX < segment.stop_ms
        )
        evoked_segments = tuple(
            segment for segment in all_segments if segment.start_ms >= ZERO_INDEX
        )
        evoked_mask = np.zeros(analysis.shape[2], dtype=bool)
        for segment in evoked_segments:
            evoked_mask[segment.start_ms : segment.stop_ms] = True

        positive_ms = int(np.sum(evoked_mask & (difference[unit_index] > 0)))
        negative_ms = int(np.sum(evoked_mask & (difference[unit_index] < 0)))
        if positive_ms > negative_ms:
            direction = "active_gt_passive"
        elif negative_ms > positive_ms:
            direction = "passive_gt_active"
        elif evoked_mask.any():
            direction = "mixed"
        else:
            direction = "not_significant"

        onset_ms = np.nan
        peak_ms = np.nan
        peak_effect_hz = np.nan
        if evoked_segments and direction in {
            "active_gt_passive",
            "passive_gt_active",
        }:
            onset_ms = float(evoked_segments[0].start_ms - ZERO_INDEX)
            candidate_indices = np.flatnonzero(evoked_mask)
            candidate_effect = difference[unit_index, candidate_indices]
            if direction == "active_gt_passive":
                local_peak = int(np.argmax(candidate_effect))
            else:
                local_peak = int(np.argmin(candidate_effect))
            peak_index = int(candidate_indices[local_peak])
            peak_ms = float(peak_index - ZERO_INDEX)
            peak_effect_hz = float(difference[unit_index, peak_index])

        rows.append(
            {
                "subject": subject,
                "run": run,
                "unit_id": int(unit_id),
                "ks_label": str(ks_labels[unit_index]),
                "baseline_window_ms": "-1000:0",
                "pointwise_test_window_ms": "-500:800",
                "pointwise_alpha": ALPHA,
                "minimum_consecutive_bins": MIN_CONSECUTIVE_BINS,
                "raw_active_baseline_mean_hz": float(raw_active_baseline[unit_index]),
                "raw_passive_baseline_mean_hz": float(raw_passive_baseline[unit_index]),
                "raw_baseline_active_minus_passive_hz": float(
                    raw_active_baseline[unit_index]
                    - raw_passive_baseline[unit_index]
                ),
                "baseline_corrected_response_active_minus_passive_hz": float(
                    bc_response_active[unit_index] - bc_response_passive[unit_index]
                ),
                "n_all_significant_segments": int(len(all_segments)),
                "n_preexisting_crossing_segments": int(len(crossing_segments)),
                "had_preexisting_crossing_segment": bool(crossing_segments),
                "n_evoked_segments": int(len(evoked_segments)),
                "has_evoked_segment": bool(evoked_segments),
                "evoked_active_gt_passive_ms": positive_ms,
                "evoked_passive_gt_active_ms": negative_ms,
                "evoked_direction": direction,
                "evoked_onset_ms": onset_ms,
                "evoked_peak_ms": peak_ms,
                "evoked_peak_active_minus_passive_hz": peak_effect_hz,
            }
        )

    del epochs, trial_baseline, baseline_corrected, analysis, active, passive
    gc.collect()
    return pd.DataFrame(rows)


def timing_long(units: pd.DataFrame) -> pd.DataFrame:
    selected = units.loc[
        units["has_evoked_segment"].astype(bool)
        & units["evoked_direction"].isin(
            ["active_gt_passive", "passive_gt_active"]
        )
    ].copy()
    rows: list[pd.DataFrame] = []
    for metric, column in (
        ("onset", "evoked_onset_ms"),
        ("peak", "evoked_peak_ms"),
    ):
        part = selected[
            ["subject", "run", "unit_id", "ks_label", "evoked_direction", column]
        ].rename(columns={"evoked_direction": "direction", column: "value_ms"})
        part["metric"] = metric
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def make_run_pairs(timing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        timing.groupby(["subject", "run", "metric", "direction"], sort=False)[
            "value_ms"
        ]
        .agg(n_units="size", median_ms="median")
        .reset_index()
    )
    medians = grouped.pivot(
        index=["subject", "run", "metric"],
        columns="direction",
        values="median_ms",
    ).reset_index()
    counts = grouped.pivot(
        index=["subject", "run", "metric"],
        columns="direction",
        values="n_units",
    ).reset_index()
    pairs = medians.merge(
        counts,
        on=["subject", "run", "metric"],
        suffixes=("_median_ms", "_n_units"),
        how="outer",
        validate="one_to_one",
    ).rename(
        columns={
            "active_gt_passive_median_ms": "active_preferring_median_ms",
            "passive_gt_active_median_ms": "passive_preferring_median_ms",
            "active_gt_passive_n_units": "n_active_preferring_units",
            "passive_gt_active_n_units": "n_passive_preferring_units",
        }
    )
    pairs["matched"] = pairs[
        ["active_preferring_median_ms", "passive_preferring_median_ms"]
    ].notna().all(axis=1)
    pairs["active_minus_passive_ms"] = (
        pairs["active_preferring_median_ms"]
        - pairs["passive_preferring_median_ms"]
    )
    return pairs, grouped


def matched_tests(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("onset", "peak"):
        part = pairs.loc[pairs["metric"].eq(metric) & pairs["matched"]].copy()
        active = part["active_preferring_median_ms"].to_numpy(float)
        passive = part["passive_preferring_median_ms"].to_numpy(float)
        difference = active - passive
        if len(part) < 2 or np.any(difference == 0):
            raise ValueError(
                f"{metric}: exact paired Wilcoxon requires >=2 non-zero matched differences"
            )
        wilcoxon = stats.wilcoxon(
            active,
            passive,
            alternative="less",
            zero_method="wilcox",
            method="exact",
        )
        paired_t = stats.ttest_rel(active, passive, alternative="less")
        rows.append(
            {
                "metric": metric,
                "matching_unit": "recording run",
                "within_run_endpoint": "median baseline-corrected evoked latency",
                "n_matched_runs": int(len(part)),
                "n_runs_active_earlier": int(np.sum(difference < 0)),
                "run_level_active_median_ms": float(np.median(active)),
                "run_level_passive_median_ms": float(np.median(passive)),
                "median_paired_difference_ms": float(np.median(difference)),
                "mean_paired_difference_ms": float(np.mean(difference)),
                "test": "exact one-sided paired Wilcoxon signed-rank",
                "alternative": "active-preferring run median < passive-preferring run median",
                "wilcoxon_w": float(wilcoxon.statistic),
                "wilcoxon_one_sided_p_value": float(wilcoxon.pvalue),
                "paired_t": float(paired_t.statistic),
                "paired_t_df": int(len(part) - 1),
                "paired_t_one_sided_p_value": float(paired_t.pvalue),
                "inference_note": "exploratory; runs are nested within five patients",
            }
        )
    return pd.DataFrame(rows)


def draw_panel(pairs: pd.DataFrame, tests: pd.DataFrame) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        sharey=True,
        facecolor="white",
    )
    for ax, metric in zip(axes, ("onset", "peak")):
        part = pairs.loc[pairs["metric"].eq(metric) & pairs["matched"]].copy()
        active = part["active_preferring_median_ms"].to_numpy(float)
        passive = part["passive_preferring_median_ms"].to_numpy(float)
        for active_value, passive_value in zip(active, passive):
            ax.plot(
                [0, 1],
                [active_value, passive_value],
                color=GRAY,
                lw=0.7,
                alpha=0.85,
                zorder=1,
            )
        ax.scatter(
            np.zeros_like(active),
            active,
            s=16,
            color=ACTIVE,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            np.ones_like(passive),
            passive,
            s=16,
            color=PASSIVE,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        for x, values in ((0, active), (1, passive)):
            median = float(np.median(values))
            ax.plot(
                [x - 0.16, x + 0.16],
                [median, median],
                color=INK,
                lw=1.2,
                zorder=4,
            )
        p_value = float(
            tests.loc[
                tests["metric"].eq(metric), "wilcoxon_one_sided_p_value"
            ].iloc[0]
        )
        ax.set_title(
            f"{metric.capitalize()}\nP = {p_value:.3f}",
            fontweight="bold",
            pad=3,
        )
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1], ["A-pref.", "P-pref."])
        ax.set_ylim(0, 820)
        ax.set_yticks([0, 200, 400, 600, 800])
        ax.grid(axis="y", color="#EEF0F2", lw=0.55)
    axes[0].set_ylabel("Run median latency (ms)")
    fig.text(
        0.035,
        0.88,
        "i",
        fontsize=PT,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.22, top=0.73, wspace=0.38)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for run in RUN_ORDER:
        subject = run[:5]
        frame = analyze_run(subject, run)
        frames.append(frame)
        selected = frame.loc[
            frame["has_evoked_segment"]
            & frame["evoked_direction"].isin(
                ["active_gt_passive", "passive_gt_active"]
            )
        ]
        print(
            f"{run}: {len(frame)} units, {len(selected)} directional evoked, "
            f"{int(frame['had_preexisting_crossing_segment'].sum())} with crossing segment",
            flush=True,
        )

    units = pd.concat(frames, ignore_index=True)
    timing = timing_long(units)
    pairs, grouped = make_run_pairs(timing)
    tests = matched_tests(pairs)

    units.to_csv(SOURCE / "fig3i_bc_evoked_all_unit_results.csv", index=False)
    timing.to_csv(SOURCE / "fig3i_bc_evoked_directional_timing.csv", index=False)
    grouped.to_csv(SOURCE / "fig3i_bc_evoked_run_direction_summary.csv", index=False)
    pairs.to_csv(SOURCE / "fig3i_bc_evoked_matched_run_pairs.csv", index=False)
    tests.to_csv(SOURCE / "fig3i_bc_evoked_matched_tests.csv", index=False)

    fig = draw_panel(pairs, tests)
    fig.savefig(f"{STEM}.pdf", facecolor="white")
    fig.savefig(f"{STEM}.svg", facecolor="white")
    fig.savefig(f"{STEM}.png", dpi=600, facecolor="white")
    fig.savefig(
        f"{STEM}.tiff",
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    selected = units.loc[
        units["has_evoked_segment"]
        & units["evoked_direction"].isin(
            ["active_gt_passive", "passive_gt_active"]
        )
    ]
    onset = tests.loc[tests["metric"].eq("onset")].iloc[0]
    peak = tests.loc[tests["metric"].eq("peak")].iloc[0]
    metadata = {
        "figure_contract": {
            "core_conclusion": "After trial-wise baseline correction and removal of pre-existing segments crossing t=0, matched-run active- and passive-preferring evoked latencies do not differ significantly.",
            "archetype": "paired quantitative comparison",
            "backend": "Python/matplotlib",
            "final_size_mm": [WIDTH_MM, HEIGHT_MM],
            "font_size_pt": PT,
        },
        "analysis": {
            "n_all_units": int(len(units)),
            "unit_population": "all registered Kilosort good + MUA units",
            "baseline_window_ms": [-1000, 0],
            "pointwise_test_window_ms": [PRETEST_START_MS, RESPONSE_STOP_MS],
            "pointwise_test": "two-sided one-way ANOVA across trials",
            "pointwise_alpha": ALPHA,
            "minimum_consecutive_bins": MIN_CONSECUTIVE_BINS,
            "preexisting_rule": "exclude any significant segment with start < 0 and stop > 0 from evoked timing",
            "n_units_with_preexisting_crossing_segment": int(
                units["had_preexisting_crossing_segment"].sum()
            ),
            "n_directional_evoked_unit_run_entries": int(len(selected)),
            "direction_counts": {
                str(k): int(v)
                for k, v in selected["evoked_direction"].value_counts().items()
            },
            "n_matched_runs": int(onset["n_matched_runs"]),
            "n_patients": 5,
        },
        "matched_tests": tests.to_dict(orient="records"),
        "reviewer_risks": [
            "Active- and passive-preferring units are different neurons; matching is by recording run, not unit identity.",
            "Runs are nested within five patients, so the matched-run analysis is exploratory rather than patient-level population inference.",
            "BA/BEI marker family is fully collinear with active/passive construction in the current stimuli.",
            "Latency remains conditional on passing the pointwise-duration screen.",
        ],
    }
    (SOURCE / "fig3i_bc_evoked_matched_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    legend = (
        "**Fig. 3i | Baseline-corrected, run-matched construction-preference timing.** "
        "Firing-rate time courses were centred within each trial and unit by the mean activity from -1,000 to 0 ms relative to corrected function-word onset. Pointwise active-passive comparisons were evaluated from -500 to 800 ms (two-sided ANOVA, P < 0.025 for at least six consecutive 1-ms bins); significant segments beginning before and continuing after t = 0 were classified as pre-existing and excluded from evoked timing. "
        "Within each recording run, points show the median evoked onset or direction-consistent peak latency across active-preferring and passive-preferring units; lines connect the same run and black bars denote medians across runs. "
        "P values are from exact one-sided paired Wilcoxon signed-rank tests of the hypothesis that active-preferring run medians are earlier "
        f"(onset W = {onset['wilcoxon_w']:.0f}, P = {onset['wilcoxon_one_sided_p_value']:.3f}; "
        f"peak W = {peak['wilcoxon_w']:.0f}, P = {peak['wilcoxon_one_sided_p_value']:.3f}; n = {int(onset['n_matched_runs'])} matched runs). "
        "Runs are nested within five patients, and BA/BEI marker family is collinear with active/passive construction; the matched-run inference is exploratory. Source data are provided with the figure.\n"
    )
    (SOURCE / "fig3i_bc_evoked_matched_legend.md").write_text(
        legend, encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
