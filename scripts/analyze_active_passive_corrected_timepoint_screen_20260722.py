#!/usr/bin/env python3
"""Re-run the active/passive unit census with corrected function-word onsets.

Primary screen
--------------
* all 1,782 registered KS good + MUA units from the eight active/passive runs;
* function-word onset = file-play onset + word onset measured in the audio file;
* 1-ms firing-rate bins smoothed with a Gaussian SDF (sigma = 25 ms);
* two-sided one-way ANOVA at every 1-ms bin in [0, 800) ms;
* pointwise p < 0.05 / 2 = 0.025;
* a unit screens positive only when at least six consecutive bins pass.

For auditability, the script also repeats the same calculation at the legacy
onsets and reports the older cumulative-bin interpretation (more than five
passing bins anywhere in the window).  The latter is a sensitivity analysis,
not the primary result.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
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
from scipy import stats
from scipy.ndimage import gaussian_filter1d


DATA_ROOT = ROOT / "neuropixel2025/data"
OUT_DIR = ROOT / "outputs/active_passive_corrected_timepoint_screen_20260722"
FIG_DIR = ROOT / "reports/active_passive_unit_deck_20260722/figures"

SUBJECTS = ("gz001", "gz002", "gz003", "gz004", "gz005")
RUN_LABELS = {
    "gz001t_g0_actpas_run1": "gz001 · run1",
    "gz002_g0_actpas_run2": "gz002 · run2",
    "gz003t2_g0_actpas_run1": "gz003 · run1",
    "gz003t2_g0_actpas_run4": "gz003 · run4",
    "gz004t1_g0_actpas_run0": "gz004 · run0",
    "gz004t1_g3_actpas_run3": "gz004 · run3",
    "gz005t0_g0_actpas_run0": "gz005 · run0",
    "gz005t0_g3_actpas_run3": "gz005 · run3",
}
RUN_ORDER = tuple(RUN_LABELS)

SDF_SIGMA_MS = 25
FILTER_PAD_MS = 4 * SDF_SIGMA_MS
BASELINE_START_MS = -1000
RESPONSE_START_MS = 0
RESPONSE_STOP_MS = 800
N_RESPONSE_BINS = RESPONSE_STOP_MS - RESPONSE_START_MS
N_CONDITIONS = 2
ALPHA = 0.05 / N_CONDITIONS
MIN_CONSECUTIVE_BINS = 6

ACTIVE = "#2E86C1"
PASSIVE = "#D66B10"
MIXED = "#7B61A8"
INK = "#17322F"
GRAY = "#6B7B78"
LIGHT = "#DCE4E1"
PAPER = "#F8F7F2"


@dataclass(frozen=True)
class Segment:
    start_ms: int
    stop_ms: int

    @property
    def duration_ms(self) -> int:
        return self.stop_ms - self.start_ms


def contiguous_segments(mask: np.ndarray, min_bins: int = MIN_CONSECUTIVE_BINS) -> tuple[Segment, ...]:
    """Return no-gap-fill true runs with stop-exclusive endpoints."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if min_bins <= 0:
        raise ValueError("min_bins must be positive")
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple(
        Segment(int(start), int(stop))
        for start, stop in zip(starts, stops)
        if int(stop - start) >= min_bins
    )


def pooled_cohens_d(active: np.ndarray, passive: np.ndarray) -> np.ndarray:
    """Cohen's d used by the existing deck, vectorized over units."""
    active = np.asarray(active, dtype=float)
    passive = np.asarray(passive, dtype=float)
    difference = active.mean(axis=0) - passive.mean(axis=0)
    pooled = np.sqrt((active.std(axis=0, ddof=1) ** 2 + passive.std(axis=0, ddof=1) ** 2) / 2.0)
    result = np.full(difference.shape, np.nan, dtype=float)
    np.divide(difference, pooled, out=result, where=pooled > 0)
    result[(pooled == 0) & (difference == 0)] = 0.0
    return result


def _task_dir(subject: str) -> Path:
    return DATA_ROOT / subject / "processed/ActPasSynSentence"


def load_trials(subject: str, run: str) -> pd.DataFrame:
    """Return matched legacy and corrected event bins for one run."""
    task_dir = _task_dir(subject)
    er_dir = task_dir / "event_related_active_passive"
    metadata = pd.read_csv(er_dir / "active_passive_event_related_trial_metadata.csv")
    metadata = metadata.loc[metadata["run"].astype(str).eq(run)].copy()
    if metadata.empty:
        raise ValueError(f"{run}: no active/passive metadata")

    events = pd.read_csv(task_dir / run / "events.csv")
    audio = pd.read_csv(er_dir / "active_passive_function_word_audio_onsets.csv")
    required_events = {"trial_index", "sound_file", "time_from_crop_start_sec"}
    required_audio = {"sound_file", "target_onset_sec_in_audio"}
    if not required_events <= set(events.columns):
        raise ValueError(f"{run}: events.csv lacks {sorted(required_events - set(events.columns))}")
    if not required_audio <= set(audio.columns):
        raise ValueError(f"{subject}: onset table lacks {sorted(required_audio - set(audio.columns))}")

    trials = metadata.merge(
        events[["trial_index", "sound_file", "time_from_crop_start_sec"]],
        on=["trial_index", "sound_file"],
        how="inner",
        validate="one_to_one",
    )
    trials = trials.merge(
        audio[["sound_file", "target_onset_sec_in_audio"]],
        on="sound_file",
        how="left",
        validate="many_to_one",
    )
    if len(trials) != len(metadata):
        raise ValueError(f"{run}: event merge retained {len(trials)} of {len(metadata)} trials")
    if trials["target_onset_sec_in_audio"].isna().any():
        raise ValueError(f"{run}: missing corrected audio onset")
    trials["legacy_event_bin"] = trials["event_bin_1000Hz"].astype(int)
    trials["corrected_event_bin"] = np.rint(
        1000.0
        * (
            trials["time_from_crop_start_sec"].astype(float)
            + trials["target_onset_sec_in_audio"].astype(float)
        )
    ).astype(int)
    trials["onset_shift_ms"] = trials["corrected_event_bin"] - trials["legacy_event_bin"]
    trials = trials.sort_values("trial_index").reset_index(drop=True)
    counts = trials["condition"].value_counts().to_dict()
    if counts.get("active") != 10 or counts.get("passive") != 10:
        raise ValueError(f"{run}: expected 10 active and 10 passive trials, got {counts}")
    return trials


def extract_smoothed_epochs(
    firing_rate_hz: np.ndarray,
    event_bins: np.ndarray,
) -> np.ndarray:
    """Extract [-1, 0.8) s after padding, then apply the registered SDF."""
    raw_start = BASELINE_START_MS - FILTER_PAD_MS
    raw_stop = RESPONSE_STOP_MS + FILTER_PAD_MS
    expected = raw_stop - raw_start
    epochs: list[np.ndarray] = []
    for event_bin in np.asarray(event_bins, dtype=int):
        start = int(event_bin + raw_start)
        stop = int(event_bin + raw_stop)
        if start < 0 or stop > firing_rate_hz.shape[1]:
            raise ValueError(f"event bin {event_bin}: padded epoch outside recording")
        epoch = firing_rate_hz[:, start:stop]
        if epoch.shape[1] != expected:
            raise ValueError("unexpected padded epoch length")
        epochs.append(epoch)
    raw = np.stack(epochs, axis=0).astype(np.float32, copy=False)
    smooth = gaussian_filter1d(
        raw,
        sigma=SDF_SIGMA_MS,
        axis=2,
        mode="constant",
        cval=0.0,
        output=np.float32,
    )
    start = FILTER_PAD_MS
    stop = start + (RESPONSE_STOP_MS - BASELINE_START_MS)
    return smooth[:, :, start:stop]


def analyze_alignment(
    *,
    subject: str,
    run: str,
    alignment: str,
    epochs: np.ndarray,
    conditions: np.ndarray,
    unit_ids: np.ndarray,
    ks_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    response_offset = RESPONSE_START_MS - BASELINE_START_MS
    response = epochs[:, :, response_offset : response_offset + N_RESPONSE_BINS]
    baseline = epochs[:, :, :response_offset]
    active_mask = conditions == "active"
    passive_mask = conditions == "passive"
    active = response[active_mask]
    passive = response[passive_mask]

    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            anova = stats.f_oneway(active, passive, axis=0)
    f_values = np.asarray(anova.statistic, dtype=float)
    p_values = np.asarray(anova.pvalue, dtype=float)
    p_values[~np.isfinite(p_values)] = 1.0
    f_values[~np.isfinite(f_values)] = np.nan
    active_mean = active.mean(axis=0)
    passive_mean = passive.mean(axis=0)
    difference = active_mean - passive_mean
    passing = p_values < ALPHA
    passing_p05 = p_values < 0.05

    active_bc = active.mean(axis=2) - baseline[active_mask].mean(axis=2)
    passive_bc = passive.mean(axis=2) - baseline[passive_mask].mean(axis=2)
    d_values = pooled_cohens_d(active_bc, passive_bc)
    aggregate_p = np.empty(len(unit_ids), dtype=float)
    for unit_index in range(len(unit_ids)):
        aggregate_p[unit_index] = stats.mannwhitneyu(
            active_bc[:, unit_index],
            passive_bc[:, unit_index],
            alternative="two-sided",
        ).pvalue

    unit_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for unit_index, unit_id in enumerate(unit_ids):
        mask = passing[unit_index]
        segments = contiguous_segments(mask)
        nominal_segments = contiguous_segments(passing_p05[unit_index])
        qualifying = np.zeros(N_RESPONSE_BINS, dtype=bool)
        for segment_id, segment in enumerate(segments):
            qualifying[segment.start_ms : segment.stop_ms] = True
            seg_diff = difference[unit_index, segment.start_ms : segment.stop_ms]
            segment_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "alignment": alignment,
                    "unit_id": int(unit_id),
                    "ks_label": str(ks_labels[unit_index]),
                    "segment_id": segment_id,
                    "start_ms": segment.start_ms,
                    "stop_ms_exclusive": segment.stop_ms,
                    "duration_ms": segment.duration_ms,
                    "direction": (
                        "active_gt_passive" if float(np.nanmean(seg_diff)) > 0
                        else "passive_gt_active" if float(np.nanmean(seg_diff)) < 0
                        else "mixed"
                    ),
                    "mean_active_minus_passive_hz": float(np.nanmean(seg_diff)),
                    "min_p": float(np.nanmin(p_values[unit_index, segment.start_ms : segment.stop_ms])),
                }
            )

        q_active_ms = int(np.sum(qualifying & (difference[unit_index] > 0)))
        q_passive_ms = int(np.sum(qualifying & (difference[unit_index] < 0)))
        if q_active_ms > q_passive_ms:
            direction = "active_gt_passive"
        elif q_passive_ms > q_active_ms:
            direction = "passive_gt_active"
        elif qualifying.any():
            direction = "mixed"
        else:
            direction = "not_significant"

        pass_indices = np.flatnonzero(mask)
        for time_index in pass_indices:
            point_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "alignment": alignment,
                    "unit_id": int(unit_id),
                    "ks_label": str(ks_labels[unit_index]),
                    "time_ms": int(time_index),
                    "anova_f": float(f_values[unit_index, time_index]) if np.isfinite(f_values[unit_index, time_index]) else np.nan,
                    "anova_p": float(p_values[unit_index, time_index]),
                    "active_mean_hz": float(active_mean[unit_index, time_index]),
                    "passive_mean_hz": float(passive_mean[unit_index, time_index]),
                    "active_minus_passive_hz": float(difference[unit_index, time_index]),
                    "in_qualifying_segment": bool(qualifying[time_index]),
                }
            )

        unit_rows.append(
            {
                "subject": subject,
                "run": run,
                "alignment": alignment,
                "unit_id": int(unit_id),
                "ks_label": str(ks_labels[unit_index]),
                "n_active_trials": int(active.shape[0]),
                "n_passive_trials": int(passive.shape[0]),
                "n_conditions": N_CONDITIONS,
                "pointwise_alpha": ALPHA,
                "min_pointwise_p": float(np.min(p_values[unit_index])),
                "n_pointwise_passing_bins": int(mask.sum()),
                "n_pointwise_p05_bins": int(passing_p05[unit_index].sum()),
                "any_pointwise_p025": bool(mask.any()),
                "cumulative_gt5_bins": bool(mask.sum() > 5),
                "has_qualifying_segment_p05": bool(nominal_segments),
                "n_qualifying_segments": len(segments),
                "first_segment_onset_ms": segments[0].start_ms if segments else np.nan,
                "longest_segment_ms": max((segment.duration_ms for segment in segments), default=0),
                "total_qualifying_duration_ms": int(qualifying.sum()),
                "has_qualifying_segment": bool(segments),
                "qualifying_active_gt_passive_ms": q_active_ms,
                "qualifying_passive_gt_active_ms": q_passive_ms,
                "qualifying_direction": direction,
                "corrected_window_active_minus_passive_bc_hz": float(active_bc[:, unit_index].mean() - passive_bc[:, unit_index].mean()),
                "corrected_window_cohens_d": float(d_values[unit_index]),
                "corrected_window_mannwhitney_p": float(aggregate_p[unit_index]),
            }
        )

    return pd.DataFrame(unit_rows), pd.DataFrame(segment_rows), pd.DataFrame(point_rows)


def session_counts(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (alignment, subject, run), frame in units.groupby(["alignment", "subject", "run"], sort=False):
        selected = frame.loc[frame["has_qualifying_segment"]]
        direction_counts = selected["qualifying_direction"].value_counts()
        rows.append(
            {
                "alignment": alignment,
                "subject": subject,
                "run": run,
                "run_label": RUN_LABELS[run],
                "n_units": len(frame),
                "n_cumulative_gt5_bins": int(frame["cumulative_gt5_bins"].sum()),
                "n_contiguous_gt5ms": int(frame["has_qualifying_segment"].sum()),
                "active_gt_passive": int(direction_counts.get("active_gt_passive", 0)),
                "passive_gt_active": int(direction_counts.get("passive_gt_active", 0)),
                "mixed": int(direction_counts.get("mixed", 0)),
            }
        )
    return pd.DataFrame(rows)


def subject_counts(sessions: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "n_units",
        "n_cumulative_gt5_bins",
        "n_contiguous_gt5ms",
        "active_gt_passive",
        "passive_gt_active",
        "mixed",
    ]
    return sessions.groupby(["alignment", "subject"], as_index=False)[numeric].sum()


def load_current_slide_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(
        DATA_ROOT.glob(
            "gz*/processed/ActPasSynSentence/event_related_active_passive/active_passive_significant_units.csv"
        )
    ):
        subject = path.parts[path.parts.index("data") + 1]
        frame = pd.read_csv(path)
        frame["subject"] = subject
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 1782:
        raise ValueError(f"current slide metric table has {len(result)} rows, expected 1782")
    return result


def build_crosswalk(units: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    key = ["subject", "run", "unit_id"]
    legacy = units.loc[units["alignment"].eq("legacy")].copy()
    corrected = units.loc[units["alignment"].eq("corrected")].copy()
    legacy = legacy.rename(
        columns={
            "cumulative_gt5_bins": "legacy_cumulative_gt5_bins",
            "has_qualifying_segment": "legacy_contiguous_gt5ms",
            "n_pointwise_passing_bins": "legacy_n_passing_bins",
            "longest_segment_ms": "legacy_longest_segment_ms",
            "qualifying_direction": "legacy_direction",
        }
    )
    corrected = corrected.rename(
        columns={
            "cumulative_gt5_bins": "corrected_cumulative_gt5_bins",
            "has_qualifying_segment": "corrected_contiguous_gt5ms",
            "n_pointwise_passing_bins": "corrected_n_passing_bins",
            "longest_segment_ms": "corrected_longest_segment_ms",
            "qualifying_direction": "corrected_direction",
        }
    )
    keep_legacy = key + [
        "legacy_cumulative_gt5_bins",
        "legacy_contiguous_gt5ms",
        "legacy_n_passing_bins",
        "legacy_longest_segment_ms",
        "legacy_direction",
    ]
    keep_corrected = key + [
        "ks_label",
        "corrected_cumulative_gt5_bins",
        "corrected_contiguous_gt5ms",
        "corrected_n_passing_bins",
        "corrected_longest_segment_ms",
        "corrected_direction",
        "corrected_window_active_minus_passive_bc_hz",
        "corrected_window_cohens_d",
        "corrected_window_mannwhitney_p",
        "n_pointwise_p05_bins",
        "any_pointwise_p025",
        "has_qualifying_segment_p05",
    ]
    current_keep = current[
        key
        + [
            "cohens_d",
            "p_value",
            "significant_uncorrected_p05",
            "direction",
        ]
    ].rename(
        columns={
            "cohens_d": "current_slide_cohens_d",
            "p_value": "current_slide_p_value",
            "significant_uncorrected_p05": "current_slide_nominal_p05",
            "direction": "current_slide_direction",
        }
    )
    crosswalk = legacy[keep_legacy].merge(corrected[keep_corrected], on=key, validate="one_to_one")
    crosswalk = crosswalk.merge(current_keep, on=key, validate="one_to_one")
    if len(crosswalk) != 1782:
        raise ValueError(f"crosswalk has {len(crosswalk)} rows, expected 1782")
    return crosswalk


def comparison_rows(crosswalk: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "Original slide: legacy-onset aggregate MWU p<0.05": crosswalk["current_slide_nominal_p05"].astype(bool),
        "Corrected onset: aggregate MWU p<0.05": crosswalk["corrected_window_mannwhitney_p"].lt(0.05),
        "Legacy onset: pointwise p<0.025 for >=6 ms": crosswalk["legacy_contiguous_gt5ms"].astype(bool),
        "Corrected onset: pointwise p<0.05 for >=6 ms": crosswalk["has_qualifying_segment_p05"].astype(bool),
        "Corrected onset: pointwise p<0.025 for >=6 ms": crosswalk["corrected_contiguous_gt5ms"].astype(bool),
    }
    rows: list[dict[str, object]] = []
    for label, mask in definitions.items():
        rows.append({"definition": label, "n_units": int(mask.sum()), "fraction_units": float(mask.mean())})
    return pd.DataFrame(rows)


def overlap_rows(crosswalk: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        (
            "current aggregate nominal vs corrected contiguous",
            "current_slide_nominal_p05",
            "corrected_contiguous_gt5ms",
        ),
        (
            "legacy cumulative vs legacy contiguous",
            "legacy_cumulative_gt5_bins",
            "legacy_contiguous_gt5ms",
        ),
        (
            "legacy contiguous vs corrected contiguous",
            "legacy_contiguous_gt5ms",
            "corrected_contiguous_gt5ms",
        ),
        (
            "corrected cumulative vs corrected contiguous",
            "corrected_cumulative_gt5_bins",
            "corrected_contiguous_gt5ms",
        ),
    ]
    rows: list[dict[str, object]] = []
    for label, old_col, new_col in pairs:
        old = crosswalk[old_col].astype(bool)
        new = crosswalk[new_col].astype(bool)
        rows.append(
            {
                "comparison": label,
                "old_total": int(old.sum()),
                "new_total": int(new.sum()),
                "shared": int((old & new).sum()),
                "lost": int((old & ~new).sum()),
                "gained": int((~old & new).sum()),
                "neither": int((~old & ~new).sum()),
                "jaccard": float((old & new).sum() / (old | new).sum()) if (old | new).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_corrected_census(crosswalk: pd.DataFrame, sessions: pd.DataFrame, path: Path) -> None:
    configure_plot()
    fig, axes = plt.subplots(4, 2, figsize=(13.2, 8.0), sharey=True, constrained_layout=True)
    corrected_sessions = sessions.loc[sessions["alignment"].eq("corrected")].set_index("run")
    for ax, run in zip(axes.flat, RUN_ORDER):
        frame = crosswalk.loc[crosswalk["run"].eq(run)].copy()
        frame["plot_d"] = frame["corrected_window_cohens_d"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame = frame.sort_values(["plot_d", "unit_id"]).reset_index(drop=True)
        x = np.arange(len(frame))
        colors = np.where(frame["plot_d"].to_numpy() >= 0, ACTIVE, PASSIVE)
        ax.scatter(x, frame["plot_d"], c=colors, s=9, alpha=0.50, linewidth=0, rasterized=True)
        sig = frame["corrected_contiguous_gt5ms"].to_numpy(bool)
        ax.scatter(
            x[sig],
            frame.loc[sig, "plot_d"],
            facecolors="none",
            edgecolors=INK,
            s=40,
            linewidth=1.0,
        )
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlim(-4, len(frame) + 3)
        ax.set_ylim(-2.25, 2.25)
        count = corrected_sessions.loc[run]
        ax.set_title(
            f"{RUN_LABELS[run]}  |  n={len(frame)}  |  screen={int(count.n_contiguous_gt5ms)} "
            f"(A{int(count.active_gt_passive)}/P{int(count.passive_gt_active)}/M{int(count.mixed)})",
            loc="left",
            fontsize=10.2,
            fontweight="bold",
        )
        ax.set_xlabel("Units sorted by corrected-window Cohen's d", fontsize=9.5)
        ax.set_ylabel("Corrected-window Cohen's d\n(+ active; − passive)", fontsize=9.5)
        ax.grid(axis="y", color=LIGHT, lw=0.6)
    total = int(crosswalk["corrected_contiguous_gt5ms"].sum())
    fig.suptitle(
        f"Corrected-onset active/passive unit census: {total}/1,782 pass the registered time-resolved screen",
        fontsize=16.5,
        fontweight="bold",
        color=INK,
    )
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ACTIVE, markeredgecolor="none", label="Active > passive (window d)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PASSIVE, markeredgecolor="none", label="Passive > active (window d)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=INK, label="p<0.025 for >=6 consecutive ms"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.025), fontsize=9.5)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def plot_definition_comparison(
    comparison: pd.DataFrame,
    overlaps: pd.DataFrame,
    onset_shifts: pd.DataFrame,
    path: Path,
) -> None:
    configure_plot()
    fig = plt.figure(figsize=(13.2, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.0, 1.1], height_ratios=[1.0, 1.0])
    ax_bar = fig.add_subplot(grid[:, 0])
    ax_overlap = fig.add_subplot(grid[0, 1:])
    ax_shift = fig.add_subplot(grid[1, 1])
    ax_rules = fig.add_subplot(grid[1, 2])

    labels = [
        "Original slide\nlegacy-onset aggregate\nMWU p<0.05",
        "Corrected onset\naggregate\nMWU p<0.05",
        "Legacy onset\npointwise p<0.025\n>=6 ms",
        "Corrected onset\npointwise p<0.05\n>=6 ms",
        "Corrected onset\npointwise p<0.025\n>=6 ms",
    ]
    # Primary comparison values (82, 197, 172) carry the color; the two
    # intermediate sensitivity values (95, 410) are intentionally muted.
    colors = ["#D99000", "#C8CFCD", "#8E7C4F", "#C8CFCD", "#009B77"]
    values = comparison["n_units"].to_numpy()
    bars = ax_bar.barh(np.arange(len(values)), values, color=colors, edgecolor="white", height=0.72)
    ax_bar.set_yticks(np.arange(len(labels)), labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Number of units")
    ax_bar.set_xlim(0, max(values) * 1.22)
    for bar, value in zip(bars, values):
        ax_bar.text(value + max(values) * 0.025, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontweight="bold", color=INK)
    ax_bar.set_title("Counts depend on the estimand and threshold", loc="left", fontweight="bold", fontsize=13)
    ax_bar.grid(axis="x", color=LIGHT, lw=0.7)

    target = overlaps.loc[
        overlaps["comparison"].eq("current aggregate nominal vs corrected contiguous")
    ].iloc[0]
    categories = ["Shared", "Only current\naggregate", "Only corrected\ntime-resolved", "Neither"]
    overlap_values = [target.shared, target.lost, target.gained, target.neither]
    overlap_colors = ["#009B77", PASSIVE, ACTIVE, "#D8DEDC"]
    ax_overlap.bar(categories, overlap_values, color=overlap_colors, edgecolor="white")
    for x, value in enumerate(overlap_values):
        ax_overlap.text(x, value + 18, str(int(value)), ha="center", fontweight="bold")
    ax_overlap.set_ylim(0, max(overlap_values) * 1.16)
    ax_overlap.set_ylabel("Units")
    ax_overlap.set_title("Only 10 units are shared between 82 and 172", loc="left", fontweight="bold", fontsize=12)
    ax_overlap.grid(axis="y", color=LIGHT, lw=0.7)

    ax_shift.hist(onset_shifts["onset_shift_ms"], bins=np.arange(145, 281, 10), color="#4C9F8B", edgecolor="white")
    median_shift = float(onset_shifts["onset_shift_ms"].median())
    ax_shift.axvline(median_shift, color=INK, lw=1.4, ls="--")
    ax_shift.text(median_shift + 3, ax_shift.get_ylim()[1] * 0.88, f"median {median_shift:.0f} ms", color=INK, fontweight="bold")
    ax_shift.set_xlabel("Corrected − legacy onset (ms)")
    ax_shift.set_ylabel("Trials")
    ax_shift.set_title("Onset shift across 160 trials", loc="left", fontweight="bold", fontsize=10.8)

    ax_rules.axis("off")
    ax_rules.set_title("Interpretation", loc="left", fontweight="bold", fontsize=10.8)
    rule_text = (
        "1  Alignment moves every trial\n"
        "    by +155 to +270 ms.\n\n"
        "2  Whole-window MWU and\n"
        "    pointwise ANOVA ask\n"
        "    different questions.\n\n"
        "3  p<0.025 is stricter than\n"
        "    p<0.05 at each bin.\n\n"
        "4  Cumulative vs contiguous\n"
        "    gives the same count here."
    )
    ax_rules.text(0.0, 0.88, rule_text, va="top", fontsize=9.7, color=INK, linespacing=1.25)

    fig.suptitle("Why the significant-unit count changes", fontsize=17, fontweight="bold", color=INK)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_units: list[pd.DataFrame] = []
    all_segments: list[pd.DataFrame] = []
    all_points: list[pd.DataFrame] = []
    trial_audit: list[pd.DataFrame] = []

    discovered_runs: list[str] = []
    for subject in SUBJECTS:
        metadata_path = _task_dir(subject) / "event_related_active_passive/active_passive_event_related_trial_metadata.csv"
        metadata = pd.read_csv(metadata_path)
        for run in metadata["run"].astype(str).drop_duplicates():
            discovered_runs.append(run)
            trials = load_trials(subject, run)
            trial_audit.append(
                trials[
                    [
                        "run",
                        "trial_index",
                        "sound_file",
                        "condition",
                        "target_word",
                        "legacy_event_bin",
                        "corrected_event_bin",
                        "onset_shift_ms",
                    ]
                ].assign(subject=subject)
            )
            fr_path = _task_dir(subject) / run / "firing_rate_1000Hz/firing_rate_1000Hz_all_clusters.npz"
            archive = np.load(fr_path, allow_pickle=True)
            firing_rate = archive["firing_rate_hz"].astype(np.float32, copy=False)
            unit_ids = archive["unit_ids"].astype(int)
            ks_labels = archive["ks_labels"].astype(str)
            conditions = trials["condition"].astype(str).to_numpy()

            for alignment, column in (
                ("legacy", "legacy_event_bin"),
                ("corrected", "corrected_event_bin"),
            ):
                epochs = extract_smoothed_epochs(firing_rate, trials[column].to_numpy())
                units, segments, points = analyze_alignment(
                    subject=subject,
                    run=run,
                    alignment=alignment,
                    epochs=epochs,
                    conditions=conditions,
                    unit_ids=unit_ids,
                    ks_labels=ks_labels,
                )
                all_units.append(units)
                all_segments.append(segments)
                all_points.append(points)

    if tuple(discovered_runs) != RUN_ORDER:
        raise ValueError(f"run order mismatch: {discovered_runs}")

    units = pd.concat(all_units, ignore_index=True)
    segments = pd.concat(all_segments, ignore_index=True) if any(len(x) for x in all_segments) else pd.DataFrame()
    points = pd.concat(all_points, ignore_index=True) if any(len(x) for x in all_points) else pd.DataFrame()
    trials_all = pd.concat(trial_audit, ignore_index=True)
    sessions = session_counts(units)
    subjects = subject_counts(sessions)
    current = load_current_slide_metrics()
    crosswalk = build_crosswalk(units, current)
    comparison = comparison_rows(crosswalk)
    overlaps = overlap_rows(crosswalk)

    # Validate that padded-epoch smoothing reproduces the existing legacy
    # cumulative-bin result before interpreting the corrected rerun.
    legacy_reference_path = ROOT / "outputs/active_passive_timepoint_anova_exact_800ms_sdf_sigma25_all_clusters_unit_results_p025_gt5ms.csv"
    legacy_reference = pd.read_csv(legacy_reference_path)
    check = crosswalk.merge(
        legacy_reference[["subject", "run", "unit_id", "n_significant_ms_total", "has_gt5ms_significant_difference"]],
        on=["subject", "run", "unit_id"],
        validate="one_to_one",
    )
    exact_bin_match = bool(
        np.array_equal(
            check["legacy_n_passing_bins"].to_numpy(int),
            check["n_significant_ms_total"].to_numpy(int),
        )
    )
    exact_screen_match = bool(
        np.array_equal(
            check["legacy_cumulative_gt5_bins"].to_numpy(bool),
            check["has_gt5ms_significant_difference"].to_numpy(bool),
        )
    )
    if not (exact_bin_match and exact_screen_match):
        raise RuntimeError("legacy validation failed: epoch-padded SDF does not reproduce archived results")

    units.to_csv(OUT_DIR / "unit_results_legacy_and_corrected.csv", index=False)
    units.loc[units["alignment"].eq("corrected")].to_csv(OUT_DIR / "corrected_unit_results.csv", index=False)
    segments.to_csv(OUT_DIR / "qualifying_segments.csv", index=False)
    points.to_csv(OUT_DIR / "pointwise_passing_bins.csv", index=False)
    trials_all.to_csv(OUT_DIR / "trial_onset_audit.csv", index=False)
    sessions.to_csv(OUT_DIR / "session_counts.csv", index=False)
    subjects.to_csv(OUT_DIR / "subject_counts.csv", index=False)
    crosswalk.to_csv(OUT_DIR / "unit_crosswalk.csv", index=False)
    comparison.to_csv(OUT_DIR / "definition_comparison.csv", index=False)
    overlaps.to_csv(OUT_DIR / "overlap_comparison.csv", index=False)

    plot_corrected_census(
        crosswalk,
        sessions,
        FIG_DIR / "Fig_all_unit_effect_census_corrected_p025_contiguous_gt5ms.png",
    )
    plot_definition_comparison(
        comparison,
        overlaps,
        trials_all,
        FIG_DIR / "Fig_corrected_significant_count_explainer.png",
    )

    corrected = crosswalk["corrected_contiguous_gt5ms"].astype(bool)
    corrected_d = crosswalk["corrected_window_cohens_d"].fillna(0.0)
    corrected_direction_counts = {
        "positive_d": int((corrected_d > 0).sum()),
        "negative_d": int((corrected_d < 0).sum()),
        "zero_d": int((corrected_d == 0).sum()),
        "median_d": float(crosswalk["corrected_window_cohens_d"].median()),
    }
    primary_selected = units.loc[
        units["alignment"].eq("corrected") & units["has_qualifying_segment"]
    ]
    primary_direction = primary_selected["qualifying_direction"].value_counts().to_dict()
    summary = {
        "analysis": {
            "unit_population": "all KS good + MUA units",
            "n_units": int(len(crosswalk)),
            "n_sessions": len(RUN_ORDER),
            "n_subjects": len(SUBJECTS),
            "corrected_event_definition": "file-play onset + target_onset_sec_in_audio",
            "sdf_sigma_ms": SDF_SIGMA_MS,
            "inferential_window_ms": [RESPONSE_START_MS, RESPONSE_STOP_MS],
            "test": "two-sided scipy.stats.f_oneway(active, passive, axis=trial)",
            "n_conditions": N_CONDITIONS,
            "pointwise_alpha": ALPHA,
            "duration_rule": "at least 6 consecutive 1-ms bins; no gap filling",
        },
        "primary_corrected_result": {
            "n_significant_units": int(corrected.sum()),
            "fraction_significant_units": float(corrected.mean()),
            "direction_counts": {str(key): int(value) for key, value in primary_direction.items()},
            "corrected_window_effect_direction": corrected_direction_counts,
        },
        "onset_shift_ms": {
            "min": int(trials_all["onset_shift_ms"].min()),
            "median": float(trials_all["onset_shift_ms"].median()),
            "max": int(trials_all["onset_shift_ms"].max()),
        },
        "legacy_validation": {
            "reference": str(legacy_reference_path),
            "exact_passing_bin_match": exact_bin_match,
            "exact_cumulative_screen_match": exact_screen_match,
        },
        "definitions": comparison.to_dict(orient="records"),
        "overlaps": overlaps.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nCorrected session counts:")
    print(sessions.loc[sessions["alignment"].eq("corrected")].to_string(index=False))


if __name__ == "__main__":
    main()
