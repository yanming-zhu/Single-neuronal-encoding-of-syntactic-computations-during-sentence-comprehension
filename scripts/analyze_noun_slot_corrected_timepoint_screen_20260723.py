#!/usr/bin/env python3
"""Corrected-onset, time-resolved noun-slot unit analysis.

The same noun identity is compared in two structural positions:

* sentence-initial subject slot;
* post-function-word predicate-position noun slot.

Primary screen
--------------
* all 1,782 registered KS good + MUA units from eight runs;
* corrected noun onset = file-play onset + noun onset measured in the audio;
* Gaussian SDF, sigma = 25 ms;
* two-level repeated-measures role test at every 1-ms bin in [0, 800) ms;
* p < 0.05 / 2 = 0.025;
* at least six consecutive passing bins, with no gap filling.

The paired test controls noun identity.  With two levels it is exactly
equivalent to a two-level repeated-measures ANOVA (F = paired-t squared).
An unpaired one-way ANOVA is retained as a sensitivity analysis because that
is the literal test used in the active/passive screen.
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
from matplotlib.colors import ListedColormap
from scipy import stats
from scipy.ndimage import gaussian_filter1d


DATA_ROOT = ROOT / "neuropixel2025/data"
LEGACY_DIR = DATA_ROOT / "combined_noun_role_structure_encoding"
OUT_DIR = ROOT / "outputs/noun_slot_corrected_timepoint_screen_20260723"
FIG_DIR = ROOT / "reports/noun_slot_unit_analysis_20260723/figures"

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
NOUN_ORDER = (
    "虫子", "交警", "小狗", "妈妈", "球员", "兔子", "爸爸", "婴儿", "奶奶", "警察",
    "树叶", "汽车", "骨头", "衣服", "足球", "菜叶", "爆竹", "牛奶", "西瓜", "道路",
)

ROLE_SUBJECT = "subject"
ROLE_PREDICATE = "predicate"
N_CONDITIONS = 2
ALPHA = 0.05 / N_CONDITIONS
MIN_CONSECUTIVE_BINS = 6
SDF_SIGMA_MS = 25
DISPLAY_SIGMA_MS = 35
FILTER_PAD_MS = 4 * DISPLAY_SIGMA_MS
DISPLAY_START_MS = -1000
DISPLAY_STOP_MS = 2000
RESPONSE_START_MS = 0
RESPONSE_STOP_MS = 800
BASELINE_START_MS = -500
BASELINE_STOP_MS = 0
LOW_FULL_WINDOW_MEAN_HZ = 2.0
HIGH_FULL_WINDOW_MEAN_HZ = 20.0

SUBJECT_COLOR = "#2E86C1"
PREDICATE_COLOR = "#D66B10"
MIXED_COLOR = "#7B61A8"
PATIENT_COLORS = {
    "gz001": "#0072B2",
    "gz002": "#E69F00",
    "gz003": "#009E73",
    "gz004": "#CC79A7",
    "gz005": "#D55E00",
}
INK = "#17322F"
GRAY = "#687673"
LIGHT = "#DCE4E1"
PAPER = "#FBFAF6"


@dataclass(frozen=True)
class Segment:
    start_ms: int
    stop_ms: int

    @property
    def duration_ms(self) -> int:
        return self.stop_ms - self.start_ms


def contiguous_segments(mask: np.ndarray, min_bins: int = MIN_CONSECUTIVE_BINS) -> tuple[Segment, ...]:
    values = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple(
        Segment(int(start), int(stop))
        for start, stop in zip(starts, stops)
        if int(stop - start) >= min_bins
    )


def task_dir(subject: str) -> Path:
    return DATA_ROOT / subject / "processed/ActPasSynSentence"


def load_audio_onsets() -> pd.DataFrame:
    path = LEGACY_DIR / "noun_role_audio_onsets.csv"
    onsets = pd.read_csv(path)
    required = {"sound_file", "role", "noun", "noun_onset_sec_in_audio", "onset_source"}
    missing = sorted(required - set(onsets.columns))
    if missing:
        raise ValueError(f"{path} lacks {missing}")
    if len(onsets) != 40 or onsets[["sound_file", "role"]].duplicated().any():
        raise ValueError("Expected exactly two unique noun locks for each of 20 stimuli")
    return onsets.rename(
        columns={
            "condition": "stimulus_condition",
            "sentence": "stimulus_sentence",
            "function_word": "stimulus_function_word",
            "onset_source": "noun_onset_source",
        }
    )


def load_corrected_events(subject: str, run: str, onsets: pd.DataFrame) -> pd.DataFrame:
    td = task_dir(subject)
    meta_path = td / "event_related_active_passive/active_passive_event_related_trial_metadata.csv"
    metadata = pd.read_csv(meta_path)
    metadata = metadata.loc[metadata["run"].astype(str).eq(run)].copy()
    events = pd.read_csv(td / run / "events.csv")
    trials = metadata.merge(
        events[["trial_index", "sound_file", "time_from_crop_start_sec"]],
        on=["trial_index", "sound_file"],
        how="inner",
        validate="one_to_one",
    )
    if len(trials) != 20:
        raise ValueError(f"{run}: expected 20 sentence trials, found {len(trials)}")
    expanded = trials.merge(onsets, on="sound_file", how="inner", validate="one_to_many")
    if len(expanded) != 40:
        raise ValueError(f"{run}: expected 40 noun-slot events, found {len(expanded)}")
    expanded["subject"] = subject
    expanded["corrected_event_bin"] = np.rint(
        1000.0
        * (
            expanded["time_from_crop_start_sec"].astype(float)
            + expanded["noun_onset_sec_in_audio"].astype(float)
        )
    ).astype(int)

    legacy_path = LEGACY_DIR / "noun_role_event_metadata.csv"
    legacy = pd.read_csv(legacy_path)
    legacy = legacy.loc[
        legacy["patient"].astype(str).eq(subject)
        & legacy["run"].astype(str).eq(run),
        ["trial_index", "sound_file", "role", "event_bin_1000Hz"],
    ].rename(columns={"event_bin_1000Hz": "legacy_event_bin"})
    expanded = expanded.merge(
        legacy,
        on=["trial_index", "sound_file", "role"],
        how="left",
        validate="one_to_one",
    )
    if expanded["legacy_event_bin"].isna().any():
        raise ValueError(f"{run}: missing legacy noun events")
    expanded["legacy_event_bin"] = expanded["legacy_event_bin"].astype(int)
    expanded["onset_shift_ms"] = expanded["corrected_event_bin"] - expanded["legacy_event_bin"]

    for role in (ROLE_SUBJECT, ROLE_PREDICATE):
        role_rows = expanded.loc[expanded["role"].eq(role)]
        if len(role_rows) != 20 or set(role_rows["noun"].astype(str)) != set(NOUN_ORDER):
            raise ValueError(f"{run}: {role} role does not contain exactly one observation per noun")
    return expanded


def paired_event_order(events: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for role in (ROLE_SUBJECT, ROLE_PREDICATE):
        indexed = events.loc[events["role"].eq(role)].set_index("noun", drop=False)
        parts.append(indexed.loc[list(NOUN_ORDER)].reset_index(drop=True))
    ordered = pd.concat(parts, ignore_index=True)
    if ordered.iloc[:20]["noun"].astype(str).tolist() != list(NOUN_ORDER):
        raise ValueError("subject noun order mismatch")
    if ordered.iloc[20:]["noun"].astype(str).tolist() != list(NOUN_ORDER):
        raise ValueError("predicate noun order mismatch")
    return ordered


def extract_raw_epochs(firing_rate_hz: np.ndarray, event_bins: np.ndarray) -> np.ndarray:
    raw_start = DISPLAY_START_MS - FILTER_PAD_MS
    raw_stop = DISPLAY_STOP_MS + FILTER_PAD_MS
    expected = raw_stop - raw_start
    epochs: list[np.ndarray] = []
    for event_bin in np.asarray(event_bins, dtype=int):
        start = int(event_bin + raw_start)
        stop = int(event_bin + raw_stop)
        if start < 0 or stop > firing_rate_hz.shape[1]:
            raise ValueError(f"event bin {event_bin}: padded noun epoch outside recording")
        epoch = firing_rate_hz[:, start:stop]
        if epoch.shape[1] != expected:
            raise ValueError("unexpected noun epoch length")
        epochs.append(epoch)
    return np.stack(epochs, axis=0).astype(np.float32, copy=False)


def smooth_and_crop(raw: np.ndarray, sigma_ms: float) -> np.ndarray:
    smoothed = gaussian_filter1d(
        raw,
        sigma=sigma_ms,
        axis=2,
        mode="constant",
        cval=0.0,
        output=np.float32,
    )
    start = FILTER_PAD_MS
    stop = start + (DISPLAY_STOP_MS - DISPLAY_START_MS)
    return smoothed[:, :, start:stop]


def paired_cohens_dz(subject_values: np.ndarray, predicate_values: np.ndarray) -> np.ndarray:
    differences = np.asarray(subject_values, float) - np.asarray(predicate_values, float)
    mean = differences.mean(axis=0)
    sd = differences.std(axis=0, ddof=1)
    result = np.full(mean.shape, np.nan, dtype=float)
    np.divide(mean, sd, out=result, where=sd > 0)
    result[(sd == 0) & (mean == 0)] = 0.0
    return result


def wilcoxon_p_by_unit(subject_values: np.ndarray, predicate_values: np.ndarray) -> np.ndarray:
    result = np.ones(subject_values.shape[1], dtype=float)
    for unit_index in range(subject_values.shape[1]):
        difference = subject_values[:, unit_index] - predicate_values[:, unit_index]
        if np.allclose(difference, 0):
            result[unit_index] = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result[unit_index] = stats.wilcoxon(
                    subject_values[:, unit_index],
                    predicate_values[:, unit_index],
                    alternative="two-sided",
                    zero_method="wilcox",
                ).pvalue
    return result


def paired_stratum_screen(
    subject_response: np.ndarray,
    predicate_response: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Screen one noun class and return has-segment, direction, longest segment."""
    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = stats.ttest_rel(subject_response, predicate_response, axis=0)
    p_values = np.asarray(test.pvalue, dtype=float)
    p_values[~np.isfinite(p_values)] = 1.0
    difference = subject_response.mean(axis=0) - predicate_response.mean(axis=0)
    has_segment = np.zeros(subject_response.shape[1], dtype=bool)
    directions = np.full(subject_response.shape[1], "not_significant", dtype=object)
    longest = np.zeros(subject_response.shape[1], dtype=int)
    for unit_index in range(subject_response.shape[1]):
        segments = contiguous_segments(p_values[unit_index] < ALPHA)
        if not segments:
            continue
        has_segment[unit_index] = True
        longest[unit_index] = max(segment.duration_ms for segment in segments)
        qualifying = np.zeros(RESPONSE_STOP_MS - RESPONSE_START_MS, dtype=bool)
        for segment in segments:
            qualifying[segment.start_ms:segment.stop_ms] = True
        positive = int(np.sum(qualifying & (difference[unit_index] > 0)))
        negative = int(np.sum(qualifying & (difference[unit_index] < 0)))
        directions[unit_index] = (
            "subject_gt_predicate" if positive > negative
            else "predicate_gt_subject" if negative > positive
            else "mixed"
        )
    return has_segment, directions.astype(str), longest


def analyze_run(
    *,
    subject: str,
    run: str,
    epochs: np.ndarray,
    unit_ids: np.ndarray,
    ks_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    response_start = RESPONSE_START_MS - DISPLAY_START_MS
    response_stop = RESPONSE_STOP_MS - DISPLAY_START_MS
    baseline_start = BASELINE_START_MS - DISPLAY_START_MS
    baseline_stop = BASELINE_STOP_MS - DISPLAY_START_MS
    subject_epochs = epochs[:20]
    predicate_epochs = epochs[20:]
    subject_response = subject_epochs[:, :, response_start:response_stop]
    predicate_response = predicate_epochs[:, :, response_start:response_stop]

    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            paired = stats.ttest_rel(subject_response, predicate_response, axis=0)
            unpaired = stats.f_oneway(subject_response, predicate_response, axis=0)
    t_values = np.asarray(paired.statistic, dtype=float)
    p_values = np.asarray(paired.pvalue, dtype=float)
    unpaired_f = np.asarray(unpaired.statistic, dtype=float)
    unpaired_p = np.asarray(unpaired.pvalue, dtype=float)
    p_values[~np.isfinite(p_values)] = 1.0
    unpaired_p[~np.isfinite(unpaired_p)] = 1.0
    t_values[~np.isfinite(t_values)] = np.nan
    unpaired_f[~np.isfinite(unpaired_f)] = np.nan
    role_f = t_values ** 2

    subject_mean = subject_response.mean(axis=0)
    predicate_mean = predicate_response.mean(axis=0)
    difference = subject_mean - predicate_mean
    passing = p_values < ALPHA
    unpaired_passing = unpaired_p < ALPHA

    subject_window = subject_response.mean(axis=2)
    predicate_window = predicate_response.mean(axis=2)
    subject_baseline = subject_epochs[:, :, baseline_start:baseline_stop].mean(axis=2)
    predicate_baseline = predicate_epochs[:, :, baseline_start:baseline_stop].mean(axis=2)
    subject_bc = subject_window - subject_baseline
    predicate_bc = predicate_window - predicate_baseline
    dz_raw = paired_cohens_dz(subject_window, predicate_window)
    dz_bc = paired_cohens_dz(subject_bc, predicate_bc)
    wilcoxon_raw = wilcoxon_p_by_unit(subject_window, predicate_window)
    wilcoxon_bc = wilcoxon_p_by_unit(subject_bc, predicate_bc)
    agent_has, agent_direction, agent_longest = paired_stratum_screen(
        subject_response[:10], predicate_response[:10]
    )
    patient_has, patient_direction, patient_longest = paired_stratum_screen(
        subject_response[10:], predicate_response[10:]
    )

    unit_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for unit_index, unit_id in enumerate(unit_ids):
        mask = passing[unit_index]
        segments = contiguous_segments(mask)
        unpaired_segments = contiguous_segments(unpaired_passing[unit_index])
        qualifying = np.zeros(RESPONSE_STOP_MS - RESPONSE_START_MS, dtype=bool)
        for segment_id, segment in enumerate(segments):
            qualifying[segment.start_ms:segment.stop_ms] = True
            seg_diff = difference[unit_index, segment.start_ms:segment.stop_ms]
            segment_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "unit_id": int(unit_id),
                    "ks_label": str(ks_labels[unit_index]),
                    "segment_id": segment_id,
                    "start_ms": segment.start_ms,
                    "stop_ms_exclusive": segment.stop_ms,
                    "duration_ms": segment.duration_ms,
                    "direction": (
                        "subject_gt_predicate" if float(np.nanmean(seg_diff)) > 0
                        else "predicate_gt_subject" if float(np.nanmean(seg_diff)) < 0
                        else "mixed"
                    ),
                    "mean_subject_minus_predicate_hz": float(np.nanmean(seg_diff)),
                    "min_p": float(np.nanmin(p_values[unit_index, segment.start_ms:segment.stop_ms])),
                }
            )

        subject_ms = int(np.sum(qualifying & (difference[unit_index] > 0)))
        predicate_ms = int(np.sum(qualifying & (difference[unit_index] < 0)))
        if subject_ms > predicate_ms:
            direction = "subject_gt_predicate"
        elif predicate_ms > subject_ms:
            direction = "predicate_gt_subject"
        elif qualifying.any():
            direction = "mixed"
        else:
            direction = "not_significant"

        if agent_has[unit_index] and patient_has[unit_index]:
            if agent_direction[unit_index] == patient_direction[unit_index] and agent_direction[unit_index] != "mixed":
                noun_class_replication = "same_direction_both"
            elif "mixed" in (agent_direction[unit_index], patient_direction[unit_index]):
                noun_class_replication = "both_with_mixed"
            else:
                noun_class_replication = "opposite_direction_both"
        elif agent_has[unit_index]:
            noun_class_replication = "agent_only"
        elif patient_has[unit_index]:
            noun_class_replication = "patient_only"
        else:
            noun_class_replication = "neither"

        for time_index in np.flatnonzero(mask):
            point_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "unit_id": int(unit_id),
                    "ks_label": str(ks_labels[unit_index]),
                    "time_ms": int(time_index),
                    "role_t": float(t_values[unit_index, time_index]) if np.isfinite(t_values[unit_index, time_index]) else np.nan,
                    "role_f": float(role_f[unit_index, time_index]) if np.isfinite(role_f[unit_index, time_index]) else np.nan,
                    "role_p": float(p_values[unit_index, time_index]),
                    "subject_mean_hz": float(subject_mean[unit_index, time_index]),
                    "predicate_mean_hz": float(predicate_mean[unit_index, time_index]),
                    "subject_minus_predicate_hz": float(difference[unit_index, time_index]),
                    "in_qualifying_segment": bool(qualifying[time_index]),
                }
            )

        unit_rows.append(
            {
                "subject": subject,
                "run": run,
                "unit_id": int(unit_id),
                "ks_label": str(ks_labels[unit_index]),
                "n_noun_pairs": 20,
                "n_conditions": N_CONDITIONS,
                "pointwise_alpha": ALPHA,
                "min_pointwise_p": float(np.min(p_values[unit_index])),
                "n_pointwise_passing_bins": int(mask.sum()),
                "n_qualifying_segments": len(segments),
                "first_segment_onset_ms": segments[0].start_ms if segments else np.nan,
                "longest_segment_ms": max((segment.duration_ms for segment in segments), default=0),
                "total_qualifying_duration_ms": int(qualifying.sum()),
                "has_qualifying_segment": bool(segments),
                "qualifying_subject_gt_predicate_ms": subject_ms,
                "qualifying_predicate_gt_subject_ms": predicate_ms,
                "qualifying_direction": direction,
                "unpaired_anova_has_qualifying_segment": bool(unpaired_segments),
                "agent_noun_has_qualifying_segment": bool(agent_has[unit_index]),
                "agent_noun_direction": str(agent_direction[unit_index]),
                "agent_noun_longest_segment_ms": int(agent_longest[unit_index]),
                "patient_noun_has_qualifying_segment": bool(patient_has[unit_index]),
                "patient_noun_direction": str(patient_direction[unit_index]),
                "patient_noun_longest_segment_ms": int(patient_longest[unit_index]),
                "noun_class_replication": noun_class_replication,
                "raw_window_subject_mean_hz": float(subject_window[:, unit_index].mean()),
                "raw_window_predicate_mean_hz": float(predicate_window[:, unit_index].mean()),
                "raw_window_subject_minus_predicate_hz": float(
                    subject_window[:, unit_index].mean() - predicate_window[:, unit_index].mean()
                ),
                "raw_window_paired_cohens_dz": float(dz_raw[unit_index]),
                "raw_window_wilcoxon_p": float(wilcoxon_raw[unit_index]),
                "bc_window_subject_mean_hz": float(subject_bc[:, unit_index].mean()),
                "bc_window_predicate_mean_hz": float(predicate_bc[:, unit_index].mean()),
                "bc_window_subject_minus_predicate_hz": float(
                    subject_bc[:, unit_index].mean() - predicate_bc[:, unit_index].mean()
                ),
                "bc_window_paired_cohens_dz": float(dz_bc[unit_index]),
                "bc_window_wilcoxon_p": float(wilcoxon_bc[unit_index]),
            }
        )
    return pd.DataFrame(unit_rows), pd.DataFrame(segment_rows), pd.DataFrame(point_rows)


def depth_order(subject: str, run: str, unit_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = task_dir(subject) / "event_related_active_passive/depth_activation" / f"{subject}_unit_depth_activation_metrics.csv"
    depth = pd.read_csv(path)
    depth = depth.loc[depth["run"].astype(str).eq(run), ["unit_id", "depth_um_median"]]
    lookup = depth.set_index("unit_id")["depth_um_median"]
    missing = sorted(set(map(int, unit_ids)) - set(map(int, lookup.index)))
    if missing:
        raise ValueError(f"{run}: missing depths for units {missing[:5]}")
    values = np.asarray([lookup.loc[int(unit_id)] for unit_id in unit_ids], dtype=float)
    order = np.lexsort((unit_ids.astype(int), values))
    return order, values[order]


def build_counts(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (subject, run), frame in units.groupby(["subject", "run"], sort=False):
        selected = frame.loc[frame["has_qualifying_segment"]]
        directions = selected["qualifying_direction"].value_counts()
        rows.append(
            {
                "subject": subject,
                "run": run,
                "run_label": RUN_LABELS[str(run)],
                "n_units": len(frame),
                "n_significant": len(selected),
                "subject_gt_predicate": int(directions.get("subject_gt_predicate", 0)),
                "predicate_gt_subject": int(directions.get("predicate_gt_subject", 0)),
                "mixed": int(directions.get("mixed", 0)),
                "unpaired_anova_significant": int(frame["unpaired_anova_has_qualifying_segment"].sum()),
            }
        )
    sessions = pd.DataFrame(rows)
    numeric = [
        "n_units", "n_significant", "subject_gt_predicate", "predicate_gt_subject", "mixed",
        "unpaired_anova_significant",
    ]
    subjects = sessions.groupby("subject", as_index=False)[numeric].sum()
    return sessions, subjects


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_screen_summary(sessions: pd.DataFrame, subjects: pd.DataFrame, total_units: int) -> Path:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7), gridspec_kw={"width_ratios": [0.9, 1.35]})
    for ax, frame, labels, title in (
        (axes[0], subjects, subjects["subject"].tolist(), "Across patients"),
        (axes[1], sessions, sessions["run_label"].tolist(), "Across sessions"),
    ):
        x = np.arange(len(frame))
        a = frame["subject_gt_predicate"].to_numpy(int)
        b = frame["predicate_gt_subject"].to_numpy(int)
        m = frame["mixed"].to_numpy(int)
        ax.bar(x, a, color=SUBJECT_COLOR, label="Subject > post-FW noun")
        ax.bar(x, b, bottom=a, color=PREDICATE_COLOR, label="Subject < post-FW noun")
        ax.bar(x, m, bottom=a + b, color=MIXED_COLOR, label="Mixed")
        totals = a + b + m
        for xpos, value in zip(x, totals):
            ax.text(xpos, value + max(1, int(totals.max() * 0.025)), str(int(value)), ha="center", fontweight="bold")
        ax.set_ylim(0, max(1, float(totals.max()) * 1.20))
        ax.set_xticks(x, labels, rotation=28 if ax is axes[1] else 0, ha="right" if ax is axes[1] else "center")
        ax.set_ylabel("Screen-positive units")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    total = int(sessions["n_significant"].sum())
    fig.suptitle(
        f"Corrected noun-slot screen: {total}/{total_units:,} units pass",
        fontsize=19,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Same-noun paired test · corrected audio onset · Gaussian SDF σ=25 ms · p<0.025 for ≥6 consecutive 1-ms bins in 0–0.8 s",
        ha="center",
        color=GRAY,
        fontsize=9.6,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.91))
    path = FIG_DIR / "Fig_corrected_noun_slot_screen_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def plot_noun_class_control(units: pd.DataFrame) -> Path:
    """Show how pooled hits behave when agent and patient nouns are tested separately."""
    configure_style()
    selected = units.loc[units["has_qualifying_segment"]].copy()
    categories = (
        "same_direction_both",
        "opposite_direction_both",
        "agent_only",
        "patient_only",
        "neither",
        "both_with_mixed",
    )
    labels = {
        "same_direction_both": "Same direction in both noun classes",
        "opposite_direction_both": "Opposite direction in both noun classes",
        "agent_only": "Agent nouns only",
        "patient_only": "Patient nouns only",
        "neither": "Pooled only; neither stratum passes",
        "both_with_mixed": "Both; one stratum mixed",
    }
    colors = {
        "same_direction_both": "#009E73",
        "opposite_direction_both": "#D55E00",
        "agent_only": "#0072B2",
        "patient_only": "#56B4E9",
        "neither": "#B8C2BF",
        "both_with_mixed": MIXED_COLOR,
    }
    overall = selected["noun_class_replication"].value_counts().reindex(categories, fill_value=0)
    by_subject = pd.crosstab(selected["subject"], selected["noun_class_replication"]).reindex(
        index=SUBJECTS, columns=categories, fill_value=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), gridspec_kw={"width_ratios": [1.05, 1.35]})
    order = np.arange(len(categories))
    bars = axes[0].barh(
        order,
        overall.to_numpy(int),
        color=[colors[category] for category in categories],
        edgecolor="white",
    )
    axes[0].set_yticks(order, [labels[category] for category in categories])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Pooled screen-positive units")
    axes[0].set_title("Noun-class control", loc="left", fontsize=13, fontweight="bold")
    axes[0].grid(axis="x", color=LIGHT, linewidth=0.7)
    for bar, value in zip(bars, overall):
        axes[0].text(value + 1.5, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontweight="bold")
    axes[0].set_xlim(0, max(1, float(overall.max()) * 1.20))

    bottom = np.zeros(len(SUBJECTS), dtype=int)
    x = np.arange(len(SUBJECTS))
    for category in categories:
        values = by_subject[category].to_numpy(int)
        axes[1].bar(x, values, bottom=bottom, color=colors[category], label=labels[category])
        bottom += values
    for xpos, value in zip(x, bottom):
        axes[1].text(xpos, value + max(1, int(bottom.max() * 0.025)), str(int(value)), ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(1, float(bottom.max()) * 1.20))
    axes[1].set_xticks(x, SUBJECTS)
    axes[1].set_ylabel("Pooled screen-positive units")
    axes[1].set_title("Control outcome by patient", loc="left", fontsize=13, fontweight="bold")
    axes[1].grid(axis="y", color=LIGHT, linewidth=0.7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    same = int(overall["same_direction_both"])
    total = len(selected)
    fig.suptitle(
        f"Only {same}/{total} pooled noun-slot hits pass both noun classes in the same direction",
        fontsize=18.2,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.02,
        "Agent and patient nouns reverse their mapping between structural slot and active/passive voice. "
        "Each stratum uses 10 noun pairs, p<0.025 and ≥6 consecutive ms; lower n reduces power.",
        ha="center",
        color=GRAY,
        fontsize=9.4,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    path = FIG_DIR / "Fig_corrected_noun_slot_noun_class_control.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def contiguous_blocks(values: np.ndarray) -> list[tuple[str, int, int]]:
    values = np.asarray(values).astype(str)
    if len(values) == 0:
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, len(values)]
    return [(str(values[start]), int(start), int(stop)) for start, stop in zip(starts, stops)]


def filter_direction_payload(
    units: pd.DataFrame,
    population_meta: pd.DataFrame,
    subject_hz: np.ndarray,
    predicate_hz: np.ndarray,
    direction: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    selected = units.loc[
        units["has_qualifying_segment"] & units["qualifying_direction"].eq(direction),
        ["subject", "run", "unit_id"],
    ].copy()
    selected["selected"] = True
    meta = population_meta.merge(selected, on=["subject", "run", "unit_id"], how="left", validate="one_to_one")
    keep_selected = meta["selected"].eq(True).to_numpy(bool)
    meta = meta.loc[keep_selected].drop(columns="selected").reset_index(drop=True)
    subj = subject_hz[keep_selected]
    pred = predicate_hz[keep_selected]
    subj_mean = subj.mean(axis=1)
    pred_mean = pred.mean(axis=1)
    persistent_low = (subj_mean < LOW_FULL_WINDOW_MEAN_HZ) & (pred_mean < LOW_FULL_WINDOW_MEAN_HZ)
    persistent_high = (subj_mean > HIGH_FULL_WINDOW_MEAN_HZ) | (pred_mean > HIGH_FULL_WINDOW_MEAN_HZ)
    remove = persistent_low | persistent_high
    audited = meta.copy()
    audited["subject_full_window_mean_hz"] = subj_mean
    audited["predicate_full_window_mean_hz"] = pred_mean
    audited["noise_stripe_reason"] = np.select(
        [persistent_high, persistent_low],
        ["persistent_high", "persistent_low"],
        default="retained",
    )
    removed = audited.loc[remove].copy()
    kept = audited.loc[~remove].reset_index(drop=True)
    kept["stacked_row"] = np.arange(len(kept), dtype=int)
    return kept, subj[~remove], pred[~remove], removed


def draw_boundaries(ax: plt.Axes, metadata: pd.DataFrame, strip: bool = False) -> None:
    for _, _, stop in contiguous_blocks(metadata["run"].to_numpy())[:-1]:
        ax.axhline(stop, color="white", lw=0.9, alpha=0.95, zorder=5)
    for _, _, stop in contiguous_blocks(metadata["subject"].to_numpy())[:-1]:
        ax.axhline(stop, color="white", lw=2.6 if not strip else 1.5, zorder=6)
        ax.axhline(stop, color=INK, lw=0.6, zorder=7)


def plot_direction_heatmap(
    metadata: pd.DataFrame,
    subject_hz: np.ndarray,
    predicate_hz: np.ndarray,
    direction: str,
    rate_vmax: float,
    difference_vmax: float,
    raw_count: int,
) -> Path:
    configure_style()
    if len(metadata) == 0:
        raise ValueError(f"No rows remain for {direction}")
    relation = "Subject slot > post-function-word noun slot" if direction == "subject_gt_predicate" else "Subject slot < post-function-word noun slot"
    stem = "Fig_corrected_noun_slot_subject_gt_predicate_stacked_heatmap" if direction == "subject_gt_predicate" else "Fig_corrected_noun_slot_subject_lt_predicate_stacked_heatmap"
    n_units = len(metadata)
    difference = subject_hz - predicate_hz
    patient_blocks = contiguous_blocks(metadata["subject"].to_numpy())
    patient_codes = np.asarray([SUBJECTS.index(patient) for patient in metadata["subject"]], dtype=int)[:, None]

    fig = plt.figure(figsize=(13.9, 8.3))
    grid = fig.add_gridspec(1, 6, width_ratios=[0.075, 1, 1, 0.035, 1, 0.035], wspace=0.16)
    ax_strip = fig.add_subplot(grid[0, 0])
    ax_subject = fig.add_subplot(grid[0, 1], sharey=ax_strip)
    ax_predicate = fig.add_subplot(grid[0, 2], sharex=ax_subject, sharey=ax_strip)
    ax_rate_cbar = fig.add_subplot(grid[0, 3])
    ax_difference = fig.add_subplot(grid[0, 4], sharex=ax_subject, sharey=ax_strip)
    ax_difference_cbar = fig.add_subplot(grid[0, 5])

    ax_strip.imshow(
        patient_codes,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[0, 1, n_units, 0],
        cmap=ListedColormap([PATIENT_COLORS[p] for p in SUBJECTS]),
        vmin=-0.5,
        vmax=len(SUBJECTS) - 0.5,
    )
    centers = [(start + stop) / 2 for _, start, stop in patient_blocks]
    labels = [f"{patient}  n={stop - start}" for patient, start, stop in patient_blocks]
    ax_strip.set_yticks(centers, labels)
    for tick, (patient, _, _) in zip(ax_strip.get_yticklabels(), patient_blocks):
        tick.set_color(PATIENT_COLORS[patient])
        tick.set_fontweight("bold")
    ax_strip.set_xticks([])
    ax_strip.tick_params(axis="y", length=0, pad=7)
    ax_strip.set_title("Patient", fontsize=10.5, fontweight="bold", pad=12)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)
    draw_boundaries(ax_strip, metadata, strip=True)

    extent = [DISPLAY_START_MS / 1000, DISPLAY_STOP_MS / 1000, n_units, 0]
    image_rate = ax_subject.imshow(subject_hz, aspect="auto", origin="upper", extent=extent, cmap="magma", vmin=0, vmax=rate_vmax, interpolation="nearest", rasterized=True)
    ax_predicate.imshow(predicate_hz, aspect="auto", origin="upper", extent=extent, cmap="magma", vmin=0, vmax=rate_vmax, interpolation="nearest", rasterized=True)
    image_diff = ax_difference.imshow(difference, aspect="auto", origin="upper", extent=extent, cmap="RdBu_r", vmin=-difference_vmax, vmax=difference_vmax, interpolation="nearest", rasterized=True)
    axes = (ax_subject, ax_predicate, ax_difference)
    for index, (ax, title) in enumerate(zip(axes, ("Subject-slot noun", "Post-FW noun", "Subject − post-FW noun"))):
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlim(-1, 2)
        ax.set_ylim(n_units, 0)
        ax.tick_params(axis="y", left=False, labelleft=False)
        if index < 2:
            ax.set_xticks([])
        else:
            ax.set_xticks([-1, 0, 0.8, 2])
            ax.set_xlabel("Time from corrected noun onset (s)")
        for value, style in ((0.0, "--"), (0.8, ":")):
            ax.axvline(value, color="white", lw=1.6, ls=style, zorder=6)
            ax.axvline(value, color=INK, lw=0.6, ls=style, zorder=7)
        draw_boundaries(ax, metadata)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cb = fig.colorbar(image_rate, cax=ax_rate_cbar)
    cb.set_label("Mean firing rate (Hz)", rotation=270, labelpad=15)
    cb = fig.colorbar(image_diff, cax=ax_difference_cbar)
    cb.set_label("Subject − post-FW noun (Hz)", rotation=270, labelpad=15)

    patient_counts = metadata["subject"].value_counts().reindex(SUBJECTS, fill_value=0)
    count_text = " · ".join(f"{p} n={int(patient_counts[p])}" for p in SUBJECTS)
    fig.suptitle(f"Corrected noun-slot units: {relation}", fontsize=17.5, fontweight="bold", y=0.988)
    fig.text(0.54, 0.922, f"After display-only stripe QC: n={n_units} of {raw_count} · {count_text}", ha="center", color=GRAY, fontsize=10.2)
    fig.text(
        0.54,
        0.02,
        "Selection: same-noun paired role test p<0.025 for ≥6 consecutive 1-ms bins in 0–0.8 s. "
        "Display QC removes rows with either full-window mean >20 Hz or both means <2 Hz.\n"
        "Condition means use Gaussian smoothing σ=35 ms; dashed line=0 s, dotted line=0.8 s. Only the right panel carries the shared x-axis.",
        ha="center",
        color=GRAY,
        fontsize=8.9,
    )
    fig.subplots_adjust(left=0.13, right=0.965, top=0.87, bottom=0.12)
    path = FIG_DIR / f"{stem}.png"
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    onsets = load_audio_onsets()
    all_units: list[pd.DataFrame] = []
    all_segments: list[pd.DataFrame] = []
    all_points: list[pd.DataFrame] = []
    event_audit: list[pd.DataFrame] = []
    pop_meta_parts: list[pd.DataFrame] = []
    pop_subject_parts: list[np.ndarray] = []
    pop_predicate_parts: list[np.ndarray] = []
    discovered_runs: list[str] = []
    global_offset = 0

    for subject in SUBJECTS:
        metadata = pd.read_csv(task_dir(subject) / "event_related_active_passive/active_passive_event_related_trial_metadata.csv")
        for run in metadata["run"].astype(str).drop_duplicates():
            discovered_runs.append(run)
            events = paired_event_order(load_corrected_events(subject, run, onsets))
            event_audit.append(
                events[
                    [
                        "subject", "run", "trial_index", "sound_file", "condition", "role", "noun",
                        "noun_onset_sec_in_audio", "noun_onset_source", "legacy_event_bin", "corrected_event_bin",
                        "onset_shift_ms",
                    ]
                ].rename(columns={"condition": "sentence_condition"})
            )
            archive = np.load(
                task_dir(subject) / run / "firing_rate_1000Hz/firing_rate_1000Hz_all_clusters.npz",
                allow_pickle=True,
            )
            firing_rate = archive["firing_rate_hz"].astype(np.float32, copy=False)
            unit_ids = archive["unit_ids"].astype(int)
            ks_labels = archive["ks_labels"].astype(str)
            raw = extract_raw_epochs(firing_rate, events["corrected_event_bin"].to_numpy())
            stats_epochs = smooth_and_crop(raw, SDF_SIGMA_MS)
            units, segments, points = analyze_run(
                subject=subject,
                run=run,
                epochs=stats_epochs,
                unit_ids=unit_ids,
                ks_labels=ks_labels,
            )
            all_units.append(units)
            all_segments.append(segments)
            all_points.append(points)

            display_epochs = smooth_and_crop(raw, DISPLAY_SIGMA_MS)
            subject_mean = display_epochs[:20].mean(axis=0)
            predicate_mean = display_epochs[20:].mean(axis=0)
            order, depths = depth_order(subject, run, unit_ids)
            n_run = len(unit_ids)
            pop_subject_parts.append(subject_mean[order].astype(np.float32))
            pop_predicate_parts.append(predicate_mean[order].astype(np.float32))
            pop_meta_parts.append(
                pd.DataFrame(
                    {
                        "subject": subject,
                        "subject_index": SUBJECTS.index(subject),
                        "run": run,
                        "unit_id": unit_ids[order],
                        "ks_label": ks_labels[order],
                        "depth_um": depths,
                        "source_unit_index": order,
                        "population_row": np.arange(global_offset, global_offset + n_run),
                    }
                )
            )
            global_offset += n_run
            del raw, stats_epochs, display_epochs

    if tuple(discovered_runs) != RUN_ORDER:
        raise ValueError(f"Run order mismatch: {discovered_runs}")
    units = pd.concat(all_units, ignore_index=True)
    segments = pd.concat(all_segments, ignore_index=True) if any(len(frame) for frame in all_segments) else pd.DataFrame()
    points = pd.concat(all_points, ignore_index=True) if any(len(frame) for frame in all_points) else pd.DataFrame()
    events_all = pd.concat(event_audit, ignore_index=True)
    population_meta = pd.concat(pop_meta_parts, ignore_index=True)
    population_subject = np.vstack(pop_subject_parts)
    population_predicate = np.vstack(pop_predicate_parts)
    if len(units) != 1782 or len(population_meta) != 1782:
        raise ValueError(f"Expected 1,782 units, got results={len(units)}, population={len(population_meta)}")

    sessions, subjects = build_counts(units)
    units.to_csv(OUT_DIR / "corrected_noun_slot_unit_results.csv", index=False)
    segments.to_csv(OUT_DIR / "qualifying_segments.csv", index=False)
    points.to_csv(OUT_DIR / "pointwise_passing_bins.csv", index=False)
    events_all.to_csv(OUT_DIR / "corrected_noun_event_audit.csv", index=False)
    sessions.to_csv(OUT_DIR / "session_counts.csv", index=False)
    subjects.to_csv(OUT_DIR / "subject_counts.csv", index=False)
    population_meta.to_csv(OUT_DIR / "population_heatmap_unit_order.csv", index=False)
    np.savez_compressed(
        OUT_DIR / "corrected_noun_slot_population_heatmaps_data.npz",
        time_sec=np.arange(DISPLAY_START_MS, DISPLAY_STOP_MS) / 1000,
        subject_hz=population_subject,
        predicate_hz=population_predicate,
        subject_minus_predicate_hz=population_subject - population_predicate,
        subjects=population_meta["subject"].astype(str).to_numpy(),
        runs=population_meta["run"].astype(str).to_numpy(),
        unit_ids=population_meta["unit_id"].to_numpy(int),
        ks_labels=population_meta["ks_label"].astype(str).to_numpy(),
        depth_um=population_meta["depth_um"].to_numpy(float),
    )

    old = pd.read_csv(LEGACY_DIR / "noun_role_structure_encoding_units.csv")
    old_crosswalk = old.merge(
        units,
        left_on=["patient", "run", "unit_id"],
        right_on=["subject", "run", "unit_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_old", "_new"),
    )
    old_crosswalk.to_csv(OUT_DIR / "legacy_719_unit_crosswalk.csv", index=False)

    summary_figure = plot_screen_summary(sessions, subjects, len(units))
    noun_class_figure = plot_noun_class_control(units)
    payloads = {}
    removed_parts = []
    for direction in ("subject_gt_predicate", "predicate_gt_subject"):
        raw_count = int((units["has_qualifying_segment"] & units["qualifying_direction"].eq(direction)).sum())
        meta, subj, pred, removed = filter_direction_payload(
            units,
            population_meta,
            population_subject,
            population_predicate,
            direction,
        )
        meta["direction"] = direction
        removed["direction"] = direction
        payloads[direction] = (meta, subj, pred, raw_count)
        removed_parts.append(removed)
    removed_rows = pd.concat(removed_parts, ignore_index=True)
    removed_rows.to_csv(OUT_DIR / "stacked_significant_heatmap_removed_noise_rows.csv", index=False)
    kept_order = pd.concat([payload[0] for payload in payloads.values()], ignore_index=True)

    rate_values = np.concatenate(
        [matrix.ravel() for meta, subj, pred, _ in payloads.values() for matrix in (subj, pred)]
    )
    difference_values = np.concatenate(
        [(subj - pred).ravel() for meta, subj, pred, _ in payloads.values()]
    )
    rate_vmax = max(1.0, float(np.nanpercentile(rate_values, 98)))
    difference_vmax = max(1.0, float(np.nanpercentile(np.abs(difference_values), 98)))
    kept_order["rate_vmax_hz_shared"] = rate_vmax
    kept_order["difference_abs_vmax_hz_shared"] = difference_vmax
    kept_order.to_csv(OUT_DIR / "stacked_significant_heatmap_unit_order.csv", index=False)
    heatmap_paths = {}
    for direction, (meta, subj, pred, raw_count) in payloads.items():
        heatmap_paths[direction] = str(
            plot_direction_heatmap(
                meta,
                subj,
                pred,
                direction,
                rate_vmax,
                difference_vmax,
                raw_count,
            )
        )

    selected = units["has_qualifying_segment"]
    direction_counts = units.loc[selected, "qualifying_direction"].value_counts()
    noun_class_counts = units.loc[selected, "noun_class_replication"].value_counts()
    old_hits = old_crosswalk["significant_run_fdr05"].astype(bool)
    summary = {
        "method": {
            "comparison": "same noun identity in sentence-initial subject slot versus post-function-word noun slot",
            "unit_set": "all_clusters: KS good + MUA",
            "n_units": len(units),
            "n_noun_pairs_per_run": 20,
            "corrected_event": "file-play onset plus audio-measured noun onset",
            "sdf_sigma_ms": SDF_SIGMA_MS,
            "primary_test": "two-level repeated-measures role ANOVA (paired t equivalent) at each 1-ms bin",
            "response_window_ms": [RESPONSE_START_MS, RESPONSE_STOP_MS],
            "pointwise_alpha": ALPHA,
            "minimum_consecutive_bins": MIN_CONSECUTIVE_BINS,
            "gap_filling": False,
        },
        "counts": {
            "n_significant": int(selected.sum()),
            "subject_gt_predicate": int(direction_counts.get("subject_gt_predicate", 0)),
            "predicate_gt_subject": int(direction_counts.get("predicate_gt_subject", 0)),
            "mixed": int(direction_counts.get("mixed", 0)),
            "unpaired_anova_sensitivity_n": int(units["unpaired_anova_has_qualifying_segment"].sum()),
            "noun_class_control_within_pooled_hits": {
                category: int(noun_class_counts.get(category, 0))
                for category in (
                    "same_direction_both",
                    "opposite_direction_both",
                    "agent_only",
                    "patient_only",
                    "neither",
                    "both_with_mixed",
                )
            },
            "by_session": sessions.to_dict(orient="records"),
            "by_subject": subjects.to_dict(orient="records"),
        },
        "onset_shift_ms": {
            role: {
                "min": int(frame["onset_shift_ms"].min()),
                "median": float(frame["onset_shift_ms"].median()),
                "max": int(frame["onset_shift_ms"].max()),
            }
            for role, frame in events_all.groupby("role")
        },
        "legacy_crosswalk": {
            "legacy_unit_set_n": len(old_crosswalk),
            "legacy_run_fdr_hits": int(old_hits.sum()),
            "legacy_hits_passing_new_screen": int((old_hits & old_crosswalk["has_qualifying_segment"].astype(bool)).sum()),
            "all_legacy_units_passing_new_screen": int(old_crosswalk["has_qualifying_segment"].astype(bool).sum()),
            "warning": "Legacy FDR and corrected time-resolved screens use different alignments, unit sets, estimands, and multiplicity rules.",
        },
        "display_qc": {
            "persistent_low_threshold_hz": LOW_FULL_WINDOW_MEAN_HZ,
            "persistent_high_threshold_hz": HIGH_FULL_WINDOW_MEAN_HZ,
            "removed_n": len(removed_rows),
            "removed_reasons": removed_rows["noise_stripe_reason"].value_counts().to_dict(),
            "retained_n": len(kept_order),
            "shared_rate_vmax_hz": rate_vmax,
            "shared_difference_abs_vmax_hz": difference_vmax,
        },
        "outputs": {
            "summary_figure": str(summary_figure),
            "noun_class_control_figure": str(noun_class_figure),
            "heatmaps": heatmap_paths,
            "unit_results": str(OUT_DIR / "corrected_noun_slot_unit_results.csv"),
            "segments": str(OUT_DIR / "qualifying_segments.csv"),
            "pointwise_bins": str(OUT_DIR / "pointwise_passing_bins.csv"),
            "event_audit": str(OUT_DIR / "corrected_noun_event_audit.csv"),
            "legacy_crosswalk": str(OUT_DIR / "legacy_719_unit_crosswalk.csv"),
            "noise_audit": str(OUT_DIR / "stacked_significant_heatmap_removed_noise_rows.csv"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
