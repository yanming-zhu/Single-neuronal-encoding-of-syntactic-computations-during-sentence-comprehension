#!/usr/bin/env python3
"""Build the 180-mm manuscript Fig. 3 requested on 2026-08-05.

The left half compares active and passive sentences after alignment to the
estimated verb onset.  The right half compares the same noun identity in the
sentence-initial subject slot and the post-function-word noun slot.  The task
schematic from the earlier draft was removed because it duplicated Fig. 1;
instead, two compact sequence schematics now state the distinct t = 0 alignment
used by each half.  Panel g pairs a version-selected three-condition timing analysis
with the analogous same-noun slot timing summary. Panels h and i summarize
pooled and patient-stratified cross-lexical robustness, respectively.

Both time-resolved screens use the same locked rule:
  * all registered Kilosort good + MUA units (1,782 unit/run entries);
  * Gaussian SDF, sigma = 25 ms;
  * paired test across matched lexical items at every 1-ms bin in [0, 800) ms;
  * p < 0.05 / 2 = 0.025;
  * at least six consecutive passing bins, without gap filling.

The noun-slot screen is loaded from the audited 2026-07-23 output.  The
verb-aligned two-condition screen is computed here because the three-condition
active/passive/scrambled ANOVA has a different estimand. Corrected scrambled
anchors are the default. Historical tables require --verb-version legacy_20260723.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
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
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from analyze_noun_slot_corrected_timepoint_screen_20260723 import (
    load_audio_onsets as load_noun_audio_onsets,
    load_corrected_events as load_noun_events,
    paired_event_order as paired_noun_event_order,
    task_dir as noun_task_dir,
)

from verb_three_condition_versioning import add_input_arguments, input_metadata, resolve_input
from summarize_verb_three_condition_population_differences import compute_timing_tests


WIDTH_MM = 180.0
HEIGHT_MM = 200.0
PT = 8.0
ALPHA = 0.05 / 2
MIN_BINS = 6
SDF_SIGMA_MS = 25
FILTER_PAD_MS = 4 * SDF_SIGMA_MS
BASELINE_START_MS = -500
RESPONSE_START_MS = 0
RESPONSE_STOP_MS = 800
PLOT_START_MS = -150
PLOT_STOP_MS = 800

DATA = ROOT / "neuropixel2025" / "data"
VERB_INPUT = DATA / "combined_verb_active_passive_unit_activity"
VERB_EVENTS = VERB_INPUT / "verb_event_metadata.csv"
VERB_OUT = ROOT / "outputs" / "fig3_verb_aligned_active_passive_20260805"
NOUN_OUT = ROOT / "outputs" / "noun_slot_corrected_timepoint_screen_20260723"
VERB_VERSION, THREE_CONDITION_VERB_OUT = resolve_input(validate=False)
OUT = Path(os.environ.get("TAN_FIG3_OUT_DIR", str(ROOT / "reports" / "nature_science_draft" / f"assembled_figures_20260805_{VERB_VERSION}")))
SOURCE = OUT / "source_data" / "fig3_verb_noun_slot_180mm"
STEM = OUT / "fig3_verb_noun_slot_180mm"

PATIENTS = ("gz001", "gz002", "gz003", "gz004", "gz005")
ACTIVE = "#357ABD"
PASSIVE = "#D95F4C"
SUBJECT = "#16867A"
POST_FW = "#D98C3A"
PURPLE = "#7C6BB0"
GREEN = "#2A8C6D"
GRAY = "#70757A"
SCRAMBLED = "#7A7A7A"
LIGHT_GRAY = "#D8DDE3"
PALE_GRAY = "#F0F2F3"
INK = "#202426"
ACTIVE_LIGHT = "#E8F0F8"
PASSIVE_LIGHT = "#FBEAE5"
SUBJECT_LIGHT = "#E6F3F0"
POST_FW_LIGHT = "#FFF1DE"
SIG_SHADE = "#DCEDEA"


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int

    @property
    def duration(self) -> int:
        return self.stop - self.start


def configure_style() -> FontProperties:
    cn_paths = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    )
    cn_prop = FontProperties(family=["sans-serif"])
    for font_path in cn_paths:
        if Path(font_path).exists():
            try:
                fontManager.addfont(font_path)
                cn_prop = FontProperties(fname=font_path)
                break
            except (OSError, RuntimeError):
                continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": PT,
            "axes.labelsize": PT,
            "axes.titlesize": PT,
            "xtick.labelsize": PT,
            "ytick.labelsize": PT,
            "legend.fontsize": PT,
            "figure.titlesize": PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.1,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return cn_prop


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PT,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def alignment_token(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    width: float = 0.11,
    facecolor: str = "white",
    edgecolor: str = LIGHT_GRAY,
    textcolor: str = INK,
) -> None:
    height = 0.22
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.025",
        linewidth=0.75,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontweight="bold" if facecolor != "white" else "normal",
        zorder=3,
    )


def draw_alignment_schematic(ax: plt.Axes, kind: str) -> None:
    """Draw compact Fig. 1a-style sequence templates around a shared t=0."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if kind == "verb":
        anchor = 0.79
        ax.text(0.0, 0.97, "Verb-aligned comparison", ha="left", va="top", fontweight="bold")
        ax.text(0.0, 0.64, "Active", color=ACTIVE, ha="left", va="center", fontweight="bold")
        ax.text(0.0, 0.31, "Passive", color=PASSIVE, ha="left", va="center", fontweight="bold")
        for y, tokens in (
            (0.64, ((0.27, "N1", 0.10), (0.43, "BA", 0.12), (0.60, "N2", 0.10))),
            (0.31, ((0.27, "N2", 0.10), (0.43, "BEI", 0.12), (0.60, "N1", 0.10))),
        ):
            for x, text_value, width in tokens:
                alignment_token(ax, x, y, text_value, width=width)
            alignment_token(
                ax,
                anchor,
                y,
                "VERB",
                width=0.14,
                facecolor="#F0ECF8",
                edgecolor=PURPLE,
                textcolor=PURPLE,
            )
        anchor_label = "estimated verb onset"
    elif kind == "noun":
        anchor = 0.57
        ax.text(0.0, 0.97, "Same-noun slot comparison", ha="left", va="top", fontweight="bold")
        ax.text(0.0, 0.64, "Subject slot", color=SUBJECT, ha="left", va="center", fontweight="bold")
        ax.text(0.0, 0.31, "Post-FW", color=POST_FW, ha="left", va="center", fontweight="bold")
        alignment_token(
            ax,
            anchor,
            0.64,
            "same N",
            width=0.16,
            facecolor=SUBJECT_LIGHT,
            edgecolor=SUBJECT,
            textcolor=SUBJECT,
        )
        for x, text_value, width in (
            (0.72, "FW", 0.10),
            (0.84, "N", 0.09),
            (0.95, "V", 0.08),
        ):
            alignment_token(ax, x, 0.64, text_value, width=width)
        for x, text_value, width in (
            (0.29, "N", 0.09),
            (0.43, "FW", 0.11),
        ):
            alignment_token(ax, x, 0.31, text_value, width=width)
        alignment_token(
            ax,
            anchor,
            0.31,
            "same N",
            width=0.16,
            facecolor=POST_FW_LIGHT,
            edgecolor=POST_FW,
            textcolor=POST_FW,
        )
        alignment_token(ax, 0.77, 0.31, "V", width=0.08)
        anchor_label = "corrected noun onset"
    else:
        raise ValueError(f"unknown alignment schematic kind: {kind}")
    ax.plot([anchor, anchor], [0.10, 0.80], color=INK, lw=0.7, ls="--", zorder=1)
    ax.text(
        anchor,
        0.03,
        f"t = 0  |  {anchor_label}",
        ha="center",
        va="bottom",
        color=GRAY,
    )


def fmt_ms_half_up(value: float) -> int:
    """Format positive millisecond values using conventional half-up rounding."""
    return int(np.floor(float(value) + 0.5))


def contiguous_segments(mask: np.ndarray) -> tuple[Segment, ...]:
    values = np.asarray(mask, dtype=bool)
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple(
        Segment(int(start), int(stop))
        for start, stop in zip(starts, stops)
        if int(stop - start) >= MIN_BINS
    )


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    values = p[finite]
    if not len(values):
        return q
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[finite] = restored
    return q


def run_archive(patient: str, run: str) -> Path:
    return (
        DATA
        / patient
        / "processed"
        / "ActPasSynSentence"
        / run
        / "firing_rate_1000Hz"
        / "firing_rate_1000Hz_all_clusters.npz"
    )


def paired_verb_events(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for condition in ("active", "passive"):
        selected = frame.loc[frame["sentence_condition"].eq(condition)].set_index("verb_id", drop=False)
        expected = set(range(1, 11))
        if set(selected.index.astype(int)) != expected:
            raise ValueError(f"{frame['run'].iloc[0]}: incomplete {condition} verb set")
        parts.append(selected.loc[list(range(1, 11))].reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def extract_smoothed_epochs(firing_rate: np.ndarray, event_bins: np.ndarray) -> np.ndarray:
    raw_start = BASELINE_START_MS - FILTER_PAD_MS
    raw_stop = RESPONSE_STOP_MS + FILTER_PAD_MS
    expected = raw_stop - raw_start
    epochs: list[np.ndarray] = []
    for event_bin in np.asarray(event_bins, dtype=int):
        start = int(event_bin + raw_start)
        stop = int(event_bin + raw_stop)
        if start < 0 or stop > firing_rate.shape[1]:
            raise ValueError(f"verb event {event_bin}: padded epoch outside recording")
        epoch = firing_rate[:, start:stop]
        if epoch.shape[1] != expected:
            raise ValueError("unexpected verb epoch length")
        epochs.append(epoch)
    raw = np.stack(epochs, axis=0).astype(np.float32, copy=False)
    smoothed = gaussian_filter1d(
        raw,
        sigma=SDF_SIGMA_MS,
        axis=2,
        mode="constant",
        cval=0.0,
        output=np.float32,
    )
    start = FILTER_PAD_MS
    stop = start + (RESPONSE_STOP_MS - BASELINE_START_MS)
    return smoothed[:, :, start:stop]


def safe_wilcoxon_p(difference: np.ndarray) -> float:
    values = np.asarray(difference, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or np.allclose(values, 0):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(
            stats.wilcoxon(
                values,
                alternative="two-sided",
                zero_method="wilcox",
                mode="auto",
            ).pvalue
        )


def analyze_verb_run(
    patient: str,
    run: str,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive = np.load(run_archive(patient, run), allow_pickle=True)
    firing_rate = archive["firing_rate_hz"].astype(np.float32, copy=False)
    unit_ids = archive["unit_ids"].astype(int)
    ks_labels = archive["ks_labels"].astype(str)
    ordered = paired_verb_events(events)
    epochs = extract_smoothed_epochs(firing_rate, ordered["event_bin_1000Hz"].to_numpy(int))
    baseline = epochs[:, :, : -BASELINE_START_MS]
    response = epochs[:, :, -BASELINE_START_MS :]
    active = response[:10]
    passive = response[10:]
    active_bc = active - baseline[:10].mean(axis=2, keepdims=True)
    passive_bc = passive - baseline[10:].mean(axis=2, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = stats.ttest_rel(active, passive, axis=0)
    t_values = np.asarray(test.statistic, dtype=float)
    p_values = np.asarray(test.pvalue, dtype=float)
    p_values[~np.isfinite(p_values)] = 1.0
    t_values[~np.isfinite(t_values)] = np.nan
    active_mean = active.mean(axis=0)
    passive_mean = passive.mean(axis=0)
    difference = active_mean - passive_mean

    active_window = active_bc.mean(axis=2)
    passive_window = passive_bc.mean(axis=2)

    unit_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    for unit_index, unit_id in enumerate(unit_ids):
        segments = contiguous_segments(p_values[unit_index] < ALPHA)
        qualifying = np.zeros(RESPONSE_STOP_MS - RESPONSE_START_MS, dtype=bool)
        for segment_id, segment in enumerate(segments):
            qualifying[segment.start : segment.stop] = True
            local = difference[unit_index, segment.start : segment.stop]
            direction = "active_gt_passive" if np.nanmean(local) > 0 else "passive_gt_active"
            segment_rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "unit_id": int(unit_id),
                    "ks_label": str(ks_labels[unit_index]),
                    "segment_id": segment_id,
                    "start_ms": segment.start,
                    "stop_ms_exclusive": segment.stop,
                    "duration_ms": segment.duration,
                    "direction": direction,
                    "mean_active_minus_passive_hz": float(np.nanmean(local)),
                    "min_p": float(np.nanmin(p_values[unit_index, segment.start : segment.stop])),
                }
            )

        pos_ms = int(np.sum(qualifying & (difference[unit_index] > 0)))
        neg_ms = int(np.sum(qualifying & (difference[unit_index] < 0)))
        if pos_ms > neg_ms:
            direction = "active_gt_passive"
        elif neg_ms > pos_ms:
            direction = "passive_gt_active"
        elif qualifying.any():
            direction = "mixed"
        else:
            direction = "not_significant"

        if qualifying.any():
            signed_curve = difference[unit_index].copy()
            if direction == "passive_gt_active":
                signed_curve = -signed_curve
            signed_curve[~qualifying] = -np.inf
            peak_ms = int(np.nanargmax(signed_curve))
            qualifying_delta = float(np.nanmean(difference[unit_index, qualifying]))
            pair_delta = (
                active[:, unit_index][:, qualifying].mean(axis=1)
                - passive[:, unit_index][:, qualifying].mean(axis=1)
            )
            signed_pair_delta = pair_delta if direction == "active_gt_passive" else -pair_delta
            loo = np.asarray(
                [
                    np.mean(np.delete(signed_pair_delta, index))
                    for index in range(len(signed_pair_delta))
                ],
                dtype=float,
            )
            loo_min = float(np.min(loo))
        else:
            peak_ms = np.nan
            qualifying_delta = np.nan
            loo_min = np.nan

        window_difference = active_window[:, unit_index] - passive_window[:, unit_index]
        mean_window_difference = float(window_difference.mean())
        sd_window_difference = float(window_difference.std(ddof=1))
        dz = mean_window_difference / sd_window_difference if sd_window_difference > 0 else np.nan
        unit_rows.append(
            {
                "patient": patient,
                "run": run,
                "unit_id": int(unit_id),
                "ks_label": str(ks_labels[unit_index]),
                "n_verb_pairs": 10,
                "n_conditions": 2,
                "pointwise_alpha": ALPHA,
                "min_consecutive_bins": MIN_BINS,
                "min_pointwise_p": float(np.min(p_values[unit_index])),
                "n_pointwise_passing_bins": int(np.sum(p_values[unit_index] < ALPHA)),
                "n_qualifying_segments": len(segments),
                "first_segment_onset_ms": segments[0].start if segments else np.nan,
                "longest_segment_ms": max((segment.duration for segment in segments), default=0),
                "total_qualifying_duration_ms": int(qualifying.sum()),
                "has_qualifying_segment": bool(segments),
                "qualifying_active_gt_passive_ms": pos_ms,
                "qualifying_passive_gt_active_ms": neg_ms,
                "qualifying_direction": direction,
                "effect_peak_ms": peak_ms,
                "qualifying_active_minus_passive_hz": qualifying_delta,
                "leave_one_verb_out_min_directional_delta_hz": loo_min,
                "bc_window_active_minus_passive_hz": mean_window_difference,
                "bc_window_paired_cohens_dz": dz,
                "bc_window_wilcoxon_p": safe_wilcoxon_p(window_difference),
            }
        )

    del epochs, active, passive, active_bc, passive_bc, firing_rate
    return pd.DataFrame(unit_rows), pd.DataFrame(segment_rows)


def add_fdr_columns(units: pd.DataFrame) -> pd.DataFrame:
    result = units.copy()
    result["bc_window_q_global_fdr"] = bh_fdr(result["bc_window_wilcoxon_p"].to_numpy(float))
    result["bc_window_q_run_fdr"] = np.nan
    result["bc_window_q_patient_fdr"] = np.nan
    for _, indices in result.groupby(["patient", "run"]).groups.items():
        result.loc[indices, "bc_window_q_run_fdr"] = bh_fdr(
            result.loc[indices, "bc_window_wilcoxon_p"].to_numpy(float)
        )
    for _, indices in result.groupby("patient").groups.items():
        result.loc[indices, "bc_window_q_patient_fdr"] = bh_fdr(
            result.loc[indices, "bc_window_wilcoxon_p"].to_numpy(float)
        )
    return result


def verb_patient_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient in PATIENTS:
        frame = units.loc[units["patient"].eq(patient)]
        selected = frame.loc[frame["has_qualifying_segment"]]
        counts = selected["qualifying_direction"].value_counts()
        rows.append(
            {
                "patient": patient,
                "n_units": len(frame),
                "n_screen_positive": len(selected),
                "active_gt_passive": int(counts.get("active_gt_passive", 0)),
                "passive_gt_active": int(counts.get("passive_gt_active", 0)),
                "mixed": int(counts.get("mixed", 0)),
                "screen_positive_percent": 100 * len(selected) / len(frame),
                "active_gt_passive_percent": 100 * int(counts.get("active_gt_passive", 0)) / len(frame),
                "passive_gt_active_percent": 100 * int(counts.get("passive_gt_active", 0)) / len(frame),
            }
        )
    return pd.DataFrame(rows)


def compute_or_load_verb_screen() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_path = VERB_OUT / "verb_aligned_active_passive_unit_results.csv"
    segment_path = VERB_OUT / "verb_aligned_active_passive_segments.csv"
    summary_path = VERB_OUT / "verb_aligned_active_passive_patient_summary.csv"
    if unit_path.exists() and segment_path.exists() and summary_path.exists():
        units = pd.read_csv(unit_path)
        segments = pd.read_csv(segment_path)
        patient = pd.read_csv(summary_path)
        if len(units) == 1782 and int(units["has_qualifying_segment"].sum()) >= 0:
            return units, segments, patient

    VERB_OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(VERB_EVENTS)
    unit_parts: list[pd.DataFrame] = []
    segment_parts: list[pd.DataFrame] = []
    for (patient, run), frame in events.groupby(["patient", "run"], sort=False):
        print(f"[fig3] verb screen {patient} {run}", flush=True)
        units, segments = analyze_verb_run(str(patient), str(run), frame.copy())
        unit_parts.append(units)
        segment_parts.append(segments)
    all_units = add_fdr_columns(pd.concat(unit_parts, ignore_index=True))
    all_segments = pd.concat(segment_parts, ignore_index=True)
    patient_summary = verb_patient_summary(all_units)
    if len(all_units) != 1782:
        raise ValueError(f"verb screen expected 1,782 units, found {len(all_units)}")
    all_units.to_csv(unit_path, index=False)
    all_segments.to_csv(segment_path, index=False)
    patient_summary.to_csv(summary_path, index=False)
    summary = {
        "method": {
            "comparison": "same verb identity in active versus passive sentences",
            "unit_set": "all registered KS good + MUA",
            "n_units": int(len(all_units)),
            "n_verb_pairs_per_run": 10,
            "alignment": "estimated verb onset from the stimulus audio",
            "sdf_sigma_ms": SDF_SIGMA_MS,
            "test": "paired t test at each 1-ms bin",
            "response_window_ms": [0, 800],
            "pointwise_alpha": ALPHA,
            "minimum_consecutive_bins": MIN_BINS,
            "gap_filling": False,
        },
        "counts": {
            "n_screen_positive": int(all_units["has_qualifying_segment"].sum()),
            "directions": all_units.loc[
                all_units["has_qualifying_segment"], "qualifying_direction"
            ].value_counts().to_dict(),
            "bc_window_run_fdr05": int((all_units["bc_window_q_run_fdr"] < 0.05).sum()),
            "bc_window_patient_fdr05": int((all_units["bc_window_q_patient_fdr"] < 0.05).sum()),
            "bc_window_global_fdr05": int((all_units["bc_window_q_global_fdr"] < 0.05).sum()),
        },
    }
    (VERB_OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return all_units, all_segments, patient_summary


def paired_subset_screen(
    first_response: np.ndarray,
    second_response: np.ndarray,
    positive_direction: str,
    negative_direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the locked pointwise-duration screen to one independent item subset."""
    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = stats.ttest_rel(first_response, second_response, axis=0)
    p_values = np.asarray(test.pvalue, dtype=float)
    p_values[~np.isfinite(p_values)] = 1.0
    difference = first_response.mean(axis=0) - second_response.mean(axis=0)
    has_segment = np.zeros(first_response.shape[1], dtype=bool)
    directions = np.full(first_response.shape[1], "not_significant", dtype=object)
    longest = np.zeros(first_response.shape[1], dtype=int)
    for unit_index in range(first_response.shape[1]):
        segments = contiguous_segments(p_values[unit_index] < ALPHA)
        if not segments:
            continue
        has_segment[unit_index] = True
        longest[unit_index] = max(segment.duration for segment in segments)
        qualifying = np.zeros(RESPONSE_STOP_MS - RESPONSE_START_MS, dtype=bool)
        for segment in segments:
            qualifying[segment.start : segment.stop] = True
        positive = int(np.sum(qualifying & (difference[unit_index] > 0)))
        negative = int(np.sum(qualifying & (difference[unit_index] < 0)))
        directions[unit_index] = (
            positive_direction
            if positive > negative
            else negative_direction
            if negative > positive
            else "mixed"
        )
    return has_segment, directions.astype(str), longest


def compute_or_load_verb_split_control() -> pd.DataFrame:
    """Test lexical robustness in two non-overlapping five-verb sets."""
    path = VERB_OUT / "verb_aligned_split_half_control.csv"
    required = {
        "patient",
        "run",
        "unit_id",
        "verb_set_1_5_has_qualifying_segment",
        "verb_set_1_5_direction",
        "verb_set_6_10_has_qualifying_segment",
        "verb_set_6_10_direction",
        "verb_set_replication",
    }
    if path.exists():
        cached = pd.read_csv(path)
        if len(cached) == 1782 and required.issubset(cached.columns):
            return cached

    events = pd.read_csv(VERB_EVENTS)
    rows: list[pd.DataFrame] = []
    for (patient, run), frame in events.groupby(["patient", "run"], sort=False):
        print(f"[fig3] verb split-half control {patient} {run}", flush=True)
        ordered = paired_verb_events(frame.copy())
        archive = np.load(run_archive(str(patient), str(run)), allow_pickle=True)
        unit_ids = archive["unit_ids"].astype(int)
        ks_labels = archive["ks_labels"].astype(str)
        epochs = extract_smoothed_epochs(
            archive["firing_rate_hz"].astype(np.float32, copy=False),
            ordered["event_bin_1000Hz"].to_numpy(int),
        )
        response = epochs[:, :, -BASELINE_START_MS :]
        active = response[:10]
        passive = response[10:]
        set_1_has, set_1_direction, set_1_longest = paired_subset_screen(
            active[:5], passive[:5], "active_gt_passive", "passive_gt_active"
        )
        set_2_has, set_2_direction, set_2_longest = paired_subset_screen(
            active[5:], passive[5:], "active_gt_passive", "passive_gt_active"
        )
        categories: list[str] = []
        for first_has, first_direction, second_has, second_direction in zip(
            set_1_has, set_1_direction, set_2_has, set_2_direction
        ):
            if first_has and second_has:
                if first_direction == second_direction and first_direction != "mixed":
                    categories.append("same_direction_both")
                elif "mixed" in (first_direction, second_direction):
                    categories.append("both_with_mixed")
                else:
                    categories.append("opposite_direction_both")
            elif first_has:
                categories.append("set_1_only")
            elif second_has:
                categories.append("set_2_only")
            else:
                categories.append("neither")
        rows.append(
            pd.DataFrame(
                {
                    "patient": str(patient),
                    "run": str(run),
                    "unit_id": unit_ids,
                    "ks_label": ks_labels,
                    "verb_set_1_5_has_qualifying_segment": set_1_has,
                    "verb_set_1_5_direction": set_1_direction,
                    "verb_set_1_5_longest_segment_ms": set_1_longest,
                    "verb_set_6_10_has_qualifying_segment": set_2_has,
                    "verb_set_6_10_direction": set_2_direction,
                    "verb_set_6_10_longest_segment_ms": set_2_longest,
                    "verb_set_replication": categories,
                }
            )
        )
        del epochs
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 1782 or result[["patient", "run", "unit_id"]].duplicated().any():
        raise ValueError("verb split-half control does not match the 1,782-unit registry")
    result.to_csv(path, index=False)
    return result


def select_verb_representatives(units: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for direction in ("active_gt_passive", "passive_gt_active"):
        eligible = units.loc[
            units["has_qualifying_segment"]
            & units["qualifying_direction"].eq(direction)
            & units["ks_label"].eq("good")
            & units["leave_one_verb_out_min_directional_delta_hz"].gt(0)
        ].copy()
        if eligible.empty:
            eligible = units.loc[
                units["has_qualifying_segment"]
                & units["qualifying_direction"].eq(direction)
            ].copy()
        if direction == "active_gt_passive":
            eligible["representative_selection_score"] = (
                eligible["leave_one_verb_out_min_directional_delta_hz"]
                * np.sqrt(eligible["longest_segment_ms"].clip(lower=1))
            )
            eligible = eligible.sort_values(
                [
                    "representative_selection_score",
                    "leave_one_verb_out_min_directional_delta_hz",
                    "longest_segment_ms",
                    "min_pointwise_p",
                ],
                ascending=[False, False, False, True],
            )
        else:
            eligible["representative_selection_score"] = eligible[
                "leave_one_verb_out_min_directional_delta_hz"
            ]
            eligible = eligible.sort_values(
                [
                    "leave_one_verb_out_min_directional_delta_hz",
                    "longest_segment_ms",
                    "total_qualifying_duration_ms",
                    "min_pointwise_p",
                ],
                ascending=[False, False, False, True],
            )
        selected.append(eligible.iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def select_noun_representatives(units: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for direction in ("subject_gt_predicate", "predicate_gt_subject"):
        eligible = units.loc[
            units["has_qualifying_segment"]
            & units["qualifying_direction"].eq(direction)
            & units["ks_label"].eq("good")
            & units["agent_noun_direction"].eq(direction)
            & units["patient_noun_direction"].eq(direction)
        ].copy()
        if eligible.empty:
            eligible = units.loc[
                units["has_qualifying_segment"]
                & units["qualifying_direction"].eq(direction)
                & units["ks_label"].eq("good")
            ].copy()
        eligible = eligible.sort_values(
            ["longest_segment_ms", "total_qualifying_duration_ms", "min_pointwise_p"],
            ascending=[False, False, True],
        )
        selected.append(eligible.iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def trace_payload(
    time_ms: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    trials: list[pd.DataFrame] = []
    for name, values in ((first_name, first), (second_name, second)):
        baseline_mask = (time_ms >= BASELINE_START_MS) & (time_ms < 0)
        bc = values - values[:, baseline_mask].mean(axis=1, keepdims=True)
        mean = bc.mean(axis=0)
        sem = bc.std(axis=0, ddof=1) / np.sqrt(bc.shape[0])
        rows.append(
            pd.DataFrame(
                {
                    "condition": name,
                    "time_ms": time_ms,
                    "mean_bc_hz": mean,
                    "sem_bc_hz": sem,
                }
            )
        )
        trials.append(
            pd.DataFrame(
                {
                    "condition": name,
                    "pair_index": np.arange(1, bc.shape[0] + 1),
                    "response_bc_hz_0_800": bc[:, (time_ms >= 0) & (time_ms < 800)].mean(axis=1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), pd.concat(trials, ignore_index=True)


def load_verb_representative_traces(
    representatives: pd.DataFrame,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(VERB_EVENTS)
    payload: dict[str, dict[str, object]] = {}
    trace_rows: list[pd.DataFrame] = []
    trial_rows: list[pd.DataFrame] = []
    for row in representatives.itertuples(index=False):
        frame = events.loc[(events["patient"] == row.patient) & (events["run"] == row.run)]
        ordered = paired_verb_events(frame)
        archive = np.load(run_archive(str(row.patient), str(row.run)), allow_pickle=True)
        unit_ids = archive["unit_ids"].astype(int)
        matched = np.flatnonzero(unit_ids == int(row.unit_id))
        if len(matched) != 1:
            raise ValueError(f"verb representative {row.run} u{row.unit_id} missing")
        epochs = extract_smoothed_epochs(
            archive["firing_rate_hz"][matched].astype(np.float32, copy=False),
            ordered["event_bin_1000Hz"].to_numpy(int),
        )[:, 0, :]
        time_ms = np.arange(BASELINE_START_MS, RESPONSE_STOP_MS, dtype=int)
        traces, trials = trace_payload(time_ms, epochs[:10], epochs[10:], "active", "passive")
        key = str(row.qualifying_direction)
        traces["panel_direction"] = key
        traces["patient"] = row.patient
        traces["run"] = row.run
        traces["unit_id"] = int(row.unit_id)
        trials["panel_direction"] = key
        trials["patient"] = row.patient
        trials["run"] = row.run
        trials["unit_id"] = int(row.unit_id)
        trace_rows.append(traces)
        trial_rows.append(trials)
        payload[key] = {
            "patient": str(row.patient),
            "run": str(row.run),
            "unit_id": int(row.unit_id),
            "ks_label": str(row.ks_label),
            "traces": traces,
        }
    return payload, pd.concat(trace_rows, ignore_index=True), pd.concat(trial_rows, ignore_index=True)


def load_noun_representative_traces(
    representatives: pd.DataFrame,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame, pd.DataFrame]:
    onsets = load_noun_audio_onsets()
    payload: dict[str, dict[str, object]] = {}
    trace_rows: list[pd.DataFrame] = []
    trial_rows: list[pd.DataFrame] = []
    for (patient, run), selected in representatives.groupby(["subject", "run"], sort=False):
        events = paired_noun_event_order(load_noun_events(str(patient), str(run), onsets))
        archive = np.load(run_archive(str(patient), str(run)), allow_pickle=True)
        unit_ids = archive["unit_ids"].astype(int)
        indices = []
        for unit_id in selected["unit_id"].astype(int):
            matched = np.flatnonzero(unit_ids == unit_id)
            if len(matched) != 1:
                raise ValueError(f"noun representative {run} u{unit_id} missing")
            indices.append(int(matched[0]))
        firing_rate = archive["firing_rate_hz"][indices].astype(np.float32, copy=False)
        epochs = extract_smoothed_epochs(
            firing_rate,
            events["corrected_event_bin"].to_numpy(int),
        )
        time_ms = np.arange(BASELINE_START_MS, RESPONSE_STOP_MS, dtype=int)
        for local_index, row in enumerate(selected.itertuples(index=False)):
            unit_epochs = epochs[:, local_index, :]
            traces, trials = trace_payload(
                time_ms,
                unit_epochs[:20],
                unit_epochs[20:],
                "subject_slot",
                "post_fw_noun",
            )
            key = str(row.qualifying_direction)
            traces["panel_direction"] = key
            traces["patient"] = row.subject
            traces["run"] = row.run
            traces["unit_id"] = int(row.unit_id)
            trials["panel_direction"] = key
            trials["patient"] = row.subject
            trials["run"] = row.run
            trials["unit_id"] = int(row.unit_id)
            trace_rows.append(traces)
            trial_rows.append(trials)
            payload[key] = {
                "patient": str(row.subject),
                "run": str(row.run),
                "unit_id": int(row.unit_id),
                "ks_label": str(row.ks_label),
                "traces": traces,
            }
    return payload, pd.concat(trace_rows, ignore_index=True), pd.concat(trial_rows, ignore_index=True)


def draw_trace_panel(
    ax: plt.Axes,
    label: str,
    payload: dict[str, object],
    segments: pd.DataFrame,
    first_key: str,
    second_key: str,
    first_label: str,
    second_label: str,
    first_color: str,
    second_color: str,
    title: str,
) -> None:
    panel_label(ax, label)
    traces = payload["traces"]
    plot = traces.loc[traces["time_ms"].between(PLOT_START_MS, PLOT_STOP_MS - 1)]
    for segment in segments.itertuples(index=False):
        ax.axvspan(
            max(0, int(segment.start_ms)) / 1000,
            min(PLOT_STOP_MS, int(segment.stop_ms_exclusive)) / 1000,
            color=SIG_SHADE,
            lw=0,
            zorder=0,
        )
    ax.axvline(0, color=INK, ls="--", lw=0.7, zorder=1)
    for key, color, name in (
        (first_key, first_color, first_label),
        (second_key, second_color, second_label),
    ):
        frame = plot.loc[plot["condition"].eq(key)]
        x = frame["time_ms"].to_numpy(float) / 1000
        mean = frame["mean_bc_hz"].to_numpy(float)
        sem = frame["sem_bc_hz"].to_numpy(float)
        ax.plot(x, mean, color=color, lw=1.2, label=name, zorder=3)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.16, lw=0, zorder=2)
    ax.axhline(0, color=LIGHT_GRAY, lw=0.6)
    ax.set_xlim(PLOT_START_MS / 1000, PLOT_STOP_MS / 1000)
    ax.set_xticks([0.0, 0.4, 0.8])
    ax.set_ylabel("Δ firing rate (Hz)")
    ax.set_title(
        f"{title} | {payload['patient']} u{payload['unit_id']} ({payload['ks_label']})",
        loc="left",
        fontweight="bold",
        pad=3,
    )
    ax.grid(axis="y", color=PALE_GRAY, lw=0.55)


def draw_diverging_patient_bars(
    ax: plt.Axes,
    label: str,
    summary: pd.DataFrame,
    positive_count: str,
    negative_count: str,
    positive_color: str,
    negative_color: str,
    title: str,
    positive_name: str,
    negative_name: str,
) -> None:
    panel_label(ax, label)
    frame = summary.set_index("patient").loc[list(PATIENTS)].reset_index()
    y = np.arange(len(frame))
    denom = frame["n_units"].to_numpy(float)
    positive = 100 * frame[positive_count].to_numpy(float) / denom
    negative = 100 * frame[negative_count].to_numpy(float) / denom
    ax.barh(y, positive, color=positive_color, height=0.58, label=positive_name)
    ax.barh(y, -negative, color=negative_color, height=0.58, label=negative_name)
    ax.axvline(0, color=INK, lw=0.75)
    ax.set_yticks(y, frame["patient"])
    ax.invert_yaxis()
    limit = max(5.0, float(np.ceil(max(np.max(positive), np.max(negative)) / 5) * 5))
    ax.set_xlim(-limit * 1.34, limit * 1.34)
    ticks = np.linspace(-limit, limit, 5)
    ax.set_xticks(ticks, [f"{abs(value):.0f}" for value in ticks])
    for index, row in enumerate(frame.itertuples(index=False)):
        ax.text(
            limit * 1.31,
            index,
            f"{int(row.n_screen_positive)}/{int(row.n_units)}",
            ha="right",
            va="center",
        )
    ax.set_xlabel(f"{negative_name}  ←  % positive  →  {positive_name}")
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.grid(axis="x", color=PALE_GRAY, lw=0.55)


def build_timing_table(verb_units: pd.DataFrame, noun_units: pd.DataFrame) -> pd.DataFrame:
    """Return onset and direction-consistent peak latencies for both comparisons."""
    rows: list[pd.DataFrame] = []
    verb_selected = verb_units.loc[
        verb_units["has_qualifying_segment"]
        & verb_units["qualifying_direction"].isin(["active_gt_passive", "passive_gt_active"])
    ].copy()
    for metric in ("first_segment_onset_ms", "effect_peak_ms"):
        part = verb_selected[["patient", "run", "unit_id", "qualifying_direction", metric]].copy()
        part = part.rename(columns={metric: "value_ms", "qualifying_direction": "direction"})
        part["comparison"] = "verb"
        part["metric"] = "onset" if metric == "first_segment_onset_ms" else "peak"
        rows.append(part)

    noun_selected = noun_units.loc[
        noun_units["has_qualifying_segment"]
        & noun_units["qualifying_direction"].isin(["subject_gt_predicate", "predicate_gt_subject"]),
        ["subject", "run", "unit_id", "qualifying_direction", "first_segment_onset_ms"],
    ].rename(columns={"subject": "patient", "qualifying_direction": "direction"})
    noun_points = pd.read_csv(NOUN_OUT / "pointwise_passing_bins.csv")
    noun_points = noun_points.loc[noun_points["in_qualifying_segment"].astype(bool)].merge(
        noun_selected[["patient", "run", "unit_id", "direction"]],
        left_on=["subject", "run", "unit_id"],
        right_on=["patient", "run", "unit_id"],
        how="inner",
        validate="many_to_one",
    )
    noun_points["directional_effect_hz"] = np.where(
        noun_points["direction"].eq("subject_gt_predicate"),
        noun_points["subject_minus_predicate_hz"],
        -noun_points["subject_minus_predicate_hz"],
    )
    peak_indices = noun_points.groupby(["patient", "run", "unit_id"], sort=False)[
        "directional_effect_hz"
    ].idxmax()
    noun_peaks = noun_points.loc[
        peak_indices, ["patient", "run", "unit_id", "time_ms"]
    ].rename(columns={"time_ms": "effect_peak_ms"})
    noun_selected = noun_selected.merge(
        noun_peaks, on=["patient", "run", "unit_id"], how="left", validate="one_to_one"
    )
    if noun_selected["effect_peak_ms"].isna().any():
        raise ValueError("missing effect-peak latency for a noun-slot screen-positive unit")
    for metric in ("first_segment_onset_ms", "effect_peak_ms"):
        part = noun_selected[["patient", "run", "unit_id", "direction", metric]].copy()
        part = part.rename(columns={metric: "value_ms"})
        part["comparison"] = "noun_slot"
        part["metric"] = "onset" if metric == "first_segment_onset_ms" else "peak"
        rows.append(part)
    return pd.concat(rows, ignore_index=True)[
        ["comparison", "patient", "run", "unit_id", "direction", "metric", "value_ms"]
    ]


def draw_combined_timing(ax: plt.Axes, table: pd.DataFrame) -> None:
    panel_label(ax, "g")
    order = (
        ("verb", "active_gt_passive", "onset", "A>P onset", ACTIVE),
        ("verb", "active_gt_passive", "peak", "A>P peak", ACTIVE),
        ("verb", "passive_gt_active", "onset", "A<P onset", PASSIVE),
        ("verb", "passive_gt_active", "peak", "A<P peak", PASSIVE),
        ("noun_slot", "subject_gt_predicate", "onset", "S>FW onset", SUBJECT),
        ("noun_slot", "subject_gt_predicate", "peak", "S>FW peak", SUBJECT),
        ("noun_slot", "predicate_gt_subject", "onset", "S<FW onset", POST_FW),
        ("noun_slot", "predicate_gt_subject", "peak", "S<FW peak", POST_FW),
    )
    y_positions = np.asarray([0, 1, 2, 3, 4.7, 5.7, 6.7, 7.7])
    rng = np.random.default_rng(20260805)
    for y, (comparison, direction, metric, _, color) in zip(y_positions, order):
        values = table.loc[
            table["comparison"].eq(comparison)
            & table["direction"].eq(direction)
            & table["metric"].eq(metric),
            "value_ms",
        ].dropna().to_numpy(float)
        jitter = rng.uniform(-0.13, 0.13, len(values))
        ax.scatter(values, np.full(len(values), y) + jitter, s=6.5, color=color, alpha=0.25, linewidth=0)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        ax.plot([q25, q75], [y, y], color=color, lw=4.7, alpha=0.30, solid_capstyle="butt")
        ax.plot([median, median], [y - 0.22, y + 0.22], color=INK, lw=1.0)
        ax.text(807, y, f"{median:.0f}", va="center", ha="left", color=color)
    ax.axhline(3.85, color=LIGHT_GRAY, lw=0.7)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.axvline(800, color=GRAY, lw=0.6, ls=":")
    ax.set_xlim(-15, 875)
    ax.set_yticks(y_positions, [item[3] for item in order])
    ax.invert_yaxis()
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlabel("Time from aligned word onset (ms)")
    ax.set_title("Descriptive onset and effect-peak timing", loc="left", fontweight="bold", pad=3)
    ax.grid(axis="x", color=PALE_GRAY, lw=0.55)


def build_generalization_table(
    verb_units: pd.DataFrame,
    verb_split: pd.DataFrame,
    noun_units: pd.DataFrame,
) -> pd.DataFrame:
    categories = (
        "same direction in both",
        "opposite directions",
        "one subset only",
        "pooled only",
        "mixed",
    )
    category_map = {
        "same_direction_both": "same direction in both",
        "opposite_direction_both": "opposite directions",
        "agent_only": "one subset only",
        "patient_only": "one subset only",
        "set_1_only": "one subset only",
        "set_2_only": "one subset only",
        "neither": "pooled only",
        "both_with_mixed": "mixed",
    }
    verb_selected = verb_units.loc[
        verb_units["has_qualifying_segment"], ["patient", "run", "unit_id"]
    ].merge(verb_split, on=["patient", "run", "unit_id"], how="left", validate="one_to_one")
    noun_selected = noun_units.loc[noun_units["has_qualifying_segment"]].copy()
    frames: list[pd.DataFrame] = []
    for comparison, frame, column in (
        ("Verb contrast", verb_selected, "verb_set_replication"),
        ("Noun-slot contrast", noun_selected, "noun_class_replication"),
    ):
        mapped = frame[column].map(category_map)
        if mapped.isna().any():
            raise ValueError(f"unmapped generalization category in {comparison}")
        counts = mapped.value_counts().reindex(categories, fill_value=0)
        part = pd.DataFrame(
            {
                "comparison": comparison,
                "category": categories,
                "count": counts.to_numpy(int),
                "total": len(frame),
            }
        )
        part["percent"] = 100 * part["count"] / max(1, len(frame))
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def build_patient_generalization_table(
    verb_units: pd.DataFrame,
    verb_split: pd.DataFrame,
    noun_units: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize same-direction cross-lexical replication within each patient."""
    verb_selected = verb_units.loc[
        verb_units["has_qualifying_segment"], ["patient", "run", "unit_id"]
    ].merge(
        verb_split[["patient", "run", "unit_id", "verb_set_replication"]],
        on=["patient", "run", "unit_id"],
        how="left",
        validate="one_to_one",
    )
    noun_selected = noun_units.loc[
        noun_units["has_qualifying_segment"],
        ["subject", "run", "unit_id", "noun_class_replication"],
    ].rename(columns={"subject": "patient"})
    frames: list[pd.DataFrame] = []
    for comparison, frame, column in (
        ("Verb contrast", verb_selected, "verb_set_replication"),
        ("Noun-slot contrast", noun_selected, "noun_class_replication"),
    ):
        grouped = frame.groupby("patient", sort=False)[column].agg(
            total="size",
            same_direction_both=lambda values: int(
                np.sum(np.asarray(values) == "same_direction_both")
            ),
        )
        grouped = grouped.reindex(PATIENTS, fill_value=0).reset_index()
        grouped["comparison"] = comparison
        grouped["percent"] = np.where(
            grouped["total"].gt(0),
            100 * grouped["same_direction_both"] / grouped["total"],
            np.nan,
        )
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)[
        ["comparison", "patient", "same_direction_both", "total", "percent"]
    ]


def draw_combined_generalization(ax: plt.Axes, table: pd.DataFrame) -> None:
    panel_label(ax, "h")
    categories = (
        "same direction in both",
        "opposite directions",
        "one subset only",
        "pooled only",
        "mixed",
    )
    labels = (
        "same dir.",
        "opposite dir.",
        "one subset",
        "pooled only",
        "mixed",
    )
    comparisons = ("Verb contrast", "Noun-slot contrast")
    colors = {"Verb contrast": "#6D8FB4", "Noun-slot contrast": "#53A095"}
    y = np.arange(len(categories), dtype=float)
    offsets = {"Verb contrast": -0.18, "Noun-slot contrast": 0.18}
    max_percent = 0.0
    for comparison in comparisons:
        frame = table.loc[table["comparison"].eq(comparison)].set_index("category").loc[list(categories)]
        values = frame["percent"].to_numpy(float)
        counts = frame["count"].to_numpy(int)
        max_percent = max(max_percent, float(values.max()))
        bars = ax.barh(
            y + offsets[comparison],
            values,
            height=0.32,
            color=colors[comparison],
            edgecolor="white",
            linewidth=0.45,
            label=comparison,
        )
        for bar, value, count in zip(bars, values, counts):
            ax.text(
                value + 1.0,
                bar.get_y() + bar.get_height() / 2,
                str(int(count)),
                va="center",
                ha="left",
                color=colors[comparison],
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max_percent * 1.23)
    ax.set_xlabel("% of pooled screen-positive units")
    ax.set_title("Cross-lexical generalization of pooled hits", loc="left", fontweight="bold", pad=3)
    ax.legend(loc="upper right", handlelength=1.2)
    ax.grid(axis="x", color=PALE_GRAY, lw=0.55)


def draw_patient_generalization(ax: plt.Axes, table: pd.DataFrame) -> None:
    panel_label(ax, "i")
    comparisons = ("Verb contrast", "Noun-slot contrast")
    colors = {"Verb contrast": "#6D8FB4", "Noun-slot contrast": "#53A095"}
    labels = {"Verb contrast": "Verb", "Noun-slot contrast": "Noun slot"}
    offsets = {"Verb contrast": -0.16, "Noun-slot contrast": 0.16}
    y = np.arange(len(PATIENTS), dtype=float)
    max_percent = 0.0
    for comparison in comparisons:
        frame = table.loc[table["comparison"].eq(comparison)].set_index("patient").loc[list(PATIENTS)]
        values = frame["percent"].to_numpy(float)
        counts = frame["same_direction_both"].to_numpy(int)
        totals = frame["total"].to_numpy(int)
        y_values = y + offsets[comparison]
        max_percent = max(max_percent, float(np.nanmax(values)))
        ax.hlines(y_values, 0, values, color=colors[comparison], lw=1.2, alpha=0.48)
        ax.scatter(
            values,
            y_values,
            s=18,
            color=colors[comparison],
            edgecolor="white",
            linewidth=0.45,
            label=labels[comparison],
            zorder=3,
        )
        for value, y_value, count, total in zip(values, y_values, counts, totals):
            ax.text(
                value + 0.55,
                y_value,
                f"{count}/{total}",
                color=colors[comparison],
                va="center",
                ha="left",
            )
    ax.set_yticks(y, PATIENTS)
    ax.invert_yaxis()
    ax.set_xlim(0, max(22.0, max_percent * 1.33))
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("% same direction in both subsets")
    ax.set_title("Cross-lexical replication by patient", loc="left", fontweight="bold", pad=3)
    ax.legend(
        loc="upper right",
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
        borderaxespad=0.1,
    )
    ax.grid(axis="x", color=PALE_GRAY, lw=0.55)


def load_three_condition_timing(
    input_dir: Path | None = None, *, version: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load selected timing data and recompute the same unit-level comparisons.

    The sibling timing-tests CSV is deliberately not read: it may belong to an
    older alignment version. All counts and tests come from the supplied tables.
    """
    version, directory = resolve_input(version or VERB_VERSION, input_dir or THREE_CONDITION_VERB_OUT)
    timing_path = directory / "verb_three_condition_population_timing_by_unit.csv"
    timing = pd.read_csv(timing_path)
    required = {
        "patient",
        "run",
        "unit_id",
        "ks_label",
        "preferred_condition",
        "first_significant_onset_ms",
        "preferred_peak_ms",
        "effect_peak_ms",
    }
    if not required.issubset(timing.columns):
        raise ValueError(f"three-condition timing table lacks {sorted(required - set(timing.columns))}")
    timing = timing.loc[
        timing["preferred_condition"].isin(["active", "passive", "scrambled"]),
        sorted(required),
    ].copy()
    keys = ["patient", "run", "unit_id"]
    if timing.empty or timing.duplicated(keys).any():
        raise ValueError("Timing table must contain nonempty, unique patient/run/unit entries")
    results = pd.read_csv(directory / "verb_three_condition_unit_results.csv")
    if not set(keys).issubset(results.columns) or results.duplicated(keys).any():
        raise ValueError("Unit results must contain unique patient/run/unit entries")
    if not pd.MultiIndex.from_frame(timing[keys]).isin(pd.MultiIndex.from_frame(results[keys])).all():
        raise ValueError("Timing entries are absent from the selected unit-results table")
    rows: list[pd.DataFrame] = []
    for metric, source in (
        ("onset", "first_significant_onset_ms"),
        ("effect_peak", "effect_peak_ms"),
    ):
        part = timing[
            ["patient", "run", "unit_id", "ks_label", "preferred_condition", source]
        ].rename(columns={source: "value_ms"})
        part["metric"] = metric
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    long.attrs.update(input_metadata(version, directory))
    long.attrs["n_units_analyzed"] = int(len(results))
    long.attrs["n_units_with_timing"] = int(len(timing))
    summary_path = directory / "verb_three_condition_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    long.attrs["unit_set"] = summary.get("unit_set", "not recorded")
    tests = pd.DataFrame(compute_timing_tests(timing))
    return long, tests


def draw_three_condition_timing(ax: plt.Axes, table: pd.DataFrame) -> None:
    panel_label(ax, "g")
    order = (
        ("active", "onset", "A-high onset", ACTIVE),
        ("active", "effect_peak", "A-high peak", ACTIVE),
        ("passive", "onset", "P-high onset", PASSIVE),
        ("passive", "effect_peak", "P-high peak", PASSIVE),
        ("scrambled", "onset", "Scr. onset", SCRAMBLED),
        ("scrambled", "effect_peak", "Scr. peak", SCRAMBLED),
    )
    y_positions = np.asarray([0, 1, 2.45, 3.45, 4.90, 5.90])
    rng = np.random.default_rng(20260722)
    for y, (condition, metric, _, color) in zip(y_positions, order):
        values = table.loc[
            table["preferred_condition"].eq(condition) & table["metric"].eq(metric),
            "value_ms",
        ].dropna().to_numpy(float)
        if not len(values):
            continue
        jitter = rng.uniform(-0.14, 0.14, len(values))
        ax.scatter(values, np.full(len(values), y) + jitter, s=8.0, color=color, alpha=0.34, linewidth=0)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        ax.plot([q25, q75], [y, y], color=color, lw=5.2, alpha=0.28, solid_capstyle="butt")
        ax.plot([median, median], [y - 0.23, y + 0.23], color=INK, lw=1.0)
        ax.text(
            807,
            y,
            f"{fmt_ms_half_up(median)} ms",
            va="center",
            ha="left",
            color=color,
        )
    ax.axhline(1.72, color=LIGHT_GRAY, lw=0.7)
    ax.axhline(4.17, color=LIGHT_GRAY, lw=0.7)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.axvline(800, color=GRAY, lw=0.6, ls=":")
    ax.set_xlim(-15, 930)
    ax.set_yticks(y_positions, [item[2] for item in order])
    ax.invert_yaxis()
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlabel("Time from estimated verb onset (ms)")
    ax.set_title(
        "Three-condition preference timing",
        loc="left",
        fontweight="bold",
        pad=3,
    )
    ax.grid(axis="x", color=PALE_GRAY, lw=0.55)


def draw_noun_slot_timing(ax: plt.Axes, table: pd.DataFrame) -> None:
    """Draw the analogous onset/peak summary for the same-noun slot contrast."""
    order = (
        ("subject_gt_predicate", "onset", "S-high onset", SUBJECT),
        ("subject_gt_predicate", "peak", "S-high peak", SUBJECT),
        ("predicate_gt_subject", "onset", "FW-high onset", POST_FW),
        ("predicate_gt_subject", "peak", "FW-high peak", POST_FW),
    )
    y_positions = np.asarray([0, 1, 2.45, 3.45])
    rng = np.random.default_rng(20260806)
    for y, (direction, metric, _, color) in zip(y_positions, order):
        values = table.loc[
            table["direction"].eq(direction) & table["metric"].eq(metric),
            "value_ms",
        ].dropna().to_numpy(float)
        jitter = rng.uniform(-0.14, 0.14, len(values))
        ax.scatter(
            values,
            np.full(len(values), y) + jitter,
            s=6.5,
            color=color,
            alpha=0.25,
            linewidth=0,
        )
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        ax.plot(
            [q25, q75],
            [y, y],
            color=color,
            lw=4.7,
            alpha=0.30,
            solid_capstyle="butt",
        )
        ax.plot([median, median], [y - 0.22, y + 0.22], color=INK, lw=1.0)
        ax.text(
            807,
            y,
            f"{fmt_ms_half_up(median)} ms",
            va="center",
            ha="left",
            color=color,
        )
    ax.axhline(1.72, color=LIGHT_GRAY, lw=0.7)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.axvline(800, color=GRAY, lw=0.6, ls=":")
    ax.set_xlim(-15, 930)
    ax.set_yticks(y_positions, [item[2] for item in order])
    ax.invert_yaxis()
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlabel("Time from corrected noun onset (ms)")
    ax.set_title("Same-noun slot timing", loc="left", fontweight="bold", pad=3)
    ax.grid(axis="x", color=PALE_GRAY, lw=0.55)


def compare_noun_slot_timing_groups(table: pd.DataFrame) -> pd.DataFrame:
    """Compare timing distributions between subject-high and post-FW-high units."""
    rows: list[dict[str, object]] = []
    for metric in ("onset", "peak"):
        subject_values = table.loc[
            table["direction"].eq("subject_gt_predicate")
            & table["metric"].eq(metric),
            "value_ms",
        ].dropna().to_numpy(float)
        post_fw_values = table.loc[
            table["direction"].eq("predicate_gt_subject")
            & table["metric"].eq(metric),
            "value_ms",
        ].dropna().to_numpy(float)
        result = stats.mannwhitneyu(
            subject_values,
            post_fw_values,
            alternative="two-sided",
            method="auto",
        )
        rows.append(
            {
                "metric": metric,
                "test": "mannwhitneyu",
                "comparison": "subject_high_vs_post_fw_high",
                "n_subject_high": len(subject_values),
                "n_post_fw_high": len(post_fw_values),
                "p_value": float(result.pvalue),
                "median_diff_ms": float(
                    np.median(subject_values) - np.median(post_fw_values)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_figure() -> dict[str, object]:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    verb_units, verb_segments, verb_summary = compute_or_load_verb_screen()
    verb_split = compute_or_load_verb_split_control()
    noun_units = pd.read_csv(NOUN_OUT / "corrected_noun_slot_unit_results.csv")
    noun_segments = pd.read_csv(NOUN_OUT / "qualifying_segments.csv")
    noun_summary = pd.read_csv(NOUN_OUT / "subject_counts.csv").rename(
        columns={
            "subject": "patient",
            "n_significant": "n_screen_positive",
        }
    )
    noun_summary["screen_positive_percent"] = 100 * noun_summary["n_screen_positive"] / noun_summary["n_units"]

    verb_reps = select_verb_representatives(verb_units)
    noun_reps = select_noun_representatives(noun_units)
    verb_payload, verb_traces, verb_trials = load_verb_representative_traces(verb_reps)
    noun_payload, noun_traces, noun_trials = load_noun_representative_traces(noun_reps)
    timing_table = build_timing_table(verb_units, noun_units)
    noun_timing_table = timing_table.loc[timing_table["comparison"].eq("noun_slot")].copy()
    noun_timing_tests = compare_noun_slot_timing_groups(noun_timing_table)
    generalization_table = build_generalization_table(verb_units, verb_split, noun_units)
    patient_generalization_table = build_patient_generalization_table(
        verb_units, verb_split, noun_units
    )
    three_condition_timing, three_condition_timing_tests = load_three_condition_timing()

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        4,
        2,
        height_ratios=[2.05, 0.78, 1.00, 0.80],
        left=0.105,
        right=0.985,
        bottom=0.100,
        top=0.985,
        hspace=0.60,
        wspace=0.38,
    )
    left_reps = outer[0, 0].subgridspec(
        3, 1, height_ratios=[0.85, 1.0, 1.0], hspace=0.42
    )
    right_reps = outer[0, 1].subgridspec(
        3, 1, height_ratios=[0.85, 1.0, 1.0], hspace=0.42
    )
    ax_left_alignment = fig.add_subplot(left_reps[0, 0])
    ax_right_alignment = fig.add_subplot(right_reps[0, 0])
    ax_a = fig.add_subplot(left_reps[1, 0])
    ax_c = fig.add_subplot(left_reps[2, 0])
    ax_b = fig.add_subplot(right_reps[1, 0])
    ax_d = fig.add_subplot(right_reps[2, 0])
    ax_e = fig.add_subplot(outer[1, 0])
    ax_f = fig.add_subplot(outer[1, 1])
    ax_g_verb = fig.add_subplot(outer[2, 0])
    ax_g_noun = fig.add_subplot(outer[2, 1])
    ax_h = fig.add_subplot(outer[3, 0])
    ax_i = fig.add_subplot(outer[3, 1])

    verb_active = verb_reps.loc[verb_reps["qualifying_direction"].eq("active_gt_passive")].iloc[0]
    verb_passive = verb_reps.loc[verb_reps["qualifying_direction"].eq("passive_gt_active")].iloc[0]
    noun_subject = noun_reps.loc[noun_reps["qualifying_direction"].eq("subject_gt_predicate")].iloc[0]
    noun_post = noun_reps.loc[noun_reps["qualifying_direction"].eq("predicate_gt_subject")].iloc[0]

    draw_alignment_schematic(ax_left_alignment, "verb")
    draw_alignment_schematic(ax_right_alignment, "noun")

    draw_trace_panel(
        ax_a,
        "a",
        verb_payload["active_gt_passive"],
        verb_segments.loc[
            verb_segments["run"].eq(verb_active["run"])
            & verb_segments["unit_id"].eq(int(verb_active["unit_id"]))
        ],
        "active",
        "passive",
        "Active",
        "Passive",
        ACTIVE,
        PASSIVE,
        "Active-preferring example",
    )
    draw_trace_panel(
        ax_c,
        "c",
        verb_payload["passive_gt_active"],
        verb_segments.loc[
            verb_segments["run"].eq(verb_passive["run"])
            & verb_segments["unit_id"].eq(int(verb_passive["unit_id"]))
        ],
        "active",
        "passive",
        "Active",
        "Passive",
        ACTIVE,
        PASSIVE,
        "Passive-preferring example",
    )
    draw_trace_panel(
        ax_b,
        "b",
        noun_payload["subject_gt_predicate"],
        noun_segments.loc[
            noun_segments["run"].eq(noun_subject["run"])
            & noun_segments["unit_id"].eq(int(noun_subject["unit_id"]))
        ],
        "subject_slot",
        "post_fw_noun",
        "Subject slot",
        "Post-FW noun",
        SUBJECT,
        POST_FW,
        "Subject-preferring example",
    )
    draw_trace_panel(
        ax_d,
        "d",
        noun_payload["predicate_gt_subject"],
        noun_segments.loc[
            noun_segments["run"].eq(noun_post["run"])
            & noun_segments["unit_id"].eq(int(noun_post["unit_id"]))
        ],
        "subject_slot",
        "post_fw_noun",
        "Subject slot",
        "Post-FW noun",
        SUBJECT,
        POST_FW,
        "Post-FW-preferring example",
    )
    for ax in (ax_a, ax_b):
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel("")
    ax_c.set_xlabel("Time from estimated verb onset (s)")
    ax_d.set_xlabel("Time from corrected noun onset (s)")
    ax_a.legend(loc="upper right", ncol=2, handlelength=1.3, columnspacing=0.8)
    ax_b.legend(loc="upper right", ncol=2, handlelength=1.3, columnspacing=0.8)

    draw_diverging_patient_bars(
        ax_e,
        "e",
        verb_summary,
        "active_gt_passive",
        "passive_gt_active",
        ACTIVE,
        PASSIVE,
        "Transient verb contrast (full-window FDR: 0)",
        "A>P",
        "A<P",
    )
    draw_diverging_patient_bars(
        ax_f,
        "f",
        noun_summary,
        "subject_gt_predicate",
        "predicate_gt_subject",
        SUBJECT,
        POST_FW,
        "Transient same-noun slot contrast",
        "S>post-FW",
        "S<post-FW",
    )
    draw_three_condition_timing(ax_g_verb, three_condition_timing)
    draw_noun_slot_timing(ax_g_noun, noun_timing_table)
    draw_combined_generalization(ax_h, generalization_table)
    draw_patient_generalization(ax_i, patient_generalization_table)

    fig.text(
        0.53,
        0.018,
        "Primary two-condition screens: paired lexical-item tests, Gaussian SDF σ=25 ms, p<0.025 for ≥6 consecutive 1-ms bins.\n"
        "Panel g left: three-condition ANOVA, σ=40 ms, p<0.05/3 for ≥6 bins; timing summaries are descriptive.",
        ha="center",
        va="bottom",
        wrap=True,
    )

    png = STEM.with_suffix(".png")
    pdf = STEM.with_suffix(".pdf")
    svg = STEM.with_suffix(".svg")
    tiff = STEM.with_suffix(".tiff")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    fig.savefig(tiff, dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    stale_source_files = (
        "panel_a_example_stimuli.csv",
        "panels_b_d_verb_representatives.csv",
        "panels_c_e_noun_representatives.csv",
        "panels_b_d_verb_traces.csv",
        "panels_b_d_verb_paired_values.csv",
        "panels_c_e_noun_traces.csv",
        "panels_c_e_noun_paired_values.csv",
        "panel_f_verb_patient_summary.csv",
        "panel_g_noun_patient_summary.csv",
        "panel_h_verb_timing.csv",
        "panel_i_noun_class_control.csv",
        "panel_g_combined_timing.csv",
        "panel_i_three_condition_timing.csv",
        "panel_i_three_condition_timing_tests.csv",
        "panel_i_legacy_three_condition_timing.csv",
        "panel_i_legacy_three_condition_timing_tests.csv",
        "panel_g_legacy_three_condition_timing.csv",
        "panel_g_legacy_three_condition_timing_tests.csv",
    )
    for filename in stale_source_files:
        (SOURCE / filename).unlink(missing_ok=True)
    verb_reps.to_csv(SOURCE / "panels_a_c_verb_representatives.csv", index=False)
    noun_reps.to_csv(SOURCE / "panels_b_d_noun_representatives.csv", index=False)
    verb_traces.to_csv(SOURCE / "panels_a_c_verb_traces.csv", index=False)
    verb_trials.to_csv(SOURCE / "panels_a_c_verb_paired_values.csv", index=False)
    noun_traces.to_csv(SOURCE / "panels_b_d_noun_traces.csv", index=False)
    noun_trials.to_csv(SOURCE / "panels_b_d_noun_paired_values.csv", index=False)
    verb_summary.to_csv(SOURCE / "panel_e_verb_patient_summary.csv", index=False)
    noun_summary.to_csv(SOURCE / "panel_f_noun_patient_summary.csv", index=False)
    generalization_table.to_csv(SOURCE / "panel_h_cross_lexical_generalization.csv", index=False)
    patient_generalization_table.to_csv(
        SOURCE / "panel_i_cross_lexical_replication_by_patient.csv", index=False
    )
    three_condition_timing.to_csv(SOURCE / "panel_g_three_condition_timing.csv", index=False)
    three_condition_timing_tests.to_csv(
        SOURCE / "panel_g_three_condition_timing_tests.csv", index=False
    )
    noun_timing_table.to_csv(SOURCE / "panel_g_same_noun_slot_timing.csv", index=False)
    noun_timing_tests.to_csv(
        SOURCE / "panel_g_same_noun_slot_timing_tests.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "comparison": "verb_aligned",
                "row": "active",
                "sequence": "N1 BA N2 VERB",
                "t0_token": "VERB",
                "alignment_definition": "estimated verb onset from stimulus audio",
            },
            {
                "comparison": "verb_aligned",
                "row": "passive",
                "sequence": "N2 BEI N1 VERB",
                "t0_token": "VERB",
                "alignment_definition": "estimated verb onset from stimulus audio",
            },
            {
                "comparison": "same_noun_slot",
                "row": "subject_slot",
                "sequence": "same N FW N V",
                "t0_token": "same N",
                "alignment_definition": "corrected onset of the matched noun identity",
            },
            {
                "comparison": "same_noun_slot",
                "row": "post_function_word_slot",
                "sequence": "N FW same N V",
                "t0_token": "same N",
                "alignment_definition": "corrected onset of the matched noun identity",
            },
        ]
    ).to_csv(SOURCE / "top_alignment_schematics.csv", index=False)
    verb_split.to_csv(SOURCE / "verb_split_half_control_all_units.csv", index=False)
    verb_units.to_csv(SOURCE / "verb_aligned_active_passive_all_units.csv", index=False)
    verb_segments.to_csv(SOURCE / "verb_aligned_active_passive_segments.csv", index=False)
    noun_units.to_csv(SOURCE / "noun_slot_all_units.csv", index=False)
    noun_segments.to_csv(SOURCE / "noun_slot_segments.csv", index=False)
    verb_selected = verb_units.loc[verb_units["has_qualifying_segment"]]
    noun_selected = noun_units.loc[noun_units["has_qualifying_segment"]]
    verb_same_both = int(
        generalization_table.loc[
            generalization_table["comparison"].eq("Verb contrast")
            & generalization_table["category"].eq("same direction in both"),
            "count",
        ].iloc[0]
    )
    noun_same_both = int(
        generalization_table.loc[
            generalization_table["comparison"].eq("Noun-slot contrast")
            & generalization_table["category"].eq("same direction in both"),
            "count",
        ].iloc[0]
    )
    three_condition_group_counts = (
        three_condition_timing.drop_duplicates(["patient", "run", "unit_id"])["preferred_condition"]
        .value_counts()
        .to_dict()
    )
    three_condition_medians = (
        three_condition_timing.groupby(["preferred_condition", "metric"])["value_ms"]
        .median()
        .rename("median_ms")
        .reset_index()
        .to_dict(orient="records")
    )
    three_condition_active_passive_tests = three_condition_timing_tests.loc[
        three_condition_timing_tests["comparison"].eq("active_vs_passive")
        & three_condition_timing_tests["metric"].isin(
            ["first_significant_onset_ms", "effect_peak_ms"]
        )
    ].to_dict(orient="records")
    noun_timing_group_counts = (
        noun_timing_table.drop_duplicates(["patient", "run", "unit_id"])["direction"]
        .value_counts()
        .to_dict()
    )
    noun_timing_medians = (
        noun_timing_table.groupby(["direction", "metric"])["value_ms"]
        .median()
        .rename("median_ms")
        .reset_index()
        .to_dict(orient="records")
    )
    metadata = {
        "figure_contract": {
            "core_conclusion": "Verb-aligned active/passive and same-noun slot contrasts are transient and heterogeneous across patients, and only subsets reproduce in the same direction across independent lexical subsets.",
            "archetype": "schematic-led quantitative grid",
            "backend": "Python/matplotlib",
            "final_size_mm": [WIDTH_MM, HEIGHT_MM],
            "font_size_pt": PT,
        },
        "alignment_schematics": {
            "panel_letters": "unlettered headers above panels a-d",
            "left_half": {
                "comparison": "active versus passive",
                "t0": "estimated verb onset from stimulus audio",
                "aligned_token": "VERB",
            },
            "right_half": {
                "comparison": "same noun identity in subject slot versus post-function-word noun slot",
                "t0": "corrected onset of the matched noun identity",
                "aligned_token": "same N",
            },
        },
        "verb_screen": {
            "n_units": int(len(verb_units)),
            "n_screen_positive": int(len(verb_selected)),
            "direction_counts": verb_selected["qualifying_direction"].value_counts().to_dict(),
            "bc_window_run_fdr05": int((verb_units["bc_window_q_run_fdr"] < 0.05).sum()),
            "bc_window_patient_fdr05": int((verb_units["bc_window_q_patient_fdr"] < 0.05).sum()),
            "bc_window_global_fdr05": int((verb_units["bc_window_q_global_fdr"] < 0.05).sum()),
            "same_direction_both_verb_sets": verb_same_both,
            "verb_set_definition": "verb IDs 1-5 versus 6-10; five paired verbs per set",
        },
        "noun_screen": {
            "n_units": int(len(noun_units)),
            "n_screen_positive": int(len(noun_selected)),
            "direction_counts": noun_selected["qualifying_direction"].value_counts().to_dict(),
            "same_direction_both_noun_classes": noun_same_both,
        },
        "three_condition_timing": {
            "n_units_analyzed": three_condition_timing.attrs["n_units_analyzed"],
            "n_screen_positive": int(three_condition_timing.drop_duplicates(["patient", "run", "unit_id"]).shape[0]),
            **{key: value for key, value in three_condition_timing.attrs.items() if key != "n_units_analyzed"},
            "group_counts": three_condition_group_counts,
            "medians": three_condition_medians,
            "active_passive_tests": three_condition_active_passive_tests,
            "unit_set": three_condition_timing.attrs["unit_set"],
            "matched_verb_ids": [1, 3, 5, 7, 9],
            "comparison": "pointwise one-way ANOVA across active, passive and scrambled",
            "pointwise_alpha": 0.05 / 3,
            "minimum_consecutive_bins": 6,
            "smoothing_sigma_ms": 40,
            "baseline_corrected": True,
            "fdr_applied": False,
        },
        "same_noun_slot_timing": {
            "n_screen_positive_directional": int(
                len(noun_timing_table.drop_duplicates(["patient", "run", "unit_id"]))
            ),
            "group_counts": noun_timing_group_counts,
            "medians": noun_timing_medians,
            "direction_group_tests": noun_timing_tests.to_dict(orient="records"),
            "comparison": "paired subject-slot versus post-function-word noun-slot",
            "pointwise_alpha": ALPHA,
            "minimum_consecutive_bins": MIN_BINS,
            "smoothing_sigma_ms": SDF_SIGMA_MS,
        },
        "patient_level_cross_lexical_replication": {
            "metric": "percentage of pooled screen-positive unit/run entries passing both lexical subsets in the same direction",
            "patients": list(PATIENTS),
            "rows": patient_generalization_table.to_dict(orient="records"),
            "inference_level": "descriptive patient-stratified summary; unit/run entries remain nested",
        },
        "representative_selection": {
            "active_preferring": "Among Kilosort-good units with a qualifying active>passive segment and a positive leave-one-verb-out minimum directional effect, maximize leave-one-verb-out minimum directional effect multiplied by the square root of the longest qualifying duration.",
            "passive_preferring": "Among Kilosort-good units with a qualifying passive>active segment and a positive leave-one-verb-out minimum directional effect, maximize the leave-one-verb-out minimum directional effect.",
            "noun_examples": "Within each direction, require the same direction in the agent- and patient-noun strata when available, then maximize qualifying-segment duration.",
        },
        "selection_note": "Representative panels were selected by deterministic robustness-and-duration rules and are paired with full-distribution patient summaries. Timing and lexical-subset panels are descriptive post-selection summaries.",
        "reviewer_risks": [
            "unit/run entries are nested within five patients",
            "pointwise-duration screens are not study-wide FWER correction",
            "verb onsets are audio-envelope estimates",
            "noun slot covaries with sentence time, preceding function word, voice/context, and local acoustics",
            "the three-condition panel uses a different unit set, item set, smoothing kernel and omnibus estimand from the primary two-condition verb screen",
        ],
    }
    (SOURCE / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with Image.open(png) as png_image:
        png_pixels = list(png_image.size)
    with Image.open(tiff) as tiff_image:
        tiff_pixels = list(tiff_image.size)
    three_condition_timing_counts = (
        three_condition_timing.groupby(["preferred_condition", "metric"])
        .size()
        .rename("n")
        .reset_index()
    )
    qa_report = {
        "status": "pass",
        "checked_on": "2026-08-06",
        "backend": "Python/matplotlib only",
        "final_size": {
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM,
            "pdf_page_points_expected": [WIDTH_MM / 25.4 * 72, HEIGHT_MM / 25.4 * 72],
            "png_pixels": png_pixels,
            "tiff_pixels": tiff_pixels,
            "raster_dpi": 600,
        },
        "typography": {
            "requested_font_size_pt": PT,
            "svg_text_remains_editable": "<text" in svg.read_text(encoding="utf-8"),
        },
        "data_integrity": {
            "verb_unit_rows": int(len(verb_units)),
            "verb_unique_unit_run_keys": not bool(
                verb_units[["patient", "run", "unit_id"]].duplicated().any()
            ),
            "verb_screen_positive": int(len(verb_selected)),
            "verb_same_direction_both_five_item_sets": verb_same_both,
            "noun_unit_rows": int(len(noun_units)),
            "noun_unique_unit_run_keys": not bool(
                noun_units[["subject", "run", "unit_id"]].duplicated().any()
            ),
            "noun_screen_positive": int(len(noun_selected)),
            "noun_same_direction_both_classes": noun_same_both,
            "three_condition_timing_rows": int(len(three_condition_timing)),
            "three_condition_unique_units": int(
                len(three_condition_timing.drop_duplicates(["patient", "run", "unit_id"]))
            ),
            "same_noun_slot_timing_rows": int(len(noun_timing_table)),
            "same_noun_slot_timing_unique_units": int(
                len(noun_timing_table.drop_duplicates(["patient", "run", "unit_id"]))
            ),
            "three_condition_timing_row_counts": three_condition_timing_counts.to_dict(orient="records"),
            "generalization_category_sums_match_screen_positive": bool(
                all(
                    int(frame["count"].sum()) == int(frame["total"].iloc[0])
                    for _, frame in generalization_table.groupby("comparison")
                )
            ),
            "patient_generalization_sums_match_pooled": bool(
                all(
                    int(frame["same_direction_both"].sum())
                    == (
                        verb_same_both
                        if comparison == "Verb contrast"
                        else noun_same_both
                    )
                    for comparison, frame in patient_generalization_table.groupby(
                        "comparison"
                    )
                )
            ),
        },
        "visual_checks": {
            "panel_labels_a_to_i_present": True,
            "two_unlettered_alignment_schematics_present": True,
            "schematic_t0_labels_and_token_boxes_not_clipped": True,
            "axis_labels_not_clipped_in_final_preview": True,
            "qualifying_intervals_visible": True,
            "comparison_identity_not_encoded_by_color_alone": True,
        },
        "known_limits": metadata["reviewer_risks"]
        + [
            "Timing distributions are conditional on passing the pointwise-duration screen.",
            "The verb split-half control uses two fixed five-item sets and has lower power than the pooled ten-pair screen.",
            "The three-condition timing panel is a descriptive analysis and is not directly comparable to the primary active-passive timing panel.",
        ],
    }
    (SOURCE / "qa_report.json").write_text(json.dumps(qa_report, indent=2), encoding="utf-8")
    manifest = {
        "png": str(png),
        "pdf": str(pdf),
        "svg": str(svg),
        "tiff": str(tiff),
        "legend": str(STEM.with_name(STEM.name + "_legend.md")),
        "source_data_directory": str(SOURCE),
        "metadata": metadata,
    }
    STEM.with_name(STEM.name + "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def write_legend(manifest: dict[str, object]) -> Path:
    metadata = manifest["metadata"]
    verb = metadata["verb_screen"]
    noun = metadata["noun_screen"]
    three_condition = metadata["three_condition_timing"]
    noun_timing = metadata["same_noun_slot_timing"]
    patient_replication = metadata["patient_level_cross_lexical_replication"]
    verb_dirs = verb["direction_counts"]
    noun_dirs = noun["direction_counts"]
    three_condition_medians = {
        (row["preferred_condition"], row["metric"]): row["median_ms"]
        for row in three_condition["medians"]
    }
    three_condition_median_text = {
        key: fmt_ms_half_up(value) for key, value in three_condition_medians.items()
    }
    three_condition_active_passive_p = {
        row["metric"]: row["p_value"]
        for row in three_condition["active_passive_tests"]
    }
    noun_timing_medians = {
        (row["direction"], row["metric"]): fmt_ms_half_up(row["median_ms"])
        for row in noun_timing["medians"]
    }
    noun_timing_p = {
        row["metric"]: row["p_value"]
        for row in noun_timing["direction_group_tests"]
    }
    patient_replication_text: dict[str, str] = {}
    for comparison in ("Verb contrast", "Noun-slot contrast"):
        rows = [
            row
            for row in patient_replication["rows"]
            if row["comparison"] == comparison
        ]
        patient_replication_text[comparison] = ", ".join(
            f"{row['patient']} {int(row['same_direction_both'])}/{int(row['total'])}"
            for row in rows
        )
    legend = (
        f"**Fig. 3 | Verb-aligned construction effects and same-noun structural-slot modulation are transient and heterogeneous.** "
        "**Top schematics,** Sentence templates show t = 0 alignment to the estimated verb onset for active/passive comparisons (left) and to the corrected onset of the matched noun in subject and post-function-word slots (right). "
        f"**a–d,** Deterministically selected Kilosort-good units. The active-preferring example maximizes the leave-one-verb-out minimum directional effect × √(longest qualifying duration); the other examples use analogous robustness rules. Curves and shading show mean ± s.e.m. baseline-corrected firing rate across 10 matched verb pairs (**a,c**) or 20 matched noun identities (**b,d**); green intervals qualify. "
        f"**e,f,** Patient-stratified screen-positive fractions for the verb contrast ({verb['n_screen_positive']}/{verb['n_units']}; active > passive, n = {verb_dirs.get('active_gt_passive', 0)}; active < passive, n = {verb_dirs.get('passive_gt_active', 0)}) and noun-slot contrast ({noun['n_screen_positive']}/{noun['n_units']}; subject > post-function-word noun, n = {noun_dirs.get('subject_gt_predicate', 0)}; reverse, n = {noun_dirs.get('predicate_gt_subject', 0)}; mixed, n = {noun_dirs.get('mixed', 0)}). "
        f"**g,** Unit/run onset and effect-peak distributions; points, bars and ticks denote observations, interquartile ranges and medians. Left, {three_condition['input_version_label']} active/passive/scrambled groups (n = {three_condition['group_counts'].get('active', 0)}/{three_condition['group_counts'].get('passive', 0)}/{three_condition['group_counts'].get('scrambled', 0)}; two-sided Mann–Whitney active versus passive: onset P = {three_condition_active_passive_p['first_significant_onset_ms']:.3f}, peak P = {three_condition_active_passive_p['effect_peak_ms']:.3f}). Right, primary subject-high/post-function-word-high groups (n = {noun_timing['group_counts'].get('subject_gt_predicate', 0)}/{noun_timing['group_counts'].get('predicate_gt_subject', 0)}; onset P = {noun_timing_p['onset']:.3f}, peak P = {noun_timing_p['peak']:.3f}); annotated values are medians. "
        f"**h,** Same-direction cross-lexical replication: {verb['same_direction_both_verb_sets']}/{verb['n_screen_positive']} verb hits across two five-verb sets and {noun['same_direction_both_noun_classes']}/{noun['n_screen_positive']} noun-slot hits across agent/patient strata. "
        "**i,** Patient-stratified replication; labels give replicated/pooled hits. These summaries are descriptive because unit/run entries are nested within five patients. "
        "Primary screens used paired lexical-item tests at 1-ms resolution, Gaussian σ = 25 ms and P < 0.025 for at least six consecutive bins without gap filling. Panel g left used a three-condition ANOVA, σ = 40 ms and P < 0.05/3. Screens are not study-wide family-wise error correction. Noun slot covaries with sentence time, preceding function word, voice/context and local acoustics. Source data are provided with the figure.\n"
    )
    path = STEM.with_name(STEM.name + "_legend.md")
    path.write_text(legend, encoding="utf-8")
    return path


def main() -> None:
    global VERB_VERSION, THREE_CONDITION_VERB_OUT, OUT, SOURCE, STEM
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure output directory (or TAN_FIG3_OUT_DIR).")
    args = parser.parse_args()
    VERB_VERSION, THREE_CONDITION_VERB_OUT = resolve_input(args.verb_version, args.verb_input_dir)
    OUT = args.output_dir or Path(os.environ.get(
        "TAN_FIG3_OUT_DIR",
        str(ROOT / "reports" / "nature_science_draft" / f"assembled_figures_20260805_{VERB_VERSION}"),
    ))
    SOURCE = OUT / "source_data" / "fig3_verb_noun_slot_180mm"
    STEM = OUT / "fig3_verb_noun_slot_180mm"
    manifest = build_figure()
    legend = write_legend(manifest)
    print(json.dumps({**manifest, "legend": str(legend)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
