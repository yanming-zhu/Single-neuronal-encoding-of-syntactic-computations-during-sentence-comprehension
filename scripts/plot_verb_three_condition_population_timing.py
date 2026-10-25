#!/usr/bin/env python3
"""Population timing summaries for active/passive/scrambled verb responses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analyze_verb_three_condition_unit_activity import (  # noqa: E402
    CONDITIONS,
    CONDITION_COLORS,
    PRIMARY_SIGNIFICANCE_COL,
    RESPONSE_SEC,
    SMOOTH_SIGMA_MS,
    extract_unit_condition_traces,
    gaussian_smooth_ms,
)


from verb_three_condition_versioning import input_metadata, parse_downstream_args, resolve_input

VERB_VERSION, OUT_DIR = resolve_input(validate=False)

def mean_trace(traces: dict[str, np.ndarray], condition: str) -> np.ndarray | None:
    arr = traces.get(condition)
    if arr is None or arr.size == 0:
        return None
    return np.nanmean(gaussian_smooth_ms(arr, SMOOTH_SIGMA_MS), axis=0)


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    finite = np.isfinite(values)
    n = finite.sum(axis=axis)
    std = np.nanstd(values, axis=axis)
    out = np.full(std.shape, np.nan, dtype=float)
    np.divide(std, np.sqrt(np.maximum(n, 1)), out=out, where=n > 0)
    return out


def percentile_summary(values: pd.Series) -> dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {"median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "median": float(vals.median()),
        "q25": float(vals.quantile(0.25)),
        "q75": float(vals.quantile(0.75)),
    }


def build_population_timing(unit_set: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    results = pd.read_csv(OUT_DIR / "verb_three_condition_unit_results.csv")
    segments = pd.read_csv(OUT_DIR / "verb_three_condition_timepoint_segments.csv")
    events = pd.read_csv(OUT_DIR / "verb_three_condition_event_metadata.csv")
    sig = results[results[PRIMARY_SIGNIFICANCE_COL].astype(bool)].copy()
    sig = sig[sig["timepoint_preferred_condition"].isin(CONDITIONS)].copy()

    rows: list[dict[str, object]] = []
    normalized_pref_traces: dict[str, list[np.ndarray]] = {"all": []}
    for condition in CONDITIONS:
        normalized_pref_traces[condition] = []
    time_sec: np.ndarray | None = None

    key_cols = ["patient", "run", "unit_id"]
    segment_lookup = segments.sort_values(["duration_ms", "start_ms"], ascending=[False, True]).groupby(key_cols)
    for row in sig.itertuples(index=False):
        patient = str(row.patient)
        run = str(row.run)
        unit_id = int(row.unit_id)
        preferred = str(row.timepoint_preferred_condition)
        try:
            unit_time, traces = extract_unit_condition_traces(patient, run, unit_id, events, unit_set)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        if time_sec is None:
            time_sec = unit_time
        response_mask = (unit_time >= RESPONSE_SEC[0]) & (unit_time < RESPONSE_SEC[1])
        response_time_ms = np.rint(unit_time[response_mask] * 1000).astype(int)
        means = {condition: mean_trace(traces, condition) for condition in CONDITIONS}
        if any(trace is None for trace in means.values()):
            continue

        preferred_trace = means[preferred]
        if preferred_trace is None:
            continue
        response_pref = preferred_trace[response_mask]
        peak_i = int(np.nanargmax(response_pref))
        preferred_peak_ms = int(response_time_ms[peak_i])
        preferred_peak_hz = float(response_pref[peak_i])

        stacked = np.stack([means[condition][response_mask] for condition in CONDITIONS], axis=0)
        effect_trace = np.nanmax(stacked, axis=0) - np.nanmin(stacked, axis=0)
        effect_peak_i = int(np.nanargmax(effect_trace))
        effect_peak_ms = int(response_time_ms[effect_peak_i])
        effect_peak_hz = float(effect_trace[effect_peak_i])

        unit_segments = segment_lookup.get_group((patient, run, unit_id))
        first_segment = unit_segments.sort_values("start_ms").iloc[0]
        longest_segment = unit_segments.iloc[0]

        norm_trace = preferred_trace.copy()
        denom = float(np.nanmax(np.abs(norm_trace[response_mask])))
        if np.isfinite(denom) and denom > 0:
            norm_trace = norm_trace / denom
            normalized_pref_traces["all"].append(norm_trace)
            normalized_pref_traces[preferred].append(norm_trace)

        rows.append(
            {
                "patient": patient,
                "run": run,
                "unit_id": unit_id,
                "ks_label": str(row.ks_label),
                "preferred_condition": preferred,
                "first_significant_onset_ms": int(first_segment.start_ms),
                "first_significant_stop_ms_exclusive": int(first_segment.stop_ms_exclusive),
                "longest_segment_onset_ms": int(longest_segment.start_ms),
                "longest_segment_stop_ms_exclusive": int(longest_segment.stop_ms_exclusive),
                "longest_segment_duration_ms": int(longest_segment.duration_ms),
                "min_pointwise_p": float(row.min_pointwise_p),
                "preferred_peak_ms": preferred_peak_ms,
                "preferred_peak_hz": preferred_peak_hz,
                "effect_peak_ms": effect_peak_ms,
                "effect_peak_max_minus_min_hz": effect_peak_hz,
            }
        )

    if time_sec is None:
        time_sec = np.arange(-500, 1201) / 1000
    trace_arrays = {
        condition: np.stack(values, axis=0) if values else np.empty((0, len(time_sec)))
        for condition, values in normalized_pref_traces.items()
    }
    return pd.DataFrame(rows), trace_arrays, time_sec


def summarize_population_timing(timing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("all", timing)]
    groups.extend((condition, timing[timing["preferred_condition"] == condition]) for condition in CONDITIONS)
    for label, group in groups:
        onset = percentile_summary(group["first_significant_onset_ms"])
        peak = percentile_summary(group["preferred_peak_ms"])
        effect_peak = percentile_summary(group["effect_peak_ms"])
        rows.append(
            {
                "preferred_condition": label,
                "n_units": int(len(group)),
                "onset_median_ms": onset["median"],
                "onset_q25_ms": onset["q25"],
                "onset_q75_ms": onset["q75"],
                "preferred_peak_median_ms": peak["median"],
                "preferred_peak_q25_ms": peak["q25"],
                "preferred_peak_q75_ms": peak["q75"],
                "effect_peak_median_ms": effect_peak["median"],
                "effect_peak_q25_ms": effect_peak["q25"],
                "effect_peak_q75_ms": effect_peak["q75"],
            }
        )
    return pd.DataFrame(rows)


def jitter_positions(n: int, center: float, width: float = 0.18) -> np.ndarray:
    if n <= 1:
        return np.array([center] * n, dtype=float)
    return center + np.linspace(-width, width, n)


def add_box_with_points(ax: plt.Axes, timing: pd.DataFrame, column: str, ylabel: str, title: str) -> None:
    labels = list(CONDITIONS)
    data = [timing.loc[timing["preferred_condition"] == condition, column].dropna().to_numpy(float) for condition in labels]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, condition in zip(bp["boxes"], labels):
        patch.set_facecolor(CONDITION_COLORS[condition])
        patch.set_alpha(0.18)
        patch.set_edgecolor(CONDITION_COLORS[condition])
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    for i, (condition, values) in enumerate(zip(labels, data), start=1):
        ax.scatter(
            jitter_positions(len(values), i),
            values,
            s=22,
            color=CONDITION_COLORS[condition],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.82,
            zorder=3,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.grid(axis="y", color="0.9", lw=0.7)


def plot_population_timing(timing: pd.DataFrame, trace_arrays: dict[str, np.ndarray], time_sec: np.ndarray) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_verb_three_condition_population_timing.png"
    pdf_path = fig_dir / "fig_verb_three_condition_population_timing.pdf"

    fig = plt.figure(figsize=(15.8, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_onset = fig.add_subplot(gs[1, 0])
    ax_peak = fig.add_subplot(gs[1, 1])

    response_mask = (time_sec >= RESPONSE_SEC[0]) & (time_sec < RESPONSE_SEC[1])
    for label, color, lw, alpha in [
        ("all", "#222222", 2.2, 0.14),
        ("active", CONDITION_COLORS["active"], 1.6, 0.10),
        ("passive", CONDITION_COLORS["passive"], 1.6, 0.10),
        ("scrambled", CONDITION_COLORS["scrambled"], 1.6, 0.10),
    ]:
        arr = trace_arrays[label]
        if arr.size == 0:
            continue
        mean = np.nanmean(arr, axis=0)
        se = sem(arr, axis=0)
        ax_trace.plot(time_sec, mean, color=color, lw=lw, label=f"{label} n={arr.shape[0]}")
        ax_trace.fill_between(time_sec, mean - se, mean + se, color=color, alpha=alpha, lw=0)
    median_onset = float(timing["first_significant_onset_ms"].median()) / 1000
    median_peak = float(timing["preferred_peak_ms"].median()) / 1000
    ax_trace.axvline(0, color="0.35", lw=0.8)
    ax_trace.axvline(median_onset, color="#7B3294", lw=1.4, linestyle="--", label=f"median onset {median_onset*1000:.0f} ms")
    ax_trace.axvline(median_peak, color="#008B8B", lw=1.4, linestyle=":", label=f"median peak {median_peak*1000:.0f} ms")
    ax_trace.axhline(0, color="0.75", lw=0.8)
    ax_trace.set_xlim(-0.25, 0.85)
    ax_trace.set_xlabel("time from estimated verb onset (s)")
    ax_trace.set_ylabel("normalized preferred response")
    ax_trace.set_title("Population preferred-condition response", loc="left", fontsize=12, weight="bold")
    ax_trace.legend(frameon=False, loc="upper right", fontsize=8)
    ax_trace.grid(color="0.92", lw=0.7)

    for condition in CONDITIONS:
        sub = timing[timing["preferred_condition"] == condition]
        ax_scatter.scatter(
            sub["first_significant_onset_ms"],
            sub["preferred_peak_ms"],
            s=38,
            color=CONDITION_COLORS[condition],
            edgecolors="white",
            linewidths=0.45,
            alpha=0.86,
            label=f"{condition} n={len(sub)}",
        )
    ax_scatter.plot([0, 800], [0, 800], color="0.7", lw=0.9, linestyle="--")
    ax_scatter.set_xlim(-10, 820)
    ax_scatter.set_ylim(-10, 820)
    ax_scatter.set_xlabel("first significant onset (ms)")
    ax_scatter.set_ylabel("preferred response peak (ms)")
    ax_scatter.set_title("Onset-to-peak timing by unit", loc="left", fontsize=12, weight="bold")
    ax_scatter.legend(frameon=False, loc="upper left", fontsize=8)
    ax_scatter.grid(color="0.92", lw=0.7)

    add_box_with_points(ax_onset, timing, "first_significant_onset_ms", "ms after verb onset", "First significant onset")
    add_box_with_points(ax_peak, timing, "preferred_peak_ms", "ms after verb onset", "Preferred-condition peak time")

    fig.suptitle("Population timing of three-condition verb responses", y=0.995, fontsize=15, weight="bold")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> None:
    global VERB_VERSION, OUT_DIR
    args = parse_downstream_args(__doc__, unit_set=True)
    VERB_VERSION, OUT_DIR = args.verb_version, args.verb_input_dir
    timing, trace_arrays, time_sec = build_population_timing(args.unit_set)
    summary = summarize_population_timing(timing)
    timing_path = OUT_DIR / "verb_three_condition_population_timing_by_unit.csv"
    summary_path = OUT_DIR / "verb_three_condition_population_timing_summary.csv"
    timing.to_csv(timing_path, index=False)
    summary.to_csv(summary_path, index=False)
    paths = plot_population_timing(timing, trace_arrays, time_sec)
    payload = {
        **input_metadata(VERB_VERSION, OUT_DIR),
        "timing_csv": str(timing_path),
        "summary_csv": str(summary_path),
        **paths,
        "summary": summary.to_dict(orient="records"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
