#!/usr/bin/env python3
"""Summarize population-level differences for three-condition verb responses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

from analyze_verb_three_condition_unit_activity import (  # noqa: E402
    BASELINE_SEC,
    CONDITIONS,
    MATCHED_THREE_CONDITION_VERB_IDS,
    POST_MS,
    PRE_MS,
    PRIMARY_SIGNIFICANCE_COL,
    RESPONSE_SEC,
    SMOOTH_SIGMA_MS,
    gaussian_smooth_ms,
    run_npz_path,
)


from verb_three_condition_versioning import input_metadata, parse_downstream_args, resolve_input

VERB_VERSION, OUT_DIR = resolve_input(validate=False)

def percentile(values: pd.Series | np.ndarray, q: float) -> float:
    vals = pd.Series(values).dropna().astype(float)
    return float(vals.quantile(q)) if not vals.empty else float("nan")


def median(values: pd.Series | np.ndarray) -> float:
    vals = pd.Series(values).dropna().astype(float)
    return float(vals.median()) if not vals.empty else float("nan")


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    finite = np.isfinite(values)
    n = finite.sum(axis=axis)
    std = np.nanstd(values, axis=axis)
    out = np.full(std.shape, np.nan, dtype=float)
    np.divide(std, np.sqrt(np.maximum(n, 1)), out=out, where=n > 0)
    return out


def extract_matched_unit_means(
    patient: str,
    run: str,
    run_events: pd.DataFrame,
    unit_ids_needed: set[int],
    unit_set: str,
) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
    time_sec = np.arange(-PRE_MS, POST_MS + 1) / 1000
    baseline_mask = (time_sec >= BASELINE_SEC[0]) & (time_sec < BASELINE_SEC[1])
    z = np.load(run_npz_path(patient, run, unit_set), allow_pickle=True)
    firing_rate = z["firing_rate_hz"]
    unit_ids = z["unit_ids"].astype(int)
    unit_index = {int(unit_id): i for i, unit_id in enumerate(unit_ids)}
    present_units = [unit_id for unit_id in unit_ids_needed if unit_id in unit_index]

    traces_by_condition: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    matched = run_events[run_events["verb_id"].isin(MATCHED_THREE_CONDITION_VERB_IDS)].copy()
    matched = matched.sort_values(["sentence_condition", "verb_id", "trial_index"])
    for event in matched.itertuples(index=False):
        start = int(event.event_bin_1000Hz) - PRE_MS
        stop = int(event.event_bin_1000Hz) + POST_MS + 1
        if start < 0 or stop > firing_rate.shape[1]:
            continue
        epoch = firing_rate[:, start:stop].astype(float, copy=False)
        smooth = gaussian_smooth_ms(epoch, SMOOTH_SIGMA_MS)
        baseline = smooth[:, baseline_mask].mean(axis=1)
        traces_by_condition[str(event.sentence_condition)].append(smooth - baseline[:, None])

    means_by_unit: dict[int, dict[str, np.ndarray]] = {}
    for unit_id in present_units:
        idx = unit_index[unit_id]
        means: dict[str, np.ndarray] = {}
        for condition in CONDITIONS:
            traces = traces_by_condition[condition]
            if not traces:
                continue
            means[condition] = np.nanmean(np.stack([trace[idx] for trace in traces], axis=0), axis=0)
        if all(condition in means for condition in CONDITIONS):
            means_by_unit[unit_id] = means
    return time_sec, means_by_unit


def compute_pairwise_direction_counts(sig: pd.DataFrame) -> list[dict[str, object]]:
    pairs = [("active", "passive"), ("active", "scrambled"), ("passive", "scrambled")]
    rows: list[dict[str, object]] = []
    for first, second in pairs:
        diff = sig[f"qualifying_{first}_mean_hz"] - sig[f"qualifying_{second}_mean_hz"]
        rows.append(
            {
                "pair": f"{first} vs {second}",
                "first_condition": first,
                "second_condition": second,
                "first_higher_count": int((diff > 0).sum()),
                "second_higher_count": int((diff < 0).sum()),
                "median_diff_hz": median(diff),
                "q25_diff_hz": percentile(diff, 0.25),
                "q75_diff_hz": percentile(diff, 0.75),
            }
        )
    return rows


def compute_timing_tests(timing: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pairs = [("active", "passive"), ("active", "scrambled"), ("passive", "scrambled")]
    for metric in ["first_significant_onset_ms", "preferred_peak_ms", "effect_peak_ms"]:
        groups = [timing.loc[timing["preferred_condition"] == c, metric].dropna().to_numpy(float) for c in CONDITIONS]
        if all(len(group) > 0 for group in groups) and len(np.unique(np.concatenate(groups))) > 1:
            kw_p = float(stats.kruskal(*groups).pvalue)
        else:
            kw_p = float("nan")
        rows.append({"metric": metric, "test": "kruskal", "comparison": "three_preference_groups", "p_value": kw_p})
        for first, second in pairs:
            a = timing.loc[timing["preferred_condition"] == first, metric].dropna().to_numpy(float)
            b = timing.loc[timing["preferred_condition"] == second, metric].dropna().to_numpy(float)
            p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) and len(b) else float("nan")
            rows.append(
                {
                    "metric": metric,
                    "test": "mannwhitneyu",
                    "comparison": f"{first}_vs_{second}",
                    "p_value": p_value,
                    "median_diff_ms": median(a) - median(b),
                }
            )
    return rows


def summarize(unit_set: str = "usable_lenient") -> dict[str, object]:
    results = pd.read_csv(OUT_DIR / "verb_three_condition_unit_results.csv")
    events = pd.read_csv(OUT_DIR / "verb_three_condition_event_metadata.csv")
    timing = pd.read_csv(OUT_DIR / "verb_three_condition_population_timing_by_unit.csv")
    sig = results[results[PRIMARY_SIGNIFICANCE_COL].astype(bool)].copy()
    sig = sig[sig["timepoint_preferred_condition"].isin(CONDITIONS)].copy()

    pref_minus_second = []
    for row in sig.itertuples(index=False):
        values = sorted([float(getattr(row, f"qualifying_{c}_mean_hz")) for c in CONDITIONS], reverse=True)
        pref_minus_second.append(values[0] - values[1])
    sig["pref_minus_second_qualifying_hz"] = pref_minus_second

    time_sec: np.ndarray | None = None
    response_mask: np.ndarray | None = None
    unit_rows: list[dict[str, object]] = []
    raw_diff_by_pref: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    norm_diff_by_pref: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}

    for (patient, run), group in sig.groupby(["patient", "run"], sort=False):
        run_events = events[(events["patient"] == patient) & (events["run"] == run)]
        unit_ids_needed = {int(v) for v in group["unit_id"]}
        unit_time, means_by_unit = extract_matched_unit_means(str(patient), str(run), run_events, unit_ids_needed, unit_set)
        if time_sec is None:
            time_sec = unit_time
            response_mask = (time_sec >= RESPONSE_SEC[0]) & (time_sec < RESPONSE_SEC[1])
        assert response_mask is not None

        for _, row in group.iterrows():
            unit_id = int(row["unit_id"])
            means = means_by_unit.get(unit_id)
            if means is None:
                continue
            preferred = str(row["timepoint_preferred_condition"])
            others = [condition for condition in CONDITIONS if condition != preferred]
            preferred_trace = means[preferred]
            other_trace = np.nanmean(np.stack([means[condition] for condition in others], axis=0), axis=0)
            diff_trace = preferred_trace - other_trace
            response_diff = diff_trace[response_mask]
            peak_i = int(np.nanargmax(response_diff))
            peak_ms = int(round(time_sec[response_mask][peak_i] * 1000))
            peak_hz = float(response_diff[peak_i])
            denom = float(np.nanmax(np.abs(response_diff)))
            norm_trace = diff_trace / denom if np.isfinite(denom) and denom > 0 else diff_trace * np.nan
            raw_diff_by_pref[preferred].append(diff_trace)
            norm_diff_by_pref[preferred].append(norm_trace)
            unit_rows.append(
                {
                    "patient": str(row["patient"]),
                    "run": str(row["run"]),
                    "unit_id": unit_id,
                    "preferred_condition": preferred,
                    "first_segment_onset_ms": float(row["first_segment_onset_ms"]),
                    "longest_segment_ms": float(row["longest_segment_ms"]),
                    "pref_minus_second_qualifying_hz": float(row["pref_minus_second_qualifying_hz"]),
                    "qualifying_max_minus_min_hz": float(row["qualifying_max_minus_min_hz"]),
                    "matched_pref_minus_other_peak_ms": peak_ms,
                    "matched_pref_minus_other_peak_hz": peak_hz,
                    "min_pointwise_p": float(row["min_pointwise_p"]),
                }
            )

    if time_sec is None or response_mask is None:
        raise RuntimeError("No significant units available for population difference summary")

    diff_units = pd.DataFrame(unit_rows)
    downsample_ms = np.arange(-200, 801, 50)
    downsample_idx = [int(np.argmin(np.abs(time_sec * 1000 - t))) for t in downsample_ms]
    timecourse: dict[str, dict[str, list[float]]] = {}
    for condition in CONDITIONS:
        raw = np.stack(raw_diff_by_pref[condition], axis=0) if raw_diff_by_pref[condition] else np.empty((0, len(time_sec)))
        norm = np.stack(norm_diff_by_pref[condition], axis=0) if norm_diff_by_pref[condition] else np.empty((0, len(time_sec)))
        source = norm
        if source.size:
            mean_values = np.nanmean(source[:, downsample_idx], axis=0)
            sem_values = sem(source[:, downsample_idx], axis=0)
        else:
            mean_values = np.full(len(downsample_idx), np.nan)
            sem_values = np.full(len(downsample_idx), np.nan)
        timecourse[condition] = {
            "n_units": int(source.shape[0]),
            "mean_normalized_pref_minus_other": [float(v) for v in mean_values],
            "sem_normalized_pref_minus_other": [float(v) for v in sem_values],
        }

    timecourse_rows = []
    for i, t in enumerate(downsample_ms):
        row = {"time_ms": int(t)}
        for condition in CONDITIONS:
            row[f"{condition}_mean_normalized_pref_minus_other"] = timecourse[condition]["mean_normalized_pref_minus_other"][i]
            row[f"{condition}_sem_normalized_pref_minus_other"] = timecourse[condition]["sem_normalized_pref_minus_other"][i]
        timecourse_rows.append(row)

    by_preference: list[dict[str, object]] = []
    for condition in CONDITIONS:
        group = diff_units[diff_units["preferred_condition"] == condition]
        timing_group = timing[timing["preferred_condition"] == condition]
        by_preference.append(
            {
                "preferred_condition": condition,
                "n_units": int(len(group)),
                "median_pref_minus_second_qualifying_hz": median(group["pref_minus_second_qualifying_hz"]),
                "q25_pref_minus_second_qualifying_hz": percentile(group["pref_minus_second_qualifying_hz"], 0.25),
                "q75_pref_minus_second_qualifying_hz": percentile(group["pref_minus_second_qualifying_hz"], 0.75),
                "median_pref_minus_other_peak_hz": median(group["matched_pref_minus_other_peak_hz"]),
                "q25_pref_minus_other_peak_hz": percentile(group["matched_pref_minus_other_peak_hz"], 0.25),
                "q75_pref_minus_other_peak_hz": percentile(group["matched_pref_minus_other_peak_hz"], 0.75),
                "median_peak_contrast_time_ms": median(group["matched_pref_minus_other_peak_ms"]),
                "median_first_significant_onset_ms": median(timing_group["first_significant_onset_ms"]),
                "median_preferred_peak_ms": median(timing_group["preferred_peak_ms"]),
            }
        )

    payload: dict[str, object] = {
        **input_metadata(VERB_VERSION, OUT_DIR),
        "unit_set": unit_set,
        "n_significant_units": int(len(diff_units)),
        "matched_verb_ids": list(MATCHED_THREE_CONDITION_VERB_IDS),
        "screen": {
            "primary_rule": "p < 0.05 / 3 and contiguous significant time > 5 ms",
            "fdr": "not applied",
        },
        "by_preference": by_preference,
        "pairwise_direction_counts": compute_pairwise_direction_counts(sig),
        "timing_tests": compute_timing_tests(timing),
        "timecourse": {
            "time_ms": [int(v) for v in downsample_ms],
            "series": timecourse,
        },
    }

    diff_units.to_csv(OUT_DIR / "verb_three_condition_population_differences_by_unit.csv", index=False)
    pd.DataFrame(timecourse_rows).to_csv(OUT_DIR / "verb_three_condition_population_difference_timecourse.csv", index=False)
    pd.DataFrame(payload["pairwise_direction_counts"]).to_csv(
        OUT_DIR / "verb_three_condition_population_pairwise_direction_counts.csv",
        index=False,
    )
    pd.DataFrame(by_preference).to_csv(OUT_DIR / "verb_three_condition_population_difference_summary.csv", index=False)
    pd.DataFrame(payload["timing_tests"]).to_csv(OUT_DIR / "verb_three_condition_population_timing_tests.csv", index=False)
    json_path = OUT_DIR / "verb_three_condition_population_differences.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    return payload


def main() -> None:
    global VERB_VERSION, OUT_DIR
    args = parse_downstream_args(__doc__, unit_set=True)
    VERB_VERSION, OUT_DIR = args.verb_version, args.verb_input_dir
    payload = summarize(args.unit_set)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
