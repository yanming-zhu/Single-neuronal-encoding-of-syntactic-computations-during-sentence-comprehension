#!/usr/bin/env python3
"""Compare active, passive, and scrambled ActPas verb responses."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.font_manager import FontProperties, fontManager  # noqa: E402

from analyze_noun_role_structure_encoding import (  # noqa: E402
    DATA_ROOT,
    PATIENTS,
    UNIT_SET_FILE,
    audio_envelope,
    audio_root,
    decode_mp3,
    onset_from_peak,
    run_npz_path,
    shortened_run_name,
)
from analyze_verb_active_passive_unit_activity import (  # noqa: E402
    BASELINE_SEC,
    POST_MS,
    PRE_MS,
    RESPONSE_SEC,
    SMOOTH_SIGMA_MS,
    STIMULI as ACTIVE_PASSIVE_STIMULI,
    VERB_LABELS,
    VERB_ORDER,
    build_verb_audio_onsets,
    gaussian_smooth_ms,
    safe_wilcoxon,
    setup_matplotlib,
)


OUT_DIR = Path(
    os.environ.get(
        "TAN_VERB_THREE_CONDITION_OUT_DIR",
        str(DATA_ROOT / "combined_verb_three_condition_unit_activity_corrected_20260812"),
    )
)
REPORT_PATH = Path(
    os.environ.get(
        "TAN_VERB_THREE_CONDITION_REPORT_PATH",
        str(ROOT / "reports/verb_three_condition_unit_activity_corrected_20260812.md"),
    )
)
ACTIVE_PASSIVE_OUT_DIR = DATA_ROOT / "combined_verb_active_passive_unit_activity"

CONDITIONS = ("active", "passive", "scrambled")
N_CONDITIONS = len(CONDITIONS)
MATCHED_THREE_CONDITION_VERB_IDS = (1, 3, 5, 7, 9)
POINTWISE_ALPHA = 0.05 / N_CONDITIONS
# Data are sampled at 1 kHz; >5 ms therefore requires at least 6 consecutive bins.
MIN_CONSECUTIVE_SIGNIFICANT_BINS = 6
PRIMARY_SIGNIFICANCE_COL = "significant_time_gt5ms_p_condition"
CONDITION_COLORS = {
    "active": "#2F6FBB",
    "passive": "#C44E35",
    "scrambled": "#7A7A7A",
}
CONDITION_LABELS = {
    "active": "active",
    "passive": "passive",
    "scrambled": "scrambled",
}

SUMMARY_EXAMPLE_UNITS = [
    ("gz004", "gz004t1_g0_actpas_run0", 93),
    ("gz002", "gz002_g0_actpas_run2", 283),
]

SCRAMBLED_SENTENCES_BY_SOUND = {
    "stim_actpas/non001.mp3": "叶虫光了把啃树子",
    "stim_actpas/non003.mp3": "了走骨叼头将小狗",
    "stim_actpas/non005.mp3": "了踢飞球给足员球",
    "stim_actpas/non007.mp3": "爸爸竹爆着将了点",
    "stim_actpas/non009.mp3": "开切奶奶瓜西将了",
}

# The recorded ActPas block contains five scrambled ActPas stimuli per run.
# Each target explicitly identifies the occurrence of the first character of
# the canonical matched verb in the scrambled string. This prevents an audio
# envelope peak from being assigned to a neighbouring noun or function word.
SCRAMBLED_VERB_TARGETS = {
    "stim_actpas/non001.mp3": {"char_index": 5, "peak_sec": 1.315, "left_limit_sec": 1.075},  # 啃
    "stim_actpas/non003.mp3": {"char_index": 3, "peak_sec": 0.850, "left_limit_sec": 0.690},  # 叼
    "stim_actpas/non005.mp3": {"char_index": 1, "peak_sec": 0.515, "left_limit_sec": 0.280},  # 踢
    "stim_actpas/non007.mp3": {"char_index": 7, "peak_sec": 1.720, "left_limit_sec": 1.555},  # 点
    "stim_actpas/non009.mp3": {"char_index": 1, "peak_sec": 0.565, "left_limit_sec": 0.335},  # 切
}

SCRAMBLED_VERB_BY_SOUND = {
    sound_file: {
        "sound_file": sound_file,
        "condition": "scrambled",
        "sentence": SCRAMBLED_SENTENCES_BY_SOUND[sound_file],
        "function_word": "",
        "verb_id": int(sound_file[-7:-4]),
        "verb": VERB_ORDER[int(sound_file[-7:-4]) - 1],
    }
    for sound_file in SCRAMBLED_VERB_TARGETS
}

STIM_BY_SOUND = {
    stim.sound_file: {
        "sound_file": stim.sound_file,
        "condition": stim.condition,
        "sentence": stim.sentence,
        "function_word": stim.function_word,
        "verb_id": stim.verb_id,
        "verb": stim.verb,
    }
    for stim in ACTIVE_PASSIVE_STIMULI
}
STIM_BY_SOUND.update(SCRAMBLED_VERB_BY_SOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-set",
        choices=sorted(UNIT_SET_FILE),
        default="usable_lenient",
        help="Cluster selection to analyze.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[verb-three-condition] {message}", flush=True)


def trial_bins_path(patient: str, run: str) -> Path:
    return DATA_ROOT / patient / "processed/ActPasSynSentence" / run / "firing_rate_1000Hz/trial_bins_1000Hz.csv"


def read_trial_bins(patient: str, run: str) -> pd.DataFrame:
    path = trial_bins_path(patient, run)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def discover_runs() -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    for patient in PATIENTS:
        task_dir = DATA_ROOT / patient / "processed/ActPasSynSentence"
        if not task_dir.exists():
            continue
        for run_dir in sorted(task_dir.iterdir()):
            if (run_dir / "firing_rate_1000Hz/trial_bins_1000Hz.csv").exists():
                runs.append((patient, run_dir.name))
    return runs


def local_peak_times(times: np.ndarray, env: np.ndarray) -> list[float]:
    peaks, _ = find_peaks(
        env,
        distance=max(1, int(0.08 / 0.005)),
        prominence=max(float(env.max()) * 0.06, 0.004),
    )
    return [float(times[i]) for i in peaks if times[i] < 2.3]


def speech_start_sec(times: np.ndarray, env: np.ndarray) -> float:
    active_thr = max(0.01, float(np.percentile(env, 95) * 0.08))
    active = np.flatnonzero(env > active_thr)
    if active.size == 0:
        raise RuntimeError("No active audio detected")
    return float(times[max(0, int(active[0]) - 2)])


def build_scrambled_audio_onsets(font_prop: FontProperties) -> pd.DataFrame:
    base_audio_dir = audio_root()
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(len(SCRAMBLED_VERB_TARGETS), 1, figsize=(11.5, 9.5), sharex=True)
    if len(SCRAMBLED_VERB_TARGETS) == 1:
        axes = np.array([axes])

    for ax, (sound_file, target) in zip(axes.flat, SCRAMBLED_VERB_TARGETS.items()):
        stim = STIM_BY_SOUND[sound_file]
        expected_char = str(stim["verb"])[0]
        target_char = str(stim["sentence"])[int(target["char_index"])]
        if target_char != expected_char:
            raise RuntimeError(
                f"Scrambled verb anchor mismatch for {sound_file}: "
                f"expected {expected_char}, found {target_char}"
            )
        verb_peak = float(target["peak_sec"])
        audio = decode_mp3(base_audio_dir / sound_file)
        times, env = audio_envelope(audio)
        peaks = local_peak_times(times, env)
        speech_start = speech_start_sec(times, env)
        left_limit = float(target["left_limit_sec"])
        verb_onset = onset_from_peak(times, env, verb_peak, left_limit_sec=left_limit)

        rows.append(
            {
                **stim,
                "speech_start_sec_in_audio": speech_start,
                "function_onset_sec_in_audio": np.nan,
                "function_peak_sec_in_audio": np.nan,
                "verb_onset_sec_in_audio": float(verb_onset),
                "verb_peak_sec_in_audio": float(verb_peak),
                "verb_onset_sec_from_sentence_onset": float(max(0.0, verb_onset - speech_start)),
                "onset_source": "manual_first_canonical_verb_character_peak_from_qc",
                "verb_char_index_in_scrambled": int(target["char_index"]),
                "verb_anchor_character": target_char,
                "detected_peak_times_sec_in_audio": json.dumps(peaks, ensure_ascii=False),
            }
        )

        ax.plot(times, env, color="#333333", linewidth=1.0)
        ax.axvline(speech_start, color="#777777", linewidth=1.0, label="speech start")
        ax.axvline(verb_onset, color="#7B3294", linewidth=1.4, label="verb onset")
        ax.axvline(verb_peak, color="#7B3294", linewidth=0.9, linestyle="--", label="verb peak")
        ax.set_xlim(0, min(float(times[-1]), 2.25))
        ax.set_ylabel("RMS")
        ax.set_title(f"{Path(sound_file).stem} -> verb {int(stim['verb_id']):02d} {stim['verb']}", fontproperties=font_prop)
    axes.flat[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes.flat[-1].set_xlabel("Audio time (s)")
    fig.suptitle("Estimated verb onsets in ActPas scrambled audio", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_DIR / "fig_scrambled_verb_audio_onset_qc.png", dpi=170)
    plt.close(fig)
    return pd.DataFrame(rows)


def build_audio_onsets(font_prop: FontProperties) -> pd.DataFrame:
    active_passive_path = ACTIVE_PASSIVE_OUT_DIR / "verb_audio_onsets.csv"
    if active_passive_path.exists():
        active_passive = pd.read_csv(active_passive_path)
    else:
        active_passive = build_verb_audio_onsets(font_prop)
    active_passive = active_passive[active_passive["sound_file"].isin(STIM_BY_SOUND)].copy()
    scrambled = build_scrambled_audio_onsets(font_prop)
    onsets = pd.concat([active_passive, scrambled], ignore_index=True, sort=False)
    onsets = onsets.sort_values(["verb_id", "condition"])
    onsets.to_csv(OUT_DIR / "verb_three_condition_audio_onsets.csv", index=False)
    return onsets


def build_event_rows(onsets: pd.DataFrame, runs: list[tuple[str, str]]) -> pd.DataFrame:
    onset_lookup = {str(row.sound_file): row for row in onsets.itertuples(index=False)}
    event_rows: list[dict[str, object]] = []
    for patient, run in runs:
        bins = read_trial_bins(patient, run)
        for trial in bins.itertuples(index=False):
            sound_file = str(trial.sound_file)
            if sound_file not in onset_lookup:
                continue
            onset = onset_lookup[sound_file]
            condition = str(onset.condition)
            event_bin = int(trial.marker_bin) + int(round(float(onset.verb_onset_sec_from_sentence_onset) * 1000))
            event_rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "trial_index": int(trial.trial_index),
                    "sound_file": sound_file,
                    "sentence_condition": condition,
                    "trial_condition": str(trial.condition),
                    "sentence": str(onset.sentence),
                    "verb_id": int(onset.verb_id),
                    "verb": str(onset.verb),
                    "verb_onset_sec_from_sentence_onset": float(onset.verb_onset_sec_from_sentence_onset),
                    "event_bin_1000Hz": event_bin,
                    "event_time_sec": event_bin / 1000,
                    "onset_source": str(onset.onset_source),
                }
            )
    events = pd.DataFrame(event_rows)
    events.to_csv(OUT_DIR / "verb_three_condition_event_metadata.csv", index=False)
    return events


def collect_response_metrics(events: pd.DataFrame, unit_set: str) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    time_sec = np.arange(-PRE_MS, POST_MS + 1) / 1000
    baseline_mask = (time_sec >= BASELINE_SEC[0]) & (time_sec < BASELINE_SEC[1])
    response_mask = (time_sec >= RESPONSE_SEC[0]) & (time_sec <= RESPONSE_SEC[1])
    metric_frames: list[pd.DataFrame] = []
    run_meta: dict[str, dict[str, object]] = {}

    for (patient, run), run_events in events.groupby(["patient", "run"], sort=False):
        z = np.load(run_npz_path(str(patient), str(run), unit_set), allow_pickle=True)
        firing_rate = z["firing_rate_hz"]
        unit_ids = z["unit_ids"].astype(int)
        labels = z["ks_labels"].astype(str)
        run_key = f"{patient}/{run}"
        run_meta[run_key] = {
            "patient": str(patient),
            "run": str(run),
            "n_units": int(len(unit_ids)),
            "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }

        for event in run_events.itertuples(index=False):
            event_bin = int(event.event_bin_1000Hz)
            start = event_bin - PRE_MS
            stop = event_bin + POST_MS + 1
            if start < 0 or stop > firing_rate.shape[1]:
                warnings.warn(
                    f"Skipping out-of-bounds event: {patient} {run} trial {event.trial_index}",
                    RuntimeWarning,
                )
                continue
            epoch = firing_rate[:, start:stop]
            baseline = epoch[:, baseline_mask].mean(axis=1)
            response = epoch[:, response_mask].mean(axis=1)
            metric_frames.append(
                pd.DataFrame(
                    {
                        "patient": str(patient),
                        "run": str(run),
                        "unit_id": unit_ids,
                        "ks_label": labels,
                        "trial_index": int(event.trial_index),
                        "sound_file": str(event.sound_file),
                        "sentence_condition": str(event.sentence_condition),
                        "verb_id": int(event.verb_id),
                        "verb": str(event.verb),
                        "event_bin_1000Hz": event_bin,
                        "baseline_hz": baseline.astype(float),
                        "response_hz": response.astype(float),
                        "response_baseline_corrected_hz": (response - baseline).astype(float),
                    }
                )
            )
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(OUT_DIR / "verb_three_condition_unit_trial_response_metrics.csv", index=False)
    return metrics, run_meta


def paired_friedman(active: np.ndarray, passive: np.ndarray, scrambled: np.ndarray) -> tuple[float, float]:
    if len(active) < 2:
        return np.nan, 1.0
    if np.allclose(active, passive) and np.allclose(active, scrambled):
        return np.nan, 1.0
    try:
        result = stats.friedmanchisquare(active, passive, scrambled)
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, 1.0


def contiguous_segments(mask: np.ndarray, min_bins: int = MIN_CONSECUTIVE_SIGNIFICANT_BINS) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if int(stop - start) >= min_bins
    ]


def extract_smoothed_bc_epoch(
    firing_rate: np.ndarray,
    event_bin: int,
    baseline_mask: np.ndarray,
) -> np.ndarray | None:
    start = int(event_bin) - PRE_MS
    stop = int(event_bin) + POST_MS + 1
    if start < 0 or stop > firing_rate.shape[1]:
        return None
    epoch = firing_rate[:, start:stop].astype(float, copy=False)
    smooth = gaussian_smooth_ms(epoch, SMOOTH_SIGMA_MS)
    baseline = smooth[:, baseline_mask].mean(axis=1)
    return smooth - baseline[:, None]


def analyze_timepoint_units(
    events: pd.DataFrame,
    unit_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_sec = np.arange(-PRE_MS, POST_MS + 1) / 1000
    baseline_mask = (time_sec >= BASELINE_SEC[0]) & (time_sec < BASELINE_SEC[1])
    response_mask = (time_sec >= RESPONSE_SEC[0]) & (time_sec < RESPONSE_SEC[1])
    response_time_ms = np.rint(time_sec[response_mask] * 1000).astype(int)

    unit_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []

    for (patient, run), run_events in events.groupby(["patient", "run"], sort=False):
        z = np.load(run_npz_path(str(patient), str(run), unit_set), allow_pickle=True)
        firing_rate = z["firing_rate_hz"]
        unit_ids = z["unit_ids"].astype(int)
        labels = z["ks_labels"].astype(str)

        matched_events = run_events[run_events["verb_id"].isin(MATCHED_THREE_CONDITION_VERB_IDS)].copy()
        traces_by_condition: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
        verb_ids_by_condition: dict[str, list[int]] = {condition: [] for condition in CONDITIONS}
        for event in matched_events.sort_values(["sentence_condition", "verb_id"]).itertuples(index=False):
            trace = extract_smoothed_bc_epoch(firing_rate, int(event.event_bin_1000Hz), baseline_mask)
            if trace is None:
                warnings.warn(
                    f"Skipping out-of-bounds timepoint event: {patient} {run} trial {event.trial_index}",
                    RuntimeWarning,
                )
                continue
            condition = str(event.sentence_condition)
            traces_by_condition[condition].append(trace[:, response_mask])
            verb_ids_by_condition[condition].append(int(event.verb_id))

        if any(len(traces_by_condition[condition]) < 2 for condition in CONDITIONS):
            warnings.warn(f"{patient} {run}: insufficient matched events for timepoint screen", RuntimeWarning)
            continue

        arrays = {condition: np.stack(traces_by_condition[condition], axis=0) for condition in CONDITIONS}
        with np.errstate(invalid="ignore", divide="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                anova = stats.f_oneway(*(arrays[condition] for condition in CONDITIONS), axis=0)
        f_values = np.asarray(anova.statistic, dtype=float)
        p_values = np.asarray(anova.pvalue, dtype=float)
        p_values[~np.isfinite(p_values)] = 1.0
        f_values[~np.isfinite(f_values)] = np.nan
        passing = p_values < POINTWISE_ALPHA
        means = {condition: arrays[condition].mean(axis=0) for condition in CONDITIONS}

        for unit_index, unit_id in enumerate(unit_ids):
            unit_mask = passing[unit_index]
            segments = contiguous_segments(unit_mask)
            qualifying = np.zeros(unit_mask.shape, dtype=bool)
            for segment_id, (start_i, stop_i) in enumerate(segments, start=1):
                qualifying[start_i:stop_i] = True
                seg_means = {
                    condition: float(np.nanmean(means[condition][unit_index, start_i:stop_i]))
                    for condition in CONDITIONS
                }
                preferred = max(seg_means, key=seg_means.get)
                segment_rows.append(
                    {
                        "patient": str(patient),
                        "run": str(run),
                        "unit_id": int(unit_id),
                        "ks_label": str(labels[unit_index]),
                        "segment_id": segment_id,
                        "start_ms": int(response_time_ms[start_i]),
                        "stop_ms_exclusive": int(response_time_ms[stop_i - 1] + 1),
                        "duration_ms": int(stop_i - start_i),
                        "preferred_condition": preferred,
                        "active_mean_hz": seg_means["active"],
                        "passive_mean_hz": seg_means["passive"],
                        "scrambled_mean_hz": seg_means["scrambled"],
                        "max_minus_min_hz": float(max(seg_means.values()) - min(seg_means.values())),
                        "min_p": float(np.nanmin(p_values[unit_index, start_i:stop_i])),
                        "max_f": float(np.nanmax(f_values[unit_index, start_i:stop_i])),
                    }
                )

            passing_bins = np.flatnonzero(unit_mask)
            for time_i in passing_bins:
                point_rows.append(
                    {
                        "patient": str(patient),
                        "run": str(run),
                        "unit_id": int(unit_id),
                        "ks_label": str(labels[unit_index]),
                        "time_ms": int(response_time_ms[time_i]),
                        "anova_f_condition": float(f_values[unit_index, time_i])
                        if np.isfinite(f_values[unit_index, time_i])
                        else np.nan,
                        "anova_p_condition": float(p_values[unit_index, time_i]),
                        "active_mean_hz": float(means["active"][unit_index, time_i]),
                        "passive_mean_hz": float(means["passive"][unit_index, time_i]),
                        "scrambled_mean_hz": float(means["scrambled"][unit_index, time_i]),
                        "in_qualifying_segment": bool(qualifying[time_i]),
                    }
                )

            if qualifying.any():
                qualifying_means = {
                    condition: float(np.nanmean(means[condition][unit_index, qualifying]))
                    for condition in CONDITIONS
                }
                preferred_condition = max(qualifying_means, key=qualifying_means.get)
                qualifying_span = float(max(qualifying_means.values()) - min(qualifying_means.values()))
            else:
                qualifying_means = {condition: np.nan for condition in CONDITIONS}
                preferred_condition = "not_significant"
                qualifying_span = np.nan

            unit_rows.append(
                {
                    "patient": str(patient),
                    "run": str(run),
                    "unit_id": int(unit_id),
                    "ks_label": str(labels[unit_index]),
                    "n_conditions": N_CONDITIONS,
                    "pointwise_alpha": POINTWISE_ALPHA,
                    "min_consecutive_bins": MIN_CONSECUTIVE_SIGNIFICANT_BINS,
                    "response_window_start_ms": int(response_time_ms[0]),
                    "response_window_stop_ms_exclusive": int(response_time_ms[-1] + 1),
                    "complete_verb_ids_timepoint": ",".join(str(v) for v in MATCHED_THREE_CONDITION_VERB_IDS),
                    "n_active_trials_timepoint": int(arrays["active"].shape[0]),
                    "n_passive_trials_timepoint": int(arrays["passive"].shape[0]),
                    "n_scrambled_trials_timepoint": int(arrays["scrambled"].shape[0]),
                    "min_pointwise_p": float(np.nanmin(p_values[unit_index])),
                    "max_pointwise_f": float(np.nanmax(f_values[unit_index]))
                    if np.isfinite(f_values[unit_index]).any()
                    else np.nan,
                    "n_pointwise_passing_bins": int(unit_mask.sum()),
                    "n_qualifying_segments": int(len(segments)),
                    "first_segment_onset_ms": int(response_time_ms[segments[0][0]]) if segments else np.nan,
                    "longest_segment_ms": int(max((stop_i - start_i for start_i, stop_i in segments), default=0)),
                    "total_qualifying_duration_ms": int(qualifying.sum()),
                    PRIMARY_SIGNIFICANCE_COL: bool(segments),
                    "timepoint_preferred_condition": preferred_condition,
                    "qualifying_active_mean_hz": qualifying_means["active"],
                    "qualifying_passive_mean_hz": qualifying_means["passive"],
                    "qualifying_scrambled_mean_hz": qualifying_means["scrambled"],
                    "qualifying_max_minus_min_hz": qualifying_span,
                }
            )

    return pd.DataFrame(unit_rows), pd.DataFrame(segment_rows), pd.DataFrame(point_rows)


def analyze_units(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (patient, run, unit_id), group in metrics.groupby(["patient", "run", "unit_id"], sort=False):
        pivot = group.pivot_table(
            index="verb_id",
            columns="sentence_condition",
            values="response_baseline_corrected_hz",
            aggfunc="mean",
        ).reindex(range(1, 11))
        if not all(condition in pivot for condition in CONDITIONS):
            continue
        complete = pivot[list(CONDITIONS)].dropna()
        if complete.empty:
            continue
        active = complete["active"].to_numpy(float)
        passive = complete["passive"].to_numpy(float)
        scrambled = complete["scrambled"].to_numpy(float)
        stat, p_value = paired_friedman(active, passive, scrambled)
        _, p_active_passive = safe_wilcoxon(active - passive)
        _, p_active_scrambled = safe_wilcoxon(active - scrambled)
        _, p_passive_scrambled = safe_wilcoxon(passive - scrambled)
        means = {
            "active": float(np.mean(active)),
            "passive": float(np.mean(passive)),
            "scrambled": float(np.mean(scrambled)),
        }
        preference = max(means, key=means.get)
        rows.append(
            {
                "patient": patient,
                "run": run,
                "unit_id": int(unit_id),
                "ks_label": str(group["ks_label"].iloc[0]),
                "n_complete_verb_triples": int(len(complete)),
                "complete_verb_ids": ",".join(str(int(v)) for v in complete.index),
                "active_mean_bc_hz": means["active"],
                "passive_mean_bc_hz": means["passive"],
                "scrambled_mean_bc_hz": means["scrambled"],
                "active_minus_passive_bc_hz": means["active"] - means["passive"],
                "active_minus_scrambled_bc_hz": means["active"] - means["scrambled"],
                "passive_minus_scrambled_bc_hz": means["passive"] - means["scrambled"],
                "max_minus_min_bc_hz": float(max(means.values()) - min(means.values())),
                "preferred_condition": preference,
                "friedman_statistic": stat,
                "p_value_friedman": p_value,
                "p_active_vs_passive": p_active_passive,
                "p_active_vs_scrambled": p_active_scrambled,
                "p_passive_vs_scrambled": p_passive_scrambled,
            }
        )

    results = pd.DataFrame(rows)
    results["aggregate_friedman_p05"] = results["p_value_friedman"] < 0.05
    # Backward-compatible alias for older helper scripts. This is not the
    # primary significance criterion for the present analysis.
    results["significant_uncorrected_p05"] = results["aggregate_friedman_p05"]
    results = results.sort_values(["p_value_friedman", "patient", "run", "unit_id"])
    return results


def merge_timepoint_results(window_results: pd.DataFrame, timepoint_results: pd.DataFrame) -> pd.DataFrame:
    key = ["patient", "run", "unit_id", "ks_label"]
    results = window_results.merge(timepoint_results, on=key, how="left", validate="one_to_one")
    results[PRIMARY_SIGNIFICANCE_COL] = results[PRIMARY_SIGNIFICANCE_COL].fillna(False).astype(bool)
    results["n_pointwise_passing_bins"] = results["n_pointwise_passing_bins"].fillna(0).astype(int)
    results["n_qualifying_segments"] = results["n_qualifying_segments"].fillna(0).astype(int)
    results["longest_segment_ms"] = results["longest_segment_ms"].fillna(0).astype(int)
    results["total_qualifying_duration_ms"] = results["total_qualifying_duration_ms"].fillna(0).astype(int)
    results["timepoint_preferred_condition"] = results["timepoint_preferred_condition"].fillna("not_significant")
    return results.sort_values(
        [PRIMARY_SIGNIFICANCE_COL, "min_pointwise_p", "longest_segment_ms", "p_value_friedman", "patient", "run", "unit_id"],
        ascending=[False, True, False, True, True, True, True],
    )


def write_unit_result_tables(
    results: pd.DataFrame,
    timepoint_segments: pd.DataFrame,
    passing_bins: pd.DataFrame,
) -> None:
    results.to_csv(OUT_DIR / "verb_three_condition_unit_results.csv", index=False)
    results.loc[results[PRIMARY_SIGNIFICANCE_COL]].to_csv(
        OUT_DIR / "verb_three_condition_significant_units_time_gt5ms_p_condition.csv",
        index=False,
    )
    results.loc[results["aggregate_friedman_p05"]].to_csv(
        OUT_DIR / "verb_three_condition_candidate_units_aggregate_friedman_p05.csv",
        index=False,
    )
    timepoint_segments.to_csv(OUT_DIR / "verb_three_condition_timepoint_segments.csv", index=False)
    passing_bins.to_csv(OUT_DIR / "verb_three_condition_timepoint_passing_bins.csv", index=False)


def condition_matrix(run_metrics: pd.DataFrame, condition: str, units: list[int], verb_ids: list[int]) -> pd.DataFrame:
    return (
        run_metrics[run_metrics["sentence_condition"] == condition]
        .pivot_table(index="unit_id", columns="verb_id", values="response_baseline_corrected_hz", aggfunc="mean")
        .reindex(index=units, columns=verb_ids)
    )


def sparse_unit_ticks(ax: plt.Axes, unit_ids: list[int]) -> None:
    n = len(unit_ids)
    if n <= 25:
        ticks = np.arange(n)
    else:
        step = max(1, int(np.ceil(n / 16)))
        ticks = np.arange(0, n, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(unit_ids[i]) for i in ticks])


def candidate_codes(run_results: pd.DataFrame, units: list[int]) -> np.ndarray:
    stats_by_unit = run_results.set_index("unit_id").reindex(units)
    codes = np.zeros(len(units), dtype=int)
    is_candidate = stats_by_unit[PRIMARY_SIGNIFICANCE_COL].fillna(False).to_numpy(bool)
    pref = stats_by_unit["timepoint_preferred_condition"].fillna("").to_numpy(str)
    for i, condition in enumerate(CONDITIONS, start=1):
        codes[is_candidate & (pref == condition)] = i
    return codes[:, None]


def plot_all_unit_heatmaps(metrics: pd.DataFrame, results: pd.DataFrame, font_prop: FontProperties) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    pdf_path = fig_dir / "verb_three_condition_all_units_response_heatmaps_by_run.pdf"
    png_paths: list[str] = []
    run_order = list(metrics[["patient", "run"]].drop_duplicates().itertuples(index=False, name=None))

    with PdfPages(pdf_path) as pdf:
        for patient, run in run_order:
            run_metrics = metrics[(metrics["patient"] == patient) & (metrics["run"] == run)]
            run_results = results[(results["patient"] == patient) & (results["run"] == run)].copy()
            units = (
                run_results.sort_values("max_minus_min_bc_hz", ascending=False)["unit_id"]
                .astype(int)
                .tolist()
            )
            verb_ids = sorted(run_metrics[run_metrics["sentence_condition"] == "scrambled"]["verb_id"].unique().astype(int))
            if not verb_ids:
                continue
            mats = {condition: condition_matrix(run_metrics, condition, units, verb_ids) for condition in CONDITIONS}
            contrasts = {
                "active - scrambled": mats["active"] - mats["scrambled"],
                "passive - scrambled": mats["passive"] - mats["scrambled"],
            }
            vals = np.concatenate([m.to_numpy(float).ravel() for m in [*mats.values(), *contrasts.values()]])
            finite = vals[np.isfinite(vals)]
            vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
            vmax = max(vmax, 1e-6)
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            height = min(18, max(7, 0.055 * len(units) + 3.5))
            fig = plt.figure(figsize=(22, height))
            gs = fig.add_gridspec(1, 7, width_ratios=[0.035, 1, 1, 1, 1, 1, 0.04], wspace=0.06)
            ax_sig = fig.add_subplot(gs[0, 0])
            axes = [fig.add_subplot(gs[0, i], sharey=ax_sig) for i in range(1, 6)]
            cax = fig.add_subplot(gs[0, 6])
            cmap = mpl.colors.ListedColormap(["#F2F2F2", CONDITION_COLORS["active"], CONDITION_COLORS["passive"], CONDITION_COLORS["scrambled"]])
            ax_sig.imshow(candidate_codes(run_results, units), aspect="auto", cmap=cmap, vmin=0, vmax=3)
            ax_sig.set_title("p<0.0167\n>5 ms", fontsize=8)
            ax_sig.set_xticks([])
            ax_sig.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            for spine in ax_sig.spines.values():
                spine.set_visible(False)

            panels = [
                (mats["active"], "active"),
                (mats["passive"], "passive"),
                (mats["scrambled"], "scrambled"),
                (contrasts["active - scrambled"], "active - scrambled"),
                (contrasts["passive - scrambled"], "passive - scrambled"),
            ]
            for ax, (mat, title) in zip(axes, panels):
                im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap="RdBu_r", norm=norm)
                ax.set_title(title)
                ax.set_xticks(np.arange(len(verb_ids)))
                ax.set_xticklabels([VERB_LABELS[i - 1] for i in verb_ids], rotation=0, fontproperties=font_prop)
                ax.set_xlabel("verb")
                sparse_unit_ticks(ax, units)
            axes[0].set_ylabel("")
            for ax in axes[1:]:
                ax.tick_params(labelleft=False)
            fig.colorbar(im, cax=cax, label="baseline-corrected Hz")
            screened = run_results[run_results[PRIMARY_SIGNIFICANCE_COL]]
            fig.legend(
                handles=[
                    mpl.patches.Patch(color=CONDITION_COLORS[c], label=f"time screen {c}-pref")
                    for c in CONDITIONS
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=3,
                frameon=False,
                fontsize=8,
            )
            fig.text(0.018, 0.53, "unit id, sorted by max condition effect", rotation=90, va="center", fontsize=9)
            fig.suptitle(
                f"{patient} {run}: active/passive/scrambled verb responses; "
                f"complete verbs={','.join(map(str, verb_ids))}; "
                f"time screen p<{POINTWISE_ALPHA:.4f}, >5 ms={len(screened)}",
                y=0.995,
            )
            fig.subplots_adjust(left=0.055, right=0.965, bottom=0.18, top=0.94)
            pdf.savefig(fig)
            png_path = fig_dir / f"verb_three_condition_all_units_heatmap_{patient}_{run}.png"
            fig.savefig(png_path, dpi=170)
            png_paths.append(str(png_path))
            plt.close(fig)
    return {"pdf": str(pdf_path), "png_dir": str(fig_dir), "first_png": png_paths[0] if png_paths else ""}


def extract_unit_condition_traces(
    patient: str,
    run: str,
    unit_id: int,
    events: pd.DataFrame,
    unit_set: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time_sec = np.arange(-PRE_MS, POST_MS + 1) / 1000
    z = np.load(run_npz_path(patient, run, unit_set), allow_pickle=True)
    firing_rate = z["firing_rate_hz"]
    unit_ids = z["unit_ids"].astype(int)
    matches = np.flatnonzero(unit_ids == unit_id)
    if matches.size == 0:
        raise ValueError(f"unit {unit_id} not found in {patient} {run}")
    unit_i = int(matches[0])
    traces: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    run_events = events[(events["patient"] == patient) & (events["run"] == run)].sort_values(
        ["verb_id", "sentence_condition"]
    )
    for event in run_events.itertuples(index=False):
        start = int(event.event_bin_1000Hz) - PRE_MS
        stop = int(event.event_bin_1000Hz) + POST_MS + 1
        if start < 0 or stop > firing_rate.shape[1]:
            continue
        epoch = firing_rate[unit_i, start:stop].astype(float)
        baseline = float(epoch[(time_sec >= BASELINE_SEC[0]) & (time_sec < BASELINE_SEC[1])].mean())
        traces[str(event.sentence_condition)].append(epoch - baseline)
    out = {
        condition: np.stack(items, axis=0) if items else np.empty((0, PRE_MS + POST_MS + 1))
        for condition, items in traces.items()
    }
    return time_sec, out


def mean_smoothed_trace(traces: dict[str, np.ndarray], condition: str) -> np.ndarray | None:
    arr = traces.get(condition)
    if arr is None or arr.size == 0:
        return None
    return np.nanmean(gaussian_smooth_ms(arr, SMOOTH_SIGMA_MS), axis=0)


def select_summary_example(
    results: pd.DataFrame,
    events: pd.DataFrame,
    unit_set: str,
) -> pd.Series | None:
    """Prefer a unit with high active/passive response and flat scrambled response."""
    if results.empty:
        return None

    preference = results["timepoint_preferred_condition"].where(
        results[PRIMARY_SIGNIFICANCE_COL],
        results["preferred_condition"],
    )
    candidates = results[preference.isin(["active", "passive"])].copy()
    candidates["plot_preferred_condition"] = preference.loc[candidates.index].to_numpy(str)
    if candidates.empty:
        return results.sort_values("min_pointwise_p").iloc[0]

    significant = candidates[candidates[PRIMARY_SIGNIFICANCE_COL]]
    if not significant.empty:
        candidates = significant

    pref_minus_scrambled = np.where(
        candidates["plot_preferred_condition"].to_numpy(str) == "active",
        candidates["active_minus_scrambled_bc_hz"].to_numpy(float),
        candidates["passive_minus_scrambled_bc_hz"].to_numpy(float),
    )
    candidates = candidates.assign(pref_minus_scrambled_bc_hz=pref_minus_scrambled)
    positive = candidates[candidates["pref_minus_scrambled_bc_hz"] > 0]
    if not positive.empty:
        candidates = positive

    candidates = candidates.sort_values(
        ["min_pointwise_p", "longest_segment_ms", "pref_minus_scrambled_bc_hz", "max_minus_min_bc_hz"],
        ascending=[True, False, False, False],
    )

    response_mask: np.ndarray | None = None
    flat_mask: np.ndarray | None = None
    passing: list[tuple[float, pd.Series]] = []
    fallbacks: list[tuple[float, pd.Series]] = []
    for _, row in candidates.head(200).iterrows():
        try:
            time_sec, traces = extract_unit_condition_traces(
                str(row["patient"]),
                str(row["run"]),
                int(row["unit_id"]),
                events,
                unit_set,
            )
        except (FileNotFoundError, ValueError, KeyError):
            continue

        if response_mask is None:
            response_mask = (time_sec >= RESPONSE_SEC[0]) & (time_sec <= RESPONSE_SEC[1])
            flat_mask = (time_sec >= -0.1) & (time_sec <= 1.1)
        means = {condition: mean_smoothed_trace(traces, condition) for condition in CONDITIONS}
        if any(mean is None for mean in means.values()):
            continue

        preferred = str(row["plot_preferred_condition"])
        preferred_trace = means[preferred]
        scrambled_trace = means["scrambled"]
        if preferred_trace is None or scrambled_trace is None:
            continue
        pref_peak = float(np.nanmax(preferred_trace[response_mask]))
        pref_mean = float(np.nanmean(preferred_trace[response_mask]))
        scrambled_abs_peak = float(np.nanmax(np.abs(scrambled_trace[flat_mask])))
        scrambled_abs_mean = float(np.nanmean(np.abs(scrambled_trace[flat_mask])))
        if not np.isfinite([pref_peak, pref_mean, scrambled_abs_peak, scrambled_abs_mean]).all():
            continue

        score = (
            pref_peak
            + 0.45 * pref_mean
            - 1.35 * scrambled_abs_peak
            - 0.45 * scrambled_abs_mean
            + 0.10 * float(row["max_minus_min_bc_hz"])
        )
        annotated = row.copy()
        annotated["example_pref_peak_hz"] = pref_peak
        annotated["example_pref_mean_hz"] = pref_mean
        annotated["example_scrambled_abs_peak_hz"] = scrambled_abs_peak
        annotated["example_scrambled_abs_mean_hz"] = scrambled_abs_mean
        annotated["example_score"] = score
        fallbacks.append((score, annotated))

        if pref_peak >= 3.5 and scrambled_abs_peak <= 0.75 and pref_peak >= scrambled_abs_peak + 3.0:
            passing.append((score, annotated))

    if passing:
        return max(passing, key=lambda item: item[0])[1]
    if fallbacks:
        return max(fallbacks, key=lambda item: item[0])[1]
    return candidates.iloc[0]


def fixed_summary_examples(results: pd.DataFrame) -> list[pd.Series]:
    examples: list[pd.Series] = []
    for patient, run, unit_id in SUMMARY_EXAMPLE_UNITS:
        match = results[
            (results["patient"] == patient)
            & (results["run"] == run)
            & (results["unit_id"] == unit_id)
        ]
        if not match.empty:
            examples.append(match.iloc[0])
    return examples


def trace_peak_summary(time_sec: np.ndarray, traces: dict[str, np.ndarray]) -> dict[str, float | str]:
    response_mask = (time_sec >= RESPONSE_SEC[0]) & (time_sec <= RESPONSE_SEC[1])
    flat_mask = (time_sec >= -0.1) & (time_sec <= 1.1)
    means = {condition: mean_smoothed_trace(traces, condition) for condition in CONDITIONS}
    active = means["active"]
    passive = means["passive"]
    scrambled = means["scrambled"]
    if active is None or passive is None or scrambled is None:
        return {
            "syntax_condition": "active/passive",
            "syntax_peak_hz": np.nan,
            "scrambled_abs_peak_hz": np.nan,
        }
    active_peak = float(np.nanmax(active[response_mask]))
    passive_peak = float(np.nanmax(passive[response_mask]))
    if passive_peak >= active_peak:
        syntax_condition = "passive"
        syntax_peak = passive_peak
    else:
        syntax_condition = "active"
        syntax_peak = active_peak
    return {
        "syntax_condition": syntax_condition,
        "syntax_peak_hz": syntax_peak,
        "scrambled_abs_peak_hz": float(np.nanmax(np.abs(scrambled[flat_mask]))),
    }


def plot_summary_figure(
    results: pd.DataFrame,
    events: pd.DataFrame,
    unit_set: str,
) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_verb_three_condition_summary.png"
    pdf_path = fig_dir / "fig_verb_three_condition_summary.pdf"

    run_order = list(results[["patient", "run"]].drop_duplicates().itertuples(index=False, name=None))
    run_labels = [f"{patient}\n{shortened_run_name(run)}" for patient, run in run_order]

    fig = plt.figure(figsize=(17, 11.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 1.35], hspace=0.38, wspace=0.32)
    ax_schematic = fig.add_subplot(gs[0, 0])
    ax_counts = fig.add_subplot(gs[0, 1])
    ax_effect = fig.add_subplot(gs[0, 2])
    ax_scatter = fig.add_subplot(gs[1, 0])
    example_gs = gs[1, 1:].subgridspec(2, 1, hspace=0.34)
    ax_examples = [fig.add_subplot(example_gs[i, 0]) for i in range(2)]

    ax_schematic.axis("off")
    ax_schematic.set_title("Matched verbs across three contexts", loc="left", fontsize=12, weight="bold")
    ax_schematic.text(0.02, 0.82, "Complete three-condition set", fontsize=9, color="0.35")
    labels = [
        ("active", "虫子把树叶  啃光  了"),
        ("passive", "树叶被虫子  啃光  了"),
        ("scrambled", "叶虫光了把啃树子  (non001)"),
    ]
    y_positions = [0.62, 0.43, 0.24]
    for y, (condition, text) in zip(y_positions, labels):
        ax_schematic.text(0.03, y, condition, color=CONDITION_COLORS[condition], fontsize=10, weight="bold", va="center")
        ax_schematic.text(0.28, y, text, fontsize=14, color="0.25", va="center")
    ax_schematic.text(
        0.02,
        0.02,
        "Only five ActPas scrambled files were presented per run:\n"
        "non001/non003/non005/non007/non009.\n"
        f"Primary screen: p<0.05/{N_CONDITIONS}={POINTWISE_ALPHA:.4f}, contiguous >5 ms; no FDR.",
        fontsize=8.8,
        color="0.25",
        va="bottom",
    )

    pref_counts = []
    for patient, run in run_order:
        group = results[
            (results["patient"] == patient)
            & (results["run"] == run)
            & results[PRIMARY_SIGNIFICANCE_COL]
        ]
        counts = group["timepoint_preferred_condition"].value_counts()
        pref_counts.append([int(counts.get(c, 0)) for c in CONDITIONS])
    y = np.arange(len(run_labels))
    left = np.zeros(len(run_labels))
    for i, condition in enumerate(CONDITIONS):
        vals = np.array([row[i] for row in pref_counts])
        ax_counts.barh(y, vals, left=left, color=CONDITION_COLORS[condition], alpha=0.9, label=condition)
        left += vals
    ax_counts.set_yticks(y)
    ax_counts.set_yticklabels(run_labels)
    ax_counts.set_xlabel(f"time-screen units (p<{POINTWISE_ALPHA:.4f}, >5 ms)")
    ax_counts.set_title("Three-condition time-screen units", loc="left", fontsize=12, weight="bold")
    ax_counts.legend(frameon=False, loc="lower right")
    ax_counts.grid(axis="x", color="0.9", lw=0.7)

    distributions = [
        results[(results["patient"] == patient) & (results["run"] == run)]["max_minus_min_bc_hz"].to_numpy()
        for patient, run in run_order
    ]
    ax_effect.boxplot(distributions, showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for patch in ax_effect.artists:
        patch.set_facecolor("#EAEAEA")
        patch.set_edgecolor("0.45")
    ax_effect.set_xticks(np.arange(1, len(run_labels) + 1))
    ax_effect.set_xticklabels(run_labels, rotation=45, ha="right")
    ax_effect.set_ylabel("max - min condition response (Hz)")
    ax_effect.set_title("Condition-effect size", loc="left", fontsize=12, weight="bold")
    ax_effect.grid(axis="y", color="0.9", lw=0.7)

    x = results["max_minus_min_bc_hz"].to_numpy(float)
    yv = -np.log10(np.clip(results["min_pointwise_p"].to_numpy(float), 1e-12, 1.0))
    plot_pref = results["timepoint_preferred_condition"].where(
        results[PRIMARY_SIGNIFICANCE_COL],
        results["preferred_condition"],
    )
    colors = np.array([CONDITION_COLORS.get(c, "#BDBDBD") for c in plot_pref.to_numpy(str)], dtype=object)
    colors[~results[PRIMARY_SIGNIFICANCE_COL].to_numpy(bool)] = "#BDBDBD"
    sizes = np.where(results[PRIMARY_SIGNIFICANCE_COL].to_numpy(bool), 34, 12)
    ax_scatter.scatter(x, yv, c=colors, s=sizes, alpha=0.82, edgecolors="none")
    ax_scatter.axhline(-np.log10(POINTWISE_ALPHA), color="0.55", lw=0.8, linestyle="--")
    ax_scatter.set_xlabel("max - min condition response (Hz)")
    ax_scatter.set_ylabel("-log10 min pointwise p")
    ax_scatter.set_title("Unit-level timepoint screen", loc="left", fontsize=12, weight="bold")
    ax_scatter.grid(color="0.92", lw=0.7)

    example_rows = fixed_summary_examples(results)
    if not example_rows:
        top = select_summary_example(results, events, unit_set)
        example_rows = [top] if top is not None else []
    for ax, top in zip(ax_examples, example_rows):
        time_sec, traces = extract_unit_condition_traces(
            str(top["patient"]),
            str(top["run"]),
            int(top["unit_id"]),
            events,
            unit_set,
        )
        for condition in CONDITIONS:
            arr = gaussian_smooth_ms(traces[condition], SMOOTH_SIGMA_MS)
            if arr.size == 0:
                continue
            mean = np.nanmean(arr, axis=0)
            n = np.sum(np.isfinite(arr), axis=0)
            se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
            color = CONDITION_COLORS[condition]
            ax.fill_between(time_sec, mean - se, mean + se, color=color, alpha=0.14, lw=0)
            ax.plot(time_sec, mean, color=color, lw=1.8, label=condition)
        peak_info = trace_peak_summary(time_sec, traces)
        ax.axvline(0, color="0.25", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.set_xlim(-0.25, 1.16)
        ax.set_ylabel("baseline-corrected Hz")
        ax.set_title(
            f"Example {peak_info['syntax_condition']}-high: {top['patient']} u{int(top['unit_id'])}, "
            f"peak={float(peak_info['syntax_peak_hz']):.1f} Hz, "
            f"scr={float(peak_info['scrambled_abs_peak_hz']):.1f} Hz, "
            f"min p={float(top['min_pointwise_p']):.3g}, "
            f"seg={int(top['longest_segment_ms'])} ms",
            loc="left",
            fontsize=11,
            weight="bold",
        )
        ax.grid(color="0.92", lw=0.7)
    for ax in ax_examples[len(example_rows):]:
        ax.axis("off")
        ax.text(0.5, 0.5, "Example unavailable", ha="center", va="center")
    ax_examples[-1].set_xlabel("time from estimated verb onset (s)")
    ax_examples[0].legend(frameon=False, loc="upper right")

    fig.suptitle("Verb responses: active, passive, and ActPas scrambled", y=0.995, fontsize=15, weight="bold")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def plot_fixed_examples_figure(
    results: pd.DataFrame,
    events: pd.DataFrame,
    unit_set: str,
) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_verb_three_condition_examples_crop.png"
    pdf_path = fig_dir / "fig_verb_three_condition_examples_crop.pdf"

    example_rows = fixed_summary_examples(results)
    if not example_rows:
        top = select_summary_example(results, events, unit_set)
        example_rows = [top] if top is not None else []

    fig, axes = plt.subplots(max(1, len(example_rows)), 1, figsize=(13.0, 3.65 * max(1, len(example_rows))), sharex=True)
    axes = np.atleast_1d(axes)
    if not example_rows:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "Example unavailable", ha="center", va="center")
    for ax, row in zip(axes, example_rows):
        time_sec, traces = extract_unit_condition_traces(
            str(row["patient"]),
            str(row["run"]),
            int(row["unit_id"]),
            events,
            unit_set,
        )
        for condition in CONDITIONS:
            arr = gaussian_smooth_ms(traces[condition], SMOOTH_SIGMA_MS)
            if arr.size == 0:
                continue
            mean = np.nanmean(arr, axis=0)
            n = np.sum(np.isfinite(arr), axis=0)
            se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
            color = CONDITION_COLORS[condition]
            ax.fill_between(time_sec, mean - se, mean + se, color=color, alpha=0.14, lw=0)
            ax.plot(time_sec, mean, color=color, lw=1.8, label=condition)
        peak_info = trace_peak_summary(time_sec, traces)
        ax.axvline(0, color="0.25", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.set_xlim(-0.25, 1.16)
        ax.set_ylabel("baseline-corrected Hz")
        ax.grid(color="0.92", lw=0.7)
        ax.set_title(
            f"Example {peak_info['syntax_condition']}-high: {row['patient']} u{int(row['unit_id'])}, "
            f"peak={float(peak_info['syntax_peak_hz']):.1f} Hz, "
            f"scr={float(peak_info['scrambled_abs_peak_hz']):.1f} Hz, "
            f"min p={float(row['min_pointwise_p']):.3g}, "
            f"seg={int(row['longest_segment_ms'])} ms",
            loc="left",
            fontsize=12,
        )
    axes[-1].set_xlabel("time from estimated verb onset (s)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def summarize_by_run(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (patient, run), group in results.groupby(["patient", "run"], sort=False):
        screened = group[group[PRIMARY_SIGNIFICANCE_COL]]
        counts = screened["timepoint_preferred_condition"].value_counts()
        rows.append(
            {
                "patient": patient,
                "run": run,
                "n_units": int(len(group)),
                "n_time_gt5ms_p_condition": int(len(screened)),
                "n_aggregate_friedman_p05": int(group["aggregate_friedman_p05"].sum()),
                "active_pref_time": int(counts.get("active", 0)),
                "passive_pref_time": int(counts.get("passive", 0)),
                "scrambled_pref_time": int(counts.get("scrambled", 0)),
                "best_min_pointwise_p": float(group["min_pointwise_p"].min()) if len(group) else np.nan,
                "max_longest_segment_ms": int(group["longest_segment_ms"].max()) if len(group) else 0,
                "best_aggregate_friedman_p": float(group["p_value_friedman"].min()) if len(group) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "verb_three_condition_summary_by_run.csv", index=False)
    return summary


def write_summary(
    unit_set: str,
    run_meta: dict[str, dict[str, object]],
    events: pd.DataFrame,
    results: pd.DataFrame,
    run_summary: pd.DataFrame,
    heatmaps: dict[str, str],
    summary_fig: dict[str, str],
    examples_fig: dict[str, str],
) -> dict[str, object]:
    event_counts = events.groupby(["sentence_condition", "verb_id"]).size().reset_index(name="n_events")
    top_units = results.sort_values(
        [PRIMARY_SIGNIFICANCE_COL, "min_pointwise_p", "longest_segment_ms", "p_value_friedman"],
        ascending=[False, True, False, True],
    ).head(20)
    summary = {
        "analysis_version": "corrected_20260812",
        "scrambled_anchor_definition": "first character of canonical matched verb",
        "unit_set": unit_set,
        "method": {
            "baseline_sec": list(BASELINE_SEC),
            "response_sec": list(RESPONSE_SEC),
            "pointwise_test": "one-way ANOVA across active/passive/scrambled trials at each 1-ms bin",
            "pointwise_alpha": POINTWISE_ALPHA,
            "n_conditions": N_CONDITIONS,
            "significant_unit_rule": "at least 6 consecutive 1-ms bins, equivalent to significant time >5 ms",
            "multiple_comparison": "FDR not applied",
            "complete_verb_ids_for_three_condition": list(MATCHED_THREE_CONDITION_VERB_IDS),
        },
        "n_runs": int(results[["patient", "run"]].drop_duplicates().shape[0]),
        "n_units": int(len(results)),
        "complete_verb_ids_for_three_condition": list(MATCHED_THREE_CONDITION_VERB_IDS),
        "pointwise_alpha": POINTWISE_ALPHA,
        "min_consecutive_significant_bins": MIN_CONSECUTIVE_SIGNIFICANT_BINS,
        "n_time_gt5ms_p_condition": int(results[PRIMARY_SIGNIFICANCE_COL].sum()),
        "n_aggregate_friedman_p05": int(results["aggregate_friedman_p05"].sum()),
        "event_counts_by_condition_verb": event_counts.to_dict(orient="records"),
        "run_meta": run_meta,
        "outputs": {
            "audio_onsets_csv": str(OUT_DIR / "verb_three_condition_audio_onsets.csv"),
            "scrambled_audio_qc_png": str(OUT_DIR / "fig_scrambled_verb_audio_onset_qc.png"),
            "event_metadata_csv": str(OUT_DIR / "verb_three_condition_event_metadata.csv"),
            "trial_response_metrics_csv": str(OUT_DIR / "verb_three_condition_unit_trial_response_metrics.csv"),
            "unit_results_csv": str(OUT_DIR / "verb_three_condition_unit_results.csv"),
            "time_screen_units_csv": str(OUT_DIR / "verb_three_condition_significant_units_time_gt5ms_p_condition.csv"),
            "aggregate_friedman_p05_units_csv": str(OUT_DIR / "verb_three_condition_candidate_units_aggregate_friedman_p05.csv"),
            "timepoint_segments_csv": str(OUT_DIR / "verb_three_condition_timepoint_segments.csv"),
            "timepoint_passing_bins_csv": str(OUT_DIR / "verb_three_condition_timepoint_passing_bins.csv"),
            "run_summary_csv": str(OUT_DIR / "verb_three_condition_summary_by_run.csv"),
            "heatmap_pdf": heatmaps["pdf"],
            "summary_png": summary_fig["png"],
            "summary_pdf": summary_fig["pdf"],
            "examples_png": examples_fig["png"],
            "examples_pdf": examples_fig["pdf"],
        },
        "top_units_by_time_screen": top_units[
            [
                "patient",
                "run",
                "unit_id",
                "ks_label",
                "n_complete_verb_triples",
                PRIMARY_SIGNIFICANCE_COL,
                "min_pointwise_p",
                "longest_segment_ms",
                "first_segment_onset_ms",
                "timepoint_preferred_condition",
                "qualifying_max_minus_min_hz",
                "max_minus_min_bc_hz",
                "preferred_condition",
                "p_value_friedman",
                "active_minus_passive_bc_hz",
                "active_minus_scrambled_bc_hz",
                "passive_minus_scrambled_bc_hz",
            ]
        ].to_dict(orient="records"),
    }
    (OUT_DIR / "verb_three_condition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = [
        "# Verb responses across active, passive, and ActPas scrambled sentences",
        "",
        "This analysis compares verb-locked unit activity across active, passive, and ActPas scrambled/no-meaning sentences.",
        "",
        f"Unit set: `{unit_set}`. Response window: {RESPONSE_SEC[0]:.1f}-{RESPONSE_SEC[1]:.1f} s after estimated verb onset; baseline: {BASELINE_SEC[0]:.1f}-{BASELINE_SEC[1]:.1f} s.",
        "",
        "Important design note: each run contains 10 active and 10 passive ActPas trials, but only 5 ActPas scrambled trials: `non001`, `non003`, `non005`, `non007`, and `non009`. The three-condition screen therefore uses the complete matched verb IDs 1, 3, 5, 7, and 9.",
        "",
        f"Primary criterion: no FDR; pointwise one-way ANOVA p < 0.05/{N_CONDITIONS} = {POINTWISE_ALPHA:.4f}; a unit is called significant only if it has at least {MIN_CONSECUTIVE_SIGNIFICANT_BINS} consecutive 1-ms bins (>5 ms).",
        "",
        "## Summary",
        "",
        f"- Units analyzed: {summary['n_units']}",
        f"- Time-resolved significant units: {summary['n_time_gt5ms_p_condition']}",
        f"- Aggregate 0-0.8 s Friedman p<0.05 reference units: {summary['n_aggregate_friedman_p05']}",
        "",
        "## By Run",
        "",
        "| patient | run | n_units | time_gt5ms | aggregate_p05 | active_pref | passive_pref | scrambled_pref | best_min_p | longest_ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_summary.itertuples(index=False):
        lines.append(
            f"| {row.patient} | {row.run} | {int(row.n_units)} | {int(row.n_time_gt5ms_p_condition)} | "
            f"{int(row.n_aggregate_friedman_p05)} | {int(row.active_pref_time)} | "
            f"{int(row.passive_pref_time)} | {int(row.scrambled_pref_time)} | "
            f"{float(row.best_min_pointwise_p):.3g} | {int(row.max_longest_segment_ms)} |"
        )
    lines.extend(
        [
            "",
            "## Top Units By Time-Resolved Screen",
            "",
            "| patient | run | unit_id | ks_label | n_triples | time_sig | time_pref | min_p | longest_ms | max-min Hz | Friedman p | A-P | A-S | P-S |",
            "|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_units.itertuples(index=False):
        row_dict = row._asdict()
        lines.append(
            f"| {row.patient} | {row.run} | {int(row.unit_id)} | {row.ks_label} | "
            f"{int(row.n_complete_verb_triples)} | {bool(row_dict[PRIMARY_SIGNIFICANCE_COL])} | "
            f"{row.timepoint_preferred_condition} | {float(row.min_pointwise_p):.3g} | "
            f"{int(row.longest_segment_ms)} | {float(row.max_minus_min_bc_hz):.3g} | "
            f"{float(row.p_value_friedman):.3g} | {float(row.active_minus_passive_bc_hz):.3g} | "
            f"{float(row.active_minus_scrambled_bc_hz):.3g} | {float(row.passive_minus_scrambled_bc_hz):.3g} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Summary figure: `{summary['outputs']['summary_png']}`",
            f"- Examples figure: `{summary['outputs']['examples_png']}`",
            f"- All-unit heatmaps: `{summary['outputs']['heatmap_pdf']}`",
            f"- Per-unit statistics: `{summary['outputs']['unit_results_csv']}`",
            f"- Time-screen unit list: `{summary['outputs']['time_screen_units_csv']}`",
            f"- Time-screen segments: `{summary['outputs']['timepoint_segments_csv']}`",
            f"- Per-trial unit metrics: `{summary['outputs']['trial_response_metrics_csv']}`",
            f"- Scrambled audio onset QC: `{summary['outputs']['scrambled_audio_qc_png']}`",
            "",
            "Interpretation note: scrambled-sentence text is taken from the ActPas material workbook; scrambled verb onsets are manually selected from audio envelopes.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"starting analysis: unit_set={args.unit_set}")
    font_prop = setup_matplotlib()
    log("building active/passive/scrambled verb audio onset table")
    onsets = build_audio_onsets(font_prop)
    runs = discover_runs()
    log(f"building event table for {len(runs)} runs")
    events = build_event_rows(onsets, runs)
    log("collecting unit response metrics")
    metrics, run_meta = collect_response_metrics(events, args.unit_set)
    log("running aggregate three-condition paired summaries")
    window_results = analyze_units(metrics)
    log("running time-resolved three-condition screen")
    timepoint_results, timepoint_segments, passing_bins = analyze_timepoint_units(events, args.unit_set)
    results = merge_timepoint_results(window_results, timepoint_results)
    write_unit_result_tables(results, timepoint_segments, passing_bins)
    log("plotting all-unit heatmaps")
    heatmaps = plot_all_unit_heatmaps(metrics, results, font_prop)
    log("plotting summary figure")
    summary_fig = plot_summary_figure(results, events, args.unit_set)
    log("plotting fixed example traces")
    examples_fig = plot_fixed_examples_figure(results, events, args.unit_set)
    log("summarizing by run")
    run_summary = summarize_by_run(results)
    log("writing summary and report")
    summary = write_summary(args.unit_set, run_meta, events, results, run_summary, heatmaps, summary_fig, examples_fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
