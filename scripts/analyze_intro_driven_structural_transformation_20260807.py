#!/usr/bin/env python3
"""Introduction-driven matched-noun structural transformation analysis.

The script implements five prespecified modules for the Mandarin active/passive
Neuropixels dataset: cross-role parallelism/CCGP, a frozen time-resolved
position-axis analysis, acoustic/timing controls, cross-fitted unit
participation/ablation, and a sparse meaningful-versus-scrambled diagnostic.

Patients are the independent units.  Surface position remains confounded with
serial time and local function-word context, so the outputs quantify a shared
construction/position-related population component rather than pure syntax.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from scripts import analyze_active_passive_geometry_transform_20260806 as base
from scripts import analyze_event_anchor_manifold_geometry_20260806 as anchor_analysis
from scripts import analyze_noun_role_structure_encoding as noun_analysis


DATE_TAG = "20260807"
REPORT_DIR = ROOT / f"reports/intro_driven_structural_transformation_{DATE_TAG}"
FIG_DIR = REPORT_DIR / "figures"
SOURCE_DIR = REPORT_DIR / "source_data"
DATA_DIR = (
    ROOT
    / "neuropixel2025/data"
    / f"combined_intro_driven_structural_transformation_{DATE_TAG}"
)
CONTRACT_PATH = REPORT_DIR / "analysis_contract.json"
EVENTS_PATH = ROOT / "reports/noun_firstchar_llm_mapping_20260725/noun_firstchar_event_metadata.csv"
MARKER_ONSETS_PATH = (
    ROOT
    / "neuropixel2025/data/gz001/processed/ActPasSynSentence/event_related_active_passive"
    / "active_passive_function_word_audio_onsets.csv"
)

UNIT_SETS = ("good_only", "usable_lenient")
PRIMARY_UNIT_SET = "good_only"
ARTIFACT_THRESHOLD = 0.2
POSITIONS = ("initial", "post_marker")
ROLES = ("agent", "patient")
FIXED_WINDOWS_MS = ((0, 200), (200, 500), (500, 800))
CONTINUOUS_START_MS = -200
CONTINUOUS_STOP_MS = 800
CONTINUOUS_STEP_MS = 5
POINT_ALPHA = 0.05 / 4
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
ABLATION_FRACTIONS = (0.05, 0.10, 0.20)
RANDOM_ABLATION_DRAWS = 100
ACOUSTIC_WINDOW_SEC = 0.250
SR = 16_000

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#7B5AA6",
    "sky": "#56B4E9",
    "gray": "#70757A",
    "light": "#D9DEE5",
    "black": "#1F2933",
}
PATIENT_COLORS = {
    "gz001": "#355070",
    "gz002": "#6D597A",
    "gz003": "#B56576",
    "gz004": "#E56B6F",
    "gz005": "#EAAC8B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-set",
        choices=("all",) + UNIT_SETS,
        default="all",
        help="Run both frozen specifications or one selected specification.",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Write numerical results without rendering the integrated figures.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reload completed numerical outputs and redraw reports/figures only.",
    )
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    joined = "::".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(joined.encode("utf-8")).digest()[:4], "little")


def ensure_dirs() -> None:
    for path in (REPORT_DIR, FIG_DIR, SOURCE_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def exact_signflip(values: Iterable[float], reference: float = 0.0) -> dict[str, float]:
    x = np.asarray(list(values), dtype=float) - reference
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "p_greater": np.nan, "p_two_sided": np.nan}
    observed = float(x.mean())
    null = np.asarray(
        [
            float(np.mean(x * np.asarray(signs, dtype=float)))
            for signs in itertools.product((-1.0, 1.0), repeat=x.size)
        ]
    )
    return {
        "n": int(x.size),
        "mean": observed + reference,
        "p_greater": float(np.mean(null >= observed - 1e-12)),
        "p_two_sided": float(np.mean(np.abs(null) >= abs(observed) - 1e-12)),
    }


def patient_equal(run_frame: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    return run_frame.groupby("patient", as_index=False)[value_cols].mean(numeric_only=True)


def norm_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    magnitude = float(np.linalg.norm(x))
    return x / magnitude if magnitude > 0 and np.isfinite(magnitude) else np.zeros_like(x)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = norm_vector(a)
    b = norm_vector(b)
    if not np.any(a) or not np.any(b):
        return np.nan
    return float(np.dot(a, b))


def training_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    return mean, sd


def apply_scale(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) - mean) / sd


def fit_centroid_axis(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    mean, sd = training_scale(x)
    z = apply_scale(x, mean, sd)
    initial = z[y == 0].mean(axis=0)
    post = z[y == 1].mean(axis=0)
    axis = norm_vector(post - initial)
    midpoint = 0.5 * (post + initial)
    return {"mean": mean, "sd": sd, "axis": axis, "midpoint": midpoint}


def axis_score(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    z = apply_scale(x, model["mean"], model["sd"])
    return (z - model["midpoint"]) @ model["axis"]


def event_matrix(
    metrics: pd.DataFrame,
    run: base.RunData,
    value_col: str = "response_baseline_corrected_hz",
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = metrics[(metrics["patient"] == run.patient) & (metrics["run"] == run.run)].copy()
    keys = [
        "patient",
        "run",
        "trial_index",
        "sound_file",
        "voice",
        "pair_id",
        "noun",
        "thematic_role",
        "surface_position",
        "event_bin_1000Hz",
    ]
    pivot = frame.pivot_table(index=keys, columns="unit_id", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(columns=run.unit_ids)
    pivot = pivot.reset_index().sort_values(
        ["thematic_role", "pair_id", "surface_position"]
    ).reset_index(drop=True)
    x = pivot[list(run.unit_ids)].to_numpy(dtype=float)
    meta = pivot[keys].copy()
    meta["position_binary"] = meta["surface_position"].map(
        {"initial": 0, "post_marker": 1}
    ).astype(int)
    if len(meta) != 40 or not np.isfinite(x).all():
        raise RuntimeError(f"Incomplete event matrix for {run.run}: {len(meta)} rows")
    return meta, x


def load_specification(unit_set: str) -> tuple[pd.DataFrame, list[base.RunData], pd.DataFrame]:
    metrics, available = base.load_filtered_metrics(unit_set, ARTIFACT_THRESHOLD)
    runs, audit = base.build_run_data(metrics, available, unit_set)
    if set(audit.loc[audit["included"], "patient"]) != set(PATIENT_COLORS):
        raise RuntimeError(f"Specification {unit_set} does not retain all five patients")
    audit = audit.assign(unit_set=unit_set)
    return metrics, runs, audit


def decode_mp3(path: Path) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-",
    ]
    raw = subprocess.check_output(command)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def estimate_f0(segment: np.ndarray) -> tuple[float, float]:
    frame_size = int(0.040 * SR)
    hop = int(0.010 * SR)
    min_lag = int(SR / 400.0)
    max_lag = int(SR / 50.0)
    estimates: list[float] = []
    total = 0
    for start in range(0, max(1, len(segment) - frame_size + 1), hop):
        frame = segment[start : start + frame_size]
        if len(frame) < frame_size:
            continue
        total += 1
        frame = frame - frame.mean()
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < 0.005:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        denom = corr[0] if corr.size and corr[0] > 0 else 1.0
        local = corr[min_lag : min(max_lag + 1, len(corr))] / denom
        if local.size == 0:
            continue
        lag_offset = int(np.argmax(local))
        peak = float(local[lag_offset])
        if peak >= 0.30:
            estimates.append(SR / float(min_lag + lag_offset))
    voiced_fraction = len(estimates) / total if total else 0.0
    return (float(np.median(estimates)) if estimates else np.nan, voiced_fraction)


def envelope_chunk_duration(audio: np.ndarray, onset_sample: int) -> float:
    frame = int(0.020 * SR)
    hop = int(0.005 * SR)
    start = max(0, onset_sample)
    stop = min(len(audio), onset_sample + int(0.700 * SR))
    segment = audio[start:stop]
    if len(segment) < frame * 2:
        return np.nan
    starts = np.arange(0, len(segment) - frame + 1, hop)
    rms = np.asarray([np.sqrt(np.mean(segment[s : s + frame] ** 2)) for s in starts])
    smooth = np.convolve(rms, np.ones(5) / 5.0, mode="same")
    peak_search = smooth[: max(2, int(0.350 / 0.005))]
    peak_index = int(np.argmax(peak_search))
    peak = float(smooth[peak_index])
    if peak <= 0:
        return np.nan
    threshold = max(0.003, 0.28 * peak)
    earliest = peak_index + int(0.080 / 0.005)
    for index in range(earliest, max(earliest, len(smooth) - 3)):
        if np.all(smooth[index : index + 4] < threshold):
            return float(index * 0.005)
    return np.nan


def build_acoustic_features() -> pd.DataFrame:
    event = pd.read_csv(EVENTS_PATH)
    event = event[event["condition"].isin(["active", "passive"])].copy()
    event = event.drop_duplicates(["sound_file", "noun"]).sort_values(
        ["pair_id", "condition", "surface_position"]
    )
    onset_table = pd.read_csv(MARKER_ONSETS_PATH)
    speech_start = dict(
        zip(onset_table["sound_file"].astype(str), onset_table["speech_start_sec_in_audio"].astype(float))
    )
    audio_root = noun_analysis.audio_root()
    audio_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    function_words = {
        stimulus.sound_file: stimulus.function_word for stimulus in noun_analysis.STIMULI
    }
    for source in event.itertuples(index=False):
        sound_file = str(source.sound_file)
        if sound_file not in audio_cache:
            audio_cache[sound_file] = decode_mp3(audio_root / sound_file)
        audio = audio_cache[sound_file]
        onset_audio = speech_start[sound_file] + float(source.noun_onset_sec_from_sentence_onset)
        left = int(round(onset_audio * SR))
        right = min(len(audio), left + int(round(ACOUSTIC_WINDOW_SEC * SR)))
        segment = audio[max(0, left) : right]
        if len(segment) < int(0.150 * SR):
            raise RuntimeError(f"Short acoustic segment: {sound_file}/{source.noun}")
        rms = float(np.sqrt(np.mean(segment**2)))
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
        frequencies = np.fft.rfftfreq(len(segment), 1.0 / SR)
        centroid = float(np.sum(frequencies * spectrum) / max(np.sum(spectrum), 1e-12))
        zcr = float(np.mean(np.diff(np.signbit(segment)).astype(float)))
        f0, voiced_fraction = estimate_f0(segment)
        duration = envelope_chunk_duration(audio, left)
        rows.append(
            {
                "sound_file": sound_file,
                "noun": str(source.noun),
                "pair_id": int(source.pair_id),
                "voice": str(source.condition),
                "thematic_role": str(source.thematic_role),
                "surface_position": str(source.surface_position),
                "function_word": function_words[sound_file],
                "preceding_marker": (
                    function_words[sound_file]
                    if str(source.surface_position) == "post_marker"
                    else "none"
                ),
                "preceding_marker_present": int(str(source.surface_position) == "post_marker"),
                "serial_onset_sec": float(source.noun_onset_sec_from_sentence_onset),
                "log_rms": float(np.log(rms + 1e-8)),
                "spectral_centroid_hz": centroid,
                "zero_crossing_rate": zcr,
                "f0_hz": f0,
                "voiced_fraction": voiced_fraction,
                "envelope_chunk_duration_sec": duration,
                "acoustic_window_sec": ACOUSTIC_WINDOW_SEC,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 40:
        raise RuntimeError(f"Expected 40 grammatical acoustic observations, found {len(frame)}")
    frame.to_csv(SOURCE_DIR / "stimulus_acoustic_features.csv", index=False)
    return frame


def permuted_labels(meta: pd.DataFrame, pair_signs: np.ndarray | None) -> np.ndarray:
    labels = meta["position_binary"].to_numpy(dtype=int).copy()
    if pair_signs is None:
        return labels
    pairs = meta["pair_id"].to_numpy(dtype=int)
    swap = np.asarray([pair_signs[pair - 1] < 0 for pair in pairs])
    labels[swap] = 1 - labels[swap]
    return labels


def cross_role_ccgp(
    meta: pd.DataFrame,
    x: np.ndarray,
    pair_signs: np.ndarray | None = None,
) -> tuple[pd.DataFrame, float]:
    labels = permuted_labels(meta, pair_signs)
    roles = meta["thematic_role"].astype(str).to_numpy()
    pairs = meta["pair_id"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    for held_pair in range(1, 11):
        for train_role, test_role in (("agent", "patient"), ("patient", "agent")):
            train = (roles == train_role) & (pairs != held_pair)
            test = (roles == test_role) & (pairs == held_pair)
            model = fit_centroid_axis(x[train], labels[train])
            score = axis_score(model, x[test])
            prediction = (score > 0).astype(int)
            for index, predicted, value in zip(np.flatnonzero(test), prediction, score):
                rows.append(
                    {
                        "held_pair": held_pair,
                        "train_role": train_role,
                        "test_role": test_role,
                        "noun": str(meta.iloc[index]["noun"]),
                        "voice": str(meta.iloc[index]["voice"]),
                        "true_position": int(labels[index]),
                        "predicted_position": int(predicted),
                        "axis_score": float(value),
                        "correct": bool(predicted == labels[index]),
                    }
                )
    predictions = pd.DataFrame(rows)
    return predictions, float(predictions["correct"].mean())


def parallelism_folds(
    meta: pd.DataFrame,
    x: np.ndarray,
    pair_signs: np.ndarray | None = None,
) -> pd.DataFrame:
    labels = permuted_labels(meta, pair_signs)
    roles = meta["thematic_role"].astype(str).to_numpy()
    pairs = meta["pair_id"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    for held_pair in range(1, 11):
        training = pairs != held_pair
        mean, sd = training_scale(x[training])
        z = apply_scale(x, mean, sd)
        role_axes: dict[str, np.ndarray] = {}
        for role in ROLES:
            cell = training & (roles == role)
            role_axes[role] = norm_vector(
                z[cell & (labels == 1)].mean(axis=0)
                - z[cell & (labels == 0)].mean(axis=0)
            )
        held_axes: dict[str, np.ndarray] = {}
        for role in ROLES:
            cell = (pairs == held_pair) & (roles == role)
            held_axes[role] = norm_vector(
                z[cell & (labels == 1)].mean(axis=0)
                - z[cell & (labels == 0)].mean(axis=0)
            )
        rows.append(
            {
                "held_pair": held_pair,
                "parallelism_score": cosine(role_axes["agent"], role_axes["patient"]),
                "heldout_cross_projection": 0.5
                * (
                    cosine(held_axes["agent"], role_axes["patient"])
                    + cosine(held_axes["patient"], role_axes["agent"])
                ),
                "agent_heldout_to_patient_axis": cosine(
                    held_axes["agent"], role_axes["patient"]
                ),
                "patient_heldout_to_agent_axis": cosine(
                    held_axes["patient"], role_axes["agent"]
                ),
            }
        )
    return pd.DataFrame(rows)


def static_run_endpoint(
    meta: pd.DataFrame,
    x: np.ndarray,
    pair_signs: np.ndarray | None = None,
) -> dict[str, float]:
    folds = parallelism_folds(meta, x, pair_signs)
    _, accuracy = cross_role_ccgp(meta, x, pair_signs)
    return {
        "parallelism_score": float(folds["parallelism_score"].mean()),
        "heldout_cross_projection": float(folds["heldout_cross_projection"].mean()),
        "ccgp_accuracy": accuracy,
    }


def static_analysis(
    unit_set: str,
    metrics: pd.DataFrame,
    runs: list[base.RunData],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    prediction_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []
    matrices: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for run in runs:
        meta, x = event_matrix(metrics, run)
        matrices[run.run] = (meta, x)
        predictions, accuracy = cross_role_ccgp(meta, x)
        folds = parallelism_folds(meta, x)
        predictions.insert(0, "run", run.run)
        predictions.insert(0, "patient", run.patient)
        predictions.insert(0, "unit_set", unit_set)
        folds.insert(0, "run", run.run)
        folds.insert(0, "patient", run.patient)
        folds.insert(0, "unit_set", unit_set)
        prediction_frames.append(predictions)
        fold_frames.append(folds)
        run_rows.append(
            {
                "unit_set": unit_set,
                "patient": run.patient,
                "run": run.run,
                "n_units": len(run.unit_ids),
                "parallelism_score": float(folds["parallelism_score"].mean()),
                "heldout_cross_projection": float(folds["heldout_cross_projection"].mean()),
                "ccgp_accuracy": accuracy,
            }
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    run_summary = pd.DataFrame(run_rows)
    patient_summary = patient_equal(
        run_summary,
        ["parallelism_score", "heldout_cross_projection", "ccgp_accuracy"],
    )
    patient_summary.insert(0, "unit_set", unit_set)

    sign_patterns = np.asarray(
        list(itertools.product((-1, 1), repeat=10)), dtype=np.int8
    )
    null_rows: list[dict[str, object]] = []
    for permutation_index, signs in enumerate(sign_patterns):
        permuted_runs: list[dict[str, object]] = []
        for run in runs:
            meta, x = matrices[run.run]
            endpoint = static_run_endpoint(meta, x, signs)
            permuted_runs.append({"patient": run.patient, **endpoint})
        run_null = pd.DataFrame(permuted_runs)
        patient_null = run_null.groupby("patient", as_index=False).mean(numeric_only=True)
        null_rows.append(
            {
                "unit_set": unit_set,
                "permutation_index": permutation_index,
                "is_observed": bool(np.all(signs == 1)),
                "patient_mean_parallelism_score": float(
                    patient_null["parallelism_score"].mean()
                ),
                "patient_mean_heldout_cross_projection": float(
                    patient_null["heldout_cross_projection"].mean()
                ),
                "patient_mean_ccgp_accuracy": float(patient_null["ccgp_accuracy"].mean()),
            }
        )
        if (permutation_index + 1) % 256 == 0:
            print(f"{unit_set}: synchronized exact null {permutation_index + 1}/1024", flush=True)
    null = pd.DataFrame(null_rows)
    observed = {
        "parallelism_score": float(patient_summary["parallelism_score"].mean()),
        "heldout_cross_projection": float(
            patient_summary["heldout_cross_projection"].mean()
        ),
        "ccgp_accuracy": float(patient_summary["ccgp_accuracy"].mean()),
    }
    inference: dict[str, object] = {
        "unit_set": unit_set,
        "n_runs": len(run_summary),
        "n_patients": len(patient_summary),
        "n_units": int(run_summary["n_units"].sum()),
    }
    for endpoint, reference in (
        ("parallelism_score", 0.0),
        ("heldout_cross_projection", 0.0),
        ("ccgp_accuracy", 0.5),
    ):
        patient_test = exact_signflip(patient_summary[endpoint], reference)
        null_column = f"patient_mean_{endpoint}"
        item_p = float(np.mean(null[null_column] >= observed[endpoint] - 1e-12))
        inference[endpoint] = {
            "patient_equal_estimate": observed[endpoint],
            "patient_exact_p_greater": patient_test["p_greater"],
            "patient_exact_p_two_sided": patient_test["p_two_sided"],
            "synchronized_pair_permutation_p_greater": item_p,
        }
    return predictions, folds, run_summary, patient_summary, null, inference


def extract_time_responses(
    run: base.RunData,
    meta: pd.DataFrame,
    unit_set: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    archive_path = base.unit_archive_path(run.patient, run.run, unit_set)
    with np.load(archive_path, allow_pickle=True) as archive:
        firing = np.asarray(archive["firing_rate_hz"], dtype=np.float32)
        archive_ids = np.asarray(archive["unit_ids"], dtype=int)
    lookup = {int(unit_id): index for index, unit_id in enumerate(archive_ids)}
    firing = firing[[lookup[int(unit_id)] for unit_id in run.unit_ids]]
    times = np.arange(
        CONTINUOUS_START_MS,
        CONTINUOUS_STOP_MS,
        CONTINUOUS_STEP_MS,
        dtype=int,
    )
    continuous = np.full(
        (len(meta), len(times), len(run.unit_ids)), np.nan, dtype=np.float32
    )
    fixed = {
        f"{left}-{right}": np.full(
            (len(meta), len(run.unit_ids)), np.nan, dtype=np.float32
        )
        for left, right in FIXED_WINDOWS_MS
    }
    for row_index, row in meta.iterrows():
        event_bin = int(row["event_bin_1000Hz"])
        if event_bin - 500 < 0 or event_bin + 800 > firing.shape[1]:
            raise RuntimeError(f"Time window outside recording for {run.run}")
        baseline = firing[:, event_bin - 500 : event_bin].mean(axis=1)
        for label, (left, right) in zip(fixed, FIXED_WINDOWS_MS):
            fixed[label][row_index] = (
                firing[:, event_bin + left : event_bin + right].mean(axis=1) - baseline
            )
        for time_index, left in enumerate(times):
            right = left + CONTINUOUS_STEP_MS
            continuous[row_index, time_index] = (
                firing[:, event_bin + left : event_bin + right].mean(axis=1) - baseline
            )
    if any(not np.isfinite(value).all() for value in fixed.values()):
        raise RuntimeError(f"Non-finite fixed time response for {run.run}")
    if not np.isfinite(continuous).all():
        raise RuntimeError(f"Non-finite continuous response for {run.run}")
    return fixed, continuous, times


def cross_role_projected_trajectory(
    meta: pd.DataFrame,
    static_x: np.ndarray,
    continuous_x: np.ndarray,
) -> np.ndarray:
    labels = meta["position_binary"].to_numpy(dtype=int)
    roles = meta["thematic_role"].astype(str).to_numpy()
    pairs = meta["pair_id"].to_numpy(dtype=int)
    margins: list[np.ndarray] = []
    for held_pair in range(1, 11):
        for train_role, test_role in (("agent", "patient"), ("patient", "agent")):
            train = (roles == train_role) & (pairs != held_pair)
            test = (roles == test_role) & (pairs == held_pair)
            model = fit_centroid_axis(static_x[train], labels[train])
            score = axis_score(model, continuous_x[test])
            test_labels = labels[test]
            margin = score[test_labels == 1].mean(axis=0) - score[test_labels == 0].mean(axis=0)
            margins.append(np.asarray(margin, dtype=float))
    return np.mean(np.vstack(margins), axis=0)


def exact_maxstat_one_sided(patient_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(patient_matrix, dtype=float)
    observed = values.mean(axis=0)
    null = np.asarray(
        [
            (values * np.asarray(signs, dtype=float)[:, None]).mean(axis=0)
            for signs in itertools.product((-1.0, 1.0), repeat=values.shape[0])
        ]
    )
    raw = np.mean(null >= observed[None, :] - 1e-12, axis=0)
    null_max = np.max(null, axis=1)
    fwer = np.asarray(
        [np.mean(null_max >= value - 1e-12) for value in observed], dtype=float
    )
    return raw, fwer


def significant_segments(
    times: np.ndarray,
    p_values: np.ndarray,
    alpha: float = POINT_ALPHA,
) -> pd.DataFrame:
    mask = np.asarray(p_values) < alpha
    rows: list[dict[str, object]] = []
    start: int | None = None
    for index in range(len(mask) + 1):
        active = index < len(mask) and bool(mask[index])
        if active and start is None:
            start = index
        if not active and start is not None:
            stop = index
            duration = int((stop - start) * CONTINUOUS_STEP_MS)
            if duration > 5:
                rows.append(
                    {
                        "start_ms": int(times[start]),
                        "stop_ms": int(times[stop - 1] + CONTINUOUS_STEP_MS),
                        "duration_ms": duration,
                        "minimum_point_p": float(np.min(p_values[start:stop])),
                    }
                )
            start = None
    return pd.DataFrame(
        rows,
        columns=["start_ms", "stop_ms", "duration_ms", "minimum_point_p"],
    )


def time_analysis(
    unit_set: str,
    metrics: pd.DataFrame,
    runs: list[base.RunData],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    fixed_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    for run in runs:
        meta, static_x = event_matrix(metrics, run)
        fixed, continuous, times = extract_time_responses(run, meta, unit_set)
        for window, x in fixed.items():
            _, accuracy = cross_role_ccgp(meta, x)
            folds = parallelism_folds(meta, x)
            fixed_rows.append(
                {
                    "unit_set": unit_set,
                    "patient": run.patient,
                    "run": run.run,
                    "window_ms": window,
                    "parallelism_score": float(folds["parallelism_score"].mean()),
                    "heldout_cross_projection": float(
                        folds["heldout_cross_projection"].mean()
                    ),
                    "ccgp_accuracy": accuracy,
                }
            )
        trajectory = cross_role_projected_trajectory(meta, static_x, continuous)
        for time_ms, value in zip(times, trajectory):
            trajectory_rows.append(
                {
                    "unit_set": unit_set,
                    "patient": run.patient,
                    "run": run.run,
                    "time_ms": int(time_ms),
                    "heldout_cross_role_projection": float(value),
                }
            )
    fixed_run = pd.DataFrame(fixed_rows)
    fixed_patient = fixed_run.groupby(
        ["unit_set", "patient", "window_ms"], as_index=False
    )[["parallelism_score", "heldout_cross_projection", "ccgp_accuracy"]].mean()
    fixed_stats_rows: list[dict[str, object]] = []
    for endpoint, reference in (
        ("parallelism_score", 0.0),
        ("heldout_cross_projection", 0.0),
        ("ccgp_accuracy", 0.5),
    ):
        pivot = fixed_patient.pivot(index="patient", columns="window_ms", values=endpoint)
        order = [f"{left}-{right}" for left, right in FIXED_WINDOWS_MS]
        matrix = pivot.reindex(index=sorted(pivot.index), columns=order).to_numpy(dtype=float)
        raw, fwer = exact_maxstat_one_sided(matrix - reference)
        for index, window in enumerate(order):
            fixed_stats_rows.append(
                {
                    "unit_set": unit_set,
                    "endpoint": endpoint,
                    "window_ms": window,
                    "patient_equal_estimate": float(matrix[:, index].mean()),
                    "patient_exact_p_greater": float(raw[index]),
                    "patient_window_maxstat_p_fwer_greater": float(fwer[index]),
                }
            )
    fixed_stats = pd.DataFrame(fixed_stats_rows)

    trajectory_run = pd.DataFrame(trajectory_rows)
    trajectory_patient = trajectory_run.groupby(
        ["unit_set", "patient", "time_ms"], as_index=False
    )["heldout_cross_role_projection"].mean()
    trajectory_pivot = trajectory_patient.pivot(
        index="patient", columns="time_ms", values="heldout_cross_role_projection"
    ).sort_index(axis=1)
    trajectory_matrix = trajectory_pivot.to_numpy(dtype=float)
    raw_p, fwer_p = exact_maxstat_one_sided(trajectory_matrix)
    trajectory_stats = pd.DataFrame(
        {
            "unit_set": unit_set,
            "time_ms": trajectory_pivot.columns.to_numpy(dtype=int),
            "patient_equal_projection": trajectory_matrix.mean(axis=0),
            "patient_sem_projection": trajectory_matrix.std(axis=0, ddof=1)
            / math.sqrt(trajectory_matrix.shape[0]),
            "patient_exact_point_p_greater": raw_p,
            "patient_time_maxstat_p_fwer_greater": fwer_p,
            "passes_legacy_point_threshold": raw_p < POINT_ALPHA,
        }
    )
    segments = significant_segments(
        trajectory_stats["time_ms"].to_numpy(),
        trajectory_stats["patient_exact_point_p_greater"].to_numpy(),
    )
    if len(segments):
        segments.insert(0, "unit_set", unit_set)
    summary = {
        "unit_set": unit_set,
        "fixed_windows": fixed_stats.to_dict(orient="records"),
        "continuous_grid_ms": [
            CONTINUOUS_START_MS,
            CONTINUOUS_STOP_MS,
            CONTINUOUS_STEP_MS,
        ],
        "legacy_point_alpha": POINT_ALPHA,
        "minimum_patient_exact_p_resolution": 1 / (2 ** len(trajectory_pivot)),
        "legacy_segments_gt_5ms": int(len(segments)),
        "time_fwer_significant_bins": int(
            (trajectory_stats["patient_time_maxstat_p_fwer_greater"] < 0.05).sum()
        ),
    }
    return fixed_run, fixed_patient, fixed_stats, trajectory_stats, segments, summary


ACOUSTIC_COLUMNS = [
    "log_rms",
    "spectral_centroid_hz",
    "zero_crossing_rate",
    "f0_hz",
    "voiced_fraction",
    "envelope_chunk_duration_sec",
]


def ridge_fit_predict(
    z_train: np.ndarray,
    x_train: np.ndarray,
    z_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    z_train = np.asarray(z_train, dtype=float)
    z_test = np.asarray(z_test, dtype=float)
    x_train = np.asarray(x_train, dtype=float)
    median = np.nanmedian(z_train, axis=0)
    median[~np.isfinite(median)] = 0.0
    z_train = np.where(np.isfinite(z_train), z_train, median)
    z_test = np.where(np.isfinite(z_test), z_test, median)
    mean = z_train.mean(axis=0)
    sd = z_train.std(axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    ztr = (z_train - mean) / sd
    zte = (z_test - mean) / sd
    design = np.column_stack([np.ones(len(ztr)), ztr])
    design_test = np.column_stack([np.ones(len(zte)), zte])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ x_train)
    return design_test @ beta


def choose_ridge_alpha(
    z: np.ndarray,
    x: np.ndarray,
    groups: np.ndarray,
) -> float:
    errors: dict[float, list[float]] = {alpha: [] for alpha in RIDGE_ALPHAS}
    for group in np.unique(groups):
        validation = groups == group
        training = ~validation
        for alpha in RIDGE_ALPHAS:
            prediction = ridge_fit_predict(
                z[training], x[training], z[validation], alpha
            )
            errors[alpha].append(float(np.mean((x[validation] - prediction) ** 2)))
    return min(RIDGE_ALPHAS, key=lambda alpha: (np.mean(errors[alpha]), alpha))


def crossfit_residualize(
    x_train: np.ndarray,
    z_train: np.ndarray,
    groups: np.ndarray,
    x_test: np.ndarray,
    z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    alpha = choose_ridge_alpha(z_train, x_train, groups)
    predicted_train = np.full_like(x_train, np.nan, dtype=float)
    for group in np.unique(groups):
        validation = groups == group
        training = ~validation
        predicted_train[validation] = ridge_fit_predict(
            z_train[training], x_train[training], z_train[validation], alpha
        )
    predicted_test = ridge_fit_predict(z_train, x_train, z_test, alpha)
    if not np.isfinite(predicted_train).all() or not np.isfinite(predicted_test).all():
        raise RuntimeError("Cross-fitted residualization produced non-finite values")
    return x_train - predicted_train, x_test - predicted_test, alpha


def matched_control_covariates(
    meta: pd.DataFrame,
    acoustic: pd.DataFrame,
) -> pd.DataFrame:
    joined = meta[["sound_file", "noun"]].merge(
        acoustic,
        on=["sound_file", "noun"],
        how="left",
        validate="many_to_one",
    )
    if joined["log_rms"].isna().any():
        raise RuntimeError("Acoustic covariate join failed")
    return joined


def controlled_cross_role_predictions(
    meta: pd.DataFrame,
    response_x: np.ndarray,
    baseline_x: np.ndarray,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    roles = meta["thematic_role"].astype(str).to_numpy()
    pairs = meta["pair_id"].to_numpy(dtype=int)
    labels = meta["position_binary"].to_numpy(dtype=int)
    acoustic_z = covariates[ACOUSTIC_COLUMNS].to_numpy(dtype=float)
    acoustic_timing_z = covariates[ACOUSTIC_COLUMNS + ["serial_onset_sec"]].to_numpy(
        dtype=float
    )
    rows: list[dict[str, object]] = []
    for held_pair in range(1, 11):
        for train_role, test_role in (("agent", "patient"), ("patient", "agent")):
            train = (roles == train_role) & (pairs != held_pair)
            test = (roles == test_role) & (pairs == held_pair)
            specifications: list[tuple[str, np.ndarray, np.ndarray, float | None]] = [
                ("primary", response_x[train], response_x[test], None),
                ("pre_noun_baseline", baseline_x[train], baseline_x[test], None),
            ]
            residual_train, residual_test, alpha = crossfit_residualize(
                response_x[train],
                acoustic_z[train],
                pairs[train],
                response_x[test],
                acoustic_z[test],
            )
            specifications.append(
                ("acoustic_residualized", residual_train, residual_test, alpha)
            )
            residual_train, residual_test, alpha = crossfit_residualize(
                response_x[train],
                acoustic_timing_z[train],
                pairs[train],
                response_x[test],
                acoustic_timing_z[test],
            )
            specifications.append(
                (
                    "acoustic_plus_serial_time_residualized",
                    residual_train,
                    residual_test,
                    alpha,
                )
            )
            for specification, x_train, x_test, alpha in specifications:
                model = fit_centroid_axis(x_train, labels[train])
                score = axis_score(model, x_test)
                prediction = (score > 0).astype(int)
                for index, predicted, value in zip(np.flatnonzero(test), prediction, score):
                    rows.append(
                        {
                            "held_pair": held_pair,
                            "train_role": train_role,
                            "test_role": test_role,
                            "specification": specification,
                            "selected_ridge_alpha": alpha,
                            "noun": str(meta.iloc[index]["noun"]),
                            "true_position": int(labels[index]),
                            "predicted_position": int(predicted),
                            "axis_score": float(value),
                            "correct": bool(predicted == labels[index]),
                        }
                    )
    return pd.DataFrame(rows)


def vif_table(design: pd.DataFrame, model: str) -> pd.DataFrame:
    x = design.to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for index, column in enumerate(design.columns):
        if column == "intercept" or np.std(x[:, index]) < 1e-12:
            vif = np.nan
        else:
            y = x[:, index]
            others = np.delete(x, index, axis=1)
            beta, *_ = np.linalg.lstsq(others, y, rcond=None)
            residual = y - others @ beta
            tss = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - float(np.sum(residual**2)) / max(tss, 1e-12)
            vif = np.inf if r2 >= 1.0 - 1e-10 else 1.0 / max(1e-12, 1.0 - r2)
        rows.append({"model": model, "term": column, "vif": vif})
    return pd.DataFrame(rows)


def design_estimability_audit(acoustic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    continuous = acoustic[ACOUSTIC_COLUMNS].copy()
    for column in continuous:
        continuous[column] = continuous[column].fillna(continuous[column].median())
        sd = continuous[column].std(ddof=1)
        continuous[column] = (
            continuous[column] - continuous[column].mean()
        ) / (sd if sd > 1e-8 else 1.0)
    position = acoustic["surface_position"].eq("post_marker").astype(float).rename("position_post")
    voice = acoustic["voice"].eq("passive").astype(float).rename("voice_passive")
    role = acoustic["thematic_role"].eq("patient").astype(float).rename("role_patient")
    marker = pd.get_dummies(
        acoustic["preceding_marker"], prefix="marker", drop_first=True, dtype=float
    )
    models: dict[str, pd.DataFrame] = {
        "position_only": pd.DataFrame({"intercept": 1.0, "position_post": position}),
        "position_plus_acoustic": pd.concat(
            [pd.DataFrame({"intercept": 1.0, "position_post": position}), continuous],
            axis=1,
        ),
        "position_plus_acoustic_and_serial_time": pd.concat(
            [
                pd.DataFrame({"intercept": 1.0, "position_post": position}),
                continuous,
                acoustic[["serial_onset_sec"]].reset_index(drop=True),
            ],
            axis=1,
        ),
        "position_plus_structural_context": pd.concat(
            [
                pd.DataFrame(
                    {
                        "intercept": 1.0,
                        "position_post": position,
                        "voice_passive": voice,
                        "role_patient": role,
                        "marker_present": acoustic["preceding_marker_present"].astype(float),
                    }
                ),
                marker.reset_index(drop=True),
            ],
            axis=1,
        ),
    }
    audit_rows: list[dict[str, object]] = []
    vif_frames: list[pd.DataFrame] = []
    for model, design in models.items():
        design = design.astype(float)
        rank = int(np.linalg.matrix_rank(design.to_numpy()))
        position_vector = design["position_post"].to_numpy(dtype=float)
        nuisance = design.drop(columns="position_post").to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(nuisance, position_vector, rcond=None)
        residual = position_vector - nuisance @ beta
        residual_fraction = float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(position_vector - position_vector.mean()), 1e-12)
        )
        audit_rows.append(
            {
                "model": model,
                "n_observations": len(design),
                "n_columns": design.shape[1],
                "matrix_rank": rank,
                "full_column_rank": bool(rank == design.shape[1]),
                "condition_number": float(np.linalg.cond(design.to_numpy())),
                "position_residual_fraction_after_nuisance": residual_fraction,
                "position_estimable_beyond_nuisance": bool(residual_fraction > 1e-8),
            }
        )
        vif_frames.append(vif_table(design, model))
    return pd.DataFrame(audit_rows), pd.concat(vif_frames, ignore_index=True)


def control_analysis(
    unit_set: str,
    metrics: pd.DataFrame,
    runs: list[base.RunData],
    acoustic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    prediction_frames: list[pd.DataFrame] = []
    for run in runs:
        meta, response_x = event_matrix(metrics, run, "response_baseline_corrected_hz")
        baseline_meta, baseline_x = event_matrix(metrics, run, "baseline_hz")
        if not meta[["sound_file", "noun"]].equals(baseline_meta[["sound_file", "noun"]]):
            raise RuntimeError(f"Response/baseline event order differs for {run.run}")
        covariates = matched_control_covariates(meta, acoustic)
        predictions = controlled_cross_role_predictions(
            meta, response_x, baseline_x, covariates
        )
        predictions.insert(0, "run", run.run)
        predictions.insert(0, "patient", run.patient)
        predictions.insert(0, "unit_set", unit_set)
        prediction_frames.append(predictions)
    prediction = pd.concat(prediction_frames, ignore_index=True)
    run_summary = prediction.groupby(
        ["unit_set", "patient", "run", "specification"], as_index=False
    ).agg(
        accuracy=("correct", "mean"),
        median_selected_ridge_alpha=("selected_ridge_alpha", "median"),
    )
    patient_summary = run_summary.groupby(
        ["unit_set", "patient", "specification"], as_index=False
    )["accuracy"].mean()
    inference: dict[str, object] = {"unit_set": unit_set}
    for specification, frame in patient_summary.groupby("specification"):
        test = exact_signflip(frame["accuracy"], 0.5)
        inference[str(specification)] = {
            "patient_equal_accuracy": float(frame["accuracy"].mean()),
            "patient_exact_p_greater": test["p_greater"],
            "patient_exact_p_two_sided": test["p_two_sided"],
        }
    return prediction, run_summary, patient_summary, inference


def gini(values: np.ndarray) -> float:
    x = np.sort(np.abs(np.asarray(values, dtype=float)))
    if x.size == 0 or x.sum() <= 0:
        return np.nan
    index = np.arange(1, x.size + 1)
    return float(np.sum((2 * index - x.size - 1) * x) / (x.size * x.sum()))


def participation_metrics(weights: np.ndarray) -> dict[str, float]:
    energy = np.square(np.asarray(weights, dtype=float))
    total = float(energy.sum())
    if total <= 0:
        return {
            "participation_ratio": np.nan,
            "participation_ratio_fraction": np.nan,
            "gini_absolute_loading": np.nan,
            "units_for_90pct_energy": np.nan,
        }
    participation = total**2 / float(np.square(energy).sum())
    ordered = np.sort(energy)[::-1]
    units90 = int(np.searchsorted(np.cumsum(ordered) / total, 0.90) + 1)
    return {
        "participation_ratio": float(participation),
        "participation_ratio_fraction": float(participation / len(energy)),
        "gini_absolute_loading": gini(weights),
        "units_for_90pct_energy": units90,
    }


def pairwise_cosine_median(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return np.nan
    values = [cosine(a, b) for a, b in itertools.combinations(vectors, 2)]
    return float(np.nanmedian(values))


def unit_contribution_analysis(
    unit_set: str,
    metrics: pd.DataFrame,
    runs: list[base.RunData],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    loading_rows: list[dict[str, object]] = []
    run_metric_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    for run in runs:
        meta, x = event_matrix(metrics, run)
        labels = meta["position_binary"].to_numpy(dtype=int)
        roles = meta["thematic_role"].astype(str).to_numpy()
        pairs = meta["pair_id"].to_numpy(dtype=int)
        axes_by_role: dict[str, list[np.ndarray]] = {role: [] for role in ROLES}
        baseline_correct: list[float] = []
        top_correct: dict[float, list[float]] = {
            fraction: [] for fraction in ABLATION_FRACTIONS
        }
        random_correct: dict[float, list[float]] = {
            fraction: [] for fraction in ABLATION_FRACTIONS
        }
        for held_pair in range(1, 11):
            for train_role, test_role in (("agent", "patient"), ("patient", "agent")):
                train = (roles == train_role) & (pairs != held_pair)
                test = (roles == test_role) & (pairs == held_pair)
                model = fit_centroid_axis(x[train], labels[train])
                axes_by_role[train_role].append(model["axis"])
                prediction = (axis_score(model, x[test]) > 0).astype(int)
                baseline_correct.extend((prediction == labels[test]).astype(float).tolist())
                order = np.argsort(np.abs(model["axis"]))[::-1]
                for fraction in ABLATION_FRACTIONS:
                    k = max(1, int(math.ceil(fraction * x.shape[1])))
                    keep_top = np.ones(x.shape[1], dtype=bool)
                    keep_top[order[:k]] = False
                    top_model = fit_centroid_axis(x[train][:, keep_top], labels[train])
                    top_prediction = (
                        axis_score(top_model, x[test][:, keep_top]) > 0
                    ).astype(int)
                    top_correct[fraction].extend(
                        (top_prediction == labels[test]).astype(float).tolist()
                    )
                    rng = np.random.default_rng(
                        stable_seed(
                            "matched-random-ablation",
                            unit_set,
                            run.run,
                            held_pair,
                            train_role,
                            fraction,
                        )
                    )
                    draw_accuracy: list[float] = []
                    for _ in range(RANDOM_ABLATION_DRAWS):
                        removed = rng.choice(x.shape[1], size=k, replace=False)
                        keep_random = np.ones(x.shape[1], dtype=bool)
                        keep_random[removed] = False
                        random_model = fit_centroid_axis(
                            x[train][:, keep_random], labels[train]
                        )
                        random_prediction = (
                            axis_score(random_model, x[test][:, keep_random]) > 0
                        ).astype(int)
                        draw_accuracy.append(
                            float(np.mean(random_prediction == labels[test]))
                        )
                    random_correct[fraction].append(float(np.mean(draw_accuracy)))
        mean_axes = {
            role: norm_vector(np.mean(np.vstack(axes_by_role[role]), axis=0))
            for role in ROLES
        }
        common_axis = norm_vector(mean_axes["agent"] + mean_axes["patient"])
        metrics_out = participation_metrics(common_axis)
        run_metric_rows.append(
            {
                "unit_set": unit_set,
                "patient": run.patient,
                "run": run.run,
                "n_units": len(run.unit_ids),
                "agent_fold_axis_stability": pairwise_cosine_median(
                    axes_by_role["agent"]
                ),
                "patient_fold_axis_stability": pairwise_cosine_median(
                    axes_by_role["patient"]
                ),
                "agent_patient_mean_axis_cosine": cosine(
                    mean_axes["agent"], mean_axes["patient"]
                ),
                **metrics_out,
            }
        )
        for unit_id, agent_weight, patient_weight, common_weight in zip(
            run.unit_ids,
            mean_axes["agent"],
            mean_axes["patient"],
            common_axis,
        ):
            loading_rows.append(
                {
                    "unit_set": unit_set,
                    "patient": run.patient,
                    "run": run.run,
                    "unit_id": int(unit_id),
                    "agent_axis_loading": float(agent_weight),
                    "patient_axis_loading": float(patient_weight),
                    "common_axis_loading": float(common_weight),
                    "absolute_common_loading": float(abs(common_weight)),
                }
            )
        baseline_accuracy = float(np.mean(baseline_correct))
        for fraction in ABLATION_FRACTIONS:
            top_accuracy = float(np.mean(top_correct[fraction]))
            random_accuracy = float(np.mean(random_correct[fraction]))
            ablation_rows.append(
                {
                    "unit_set": unit_set,
                    "patient": run.patient,
                    "run": run.run,
                    "fraction_removed": fraction,
                    "baseline_accuracy": baseline_accuracy,
                    "top_loading_removed_accuracy": top_accuracy,
                    "matched_random_removed_accuracy": random_accuracy,
                    "top_minus_random_accuracy": top_accuracy - random_accuracy,
                    "baseline_minus_top_accuracy": baseline_accuracy - top_accuracy,
                    "baseline_minus_random_accuracy": baseline_accuracy - random_accuracy,
                }
            )
    loadings = pd.DataFrame(loading_rows)
    run_metrics = pd.DataFrame(run_metric_rows)
    ablation_run = pd.DataFrame(ablation_rows)
    ablation_patient = ablation_run.groupby(
        ["unit_set", "patient", "fraction_removed"], as_index=False
    )[
        [
            "baseline_accuracy",
            "top_loading_removed_accuracy",
            "matched_random_removed_accuracy",
            "top_minus_random_accuracy",
            "baseline_minus_top_accuracy",
            "baseline_minus_random_accuracy",
        ]
    ].mean()
    inference: dict[str, object] = {
        "unit_set": unit_set,
        "patient_equal_participation_ratio_fraction": float(
            run_metrics.groupby("patient")["participation_ratio_fraction"].mean().mean()
        ),
        "patient_equal_gini_absolute_loading": float(
            run_metrics.groupby("patient")["gini_absolute_loading"].mean().mean()
        ),
    }
    for fraction, frame in ablation_patient.groupby("fraction_removed"):
        random_minus_top = -frame["top_minus_random_accuracy"].to_numpy(dtype=float)
        test = exact_signflip(random_minus_top, 0.0)
        inference[f"ablation_{fraction:.2f}"] = {
            "patient_equal_baseline_accuracy": float(frame["baseline_accuracy"].mean()),
            "patient_equal_top_removed_accuracy": float(
                frame["top_loading_removed_accuracy"].mean()
            ),
            "patient_equal_random_removed_accuracy": float(
                frame["matched_random_removed_accuracy"].mean()
            ),
            "patient_equal_random_minus_top": float(random_minus_top.mean()),
            "patient_exact_p_random_greater_than_top": test["p_greater"],
        }
    return loadings, run_metrics, ablation_run, ablation_patient, inference


def extract_scrambled_response(
    run: base.RunData,
    unit_set: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    events = pd.read_csv(EVENTS_PATH)
    meta = events[
        (events["run"] == run.run) & (events["condition"] == "scrambled")
    ].copy()
    meta = meta.sort_values(["pair_id", "char_index"]).reset_index(drop=True)
    if len(meta) != 10:
        raise RuntimeError(f"Expected ten scrambled noun events for {run.run}")
    archive_path = base.unit_archive_path(run.patient, run.run, unit_set)
    with np.load(archive_path, allow_pickle=True) as archive:
        firing = np.asarray(archive["firing_rate_hz"], dtype=np.float32)
        archive_ids = np.asarray(archive["unit_ids"], dtype=int)
    lookup = {int(unit_id): index for index, unit_id in enumerate(archive_ids)}
    firing = firing[[lookup[int(unit_id)] for unit_id in run.unit_ids]]
    x = np.full((len(meta), len(run.unit_ids)), np.nan, dtype=np.float32)
    for index, row in meta.iterrows():
        event_bin = int(row["event_bin_1000Hz"])
        if event_bin - 500 < 0 or event_bin + 800 > firing.shape[1]:
            raise RuntimeError(f"Scrambled window outside recording for {run.run}")
        baseline = firing[:, event_bin - 500 : event_bin].mean(axis=1)
        x[index] = firing[:, event_bin : event_bin + 800].mean(axis=1) - baseline
    return meta, x


def scrambled_analysis(
    unit_set: str,
    metrics: pd.DataFrame,
    runs: list[base.RunData],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pair_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    recorded_pairs = (1, 3, 5, 7, 9)
    for run in runs:
        grammar_meta, grammar_x = event_matrix(metrics, run)
        scrambled_meta, scrambled_x = extract_scrambled_response(run, unit_set)
        grammar_labels = grammar_meta["position_binary"].to_numpy(dtype=int)
        grammar_pairs = grammar_meta["pair_id"].to_numpy(dtype=int)
        crossfit_scrambled_scores: list[float] = []
        crossfit_char_indices: list[int] = []
        for held_pair in recorded_pairs:
            train = grammar_pairs != held_pair
            model = fit_centroid_axis(grammar_x[train], grammar_labels[train])
            grammar_test = grammar_pairs == held_pair
            grammar_score = axis_score(model, grammar_x[grammar_test])
            grammar_test_labels = grammar_labels[grammar_test]
            grammar_margin = float(
                grammar_score[grammar_test_labels == 1].mean()
                - grammar_score[grammar_test_labels == 0].mean()
            )
            scrambled_test = scrambled_meta["pair_id"].to_numpy(dtype=int) == held_pair
            pair_meta = scrambled_meta.loc[scrambled_test].reset_index(drop=True)
            pair_x = scrambled_x[scrambled_test]
            score = axis_score(model, pair_x)
            order = np.argsort(pair_meta["char_index"].to_numpy(dtype=int))
            scrambled_margin = float(score[order[-1]] - score[order[0]])
            crossfit_scrambled_scores.extend(score.tolist())
            crossfit_char_indices.extend(pair_meta["char_index"].astype(int).tolist())
            pair_rows.append(
                {
                    "unit_set": unit_set,
                    "patient": run.patient,
                    "run": run.run,
                    "pair_id": held_pair,
                    "earlier_noun": str(pair_meta.iloc[order[0]]["noun"]),
                    "later_noun": str(pair_meta.iloc[order[-1]]["noun"]),
                    "earlier_char_index": int(pair_meta.iloc[order[0]]["char_index"]),
                    "later_char_index": int(pair_meta.iloc[order[-1]]["char_index"]),
                    "grammar_post_minus_initial_projection": grammar_margin,
                    "scrambled_later_minus_earlier_projection": scrambled_margin,
                }
            )
        rho = float(
            stats.spearmanr(crossfit_char_indices, crossfit_scrambled_scores).correlation
        )
        run_pair_frame = pd.DataFrame(
            [row for row in pair_rows if row["run"] == run.run]
        )
        run_rows.append(
            {
                "unit_set": unit_set,
                "patient": run.patient,
                "run": run.run,
                "n_scrambled_pairs": len(run_pair_frame),
                "grammar_post_minus_initial_projection": float(
                    run_pair_frame["grammar_post_minus_initial_projection"].mean()
                ),
                "scrambled_later_minus_earlier_projection": float(
                    run_pair_frame["scrambled_later_minus_earlier_projection"].mean()
                ),
                "scrambled_char_index_spearman": rho,
            }
        )
    pair_frame = pd.DataFrame(pair_rows)
    run_summary = pd.DataFrame(run_rows)
    patient_summary = run_summary.groupby(
        ["unit_set", "patient"], as_index=False
    )[
        [
            "grammar_post_minus_initial_projection",
            "scrambled_later_minus_earlier_projection",
            "scrambled_char_index_spearman",
        ]
    ].mean()
    patient_summary["scrambled_minus_grammar_projection"] = (
        patient_summary["scrambled_later_minus_earlier_projection"]
        - patient_summary["grammar_post_minus_initial_projection"]
    )
    grammar_test = exact_signflip(
        patient_summary["grammar_post_minus_initial_projection"], 0.0
    )
    scrambled_test = exact_signflip(
        patient_summary["scrambled_later_minus_earlier_projection"], 0.0
    )
    rho_test = exact_signflip(patient_summary["scrambled_char_index_spearman"], 0.0)

    null_values: list[float] = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(recorded_pairs)):
        sign_lookup = dict(zip(recorded_pairs, signs))
        signed = pair_frame.copy()
        signed["signed_margin"] = signed.apply(
            lambda row: row["scrambled_later_minus_earlier_projection"]
            * sign_lookup[int(row["pair_id"])],
            axis=1,
        )
        patient_null = signed.groupby("patient")["signed_margin"].mean()
        null_values.append(float(patient_null.mean()))
    observed = float(
        patient_summary["scrambled_later_minus_earlier_projection"].mean()
    )
    stimulus_p = float(np.mean(np.asarray(null_values) >= observed - 1e-12))
    inference = {
        "unit_set": unit_set,
        "recorded_scrambled_pairs": len(recorded_pairs),
        "grammar": {
            "patient_equal_projection": float(
                patient_summary["grammar_post_minus_initial_projection"].mean()
            ),
            "patient_exact_p_greater": grammar_test["p_greater"],
        },
        "scrambled": {
            "patient_equal_later_minus_earlier_projection": observed,
            "patient_exact_p_greater": scrambled_test["p_greater"],
            "synchronized_five_pair_permutation_p_greater": stimulus_p,
            "patient_equal_char_index_spearman": float(
                patient_summary["scrambled_char_index_spearman"].mean()
            ),
            "char_index_patient_exact_p_greater": rho_test["p_greater"],
        },
        "interpretation": "supplementary serial-position diagnostic; five scrambled trials per run and role/identity/order remain confounded",
    }
    return pair_frame, run_summary, patient_summary, inference


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def p_label(value: float) -> str:
    return f"P={value:.3f}" if value >= 0.001 else f"P={value:.1e}"


def patient_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    order: list[str],
    connect: bool = True,
) -> None:
    for patient, cell in frame.groupby("patient"):
        indexed = cell.set_index(x_column).reindex(order)
        x = np.arange(len(order), dtype=float)
        y = indexed[y_column].to_numpy(dtype=float)
        if connect:
            ax.plot(x, y, color=PATIENT_COLORS[str(patient)], alpha=0.55, lw=0.8)
        ax.scatter(
            x,
            y,
            color=PATIENT_COLORS[str(patient)],
            s=18,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    means = frame.groupby(x_column)[y_column].mean().reindex(order).to_numpy(dtype=float)
    ax.scatter(
        np.arange(len(order)),
        means,
        marker="_",
        s=210,
        linewidth=2.0,
        color="black",
        zorder=4,
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def plot_main_figure(
    frames: dict[str, dict[str, pd.DataFrame]],
    summaries: dict[str, dict[str, object]],
    anchor_patient: pd.DataFrame,
    anchor_summary: pd.DataFrame,
) -> None:
    configure_style()
    primary = frames[PRIMARY_UNIT_SET]
    inference = summaries[PRIMARY_UNIT_SET]
    figure, axes = plt.subplots(3, 2, figsize=(7.09, 6.25))
    figure.subplots_adjust(hspace=0.55, wspace=0.36)

    ax = axes[0, 0]
    static_patient = primary["static_patient"].copy()
    long = static_patient.melt(
        id_vars=["unit_set", "patient"],
        value_vars=["parallelism_score", "heldout_cross_projection", "ccgp_accuracy"],
        var_name="endpoint",
        value_name="value",
    )
    long.loc[long["endpoint"].eq("ccgp_accuracy"), "value"] -= 0.5
    order = ["parallelism_score", "heldout_cross_projection", "ccgp_accuracy"]
    patient_points(ax, long, "endpoint", "value", order)
    ax.axhline(0, color=COLORS["gray"], ls="--", lw=0.7)
    labels = ["PS", "Held-out\nprojection", "CCGP − chance"]
    p_values = [
        inference["static"][endpoint]["patient_exact_p_greater"]
        for endpoint in order
    ]
    ax.set_xticks(range(3), [f"{label}\n{p_label(p)}" for label, p in zip(labels, p_values)])
    ax.set_ylabel("Patient endpoint")
    ax.set_title("Shared position transformation", loc="left")
    panel_label(ax, "a")

    ax = axes[0, 1]
    trajectory = primary["trajectory_stats"]
    ax.plot(
        trajectory["time_ms"],
        trajectory["patient_equal_projection"],
        color=COLORS["blue"],
        lw=1.4,
        label="cross-role projection",
    )
    ax.fill_between(
        trajectory["time_ms"].to_numpy(dtype=float),
        (
            trajectory["patient_equal_projection"]
            - trajectory["patient_sem_projection"]
        ).to_numpy(dtype=float),
        (
            trajectory["patient_equal_projection"]
            + trajectory["patient_sem_projection"]
        ).to_numpy(dtype=float),
        color=COLORS["blue"],
        alpha=0.18,
        lw=0,
    )
    ax.axhline(0, color=COLORS["gray"], ls="--", lw=0.7)
    ax.axvline(0, color=COLORS["black"], lw=0.7)
    ax.set_xlim(CONTINUOUS_START_MS, CONTINUOUS_STOP_MS)
    ax.set_xlabel("Time from target-noun onset (ms)")
    ax.set_ylabel("Held-out projection")
    fixed = primary["fixed_stats"]
    fixed = fixed[fixed["endpoint"] == "ccgp_accuracy"].set_index("window_ms")
    annotation = "   ".join(
        f"{window}: {fixed.loc[window, 'patient_equal_estimate']:.2f}"
        for window in [f"{a}-{b}" for a, b in FIXED_WINDOWS_MS]
    )
    ax.text(0.01, 0.98, f"Fixed-window CCGP  {annotation}", transform=ax.transAxes, va="top", fontsize=6.2)
    ax.text(
        0.01,
        0.05,
        "No bin can pass P<0.0125 with n=5; max-stat reported",
        transform=ax.transAxes,
        fontsize=6.2,
        color=COLORS["gray"],
    )
    ax.set_title("Frozen-axis temporal expression", loc="left")
    panel_label(ax, "b")

    ax = axes[1, 0]
    controls = primary["control_patient"].copy()
    control_order = [
        "primary",
        "acoustic_residualized",
        "acoustic_plus_serial_time_residualized",
        "pre_noun_baseline",
    ]
    patient_points(ax, controls, "specification", "accuracy", control_order)
    ax.axhline(0.5, color=COLORS["gray"], ls="--", lw=0.7)
    short = ["Primary", "Acoustic\nresidual", "Acoustic +\ntime residual", "Pre-noun\nbaseline"]
    cp = [
        inference["controls"][specification]["patient_exact_p_greater"]
        for specification in control_order
    ]
    ax.set_xticks(range(4), [f"{label}\n{p_label(p)}" for label, p in zip(short, cp)])
    ax.set_ylabel("Cross-role position accuracy")
    ax.set_title("Controls and estimability", loc="left")
    panel_label(ax, "c")

    ax = axes[1, 1]
    ablation = primary["ablation_patient"]
    for patient, cell in ablation.groupby("patient"):
        cell = cell.sort_values("fraction_removed")
        ax.plot(
            cell["fraction_removed"] * 100,
            cell["top_loading_removed_accuracy"],
            color=PATIENT_COLORS[str(patient)],
            alpha=0.25,
            lw=0.7,
        )
    grouped = ablation.groupby("fraction_removed").agg(
        top=("top_loading_removed_accuracy", "mean"),
        random=("matched_random_removed_accuracy", "mean"),
        baseline=("baseline_accuracy", "mean"),
    )
    x_fraction = grouped.index.to_numpy(dtype=float) * 100
    ax.plot(x_fraction, grouped["top"], "o-", color=COLORS["vermillion"], label="top-loading")
    ax.plot(x_fraction, grouped["random"], "o--", color=COLORS["gray"], label="matched random")
    ax.axhline(grouped["baseline"].mean(), color=COLORS["blue"], ls=":", lw=1.0, label="no removal")
    ax.axhline(0.5, color=COLORS["light"], ls="--", lw=0.7)
    ax.set_xticks([5, 10, 20])
    ax.set_xlabel("Units removed (%)")
    ax.set_ylabel("Cross-role position accuracy")
    pr = inference["units"]["patient_equal_participation_ratio_fraction"]
    gi = inference["units"]["patient_equal_gini_absolute_loading"]
    ax.text(0.02, 0.04, f"PR/n={pr:.2f}; loading Gini={gi:.2f}", transform=ax.transAxes, fontsize=6.5)
    ax.legend(loc="best")
    ax.set_title("Distributed readout contribution", loc="left")
    panel_label(ax, "d")

    ax = axes[2, 0]
    scrambled = primary["scrambled_patient"].melt(
        id_vars=["unit_set", "patient"],
        value_vars=[
            "grammar_post_minus_initial_projection",
            "scrambled_later_minus_earlier_projection",
        ],
        var_name="context",
        value_name="projection",
    )
    scramble_order = [
        "grammar_post_minus_initial_projection",
        "scrambled_later_minus_earlier_projection",
    ]
    patient_points(ax, scrambled, "context", "projection", scramble_order)
    ax.axhline(0, color=COLORS["gray"], ls="--", lw=0.7)
    sp = inference["scrambled"]
    ax.set_xticks(
        [0, 1],
        [
            f"Grammar post − initial\n{p_label(sp['grammar']['patient_exact_p_greater'])}",
            f"Scrambled later − earlier\n{p_label(sp['scrambled']['patient_exact_p_greater'])}",
        ],
    )
    ax.set_ylabel("Position-axis projection")
    ax.set_title("Meaningful versus scrambled", loc="left")
    panel_label(ax, "e")

    ax = axes[2, 1]
    anchor_order = ["noun", "marker", "verb"]
    patient_points(
        ax,
        anchor_patient,
        "anchor",
        "retrieval_margin",
        anchor_order,
    )
    ax.axhline(0, color=COLORS["gray"], ls="--", lw=0.7)
    indexed_summary = anchor_summary.set_index("anchor")
    anchor_labels = [
        f"{name.title()}\npatient P={indexed_summary.loc[name, 'margin_patient_p']:.3f}\nanchor-FWER={indexed_summary.loc[name, 'margin_patient_anchor_fwer_p']:.3f}"
        for name in anchor_order
    ]
    ax.set_xticks(range(3), anchor_labels)
    ax.set_ylabel("Rank-4 retrieval margin")
    ax.set_title("Separate event-anchor manifold test", loc="left")
    panel_label(ax, "f")

    figure.suptitle(
        "A shared but confounded position-related population transformation",
        fontsize=10,
        fontweight="bold",
        y=0.995,
    )
    save_figure(figure, FIG_DIR / "Fig_9_intro_driven_structural_transformation")
    plt.close(figure)


def _plot_panel_b_exact_size(
    frames: dict[str, dict[str, pd.DataFrame]],
    width_mm: float,
    stem_name: str,
    height_mm: float = 60.0,
) -> None:
    """Export Fig. 9b on an exact-size canvas."""
    configure_style()
    compact = width_mm < 55.0
    short = height_mm < 50.0
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    ax = figure.add_axes(
        [
            0.23,
            0.25 if short else 0.18,
            0.69 if compact else 0.72,
            0.55 if short else 0.63,
        ]
    )
    trajectory = frames[PRIMARY_UNIT_SET]["trajectory_stats"]
    x = trajectory["time_ms"].to_numpy(dtype=float)
    mean = trajectory["patient_equal_projection"].to_numpy(dtype=float)
    sem = trajectory["patient_sem_projection"].to_numpy(dtype=float)
    ax.fill_between(x, mean - sem, mean + sem, color=COLORS["sky"], alpha=0.24, lw=0)
    ax.plot(x, mean, color=COLORS["blue"], lw=1.0)
    ax.axhline(0, color=COLORS["gray"], ls="--", lw=0.55)
    ax.axvline(0, color=COLORS["black"], lw=0.6)
    ax.set_xlim(CONTINUOUS_START_MS, CONTINUOUS_STOP_MS)
    limit = max(5.0, float(np.nanmax(np.abs(np.r_[mean - sem, mean + sem]))) * 1.03)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([-200, 0, 400, 800])
    ax.set_yticks([-5, 0, 5])
    x_label = (
        "Time from target-noun\nonset (ms)"
        if compact
        else "Time from target-noun onset (ms)"
    )
    title = (
        "Frozen position-axis\nexpression"
        if compact
        else "Frozen position-axis expression"
    )
    ax.set_xlabel(x_label, fontsize=8.0, labelpad=2.0)
    ax.set_ylabel("Held-out projection", fontsize=8.0, labelpad=1.5)
    ax.tick_params(axis="both", labelsize=8.0, length=2.1, pad=1.5)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)

    figure.text(0.035, 0.965, "b", fontsize=8.0, fontweight="bold", ha="left", va="top")
    figure.text(
        0.23,
        0.955,
        title,
        fontsize=8.0,
        ha="left",
        va="top",
    )
    stem = FIG_DIR / stem_name
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def plot_panel_b_45x60mm(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> None:
    _plot_panel_b_exact_size(
        frames,
        width_mm=45.0,
        stem_name="Fig_9b_frozen_axis_temporal_expression_45x60mm",
    )


def plot_panel_b_60x60mm(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> None:
    _plot_panel_b_exact_size(
        frames,
        width_mm=60.0,
        stem_name="Fig_9b_frozen_axis_temporal_expression_60x60mm",
    )


def plot_panel_b_60x40mm(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> None:
    _plot_panel_b_exact_size(
        frames,
        width_mm=60.0,
        height_mm=40.0,
        stem_name="Fig_9b_frozen_axis_temporal_expression_60x40mm",
    )


def plot_panel_b_60x45mm(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> None:
    _plot_panel_b_exact_size(
        frames,
        width_mm=60.0,
        height_mm=45.0,
        stem_name="Fig_9b_frozen_axis_temporal_expression_60x45mm",
    )


def plot_extended_figure(
    frames: dict[str, dict[str, pd.DataFrame]],
    design_audit: pd.DataFrame,
    vif: pd.DataFrame,
) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 3, figsize=(7.09, 4.45))
    figure.subplots_adjust(hspace=0.55, wspace=0.40)

    ax = axes[0, 0]
    audit = design_audit.copy()
    labels = ["Position", "+ acoustic", "+ serial time", "+ structural"]
    x = np.arange(len(audit))
    ax.bar(x, audit["position_residual_fraction_after_nuisance"], color=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["vermillion"]])
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Unique position-vector fraction")
    ax.set_title("Design estimability", loc="left")
    panel_label(ax, "a")

    ax = axes[0, 1]
    position_vif = vif[vif["term"] == "position_post"].copy()
    clipped = position_vif["vif"].replace(np.inf, 1e3).to_numpy(dtype=float)
    ax.bar(range(len(clipped)), clipped, color=COLORS["gray"])
    ax.set_yscale("log")
    ax.axhline(10, color=COLORS["vermillion"], ls="--", lw=0.8)
    ax.set_xticks(range(len(clipped)), labels, rotation=25, ha="right")
    ax.set_ylabel("Position VIF (∞ clipped at 10³)")
    ax.set_title("Collinearity audit", loc="left")
    panel_label(ax, "b")

    ax = axes[0, 2]
    rows = []
    for unit_set, data in frames.items():
        cell = data["static_patient"].copy()
        cell["ccgp_excess"] = cell["ccgp_accuracy"] - 0.5
        rows.append(cell[["unit_set", "patient", "parallelism_score", "ccgp_excess"]])
    compare = pd.concat(rows, ignore_index=True).melt(
        id_vars=["unit_set", "patient"], var_name="endpoint", value_name="value"
    )
    offsets = {"good_only": -0.12, "usable_lenient": 0.12}
    color = {"good_only": COLORS["blue"], "usable_lenient": COLORS["orange"]}
    for unit_set, cell in compare.groupby("unit_set"):
        means = cell.groupby("endpoint")["value"].mean().reindex(["parallelism_score", "ccgp_excess"])
        ax.scatter(np.arange(2) + offsets[unit_set], means, s=35, color=color[unit_set], label=unit_set)
    ax.axhline(0, color=COLORS["gray"], ls="--", lw=0.7)
    ax.set_xticks([0, 1], ["PS", "CCGP − chance"])
    ax.set_ylabel("Patient-equal endpoint")
    ax.legend()
    ax.set_title("Unit-set sensitivity", loc="left")
    panel_label(ax, "c")

    ax = axes[1, 0]
    for unit_set, data in frames.items():
        values = data["unit_run_metrics"]["participation_ratio_fraction"]
        positions = 0 if unit_set == "good_only" else 1
        jitter = np.linspace(-0.07, 0.07, len(values))
        ax.scatter(np.full(len(values), positions) + jitter, values, s=18, alpha=0.75, color=color[unit_set])
        ax.scatter(positions, values.mean(), marker="_", s=180, lw=2, color="black")
    ax.set_xticks([0, 1], ["Good only", "Usable lenient"])
    ax.set_ylabel("Participation ratio / units")
    ax.set_ylim(0, 1)
    ax.set_title("Readout participation", loc="left")
    panel_label(ax, "d")

    ax = axes[1, 1]
    for unit_set, data in frames.items():
        trajectory = data["trajectory_stats"]
        ax.plot(trajectory["time_ms"], trajectory["patient_exact_point_p_greater"], color=color[unit_set], label=unit_set)
    ax.axhline(POINT_ALPHA, color=COLORS["vermillion"], ls="--", lw=0.8, label="0.05/4")
    ax.axhline(1 / 32, color=COLORS["black"], ls=":", lw=0.8, label="n=5 minimum P")
    ax.set_yscale("log")
    ax.set_xlabel("Time from target noun (ms)")
    ax.set_ylabel("Exact point P")
    ax.legend(fontsize=5.5, loc="upper right", ncol=2)
    ax.set_title("Time-point resolution", loc="left")
    panel_label(ax, "e")

    ax = axes[1, 2]
    acoustic = pd.read_csv(SOURCE_DIR / "stimulus_acoustic_features.csv")
    for position_index, position in enumerate(POSITIONS):
        cell = acoustic[acoustic["surface_position"] == position]
        values = cell["serial_onset_sec"].to_numpy(dtype=float)
        ax.scatter(np.full(len(values), position_index) + np.linspace(-0.06, 0.06, len(values)), values, s=13, alpha=0.55, color=[COLORS["sky"], COLORS["orange"]][position_index])
    ax.set_xticks([0, 1], ["Initial", "Post-marker"])
    ax.set_ylabel("Noun onset in sentence (s)")
    ax.text(0.03, 0.94, "Marker presence = position exactly", transform=ax.transAxes, va="top", fontsize=6.5)
    ax.set_title("Serial/local-context confound", loc="left")
    panel_label(ax, "f")

    figure.suptitle("Validation and sensitivity analyses", fontsize=10, fontweight="bold", y=0.995)
    save_figure(figure, FIG_DIR / "Extended_Data_Fig_9_validation_sensitivity")
    plt.close(figure)


def plot_fig4_integration(
    frames: dict[str, dict[str, pd.DataFrame]],
    summaries: dict[str, dict[str, object]],
) -> None:
    configure_style()
    old_units = pd.read_csv(
        ROOT
        / "reports/fig4_noun_identity_position_kimi_20260806/source_data/panel_b_patient_unit_rates.csv"
    )
    old_units = old_units[old_units["metric"] == "bc"]
    old_decode = pd.read_csv(
        ROOT
        / "reports/fig4_noun_identity_position_kimi_20260806/source_data/panel_c_patient_decoding_endpoints.csv"
    )
    old_decode = old_decode[old_decode["target"] == "position across semantic roles"]
    primary = frames[PRIMARY_UNIT_SET]
    inference = summaries[PRIMARY_UNIT_SET]

    figure, axes = plt.subplots(1, 4, figsize=(7.09, 2.35), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    figure.subplots_adjust(wspace=0.50, top=0.72, bottom=0.23)
    ax = axes[0]
    patient_points(ax, old_units, "factor", "rate", ["noun identity", "surface position"])
    ax.set_xticks([0, 1], ["Noun\nidentity", "Surface\nposition"])
    ax.set_ylabel("Unit screening rate")
    ax.set_title("Fig. 4: unit level", loc="left")
    panel_label(ax, "a")

    ax = axes[1]
    for index, row in old_decode.reset_index(drop=True).iterrows():
        ax.scatter(0, row["accuracy"], color=PATIENT_COLORS[row["patient"]], s=20, edgecolor="white", lw=0.35)
    ax.scatter(0, old_decode["accuracy"].mean(), marker="_", s=200, lw=2, color="black")
    ax.axhline(0.5, color=COLORS["gray"], ls="--", lw=0.7)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([0], ["Position across roles\nP=0.031"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig. 4: generalization", loc="left")
    panel_label(ax, "b")

    ax = axes[2]
    transformed = primary["static_patient"].melt(
        id_vars=["patient"],
        value_vars=["parallelism_score", "heldout_cross_projection"],
        var_name="endpoint",
        value_name="value",
    )
    patient_points(ax, transformed, "endpoint", "value", ["parallelism_score", "heldout_cross_projection"])
    ax.axhline(0, color=COLORS["gray"], ls="--", lw=0.7)
    ax.set_xticks([0, 1], ["Parallelism\nscore", "Held-out\nprojection"])
    ax.set_ylabel("Geometry endpoint")
    ax.set_title("New: transformation", loc="left")
    panel_label(ax, "c")

    ax = axes[3]
    ax.axis("off")
    static = inference["static"]
    controls = inference["controls"]
    text = (
        "Evidence chain\n\n"
        "1  More position than noun-screened units\n"
        "2  Position decoder crosses Agent/Patient\n"
        f"3  PS={static['parallelism_score']['patient_equal_estimate']:.2f}, "
        f"{p_label(static['parallelism_score']['patient_exact_p_greater'])}\n"
        f"4  Acoustic residual CCGP={controls['acoustic_residualized']['patient_equal_accuracy']:.2f}\n\n"
        "Bounded claim:\ntransient distributed construction/position state;\n"
        "not pure syntax, voice or noun semantics."
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.0,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F5F7FA", "edgecolor": COLORS["light"]},
    )
    panel_label(ax, "d")
    figure.suptitle("How the new manifold analysis extends Fig. 4", fontsize=10, fontweight="bold", y=0.97)
    save_figure(figure, FIG_DIR / "Fig_10_Fig4_plus_structural_transformation")
    plt.close(figure)


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_ordered = np.maximum.accumulate(
        np.asarray([(len(p) - index) * p[item] for index, item in enumerate(order)])
    )
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def module_family_table(
    summaries: dict[str, dict[str, object]],
    frames: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    summary = summaries[PRIMARY_UNIT_SET]
    fixed_ccgp = frames[PRIMARY_UNIT_SET]["fixed_stats"]
    fixed_ccgp = fixed_ccgp[fixed_ccgp["endpoint"] == "ccgp_accuracy"]
    rows = [
        {
            "module": "parallelism_score",
            "endpoint": "patient-equal PS > 0",
            "p_within_module": summary["static"]["parallelism_score"]["patient_exact_p_greater"],
        },
        {
            "module": "time_resolved_position_axis",
            "endpoint": "minimum patient max-stat P across three fixed windows",
            "p_within_module": float(
                fixed_ccgp["patient_window_maxstat_p_fwer_greater"].min()
            ),
        },
        {
            "module": "acoustic_control",
            "endpoint": "acoustic-residualized cross-role CCGP > 0.5",
            "p_within_module": summary["controls"]["acoustic_residualized"]["patient_exact_p_greater"],
        },
        {
            "module": "unit_ablation",
            "endpoint": "20% matched-random removal accuracy > top-loading removal accuracy",
            "p_within_module": summary["units"]["ablation_0.20"]["patient_exact_p_random_greater_than_top"],
        },
        {
            "module": "scrambled_transfer",
            "endpoint": "scrambled later-minus-earlier projection > 0",
            "p_within_module": summary["scrambled"]["scrambled"]["patient_exact_p_greater"],
        },
    ]
    frame = pd.DataFrame(rows)
    frame["holm_p_across_five_modules"] = holm_adjust(
        frame["p_within_module"].to_numpy(dtype=float)
    )
    return frame


def write_reports(
    summaries: dict[str, dict[str, object]],
    frames: dict[str, dict[str, pd.DataFrame]],
    design_audit: pd.DataFrame,
    module_family: pd.DataFrame,
    anchor_summary: pd.DataFrame,
) -> None:
    primary = summaries[PRIMARY_UNIT_SET]
    sensitivity = summaries["usable_lenient"]
    static = primary["static"]
    controls = primary["controls"]
    units = primary["units"]
    scrambled = primary["scrambled"]
    time = primary["time"]
    structural = design_audit.set_index("model").loc["position_plus_structural_context"]
    noun_anchor = anchor_summary.set_index("anchor").loc["noun"]
    module_survivors = int((module_family["holm_p_across_five_modules"] < 0.05).sum())

    fixed = frames[PRIMARY_UNIT_SET]["fixed_stats"]
    fixed_ccgp = fixed[fixed["endpoint"] == "ccgp_accuracy"].set_index("window_ms")
    sensitivity_fixed = frames["usable_lenient"]["fixed_stats"]
    sensitivity_ps = sensitivity_fixed[
        sensitivity_fixed["endpoint"] == "parallelism_score"
    ].set_index("window_ms")
    sensitivity_trajectory = frames["usable_lenient"]["trajectory_stats"]
    sensitivity_fwer_bins = sensitivity_trajectory[
        sensitivity_trajectory["patient_time_maxstat_p_fwer_greater"] < 0.05
    ]
    sensitivity_fwer_text = (
        ", ".join(
            f"{int(row.time_ms)} ms (P={row.patient_time_maxstat_p_fwer_greater:.5f})"
            for row in sensitivity_fwer_bins.itertuples(index=False)
        )
        if len(sensitivity_fwer_bins)
        else "none"
    )
    report = f"""# Introduction 驱动的结构位置变换分析

## 一句话结论

在 good-only 主规格中，五位患者的群体活动可检验是否存在从句首 NP 到标记词后 NP 的共同变换；结果需要同时结合 Parallelism、跨角色泛化、声学/时间控制、unit 消融和 scrambled 反证来判断。无论具体显著性如何，这套材料仍不能把位置、句中时刻和前置功能词语境分离，因此最稳妥的结论层级是 **transient, distributed construction/position-related population state**，不是纯句法、抽象语态或名词语义神经元。

## 1. 共同位置变换：Parallelism Score 与 CCGP

- Parallelism Score（post-marker − initial，患者等权）为 **{static['parallelism_score']['patient_equal_estimate']:.4f}**；患者 exact P={static['parallelism_score']['patient_exact_p_greater']:.5f}，10 个句对同步置换 P={static['parallelism_score']['synchronized_pair_permutation_p_greater']:.5f}。
- held-out cross-role projection 为 **{static['heldout_cross_projection']['patient_equal_estimate']:.4f}**；患者 exact P={static['heldout_cross_projection']['patient_exact_p_greater']:.5f}，句对置换 P={static['heldout_cross_projection']['synchronized_pair_permutation_p_greater']:.5f}。
- 最近质心轴的跨 Agent/Patient CCGP 为 **{static['ccgp_accuracy']['patient_equal_estimate']:.3f}**（机会水平 0.5）；患者 exact P={static['ccgp_accuracy']['patient_exact_p_greater']:.5f}，句对置换 P={static['ccgp_accuracy']['synchronized_pair_permutation_p_greater']:.5f}。
- usable-lenient 敏感性：PS={sensitivity['static']['parallelism_score']['patient_equal_estimate']:.4f}；CCGP={sensitivity['static']['ccgp_accuracy']['patient_equal_estimate']:.3f}。

解释：PS 检验 Agent 和 Patient 两类名词的“位置移动方向”是否平行；CCGP 检验在一类角色上学到的位置边界能否用于另一类角色。二者共同阳性才比较接近“shared transformation”；仅 CCGP 阳性也可能来自较粗的共同读出，而不是平行流形位移。

## 2. 时间分辨结果

- 0–200 ms CCGP={fixed_ccgp.loc['0-200', 'patient_equal_estimate']:.3f}，三窗口 max-stat P={fixed_ccgp.loc['0-200', 'patient_window_maxstat_p_fwer_greater']:.5f}。
- 200–500 ms CCGP={fixed_ccgp.loc['200-500', 'patient_equal_estimate']:.3f}，三窗口 max-stat P={fixed_ccgp.loc['200-500', 'patient_window_maxstat_p_fwer_greater']:.5f}。
- 500–800 ms CCGP={fixed_ccgp.loc['500-800', 'patient_equal_estimate']:.3f}，三窗口 max-stat P={fixed_ccgp.loc['500-800', 'patient_window_maxstat_p_fwer_greater']:.5f}。
- 5-ms 连续曲线严格使用 0–800 ms 训练轴投射 held-out 轨迹，没有逐时刻重新拟合。旧规则要求 P<0.0125 且持续 >5 ms；n=5 的 exact sign-flip 最小可达 P 为 {time['minimum_patient_exact_p_resolution']:.5f}，因此 **{time['legacy_segments_gt_5ms']} 个区段**能满足该点阈值；全时间 max-stat 通过的 bin 为 **{time['time_fwer_significant_bins']}**。
- usable-lenient 中 PS 在 200–500 ms 为 {sensitivity_ps.loc['200-500', 'patient_equal_estimate']:.3f}（三窗口 FWER P={sensitivity_ps.loc['200-500', 'patient_window_maxstat_p_fwer_greater']:.5f}），500–800 ms 为 {sensitivity_ps.loc['500-800', 'patient_equal_estimate']:.3f}（FWER P={sensitivity_ps.loc['500-800', 'patient_window_maxstat_p_fwer_greater']:.5f}）。连续曲线仅出现 {sensitivity_fwer_text} 的孤立 max-stat bin；因只有 5 ms 且 point P 不可能低于 0.03125，它不满足“持续 >5 ms 且 P<0.0125”的旧规则。
- Marker 和 verb 对齐不能被重新标成某一个名词的“位置”观测，因为同一句同时含一个 initial NP 和一个 post-marker NP。独立、已冻结的 Rank-4 anchor 分析中，noun-anchor margin={noun_anchor.static_margin:.4f}，患者 P={noun_anchor.margin_patient_p:.5f}，三-anchor FWER P={noun_anchor.margin_patient_anchor_fwer_p:.5f}；该结果只作为事件锚点依赖性证据。

## 3. 声学、句中时刻与基线控制

- 主规格 CCGP={controls['primary']['patient_equal_accuracy']:.3f}，P={controls['primary']['patient_exact_p_greater']:.5f}。
- 训练折内声学残差化后 CCGP={controls['acoustic_residualized']['patient_equal_accuracy']:.3f}，P={controls['acoustic_residualized']['patient_exact_p_greater']:.5f}。
- 再加入 serial onset 的强残差化后 CCGP={controls['acoustic_plus_serial_time_residualized']['patient_equal_accuracy']:.3f}，P={controls['acoustic_plus_serial_time_residualized']['patient_exact_p_greater']:.5f}。
- 单独使用 pre-noun baseline 的 CCGP={controls['pre_noun_baseline']['patient_equal_accuracy']:.3f}，P={controls['pre_noun_baseline']['patient_exact_p_greater']:.5f}。
- 完整结构控制矩阵 rank={int(structural.matrix_rank)}/{int(structural.n_columns)}，position 在其他结构项后的残差比例={structural.position_residual_fraction_after_nuisance:.3g}。前置 marker 是否存在与 position 完全相同，因此“控制 marker 后的纯位置/纯句法效应”在本刺激设计中不可估计。加入 serial onset 的残差化是强敏感性分析，不是已识别的因果调整。

## 4. Unit 贡献是否分布式

- 患者等权的 participation ratio / unit 数={units['patient_equal_participation_ratio_fraction']:.3f}；绝对 loading Gini={units['patient_equal_gini_absolute_loading']:.3f}。
- 移除训练折 top 20% loading units 后，患者等权 CCGP={units['ablation_0.20']['patient_equal_top_removed_accuracy']:.3f}；匹配随机移除={units['ablation_0.20']['patient_equal_random_removed_accuracy']:.3f}；random−top={units['ablation_0.20']['patient_equal_random_minus_top']:.3f}，患者 exact P={units['ablation_0.20']['patient_exact_p_random_greater_than_top']:.5f}。

解释：较高 PR 和较低 Gini 支持分布式弱贡献；top-unit 消融若比随机移除下降更多，则说明仍存在权重集中的关键子群。两类指标要一起读，不能用单个“代表 unit”代替群体证据。

## 5. Meaningful 与 scrambled 反证

- grammar held-out pair 的 post−initial 投影={scrambled['grammar']['patient_equal_projection']:.4f}，患者 P={scrambled['grammar']['patient_exact_p_greater']:.5f}。
- scrambled 的 later−earlier 投影={scrambled['scrambled']['patient_equal_later_minus_earlier_projection']:.4f}，患者 P={scrambled['scrambled']['patient_exact_p_greater']:.5f}，五句对同步置换 P={scrambled['scrambled']['synchronized_five_pair_permutation_p_greater']:.5f}。
- scrambled 投影与字符位置的患者等权 Spearman={scrambled['scrambled']['patient_equal_char_index_spearman']:.3f}，P={scrambled['scrambled']['char_index_patient_exact_p_greater']:.5f}。

该模块每个 run 只有五个 scrambled 句子，而且先后顺序与 noun identity、thematic role 仍相互纠缠，只能作为 serial-position 反证或描述性补充；不能单独证明“语法存在/消失”。

## 五模块联合判断

五个预先列出的模块在各自内部控制后，再做五模块 Holm 校正；通过 0.05 的模块数为 **{module_survivors}/5**。完整结果见 `source_data/module_family_correction.csv`，阴性模块保留在主报告和图中，没有按显著性筛选。

本文正文中的患者 exact P 均为事先规定方向的 one-sided 检验；n=5 时五位患者同方向的最小 P=0.03125，对应 two-sided P=0.0625。因而“P=0.03125”应读作方向一致性证据，而不是高精度的显著性估计。

## 与原 Fig. 4 的故事关系

原 Fig. 4 的 A/D 类结果回答“position 信息是否比 noun identity 更突出、是否可跨 Agent/Patient 读出”；本分析进一步回答“这种读出是否对应共享方向、何时表达、能否经受声学/时间控制、由多少 units 承担、以及 scrambled 是否也沿同一轴变化”。最合理的故事顺序是：

1. 单 unit 层面 noun identity 没有富集，而 position screening 更高；
2. 群体层面 position 可跨语义角色泛化；
3. 共享几何用 PS/held-out projection 直接检验，而不是把 decoder accuracy 自动称为 manifold；
4. 时间、声学、unit 消融和 scrambled 结果限定这种几何究竟更像 construction/position state，还是可由线性位置/局部语境解释。

## 文件

- `figures/Fig_9_intro_driven_structural_transformation.*`：六面板主图。
- `figures/Fig_10_Fig4_plus_structural_transformation.*`：与原 Fig. 4 的整合图。
- `figures/Extended_Data_Fig_9_validation_sensitivity.*`：可估计性、unit-set 和时间统计分辨率。
- `source_data/`：图源数据、五模块校正和 anchor 整合表。
- `../../neuropixel2025/data/combined_intro_driven_structural_transformation_20260807/`：逐 fold、逐 run、逐患者完整数值。
"""
    (REPORT_DIR / "REPORT_zh.md").write_text(report, encoding="utf-8")

    caption = f"""Fig. 9 | Cross-role neural transformation from sentence-initial to post-marker noun position. a, Patient-level parallelism score (PS), held-out cross-role projection and cross-condition generalization performance (CCGP; accuracy minus 0.5) for good-only units. Each coloured point is one patient; black bars denote patient-equal means. b, A position axis fitted only to the frozen 0–800-ms response in training nouns was used to project held-out, cross-role 5-ms activity around target-noun onset. Shading denotes the s.e.m. across five patients; fixed-window CCGP is reported for 0–200, 200–500 and 500–800 ms. No point can satisfy P<0.0125 under an exact five-patient sign-flip test, whose minimum attainable P is 0.03125; max-statistic P values are provided in the source data. c, Cross-role position decoding before and after training-fold acoustic residualization, after the additional removal of serial-onset-predictable activity, and in the pre-noun baseline. d, Decoder accuracy after training-fold removal of the largest absolute position-axis loadings or the same number of randomly selected units. e, Cross-fitted position-axis projection for held-out grammatical pairs and for later-minus-earlier nouns in the five recorded scrambled sentences. f, Separately frozen Rank-4 retrieval margins for noun-, marker- and verb-aligned activity; marker- and verb-aligned sentence responses are not treated as noun-position observations. Patient-level inference used exhaustive exact sign flips; stimulus inference used synchronized within-pair label swaps. Surface position remains confounded with serial time and preceding function-word context, so the results describe a distributed construction/position-related population state rather than syntax-selective neurons."""
    (REPORT_DIR / "Fig9_caption_en.txt").write_text(caption + "\n", encoding="utf-8")

    methods = f"""# Insertable Methods and Results text

## Methods

We tested whether matched Agent and Patient nouns shared a population transformation from sentence-initial to post-marker position. Neural responses were defined a priori as mean firing from 0–800 ms after target-noun onset minus the −500–0-ms baseline. Within each run, post-marker-minus-initial difference vectors were estimated after training-only unit standardization. We quantified their cosine similarity (parallelism score), and evaluated cross-condition generalization by leaving out one complete sentence pair, fitting a nearest-centroid position axis in one thematic-role class and testing the other. All 2^10 synchronized within-pair position-label swaps provided a stimulus-level null. Runs and units were nested within patients; final inference used exhaustive patient-level sign flips (n=5). Three time windows (0–200, 200–500 and 500–800 ms) were fixed before analysis. A continuous 5-ms trajectory projected held-out responses through an axis fitted only to the 0–800-ms training response. Acoustic controls used cross-fitted multivariate ridge regression of log RMS, spectral centroid, zero-crossing rate, autocorrelation-based F0, voiced-frame fraction and envelope-derived duration; a stronger sensitivity model additionally included noun onset within the sentence. Unit participation was summarized by the participation ratio, Gini coefficient and training-fold top-loading ablation. Finally, grammar-trained axes were applied to the two matched nouns in each of five recorded scrambled sentences.

## Results

For good-only units, the patient-equal parallelism score was {static['parallelism_score']['patient_equal_estimate']:.3f} (exact P={static['parallelism_score']['patient_exact_p_greater']:.4f}) and cross-role CCGP was {static['ccgp_accuracy']['patient_equal_estimate']:.3f} (chance=0.5; exact P={static['ccgp_accuracy']['patient_exact_p_greater']:.4f}). Acoustic-residualized CCGP was {controls['acoustic_residualized']['patient_equal_accuracy']:.3f} (P={controls['acoustic_residualized']['patient_exact_p_greater']:.4f}), whereas the pre-noun baseline yielded {controls['pre_noun_baseline']['patient_equal_accuracy']:.3f} (P={controls['pre_noun_baseline']['patient_exact_p_greater']:.4f}). The common readout had a patient-equal normalized participation ratio of {units['patient_equal_participation_ratio_fraction']:.3f} and an absolute-loading Gini coefficient of {units['patient_equal_gini_absolute_loading']:.3f}. In scrambled sentences, the later-minus-earlier projection was {scrambled['scrambled']['patient_equal_later_minus_earlier_projection']:.3f} (P={scrambled['scrambled']['patient_exact_p_greater']:.4f}). Because noun position was exactly confounded with preceding-marker presence and strongly confounded with serial onset, these analyses support, at most, a transient and distributed construction/position-related population state that generalizes across the role-reversed noun sets; they do not isolate abstract syntactic hierarchy or voice.
"""
    (REPORT_DIR / "METHODS_RESULTS_en.md").write_text(methods, encoding="utf-8")


def validate_outputs(
    frames: dict[str, dict[str, pd.DataFrame]],
    design_audit: pd.DataFrame,
) -> dict[str, object]:
    checks: dict[str, object] = {
        "all_checks_pass": True,
        "unit_sets": {},
        "figures": {},
        "design": {},
    }
    for unit_set, data in frames.items():
        unit_checks = {
            "five_patients_static": int(data["static_patient"]["patient"].nunique()) == 5,
            "exact_1024_pair_null": len(data["static_null"]) == 1024,
            "three_fixed_windows": int(data["fixed_stats"]["window_ms"].nunique()) == 3,
            "continuous_grid_complete": len(data["trajectory_stats"])
            == int((CONTINUOUS_STOP_MS - CONTINUOUS_START_MS) / CONTINUOUS_STEP_MS),
            "five_scrambled_pairs_per_run": bool(
                (data["scrambled_run"]["n_scrambled_pairs"] == 5).all()
            ),
            "finite_primary_endpoints": bool(
                np.isfinite(
                    data["static_patient"][
                        ["parallelism_score", "heldout_cross_projection", "ccgp_accuracy"]
                    ].to_numpy(dtype=float)
                ).all()
            ),
        }
        checks["unit_sets"][unit_set] = unit_checks
        checks["all_checks_pass"] = checks["all_checks_pass"] and all(unit_checks.values())
    structural = design_audit.set_index("model").loc["position_plus_structural_context"]
    checks["design"] = {
        "structural_design_rank_deficient": not bool(structural.full_column_rank),
        "position_not_estimable_beyond_structural_context": not bool(
            structural.position_estimable_beyond_nuisance
        ),
    }
    checks["all_checks_pass"] = checks["all_checks_pass"] and all(
        checks["design"].values()
    )
    for stem in (
        "Fig_9_intro_driven_structural_transformation",
        "Fig_9b_frozen_axis_temporal_expression_45x60mm",
        "Fig_10_Fig4_plus_structural_transformation",
        "Extended_Data_Fig_9_validation_sensitivity",
    ):
        files = {}
        for suffix in (".pdf", ".svg", ".png", ".tiff"):
            path = FIG_DIR / f"{stem}{suffix}"
            files[suffix] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        checks["figures"][stem] = files
        checks["all_checks_pass"] = checks["all_checks_pass"] and all(
            item["exists"] and item["bytes"] > 1000 for item in files.values()
        )
    checks["figures"]["Fig_9b_frozen_axis_temporal_expression_45x60mm"][
        "physical_size_mm"
    ] = [45.0, 60.0]
    checks["figures"]["Fig_9b_frozen_axis_temporal_expression_45x60mm"][
        "raster_contract"
    ] = {"dpi": 600, "pixel_width": 1062, "pixel_height": 1417}
    return checks


def write_specification_outputs(
    unit_set: str,
    frames: dict[str, pd.DataFrame],
    summary: dict[str, object],
) -> None:
    directory = DATA_DIR / (
        "good_only" if unit_set == "good_only" else "sensitivity_usable_lenient"
    )
    directory.mkdir(parents=True, exist_ok=True)
    file_map = {
        "static_predictions": "static_cross_role_predictions.csv",
        "static_folds": "static_parallelism_cv_folds.csv",
        "static_run": "static_run_endpoints.csv",
        "static_patient": "static_patient_endpoints.csv",
        "static_null": "static_synchronized_pair_exact_null.csv",
        "fixed_run": "fixed_time_run_endpoints.csv",
        "fixed_patient": "fixed_time_patient_endpoints.csv",
        "fixed_stats": "fixed_time_window_stats.csv",
        "trajectory_stats": "continuous_time_patient_stats.csv",
        "time_segments": "continuous_time_legacy_segments.csv",
        "control_predictions": "control_cross_role_predictions.csv",
        "control_run": "control_run_endpoints.csv",
        "control_patient": "control_patient_endpoints.csv",
        "unit_loadings": "crossfitted_unit_loadings.csv",
        "unit_run_metrics": "unit_participation_run_metrics.csv",
        "ablation_run": "unit_ablation_run_endpoints.csv",
        "ablation_patient": "unit_ablation_patient_endpoints.csv",
        "scrambled_pair": "scrambled_pair_projections.csv",
        "scrambled_run": "scrambled_run_endpoints.csv",
        "scrambled_patient": "scrambled_patient_endpoints.csv",
    }
    for key, filename in file_map.items():
        frames[key].to_csv(directory / filename, index=False)
    write_json(directory / "analysis_summary.json", summary)


def reload_completed_outputs() -> tuple[
    dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, object]]
]:
    file_map = {
        "static_predictions": "static_cross_role_predictions.csv",
        "static_folds": "static_parallelism_cv_folds.csv",
        "static_run": "static_run_endpoints.csv",
        "static_patient": "static_patient_endpoints.csv",
        "static_null": "static_synchronized_pair_exact_null.csv",
        "fixed_run": "fixed_time_run_endpoints.csv",
        "fixed_patient": "fixed_time_patient_endpoints.csv",
        "fixed_stats": "fixed_time_window_stats.csv",
        "trajectory_stats": "continuous_time_patient_stats.csv",
        "time_segments": "continuous_time_legacy_segments.csv",
        "control_predictions": "control_cross_role_predictions.csv",
        "control_run": "control_run_endpoints.csv",
        "control_patient": "control_patient_endpoints.csv",
        "unit_loadings": "crossfitted_unit_loadings.csv",
        "unit_run_metrics": "unit_participation_run_metrics.csv",
        "ablation_run": "unit_ablation_run_endpoints.csv",
        "ablation_patient": "unit_ablation_patient_endpoints.csv",
        "scrambled_pair": "scrambled_pair_projections.csv",
        "scrambled_run": "scrambled_run_endpoints.csv",
        "scrambled_patient": "scrambled_patient_endpoints.csv",
    }
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for unit_set in UNIT_SETS:
        directory = DATA_DIR / (
            "good_only" if unit_set == "good_only" else "sensitivity_usable_lenient"
        )
        frames[unit_set] = {
            key: pd.read_csv(directory / filename) for key, filename in file_map.items()
        }
        summaries[unit_set] = json.loads(
            (directory / "analysis_summary.json").read_text(encoding="utf-8")
        )
    return frames, summaries


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Frozen analysis contract is missing: {CONTRACT_PATH}")
    acoustic = build_acoustic_features()
    design_audit, vif = design_estimability_audit(acoustic)
    design_audit.to_csv(SOURCE_DIR / "design_estimability_audit.csv", index=False)
    vif.to_csv(SOURCE_DIR / "design_vif_audit.csv", index=False)

    if args.plot_only:
        frames, summaries = reload_completed_outputs()
        anchor_patient = pd.read_csv(
            ROOT
            / "neuropixel2025/data/combined_event_anchor_manifold_geometry_20260806"
            / "static_rank4_patient_anchor_comparison.csv"
        )
        anchor_summary = pd.read_csv(
            ROOT / "reports/event_anchor_manifold_geometry_20260806/event_anchor_summary.csv"
        )
        module_family = module_family_table(summaries, frames)
        module_family.to_csv(SOURCE_DIR / "module_family_correction.csv", index=False)
        plot_main_figure(frames, summaries, anchor_patient, anchor_summary)
        plot_panel_b_45x60mm(frames)
        plot_extended_figure(frames, design_audit, vif)
        plot_fig4_integration(frames, summaries)
        write_reports(
            summaries,
            frames,
            design_audit,
            module_family,
            anchor_summary,
        )
        validation = validate_outputs(frames, design_audit)
        write_json(REPORT_DIR / "validation.json", validation)
        write_json(DATA_DIR / "validation.json", validation)
        print(json.dumps({"stage": "plot-only complete", **validation}, ensure_ascii=False, default=json_default), flush=True)
        return

    selected = UNIT_SETS if args.unit_set == "all" else (args.unit_set,)
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    summaries: dict[str, dict[str, object]] = {}
    audit_frames: list[pd.DataFrame] = []
    for unit_set in selected:
        print(f"Loading frozen specification: {unit_set}", flush=True)
        metrics, runs, audit = load_specification(unit_set)
        audit_frames.append(audit)

        (
            static_predictions,
            static_folds,
            static_run,
            static_patient,
            static_null,
            static_summary,
        ) = static_analysis(unit_set, metrics, runs)
        print(f"{unit_set}: static analysis complete", flush=True)
        (
            fixed_run,
            fixed_patient,
            fixed_stats,
            trajectory_stats,
            time_segments,
            time_summary,
        ) = time_analysis(unit_set, metrics, runs)
        print(f"{unit_set}: time analysis complete", flush=True)
        (
            control_predictions,
            control_run,
            control_patient,
            control_summary,
        ) = control_analysis(unit_set, metrics, runs, acoustic)
        print(f"{unit_set}: control analysis complete", flush=True)
        (
            unit_loadings,
            unit_run_metrics,
            ablation_run,
            ablation_patient,
            unit_summary,
        ) = unit_contribution_analysis(unit_set, metrics, runs)
        print(f"{unit_set}: unit-contribution analysis complete", flush=True)
        (
            scrambled_pair,
            scrambled_run,
            scrambled_patient,
            scrambled_summary,
        ) = scrambled_analysis(unit_set, metrics, runs)
        print(f"{unit_set}: scrambled analysis complete", flush=True)

        frame_bundle = {
            "static_predictions": static_predictions,
            "static_folds": static_folds,
            "static_run": static_run,
            "static_patient": static_patient,
            "static_null": static_null,
            "fixed_run": fixed_run,
            "fixed_patient": fixed_patient,
            "fixed_stats": fixed_stats,
            "trajectory_stats": trajectory_stats,
            "time_segments": time_segments,
            "control_predictions": control_predictions,
            "control_run": control_run,
            "control_patient": control_patient,
            "unit_loadings": unit_loadings,
            "unit_run_metrics": unit_run_metrics,
            "ablation_run": ablation_run,
            "ablation_patient": ablation_patient,
            "scrambled_pair": scrambled_pair,
            "scrambled_run": scrambled_run,
            "scrambled_patient": scrambled_patient,
        }
        summary_bundle = {
            "analysis_contract": json.loads(CONTRACT_PATH.read_text(encoding="utf-8")),
            "static": static_summary,
            "time": time_summary,
            "controls": control_summary,
            "units": unit_summary,
            "scrambled": scrambled_summary,
        }
        frames[unit_set] = frame_bundle
        summaries[unit_set] = summary_bundle
        write_specification_outputs(unit_set, frame_bundle, summary_bundle)

    if set(frames) != set(UNIT_SETS):
        print("Single-specification run complete; integrated figures require --unit-set all")
        return

    inclusion_audit = pd.concat(audit_frames, ignore_index=True)
    inclusion_audit.to_csv(DATA_DIR / "inclusion_audit.csv", index=False)
    inclusion_audit.to_csv(SOURCE_DIR / "inclusion_audit.csv", index=False)

    combined_keys = [
        "static_patient",
        "fixed_stats",
        "trajectory_stats",
        "control_patient",
        "unit_run_metrics",
        "ablation_patient",
        "scrambled_patient",
    ]
    combined_names = {
        "static_patient": "static_patient_endpoints.csv",
        "fixed_stats": "fixed_time_window_stats.csv",
        "trajectory_stats": "continuous_time_patient_stats.csv",
        "control_patient": "control_patient_endpoints.csv",
        "unit_run_metrics": "unit_participation_run_metrics.csv",
        "ablation_patient": "unit_ablation_patient_endpoints.csv",
        "scrambled_patient": "scrambled_patient_endpoints.csv",
    }
    for key in combined_keys:
        pd.concat([frames[unit_set][key] for unit_set in UNIT_SETS], ignore_index=True).to_csv(
            SOURCE_DIR / combined_names[key], index=False
        )

    anchor_patient = pd.read_csv(
        ROOT
        / "neuropixel2025/data/combined_event_anchor_manifold_geometry_20260806"
        / "static_rank4_patient_anchor_comparison.csv"
    )
    anchor_summary = pd.read_csv(
        ROOT / "reports/event_anchor_manifold_geometry_20260806/event_anchor_summary.csv"
    )
    anchor_patient.to_csv(SOURCE_DIR / "event_anchor_patient_endpoints_good_only.csv", index=False)
    anchor_summary.to_csv(SOURCE_DIR / "event_anchor_summary_good_only.csv", index=False)

    module_family = module_family_table(summaries, frames)
    module_family.to_csv(SOURCE_DIR / "module_family_correction.csv", index=False)
    write_json(REPORT_DIR / "analysis_summary.json", summaries)
    write_json(DATA_DIR / "analysis_summary.json", summaries)

    if not args.skip_figure:
        plot_main_figure(frames, summaries, anchor_patient, anchor_summary)
        plot_panel_b_45x60mm(frames)
        plot_extended_figure(frames, design_audit, vif)
        plot_fig4_integration(frames, summaries)
    write_reports(
        summaries,
        frames,
        design_audit,
        module_family,
        anchor_summary,
    )
    validation = validate_outputs(frames, design_audit)
    write_json(REPORT_DIR / "validation.json", validation)
    write_json(DATA_DIR / "validation.json", validation)
    print(json.dumps({"stage": "complete", **validation}, ensure_ascii=False, default=json_default), flush=True)


if __name__ == "__main__":
    main()
