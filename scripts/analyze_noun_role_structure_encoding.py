#!/usr/bin/env python3
"""Analyze noun subject vs predicate-position structure coding."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


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


DATA_ROOT = ROOT / "neuropixel2025/data"
OUT_DIR = DATA_ROOT / "combined_noun_role_structure_encoding"
REPORT_PATH = ROOT / "reports/noun_role_structure_encoding_5patients.md"

PATIENTS = ("gz001", "gz002", "gz003", "gz004", "gz005")
UNIT_SET_FILE = {
    "usable_lenient": "firing_rate_1000Hz_usable_lenient_clusters.npz",
    "all_clusters": "firing_rate_1000Hz_all_clusters.npz",
    "good_clusters": "firing_rate_1000Hz_good_clusters.npz",
    "strict_qc": "firing_rate_1000Hz_strict_qc_clusters.npz",
}

PRE_MS = 500
POST_MS = 1200
BASELINE_SEC = (-0.5, 0.0)
RESPONSE_SEC = (0.0, 0.8)
SMOOTH_SIGMA_MS = 40
ALPHA = 0.05
TIME_ANOVA_ALPHA = 0.05
TIME_ANOVA_MIN_SEGMENT_MS = 50
TIME_ANOVA_MAX_GAP_MS = 20
SR = 16_000

ROLE_SUBJECT = "subject"
ROLE_PREDICATE = "predicate"
ACTIVE = "#2F6FBB"
PREDICATE = "#C44E35"


@dataclass(frozen=True)
class Stimulus:
    sound_file: str
    condition: str
    sentence: str
    function_word: str
    subject_noun: str
    predicate_noun: str


STIMULI: tuple[Stimulus, ...] = (
    Stimulus("stim_actpas/act001.mp3", "active", "虫子把树叶啃光了", "把", "虫子", "树叶"),
    Stimulus("stim_actpas/act002.mp3", "active", "交警给汽车拦住了", "给", "交警", "汽车"),
    Stimulus("stim_actpas/act003.mp3", "active", "小狗将骨头叼走了", "将", "小狗", "骨头"),
    Stimulus("stim_actpas/act004.mp3", "active", "妈妈把衣服晾好了", "把", "妈妈", "衣服"),
    Stimulus("stim_actpas/act005.mp3", "active", "球员给足球踢飞了", "给", "球员", "足球"),
    Stimulus("stim_actpas/act006.mp3", "active", "兔子把菜叶吃光了", "把", "兔子", "菜叶"),
    Stimulus("stim_actpas/act007.mp3", "active", "爸爸将爆竹点着了", "将", "爸爸", "爆竹"),
    Stimulus("stim_actpas/act008.mp3", "active", "婴儿把牛奶喝完了", "把", "婴儿", "牛奶"),
    Stimulus("stim_actpas/act009.mp3", "active", "奶奶将西瓜切开了", "将", "奶奶", "西瓜"),
    Stimulus("stim_actpas/act010.mp3", "active", "警察给道路封锁了", "给", "警察", "道路"),
    Stimulus("stim_actpas/pass001.mp3", "passive", "树叶被虫子啃光了", "被", "树叶", "虫子"),
    Stimulus("stim_actpas/pass002.mp3", "passive", "汽车让交警拦住了", "让", "汽车", "交警"),
    Stimulus("stim_actpas/pass003.mp3", "passive", "骨头叫小狗叼走了", "叫", "骨头", "小狗"),
    Stimulus("stim_actpas/pass004.mp3", "passive", "衣服被妈妈晾好了", "被", "衣服", "妈妈"),
    Stimulus("stim_actpas/pass005.mp3", "passive", "足球让球员踢飞了", "让", "足球", "球员"),
    Stimulus("stim_actpas/pass006.mp3", "passive", "菜叶被兔子吃光了", "被", "菜叶", "兔子"),
    Stimulus("stim_actpas/pass007.mp3", "passive", "爆竹叫爸爸点着了", "叫", "爆竹", "爸爸"),
    Stimulus("stim_actpas/pass008.mp3", "passive", "牛奶被婴儿喝完了", "被", "牛奶", "婴儿"),
    Stimulus("stim_actpas/pass009.mp3", "passive", "西瓜叫奶奶切开了", "叫", "西瓜", "奶奶"),
    Stimulus("stim_actpas/pass010.mp3", "passive", "道路让警察封锁了", "让", "道路", "警察"),
)

STIM_BY_SOUND = {stim.sound_file: stim for stim in STIMULI}
NOUN_ORDER = [
    "虫子",
    "交警",
    "小狗",
    "妈妈",
    "球员",
    "兔子",
    "爸爸",
    "婴儿",
    "奶奶",
    "警察",
    "树叶",
    "汽车",
    "骨头",
    "衣服",
    "足球",
    "菜叶",
    "爆竹",
    "牛奶",
    "西瓜",
    "道路",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-set",
        choices=sorted(UNIT_SET_FILE),
        default="usable_lenient",
        help="Cluster selection to analyze.",
    )
    parser.add_argument(
        "--top-units-per-run",
        type=int,
        default=8,
        help="Number of top p-value units per run to include in detailed noun-role pages.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[noun-role] {message}", flush=True)


def setup_matplotlib() -> FontProperties:
    font_paths = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    font_prop: FontProperties | None = None
    for font_path in font_paths:
        if Path(font_path).exists():
            fontManager.addfont(font_path)
            font_prop = FontProperties(fname=font_path)
            break
    if font_prop is None:
        font_prop = FontProperties()

    mpl.rcParams.update(
        {
            "font.family": font_prop.get_name(),
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    return font_prop


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    q[valid] = out
    return q


def gaussian_smooth_ms(x: np.ndarray, sigma_ms: float) -> np.ndarray:
    if sigma_ms <= 0:
        return x
    radius = int(round(sigma_ms * 4))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma_ms) ** 2)
    kernel /= kernel.sum()

    def smooth_vector(v: np.ndarray) -> np.ndarray:
        padded = np.pad(v, (radius, radius), mode="reflect")
        return np.convolve(padded, kernel, mode="same")[radius:-radius]

    return np.apply_along_axis(smooth_vector, -1, x)


def close_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = mask.copy()
    start: int | None = None
    for i, is_gap in enumerate(~mask):
        if is_gap and start is None:
            start = i
        elif not is_gap and start is not None:
            if start > 0 and i < len(mask) and i - start <= max_gap:
                mask[start:i] = True
            start = None
    if start is not None and start > 0 and len(mask) - start <= max_gap:
        mask[start:] = True
    return mask


def find_segments(mask: np.ndarray, time_sec: np.ndarray, min_len: int) -> list[tuple[int, int, float, float]]:
    segments: list[tuple[int, int, float, float]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_len:
                segments.append((start, i, float(time_sec[start]), float(time_sec[i - 1])))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask), float(time_sec[start]), float(time_sec[-1])))
    return segments


def sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    finite = np.isfinite(x)
    n = finite.sum(axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    out = sd / np.sqrt(np.maximum(n, 1))
    out = np.where(n <= 1, 0.0, out)
    return out


def audio_root() -> Path:
    for version in ("202511", "202509"):
        path = (
            ROOT
            / "neuropixel2025/files/实验设计/程序设计"
            / f"neuropixel_m255_{version}/ActPasSynSentence/scripts"
        )
        if (path / "stim_actpas/act001.mp3").exists():
            return path
    raise FileNotFoundError("Cannot find ActPasSynSentence audio directory")


def decode_mp3(path: Path) -> np.ndarray:
    cmd = [
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
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(x, kernel, mode="same")


def audio_envelope(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hop = int(0.005 * SR)
    frame = int(0.025 * SR)
    starts = np.arange(0, max(1, audio.size - frame), hop)
    rms = np.array([np.sqrt(np.mean(audio[s : s + frame] ** 2)) for s in starts])
    return starts / SR, moving_average(rms, 9)


def nearest_index(times: np.ndarray, value: float) -> int:
    return int(np.clip(np.searchsorted(times, value), 0, len(times) - 1))


def onset_from_peak(
    times: np.ndarray,
    env: np.ndarray,
    peak_sec: float,
    left_limit_sec: float,
) -> float:
    peak_i = nearest_index(times, peak_sec)
    left_limit_i = nearest_index(times, left_limit_sec)
    left_limit_i = min(left_limit_i, max(0, peak_i - 1))
    left_i = peak_i
    while left_i > left_limit_i and env[left_i - 1] <= env[left_i]:
        left_i -= 1
    if left_i == peak_i and peak_i > left_limit_i:
        left_i = left_limit_i + int(np.argmin(env[left_limit_i:peak_i]))
    if env[peak_i] <= env[left_i]:
        return float(times[max(left_limit_i, peak_i - 1)])
    threshold = float(env[left_i] + 0.25 * (env[peak_i] - env[left_i]))
    onset_i = left_i
    while onset_i < peak_i and env[onset_i] < threshold:
        onset_i += 1
    return float(times[onset_i])


def estimate_post_function_noun_onset(
    times: np.ndarray,
    env: np.ndarray,
    function_onset_sec: float,
) -> tuple[float, str]:
    start_i = nearest_index(times, function_onset_sec + 0.06)
    stop_i = nearest_index(times, function_onset_sec + 0.55)
    if stop_i <= start_i + 3:
        return function_onset_sec + 0.22, "fallback_function_plus_220ms"

    valley_stop_i = min(stop_i, nearest_index(times, function_onset_sec + 0.32))
    if valley_stop_i <= start_i:
        valley_i = start_i
    else:
        valley_i = start_i + int(np.argmin(env[start_i:valley_stop_i]))

    peak_start_i = max(valley_i + 1, nearest_index(times, function_onset_sec + 0.10))
    peak_stop_i = min(stop_i, nearest_index(times, function_onset_sec + 0.50))
    if peak_stop_i <= peak_start_i + 2:
        return function_onset_sec + 0.22, "fallback_function_plus_220ms"
    peak_i = peak_start_i + int(np.argmax(env[peak_start_i:peak_stop_i]))
    if env[peak_i] <= env[valley_i]:
        return function_onset_sec + 0.22, "fallback_function_plus_220ms"

    threshold = float(env[valley_i] + 0.22 * (env[peak_i] - env[valley_i]))
    onset_i = valley_i
    while onset_i < peak_i and env[onset_i] < threshold:
        onset_i += 1
    onset_sec = float(times[onset_i])
    if not (function_onset_sec + 0.06 <= onset_sec <= function_onset_sec + 0.50):
        return function_onset_sec + 0.22, "fallback_function_plus_220ms"
    return onset_sec, "audio_envelope_after_function_word"


def estimate_predicate_onset_from_peaks(
    times: np.ndarray,
    env: np.ndarray,
    function_onset_sec: float,
    peak_times_json: str,
) -> tuple[float | None, str | None]:
    try:
        peak_times = json.loads(peak_times_json)
    except json.JSONDecodeError:
        return None, None
    if len(peak_times) < 4:
        return None, None
    fourth_peak_sec = float(peak_times[3])
    if not (function_onset_sec + 0.05 <= fourth_peak_sec <= function_onset_sec + 0.50):
        return None, None
    onset_sec = onset_from_peak(
        times,
        env,
        fourth_peak_sec,
        left_limit_sec=function_onset_sec + 0.04,
    )
    if function_onset_sec + 0.04 <= onset_sec <= function_onset_sec + 0.50:
        return onset_sec, "audio_envelope_fourth_syllable_peak"
    return None, None


def first_patient_offsets_path() -> Path:
    for patient in PATIENTS:
        path = (
            DATA_ROOT
            / patient
            / "processed/ActPasSynSentence/event_related_active_passive/active_passive_function_word_audio_onsets.csv"
        )
        if path.exists():
            return path
    raise FileNotFoundError("No active_passive_function_word_audio_onsets.csv found")


def build_noun_audio_onsets(font_prop: FontProperties) -> pd.DataFrame:
    offsets = pd.read_csv(first_patient_offsets_path())
    offsets = offsets[offsets["sound_file"].isin(STIM_BY_SOUND)].copy()
    base_audio_dir = audio_root()
    rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(10, 2, figsize=(13, 22), sharex=False)
    for ax, row in zip(axes.flat, offsets.itertuples(index=False)):
        stim = STIM_BY_SOUND[str(row.sound_file)]
        audio = decode_mp3(base_audio_dir / str(row.sound_file))
        times, env = audio_envelope(audio)
        speech_start_sec = float(row.speech_start_sec_in_audio)
        function_onset_sec = float(row.target_onset_sec_in_audio)
        predicate_onset_audio_sec, predicate_source = estimate_predicate_onset_from_peaks(
            times,
            env,
            function_onset_sec,
            str(row.detected_peak_times_sec_in_audio),
        )
        if predicate_onset_audio_sec is None:
            predicate_onset_audio_sec, predicate_source = estimate_post_function_noun_onset(
                times,
                env,
                function_onset_sec,
            )
        subject_onset_sec = 0.0
        predicate_onset_sec = max(0.0, predicate_onset_audio_sec - speech_start_sec)

        rows.extend(
            [
                {
                    "sound_file": stim.sound_file,
                    "condition": stim.condition,
                    "sentence": stim.sentence,
                    "function_word": stim.function_word,
                    "role": ROLE_SUBJECT,
                    "noun": stim.subject_noun,
                    "noun_onset_sec_from_sentence_onset": subject_onset_sec,
                    "noun_onset_sec_in_audio": speech_start_sec,
                    "onset_source": "speech_start",
                },
                {
                    "sound_file": stim.sound_file,
                    "condition": stim.condition,
                    "sentence": stim.sentence,
                    "function_word": stim.function_word,
                    "role": ROLE_PREDICATE,
                    "noun": stim.predicate_noun,
                    "noun_onset_sec_from_sentence_onset": predicate_onset_sec,
                    "noun_onset_sec_in_audio": predicate_onset_audio_sec,
                    "onset_source": predicate_source,
                },
            ]
        )

        ax.plot(times, env, color="#333333", linewidth=1.0)
        ax.axvline(speech_start_sec, color=ACTIVE, linewidth=1.2, label="subject noun")
        ax.axvline(function_onset_sec, color="#777777", linewidth=1.0, linestyle="--", label="function word")
        ax.axvline(predicate_onset_audio_sec, color=PREDICATE, linewidth=1.2, label="predicate noun")
        ax.set_xlim(0, min(float(times[-1]), 2.25))
        ax.set_title(f"{Path(stim.sound_file).stem} {stim.sentence}", fontproperties=font_prop, fontsize=9)
        ax.set_ylabel("RMS")
    for ax in axes[-1]:
        ax.set_xlabel("Audio time (s)")
    axes.flat[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Noun-role audio locks: subject noun and post-function-word noun", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    qc_path = OUT_DIR / "fig_noun_role_audio_onset_qc.png"
    fig.savefig(qc_path, dpi=170)
    plt.close(fig)

    onsets = pd.DataFrame(rows)
    onsets_path = OUT_DIR / "noun_role_audio_onsets.csv"
    onsets.to_csv(onsets_path, index=False)
    return onsets


def active_passive_metadata(patient: str) -> pd.DataFrame:
    path = (
        DATA_ROOT
        / patient
        / "processed/ActPasSynSentence/event_related_active_passive/active_passive_event_related_trial_metadata.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df[df["sound_file"].isin(STIM_BY_SOUND)].copy()


def run_npz_path(patient: str, run: str, unit_set: str) -> Path:
    path = (
        DATA_ROOT
        / patient
        / "processed/ActPasSynSentence"
        / run
        / "firing_rate_1000Hz"
        / UNIT_SET_FILE[unit_set]
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_event_rows(onsets: pd.DataFrame) -> pd.DataFrame:
    onset_lookup = {
        (row.sound_file, row.role): row
        for row in onsets.itertuples(index=False)
    }
    event_rows: list[dict[str, object]] = []
    for patient in PATIENTS:
        meta = active_passive_metadata(patient)
        for trial in meta.itertuples(index=False):
            stim = STIM_BY_SOUND[str(trial.sound_file)]
            for role, noun, counterpart in (
                (ROLE_SUBJECT, stim.subject_noun, stim.predicate_noun),
                (ROLE_PREDICATE, stim.predicate_noun, stim.subject_noun),
            ):
                onset = onset_lookup[(stim.sound_file, role)]
                noun_onset_sec = float(onset.noun_onset_sec_from_sentence_onset)
                event_bin = int(trial.marker_bin) + int(round(noun_onset_sec * 1000))
                event_rows.append(
                    {
                        "patient": patient,
                        "run": str(trial.run),
                        "trial_index": int(trial.trial_index),
                        "sound_file": stim.sound_file,
                        "sentence_condition": stim.condition,
                        "sentence": stim.sentence,
                        "function_word": stim.function_word,
                        "role": role,
                        "noun": noun,
                        "counterpart_noun": counterpart,
                        "noun_onset_sec_from_sentence_onset": noun_onset_sec,
                        "event_bin_1000Hz": event_bin,
                        "event_time_sec": event_bin / 1000,
                        "onset_source": str(onset.onset_source),
                    }
                )
    events = pd.DataFrame(event_rows)
    events.to_csv(OUT_DIR / "noun_role_event_metadata.csv", index=False)
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
                        "sentence": str(event.sentence),
                        "role": str(event.role),
                        "noun": str(event.noun),
                        "counterpart_noun": str(event.counterpart_noun),
                        "event_bin_1000Hz": event_bin,
                        "baseline_hz": baseline.astype(float),
                        "response_hz": response.astype(float),
                        "response_baseline_corrected_hz": (response - baseline).astype(float),
                    }
                )
            )

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(OUT_DIR / "noun_role_unit_trial_response_metrics.csv", index=False)
    return metrics, run_meta


def safe_wilcoxon(diffs: np.ndarray) -> tuple[float, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0 or np.allclose(diffs, 0):
        return np.nan, 1.0
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros.*")
            warnings.filterwarnings("ignore", message="Sample size too small for normal approximation.*")
            result = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, 1.0


def analyze_units(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (patient, run, unit_id), group in metrics.groupby(["patient", "run", "unit_id"], sort=False):
        pivot = group.pivot_table(
            index="noun",
            columns="role",
            values="response_baseline_corrected_hz",
            aggfunc="mean",
        ).reindex(NOUN_ORDER)
        if ROLE_SUBJECT not in pivot or ROLE_PREDICATE not in pivot:
            continue
        valid = pivot[[ROLE_SUBJECT, ROLE_PREDICATE]].dropna()
        diffs = valid[ROLE_SUBJECT].to_numpy(float) - valid[ROLE_PREDICATE].to_numpy(float)
        stat, p_value = safe_wilcoxon(diffs)
        mean_subject = float(valid[ROLE_SUBJECT].mean()) if len(valid) else np.nan
        mean_predicate = float(valid[ROLE_PREDICATE].mean()) if len(valid) else np.nan
        mean_diff = float(np.mean(diffs)) if len(diffs) else np.nan
        sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else np.nan
        dz = float(mean_diff / sd_diff) if sd_diff and np.isfinite(sd_diff) and sd_diff > 0 else np.nan
        rows.append(
            {
                "patient": patient,
                "run": run,
                "unit_id": int(unit_id),
                "ks_label": str(group["ks_label"].iloc[0]),
                "n_noun_pairs": int(len(valid)),
                "subject_mean_bc_hz": mean_subject,
                "predicate_mean_bc_hz": mean_predicate,
                "subject_minus_predicate_bc_hz": mean_diff,
                "paired_cohens_dz": dz,
                "wilcoxon_statistic": stat,
                "p_value": p_value,
                "direction": "subject_gt_predicate"
                if mean_diff > 0
                else "predicate_gt_subject"
                if mean_diff < 0
                else "no_difference",
            }
        )

    results = pd.DataFrame(rows)
    results["q_value_global_fdr"] = bh_fdr(results["p_value"].to_numpy())
    results["q_value_run_fdr"] = np.nan
    results["q_value_patient_fdr"] = np.nan
    for (_, _), idx in results.groupby(["patient", "run"]).groups.items():
        results.loc[idx, "q_value_run_fdr"] = bh_fdr(results.loc[idx, "p_value"].to_numpy())
    for _, idx in results.groupby("patient").groups.items():
        results.loc[idx, "q_value_patient_fdr"] = bh_fdr(results.loc[idx, "p_value"].to_numpy())
    results["significant_uncorrected_p05"] = results["p_value"] < ALPHA
    results["significant_run_fdr05"] = results["q_value_run_fdr"] < ALPHA
    results["significant_patient_fdr05"] = results["q_value_patient_fdr"] < ALPHA
    results["significant_global_fdr05"] = results["q_value_global_fdr"] < ALPHA
    results = results.sort_values(["q_value_run_fdr", "p_value", "patient", "run", "unit_id"])
    results.to_csv(OUT_DIR / "noun_role_structure_encoding_units.csv", index=False)
    results.loc[results["significant_uncorrected_p05"]].to_csv(
        OUT_DIR / "noun_role_candidate_units_uncorrected_p05.csv",
        index=False,
    )
    results.loc[results["significant_run_fdr05"]].to_csv(
        OUT_DIR / "noun_role_significant_units_run_fdr05.csv",
        index=False,
    )
    return results


def matrix_for_role(run_metrics: pd.DataFrame, role: str, units: list[int]) -> pd.DataFrame:
    return (
        run_metrics[run_metrics["role"] == role]
        .pivot_table(index="unit_id", columns="noun", values="response_baseline_corrected_hz", aggfunc="mean")
        .reindex(index=units, columns=NOUN_ORDER)
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


def uncorrected_candidate_codes(run_results: pd.DataFrame, units: list[int]) -> np.ndarray:
    stats_by_unit = run_results.set_index("unit_id").reindex(units)
    codes = np.zeros(len(units), dtype=int)
    is_candidate = stats_by_unit["significant_uncorrected_p05"].fillna(False).to_numpy(bool)
    directions = stats_by_unit["direction"].fillna("").to_numpy(str)
    codes[is_candidate & (directions == "subject_gt_predicate")] = 1
    codes[is_candidate & (directions == "predicate_gt_subject")] = 2
    return codes[:, None]


def plot_all_unit_heatmaps(metrics: pd.DataFrame, results: pd.DataFrame, font_prop: FontProperties) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    pdf_path = fig_dir / "noun_role_all_units_response_heatmaps_by_run.pdf"
    png_paths: list[str] = []
    run_order = list(metrics[["patient", "run"]].drop_duplicates().itertuples(index=False, name=None))

    with PdfPages(pdf_path) as pdf:
        for patient, run in run_order:
            run_metrics = metrics[(metrics["patient"] == patient) & (metrics["run"] == run)]
            run_results = results[(results["patient"] == patient) & (results["run"] == run)].copy()
            units = (
                run_results.sort_values("subject_minus_predicate_bc_hz", ascending=False)["unit_id"]
                .astype(int)
                .tolist()
            )
            subject_mat = matrix_for_role(run_metrics, ROLE_SUBJECT, units)
            predicate_mat = matrix_for_role(run_metrics, ROLE_PREDICATE, units)
            diff_mat = subject_mat - predicate_mat
            vals = np.concatenate(
                [
                    subject_mat.to_numpy(float).ravel(),
                    predicate_mat.to_numpy(float).ravel(),
                    diff_mat.to_numpy(float).ravel(),
                ]
            )
            finite = vals[np.isfinite(vals)]
            vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
            vmax = max(vmax, 1e-6)
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            height = min(18, max(7, 0.055 * len(units) + 3.5))
            fig = plt.figure(figsize=(20, height))
            gs = fig.add_gridspec(1, 5, width_ratios=[0.035, 1, 1, 1, 0.035], wspace=0.06)
            ax_sig = fig.add_subplot(gs[0, 0])
            axes = [
                fig.add_subplot(gs[0, 1], sharey=ax_sig),
                fig.add_subplot(gs[0, 2], sharey=ax_sig),
                fig.add_subplot(gs[0, 3], sharey=ax_sig),
            ]
            cax = fig.add_subplot(gs[0, 4])
            candidate_codes = uncorrected_candidate_codes(run_results, units)
            candidate_cmap = mpl.colors.ListedColormap(["#F2F2F2", ACTIVE, PREDICATE])
            ax_sig.imshow(candidate_codes, aspect="auto", cmap=candidate_cmap, vmin=0, vmax=2)
            ax_sig.set_title("p<0.05\ncandidate", fontsize=8)
            ax_sig.set_xticks([])
            ax_sig.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            for spine in ax_sig.spines.values():
                spine.set_visible(False)
            for ax, mat, title in zip(
                axes,
                [subject_mat, predicate_mat, diff_mat],
                ["subject noun", "predicate-position noun", "subject - predicate"],
            ):
                im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap="RdBu_r", norm=norm)
                ax.set_title(title)
                ax.set_xticks(np.arange(len(NOUN_ORDER)))
                ax.set_xticklabels(NOUN_ORDER, rotation=90, fontproperties=font_prop)
                ax.set_xlabel("noun")
                sparse_unit_ticks(ax, units)
            axes[0].set_ylabel("")
            axes[1].tick_params(labelleft=False)
            axes[2].tick_params(labelleft=False)
            fig.colorbar(im, cax=cax, label="baseline-corrected Hz")
            fig.text(0.018, 0.53, "unit id, sorted by mean role effect", rotation=90, va="center", fontsize=9)
            fig.legend(
                handles=[
                    mpl.patches.Patch(color=ACTIVE, label="uncorrected p<0.05 subject-pref"),
                    mpl.patches.Patch(color=PREDICATE, label="uncorrected p<0.05 predicate-pref"),
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=2,
                frameon=False,
                fontsize=8,
            )
            unc = run_results[run_results["significant_uncorrected_p05"]]
            n_unc_subject = int((unc["direction"] == "subject_gt_predicate").sum())
            n_unc_predicate = int((unc["direction"] == "predicate_gt_subject").sum())
            n_unc = int(len(unc))
            n_fdr = int(run_results["significant_run_fdr05"].sum())
            fig.suptitle(
                f"{patient} {run}: unit x noun role response, "
                f"uncorrected p<0.05 candidates={n_unc} "
                f"(subject={n_unc_subject}, predicate={n_unc_predicate}); run-FDR<0.05={n_fdr}",
                y=0.995,
            )
            fig.subplots_adjust(left=0.055, right=0.965, bottom=0.18, top=0.94)
            pdf.savefig(fig)
            png_path = fig_dir / f"noun_role_all_units_heatmap_{patient}_{run}.png"
            fig.savefig(png_path, dpi=170)
            png_paths.append(str(png_path))
            plt.close(fig)

    return {"pdf": str(pdf_path), "png_dir": str(fig_dir), "first_png": png_paths[0] if png_paths else ""}


def shortened_run_name(run: str) -> str:
    return (
        run.replace("_actpas_", "_")
        .replace("gz001t_g0_", "g0_")
        .replace("gz002_g0_", "g0_")
        .replace("gz003t2_g0_", "g0_")
        .replace("gz004t1_g0_", "g0_")
        .replace("gz004t1_g3_", "g3_")
        .replace("gz005t0_g0_", "g0_")
        .replace("gz005t0_g3_", "g3_")
    )


def plot_explanatory_summary(
    results: pd.DataFrame,
    font_prop: FontProperties,
    events: pd.DataFrame | None = None,
    unit_set: str = "usable_lenient",
) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_noun_role_structure_encoding_explanatory_summary.png"
    pdf_path = fig_dir / "fig_noun_role_structure_encoding_explanatory_summary.pdf"

    run_order = list(results[["patient", "run"]].drop_duplicates().itertuples(index=False, name=None))
    run_labels = [f"{patient}\n{shortened_run_name(run)}" for patient, run in run_order]

    fig = plt.figure(figsize=(18, 10.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.05], hspace=0.38, wspace=0.34)
    ax_schematic = fig.add_subplot(gs[0, 0])
    ax_counts = fig.add_subplot(gs[0, 1])
    ax_example_subject = fig.add_subplot(gs[0, 2])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_volcano = fig.add_subplot(gs[1, 1])
    ax_example_predicate = fig.add_subplot(gs[1, 2])

    ax_schematic.axis("off")
    ax_schematic.set_title("Same noun, different structural role", loc="left", fontsize=12, weight="bold")
    ax_schematic.text(0.02, 0.84, "Example pair", fontsize=9, color="0.35")
    examples = [
        [("虫子", ACTIVE, "subject"), ("把", "#555555", ""), ("树叶", PREDICATE, "predicate"), ("啃光了", "#222222", "")],
        [("树叶", ACTIVE, "subject"), ("被", "#555555", ""), ("虫子", PREDICATE, "predicate"), ("啃光了", "#222222", "")],
    ]
    y_positions = [0.64, 0.37]
    for y, parts in zip(y_positions, examples):
        x = 0.04
        for text, color, role in parts:
            ax_schematic.text(
                x,
                y,
                text,
                fontproperties=font_prop,
                fontsize=18 if role else 15,
                color=color,
                weight="bold" if role else "normal",
                va="center",
            )
            if role:
                ax_schematic.text(
                    x,
                    y - 0.11,
                    role,
                    fontsize=8,
                    color=color,
                    va="center",
                )
            x += 0.13 if len(text) <= 2 else 0.18
    pref_summary = []
    for label, group in (
        ("All units", results),
        ("Uncorrected p<0.05", results[results["significant_uncorrected_p05"]]),
        ("Run-FDR<0.05", results[results["significant_run_fdr05"]]),
    ):
        counts = group["direction"].value_counts()
        pref_summary.append(
            f"{label}: {len(group)} units; subject {int(counts.get('subject_gt_predicate', 0))}, "
            f"predicate {int(counts.get('predicate_gt_subject', 0))}"
        )
    ax_schematic.text(
        0.02,
        0.14,
        "Per unit: compare each noun in subject position\n"
        "against the same noun in predicate-position.",
        fontsize=9.2,
        color="0.25",
        va="top",
    )
    ax_schematic.text(0.02, -0.02, "\n".join(pref_summary), fontsize=8.8, color="0.20", va="top")

    sig_counts = []
    for patient, run in run_order:
        group = results[
            (results["patient"] == patient)
            & (results["run"] == run)
            & results["significant_uncorrected_p05"]
        ]
        counts = group["direction"].value_counts()
        sig_counts.append(
            {
                "subject": int(counts.get("subject_gt_predicate", 0)),
                "predicate": int(counts.get("predicate_gt_subject", 0)),
                "total": int(len(group)),
            }
        )
    y = np.arange(len(run_labels))
    subject_counts = np.array([row["subject"] for row in sig_counts])
    predicate_counts = np.array([row["predicate"] for row in sig_counts])
    ax_counts.barh(y, subject_counts, color=ACTIVE, alpha=0.88, label="subject-pref")
    ax_counts.barh(y, -predicate_counts, color=PREDICATE, alpha=0.88, label="predicate-pref")
    for yi, row in enumerate(sig_counts):
        if row["subject"]:
            ax_counts.text(row["subject"] + 0.25, yi, str(row["subject"]), va="center", fontsize=8)
        if row["predicate"]:
            ax_counts.text(-row["predicate"] - 0.25, yi, str(row["predicate"]), va="center", ha="right", fontsize=8)
    ax_counts.axvline(0, color="0.35", lw=0.8)
    ax_counts.set_yticks(y)
    ax_counts.set_yticklabels(run_labels)
    ax_counts.tick_params(axis="y", labelsize=8.5)
    ax_counts.set_xlabel("uncorrected p<0.05 candidate units")
    ax_counts.set_title("Direction of uncorrected candidate units", loc="left", fontsize=12, weight="bold")
    ax_counts.legend(frameon=False, loc="lower right")
    max_count = max(1, int(np.max(np.abs(np.r_[subject_counts, predicate_counts]))))
    ax_counts.set_xlim(-max_count - 2, max_count + 2)
    ax_counts.grid(axis="x", color="0.9", lw=0.7)

    def plot_role_trace(ax: plt.Axes, row: pd.Series | None, expected_direction: str) -> None:
        if events is None or row is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "Example trace unavailable", ha="center", va="center", color="0.4")
            return
        time_sec, traces, _ = extract_unit_role_traces(
            str(row["patient"]),
            str(row["run"]),
            int(row["unit_id"]),
            events,
            unit_set,
        )
        subject = traces[ROLE_SUBJECT]
        predicate = traces[ROLE_PREDICATE]
        for arr, color, label in (
            (subject, ACTIVE, "subject"),
            (predicate, PREDICATE, "predicate"),
        ):
            mean = np.nanmean(arr, axis=0)
            n = np.sum(np.isfinite(arr), axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
            ax.fill_between(time_sec, mean - sem, mean + sem, color=color, alpha=0.16, lw=0)
            ax.plot(time_sec, mean, color=color, lw=1.8, label=label)
            noun_means = np.nanmean(arr[:, (time_sec >= 0) & (time_sec <= 0.8)], axis=1)
            ax.scatter(
                np.full_like(noun_means, 1.06 if label == "subject" else 1.12, dtype=float),
                noun_means,
                color=color,
                s=13,
                alpha=0.58,
                edgecolors="none",
                clip_on=False,
            )
        ax.axvline(0, color="0.25", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.set_xlim(-0.25, 1.18)
        ax.set_xlabel("time from noun onset (s)")
        ax.set_ylabel("baseline-corrected Hz")
        direction_label = "subject-pref" if expected_direction == "subject_gt_predicate" else "predicate-pref"
        ax.set_title(
            f"Example {direction_label}: {row['patient']} u{int(row['unit_id'])}",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        ax.text(
            0.02,
            0.96,
            f"effect={float(row['subject_minus_predicate_bc_hz']):.1f} Hz\n"
            f"p={float(row['p_value']):.2g}, q_run={float(row.get('q_value_run_fdr', np.nan)):.2g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="0.25",
        )
        ax.legend(frameon=False, loc="lower left", fontsize=8)
        ax.grid(color="0.92", lw=0.7)

    significant = results[results["significant_uncorrected_p05"]].copy()
    subject_examples = significant[significant["direction"] == "subject_gt_predicate"].sort_values("p_value")
    predicate_examples = significant[significant["direction"] == "predicate_gt_subject"].sort_values("p_value")
    if subject_examples.empty:
        subject_examples = results[results["direction"] == "subject_gt_predicate"].sort_values("p_value")
    if predicate_examples.empty:
        predicate_examples = results[results["direction"] == "predicate_gt_subject"].sort_values("p_value")
    plot_role_trace(
        ax_example_subject,
        subject_examples.iloc[0] if not subject_examples.empty else None,
        "subject_gt_predicate",
    )
    plot_role_trace(
        ax_example_predicate,
        predicate_examples.iloc[0] if not predicate_examples.empty else None,
        "predicate_gt_subject",
    )

    distributions = [
        results[(results["patient"] == patient) & (results["run"] == run)]["subject_minus_predicate_bc_hz"].to_numpy()
        for patient, run in run_order
    ]
    ax_dist.boxplot(distributions, showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for patch in ax_dist.artists:
        patch.set_facecolor("#EAEAEA")
        patch.set_edgecolor("0.45")
    for i, (patient, run) in enumerate(run_order, start=1):
        group = results[(results["patient"] == patient) & (results["run"] == run)]
        sig = group[group["significant_uncorrected_p05"]]
        if not sig.empty:
            colors = np.where(sig["direction"].to_numpy() == "subject_gt_predicate", ACTIVE, PREDICATE)
            ax_dist.scatter(
                np.full(len(sig), i),
                sig["subject_minus_predicate_bc_hz"],
                s=26,
                color=colors,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
    ax_dist.axhline(0, color="0.25", lw=0.9)
    ax_dist.set_xticks(np.arange(1, len(run_labels) + 1))
    ax_dist.set_xticklabels(run_labels, rotation=45, ha="right")
    ax_dist.set_ylabel("subject - predicate response (Hz)")
    ax_dist.set_title("Effect-size distribution by run", loc="left", fontsize=12, weight="bold")
    ax_dist.grid(axis="y", color="0.9", lw=0.7)

    x = results["subject_minus_predicate_bc_hz"].to_numpy(float)
    yv = -np.log10(np.clip(results["p_value"].to_numpy(float), 1e-12, 1.0))
    colors = np.full(len(results), "#BDBDBD", dtype=object)
    colors[(results["direction"] == "subject_gt_predicate").to_numpy() & results["significant_uncorrected_p05"].to_numpy()] = ACTIVE
    colors[(results["direction"] == "predicate_gt_subject").to_numpy() & results["significant_uncorrected_p05"].to_numpy()] = PREDICATE
    sizes = np.where(results["significant_uncorrected_p05"].to_numpy(), 34, 12)
    ax_volcano.scatter(x, yv, c=colors, s=sizes, alpha=0.82, edgecolors="none")
    ax_volcano.axvline(0, color="0.3", lw=0.8)
    ax_volcano.axhline(-np.log10(0.05), color="0.45", lw=0.8, ls="--")
    for row in results[results["significant_uncorrected_p05"]].sort_values("p_value").head(12).itertuples(index=False):
        ax_volcano.text(
            float(row.subject_minus_predicate_bc_hz),
            -np.log10(max(float(row.p_value), 1e-12)) + 0.08,
            f"{row.patient} u{int(row.unit_id)}",
            fontsize=7,
            ha="center",
            color="0.2",
        )
    ax_volcano.set_xlabel("subject - predicate response (Hz)")
    ax_volcano.set_ylabel("-log10 paired noun-role p")
    ax_volcano.set_title("Unit-level role effect and uncorrected significance", loc="left", fontsize=12, weight="bold")
    ax_volcano.grid(color="0.92", lw=0.7)

    fig.suptitle("Noun structural-role encoding across 5 patients", fontsize=15, weight="bold", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.975, top=0.93, bottom=0.08)
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def selected_detailed_units(results: pd.DataFrame, top_units_per_run: int) -> pd.DataFrame:
    selections = []
    for (patient, run), group in results.groupby(["patient", "run"], sort=False):
        group = group.sort_values(["p_value", "unit_id"]).copy()
        sig = group[group["significant_run_fdr05"]].copy()
        sig["plot_selection_reason"] = "run_fdr05"
        top = group.head(top_units_per_run).copy()
        top["plot_selection_reason"] = "top_p"
        selected = pd.concat([sig, top], ignore_index=True)
        selected = selected.drop_duplicates(["patient", "run", "unit_id"], keep="first")
        selections.append(selected)
    out = pd.concat(selections, ignore_index=True)
    out.to_csv(OUT_DIR / "noun_role_detailed_unit_plot_selection.csv", index=False)
    return out


def extract_unit_role_traces(
    patient: str,
    run: str,
    unit_id: int,
    events: pd.DataFrame,
    unit_set: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    z = np.load(run_npz_path(patient, run, unit_set), allow_pickle=True)
    firing_rate = z["firing_rate_hz"]
    unit_ids = z["unit_ids"].astype(int)
    matches = np.flatnonzero(unit_ids == int(unit_id))
    if len(matches) == 0:
        raise ValueError(f"unit {unit_id} not found in {patient} {run}")
    unit_i = int(matches[0])
    time_sec = np.arange(-PRE_MS, POST_MS + 1) / 1000
    baseline_mask = (time_sec >= BASELINE_SEC[0]) & (time_sec < BASELINE_SEC[1])
    response_mask = (time_sec >= RESPONSE_SEC[0]) & (time_sec <= RESPONSE_SEC[1])

    traces = {
        ROLE_SUBJECT: np.full((len(NOUN_ORDER), len(time_sec)), np.nan, dtype=float),
        ROLE_PREDICATE: np.full((len(NOUN_ORDER), len(time_sec)), np.nan, dtype=float),
    }
    responses = {
        ROLE_SUBJECT: np.full(len(NOUN_ORDER), np.nan, dtype=float),
        ROLE_PREDICATE: np.full(len(NOUN_ORDER), np.nan, dtype=float),
    }

    run_events = events[(events["patient"] == patient) & (events["run"] == run)]
    for event in run_events.itertuples(index=False):
        noun_i = NOUN_ORDER.index(str(event.noun))
        event_bin = int(event.event_bin_1000Hz)
        start = event_bin - PRE_MS
        stop = event_bin + POST_MS + 1
        if start < 0 or stop > firing_rate.shape[1]:
            continue
        trace = firing_rate[unit_i, start:stop].astype(float)
        baseline = float(trace[baseline_mask].mean())
        trace_bc = trace - baseline
        traces[str(event.role)][noun_i] = gaussian_smooth_ms(trace_bc[None, :], SMOOTH_SIGMA_MS)[0]
        responses[str(event.role)][noun_i] = float(trace_bc[response_mask].mean())
    return time_sec, traces, responses


def timepoint_role_anova(
    subject_traces: np.ndarray,
    predicate_traces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeated-measures role ANOVA at each time bin.

    With two role levels, the within-noun repeated-measures ANOVA F statistic
    is the paired t statistic squared.
    """
    n_time = subject_traces.shape[1]
    f_values = np.full(n_time, np.nan, dtype=float)
    p_values = np.full(n_time, np.nan, dtype=float)
    n_pairs = np.zeros(n_time, dtype=int)

    for t in range(n_time):
        subj = subject_traces[:, t]
        pred = predicate_traces[:, t]
        valid = np.isfinite(subj) & np.isfinite(pred)
        diffs = subj[valid] - pred[valid]
        n = len(diffs)
        n_pairs[t] = n
        if n < 2:
            continue
        sd = float(np.std(diffs, ddof=1))
        if sd == 0 or not np.isfinite(sd):
            f_values[t] = 0.0
            p_values[t] = 1.0
            continue
        t_stat = float(np.mean(diffs) / (sd / np.sqrt(n)))
        f_stat = t_stat * t_stat
        f_values[t] = f_stat
        p_values[t] = float(stats.f.sf(f_stat, 1, n - 1))
    return f_values, p_values, n_pairs


def plot_detailed_units(
    selected: pd.DataFrame,
    events: pd.DataFrame,
    unit_set: str,
    font_prop: FontProperties,
) -> dict[str, str]:
    fig_dir = OUT_DIR / "detailed_unit_pages"
    fig_dir.mkdir(exist_ok=True)
    pdf_path = fig_dir / "noun_role_detailed_candidate_unit_pages.pdf"
    png_paths: list[str] = []
    segment_rows: list[dict[str, object]] = []
    timepoint_rows: list[dict[str, object]] = []

    with PdfPages(pdf_path) as pdf:
        for page_i, row in enumerate(selected.itertuples(index=False), start=1):
            time_sec, traces, responses = extract_unit_role_traces(
                str(row.patient),
                str(row.run),
                int(row.unit_id),
                events,
                unit_set,
            )
            subj = traces[ROLE_SUBJECT]
            pred = traces[ROLE_PREDICATE]
            diff = subj - pred
            f_values, anova_p_values, n_pairs = timepoint_role_anova(subj, pred)
            sig_mask = np.isfinite(anova_p_values) & (anova_p_values < TIME_ANOVA_ALPHA)
            sig_mask = close_short_gaps(sig_mask, TIME_ANOVA_MAX_GAP_MS)
            sig_segments = find_segments(sig_mask, time_sec, TIME_ANOVA_MIN_SEGMENT_MS)
            segment_lookup = np.full(len(time_sec), "", dtype=object)
            for segment_i, (start_i, stop_i, start_sec, stop_sec) in enumerate(sig_segments, start=1):
                segment_id = f"{row.patient}_{row.run}_unit{int(row.unit_id)}_seg{segment_i:02d}"
                segment_lookup[start_i:stop_i] = segment_id
                segment_rows.append(
                    {
                        "patient": str(row.patient),
                        "run": str(row.run),
                        "unit_id": int(row.unit_id),
                        "ks_label": str(row.ks_label),
                        "segment_id": segment_id,
                        "segment_start_sec": start_sec,
                        "segment_stop_sec": stop_sec,
                        "duration_ms": int(stop_i - start_i),
                        "min_p_value": float(np.nanmin(anova_p_values[start_i:stop_i])),
                        "max_f_value": float(np.nanmax(f_values[start_i:stop_i])),
                        "mean_subject_minus_predicate_hz": float(np.nanmean(diff[:, start_i:stop_i])),
                    }
                )
            for t_i, time_value in enumerate(time_sec):
                timepoint_rows.append(
                    {
                        "patient": str(row.patient),
                        "run": str(row.run),
                        "unit_id": int(row.unit_id),
                        "ks_label": str(row.ks_label),
                        "time_sec": float(time_value),
                        "n_noun_pairs": int(n_pairs[t_i]),
                        "anova_f_role": float(f_values[t_i]) if np.isfinite(f_values[t_i]) else np.nan,
                        "anova_p_role": float(anova_p_values[t_i]) if np.isfinite(anova_p_values[t_i]) else np.nan,
                        "significant_uncorrected_p05": bool(
                            np.isfinite(anova_p_values[t_i]) and anova_p_values[t_i] < TIME_ANOVA_ALPHA
                        ),
                        "marked_segment_id": str(segment_lookup[t_i]),
                    }
                )
            vals = np.concatenate([subj.ravel(), pred.ravel(), diff.ravel()])
            finite = vals[np.isfinite(vals)]
            vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
            vmax = max(vmax, 1e-6)
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

            fig = plt.figure(figsize=(14, 13.5))
            gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 2.1, 0.85], hspace=0.35, wspace=0.28)
            ax_trace = fig.add_subplot(gs[0, :2])
            ax_scatter = fig.add_subplot(gs[0, 2])
            ax_subj = fig.add_subplot(gs[1, 0])
            ax_pred = fig.add_subplot(gs[1, 1], sharey=ax_subj)
            ax_diff = fig.add_subplot(gs[1, 2], sharey=ax_subj)
            ax_text = fig.add_subplot(gs[2, :])

            subj_mean = np.nanmean(subj, axis=0)
            pred_mean = np.nanmean(pred, axis=0)
            subj_sem = sem(subj, axis=0)
            pred_sem = sem(pred, axis=0)
            ax_trace.plot(time_sec, subj_mean, color=ACTIVE, lw=1.8, label="subject")
            ax_trace.fill_between(time_sec, subj_mean - subj_sem, subj_mean + subj_sem, color=ACTIVE, alpha=0.16, lw=0)
            ax_trace.plot(time_sec, pred_mean, color=PREDICATE, lw=1.8, label="predicate")
            ax_trace.fill_between(time_sec, pred_mean - pred_sem, pred_mean + pred_sem, color=PREDICATE, alpha=0.16, lw=0)
            ax_trace.axvline(0, color="black", lw=0.9, linestyle="--")
            ax_trace.axhline(0, color="0.78", lw=0.8)
            ax_trace.set_xlim(time_sec[0], time_sec[-1])
            ax_trace.set_xlabel("time from noun onset (s)")
            ax_trace.set_ylabel("baseline-corrected Hz")
            sig_label_used = False
            for _, _, start_sec, stop_sec in sig_segments:
                ax_trace.axvspan(
                    start_sec,
                    stop_sec,
                    ymin=0.955,
                    ymax=0.995,
                    color="#7B3294",
                    alpha=0.45,
                    lw=0,
                    label="role ANOVA p<0.05" if not sig_label_used else None,
                )
                sig_label_used = True
            ax_trace.legend(frameon=False, loc="upper right")

            x_subject = np.ones(len(NOUN_ORDER))
            x_predicate = np.ones(len(NOUN_ORDER)) * 2
            for i in range(len(NOUN_ORDER)):
                ax_scatter.plot(
                    [1, 2],
                    [responses[ROLE_SUBJECT][i], responses[ROLE_PREDICATE][i]],
                    color="0.75",
                    lw=0.8,
                    zorder=1,
                )
            ax_scatter.scatter(x_subject, responses[ROLE_SUBJECT], color=ACTIVE, s=18, zorder=2)
            ax_scatter.scatter(x_predicate, responses[ROLE_PREDICATE], color=PREDICATE, s=18, zorder=2)
            ax_scatter.axhline(0, color="0.78", lw=0.8)
            ax_scatter.set_xticks([1, 2])
            ax_scatter.set_xticklabels(["subject", "predicate"])
            ax_scatter.set_ylabel("0-0.8 s bc Hz")
            ax_scatter.set_xlim(0.55, 2.45)

            for ax, mat, title in (
                (ax_subj, subj, "subject nouns"),
                (ax_pred, pred, "predicate-position nouns"),
                (ax_diff, diff, "subject - predicate"),
            ):
                im = ax.imshow(
                    mat,
                    aspect="auto",
                    cmap="RdBu_r",
                    norm=norm,
                    extent=[time_sec[0], time_sec[-1], len(NOUN_ORDER) - 0.5, -0.5],
                )
                ax.axvline(0, color="black", lw=0.6, linestyle="--")
                ax.set_title(title)
                ax.set_xlabel("time (s)")
                ax.set_yticks(np.arange(len(NOUN_ORDER)))
                ax.set_yticklabels(NOUN_ORDER, fontproperties=font_prop, fontsize=8)
            ax_pred.tick_params(labelleft=False)
            ax_diff.tick_params(labelleft=False)
            fig.colorbar(im, ax=[ax_subj, ax_pred, ax_diff], fraction=0.025, pad=0.012, label="bc Hz")

            ax_text.axis("off")
            noun_rows = []
            diffs = responses[ROLE_SUBJECT] - responses[ROLE_PREDICATE]
            order = np.argsort(np.abs(diffs))[::-1][:10]
            for noun_i in order:
                noun_rows.append(
                    f"{NOUN_ORDER[noun_i]}: subj={responses[ROLE_SUBJECT][noun_i]:.2f}, "
                    f"pred={responses[ROLE_PREDICATE][noun_i]:.2f}, diff={diffs[noun_i]:.2f} Hz"
                )
            ax_text.text(
                0.01,
                0.98,
                "Largest noun-level role differences (0-0.8 s, baseline-corrected):\n"
                + "\n".join(noun_rows),
                va="top",
                ha="left",
                fontproperties=font_prop,
                fontsize=9,
            )

            title = (
                f"{row.patient} {row.run} unit {int(row.unit_id)} ({row.ks_label}) | "
                f"mean diff={row.subject_minus_predicate_bc_hz:.2f} Hz, "
                f"p={row.p_value:.3g}, q_run={row.q_value_run_fdr:.3g}, "
                f"{row.plot_selection_reason}"
            )
            fig.suptitle(title, y=0.995)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible.*")
                fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            png_path = fig_dir / f"page_{page_i:03d}_{row.patient}_{row.run}_unit{int(row.unit_id)}.png"
            fig.savefig(png_path, dpi=170)
            png_paths.append(str(png_path))
            plt.close(fig)

    pd.DataFrame(segment_rows).to_csv(
        fig_dir / "noun_role_detailed_timepoint_anova_segments.csv",
        index=False,
    )
    pd.DataFrame(timepoint_rows).to_csv(
        fig_dir / "noun_role_detailed_timepoint_anova_pvalues.csv",
        index=False,
    )
    return {"pdf": str(pdf_path), "png_dir": str(fig_dir), "first_png": png_paths[0] if png_paths else ""}


def write_summary(
    unit_set: str,
    run_meta: dict[str, dict[str, object]],
    results: pd.DataFrame,
    all_heatmaps: dict[str, str],
    explanatory_summary: dict[str, str],
    detailed_pages: dict[str, str],
) -> dict[str, object]:
    by_run_rows = []
    for (patient, run), group in results.groupby(["patient", "run"], sort=False):
        by_run_rows.append(
            {
                "patient": patient,
                "run": run,
                "n_units": int(len(group)),
                "n_uncorrected_p05": int(group["significant_uncorrected_p05"].sum()),
                "n_run_fdr05": int(group["significant_run_fdr05"].sum()),
                "best_p": float(group["p_value"].min()),
                "best_q_run": float(group["q_value_run_fdr"].min()),
            }
        )
    by_run = pd.DataFrame(by_run_rows)
    by_run.to_csv(OUT_DIR / "noun_role_structure_encoding_summary_by_run.csv", index=False)
    preference_rows = []
    for label, group in (
        ("all_units", results),
        ("uncorrected_p05", results[results["significant_uncorrected_p05"]]),
        ("run_fdr05", results[results["significant_run_fdr05"]]),
        ("global_fdr05", results[results["significant_global_fdr05"]]),
    ):
        counts = group["direction"].value_counts()
        preference_rows.append(
            {
                "set": label,
                "n_units": int(len(group)),
                "subject_pref": int(counts.get("subject_gt_predicate", 0)),
                "predicate_pref": int(counts.get("predicate_gt_subject", 0)),
                "no_difference": int(counts.get("no_difference", 0)),
            }
        )
    preference_counts = pd.DataFrame(preference_rows)
    preference_counts.to_csv(OUT_DIR / "noun_role_preference_count_summary.csv", index=False)

    summary = {
        "unit_set": unit_set,
        "method": {
            "roles": {
                ROLE_SUBJECT: "sentence-initial noun",
                ROLE_PREDICATE: "noun immediately after active/passive function word",
            },
            "nouns": NOUN_ORDER,
            "lock": "subject noun locked to speech start; predicate-position noun locked to post-function-word envelope rise",
            "baseline_sec": list(BASELINE_SEC),
            "response_sec": list(RESPONSE_SEC),
            "unit_test": "paired Wilcoxon signed-rank across 20 noun identities, subject vs predicate response",
            "multiple_comparison": "Benjamini-Hochberg FDR across units globally, per patient, and per run",
            "alpha": ALPHA,
        },
        "counts": {
            "n_runs": int(results[["patient", "run"]].drop_duplicates().shape[0]),
            "n_units_total": int(len(results)),
            "n_uncorrected_p05": int(results["significant_uncorrected_p05"].sum()),
            "n_run_fdr05": int(results["significant_run_fdr05"].sum()),
            "n_patient_fdr05": int(results["significant_patient_fdr05"].sum()),
            "n_global_fdr05": int(results["significant_global_fdr05"].sum()),
            "by_run": by_run.to_dict(orient="records"),
            "run_meta": run_meta,
        },
        "outputs": {
            "audio_onsets_csv": str(OUT_DIR / "noun_role_audio_onsets.csv"),
            "event_metadata_csv": str(OUT_DIR / "noun_role_event_metadata.csv"),
            "trial_response_metrics_csv": str(OUT_DIR / "noun_role_unit_trial_response_metrics.csv"),
            "unit_results_csv": str(OUT_DIR / "noun_role_structure_encoding_units.csv"),
            "candidate_units_csv": str(OUT_DIR / "noun_role_candidate_units_uncorrected_p05.csv"),
            "run_fdr_units_csv": str(OUT_DIR / "noun_role_significant_units_run_fdr05.csv"),
            "detailed_unit_selection_csv": str(OUT_DIR / "noun_role_detailed_unit_plot_selection.csv"),
            "preference_count_summary_csv": str(OUT_DIR / "noun_role_preference_count_summary.csv"),
            "detailed_timepoint_anova_segments_csv": str(
                OUT_DIR / "detailed_unit_pages" / "noun_role_detailed_timepoint_anova_segments.csv"
            ),
            "detailed_timepoint_anova_pvalues_csv": str(
                OUT_DIR / "detailed_unit_pages" / "noun_role_detailed_timepoint_anova_pvalues.csv"
            ),
            "all_unit_heatmaps_pdf": all_heatmaps["pdf"],
            "explanatory_summary_png": explanatory_summary["png"],
            "explanatory_summary_pdf": explanatory_summary["pdf"],
            "detailed_unit_pages_pdf": detailed_pages["pdf"],
            "audio_onset_qc_png": str(OUT_DIR / "fig_noun_role_audio_onset_qc.png"),
            "summary_by_run_csv": str(OUT_DIR / "noun_role_structure_encoding_summary_by_run.csv"),
            "summary_json": str(OUT_DIR / "noun_role_structure_encoding_summary.json"),
            "report_md": str(REPORT_PATH),
        },
    }
    with (OUT_DIR / "noun_role_structure_encoding_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_markdown_report(summary, by_run, preference_counts, results)
    return summary


def markdown_table(df: pd.DataFrame, columns: list[str], float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if col in float_cols and pd.notna(value):
                vals.append(f"{float(value):.3g}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_markdown_report(
    summary: dict[str, object],
    by_run: pd.DataFrame,
    preference_counts: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    top = results.sort_values(["p_value", "q_value_run_fdr"]).head(20).copy()
    top_cols = [
        "patient",
        "run",
        "unit_id",
        "ks_label",
        "n_noun_pairs",
        "subject_minus_predicate_bc_hz",
        "paired_cohens_dz",
        "p_value",
        "q_value_run_fdr",
        "direction",
    ]
    lines = [
        "# Noun subject/predicate-position structure encoding across 5 patients",
        "",
        "This analysis treats each noun identity as a paired sample: the same noun appears once as the sentence-initial subject noun and once as the noun after the active/passive function word.",
        "",
        f"Unit set: `{summary['unit_set']}`.",
        f"Response metric: mean firing rate {RESPONSE_SEC[0]}-{RESPONSE_SEC[1]} s after noun onset minus baseline {BASELINE_SEC[0]}-{BASELINE_SEC[1]} s.",
        "Statistics: paired Wilcoxon signed-rank across the 20 noun identities for each unit; FDR is reported globally, by patient, and by run.",
        "",
        "Note: the user-provided list included `因而`; the stimulus table contains `婴儿`, which is used here.",
        "",
        "## Preference summary",
        "",
        markdown_table(
            preference_counts,
            ["set", "n_units", "subject_pref", "predicate_pref", "no_difference"],
        ),
        "",
        "Time-resolved ANOVA was added to the upper-left trace in the detailed candidate-unit pages. Purple bars mark contiguous role-effect periods with per-time-bin repeated-measures role ANOVA p < 0.05, after filling gaps <=20 ms and keeping segments >=50 ms.",
        "",
        "## Counts by run",
        "",
        markdown_table(
            by_run,
            ["patient", "run", "n_units", "n_uncorrected_p05", "n_run_fdr05", "best_p", "best_q_run"],
            {"best_p", "best_q_run"},
        ),
        "",
        "## Top units by paired noun-role p value",
        "",
        markdown_table(
            top[top_cols],
            top_cols,
            {"subject_minus_predicate_bc_hz", "paired_cohens_dz", "p_value", "q_value_run_fdr"},
        ),
        "",
        "## Outputs",
        "",
    ]
    outputs = summary["outputs"]  # type: ignore[index]
    for key, value in outputs.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- A positive effect means stronger baseline-corrected response when the noun is in subject position.",
            "- A negative effect means stronger response when the same noun is in the post-function-word predicate-position noun slot.",
            "- The detailed candidate-unit pages show all 20 nouns as subject, predicate-position noun, and subject-minus-predicate heatmaps.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"starting analysis: unit_set={args.unit_set}, top_units_per_run={args.top_units_per_run}")
    font_prop = setup_matplotlib()
    log("estimating noun audio onsets")
    onsets = build_noun_audio_onsets(font_prop)
    log("building noun-role event table")
    events = build_event_rows(onsets)
    log("collecting unit response metrics")
    metrics, run_meta = collect_response_metrics(events, args.unit_set)
    log("running paired noun-role unit tests")
    results = analyze_units(metrics)
    log("plotting all-unit response heatmaps")
    all_heatmaps = plot_all_unit_heatmaps(metrics, results, font_prop)
    log("plotting explanatory summary figure")
    explanatory_summary = plot_explanatory_summary(results, font_prop, events, args.unit_set)
    log("selecting detailed candidate units")
    selected = selected_detailed_units(results, args.top_units_per_run)
    log(f"plotting detailed pages for {len(selected)} units")
    detailed_pages = plot_detailed_units(selected, events, args.unit_set, font_prop)
    log("writing summary and report")
    summary = write_summary(args.unit_set, run_meta, results, all_heatmaps, explanatory_summary, detailed_pages)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
