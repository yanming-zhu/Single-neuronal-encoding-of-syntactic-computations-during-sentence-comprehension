#!/usr/bin/env python3
"""Analyze unit responses to the 10 verbs in active and passive sentences."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
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
    active_passive_metadata,
    audio_envelope,
    audio_root,
    bh_fdr,
    decode_mp3,
    first_patient_offsets_path,
    gaussian_smooth_ms,
    onset_from_peak,
    run_npz_path,
    shortened_run_name,
)


OUT_DIR = DATA_ROOT / "combined_verb_active_passive_unit_activity"
REPORT_PATH = ROOT / "reports/verb_active_passive_unit_activity_5patients.md"

PRE_MS = 500
POST_MS = 1200
BASELINE_SEC = (-0.5, 0.0)
RESPONSE_SEC = (0.0, 0.8)
SMOOTH_SIGMA_MS = 40
ALPHA = 0.05

ACTIVE = "#2F6FBB"
PASSIVE = "#C44E35"


@dataclass(frozen=True)
class VerbStimulus:
    sound_file: str
    condition: str
    sentence: str
    function_word: str
    verb_id: int
    verb: str


STIMULI: tuple[VerbStimulus, ...] = (
    VerbStimulus("stim_actpas/act001.mp3", "active", "虫子把树叶啃光了", "把", 1, "啃光"),
    VerbStimulus("stim_actpas/act002.mp3", "active", "交警给汽车拦住了", "给", 2, "拦住"),
    VerbStimulus("stim_actpas/act003.mp3", "active", "小狗将骨头叼走了", "将", 3, "叼走"),
    VerbStimulus("stim_actpas/act004.mp3", "active", "妈妈把衣服晾好了", "把", 4, "晾好"),
    VerbStimulus("stim_actpas/act005.mp3", "active", "球员给足球踢飞了", "给", 5, "踢飞"),
    VerbStimulus("stim_actpas/act006.mp3", "active", "兔子把菜叶吃光了", "把", 6, "吃光"),
    VerbStimulus("stim_actpas/act007.mp3", "active", "爸爸将爆竹点着了", "将", 7, "点着"),
    VerbStimulus("stim_actpas/act008.mp3", "active", "婴儿把牛奶喝完了", "把", 8, "喝完"),
    VerbStimulus("stim_actpas/act009.mp3", "active", "奶奶将西瓜切开了", "将", 9, "切开"),
    VerbStimulus("stim_actpas/act010.mp3", "active", "警察给道路封锁了", "给", 10, "封锁"),
    VerbStimulus("stim_actpas/pass001.mp3", "passive", "树叶被虫子啃光了", "被", 1, "啃光"),
    VerbStimulus("stim_actpas/pass002.mp3", "passive", "汽车让交警拦住了", "让", 2, "拦住"),
    VerbStimulus("stim_actpas/pass003.mp3", "passive", "骨头叫小狗叼走了", "叫", 3, "叼走"),
    VerbStimulus("stim_actpas/pass004.mp3", "passive", "衣服被妈妈晾好了", "被", 4, "晾好"),
    VerbStimulus("stim_actpas/pass005.mp3", "passive", "足球让球员踢飞了", "让", 5, "踢飞"),
    VerbStimulus("stim_actpas/pass006.mp3", "passive", "菜叶被兔子吃光了", "被", 6, "吃光"),
    VerbStimulus("stim_actpas/pass007.mp3", "passive", "爆竹叫爸爸点着了", "叫", 7, "点着"),
    VerbStimulus("stim_actpas/pass008.mp3", "passive", "牛奶被婴儿喝完了", "被", 8, "喝完"),
    VerbStimulus("stim_actpas/pass009.mp3", "passive", "西瓜叫奶奶切开了", "叫", 9, "切开"),
    VerbStimulus("stim_actpas/pass010.mp3", "passive", "道路让警察封锁了", "让", 10, "封锁"),
)

STIM_BY_SOUND = {stim.sound_file: stim for stim in STIMULI}
VERB_ORDER = [stim.verb for stim in STIMULI[:10]]
VERB_LABELS = [f"{i:02d}\n{verb}" for i, verb in enumerate(VERB_ORDER, start=1)]

# A few envelope traces contain a small transition peak between the second noun
# and the verb, or miss one weak noun syllable. These peak targets were selected
# from the verb-onset QC figure and keep the analysis locked to the first verb
# syllable rather than the neighboring noun/transition peak.
MANUAL_VERB_PEAK_AUDIO_SEC = {
    "stim_actpas/act002.mp3": 1.355,
    "stim_actpas/pass006.mp3": 1.575,
    "stim_actpas/pass008.mp3": 1.355,
}


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
    print(f"[verb-active-passive] {message}", flush=True)


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


def local_peak_times(times: np.ndarray, env: np.ndarray) -> list[float]:
    peaks, _ = find_peaks(
        env,
        distance=max(1, int(0.08 / 0.005)),
        prominence=max(float(env.max()) * 0.06, 0.004),
    )
    return [float(times[i]) for i in peaks if times[i] < 2.3]


def choose_function_peak(row: pd.Series, peaks: list[float]) -> float:
    onset = float(row["target_onset_sec_in_audio"])
    source = str(row["onset_source"])
    if source.startswith("manual"):
        candidates = [p for p in peaks if onset - 0.03 <= p <= onset + 0.35]
        if candidates:
            return min(candidates, key=lambda p: abs(p - onset))
    return float(row["target_peak_sec_in_audio"])


def estimate_verb_onset(
    times: np.ndarray,
    env: np.ndarray,
    row: pd.Series,
) -> dict[str, object]:
    peaks = local_peak_times(times, env)
    function_peak = choose_function_peak(row, peaks)
    sound_file = str(row["sound_file"])
    if sound_file in MANUAL_VERB_PEAK_AUDIO_SEC:
        verb_peak = MANUAL_VERB_PEAK_AUDIO_SEC[sound_file]
        prior_peaks = [p for p in peaks if function_peak + 0.04 < p < verb_peak - 0.04]
        prev_peak = prior_peaks[-1] if prior_peaks else function_peak
        source = "manual_verb_peak_from_qc"
    else:
        after_function = [p for p in peaks if p > function_peak + 0.04]
        if len(after_function) >= 3:
            verb_peak = after_function[2]
            prev_peak = after_function[1]
            source = "third_envelope_peak_after_function"
        elif len(after_function) >= 2:
            verb_peak = after_function[1]
            prev_peak = after_function[0]
            source = "second_envelope_peak_after_function_fallback"
        elif len(after_function) == 1:
            verb_peak = after_function[0]
            prev_peak = function_peak
            source = "first_envelope_peak_after_function_fallback"
        else:
            verb_peak = function_peak + 0.65
            prev_peak = function_peak
            source = "function_peak_plus_650ms_fallback"

    left_limit = min(verb_peak - 0.02, max(float(row["target_onset_sec_in_audio"]) + 0.08, prev_peak + 0.02))
    verb_onset = onset_from_peak(times, env, verb_peak, left_limit_sec=left_limit)
    return {
        "verb_onset_sec_in_audio": float(verb_onset),
        "verb_peak_sec_in_audio": float(verb_peak),
        "function_peak_sec_in_audio": float(function_peak),
        "detected_peak_times_sec_in_audio": json.dumps(peaks, ensure_ascii=False),
        "onset_source": source,
    }


def build_verb_audio_onsets(font_prop: FontProperties) -> pd.DataFrame:
    offsets = pd.read_csv(first_patient_offsets_path())
    offsets = offsets[offsets["sound_file"].isin(STIM_BY_SOUND)].copy()
    base_audio_dir = audio_root()
    rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(10, 2, figsize=(13, 23), sharex=False)
    for ax, row in zip(axes.flat, offsets.itertuples(index=False)):
        row_s = pd.Series(row._asdict())
        stim = STIM_BY_SOUND[str(row.sound_file)]
        audio = decode_mp3(base_audio_dir / str(row.sound_file))
        times, env = audio_envelope(audio)
        est = estimate_verb_onset(times, env, row_s)
        speech_start = float(row.speech_start_sec_in_audio)
        verb_onset_audio = float(est["verb_onset_sec_in_audio"])
        verb_peak_audio = float(est["verb_peak_sec_in_audio"])
        verb_onset_sentence = max(0.0, verb_onset_audio - speech_start)

        rows.append(
            {
                "sound_file": stim.sound_file,
                "condition": stim.condition,
                "sentence": stim.sentence,
                "function_word": stim.function_word,
                "verb_id": stim.verb_id,
                "verb": stim.verb,
                "speech_start_sec_in_audio": speech_start,
                "function_onset_sec_in_audio": float(row.target_onset_sec_in_audio),
                "function_peak_sec_in_audio": float(est["function_peak_sec_in_audio"]),
                "verb_onset_sec_in_audio": verb_onset_audio,
                "verb_peak_sec_in_audio": verb_peak_audio,
                "verb_onset_sec_from_sentence_onset": verb_onset_sentence,
                "onset_source": str(est["onset_source"]),
                "detected_peak_times_sec_in_audio": str(est["detected_peak_times_sec_in_audio"]),
            }
        )

        ax.plot(times, env, color="#333333", linewidth=1.0)
        ax.axvline(speech_start, color="#777777", linewidth=1.0, label="speech start")
        ax.axvline(float(row.target_onset_sec_in_audio), color="#777777", linewidth=1.0, linestyle="--", label="function word")
        ax.axvline(verb_onset_audio, color="#7B3294", linewidth=1.4, label="verb onset")
        ax.axvline(verb_peak_audio, color="#7B3294", linewidth=0.9, linestyle="--", label="verb peak")
        ax.set_xlim(0, min(float(times[-1]), 2.25))
        ax.set_title(
            f"{Path(stim.sound_file).stem} {stim.verb}: {stim.sentence}",
            fontproperties=font_prop,
            fontsize=9,
        )
        ax.set_ylabel("RMS")
    for ax in axes[-1]:
        ax.set_xlabel("Audio time (s)")
    axes.flat[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Estimated verb onsets in active/passive sentence audio", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    qc_path = OUT_DIR / "fig_verb_audio_onset_qc.png"
    fig.savefig(qc_path, dpi=170)
    plt.close(fig)

    onsets = pd.DataFrame(rows).sort_values(["verb_id", "condition"])
    onsets.to_csv(OUT_DIR / "verb_audio_onsets.csv", index=False)
    return onsets


def build_event_rows(onsets: pd.DataFrame) -> pd.DataFrame:
    onset_lookup = {row.sound_file: row for row in onsets.itertuples(index=False)}
    event_rows: list[dict[str, object]] = []
    for patient in PATIENTS:
        meta = active_passive_metadata(patient)
        for trial in meta.itertuples(index=False):
            if str(trial.sound_file) not in onset_lookup:
                continue
            onset = onset_lookup[str(trial.sound_file)]
            event_bin = int(trial.marker_bin) + int(round(float(onset.verb_onset_sec_from_sentence_onset) * 1000))
            event_rows.append(
                {
                    "patient": patient,
                    "run": str(trial.run),
                    "trial_index": int(trial.trial_index),
                    "sound_file": str(trial.sound_file),
                    "sentence_condition": str(onset.condition),
                    "sentence": str(onset.sentence),
                    "function_word": str(onset.function_word),
                    "verb_id": int(onset.verb_id),
                    "verb": str(onset.verb),
                    "verb_onset_sec_from_sentence_onset": float(onset.verb_onset_sec_from_sentence_onset),
                    "event_bin_1000Hz": event_bin,
                    "event_time_sec": event_bin / 1000,
                    "onset_source": str(onset.onset_source),
                }
            )
    events = pd.DataFrame(event_rows)
    events.to_csv(OUT_DIR / "verb_event_metadata.csv", index=False)
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
                        "sentence": str(event.sentence),
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
    metrics.to_csv(OUT_DIR / "verb_unit_trial_response_metrics.csv", index=False)
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
            index="verb_id",
            columns="sentence_condition",
            values="response_baseline_corrected_hz",
            aggfunc="mean",
        ).reindex(range(1, 11))
        if "active" not in pivot or "passive" not in pivot:
            continue
        valid = pivot[["active", "passive"]].dropna()
        diffs = valid["active"].to_numpy(float) - valid["passive"].to_numpy(float)
        stat, p_value = safe_wilcoxon(diffs)
        active_mean = float(valid["active"].mean()) if len(valid) else np.nan
        passive_mean = float(valid["passive"].mean()) if len(valid) else np.nan
        mean_diff = float(np.mean(diffs)) if len(diffs) else np.nan
        sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else np.nan
        dz = float(mean_diff / sd_diff) if sd_diff and np.isfinite(sd_diff) and sd_diff > 0 else np.nan
        rows.append(
            {
                "patient": patient,
                "run": run,
                "unit_id": int(unit_id),
                "ks_label": str(group["ks_label"].iloc[0]),
                "n_verb_pairs": int(len(valid)),
                "active_mean_bc_hz": active_mean,
                "passive_mean_bc_hz": passive_mean,
                "active_minus_passive_bc_hz": mean_diff,
                "paired_cohens_dz": dz,
                "wilcoxon_statistic": stat,
                "p_value": p_value,
                "direction": "active_gt_passive"
                if mean_diff > 0
                else "passive_gt_active"
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
    results.to_csv(OUT_DIR / "verb_active_passive_unit_results.csv", index=False)
    results.loc[results["significant_uncorrected_p05"]].to_csv(
        OUT_DIR / "verb_active_passive_candidate_units_uncorrected_p05.csv",
        index=False,
    )
    results.loc[results["significant_run_fdr05"]].to_csv(
        OUT_DIR / "verb_active_passive_significant_units_run_fdr05.csv",
        index=False,
    )
    return results


def matrix_for_condition(run_metrics: pd.DataFrame, condition: str, units: list[int]) -> pd.DataFrame:
    return (
        run_metrics[run_metrics["sentence_condition"] == condition]
        .pivot_table(index="unit_id", columns="verb_id", values="response_baseline_corrected_hz", aggfunc="mean")
        .reindex(index=units, columns=range(1, 11))
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
    codes[is_candidate & (directions == "active_gt_passive")] = 1
    codes[is_candidate & (directions == "passive_gt_active")] = 2
    return codes[:, None]


def plot_all_unit_heatmaps(metrics: pd.DataFrame, results: pd.DataFrame, font_prop: FontProperties) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    pdf_path = fig_dir / "verb_active_passive_all_units_response_heatmaps_by_run.pdf"
    png_paths: list[str] = []
    run_order = list(metrics[["patient", "run"]].drop_duplicates().itertuples(index=False, name=None))

    with PdfPages(pdf_path) as pdf:
        for patient, run in run_order:
            run_metrics = metrics[(metrics["patient"] == patient) & (metrics["run"] == run)]
            run_results = results[(results["patient"] == patient) & (results["run"] == run)].copy()
            units = (
                run_results.sort_values("active_minus_passive_bc_hz", ascending=False)["unit_id"]
                .astype(int)
                .tolist()
            )
            active_mat = matrix_for_condition(run_metrics, "active", units)
            passive_mat = matrix_for_condition(run_metrics, "passive", units)
            diff_mat = active_mat - passive_mat
            vals = np.concatenate(
                [
                    active_mat.to_numpy(float).ravel(),
                    passive_mat.to_numpy(float).ravel(),
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
            candidate_cmap = mpl.colors.ListedColormap(["#F2F2F2", ACTIVE, PASSIVE])
            ax_sig.imshow(candidate_codes, aspect="auto", cmap=candidate_cmap, vmin=0, vmax=2)
            ax_sig.set_title("p<0.05\ncandidate", fontsize=8)
            ax_sig.set_xticks([])
            ax_sig.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            for spine in ax_sig.spines.values():
                spine.set_visible(False)

            for ax, mat, title in zip(
                axes,
                [active_mat, passive_mat, diff_mat],
                ["active sentence", "passive sentence", "active - passive"],
            ):
                im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap="RdBu_r", norm=norm)
                ax.set_title(title)
                ax.set_xticks(np.arange(10))
                ax.set_xticklabels(VERB_LABELS, rotation=0, fontproperties=font_prop)
                ax.set_xlabel("verb")
                sparse_unit_ticks(ax, units)
            axes[0].set_ylabel("")
            axes[1].tick_params(labelleft=False)
            axes[2].tick_params(labelleft=False)
            fig.colorbar(im, cax=cax, label="baseline-corrected Hz")
            fig.text(0.018, 0.53, "unit id, sorted by mean active-passive effect", rotation=90, va="center", fontsize=9)

            unc = run_results[run_results["significant_uncorrected_p05"]]
            n_unc_active = int((unc["direction"] == "active_gt_passive").sum())
            n_unc_passive = int((unc["direction"] == "passive_gt_active").sum())
            n_unc = int(len(unc))
            n_fdr = int(run_results["significant_run_fdr05"].sum())
            fig.legend(
                handles=[
                    mpl.patches.Patch(color=ACTIVE, label="uncorrected p<0.05 active-pref"),
                    mpl.patches.Patch(color=PASSIVE, label="uncorrected p<0.05 passive-pref"),
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=2,
                frameon=False,
                fontsize=8,
            )
            fig.suptitle(
                f"{patient} {run}: unit x verb response, "
                f"uncorrected p<0.05 candidates={n_unc} "
                f"(active={n_unc_active}, passive={n_unc_passive}); run-FDR<0.05={n_fdr}",
                y=0.995,
            )
            fig.subplots_adjust(left=0.055, right=0.965, bottom=0.18, top=0.94)
            pdf.savefig(fig)
            png_path = fig_dir / f"verb_active_passive_all_units_heatmap_{patient}_{run}.png"
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
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    time_sec = np.arange(-PRE_MS, POST_MS + 1) / 1000
    z = np.load(run_npz_path(patient, run, unit_set), allow_pickle=True)
    firing_rate = z["firing_rate_hz"]
    unit_ids = z["unit_ids"].astype(int)
    matches = np.flatnonzero(unit_ids == unit_id)
    if matches.size == 0:
        raise ValueError(f"unit {unit_id} not found in {patient} {run}")
    unit_i = int(matches[0])
    traces: dict[str, list[np.ndarray]] = {"active": [], "passive": []}
    verbs: list[str] = []
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
        if str(event.sentence_condition) == "active":
            verbs.append(str(event.verb))
    out = {
        condition: np.stack(items, axis=0) if items else np.empty((0, PRE_MS + POST_MS + 1))
        for condition, items in traces.items()
    }
    return time_sec, out, verbs


def plot_summary_figure(
    results: pd.DataFrame,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    unit_set: str,
) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_verb_active_passive_summary.png"
    pdf_path = fig_dir / "fig_verb_active_passive_summary.pdf"

    run_order = list(results[["patient", "run"]].drop_duplicates().itertuples(index=False, name=None))
    run_labels = [f"{patient}\n{shortened_run_name(run)}" for patient, run in run_order]

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.1], hspace=0.35, wspace=0.32)
    ax_schematic = fig.add_subplot(gs[0, 0])
    ax_counts = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[0, 2])
    ax_volcano = fig.add_subplot(gs[1, 0])
    ax_example_active = fig.add_subplot(gs[1, 1])
    ax_example_passive = fig.add_subplot(gs[1, 2])

    ax_schematic.axis("off")
    ax_schematic.set_title("Same verb, active vs passive context", loc="left", fontsize=12, weight="bold")
    ax_schematic.text(0.02, 0.82, "10 paired verbs", fontsize=9, color="0.35")
    examples = [
        ("虫子把树叶", "啃光", "了", ACTIVE),
        ("树叶被虫子", "啃光", "了", PASSIVE),
    ]
    for y, (left, verb, right, color) in zip([0.62, 0.42], examples):
        ax_schematic.text(0.03, y, left, fontsize=15, color="0.25", va="center")
        ax_schematic.text(0.45, y, verb, fontsize=20, color=color, weight="bold", va="center")
        ax_schematic.text(0.64, y, right, fontsize=15, color="0.25", va="center")
    ax_schematic.text(
        0.02,
        0.17,
        "Per unit: compare response to the same verb\n"
        "in active and passive sentences, paired across 10 verbs.",
        fontsize=9.2,
        color="0.25",
        va="top",
    )
    ax_schematic.text(
        0.02,
        -0.03,
        f"Units analyzed: {len(results)}; run-FDR<0.05: {int(results['significant_run_fdr05'].sum())}; "
        f"uncorrected p<0.05: {int(results['significant_uncorrected_p05'].sum())}",
        fontsize=8.8,
        color="0.20",
        va="top",
    )

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
                "active": int(counts.get("active_gt_passive", 0)),
                "passive": int(counts.get("passive_gt_active", 0)),
            }
        )
    y = np.arange(len(run_labels))
    active_counts = np.array([row["active"] for row in sig_counts])
    passive_counts = np.array([row["passive"] for row in sig_counts])
    ax_counts.barh(y, active_counts, color=ACTIVE, alpha=0.88, label="active-pref")
    ax_counts.barh(y, -passive_counts, color=PASSIVE, alpha=0.88, label="passive-pref")
    ax_counts.axvline(0, color="0.35", lw=0.8)
    ax_counts.set_yticks(y)
    ax_counts.set_yticklabels(run_labels)
    ax_counts.tick_params(axis="y", labelsize=8.5)
    ax_counts.set_xlabel("uncorrected p<0.05 candidate units")
    ax_counts.set_title("Candidate direction by run", loc="left", fontsize=12, weight="bold")
    ax_counts.legend(frameon=False, loc="lower right")
    max_count = max(1, int(np.max(np.abs(np.r_[active_counts, passive_counts]))))
    ax_counts.set_xlim(-max_count - 2, max_count + 2)
    ax_counts.grid(axis="x", color="0.9", lw=0.7)

    distributions = [
        results[(results["patient"] == patient) & (results["run"] == run)]["active_minus_passive_bc_hz"].to_numpy()
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
            colors = np.where(sig["direction"].to_numpy() == "active_gt_passive", ACTIVE, PASSIVE)
            ax_dist.scatter(
                np.full(len(sig), i),
                sig["active_minus_passive_bc_hz"],
                s=26,
                color=colors,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
    ax_dist.axhline(0, color="0.25", lw=0.9)
    ax_dist.set_xticks(np.arange(1, len(run_labels) + 1))
    ax_dist.set_xticklabels(run_labels, rotation=45, ha="right")
    ax_dist.set_ylabel("active - passive response (Hz)")
    ax_dist.set_title("Effect-size distribution", loc="left", fontsize=12, weight="bold")
    ax_dist.grid(axis="y", color="0.9", lw=0.7)

    x = results["active_minus_passive_bc_hz"].to_numpy(float)
    yv = -np.log10(np.clip(results["p_value"].to_numpy(float), 1e-12, 1.0))
    colors = np.full(len(results), "#BDBDBD", dtype=object)
    colors[(results["direction"] == "active_gt_passive").to_numpy() & results["significant_uncorrected_p05"].to_numpy()] = ACTIVE
    colors[(results["direction"] == "passive_gt_active").to_numpy() & results["significant_uncorrected_p05"].to_numpy()] = PASSIVE
    sizes = np.where(results["significant_uncorrected_p05"].to_numpy(), 34, 12)
    ax_volcano.scatter(x, yv, c=colors, s=sizes, alpha=0.82, edgecolors="none")
    ax_volcano.axvline(0, color="0.3", lw=0.8)
    ax_volcano.axhline(-np.log10(0.05), color="0.55", lw=0.8, linestyle="--")
    ax_volcano.set_xlabel("active - passive response (Hz)")
    ax_volcano.set_ylabel("-log10 paired Wilcoxon p")
    ax_volcano.set_title("Unit-level paired active/passive test", loc="left", fontsize=12, weight="bold")
    ax_volcano.grid(color="0.92", lw=0.7)

    def plot_example_trace(ax: plt.Axes, row: pd.Series | None, expected_direction: str) -> None:
        if row is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "Example trace unavailable", ha="center", va="center", color="0.4")
            return
        time_sec, traces, _ = extract_unit_condition_traces(
            str(row["patient"]),
            str(row["run"]),
            int(row["unit_id"]),
            events,
            unit_set,
        )
        for condition, color in (("active", ACTIVE), ("passive", PASSIVE)):
            arr = gaussian_smooth_ms(traces[condition], SMOOTH_SIGMA_MS)
            if arr.size == 0:
                continue
            mean = np.nanmean(arr, axis=0)
            n = np.sum(np.isfinite(arr), axis=0)
            se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
            ax.fill_between(time_sec, mean - se, mean + se, color=color, alpha=0.16, lw=0)
            ax.plot(time_sec, mean, color=color, lw=1.8, label=condition)
            window = (time_sec >= RESPONSE_SEC[0]) & (time_sec <= RESPONSE_SEC[1])
            verb_means = np.nanmean(arr[:, window], axis=1)
            ax.scatter(
                np.full_like(verb_means, 1.04 if condition == "active" else 1.10, dtype=float),
                verb_means,
                color=color,
                s=13,
                alpha=0.58,
                edgecolors="none",
                clip_on=False,
            )
        ax.axvline(0, color="0.25", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.set_xlim(-0.25, 1.16)
        ax.set_xlabel("time from verb onset (s)")
        ax.set_ylabel("baseline-corrected Hz")
        direction_label = "active-pref" if expected_direction == "active_gt_passive" else "passive-pref"
        ax.set_title(
            f"Example {direction_label}: {row['patient']} u{int(row['unit_id'])}",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        ax.text(
            0.02,
            0.96,
            f"effect={float(row['active_minus_passive_bc_hz']):.1f} Hz\n"
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
    active_examples = significant[significant["direction"] == "active_gt_passive"].sort_values("p_value")
    passive_examples = significant[significant["direction"] == "passive_gt_active"].sort_values("p_value")
    if active_examples.empty:
        active_examples = results[results["direction"] == "active_gt_passive"].sort_values("p_value")
    if passive_examples.empty:
        passive_examples = results[results["direction"] == "passive_gt_active"].sort_values("p_value")
    plot_example_trace(
        ax_example_active,
        active_examples.iloc[0] if not active_examples.empty else None,
        "active_gt_passive",
    )
    plot_example_trace(
        ax_example_passive,
        passive_examples.iloc[0] if not passive_examples.empty else None,
        "passive_gt_active",
    )

    fig.suptitle("Verb responses in active and passive sentences", y=0.995, fontsize=15, weight="bold")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def summarize_by_run(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (patient, run), group in results.groupby(["patient", "run"], sort=False):
        unc = group[group["significant_uncorrected_p05"]]
        rows.append(
            {
                "patient": patient,
                "run": run,
                "n_units": int(len(group)),
                "n_uncorrected_p05": int(len(unc)),
                "n_run_fdr05": int(group["significant_run_fdr05"].sum()),
                "active_pref_uncorrected": int((unc["direction"] == "active_gt_passive").sum()),
                "passive_pref_uncorrected": int((unc["direction"] == "passive_gt_active").sum()),
                "best_p": float(group["p_value"].min()) if len(group) else np.nan,
                "best_q_run": float(group["q_value_run_fdr"].min()) if len(group) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "verb_active_passive_summary_by_run.csv", index=False)
    return summary


def write_summary(
    unit_set: str,
    run_meta: dict[str, dict[str, object]],
    results: pd.DataFrame,
    run_summary: pd.DataFrame,
    heatmaps: dict[str, str],
    summary_fig: dict[str, str],
) -> dict[str, object]:
    direction_counts = {str(k): int(v) for k, v in results["direction"].value_counts().items()}
    top_units = results.sort_values("p_value").head(20)
    summary = {
        "unit_set": unit_set,
        "n_runs": int(results[["patient", "run"]].drop_duplicates().shape[0]),
        "n_units": int(len(results)),
        "direction_counts": direction_counts,
        "n_uncorrected_p05": int(results["significant_uncorrected_p05"].sum()),
        "n_run_fdr05": int(results["significant_run_fdr05"].sum()),
        "n_patient_fdr05": int(results["significant_patient_fdr05"].sum()),
        "n_global_fdr05": int(results["significant_global_fdr05"].sum()),
        "run_meta": run_meta,
        "outputs": {
            "verb_audio_onsets_csv": str(OUT_DIR / "verb_audio_onsets.csv"),
            "verb_event_metadata_csv": str(OUT_DIR / "verb_event_metadata.csv"),
            "trial_response_metrics_csv": str(OUT_DIR / "verb_unit_trial_response_metrics.csv"),
            "unit_results_csv": str(OUT_DIR / "verb_active_passive_unit_results.csv"),
            "candidate_units_csv": str(OUT_DIR / "verb_active_passive_candidate_units_uncorrected_p05.csv"),
            "run_fdr_units_csv": str(OUT_DIR / "verb_active_passive_significant_units_run_fdr05.csv"),
            "run_summary_csv": str(OUT_DIR / "verb_active_passive_summary_by_run.csv"),
            "audio_qc_png": str(OUT_DIR / "fig_verb_audio_onset_qc.png"),
            "heatmap_pdf": heatmaps["pdf"],
            "summary_png": summary_fig["png"],
            "summary_pdf": summary_fig["pdf"],
        },
        "top_units_by_p": top_units[
            [
                "patient",
                "run",
                "unit_id",
                "ks_label",
                "n_verb_pairs",
                "active_minus_passive_bc_hz",
                "paired_cohens_dz",
                "p_value",
                "q_value_run_fdr",
                "direction",
            ]
        ].to_dict(orient="records"),
    }
    (OUT_DIR / "verb_active_passive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = [
        "# Verb active/passive unit activity across 5 patients",
        "",
        "This analysis treats each verb identity as a paired sample: the same verb appears once in an active sentence and once in a passive sentence.",
        "",
        f"Unit set: `{unit_set}`. Response window: {RESPONSE_SEC[0]:.1f}-{RESPONSE_SEC[1]:.1f} s after estimated verb onset; baseline: {BASELINE_SEC[0]:.1f}-{BASELINE_SEC[1]:.1f} s.",
        "",
        "## Summary",
        "",
        f"- Units analyzed: {summary['n_units']}",
        f"- Uncorrected p<0.05 candidate units: {summary['n_uncorrected_p05']}",
        f"- Run-wise FDR<0.05 units: {summary['n_run_fdr05']}",
        f"- Patient-wise FDR<0.05 units: {summary['n_patient_fdr05']}",
        f"- Global FDR<0.05 units: {summary['n_global_fdr05']}",
        "",
        "## By Run",
        "",
        "| patient | run | n_units | uncorrected_p05 | run_fdr05 | active_pref | passive_pref | best_p | best_q_run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_summary.itertuples(index=False):
        lines.append(
            f"| {row.patient} | {row.run} | {int(row.n_units)} | {int(row.n_uncorrected_p05)} | "
            f"{int(row.n_run_fdr05)} | {int(row.active_pref_uncorrected)} | "
            f"{int(row.passive_pref_uncorrected)} | {float(row.best_p):.3g} | {float(row.best_q_run):.3g} |"
        )
    lines.extend(
        [
            "",
            "## Top Units By Paired Verb Test",
            "",
            "| patient | run | unit_id | ks_label | n_pairs | active-passive Hz | dz | p | q_run | direction |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in top_units.itertuples(index=False):
        lines.append(
            f"| {row.patient} | {row.run} | {int(row.unit_id)} | {row.ks_label} | "
            f"{int(row.n_verb_pairs)} | {float(row.active_minus_passive_bc_hz):.3g} | "
            f"{float(row.paired_cohens_dz):.3g} | {float(row.p_value):.3g} | "
            f"{float(row.q_value_run_fdr):.3g} | {row.direction} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Audio onset QC: `{summary['outputs']['audio_qc_png']}`",
            f"- Summary figure: `{summary['outputs']['summary_png']}`",
            f"- All-unit heatmaps: `{summary['outputs']['heatmap_pdf']}`",
            f"- Per-unit statistics: `{summary['outputs']['unit_results_csv']}`",
            f"- Per-trial unit metrics: `{summary['outputs']['trial_response_metrics_csv']}`",
            "",
            "Interpretation note: verb onsets are audio-envelope estimates. The QC figure should be checked before treating millisecond-level latency as exact.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"starting analysis: unit_set={args.unit_set}")
    font_prop = setup_matplotlib()
    log("estimating verb audio onsets")
    onsets = build_verb_audio_onsets(font_prop)
    log("building verb event table")
    events = build_event_rows(onsets)
    log("collecting unit response metrics")
    metrics, run_meta = collect_response_metrics(events, args.unit_set)
    log("running paired active/passive tests across the 10 verbs")
    results = analyze_units(metrics)
    log("plotting all-unit heatmaps")
    heatmaps = plot_all_unit_heatmaps(metrics, results, font_prop)
    log("plotting summary figure")
    summary_fig = plot_summary_figure(results, metrics, events, args.unit_set)
    log("summarizing by run")
    run_summary = summarize_by_run(results)
    log("writing summary and report")
    summary = write_summary(args.unit_set, run_meta, results, run_summary, heatmaps, summary_fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
