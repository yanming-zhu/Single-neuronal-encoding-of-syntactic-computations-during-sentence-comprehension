#!/usr/bin/env python3
"""Variance partitioning for active/passive function-word responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025/data"
AUDIO_DIR = ROOT / "neuropixel2025/files/实验设计/程序设计/neuropixel_m255_202511/ActPasSynSentence/scripts"
OUT_DIR = DATA_ROOT / "combined_active_passive_variance_partitioning"
REPORT_PATH = ROOT / "reports/active_passive_variance_partitioning.md"
SR = 16_000
METRIC = "response_baseline_corrected_hz"


STIMULI = {
    "stim_actpas/act001.mp3": ("active", "虫子把树叶啃光了", "把"),
    "stim_actpas/act002.mp3": ("active", "交警给汽车拦住了", "给"),
    "stim_actpas/act003.mp3": ("active", "小狗将骨头叼走了", "将"),
    "stim_actpas/act004.mp3": ("active", "妈妈把衣服晾好了", "把"),
    "stim_actpas/act005.mp3": ("active", "球员给足球踢飞了", "给"),
    "stim_actpas/act006.mp3": ("active", "兔子把菜叶吃光了", "把"),
    "stim_actpas/act007.mp3": ("active", "爸爸将爆竹点着了", "将"),
    "stim_actpas/act008.mp3": ("active", "婴儿把牛奶喝完了", "把"),
    "stim_actpas/act009.mp3": ("active", "奶奶将西瓜切开了", "将"),
    "stim_actpas/act010.mp3": ("active", "警察给道路封锁了", "给"),
    "stim_actpas/pass001.mp3": ("passive", "树叶被虫子啃光了", "被"),
    "stim_actpas/pass002.mp3": ("passive", "汽车让交警拦住了", "让"),
    "stim_actpas/pass003.mp3": ("passive", "骨头叫小狗叼走了", "叫"),
    "stim_actpas/pass004.mp3": ("passive", "衣服被妈妈晾好了", "被"),
    "stim_actpas/pass005.mp3": ("passive", "足球让球员踢飞了", "让"),
    "stim_actpas/pass006.mp3": ("passive", "菜叶被兔子吃光了", "被"),
    "stim_actpas/pass007.mp3": ("passive", "爆竹叫爸爸点着了", "叫"),
    "stim_actpas/pass008.mp3": ("passive", "牛奶被婴儿喝完了", "被"),
    "stim_actpas/pass009.mp3": ("passive", "西瓜叫奶奶切开了", "叫"),
    "stim_actpas/pass010.mp3": ("passive", "道路让警察封锁了", "让"),
}

MANUAL_TARGET_ONSET_AUDIO_SEC = {
    "stim_actpas/act008.mp3": 0.620,
    "stim_actpas/pass004.mp3": 0.655,
}


@dataclass(frozen=True)
class ModelScoreSet:
    full_r2: float
    reduced_r2: dict[str, float]
    unique_r2: dict[str, float]


def simple_condition(value: object) -> str | None:
    text = str(value)
    if text.startswith("active"):
        return "active"
    if text.startswith("passive"):
        return "passive"
    return None


def subject_from_er_dir(er_dir: Path) -> str:
    return er_dir.relative_to(DATA_ROOT).parts[0]


def find_event_dirs() -> list[Path]:
    return sorted(DATA_ROOT.glob("gz*/processed/ActPasSynSentence/event_related_active_passive"))


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


def estimate_target_onset(audio: np.ndarray, sound_file: str) -> dict[str, object]:
    hop = int(0.005 * SR)
    frame = int(0.025 * SR)
    starts = np.arange(0, max(1, audio.size - frame), hop)
    rms = np.array([np.sqrt(np.mean(audio[s : s + frame] ** 2)) for s in starts])
    env = moving_average(rms, 9)
    times = starts / SR

    active_thr = max(0.01, float(np.percentile(env, 95) * 0.08))
    active = np.flatnonzero(env > active_thr)
    if active.size == 0:
        raise RuntimeError(f"No active audio detected in {sound_file}")
    speech_start = max(0, int(active[0]) - 2)
    speech_start_sec = float(times[speech_start])

    search_start = speech_start + int(0.08 / 0.005)
    min_peak_gap = int(0.11 / 0.005)
    peak_thr = max(float(np.percentile(env, 75)), float(env.max() * 0.22))
    candidates: list[int] = []
    last = -10_000
    for i in range(search_start + 1, len(env) - 1):
        if env[i] >= peak_thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            if i - last >= min_peak_gap:
                candidates.append(i)
                last = i
            elif candidates and env[i] > env[candidates[-1]]:
                candidates[-1] = i
                last = i

    if len(candidates) < 3:
        early_end = min(len(env), speech_start + int(1.4 / 0.005))
        order = np.argsort(env[search_start:early_end])[::-1] + search_start
        candidates = []
        for idx in order:
            if all(abs(int(idx) - j) >= min_peak_gap for j in candidates):
                candidates.append(int(idx))
            if len(candidates) >= 6:
                break
        candidates.sort()

    if len(candidates) < 3:
        raise RuntimeError(f"Could not identify three syllable peaks in {sound_file}")

    target_peak = int(candidates[2])
    left = target_peak
    valley_limit = speech_start + int(0.12 / 0.005)
    while left > valley_limit and env[left - 1] <= env[left]:
        left -= 1
    onset_thr = env[left] + 0.25 * (env[target_peak] - env[left])
    onset = left
    while onset < target_peak and env[onset] < onset_thr:
        onset += 1
    target_onset_sec = float(times[onset])
    if sound_file in MANUAL_TARGET_ONSET_AUDIO_SEC:
        target_onset_sec = MANUAL_TARGET_ONSET_AUDIO_SEC[sound_file]

    return {
        "speech_start_sec": speech_start_sec,
        "target_onset_sec": target_onset_sec,
        "target_peak_sec": float(times[target_peak]),
        "env_times_sec": times,
        "env": env,
    }


def pair_id_from_sound_file(sound_file: str) -> int:
    match = re.search(r"(?:act|pass)(\d+)\.mp3$", sound_file)
    if not match:
        raise ValueError(f"Cannot infer pair id from {sound_file}")
    return int(match.group(1))


def extract_audio_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sound_file, (condition, sentence, target_word) in STIMULI.items():
        audio = decode_mp3(AUDIO_DIR / sound_file)
        est = estimate_target_onset(audio, sound_file)
        duration_sec = audio.size / SR
        onset = int(round(float(est["target_onset_sec"]) * SR))
        short_stop = min(audio.size, onset + int(0.25 * SR))
        long_stop = min(audio.size, onset + int(0.80 * SR))
        pre_start = max(0, onset - int(0.25 * SR))
        short = audio[onset:short_stop]
        long = audio[onset:long_stop]
        pre = audio[pre_start:onset]

        env = est["env"]
        env_times = est["env_times_sec"]
        env_mask_short = (env_times >= float(est["target_onset_sec"])) & (
            env_times < float(est["target_onset_sec"]) + 0.25
        )
        env_mask_onset = (env_times >= float(est["target_onset_sec"])) & (
            env_times < float(est["target_onset_sec"]) + 0.12
        )
        env_short = env[env_mask_short]
        env_onset = env[env_mask_onset]
        env_slope = float(np.nanmax(np.diff(env_onset))) if len(env_onset) > 1 else 0.0

        rows.append(
            {
                "sound_file": sound_file,
                "condition_simple": condition,
                "sentence": sentence,
                "target_word": target_word,
                "pair_id": pair_id_from_sound_file(sound_file),
                "audio_duration_sec": duration_sec,
                "target_onset_sec_in_audio": float(est["target_onset_sec"]),
                "target_onset_sec_from_sentence_onset": float(est["target_onset_sec"]) - float(est["speech_start_sec"]),
                "full_rms": float(np.sqrt(np.mean(audio**2))),
                "pre_fw_rms_250ms": float(np.sqrt(np.mean(pre**2))) if pre.size else 0.0,
                "fw_rms_250ms": float(np.sqrt(np.mean(short**2))) if short.size else 0.0,
                "fw_peak_abs_250ms": float(np.max(np.abs(short))) if short.size else 0.0,
                "fw_log_energy_250ms": float(np.log1p(np.sum(short**2))) if short.size else 0.0,
                "post_fw_rms_800ms": float(np.sqrt(np.mean(long**2))) if long.size else 0.0,
                "fw_env_peak_250ms": float(np.max(env_short)) if len(env_short) else 0.0,
                "fw_env_slope_120ms": env_slope,
            }
        )
    return pd.DataFrame(rows)


def load_artifact_table(er_dir: Path) -> pd.DataFrame | None:
    path = er_dir / "trial_artifact_overlap_minus1_to_plus2s.csv"
    if not path.exists():
        return None
    artifact = pd.read_csv(path)
    artifact["condition_simple"] = artifact["condition"].map(simple_condition)
    artifact = artifact[artifact["condition_simple"].isin(["active", "passive"])].copy()
    if artifact.empty:
        return None
    cols = ["run", "trial_index", "artifact_fraction_minus1_to_plus2s"]
    return artifact[cols].drop_duplicates(["run", "trial_index"])


def load_metrics(er_dir: Path) -> pd.DataFrame:
    subject = subject_from_er_dir(er_dir)
    metrics = pd.read_csv(er_dir / "active_passive_unit_trial_response_metrics.csv")
    metrics["subject"] = subject
    metrics["condition_simple"] = metrics["condition"].map(simple_condition)
    metrics = metrics[metrics["condition_simple"].isin(["active", "passive"])].copy()
    metrics["pair_id"] = metrics["sound_file"].map(pair_id_from_sound_file)
    return metrics


def apply_filters(
    metrics: pd.DataFrame,
    artifact: pd.DataFrame | None,
    unit_set: str,
    artifact_threshold: float | None,
) -> pd.DataFrame:
    out = metrics.copy()
    if unit_set == "good_only":
        out = out[out["ks_label"].astype(str) == "good"].copy()
    elif unit_set != "all_clusters":
        raise ValueError(f"Unsupported unit_set: {unit_set}")
    if artifact_threshold is not None and artifact is not None:
        out = out.merge(artifact, on=["run", "trial_index"], how="left")
        keep = out["artifact_fraction_minus1_to_plus2s"].isna() | (
            out["artifact_fraction_minus1_to_plus2s"] <= artifact_threshold
        )
        out = out.loc[keep].copy()
    else:
        out["artifact_fraction_minus1_to_plus2s"] = np.nan
    return out


def one_hot(values: pd.Series, prefix: str) -> np.ndarray:
    return pd.get_dummies(values.astype(str), prefix=prefix, dtype=float).to_numpy(dtype=float)


def design_groups_for_trials(trials: pd.DataFrame, acoustic_columns: list[str]) -> dict[str, np.ndarray]:
    return {
        "acoustic": trials[acoustic_columns].to_numpy(dtype=float),
        "sentence_pair": one_hot(trials["pair_id"], "pair"),
        "word_identity": one_hot(trials["target_word"], "word"),
        "syntax": (trials["condition_simple"].to_numpy() == "active").astype(float)[:, None],
    }


def hstack_groups(groups: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    if not names:
        return np.empty((next(iter(groups.values())).shape[0], 0), dtype=float)
    return np.hstack([groups[name] for name in names])


def cv_predictions(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray | None,
    n_splits: int,
    n_repeats: int,
    alpha: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    yhat = np.zeros_like(y, dtype=float)
    yhat_null = np.zeros_like(y, dtype=float)
    counts = np.zeros_like(y, dtype=float)
    min_class = np.min(np.bincount(strata.astype(int))) if strata is not None else len(y)
    effective_splits = min(n_splits, int(min_class)) if strata is not None else min(n_splits, len(y))
    if effective_splits < 2:
        raise ValueError("Need at least two folds")
    if strata is None:
        splitter = RepeatedKFold(n_splits=effective_splits, n_repeats=n_repeats, random_state=random_state)
        split_iter = splitter.split(x)
    elif n_repeats > 1:
        splitter = RepeatedStratifiedKFold(
            n_splits=effective_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        split_iter = splitter.split(x, strata)
    else:
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(x, strata)

    for train_idx, test_idx in split_iter:
        yhat_null[test_idx] += float(np.mean(y[train_idx]))
        if x.shape[1] == 0:
            yhat[test_idx] += float(np.mean(y[train_idx]))
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[train_idx], y[train_idx])
            yhat[test_idx] += model.predict(x[test_idx])
        counts[test_idx] += 1
    return yhat / counts, yhat_null / counts


def cv_r2(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    alpha: float = 1.0,
    random_state: int = 0,
) -> float:
    y = np.asarray(y, dtype=float)
    yhat, yhat_null = cv_predictions(
        x,
        y,
        strata=strata,
        n_splits=n_splits,
        n_repeats=n_repeats,
        alpha=alpha,
        random_state=random_state,
    )
    sse = float(np.sum((y - yhat) ** 2))
    sse_null = float(np.sum((y - yhat_null) ** 2))
    return float(1.0 - sse / sse_null) if sse_null > 0 else np.nan


def cross_validated_group_r2(
    groups: dict[str, np.ndarray],
    y: np.ndarray,
    strata: np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    alpha: float = 1.0,
    random_state: int = 0,
) -> dict[str, object]:
    names = list(groups)
    full_x = hstack_groups(groups, names)
    full_r2 = cv_r2(
        full_x,
        y,
        strata=strata,
        n_splits=n_splits,
        n_repeats=n_repeats,
        alpha=alpha,
        random_state=random_state,
    )
    reduced: dict[str, float] = {}
    unique: dict[str, float] = {}
    for name in names:
        reduced_names = [other for other in names if other != name]
        reduced_r2 = cv_r2(
            hstack_groups(groups, reduced_names),
            y,
            strata=strata,
            n_splits=n_splits,
            n_repeats=n_repeats,
            alpha=alpha,
            random_state=random_state,
        )
        reduced[name] = reduced_r2
        unique[name] = full_r2 - reduced_r2
    return {"full_r2": full_r2, "reduced_r2": reduced, "unique_r2": unique}


def model_r2_for_names(
    groups: dict[str, np.ndarray],
    y: np.ndarray,
    names: list[str],
    strata: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> float:
    return cv_r2(
        hstack_groups(groups, names),
        y,
        strata=strata,
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        alpha=args.ridge_alpha,
        random_state=seed,
    )


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False) % 100_000


def analyze_unit(
    groups: dict[str, np.ndarray],
    y: np.ndarray,
    strata: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    full = model_r2_for_names(groups, y, ["acoustic", "sentence_pair", "word_identity", "syntax"], strata, args, seed)
    acoustic_pair = model_r2_for_names(groups, y, ["acoustic", "sentence_pair"], strata, args, seed)
    acoustic_pair_syntax = model_r2_for_names(groups, y, ["acoustic", "sentence_pair", "syntax"], strata, args, seed)
    acoustic_pair_word = model_r2_for_names(groups, y, ["acoustic", "sentence_pair", "word_identity"], strata, args, seed)
    acoustic_pair_word_syntax = full
    acoustic_pair_syntax_word = full

    reduced_without_acoustic = model_r2_for_names(groups, y, ["sentence_pair", "word_identity", "syntax"], strata, args, seed)
    reduced_without_pair = model_r2_for_names(groups, y, ["acoustic", "word_identity", "syntax"], strata, args, seed)
    reduced_without_word = model_r2_for_names(groups, y, ["acoustic", "sentence_pair", "syntax"], strata, args, seed)
    reduced_without_syntax = model_r2_for_names(groups, y, ["acoustic", "sentence_pair", "word_identity"], strata, args, seed)

    return {
        "full_r2": full,
        "acoustic_pair_r2": acoustic_pair,
        "acoustic_pair_syntax_r2": acoustic_pair_syntax,
        "acoustic_pair_word_r2": acoustic_pair_word,
        "unique_acoustic_all": full - reduced_without_acoustic,
        "unique_sentence_pair_all": full - reduced_without_pair,
        "unique_word_identity_all": full - reduced_without_word,
        "unique_syntax_all": full - reduced_without_syntax,
        "syntax_candidate_no_word": acoustic_pair_syntax - acoustic_pair,
        "word_within_syntax": acoustic_pair_syntax_word - acoustic_pair_syntax,
        "syntax_after_word_identity": acoustic_pair_word_syntax - acoustic_pair_word,
    }


def signed_rank(values: np.ndarray, reference: float = 0.0) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out = {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else np.nan,
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "wilcoxon_p_greater": np.nan,
    }
    if len(values) >= 3 and np.any(np.abs(values - reference) > 1e-12):
        out["wilcoxon_p_greater"] = float(stats.wilcoxon(values - reference, alternative="greater").pvalue)
    return out


def make_figures(unit_results: pd.DataFrame, run_summary: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    if unit_results.empty:
        return paths
    components = [
        "unique_acoustic_all",
        "unique_sentence_pair_all",
        "syntax_candidate_no_word",
        "word_within_syntax",
        "syntax_after_word_identity",
    ]
    labels = {
        "unique_acoustic_all": "Acoustic\nunique",
        "unique_sentence_pair_all": "Sentence pair\nunique",
        "syntax_candidate_no_word": "Syntax candidate\n(no word control)",
        "word_within_syntax": "Word identity\nwithin syntax",
        "syntax_after_word_identity": "Syntax after\nword identity",
    }

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    positions = np.arange(len(components))
    for i, component in enumerate(components):
        vals = [
            unit_results.loc[unit_results["run"] == run, component].median()
            for run in sorted(unit_results["run"].unique())
        ]
        jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0]
        ax.scatter(np.full(len(vals), positions[i]) + jitter, vals, color="#2f6fbb", alpha=0.8, s=36)
        ax.plot([positions[i] - 0.26, positions[i] + 0.26], [np.nanmedian(vals)] * 2, color="black", lw=2.2)
    ax.axhline(0, color="0.45", ls="--", lw=1)
    ax.set_xticks(positions, [labels[c] for c in components])
    ax.set_ylabel("Run median unique CV-R2 across units")
    ax.set_title("Variance partitioning of function-word responses")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_variance_partitioning_run_medians.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths["run_medians_png"] = str(path)

    fig, axes = plt.subplots(1, len(components), figsize=(15, 4.4), sharey=True)
    for ax, component in zip(axes, components):
        values = [
            unit_results.loc[unit_results["run"] == run, component].dropna().to_numpy(float)
            for run in sorted(unit_results["run"].unique())
        ]
        ax.boxplot(values, showfliers=False, widths=0.65)
        ax.axhline(0, color="0.45", ls="--", lw=1)
        ax.set_title(labels[component], fontsize=9)
        ax.set_xticks(np.arange(1, len(values) + 1), [str(i + 1) for i in range(len(values))], fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Unit-level unique CV-R2")
    fig.suptitle("Unit-level variance partitioning distributions by run", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig_variance_partitioning_unit_distributions.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths["unit_distributions_png"] = str(path)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(
        unit_results["syntax_candidate_no_word"],
        unit_results["word_within_syntax"],
        s=10,
        alpha=0.45,
        color="#444444",
    )
    ax.axhline(0, color="0.6", lw=1, ls="--")
    ax.axvline(0, color="0.6", lw=1, ls="--")
    ax.set_xlabel("Syntax candidate unique CV-R2\n(acoustic + sentence pair controlled)")
    ax.set_ylabel("Function-word identity unique CV-R2\n(acoustic + pair + syntax controlled)")
    ax.set_title("Condition-level vs marker-identity explanatory power")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_syntax_candidate_vs_word_identity.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths["syntax_vs_word_identity_png"] = str(path)
    return paths


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report(
    run_summary: pd.DataFrame,
    aggregate: dict[str, object],
    figure_paths: dict[str, str],
) -> None:
    lines = [
        "# Active/passive acoustic, word-identity and syntax variance partitioning",
        "",
        "This analysis uses cross-validated ridge-regression R2 on 0-0.8 s baseline-corrected function-word responses.",
        "The main predictors are acoustic features, matched sentence-pair identity, function-word identity, and active/passive condition.",
        "",
        "Important identifiability note: function-word identity fully determines active/passive condition in this stimulus set.",
        "Therefore `syntax_after_word_identity` is a diagnostic for non-identifiability, not a clean abstract-syntax effect.",
        "",
        "## Run Summary",
        "",
        markdown_table(run_summary),
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Figures",
        "",
    ]
    for key, path in figure_paths.items():
        lines.append(f"- {key}: `{path}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    acoustic = extract_audio_features()
    acoustic_columns = [
        "audio_duration_sec",
        "target_onset_sec_from_sentence_onset",
        "full_rms",
        "pre_fw_rms_250ms",
        "fw_rms_250ms",
        "fw_peak_abs_250ms",
        "fw_log_energy_250ms",
        "post_fw_rms_800ms",
        "fw_env_peak_250ms",
        "fw_env_slope_120ms",
    ]
    acoustic_path = OUT_DIR / "active_passive_acoustic_features.csv"
    acoustic.to_csv(acoustic_path, index=False)

    unit_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    analysis_mode = "all_trials" if args.artifact_threshold is None else (
        f"artifact_clean_le_{str(args.artifact_threshold).replace('.', 'p')}"
    )

    for er_dir in find_event_dirs():
        subject = subject_from_er_dir(er_dir)
        metrics = load_metrics(er_dir)
        artifact = load_artifact_table(er_dir)
        filtered = apply_filters(metrics, artifact, args.unit_set, args.artifact_threshold)
        filtered = filtered.merge(acoustic, on=["sound_file", "condition_simple", "target_word", "pair_id"], how="left")
        for run in sorted(metrics["run"].unique()):
            run_df = filtered[filtered["run"] == run].copy()
            n_trials = run_df["trial_index"].nunique()
            n_active = run_df.loc[run_df["condition_simple"] == "active", "trial_index"].nunique()
            n_passive = run_df.loc[run_df["condition_simple"] == "passive", "trial_index"].nunique()
            common = {
                "analysis_mode": analysis_mode,
                "unit_set": args.unit_set,
                "subject": subject,
                "run": run,
                "n_trials": int(n_trials),
                "n_active": int(n_active),
                "n_passive": int(n_passive),
            }
            if n_trials < args.min_trials_total or min(n_active, n_passive) < args.min_trials_per_class:
                skipped_rows.append({**common, "reason": "insufficient artifact-clean active/passive trials"})
                continue
            trial_table = (
                run_df.drop_duplicates("trial_index")
                .sort_values("trial_index")
                .reset_index(drop=True)
            )
            groups = design_groups_for_trials(trial_table, acoustic_columns)
            strata = (trial_table["condition_simple"].to_numpy() == "active").astype(int)

            for unit_id, unit_df in run_df.groupby("unit_id"):
                unit_df = unit_df.set_index("trial_index").loc[trial_table["trial_index"]].reset_index()
                y = unit_df[METRIC].to_numpy(dtype=float)
                if not np.isfinite(y).all() or np.nanstd(y) <= 0:
                    continue
                seed = args.random_state + stable_seed(subject, run, unit_id)
                scores = analyze_unit(groups, y, strata, args, seed)
                unit_rows.append(
                    {
                        **common,
                        "unit_id": int(unit_id),
                        "ks_label": str(unit_df["ks_label"].iloc[0]),
                        **scores,
                    }
                )
            print(f"analyzed {subject} {run}: trials={n_trials} units={run_df['unit_id'].nunique()}", flush=True)

    unit_results = pd.DataFrame(unit_rows)
    skipped = pd.DataFrame(skipped_rows)
    component_cols = [
        "full_r2",
        "unique_acoustic_all",
        "unique_sentence_pair_all",
        "unique_word_identity_all",
        "unique_syntax_all",
        "syntax_candidate_no_word",
        "word_within_syntax",
        "syntax_after_word_identity",
    ]
    run_summary = (
        unit_results.groupby(["analysis_mode", "unit_set", "subject", "run"], as_index=False)
        .agg(
            n_units=("unit_id", "nunique"),
            **{f"median_{col}": (col, "median") for col in component_cols},
            **{f"mean_{col}": (col, "mean") for col in component_cols},
        )
        if not unit_results.empty
        else pd.DataFrame()
    )

    aggregate: dict[str, object] = {
        "analysis_mode": analysis_mode,
        "unit_set": args.unit_set,
        "metric": METRIC,
        "cv_splits": args.cv_splits,
        "cv_repeats": args.cv_repeats,
        "ridge_alpha": args.ridge_alpha,
        "n_runs_analyzed": int(run_summary[["subject", "run"]].drop_duplicates().shape[0]) if not run_summary.empty else 0,
        "n_units_analyzed": int(unit_results.shape[0]),
        "n_runs_skipped": int(skipped.shape[0]),
        "acoustic_columns": acoustic_columns,
    }
    for col in [
        "median_unique_acoustic_all",
        "median_unique_sentence_pair_all",
        "median_syntax_candidate_no_word",
        "median_word_within_syntax",
        "median_syntax_after_word_identity",
    ]:
        if col in run_summary:
            aggregate[col] = signed_rank(run_summary[col].to_numpy(float), reference=0.0)

    unit_path = OUT_DIR / "active_passive_variance_partitioning_unit_results.csv"
    run_path = OUT_DIR / "active_passive_variance_partitioning_run_summary.csv"
    skipped_path = OUT_DIR / "active_passive_variance_partitioning_skipped_runs.csv"
    summary_path = OUT_DIR / "active_passive_variance_partitioning_summary.json"
    unit_results.to_csv(unit_path, index=False)
    run_summary.to_csv(run_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    figure_paths = make_figures(unit_results, run_summary, OUT_DIR)
    aggregate["outputs"] = {
        "unit_results_csv": str(unit_path),
        "run_summary_csv": str(run_path),
        "skipped_runs_csv": str(skipped_path),
        "acoustic_features_csv": str(acoustic_path),
        "summary_json": str(summary_path),
        "report_md": str(REPORT_PATH),
        **figure_paths,
    }
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(run_summary, aggregate, figure_paths)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=["good_only", "all_clusters"], default="good_only")
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-trials-total", type=int, default=10)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=23)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
