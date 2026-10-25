#!/usr/bin/env python3
"""Semantic and grammatical feature models for active/passive responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025/data"
OUT_DIR = DATA_ROOT / "combined_active_passive_semantic_grammatical_feature_model"
REPORT_PATH = ROOT / "reports/active_passive_semantic_grammatical_feature_model.md"
ACOUSTIC_FEATURE_PATH = (
    DATA_ROOT
    / "combined_active_passive_variance_partitioning"
    / "active_passive_acoustic_features.csv"
)
ARTIFACT_FEATURE_PATH = (
    DATA_ROOT
    / "combined_active_passive_ied_artifact_control"
    / "active_passive_trial_artifact_spikefield_features.csv"
)
METRIC = "response_baseline_corrected_hz"


@dataclass(frozen=True)
class Proposition:
    pair_id: int
    agent_noun: str
    patient_noun: str
    verb: str
    result_complement: str
    active_function_word: str
    passive_function_word: str
    active_sentence: str
    passive_sentence: str


PROPOSITIONS: tuple[Proposition, ...] = (
    Proposition(1, "虫子", "树叶", "啃", "光", "把", "被", "虫子把树叶啃光了", "树叶被虫子啃光了"),
    Proposition(2, "交警", "汽车", "拦", "住", "给", "让", "交警给汽车拦住了", "汽车让交警拦住了"),
    Proposition(3, "小狗", "骨头", "叼", "走", "将", "叫", "小狗将骨头叼走了", "骨头叫小狗叼走了"),
    Proposition(4, "妈妈", "衣服", "晾", "好", "把", "被", "妈妈把衣服晾好了", "衣服被妈妈晾好了"),
    Proposition(5, "球员", "足球", "踢", "飞", "给", "让", "球员给足球踢飞了", "足球让球员踢飞了"),
    Proposition(6, "兔子", "菜叶", "吃", "光", "把", "被", "兔子把菜叶吃光了", "菜叶被兔子吃光了"),
    Proposition(7, "爸爸", "爆竹", "点", "着", "将", "叫", "爸爸将爆竹点着了", "爆竹叫爸爸点着了"),
    Proposition(8, "婴儿", "牛奶", "喝", "完", "把", "被", "婴儿把牛奶喝完了", "牛奶被婴儿喝完了"),
    Proposition(9, "奶奶", "西瓜", "切", "开", "将", "叫", "奶奶将西瓜切开了", "西瓜叫奶奶切开了"),
    Proposition(10, "警察", "道路", "封锁", "none", "给", "让", "警察给道路封锁了", "道路让警察封锁了"),
)


def make_stimulus_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prop in PROPOSITIONS:
        proposition_label = f"{prop.agent_noun}->{prop.patient_noun}:{prop.verb}:{prop.result_complement}"
        rows.extend(
            [
                {
                    "sound_file": f"stim_actpas/act{prop.pair_id:03d}.mp3",
                    "pair_id": prop.pair_id,
                    "condition_simple": "active",
                    "sentence": prop.active_sentence,
                    "target_word": prop.active_function_word,
                    "agent_noun": prop.agent_noun,
                    "patient_noun": prop.patient_noun,
                    "verb": prop.verb,
                    "result_complement": prop.result_complement,
                    "proposition_label": proposition_label,
                    "surface_subject_noun": prop.agent_noun,
                    "post_marker_noun": prop.patient_noun,
                    "surface_subject_role": "agent",
                    "post_marker_role": "patient",
                },
                {
                    "sound_file": f"stim_actpas/pass{prop.pair_id:03d}.mp3",
                    "pair_id": prop.pair_id,
                    "condition_simple": "passive",
                    "sentence": prop.passive_sentence,
                    "target_word": prop.passive_function_word,
                    "agent_noun": prop.agent_noun,
                    "patient_noun": prop.patient_noun,
                    "verb": prop.verb,
                    "result_complement": prop.result_complement,
                    "proposition_label": proposition_label,
                    "surface_subject_noun": prop.patient_noun,
                    "post_marker_noun": prop.agent_noun,
                    "surface_subject_role": "patient",
                    "post_marker_role": "agent",
                },
            ]
        )
    out = pd.DataFrame(rows).sort_values(["pair_id", "condition_simple"]).reset_index(drop=True)
    out = out.set_index("sound_file", drop=False)
    return out


STIMULUS_FEATURES = make_stimulus_features()


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False) % 100_000


def simple_condition(value: object) -> str | None:
    text = str(value)
    if text.startswith("active"):
        return "active"
    if text.startswith("passive"):
        return "passive"
    return None


def pair_id_from_sound_file(sound_file: str) -> int:
    match = re.search(r"(?:act|pass)(\d+)\.mp3$", str(sound_file))
    if not match:
        raise ValueError(f"Cannot infer pair id from {sound_file}")
    return int(match.group(1))


def subject_from_er_dir(er_dir: Path) -> str:
    return er_dir.relative_to(DATA_ROOT).parts[0]


def find_event_dirs() -> list[Path]:
    return sorted(DATA_ROOT.glob("gz*/processed/ActPasSynSentence/event_related_active_passive"))


def one_hot(values: pd.Series | np.ndarray, prefix: str) -> np.ndarray:
    series = pd.Series(values).astype(str)
    return pd.get_dummies(series, prefix=prefix, dtype=float).to_numpy(dtype=float)


def numeric_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(df), 0), dtype=float)
    x = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if x.size == 0:
        return np.empty((len(df), 0), dtype=float)
    col_median = np.nanmedian(x, axis=0)
    col_median = np.where(np.isfinite(col_median), col_median, 0.0)
    inds = np.where(~np.isfinite(x))
    if inds[0].size:
        x[inds] = np.take(col_median, inds[1])
    return x


def build_design_groups(trials: pd.DataFrame, acoustic_columns: list[str]) -> dict[str, np.ndarray]:
    if "surface_subject_role" in trials:
        surface_subject_role = trials["surface_subject_role"]
    else:
        surface_subject_role = np.where(trials["condition_simple"].astype(str) == "active", "agent", "patient")
    if "post_marker_role" in trials:
        post_marker_role = trials["post_marker_role"]
    else:
        post_marker_role = np.where(trials["condition_simple"].astype(str) == "active", "patient", "agent")
    return {
        "acoustic": numeric_matrix(trials, acoustic_columns),
        "semantic_pair": one_hot(trials["pair_id"], "pair"),
        "semantic_roles": np.hstack(
            [
                one_hot(trials["agent_noun"], "agent"),
                one_hot(trials["patient_noun"], "patient"),
                one_hot(trials["verb"], "verb"),
                one_hot(trials["result_complement"], "result"),
            ]
        ),
        "grammar_voice": one_hot(trials["condition_simple"], "voice"),
        "marker_identity": one_hot(trials["target_word"], "marker"),
        "surface_order": np.hstack(
            [
                one_hot(trials["surface_subject_noun"], "surface_subject"),
                one_hot(trials["post_marker_noun"], "post_marker"),
                one_hot(surface_subject_role, "surface_subject_role"),
                one_hot(post_marker_role, "post_marker_role"),
            ]
        ),
    }


def hstack_groups(groups: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    if not names:
        n_rows = next(iter(groups.values())).shape[0]
        return np.empty((n_rows, 0), dtype=float)
    return np.hstack([groups[name] for name in names])


def make_leave_pair_out_splits(meta: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    pair_values = pd.unique(meta["pair_id"])
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    indices = np.arange(len(meta))
    for pair in pair_values:
        test = indices[meta["pair_id"].to_numpy() == pair]
        train = indices[meta["pair_id"].to_numpy() != pair]
        if len(train) == 0 or len(test) == 0:
            continue
        splits.append((train.astype(int), test.astype(int)))
    return splits


def make_cross_voice_splits(
    meta: pd.DataFrame,
    n_repeats: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    meta = meta.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    pair_values = list(pd.unique(meta["pair_id"]))
    pair_values = [pair for pair in pair_values if meta.loc[meta["pair_id"] == pair, "condition_simple"].nunique() == 2]
    if len(pair_values) < 2:
        return []
    offsets = rng.integers(0, 2, size=max(n_repeats, 1))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(max(n_repeats, 1)):
        train_idx: list[int] = []
        test_idx: list[int] = []
        for pair_pos, pair in enumerate(pair_values):
            sub = np.flatnonzero(meta["pair_id"].to_numpy() == pair)
            active = sub[meta.loc[sub, "condition_simple"].to_numpy() == "active"]
            passive = sub[meta.loc[sub, "condition_simple"].to_numpy() == "passive"]
            if len(active) != 1 or len(passive) != 1:
                continue
            train_active = (pair_pos + repeat + int(offsets[repeat])) % 2 == 0
            train_idx.append(int(active[0] if train_active else passive[0]))
            test_idx.append(int(passive[0] if train_active else active[0]))
        if len(train_idx) >= 2 and len(test_idx) >= 2:
            splits.append((np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)))
    return splits


def cv_predictions(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    pred_sum = np.zeros(len(y), dtype=float)
    null_sum = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in splits:
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        train_y = y[train_idx]
        null = float(np.mean(train_y))
        null_sum[test_idx] += null
        if x.shape[1] == 0 or np.nanstd(train_y) <= 1e-12:
            pred = np.full(len(test_idx), null, dtype=float)
        else:
            variable = np.nanstd(x[train_idx], axis=0) > 1e-12
            if not np.any(variable):
                pred = np.full(len(test_idx), null, dtype=float)
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(x[train_idx][:, variable], train_y)
                pred = model.predict(x[test_idx][:, variable])
        pred_sum[test_idx] += pred
        counts[test_idx] += 1
    return pred_sum, null_sum, counts


def cv_r2(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    alpha: float = 1.0,
) -> float:
    if not splits:
        return np.nan
    pred_sum, null_sum, counts = cv_predictions(x, y, splits, alpha)
    valid = counts > 0
    if valid.sum() < 3:
        return np.nan
    y_valid = np.asarray(y, dtype=float)[valid]
    yhat = pred_sum[valid] / counts[valid]
    yhat_null = null_sum[valid] / counts[valid]
    sse = float(np.sum((y_valid - yhat) ** 2))
    sse_null = float(np.sum((y_valid - yhat_null) ** 2))
    return float(1.0 - sse / sse_null) if sse_null > 1e-12 else np.nan


MODEL_GROUPS: dict[str, list[str]] = {
    "acoustic": ["acoustic"],
    "semantic_pair": ["acoustic", "semantic_pair"],
    "semantic_roles": ["acoustic", "semantic_roles"],
    "grammar_voice": ["acoustic", "grammar_voice"],
    "grammar_marker": ["acoustic", "grammar_voice", "marker_identity"],
    "semantic_plus_grammar": ["acoustic", "semantic_pair", "grammar_voice"],
    "full": ["acoustic", "semantic_pair", "grammar_voice", "marker_identity", "surface_order"],
}


def score_feature_models(
    groups: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
    prefix: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for model_name, group_names in MODEL_GROUPS.items():
        scores[f"{prefix}_{model_name}_r2"] = cv_r2(hstack_groups(groups, group_names), y, splits, alpha=alpha)

    scores[f"{prefix}_semantic_pair_delta"] = scores[f"{prefix}_semantic_pair_r2"] - scores[f"{prefix}_acoustic_r2"]
    scores[f"{prefix}_semantic_roles_delta"] = scores[f"{prefix}_semantic_roles_r2"] - scores[f"{prefix}_acoustic_r2"]
    scores[f"{prefix}_grammar_voice_delta"] = scores[f"{prefix}_grammar_voice_r2"] - scores[f"{prefix}_acoustic_r2"]
    scores[f"{prefix}_marker_after_voice_delta"] = scores[f"{prefix}_grammar_marker_r2"] - scores[f"{prefix}_grammar_voice_r2"]
    scores[f"{prefix}_grammar_after_semantic_delta"] = (
        scores[f"{prefix}_semantic_plus_grammar_r2"] - scores[f"{prefix}_semantic_pair_r2"]
    )
    scores[f"{prefix}_full_after_semantic_grammar_delta"] = (
        scores[f"{prefix}_full_r2"] - scores[f"{prefix}_semantic_plus_grammar_r2"]
    )
    return scores


def signed_rank(values: pd.Series | np.ndarray, alternative: str = "greater") -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    out = {
        "n": int(len(arr)),
        "median": float(np.median(arr)) if len(arr) else np.nan,
        "mean": float(np.mean(arr)) if len(arr) else np.nan,
        "wilcoxon_p": np.nan,
    }
    if len(arr) >= 3 and np.any(np.abs(arr) > 1e-12):
        out["wilcoxon_p"] = float(stats.wilcoxon(arr, alternative=alternative).pvalue)
    return out


def load_metrics(er_dir: Path) -> pd.DataFrame:
    subject = subject_from_er_dir(er_dir)
    df = pd.read_csv(er_dir / "active_passive_unit_trial_response_metrics.csv")
    df["subject"] = subject
    df["condition_simple"] = df["condition"].map(simple_condition)
    df = df[df["condition_simple"].isin(["active", "passive"])].copy()
    df["pair_id"] = df["sound_file"].map(pair_id_from_sound_file)
    return df


def load_acoustic_features() -> tuple[pd.DataFrame, list[str]]:
    if not ACOUSTIC_FEATURE_PATH.exists():
        return pd.DataFrame({"sound_file": STIMULUS_FEATURES["sound_file"].to_numpy()}), []
    acoustic = pd.read_csv(ACOUSTIC_FEATURE_PATH)
    acoustic_columns = [
        col
        for col in acoustic.columns
        if col
        not in {
            "sound_file",
            "condition_simple",
            "sentence",
            "target_word",
            "pair_id",
            "target_onset_sec_in_audio",
        }
        and pd.api.types.is_numeric_dtype(acoustic[col])
    ]
    return acoustic[["sound_file", *acoustic_columns]].copy(), acoustic_columns


def load_artifact_features() -> pd.DataFrame | None:
    if not ARTIFACT_FEATURE_PATH.exists():
        return None
    artifact = pd.read_csv(ARTIFACT_FEATURE_PATH)
    cols = ["subject", "run", "trial_index", "artifact_fraction", "max_abs_count_max", "p999_abs_count_max"]
    return artifact[cols].drop_duplicates(["subject", "run", "trial_index"])


def augment_metrics(metrics: pd.DataFrame, acoustic: pd.DataFrame, artifact: pd.DataFrame | None) -> pd.DataFrame:
    stim = STIMULUS_FEATURES.reset_index(drop=True)
    out = metrics.merge(
        stim.drop(columns=["condition_simple"], errors="ignore"),
        on="sound_file",
        how="left",
        suffixes=("", "_stim"),
    )
    out = out.merge(acoustic, on="sound_file", how="left")
    if artifact is not None:
        out = out.merge(artifact, on=["subject", "run", "trial_index"], how="left")
    else:
        out["artifact_fraction"] = np.nan
        out["max_abs_count_max"] = np.nan
        out["p999_abs_count_max"] = np.nan
    return out


def apply_filters(
    metrics: pd.DataFrame,
    unit_set: str,
    artifact_threshold: float | None,
) -> pd.DataFrame:
    out = metrics.copy()
    if unit_set == "good_only":
        out = out[out["ks_label"].astype(str) == "good"].copy()
    elif unit_set != "all_clusters":
        raise ValueError(f"Unsupported unit set: {unit_set}")

    if artifact_threshold is not None:
        keep = out["artifact_fraction"].notna() & (out["artifact_fraction"] <= artifact_threshold)
        out = out.loc[keep].copy()
    return out


def make_run_trials(run_df: pd.DataFrame) -> pd.DataFrame:
    trial_cols = [
        "subject",
        "run",
        "trial_index",
        "sound_file",
        "condition_simple",
        "sentence",
        "target_word",
        "pair_id",
        "agent_noun",
        "patient_noun",
        "verb",
        "result_complement",
        "proposition_label",
        "surface_subject_noun",
        "post_marker_noun",
        "surface_subject_role",
        "post_marker_role",
        "artifact_fraction",
    ]
    acoustic_cols = [col for col in run_df.columns if col.startswith(("audio_", "target_onset_", "full_", "pre_", "fw_", "post_"))]
    trial_cols.extend([col for col in acoustic_cols if col not in trial_cols])
    trials = run_df[trial_cols].drop_duplicates(["trial_index"]).sort_values("trial_index").reset_index(drop=True)
    return trials


def analyze_run(
    subject: str,
    run: str,
    run_df: pd.DataFrame,
    acoustic_columns: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object] | None, dict[str, object] | None]:
    trials = make_run_trials(run_df)
    n_active = int((trials["condition_simple"] == "active").sum())
    n_passive = int((trials["condition_simple"] == "passive").sum())
    n_pairs = int(trials["pair_id"].nunique())
    common = {
        "subject": subject,
        "run": run,
        "n_trials": int(len(trials)),
        "n_active": n_active,
        "n_passive": n_passive,
        "n_pairs": n_pairs,
    }
    if len(trials) < args.min_trials_total or min(n_active, n_passive) < args.min_trials_per_class or n_pairs < args.min_pairs:
        return pd.DataFrame(), None, {**common, "reason": "insufficient clean active/passive trials or semantic pairs"}

    trial_index_to_pos = {int(row.trial_index): i for i, row in trials.iterrows()}
    groups = build_design_groups(trials, acoustic_columns)
    leave_pair = make_leave_pair_out_splits(trials)
    cross_voice = make_cross_voice_splits(
        trials,
        n_repeats=args.cross_voice_repeats,
        random_state=stable_seed(subject, run, "cross_voice"),
    )
    if len(leave_pair) < args.min_pairs or len(cross_voice) < 2:
        return pd.DataFrame(), None, {**common, "reason": "not enough cross-validation splits"}

    rows: list[dict[str, object]] = []
    for unit_id, sub in run_df.groupby("unit_id"):
        sub = sub.drop_duplicates(["trial_index"]).copy()
        if sub["trial_index"].nunique() != len(trials):
            continue
        ordered = sub.assign(_pos=sub["trial_index"].map(trial_index_to_pos)).sort_values("_pos")
        y = ordered[METRIC].to_numpy(dtype=float)
        if len(y) != len(trials) or not np.isfinite(y).all() or np.nanstd(y) <= 1e-12:
            continue
        cross_scores = score_feature_models(groups, y, cross_voice, args.ridge_alpha, "cross_voice")
        leave_scores = score_feature_models(groups, y, leave_pair, args.ridge_alpha, "leave_pair")
        rows.append(
            {
                **common,
                "unit_id": int(unit_id),
                "ks_label": str(ordered["ks_label"].mode().iloc[0]),
                "response_sd_hz": float(np.std(y, ddof=1)),
                **cross_scores,
                **leave_scores,
            }
        )

    unit_df = pd.DataFrame(rows)
    if unit_df.empty:
        return unit_df, None, {**common, "reason": "no units with complete finite trial responses"}
    run_summary = summarize_run(unit_df, common)
    return unit_df, run_summary, None


def summarize_run(unit_df: pd.DataFrame, common: dict[str, object]) -> dict[str, object]:
    metrics = [
        "cross_voice_semantic_pair_delta",
        "cross_voice_grammar_voice_delta",
        "cross_voice_marker_after_voice_delta",
        "cross_voice_full_after_semantic_grammar_delta",
        "leave_pair_grammar_voice_delta",
        "leave_pair_marker_after_voice_delta",
        "leave_pair_semantic_pair_delta",
        "leave_pair_full_r2",
    ]
    out: dict[str, object] = {**common, "n_units": int(unit_df.shape[0])}
    for metric in metrics:
        values = unit_df[metric].to_numpy(dtype=float)
        out[f"median_{metric}"] = float(np.nanmedian(values)) if len(values) else np.nan
        out[f"fraction_positive_{metric}"] = float(np.nanmean(values > 0)) if len(values) else np.nan
    out["median_cross_voice_semantic_pair_r2"] = float(np.nanmedian(unit_df["cross_voice_semantic_pair_r2"]))
    out["median_cross_voice_grammar_voice_r2"] = float(np.nanmedian(unit_df["cross_voice_grammar_voice_r2"]))
    out["median_leave_pair_grammar_voice_r2"] = float(np.nanmedian(unit_df["leave_pair_grammar_voice_r2"]))
    out["median_leave_pair_grammar_marker_r2"] = float(np.nanmedian(unit_df["leave_pair_grammar_marker_r2"]))
    return out


def aggregate_run_medians(run_summary: pd.DataFrame) -> dict[str, object]:
    metrics = [
        "median_cross_voice_semantic_pair_delta",
        "median_cross_voice_grammar_voice_delta",
        "median_leave_pair_grammar_voice_delta",
        "median_leave_pair_marker_after_voice_delta",
        "median_leave_pair_semantic_pair_delta",
    ]
    aggregate: dict[str, object] = {}
    for metric in metrics:
        aggregate[metric] = signed_rank(run_summary[metric])
    return aggregate


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


def make_run_median_plot(run_summary: pd.DataFrame, out_dir: Path) -> str | None:
    if run_summary.empty:
        return None
    components = [
        "median_cross_voice_semantic_pair_delta",
        "median_cross_voice_grammar_voice_delta",
        "median_leave_pair_grammar_voice_delta",
        "median_leave_pair_marker_after_voice_delta",
        "median_leave_pair_semantic_pair_delta",
    ]
    labels = [
        "Semantic pair\ncross-voice",
        "Voice\ncross-voice",
        "Voice\nleave-pair-out",
        "Marker after voice\nleave-pair-out",
        "Semantic pair\nleave-pair-out",
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for i, component in enumerate(components):
        vals = run_summary[component].to_numpy(dtype=float)
        jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=54, color="#2f6fbb", alpha=0.82)
        if np.isfinite(vals).any():
            ax.plot([i - 0.27, i + 0.27], [np.nanmedian(vals)] * 2, color="black", lw=2.2)
    ax.axhline(0, color="0.45", lw=1.0, ls="--")
    ax.set_xticks(np.arange(len(components)), labels)
    ax.set_ylabel("Run median delta CV-R2 across units")
    ax.set_title("Semantic and grammatical feature-model contributions")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_semantic_grammar_feature_run_medians.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_unit_scatter_plot(unit_df: pd.DataFrame, out_dir: Path) -> str | None:
    if unit_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.7, 5.5))
    for run, sub in unit_df.groupby("run"):
        ax.scatter(
            sub["cross_voice_semantic_pair_delta"],
            sub["leave_pair_grammar_voice_delta"],
            s=12,
            alpha=0.45,
            label=str(run).replace("_actpas_", "\n"),
        )
    ax.axhline(0, color="0.55", lw=1, ls="--")
    ax.axvline(0, color="0.55", lw=1, ls="--")
    ax.set_xlabel("Semantic pair delta CV-R2\n(cross-voice split)")
    ax.set_ylabel("Voice delta CV-R2\n(leave-pair-out split)")
    ax.set_title("Unit-wise semantic vs abstract-grammar evidence")
    ax.legend(frameon=False, fontsize=6, ncol=1, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_semantic_vs_grammar_unit_scatter.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_model_r2_plot(unit_df: pd.DataFrame, out_dir: Path) -> str | None:
    if unit_df.empty:
        return None
    rows = []
    for run, sub in unit_df.groupby("run"):
        rows.extend(
            [
                {"run": run, "model": "cross_voice semantic", "median_r2": np.nanmedian(sub["cross_voice_semantic_pair_r2"])},
                {"run": run, "model": "cross_voice voice", "median_r2": np.nanmedian(sub["cross_voice_grammar_voice_r2"])},
                {"run": run, "model": "leave_pair voice", "median_r2": np.nanmedian(sub["leave_pair_grammar_voice_r2"])},
                {"run": run, "model": "leave_pair marker", "median_r2": np.nanmedian(sub["leave_pair_grammar_marker_r2"])},
                {"run": run, "model": "full", "median_r2": np.nanmedian(sub["leave_pair_full_r2"])},
            ]
        )
    plot_df = pd.DataFrame(rows)
    model_order = ["cross_voice semantic", "cross_voice voice", "leave_pair voice", "leave_pair marker", "full"]
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    for run, sub in plot_df.groupby("run"):
        y = [sub.loc[sub["model"] == model, "median_r2"].iloc[0] for model in model_order]
        ax.plot(range(len(model_order)), y, marker="o", lw=1.1, alpha=0.62)
    med = [np.nanmedian(plot_df.loc[plot_df["model"] == model, "median_r2"]) for model in model_order]
    ax.plot(range(len(model_order)), med, marker="s", lw=3.0, color="#c43c39", label="median across runs")
    ax.axhline(0, color="0.5", lw=1, ls="--")
    ax.set_xticks(range(len(model_order)), model_order, rotation=20, ha="right")
    ax.set_ylabel("Run median CV-R2 across units")
    ax.set_title("Feature-model predictive performance")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_semantic_grammar_model_r2.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_figures(unit_df: pd.DataFrame, run_summary: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key, path in [
        ("run_medians_png", make_run_median_plot(run_summary, out_dir)),
        ("unit_scatter_png", make_unit_scatter_plot(unit_df, out_dir)),
        ("model_r2_png", make_model_r2_plot(unit_df, out_dir)),
    ]:
        if path:
            paths[key] = path
    return paths


def write_report(
    run_summary: pd.DataFrame,
    unit_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    aggregate: dict[str, object],
    figure_paths: dict[str, str],
    args: argparse.Namespace,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_label = "all trials" if args.artifact_threshold is None else f"artifact_fraction <= {args.artifact_threshold:g}"
    lines = [
        "# Active/passive semantic and grammatical feature model",
        "",
        "This analysis fits cross-validated ridge encoding models to unit-level baseline-corrected function-word responses.",
        f"Main trial filter: {clean_label}; unit set: {args.unit_set}.",
        "",
        "Two complementary split schemes are used:",
        "",
        "- `cross_voice`: one member of each active/passive semantic pair is in training and the opposite voice is held out. Semantic-pair features can succeed here only if responses carry grammar-invariant proposition identity.",
        "- `leave_pair`: both active/passive versions of a semantic pair are held out together. Voice/marker features can succeed here only if grammatical effects generalize to unseen propositions.",
        "",
        "Identifiability note: in this stimulus set, each proposition has unique nouns/verb, and marker identity is nested within voice class. Therefore semantic-pair and marker effects are interpretable as feature-model diagnostics, not as perfectly separable latent causes.",
        "Because each run has only 20 sentence trials, absolute single-unit CV-R2 is often negative; the main readout is the within-split delta over acoustic baseline.",
        "",
    ]
    if not run_summary.empty:
        lines.extend(
            [
                "## Interpretation Notes",
                "",
                f"- Clean analyzed runs: {run_summary.shape[0]}; analyzed units: {unit_df[['subject', 'run', 'unit_id']].drop_duplicates().shape[0]}.",
                f"- Cross-voice semantic-pair delta median across runs: {run_summary['median_cross_voice_semantic_pair_delta'].median():.3g}.",
                f"- Leave-pair-out voice delta median across runs: {run_summary['median_leave_pair_grammar_voice_delta'].median():.3g}.",
                f"- Leave-pair-out marker-after-voice delta median across runs: {run_summary['median_leave_pair_marker_after_voice_delta'].median():.3g}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Run Summary",
            "",
            markdown_table(run_summary),
            "",
            "## Run-Median Signed-Rank Summary",
            "",
            "```json",
            json.dumps(aggregate, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Unit Results Preview",
            "",
            markdown_table(unit_df.head(60)),
            "",
            "## Skipped Runs",
            "",
            markdown_table(skipped_df),
            "",
            "## Figures",
            "",
        ]
    )
    for key, path in figure_paths.items():
        lines.append(f"- {key}: `{path}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    acoustic, acoustic_columns = load_acoustic_features()
    artifact = load_artifact_features()

    unit_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for er_dir in find_event_dirs():
        subject = subject_from_er_dir(er_dir)
        metrics = augment_metrics(load_metrics(er_dir), acoustic, artifact)
        metrics = apply_filters(metrics, args.unit_set, args.artifact_threshold)
        for run in sorted(metrics["run"].unique()):
            run_df = metrics[metrics["run"] == run].copy()
            unit_df, run_summary, skipped = analyze_run(subject, run, run_df, acoustic_columns, args)
            if skipped is not None:
                skipped_rows.append(skipped)
                print(f"skipped {subject} {run}: {skipped['reason']}", flush=True)
                continue
            if run_summary is not None:
                unit_frames.append(unit_df)
                run_rows.append(run_summary)
                print(
                    f"analyzed {subject} {run}: trials={run_summary['n_trials']} pairs={run_summary['n_pairs']} units={run_summary['n_units']}",
                    flush=True,
                )

    unit_df = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    run_summary = pd.DataFrame(run_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    if not unit_df.empty:
        unit_df = unit_df.sort_values(["subject", "run", "unit_id"]).reset_index(drop=True)
    if not run_summary.empty:
        run_summary = run_summary.sort_values(["subject", "run"]).reset_index(drop=True)
    aggregate = {
        "unit_set": args.unit_set,
        "metric": METRIC,
        "artifact_threshold": args.artifact_threshold,
        "ridge_alpha": args.ridge_alpha,
        "cross_voice_repeats": args.cross_voice_repeats,
        "n_runs_analyzed": int(run_summary.shape[0]),
        "n_units_analyzed": int(unit_df.shape[0]),
        "n_runs_skipped": int(skipped_df.shape[0]),
        "acoustic_columns": acoustic_columns,
        "run_median_signed_rank": aggregate_run_medians(run_summary) if not run_summary.empty else {},
    }

    unit_path = OUT_DIR / "active_passive_semantic_grammatical_unit_results.csv"
    run_path = OUT_DIR / "active_passive_semantic_grammatical_run_summary.csv"
    skipped_path = OUT_DIR / "active_passive_semantic_grammatical_skipped_runs.csv"
    stim_path = OUT_DIR / "active_passive_semantic_grammatical_stimulus_features.csv"
    summary_path = OUT_DIR / "active_passive_semantic_grammatical_summary.json"
    unit_df.to_csv(unit_path, index=False)
    run_summary.to_csv(run_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    STIMULUS_FEATURES.to_csv(stim_path, index=False)

    figure_paths = make_figures(unit_df, run_summary, OUT_DIR)
    aggregate["outputs"] = {
        "unit_results_csv": str(unit_path),
        "run_summary_csv": str(run_path),
        "skipped_runs_csv": str(skipped_path),
        "stimulus_features_csv": str(stim_path),
        "summary_json": str(summary_path),
        "report_md": str(REPORT_PATH),
        **figure_paths,
    }
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(run_summary, unit_df, skipped_df, aggregate, figure_paths, args)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=["good_only", "all_clusters"], default="good_only")
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--cross-voice-repeats", type=int, default=40)
    parser.add_argument("--min-trials-total", type=int, default=12)
    parser.add_argument("--min-trials-per-class", type=int, default=4)
    parser.add_argument("--min-pairs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
