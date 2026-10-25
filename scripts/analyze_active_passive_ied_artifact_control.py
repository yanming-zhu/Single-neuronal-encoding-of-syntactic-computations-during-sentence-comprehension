#!/usr/bin/env python3
"""IED/artifact and spike-field-load controls for active/passive responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025/data"
OUT_DIR = DATA_ROOT / "combined_active_passive_ied_artifact_control"
REPORT_PATH = ROOT / "reports/active_passive_ied_artifact_control.md"
METRIC = "response_baseline_corrected_hz"
ARTIFACT_FEATURES = ["artifact_fraction", "max_abs_count_max", "p999_abs_count_max"]


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


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


def pair_id_from_sound_file(sound_file: str) -> int:
    match = re.search(r"(?:act|pass)(\d+)\.mp3$", str(sound_file))
    if not match:
        raise ValueError(f"Cannot infer pair id from {sound_file}")
    return int(match.group(1))


def load_metrics(er_dir: Path) -> pd.DataFrame:
    subject = subject_from_er_dir(er_dir)
    metrics = pd.read_csv(er_dir / "active_passive_unit_trial_response_metrics.csv")
    metrics["subject"] = subject
    metrics["condition_simple"] = metrics["condition"].map(simple_condition)
    metrics = metrics[metrics["condition_simple"].isin(["active", "passive"])].copy()
    metrics["pair_id"] = metrics["sound_file"].map(pair_id_from_sound_file)
    return metrics


def load_trial_metadata(er_dir: Path) -> pd.DataFrame:
    subject = subject_from_er_dir(er_dir)
    path = er_dir / "active_passive_event_related_trial_metadata.csv"
    trial_meta = pd.read_csv(path)
    trial_meta["subject"] = subject
    trial_meta["condition_simple"] = trial_meta["condition"].map(simple_condition)
    trial_meta = trial_meta[trial_meta["condition_simple"].isin(["active", "passive"])].copy()
    trial_meta["pair_id"] = trial_meta["sound_file"].map(pair_id_from_sound_file)
    return trial_meta


def run_dir_from_event_dir(er_dir: Path, run: str) -> Path:
    return er_dir.parent / run


def load_artifact_npz(er_dir: Path, run: str) -> dict[str, np.ndarray]:
    path = run_dir_from_event_dir(er_dir, run) / "artifact_rejection" / "ap_artifact_mask_1000Hz.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path)
    return {
        "artifact_mask": z["artifact_mask"].astype(bool),
        "raw_artifact_mask": z["raw_artifact_mask"].astype(bool),
        "max_abs_count": z["max_abs_count"].astype(float),
        "p999_abs_count": z["p999_abs_count"].astype(float),
    }


def compute_window_artifact_features(
    events: pd.DataFrame,
    artifact_mask: np.ndarray,
    max_abs_count: np.ndarray,
    p999_abs_count: np.ndarray,
    pre_bins: int = 1000,
    post_bins: int = 2001,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_bins = int(len(artifact_mask))
    for _, row in events.iterrows():
        marker = int(row["marker_bin"])
        start = max(0, marker - pre_bins)
        stop = min(n_bins, marker + post_bins)
        window_mask = artifact_mask[start:stop]
        window_max_abs = max_abs_count[start:stop]
        window_p999 = p999_abs_count[start:stop]
        artifact_bins = int(window_mask.sum())
        window_n = int(stop - start)
        rows.append(
            {
                "subject": row.get("subject", ""),
                "run": row["run"],
                "trial_index": int(row["trial_index"]),
                "sound_file": row["sound_file"],
                "condition": row["condition"],
                "condition_simple": simple_condition(row["condition"]),
                "pair_id": pair_id_from_sound_file(row["sound_file"]),
                "marker_bin": marker,
                "window_start_bin": start,
                "window_stop_bin_exclusive": stop,
                "window_n_bins": window_n,
                "artifact_bins": artifact_bins,
                "artifact_fraction": float(artifact_bins / window_n) if window_n else np.nan,
                "artifact_ms": float(artifact_bins),
                "artifact_free": bool(artifact_bins == 0),
                "max_abs_count_mean": float(np.mean(window_max_abs)) if window_n else np.nan,
                "max_abs_count_max": float(np.max(window_max_abs)) if window_n else np.nan,
                "p999_abs_count_mean": float(np.mean(window_p999)) if window_n else np.nan,
                "p999_abs_count_max": float(np.max(window_p999)) if window_n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_artifact_features_for_event_dir(er_dir: Path) -> pd.DataFrame:
    trial_meta = load_trial_metadata(er_dir)
    frames: list[pd.DataFrame] = []
    for run, run_meta in trial_meta.groupby("run"):
        artifact = load_artifact_npz(er_dir, run)
        frames.append(
            compute_window_artifact_features(
                run_meta,
                artifact["artifact_mask"],
                artifact["max_abs_count"],
                artifact["p999_abs_count"],
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_unit_filter(metrics: pd.DataFrame, unit_set: str) -> pd.DataFrame:
    if unit_set == "good_only":
        return metrics[metrics["ks_label"].astype(str) == "good"].copy()
    if unit_set == "all_clusters":
        return metrics.copy()
    raise ValueError(f"Unsupported unit_set: {unit_set}")


def threshold_label(threshold: float | None) -> str:
    if threshold is None:
        return "all_trials"
    if threshold == 0:
        return "artifact_free"
    return f"artifact_le_{str(threshold).replace('.', 'p')}"


def filter_trials_by_threshold(trials: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if threshold is None:
        return trials.copy()
    return trials[trials["artifact_fraction"] <= threshold].copy()


def zscore_covariates(covariates: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=covariates.index)
    for col in covariates.columns:
        values = covariates[col].to_numpy(float)
        if not np.isfinite(values).all() or np.nanstd(values) <= 0:
            continue
        out[col] = (values - np.nanmean(values)) / np.nanstd(values)
    return out


def fit_condition_effect_with_covariates(
    response: np.ndarray,
    condition: np.ndarray,
    covariates: pd.DataFrame,
) -> dict[str, float]:
    y = np.asarray(response, dtype=float)
    condition = np.asarray(condition, dtype=float)
    valid = np.isfinite(y) & np.isfinite(condition)
    cov = covariates.copy()
    for col in cov.columns:
        valid &= np.isfinite(cov[col].to_numpy(float))
    y = y[valid]
    condition = condition[valid]
    cov = cov.loc[valid].reset_index(drop=True)
    cov_z = zscore_covariates(cov)
    if len(y) < 3 or np.unique(condition).size < 2:
        return {
            "condition_beta": np.nan,
            "condition_t": np.nan,
            "condition_p": np.nan,
            "n_trials_used": int(len(y)),
            "n_covariates_used": int(cov_z.shape[1]),
            "model_r2": np.nan,
        }
    x = np.column_stack([np.ones(len(y)), condition])
    if cov_z.shape[1] > 0:
        x = np.column_stack([x, cov_z.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    resid = y - yhat
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / denom) if denom > 0 else np.nan
    dof = len(y) - x.shape[1]
    if dof > 0:
        sigma2 = float(np.sum(resid**2) / dof)
        xtx_inv = np.linalg.pinv(x.T @ x)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
        if se[1] > 0:
            t_val = float(beta[1] / se[1])
            p_val = float(2 * stats.t.sf(abs(t_val), df=dof))
        else:
            t_val = np.nan
            p_val = np.nan
    else:
        t_val = np.nan
        p_val = np.nan
    return {
        "condition_beta": float(beta[1]),
        "condition_t": t_val,
        "condition_p": p_val,
        "n_trials_used": int(len(y)),
        "n_covariates_used": int(cov_z.shape[1]),
        "model_r2": r2,
    }


def raw_condition_effect(response: np.ndarray, condition: np.ndarray) -> dict[str, float]:
    y = np.asarray(response, dtype=float)
    condition = np.asarray(condition, dtype=int)
    active = y[condition == 1]
    passive = y[condition == 0]
    if len(active) == 0 or len(passive) == 0:
        return {"raw_effect": np.nan, "raw_p": np.nan}
    raw_effect = float(np.mean(active) - np.mean(passive))
    if len(active) >= 2 and len(passive) >= 2:
        raw_p = float(stats.mannwhitneyu(active, passive, alternative="two-sided").pvalue)
    else:
        raw_p = np.nan
    return {"raw_effect": raw_effect, "raw_p": raw_p}


def paired_condition_artifact_balance(trials: pd.DataFrame) -> dict[str, float]:
    diffs: list[float] = []
    for _, sub in trials.groupby("pair_id"):
        active = sub.loc[sub["condition_simple"] == "active", "artifact_fraction"].to_numpy(float)
        passive = sub.loc[sub["condition_simple"] == "passive", "artifact_fraction"].to_numpy(float)
        if len(active) == 1 and len(passive) == 1:
            diffs.append(float(active[0] - passive[0]))
    diffs_arr = np.asarray(diffs, dtype=float)
    out = {
        "n_pairs": int(len(diffs_arr)),
        "mean_active_minus_passive_artifact_fraction": float(np.mean(diffs_arr)) if len(diffs_arr) else np.nan,
        "median_active_minus_passive_artifact_fraction": float(np.median(diffs_arr)) if len(diffs_arr) else np.nan,
        "fraction_pairs_active_gt_passive_artifact": float(np.mean(diffs_arr > 0)) if len(diffs_arr) else np.nan,
        "wilcoxon_artifact_balance_p": np.nan,
    }
    if len(diffs_arr) >= 3 and np.any(np.abs(diffs_arr) > 1e-12):
        out["wilcoxon_artifact_balance_p"] = float(stats.wilcoxon(diffs_arr, alternative="two-sided").pvalue)
    return out


def leave_pair_out_artifact_decoding(features: pd.DataFrame, y: np.ndarray, pair_id: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pair_id = np.asarray(pair_id)
    x = features.copy()
    keep_cols = []
    for col in x.columns:
        values = x[col].to_numpy(float)
        if np.isfinite(values).all() and np.nanstd(values) > 0:
            keep_cols.append(col)
    if len(keep_cols) == 0 or np.unique(y).size < 2:
        return {"auc": np.nan, "balanced_accuracy": np.nan, "n_heldout_pairs": 0}
    x = x[keep_cols].to_numpy(float)
    prob = np.full(len(y), np.nan, dtype=float)
    n_pairs = 0
    for pair in pd.unique(pair_id):
        test = pair_id == pair
        if np.unique(y[test]).size < 2:
            continue
        train = ~test
        if np.unique(y[train]).size < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=1000),
        )
        model.fit(x[train], y[train])
        prob[test] = model.predict_proba(x[test])[:, 1]
        n_pairs += 1
    valid = np.isfinite(prob)
    if valid.sum() == 0 or np.unique(y[valid]).size < 2:
        return {"auc": np.nan, "balanced_accuracy": np.nan, "n_heldout_pairs": int(n_pairs)}
    return {
        "auc": float(roc_auc_score(y[valid], prob[valid])),
        "balanced_accuracy": float(balanced_accuracy_score(y[valid], prob[valid] >= 0.5)),
        "n_heldout_pairs": int(n_pairs),
    }


def compute_unit_controls(
    metrics: pd.DataFrame,
    trials: pd.DataFrame,
    threshold_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    merged = metrics.merge(
        trials[["run", "trial_index", "artifact_fraction", "max_abs_count_max", "p999_abs_count_max"]],
        on=["run", "trial_index"],
        how="inner",
    )
    for (subject, run, unit_id), sub in merged.groupby(["subject", "run", "unit_id"]):
        condition = (sub["condition_simple"].to_numpy() == "active").astype(int)
        response = sub[METRIC].to_numpy(float)
        raw = raw_condition_effect(response, condition)
        controlled = fit_condition_effect_with_covariates(response, condition, sub[ARTIFACT_FEATURES])
        rows.append(
            {
                "subject": subject,
                "run": run,
                "unit_id": int(unit_id),
                "ks_label": sub["ks_label"].astype(str).mode().iloc[0],
                "threshold": threshold_name,
                "n_trials": int(sub["trial_index"].nunique()),
                "n_active": int(np.sum(condition == 1)),
                "n_passive": int(np.sum(condition == 0)),
                **raw,
                "artifact_controlled_effect": controlled["condition_beta"],
                "artifact_controlled_t": controlled["condition_t"],
                "artifact_controlled_p": controlled["condition_p"],
                "n_covariates_used": controlled["n_covariates_used"],
                "artifact_control_model_r2": controlled["model_r2"],
                "effect_delta_controlled_minus_raw": controlled["condition_beta"] - raw["raw_effect"],
                "abs_effect_delta": abs(controlled["condition_beta"] - raw["raw_effect"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_unit_controls(unit_df: pd.DataFrame) -> dict[str, float]:
    if unit_df.empty:
        return {
            "n_units": 0,
            "median_raw_effect": np.nan,
            "median_artifact_controlled_effect": np.nan,
            "median_abs_raw_effect": np.nan,
            "median_abs_artifact_controlled_effect": np.nan,
            "median_abs_effect_delta": np.nan,
            "raw_controlled_spearman_r": np.nan,
            "fraction_sign_flip": np.nan,
            "n_raw_p05": 0,
            "n_artifact_controlled_p05": 0,
        }
    raw = unit_df["raw_effect"].to_numpy(float)
    controlled = unit_df["artifact_controlled_effect"].to_numpy(float)
    finite = np.isfinite(raw) & np.isfinite(controlled)
    if finite.sum() >= 3 and np.nanstd(raw[finite]) > 0 and np.nanstd(controlled[finite]) > 0:
        corr = float(stats.spearmanr(raw[finite], controlled[finite]).correlation)
    else:
        corr = np.nan
    nonzero = finite & (np.abs(raw) > 1e-12) & (np.abs(controlled) > 1e-12)
    sign_flip = float(np.mean(np.sign(raw[nonzero]) != np.sign(controlled[nonzero]))) if nonzero.sum() else np.nan
    return {
        "n_units": int(unit_df.shape[0]),
        "median_raw_effect": float(np.nanmedian(raw)),
        "median_artifact_controlled_effect": float(np.nanmedian(controlled)),
        "median_abs_raw_effect": float(np.nanmedian(np.abs(raw))),
        "median_abs_artifact_controlled_effect": float(np.nanmedian(np.abs(controlled))),
        "median_abs_effect_delta": float(np.nanmedian(unit_df["abs_effect_delta"].to_numpy(float))),
        "raw_controlled_spearman_r": corr,
        "fraction_sign_flip": sign_flip,
        "n_raw_p05": int((unit_df["raw_p"] < 0.05).sum()),
        "n_artifact_controlled_p05": int((unit_df["artifact_controlled_p"] < 0.05).sum()),
    }


def summarize_run(
    subject: str,
    run: str,
    trials: pd.DataFrame,
    unit_df: pd.DataFrame,
    threshold_name: str,
) -> dict[str, object]:
    y = (trials["condition_simple"].to_numpy() == "active").astype(int)
    decoding = leave_pair_out_artifact_decoding(trials[ARTIFACT_FEATURES], y, trials["pair_id"].to_numpy())
    balance = paired_condition_artifact_balance(trials)
    active = trials[trials["condition_simple"] == "active"]
    passive = trials[trials["condition_simple"] == "passive"]
    return {
        "subject": subject,
        "run": run,
        "threshold": threshold_name,
        "n_trials": int(trials.shape[0]),
        "n_active": int(active.shape[0]),
        "n_passive": int(passive.shape[0]),
        "mean_artifact_fraction_active": float(active["artifact_fraction"].mean()) if len(active) else np.nan,
        "mean_artifact_fraction_passive": float(passive["artifact_fraction"].mean()) if len(passive) else np.nan,
        "median_artifact_fraction_active": float(active["artifact_fraction"].median()) if len(active) else np.nan,
        "median_artifact_fraction_passive": float(passive["artifact_fraction"].median()) if len(passive) else np.nan,
        "max_artifact_fraction": float(trials["artifact_fraction"].max()) if len(trials) else np.nan,
        "artifact_only_auc": decoding["auc"],
        "artifact_only_balanced_accuracy": decoding["balanced_accuracy"],
        "artifact_only_n_heldout_pairs": decoding["n_heldout_pairs"],
        **balance,
        **summarize_unit_controls(unit_df),
    }


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


def make_artifact_balance_plot(run_df: pd.DataFrame, out_dir: Path) -> str | None:
    df = run_df[run_df["threshold"] == "all_trials"].copy()
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = [f"{r.subject}\n{r.run.split('_')[-1]}" for r in df.itertuples()]
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["mean_artifact_fraction_active"], width=0.36, label="active", color="#2f6fbb")
    ax.bar(x + 0.18, df["mean_artifact_fraction_passive"], width=0.36, label="passive", color="#c43c39")
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("Mean artifact fraction (-1 to +2 s)")
    ax.set_title("Trial-window artifact load by condition")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_artifact_balance_by_run.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_artifact_decoding_plot(run_df: pd.DataFrame, out_dir: Path) -> str | None:
    if run_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    thresholds = list(run_df["threshold"].drop_duplicates())
    x_lookup = {name: i for i, name in enumerate(thresholds)}
    for run, sub in run_df.groupby("run"):
        xs = [x_lookup[name] for name in sub["threshold"]]
        ax.plot(xs, sub["artifact_only_auc"], marker="o", lw=1.0, alpha=0.75, label=run)
    ax.axhline(0.5, color="0.5", ls="--", lw=1.0)
    ax.set_xticks(range(len(thresholds)), thresholds, rotation=25, ha="right")
    ax.set_ylabel("Leave-pair-out AUC")
    ax.set_title("Condition decoding from artifact/spike-field load only")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_artifact_only_decoding_auc.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_raw_vs_controlled_plot(unit_df: pd.DataFrame, main_threshold: str, out_dir: Path) -> str | None:
    df = unit_df[unit_df["threshold"] == main_threshold].copy()
    if df.empty:
        return None
    runs = list(df["run"].drop_duplicates())
    n_cols = min(4, len(runs))
    n_rows = int(np.ceil(len(runs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        sub = df[df["run"] == run]
        ax.scatter(sub["raw_effect"], sub["artifact_controlled_effect"], s=18, alpha=0.6, color="#2f6fbb")
        lim = np.nanmax(np.abs(np.r_[sub["raw_effect"].to_numpy(float), sub["artifact_controlled_effect"].to_numpy(float)]))
        if np.isfinite(lim) and lim > 0:
            ax.plot([-lim, lim], [-lim, lim], color="0.4", ls="--", lw=1.0)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_title(run.split("_")[-1], fontsize=9)
        ax.set_xlabel("Raw effect (Hz)")
        ax.set_ylabel("Artifact-controlled beta (Hz)")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(runs) :]:
        ax.axis("off")
    fig.suptitle(f"Raw versus artifact-controlled unit effects ({main_threshold})", y=1.01)
    fig.tight_layout()
    path = out_dir / "fig_raw_vs_artifact_controlled_effects.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def make_threshold_sensitivity_plot(run_df: pd.DataFrame, out_dir: Path) -> str | None:
    if run_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    thresholds = list(run_df["threshold"].drop_duplicates())
    x_lookup = {name: i for i, name in enumerate(thresholds)}
    for _, sub in run_df.groupby("run"):
        sub = sub.sort_values("threshold", key=lambda s: s.map(x_lookup))
        xs = [x_lookup[name] for name in sub["threshold"]]
        ax.plot(xs, sub["median_abs_raw_effect"], marker="o", lw=1.0, alpha=0.45, color="#2f6fbb")
        ax.plot(xs, sub["median_abs_artifact_controlled_effect"], marker="s", lw=1.0, alpha=0.45, color="#c43c39")
    raw_median = run_df.groupby("threshold", observed=False)["median_abs_raw_effect"].median().reindex(thresholds)
    ctrl_median = run_df.groupby("threshold", observed=False)["median_abs_artifact_controlled_effect"].median().reindex(thresholds)
    ax.plot(range(len(thresholds)), raw_median, marker="o", lw=2.5, color="#2f6fbb", label="Raw median across runs")
    ax.plot(range(len(thresholds)), ctrl_median, marker="s", lw=2.5, color="#c43c39", label="Controlled median across runs")
    ax.set_xticks(range(len(thresholds)), thresholds, rotation=25, ha="right")
    ax.set_ylabel("Median absolute unit effect (Hz)")
    ax.set_title("Artifact threshold sensitivity")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_artifact_threshold_sensitivity.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_figures(run_df: pd.DataFrame, unit_df: pd.DataFrame, out_dir: Path, main_threshold: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key, path in [
        ("artifact_balance_png", make_artifact_balance_plot(run_df, out_dir)),
        ("artifact_only_decoding_auc_png", make_artifact_decoding_plot(run_df, out_dir)),
        ("raw_vs_artifact_controlled_effects_png", make_raw_vs_controlled_plot(unit_df, main_threshold, out_dir)),
        ("artifact_threshold_sensitivity_png", make_threshold_sensitivity_plot(run_df, out_dir)),
    ]:
        if path:
            paths[key] = path
    return paths


def write_report(
    run_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    aggregate: dict[str, object],
    figure_paths: dict[str, str],
    main_threshold: str,
) -> None:
    main = run_df[run_df["threshold"] == main_threshold].copy()
    all_trials = run_df[run_df["threshold"] == "all_trials"].copy()
    lines = [
        "# Active/passive IED, artifact and spike-field-load controls",
        "",
        "This control analysis recomputes trial-window artifact/IED load from each run's 1000 Hz artifact mask, then asks whether active/passive effects survive artifact-thresholding and unitwise regression against artifact/spike-field-load covariates.",
        "The continuous covariates are artifact fraction, max_abs_count_max and p999_abs_count_max in the -1 to +2 s sentence-onset window that contains the function-word response window.",
        "",
        "## Interpretation Notes",
        "",
    ]
    if not all_trials.empty:
        auc = all_trials["artifact_only_auc"].dropna()
        balance = all_trials["mean_active_minus_passive_artifact_fraction"].dropna()
        lines.append(
            f"- Artifact-only condition decoding across all available trials: median AUC={auc.median():.3g} across {len(auc)} runs. Values near 0.5 mean artifact load alone does not reconstruct active/passive labels."
        )
        lines.append(
            f"- Active-minus-passive artifact fraction imbalance: median paired difference={balance.median():.3g} across {len(balance)} runs."
        )
    if not main.empty:
        lines.append(
            f"- At the main threshold ({main_threshold}), median raw/control Spearman across runs={main['raw_controlled_spearman_r'].median():.3g}; median sign-flip fraction={main['fraction_sign_flip'].median():.3g}."
        )
        lines.append(
            f"- At the main threshold, median absolute unit effect changes by {main['median_abs_effect_delta'].median():.3g} Hz after artifact/spike-field-load control."
        )
    lines.extend(
        [
            "",
            "A strong artifact confound would show condition-decodable artifact load, systematic active/passive artifact imbalance, and large collapse or sign flips after covariate control. A robust neural result should not depend on those patterns.",
            "",
            "## Main Threshold Run Summary",
            "",
            markdown_table(main),
            "",
            "## All Threshold Run Summary",
            "",
            markdown_table(run_df),
            "",
            "## Unit-Level Controls",
            "",
            markdown_table(unit_df.head(60)),
            "",
            "## Skipped Run/Thresholds",
            "",
            markdown_table(skipped_df),
            "",
            "## Aggregate Metadata",
            "",
            "```json",
            json.dumps(aggregate, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Figures",
            "",
        ]
    )
    for key, path in figure_paths.items():
        lines.append(f"- {key}: `{path}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_thresholds(text: str) -> list[float | None]:
    out: list[float | None] = []
    for part in text.split(","):
        value = part.strip().lower()
        if value in {"none", "all", "all_trials"}:
            out.append(None)
        else:
            out.append(float(value))
    return out


def run_analysis(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    main_threshold_name = threshold_label(args.main_threshold)

    trial_feature_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []
    unit_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []

    for er_dir in find_event_dirs():
        subject = subject_from_er_dir(er_dir)
        metrics = apply_unit_filter(load_metrics(er_dir), args.unit_set)
        trial_features = compute_artifact_features_for_event_dir(er_dir)
        trial_feature_frames.append(trial_features)
        for run in sorted(trial_features["run"].unique()):
            run_metrics = metrics[metrics["run"] == run].copy()
            run_trials_all = trial_features[trial_features["run"] == run].copy()
            for threshold in thresholds:
                name = threshold_label(threshold)
                run_trials = filter_trials_by_threshold(run_trials_all, threshold)
                n_active = int((run_trials["condition_simple"] == "active").sum())
                n_passive = int((run_trials["condition_simple"] == "passive").sum())
                common = {
                    "subject": subject,
                    "run": run,
                    "threshold": name,
                    "n_trials": int(run_trials.shape[0]),
                    "n_active": n_active,
                    "n_passive": n_passive,
                }
                if run_trials.shape[0] < args.min_trials_total or min(n_active, n_passive) < args.min_trials_per_class:
                    skipped_rows.append({**common, "reason": "insufficient active/passive trials after artifact threshold"})
                    continue
                if run_metrics.empty:
                    skipped_rows.append({**common, "reason": "no units after unit filter"})
                    continue
                unit_df = compute_unit_controls(run_metrics, run_trials, name)
                if unit_df.empty:
                    skipped_rows.append({**common, "reason": "unit control table empty"})
                    continue
                unit_frames.append(unit_df)
                run_rows.append(summarize_run(subject, run, run_trials, unit_df, name))
                print(f"analyzed {subject} {run} {name}: trials={run_trials.shape[0]} units={unit_df.shape[0]}", flush=True)

    trial_features_df = pd.concat(trial_feature_frames, ignore_index=True) if trial_feature_frames else pd.DataFrame()
    run_df = pd.DataFrame(run_rows)
    unit_df = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped_rows)
    threshold_order = [threshold_label(t) for t in thresholds]
    if not run_df.empty:
        run_df["threshold"] = pd.Categorical(run_df["threshold"], categories=threshold_order, ordered=True)
        run_df = run_df.sort_values(["subject", "run", "threshold"]).reset_index(drop=True)
    if not unit_df.empty:
        unit_df["threshold"] = pd.Categorical(unit_df["threshold"], categories=threshold_order, ordered=True)
        unit_df = unit_df.sort_values(["subject", "run", "threshold", "unit_id"]).reset_index(drop=True)

    trial_path = OUT_DIR / "active_passive_trial_artifact_spikefield_features.csv"
    run_path = OUT_DIR / "active_passive_ied_artifact_control_run_summary.csv"
    unit_path = OUT_DIR / "active_passive_ied_artifact_control_unit_effects.csv"
    skipped_path = OUT_DIR / "active_passive_ied_artifact_control_skipped.csv"
    summary_path = OUT_DIR / "active_passive_ied_artifact_control_summary.json"
    trial_features_df.to_csv(trial_path, index=False)
    run_df.to_csv(run_path, index=False)
    unit_df.to_csv(unit_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    figure_paths = make_figures(run_df, unit_df, OUT_DIR, main_threshold_name)
    aggregate = {
        "unit_set": args.unit_set,
        "metric": METRIC,
        "thresholds": threshold_order,
        "main_threshold": main_threshold_name,
        "n_runs_analyzed_threshold_rows": int(run_df.shape[0]),
        "n_unit_effect_rows": int(unit_df.shape[0]),
        "n_skipped_threshold_rows": int(skipped_df.shape[0]),
        "artifact_feature_window": "-1 to +2 s around sentence onset marker, matching prior artifact screening",
        "outputs": {
            "trial_artifact_spikefield_features_csv": str(trial_path),
            "run_summary_csv": str(run_path),
            "unit_effects_csv": str(unit_path),
            "skipped_csv": str(skipped_path),
            "summary_json": str(summary_path),
            "report_md": str(REPORT_PATH),
            **figure_paths,
        },
    }
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(run_df, unit_df, skipped_df, aggregate, figure_paths, main_threshold_name)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=["good_only", "all_clusters"], default="good_only")
    parser.add_argument("--thresholds", default="none,0.5,0.2,0.1,0.0")
    parser.add_argument("--main-threshold", type=float, default=0.2)
    parser.add_argument("--min-trials-total", type=int, default=10)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
