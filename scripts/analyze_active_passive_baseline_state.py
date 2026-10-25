#!/usr/bin/env python3
"""Baseline-state analysis for active/passive function-word responses."""

from __future__ import annotations

import argparse
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
OUT_DIR = DATA_ROOT / "combined_active_passive_baseline_state"
REPORT_PATH = ROOT / "reports/active_passive_baseline_state.md"


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


def standardize_units(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mean = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (x - mean) / sd


def make_run_matrices(run_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[int]]:
    baseline = run_df.pivot_table(index="trial_index", columns="unit_id", values="baseline_hz", aggfunc="mean").sort_index()
    response = run_df.pivot_table(index="trial_index", columns="unit_id", values="response_baseline_corrected_hz", aggfunc="mean").sort_index()
    raw_response = run_df.pivot_table(index="trial_index", columns="unit_id", values="response_hz", aggfunc="mean").sort_index()
    common_trials = baseline.index.intersection(response.index).intersection(raw_response.index)
    common_units = baseline.columns.intersection(response.columns).intersection(raw_response.columns)
    baseline = baseline.loc[common_trials, common_units]
    response = response.loc[common_trials, common_units]
    raw_response = raw_response.loc[common_trials, common_units]
    trial_meta = (
        run_df.drop_duplicates("trial_index")
        .set_index("trial_index")
        .loc[common_trials]
        .reset_index()
        .sort_values("trial_index")
        .reset_index(drop=True)
    )

    xb = baseline.to_numpy(dtype=float)
    xr = response.to_numpy(dtype=float)
    xraw = raw_response.to_numpy(dtype=float)
    finite = np.isfinite(xb).all(axis=0) & np.isfinite(xr).all(axis=0) & np.isfinite(xraw).all(axis=0)
    variable = (np.nanstd(xb, axis=0) > 0) | (np.nanstd(xr, axis=0) > 0) | (np.nanstd(xraw, axis=0) > 0)
    keep = finite & variable
    return xb[:, keep], xr[:, keep], xraw[:, keep], trial_meta, [int(unit_id) for unit_id in common_units[keep]]


def training_t_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    active = x[y == 1]
    passive = x[y == 0]
    active_var = active.var(axis=0, ddof=1) if active.shape[0] > 1 else np.zeros(x.shape[1])
    passive_var = passive.var(axis=0, ddof=1) if passive.shape[0] > 1 else np.zeros(x.shape[1])
    denom = np.sqrt(active_var / max(active.shape[0], 1) + passive_var / max(passive.shape[0], 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs((active.mean(axis=0) - passive.mean(axis=0)) / denom)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def select_top_units(x: np.ndarray, y: np.ndarray, top_k: int | None) -> np.ndarray:
    if top_k is None or top_k >= x.shape[1]:
        return np.arange(x.shape[1])
    order = np.argsort(training_t_scores(x, y))[::-1]
    return order[: max(1, min(top_k, x.shape[1]))]


def leave_pair_out_decoding(
    x: np.ndarray,
    y: np.ndarray,
    pair_id: np.ndarray,
    top_k: int | None = 5,
    c: float = 0.1,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    pair_id = np.asarray(pair_id)
    prob = np.full(len(y), np.nan, dtype=float)
    heldout_pairs: list[object] = []
    for pair in pd.unique(pair_id):
        test = pair_id == pair
        if np.unique(y[test]).size < 2:
            continue
        train = ~test
        if np.unique(y[train]).size < 2:
            continue
        cols = select_top_units(x[train], y[train], top_k)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, solver="liblinear", class_weight="balanced", max_iter=1000),
        )
        model.fit(x[train][:, cols], y[train])
        prob[test] = model.predict_proba(x[test][:, cols])[:, 1]
        heldout_pairs.append(pair)

    valid = np.isfinite(prob)
    if valid.sum() == 0 or np.unique(y[valid]).size < 2:
        auc = np.nan
        balanced_accuracy = np.nan
    else:
        auc = float(roc_auc_score(y[valid], prob[valid]))
        balanced_accuracy = float(balanced_accuracy_score(y[valid], prob[valid] >= 0.5))

    pair_positive: list[float] = []
    pair_diffs: list[float] = []
    for pair in heldout_pairs:
        sub = (pair_id == pair) & valid
        active_prob = prob[sub & (y == 1)]
        passive_prob = prob[sub & (y == 0)]
        if len(active_prob) == 1 and len(passive_prob) == 1:
            diff = float(active_prob[0] - passive_prob[0])
            pair_diffs.append(diff)
            pair_positive.append(float(diff > 0))

    return {
        "auc": auc,
        "balanced_accuracy": balanced_accuracy,
        "frac_active_probability_gt_passive": float(np.mean(pair_positive)) if pair_positive else np.nan,
        "mean_active_minus_passive_probability": float(np.mean(pair_diffs)) if pair_diffs else np.nan,
        "n_heldout_pairs": int(len(pair_positive)),
    }


def condition_axis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    axis = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
    norm = float(np.linalg.norm(axis))
    if norm <= 0 or not np.isfinite(norm):
        return np.zeros(x.shape[1], dtype=float)
    return axis / norm


def syntax_axis_dprime(x: np.ndarray, y: np.ndarray) -> float:
    axis = condition_axis(x, y)
    if not np.any(axis):
        return np.nan
    scores = x @ axis
    active = scores[y == 1]
    passive = scores[y == 0]
    denom = np.sqrt(((len(active) - 1) * np.var(active, ddof=1) + (len(passive) - 1) * np.var(passive, ddof=1)) / max(len(active) + len(passive) - 2, 1))
    return float((np.mean(active) - np.mean(passive)) / denom) if denom > 0 else np.nan


def leave_one_pair_out_axis_projection(x: np.ndarray, y: np.ndarray, pair_id: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    pair_id = np.asarray(pair_id)
    diffs: list[float] = []
    for pair in pd.unique(pair_id):
        test = pair_id == pair
        if np.unique(y[test]).size < 2:
            continue
        train = ~test
        if np.unique(y[train]).size < 2:
            continue
        axis = condition_axis(x[train], y[train])
        if not np.any(axis):
            continue
        projection = x @ axis
        active = projection[test & (y == 1)]
        passive = projection[test & (y == 0)]
        if len(active) == 1 and len(passive) == 1:
            diffs.append(float(active[0] - passive[0]))
    diffs_arr = np.asarray(diffs, dtype=float)
    if len(diffs_arr) == 0:
        return {
            "cv_axis_mean_diff": np.nan,
            "cv_axis_median_diff": np.nan,
            "cv_axis_frac_pair_positive": np.nan,
        }
    return {
        "cv_axis_mean_diff": float(np.mean(diffs_arr)),
        "cv_axis_median_diff": float(np.median(diffs_arr)),
        "cv_axis_frac_pair_positive": float(np.mean(diffs_arr > 0)),
    }


def residualize_response_by_baseline(response: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    residual = np.empty_like(response)
    for col in range(response.shape[1]):
        design = np.column_stack([np.ones(response.shape[0]), baseline[:, col]])
        beta, *_ = np.linalg.lstsq(design, response[:, col], rcond=None)
        residual[:, col] = response[:, col] - design @ beta
    return residual


def median_unit_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for col in range(a.shape[1]):
        if np.std(a[:, col]) == 0 or np.std(b[:, col]) == 0:
            r = np.nan
        else:
            r = float(np.corrcoef(a[:, col], b[:, col])[0, 1])
        rows.append({"unit_col": col, "baseline_response_r": r})
    values = np.array([row["baseline_response_r"] for row in rows], dtype=float)
    return (float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan), pd.DataFrame(rows)


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def analyze_run(
    subject: str,
    run: str,
    run_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    baseline_raw, response_bc_raw, response_raw_raw, trial_meta, unit_ids = make_run_matrices(run_df)
    if baseline_raw.shape[0] < 2 or baseline_raw.shape[1] < 1:
        raise ValueError("Need at least two trials and one unit")

    baseline = standardize_units(baseline_raw)
    response_bc = standardize_units(response_bc_raw)
    response_raw = standardize_units(response_raw_raw)
    response_residual = standardize_units(residualize_response_by_baseline(response_bc, baseline))

    y = (trial_meta["condition_simple"].to_numpy() == "active").astype(int)
    pair_id = trial_meta["pair_id"].to_numpy()

    baseline_axis = condition_axis(baseline, y)
    response_axis = condition_axis(response_bc, y)
    residual_axis = condition_axis(response_residual, y)
    axis_cosine = float(np.dot(baseline_axis, response_axis)) if np.any(baseline_axis) and np.any(response_axis) else np.nan
    residual_axis_cosine = float(np.dot(baseline_axis, residual_axis)) if np.any(baseline_axis) and np.any(residual_axis) else np.nan

    median_r, unit_corr = median_unit_correlation(baseline, response_bc)
    unit_corr["subject"] = subject
    unit_corr["run"] = run
    unit_corr["unit_id"] = unit_ids

    baseline_decoding = leave_pair_out_decoding(baseline, y, pair_id, top_k=args.top_k, c=args.c)
    response_decoding = leave_pair_out_decoding(response_bc, y, pair_id, top_k=args.top_k, c=args.c)
    raw_response_decoding = leave_pair_out_decoding(response_raw, y, pair_id, top_k=args.top_k, c=args.c)
    residual_decoding = leave_pair_out_decoding(response_residual, y, pair_id, top_k=args.top_k, c=args.c)

    row: dict[str, object] = {
        "subject": subject,
        "run": run,
        "n_trials": int(baseline.shape[0]),
        "n_units_variable": int(baseline.shape[1]),
        "baseline_response_axis_cosine": axis_cosine,
        "baseline_residual_axis_cosine": residual_axis_cosine,
        "baseline_syntax_axis_dprime": syntax_axis_dprime(baseline, y),
        "response_syntax_axis_dprime": syntax_axis_dprime(response_bc, y),
        "raw_response_syntax_axis_dprime": syntax_axis_dprime(response_raw, y),
        "residual_response_syntax_axis_dprime": syntax_axis_dprime(response_residual, y),
        "median_unit_baseline_response_r": median_r,
        **prefixed("baseline", baseline_decoding),
        **prefixed("response", response_decoding),
        **prefixed("raw_response", raw_response_decoding),
        **prefixed("residual_response", residual_decoding),
        **prefixed("baseline", leave_one_pair_out_axis_projection(baseline, y, pair_id)),
        **prefixed("response", leave_one_pair_out_axis_projection(response_bc, y, pair_id)),
        **prefixed("residual_response", leave_one_pair_out_axis_projection(response_residual, y, pair_id)),
    }
    row["response_auc_minus_baseline_auc"] = float(row["response_auc"] - row["baseline_auc"])
    row["response_auc_minus_raw_response_auc"] = float(row["response_auc"] - row["raw_response_auc"])
    row["response_auc_minus_residual_auc"] = float(row["response_auc"] - row["residual_response_auc"])
    row["response_cv_axis_mean_minus_baseline"] = float(row["response_cv_axis_mean_diff"] - row["baseline_cv_axis_mean_diff"])
    row["response_cv_axis_mean_minus_residual"] = float(row["response_cv_axis_mean_diff"] - row["residual_response_cv_axis_mean_diff"])
    return row, unit_corr


def signed_rank(values: np.ndarray, reference: float = 0.0, alternative: str = "greater") -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out = {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else np.nan,
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "reference": float(reference),
        "wilcoxon_p": np.nan,
    }
    if len(values) >= 3 and np.any(np.abs(values - reference) > 1e-12):
        out["wilcoxon_p"] = float(stats.wilcoxon(values - reference, alternative=alternative).pvalue)
    return out


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


def make_metric_dot_plot(
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    ylabel: str,
    title: str,
    path: Path,
    reference: float | None = None,
) -> str | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x_pos = np.arange(len(columns))
    for i, col in enumerate(columns):
        vals = df[col].to_numpy(float)
        jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0]
        ax.scatter(np.full(len(vals), x_pos[i]) + jitter, vals, s=44, color="#2f6fbb", alpha=0.84)
        ax.plot([x_pos[i] - 0.26, x_pos[i] + 0.26], [np.nanmedian(vals)] * 2, color="black", lw=2.2)
    if reference is not None:
        ax.axhline(reference, color="0.45", ls="--", lw=1.2)
    ax.set_xticks(x_pos, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_figures(run_summary: pd.DataFrame, unit_corr: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    auc_path = make_metric_dot_plot(
        run_summary,
        ["baseline_auc", "raw_response_auc", "response_auc", "residual_response_auc"],
        ["Baseline", "Raw evoked", "Baseline-corrected", "Corrected residual"],
        "Leave-pair-out AUC",
        "Active/passive decoding from baseline and evoked states",
        out_dir / "fig_baseline_vs_evoked_decoding_auc.png",
        reference=0.5,
    )
    if auc_path:
        paths["decoding_auc_png"] = auc_path
    axis_path = make_metric_dot_plot(
        run_summary,
        ["baseline_cv_axis_frac_pair_positive", "response_cv_axis_frac_pair_positive", "residual_response_cv_axis_frac_pair_positive"],
        ["Baseline", "Evoked", "Evoked residual"],
        "Fraction held-out pairs active > passive",
        "Leave-one-pair-out syntax-axis projection",
        out_dir / "fig_baseline_vs_evoked_axis_projection.png",
        reference=0.5,
    )
    if axis_path:
        paths["axis_projection_png"] = axis_path
    cosine_path = make_metric_dot_plot(
        run_summary,
        ["baseline_response_axis_cosine", "baseline_residual_axis_cosine"],
        ["Baseline vs evoked", "Baseline vs residual"],
        "Cosine similarity",
        "Baseline syntax-axis alignment with evoked axes",
        out_dir / "fig_baseline_evoked_axis_cosine.png",
        reference=0.0,
    )
    if cosine_path:
        paths["axis_cosine_png"] = cosine_path
    if not unit_corr.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.hist(unit_corr["baseline_response_r"].dropna().to_numpy(float), bins=35, color="#2f6fbb", alpha=0.82)
        ax.axvline(0, color="0.45", ls="--", lw=1.2)
        ax.set_xlabel("Unitwise baseline-response correlation")
        ax.set_ylabel("Unit count")
        ax.set_title("Trial-by-trial baseline/evoked coupling across units")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = out_dir / "fig_unit_baseline_response_correlations.png"
        fig.savefig(path, dpi=240)
        plt.close(fig)
        paths["unit_baseline_response_correlation_png"] = str(path)
    return paths


def aggregate_summary(run_summary: pd.DataFrame) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for col in [
        "baseline_auc",
        "response_auc",
        "raw_response_auc",
        "residual_response_auc",
        "baseline_cv_axis_frac_pair_positive",
        "response_cv_axis_frac_pair_positive",
        "residual_response_cv_axis_frac_pair_positive",
    ]:
        if col in run_summary:
            aggregate[col] = signed_rank(run_summary[col].to_numpy(float), reference=0.5, alternative="greater")
    for col in [
        "baseline_response_axis_cosine",
        "baseline_residual_axis_cosine",
        "baseline_syntax_axis_dprime",
        "response_syntax_axis_dprime",
        "residual_response_syntax_axis_dprime",
        "median_unit_baseline_response_r",
        "response_auc_minus_baseline_auc",
        "response_auc_minus_raw_response_auc",
        "response_auc_minus_residual_auc",
        "response_cv_axis_mean_minus_baseline",
        "response_cv_axis_mean_minus_residual",
    ]:
        if col in run_summary:
            aggregate[col] = signed_rank(run_summary[col].to_numpy(float), reference=0.0, alternative="greater")
    return aggregate


def agg_value(aggregate: dict[str, object], key: str, field: str = "median") -> str:
    value = aggregate.get(key, {})
    if not isinstance(value, dict) or field not in value:
        return "NA"
    number = value[field]
    return "NA" if not np.isfinite(number) else f"{number:.4g}"


def write_report(
    run_summary: pd.DataFrame,
    skipped: pd.DataFrame,
    aggregate: dict[str, object],
    figure_paths: dict[str, str],
) -> None:
    lines = [
        "# Active/passive baseline-state hypothesis",
        "",
        "This analysis asks whether the pre-function-word baseline population state already carries active/passive information, and whether that state can account for evoked active/passive structure.",
        "The baseline window comes from the existing event-related metrics and should be interpreted as pre-function-word context, not a silent pre-sentence rest state.",
        "",
        "## Interpretation Notes",
        "",
        f"- Baseline leave-pair-out decoding: median AUC = {agg_value(aggregate, 'baseline_auc')}, Wilcoxon vs 0.5 p={agg_value(aggregate, 'baseline_auc', 'wilcoxon_p')}.",
        f"- Raw evoked response decoding: median AUC = {agg_value(aggregate, 'raw_response_auc')}, Wilcoxon vs 0.5 p={agg_value(aggregate, 'raw_response_auc', 'wilcoxon_p')}.",
        f"- Evoked baseline-corrected decoding: median AUC = {agg_value(aggregate, 'response_auc')}, Wilcoxon vs 0.5 p={agg_value(aggregate, 'response_auc', 'wilcoxon_p')}.",
        f"- Evoked residualized against unitwise baseline: median AUC = {agg_value(aggregate, 'residual_response_auc')}, Wilcoxon vs 0.5 p={agg_value(aggregate, 'residual_response_auc', 'wilcoxon_p')}.",
        f"- Baseline and evoked syntax-axis alignment: median cosine = {agg_value(aggregate, 'baseline_response_axis_cosine')}, Wilcoxon vs 0 p={agg_value(aggregate, 'baseline_response_axis_cosine', 'wilcoxon_p')}.",
        f"- Baseline-corrected evoked AUC minus raw evoked AUC: median = {agg_value(aggregate, 'response_auc_minus_raw_response_auc')}, Wilcoxon vs 0 p={agg_value(aggregate, 'response_auc_minus_raw_response_auc', 'wilcoxon_p')}.",
        f"- Evoked AUC minus residualized evoked AUC: median = {agg_value(aggregate, 'response_auc_minus_residual_auc')}, Wilcoxon vs 0 p={agg_value(aggregate, 'response_auc_minus_residual_auc', 'wilcoxon_p')}.",
        "",
        "A strong baseline-state account would predict above-chance baseline decoding, positive baseline/evoked axis alignment, and a clear drop after residualizing evoked responses against baseline. A weak baseline-state account would show only one or two of these. Because response_baseline_corrected_hz is defined as response_hz minus baseline_hz, residualization of that metric against baseline is a conservative diagnostic rather than causal evidence.",
        "",
        "## Run Summary",
        "",
        markdown_table(run_summary),
        "",
        "## Skipped Runs",
        "",
        markdown_table(skipped),
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
    analysis_mode = "all_trials" if args.artifact_threshold is None else (
        f"artifact_clean_le_{str(args.artifact_threshold).replace('.', 'p')}"
    )

    rows: list[dict[str, object]] = []
    unit_corr_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []
    for er_dir in find_event_dirs():
        subject = subject_from_er_dir(er_dir)
        metrics = load_metrics(er_dir)
        artifact = load_artifact_table(er_dir)
        filtered = apply_filters(metrics, artifact, args.unit_set, args.artifact_threshold)
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
            try:
                row, unit_corr = analyze_run(subject, run, run_df, args)
            except Exception as exc:
                skipped_rows.append({**common, "reason": f"analysis failed: {exc}"})
                continue
            rows.append({**common, **row})
            unit_corr["analysis_mode"] = analysis_mode
            unit_corr["unit_set"] = args.unit_set
            unit_corr_frames.append(unit_corr)
            print(f"analyzed {subject} {run}: trials={n_trials} units={row['n_units_variable']}", flush=True)

    run_summary = pd.DataFrame(rows)
    unit_corr_df = pd.concat(unit_corr_frames, ignore_index=True) if unit_corr_frames else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows)
    run_summary_path = OUT_DIR / "active_passive_baseline_state_run_summary.csv"
    unit_corr_path = OUT_DIR / "active_passive_baseline_state_unit_correlations.csv"
    skipped_path = OUT_DIR / "active_passive_baseline_state_skipped_runs.csv"
    summary_path = OUT_DIR / "active_passive_baseline_state_summary.json"
    run_summary.to_csv(run_summary_path, index=False)
    unit_corr_df.to_csv(unit_corr_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    figure_paths = make_figures(run_summary, unit_corr_df, OUT_DIR)
    aggregate: dict[str, object] = {
        "analysis_mode": analysis_mode,
        "unit_set": args.unit_set,
        "top_k": args.top_k,
        "logistic_c": args.c,
        "n_runs_analyzed": int(run_summary[["subject", "run"]].drop_duplicates().shape[0]) if not run_summary.empty else 0,
        "n_runs_skipped": int(skipped.shape[0]),
        "outputs": {
            "run_summary_csv": str(run_summary_path),
            "unit_correlations_csv": str(unit_corr_path),
            "skipped_runs_csv": str(skipped_path),
            "summary_json": str(summary_path),
            "report_md": str(REPORT_PATH),
            **figure_paths,
        },
        **aggregate_summary(run_summary),
    }
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(run_summary, skipped, aggregate, figure_paths)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=["good_only", "all_clusters"], default="good_only")
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--min-trials-total", type=int, default=10)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
