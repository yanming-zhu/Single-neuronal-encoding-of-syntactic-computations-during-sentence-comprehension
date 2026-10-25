#!/usr/bin/env python3
"""State-space, demixed-style marginal variance, and RSA analyses."""

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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025/data"
OUT_DIR = DATA_ROOT / "combined_active_passive_state_space_rsa"
REPORT_PATH = ROOT / "reports/active_passive_state_space_rsa.md"
METRIC = "response_baseline_corrected_hz"


def simple_condition(value: object) -> str | None:
    text = str(value)
    if text.startswith("active"):
        return "active"
    if text.startswith("passive"):
        return "passive"
    return None


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def subject_from_er_dir(er_dir: Path) -> str:
    return er_dir.relative_to(DATA_ROOT).parts[0]


def find_event_dirs() -> list[Path]:
    return sorted(DATA_ROOT.glob("gz*/processed/ActPasSynSentence/event_related_active_passive"))


def pair_id_from_sound_file(sound_file: str) -> int:
    match = re.search(r"(?:act|pass)(\d+)\.mp3$", sound_file)
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


def make_run_matrix(run_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    pivot = run_df.pivot_table(index="trial_index", columns="unit_id", values=METRIC, aggfunc="mean").sort_index()
    trial_meta = (
        run_df.drop_duplicates("trial_index")
        .set_index("trial_index")
        .loc[pivot.index]
        .reset_index()
        .sort_values("trial_index")
        .reset_index(drop=True)
    )
    x = pivot.to_numpy(dtype=float)
    finite = np.isfinite(x).all(axis=0)
    variable = np.nanstd(x, axis=0) > 0
    keep = finite & variable
    return x[:, keep], trial_meta, [int(unit_id) for unit_id in pivot.columns[keep]]


def neural_correlation_rdm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    xn = x / denom
    corr = np.clip(xn @ xn.T, -1.0, 1.0)
    return 1.0 - corr


def categorical_model_rdm(labels: np.ndarray | pd.Series | list[object]) -> np.ndarray:
    labels = np.asarray(labels)
    return (labels[:, None] != labels[None, :]).astype(float)


def rdm_vector(rdm: np.ndarray) -> np.ndarray:
    rdm = np.asarray(rdm, dtype=float)
    idx = np.tril_indices(rdm.shape[0], k=-1)
    return rdm[idx]


def rsa_spearman(neural_rdm: np.ndarray, model_rdm: np.ndarray) -> float:
    neural_vec = rdm_vector(neural_rdm)
    model_vec = rdm_vector(model_rdm)
    if np.nanstd(neural_vec) == 0 or np.nanstd(model_vec) == 0:
        return np.nan
    return float(stats.spearmanr(neural_vec, model_vec, nan_policy="omit").correlation)


def rank_zscore(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(values, method="average")
    sd = np.std(ranks, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(ranks, dtype=float)
    return (ranks - np.mean(ranks)) / sd


def rsa_multiple_regression(neural_rdm: np.ndarray, model_rdms: dict[str, np.ndarray]) -> dict[str, float]:
    y = rank_zscore(rdm_vector(neural_rdm))
    x_cols = [rank_zscore(rdm_vector(model_rdms[name])) for name in model_rdms]
    x = np.column_stack(x_cols)
    x = np.column_stack([np.ones(x.shape[0]), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((y - yhat) ** 2) / denom) if denom > 0 else np.nan
    return {**{f"beta_{name}": float(beta[i + 1]) for i, name in enumerate(model_rdms)}, "model_r2": r2}


def marginal_variance_fractions(
    x: np.ndarray,
    factor_labels: dict[str, np.ndarray | pd.Series | list[object]],
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=0, keepdims=True)
    total_ss = float(np.sum(centered**2))
    if total_ss <= 0:
        return {name: np.nan for name in factor_labels}
    out: dict[str, float] = {}
    for name, labels_raw in factor_labels.items():
        labels = np.asarray(labels_raw)
        marginal = np.zeros_like(centered)
        for level in pd.unique(labels):
            mask = labels == level
            marginal[mask] = centered[mask].mean(axis=0, keepdims=True)
        out[name] = float(np.sum(marginal**2) / total_ss)
    return out


def syntax_axis_dprime(x: np.ndarray, condition: np.ndarray) -> float:
    active = x[condition == "active"]
    passive = x[condition == "passive"]
    axis = active.mean(axis=0) - passive.mean(axis=0)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return np.nan
    axis = axis / norm
    scores = x @ axis
    a = scores[condition == "active"]
    p = scores[condition == "passive"]
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(p) - 1) * np.var(p, ddof=1)) / max(len(a) + len(p) - 2, 1))
    return float((np.mean(a) - np.mean(p)) / pooled) if pooled > 0 else np.nan


def paired_condition_distance(x: np.ndarray, trial_meta: pd.DataFrame) -> float:
    distances: list[float] = []
    for _, sub in trial_meta.groupby("pair_id"):
        active_idx = sub.index[sub["condition_simple"].to_numpy() == "active"].to_numpy()
        passive_idx = sub.index[sub["condition_simple"].to_numpy() == "passive"].to_numpy()
        if len(active_idx) == 1 and len(passive_idx) == 1:
            distances.append(float(np.linalg.norm(x[active_idx[0]] - x[passive_idx[0]])))
    return float(np.median(distances)) if distances else np.nan


def leave_one_pair_out_syntax_projection(x: np.ndarray, trial_meta: pd.DataFrame) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    condition = trial_meta["condition_simple"].to_numpy()
    diffs: list[float] = []
    for _, sub in trial_meta.groupby("pair_id"):
        active_idx = sub.index[sub["condition_simple"].to_numpy() == "active"].to_numpy()
        passive_idx = sub.index[sub["condition_simple"].to_numpy() == "passive"].to_numpy()
        if len(active_idx) != 1 or len(passive_idx) != 1:
            continue
        heldout_idx = np.concatenate([active_idx, passive_idx])
        train = np.ones(x.shape[0], dtype=bool)
        train[heldout_idx] = False
        if np.sum(train & (condition == "active")) < 1 or np.sum(train & (condition == "passive")) < 1:
            continue
        axis = x[train & (condition == "active")].mean(axis=0) - x[train & (condition == "passive")].mean(axis=0)
        norm = float(np.linalg.norm(axis))
        if norm <= 0 or not np.isfinite(norm):
            continue
        projection = x @ (axis / norm)
        diffs.append(float(projection[active_idx[0]] - projection[passive_idx[0]]))
    diffs_arr = np.asarray(diffs, dtype=float)
    if len(diffs_arr) == 0:
        return {
            "cv_syntax_axis_mean_diff": np.nan,
            "cv_syntax_axis_median_diff": np.nan,
            "cv_syntax_axis_frac_pair_positive": np.nan,
            "cv_syntax_axis_pair_dprime": np.nan,
        }
    sd = float(np.std(diffs_arr, ddof=1)) if len(diffs_arr) > 1 else np.nan
    return {
        "cv_syntax_axis_mean_diff": float(np.mean(diffs_arr)),
        "cv_syntax_axis_median_diff": float(np.median(diffs_arr)),
        "cv_syntax_axis_frac_pair_positive": float(np.mean(diffs_arr > 0)),
        "cv_syntax_axis_pair_dprime": float(np.mean(diffs_arr) / sd) if sd and np.isfinite(sd) else np.nan,
    }


def analyze_run(
    subject: str,
    run: str,
    run_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, object]]:
    x_raw, trial_meta, unit_ids = make_run_matrix(run_df)
    x = standardize_units(x_raw)
    pca_n = min(3, x.shape[0] - 1, x.shape[1])
    if pca_n < 2:
        raise ValueError("Need at least two PCA components")
    pca = PCA(n_components=pca_n, random_state=args.random_state)
    scores = pca.fit_transform(x)

    trial_rows: list[dict[str, object]] = []
    for i, row in trial_meta.iterrows():
        trial_rows.append(
            {
                "subject": subject,
                "run": run,
                "trial_index": int(row["trial_index"]),
                "condition_simple": row["condition_simple"],
                "target_word": row["target_word"],
                "pair_id": int(row["pair_id"]),
                "sound_file": row["sound_file"],
                "pc1": float(scores[i, 0]),
                "pc2": float(scores[i, 1]),
                "pc3": float(scores[i, 2]) if pca_n >= 3 else np.nan,
            }
        )

    labels_condition = trial_meta["condition_simple"].to_numpy()
    labels_pair = trial_meta["pair_id"].astype(str).to_numpy()
    labels_word = trial_meta["target_word"].astype(str).to_numpy()
    marginal = marginal_variance_fractions(
        x,
        {
            "syntax": labels_condition,
            "sentence_pair": labels_pair,
            "word_identity": labels_word,
        },
    )
    neural = neural_correlation_rdm(x)
    model_rdms = {
        "syntax": categorical_model_rdm(labels_condition),
        "sentence_pair": categorical_model_rdm(labels_pair),
        "word_identity": categorical_model_rdm(labels_word),
    }
    rsa = {f"rho_{name}": rsa_spearman(neural, rdm) for name, rdm in model_rdms.items()}
    rsa.update(rsa_multiple_regression(neural, model_rdms))

    try:
        sil = float(silhouette_score(x, labels_condition, metric="euclidean"))
    except Exception:
        sil = np.nan

    summary = {
        "subject": subject,
        "run": run,
        "n_trials": int(x.shape[0]),
        "n_units_variable": int(x.shape[1]),
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
        "pc3_var": float(pca.explained_variance_ratio_[2]) if pca_n >= 3 else np.nan,
        "syntax_axis_dprime": syntax_axis_dprime(x, labels_condition),
        "syntax_silhouette": sil,
        "median_active_passive_pair_distance": paired_condition_distance(x, trial_meta),
        **leave_one_pair_out_syntax_projection(x, trial_meta),
    }
    return pd.DataFrame(trial_rows), summary, marginal, rsa


def signed_rank(values: np.ndarray, reference: float = 0.0, alternative: str = "greater") -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out = {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else np.nan,
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "wilcoxon_p": np.nan,
    }
    if len(values) >= 3 and np.any(np.abs(values - reference) > 1e-12):
        out["wilcoxon_p"] = float(stats.wilcoxon(values - reference, alternative=alternative).pvalue)
    return out


def make_state_space_plot(trial_scores: pd.DataFrame, out_dir: Path) -> str | None:
    if trial_scores.empty:
        return None
    runs = list(trial_scores["run"].drop_duplicates())
    n = len(runs)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.1 * n_rows), squeeze=False)
    colors = {"active": "#2f6fbb", "passive": "#c43c39"}
    for ax, run in zip(axes.ravel(), runs):
        sub = trial_scores[trial_scores["run"] == run].copy()
        for _, pair_df in sub.groupby("pair_id"):
            if pair_df.shape[0] == 2:
                ax.plot(pair_df["pc1"], pair_df["pc2"], color="0.75", lw=0.8, zorder=1)
        for condition, cond_df in sub.groupby("condition_simple"):
            ax.scatter(
                cond_df["pc1"],
                cond_df["pc2"],
                s=34,
                label=condition,
                color=colors.get(condition, "0.4"),
                edgecolor="white",
                lw=0.5,
                zorder=2,
            )
        for _, row in sub.iterrows():
            ax.text(row["pc1"], row["pc2"], str(int(row["pair_id"])), fontsize=6, ha="center", va="center", zorder=3)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_title(f"{sub['subject'].iloc[0]} {run.split('_')[-1]}", fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right")
    fig.suptitle("Function-word response state space", y=1.01)
    fig.tight_layout()
    path = out_dir / "fig_state_space_pca_by_run.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def make_run_dot_plot(
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    ylabel: str,
    title: str,
    path: Path,
) -> str | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(columns))
    for i, col in enumerate(columns):
        vals = df[col].to_numpy(float)
        jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0]
        ax.scatter(np.full(len(vals), x[i]) + jitter, vals, s=44, alpha=0.82, color="#2f6fbb")
        ax.plot([x[i] - 0.26, x[i] + 0.26], [np.nanmedian(vals)] * 2, color="black", lw=2.2)
    ax.axhline(0, color="0.45", ls="--", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_figures(
    trial_scores: pd.DataFrame,
    marginal_df: pd.DataFrame,
    rsa_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    state_path = make_state_space_plot(trial_scores, out_dir)
    if state_path:
        paths["state_space_pca_png"] = state_path
    marginal_path = make_run_dot_plot(
        marginal_df,
        ["syntax", "sentence_pair", "word_identity"],
        ["Syntax", "Sentence pair", "Word identity"],
        "Marginal variance fraction",
        "Demixed-style marginal variance by run",
        out_dir / "fig_demixed_marginal_variance.png",
    )
    if marginal_path:
        paths["demixed_marginal_variance_png"] = marginal_path
    rsa_path = make_run_dot_plot(
        rsa_df,
        ["rho_syntax", "rho_sentence_pair", "rho_word_identity"],
        ["Syntax", "Sentence pair", "Word identity"],
        "Spearman rho with neural RDM",
        "RSA model correlations by run",
        out_dir / "fig_rsa_model_correlations.png",
    )
    if rsa_path:
        paths["rsa_model_correlations_png"] = rsa_path
    beta_path = make_run_dot_plot(
        rsa_df,
        ["beta_syntax", "beta_sentence_pair", "beta_word_identity"],
        ["Syntax beta", "Pair beta", "Word beta"],
        "Rank-regression standardized beta",
        "Multiple-regression RSA betas by run",
        out_dir / "fig_rsa_multiple_betas.png",
    )
    if beta_path:
        paths["rsa_multiple_betas_png"] = beta_path
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
    state_summary: pd.DataFrame,
    marginal_df: pd.DataFrame,
    rsa_df: pd.DataFrame,
    aggregate: dict[str, object],
    figure_paths: dict[str, str],
) -> None:
    def agg_value(key: str, field: str = "median") -> str:
        value = aggregate.get(key, {})
        if not isinstance(value, dict) or field not in value:
            return "NA"
        number = value[field]
        return "NA" if not np.isfinite(number) else f"{number:.4g}"

    lines = [
        "# Active/passive state-space, demixed-style variance and RSA",
        "",
        "This analysis uses artifact-clean good-unit function-word responses from 0-0.8 s after function-word onset.",
        "Units are z-scored within each run before PCA, demixed-style marginal variance, and RSA.",
        "",
        "The marginal variance analysis is dPCA-inspired rather than a full dPCA model, because each sentence-pair by syntax condition has only one trial.",
        "The raw syntax-axis d-prime is descriptive because the axis is estimated and evaluated on the same trials; the leave-one-pair-out syntax projection is the more conservative generalization check.",
        "",
        "## Interpretation Notes",
        "",
        f"- Leave-one-pair-out syntax projection: median held-out mean active-passive projection difference = {agg_value('cv_syntax_axis_mean_diff')}; median fraction of sentence pairs with active > passive = {agg_value('cv_syntax_axis_frac_pair_positive')} (Wilcoxon p={agg_value('cv_syntax_axis_frac_pair_positive', 'wilcoxon_p')}).",
        f"- Demixed-style marginal variance is dominated by sentence-pair structure (median {agg_value('marginal_fraction_sentence_pair')}) and word identity (median {agg_value('marginal_fraction_word_identity')}), with syntax lower but consistently positive (median {agg_value('marginal_fraction_syntax')}).",
        f"- RSA model correlations are near zero for syntax (median rho={agg_value('rho_syntax')}) and do not show a strong global neural geometry organized by syntax, sentence pair, or word identity.",
        "- Taken together, the data support a weak-to-moderate syntax-sensitive projection axis, but not a broad categorical clustering of active versus passive trials in the full population state space.",
        "",
        "## State-Space Summary",
        "",
        markdown_table(state_summary),
        "",
        "## Demixed-Style Marginal Variance",
        "",
        markdown_table(marginal_df),
        "",
        "## RSA Summary",
        "",
        markdown_table(rsa_df),
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

    trial_score_frames: list[pd.DataFrame] = []
    state_rows: list[dict[str, object]] = []
    marginal_rows: list[dict[str, object]] = []
    rsa_rows: list[dict[str, object]] = []
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
                trial_scores, state_summary, marginal, rsa = analyze_run(subject, run, run_df, args)
            except Exception as exc:
                skipped_rows.append({**common, "reason": f"analysis failed: {exc}"})
                continue
            trial_score_frames.append(trial_scores.assign(analysis_mode=analysis_mode, unit_set=args.unit_set))
            state_rows.append({**common, **state_summary})
            marginal_rows.append({**common, **marginal})
            rsa_rows.append({**common, **rsa})
            print(f"analyzed {subject} {run}: trials={n_trials} units={state_summary['n_units_variable']}", flush=True)

    trial_scores_df = pd.concat(trial_score_frames, ignore_index=True) if trial_score_frames else pd.DataFrame()
    state_summary_df = pd.DataFrame(state_rows)
    marginal_df = pd.DataFrame(marginal_rows)
    rsa_df = pd.DataFrame(rsa_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    trial_scores_path = OUT_DIR / "active_passive_state_space_trial_scores.csv"
    state_summary_path = OUT_DIR / "active_passive_state_space_run_summary.csv"
    marginal_path = OUT_DIR / "active_passive_demixed_marginal_variance.csv"
    rsa_path = OUT_DIR / "active_passive_rsa_run_summary.csv"
    skipped_path = OUT_DIR / "active_passive_state_space_rsa_skipped_runs.csv"
    summary_path = OUT_DIR / "active_passive_state_space_rsa_summary.json"

    trial_scores_df.to_csv(trial_scores_path, index=False)
    state_summary_df.to_csv(state_summary_path, index=False)
    marginal_df.to_csv(marginal_path, index=False)
    rsa_df.to_csv(rsa_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    figure_paths = make_figures(trial_scores_df, marginal_df, rsa_df, OUT_DIR)
    aggregate: dict[str, object] = {
        "analysis_mode": analysis_mode,
        "unit_set": args.unit_set,
        "metric": METRIC,
        "n_runs_analyzed": int(state_summary_df[["subject", "run"]].drop_duplicates().shape[0]) if not state_summary_df.empty else 0,
        "n_runs_skipped": int(skipped_df.shape[0]),
        "outputs": {
            "trial_scores_csv": str(trial_scores_path),
            "state_summary_csv": str(state_summary_path),
            "demixed_marginal_variance_csv": str(marginal_path),
            "rsa_run_summary_csv": str(rsa_path),
            "skipped_runs_csv": str(skipped_path),
            "summary_json": str(summary_path),
            "report_md": str(REPORT_PATH),
            **figure_paths,
        },
    }
    for col in [
        "syntax_axis_dprime",
        "syntax_silhouette",
        "cv_syntax_axis_mean_diff",
        "cv_syntax_axis_median_diff",
        "cv_syntax_axis_pair_dprime",
        "pc1_var",
        "pc2_var",
    ]:
        if col in state_summary_df:
            aggregate[col] = signed_rank(state_summary_df[col].to_numpy(float), reference=0.0)
    if "cv_syntax_axis_frac_pair_positive" in state_summary_df:
        aggregate["cv_syntax_axis_frac_pair_positive"] = signed_rank(
            state_summary_df["cv_syntax_axis_frac_pair_positive"].to_numpy(float),
            reference=0.5,
        )
    for col in ["syntax", "sentence_pair", "word_identity"]:
        if col in marginal_df:
            aggregate[f"marginal_fraction_{col}"] = signed_rank(marginal_df[col].to_numpy(float), reference=0.0)
    for col in ["rho_syntax", "rho_sentence_pair", "rho_word_identity", "beta_syntax", "beta_sentence_pair", "beta_word_identity"]:
        if col in rsa_df:
            aggregate[col] = signed_rank(rsa_df[col].to_numpy(float), reference=0.0)

    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(state_summary_df, marginal_df, rsa_df, aggregate, figure_paths)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=["good_only", "all_clusters"], default="good_only")
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--min-trials-total", type=int, default=10)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=31)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
