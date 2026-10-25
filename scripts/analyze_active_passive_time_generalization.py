#!/usr/bin/env python3
"""Time-resolved and cross-temporal active/passive population decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025/data"
OUT_DIR = DATA_ROOT / "combined_active_passive_time_generalization"
REPORT_PATH = ROOT / "reports/active_passive_time_generalization.md"


@dataclass(frozen=True)
class RunResult:
    subject: str
    run: str
    n_trials: int
    n_active: int
    n_passive: int
    n_units_input: int
    n_units_variable: int
    auc_matrix: np.ndarray
    balanced_accuracy_diag: np.ndarray


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


def load_event_metadata(er_dir: Path) -> pd.DataFrame:
    subject = subject_from_er_dir(er_dir)
    path = er_dir / "active_passive_event_related_trial_metadata.csv"
    events = pd.read_csv(path)
    events["subject"] = subject
    events["condition_simple"] = events["condition"].map(simple_condition)
    events = events[events["condition_simple"].isin(["active", "passive"])].copy()
    return events


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


def apply_artifact_filter(
    events: pd.DataFrame,
    artifact: pd.DataFrame | None,
    threshold: float | None,
) -> pd.DataFrame:
    if threshold is None or artifact is None:
        out = events.copy()
        out["artifact_fraction_minus1_to_plus2s"] = np.nan
        return out
    out = events.merge(artifact, on=["run", "trial_index"], how="left")
    keep = out["artifact_fraction_minus1_to_plus2s"].isna() | (
        out["artifact_fraction_minus1_to_plus2s"] <= threshold
    )
    return out.loc[keep].copy()


def npz_name_for_unit_set(unit_set: str) -> str:
    names = {
        "good_only": "firing_rate_1000Hz_good_clusters.npz",
        "strict_qc": "firing_rate_1000Hz_strict_qc_clusters.npz",
        "usable_lenient": "firing_rate_1000Hz_usable_lenient_clusters.npz",
        "usable_very_lenient": "firing_rate_1000Hz_usable_very_lenient_clusters.npz",
        "all_clusters": "firing_rate_1000Hz_all_clusters.npz",
    }
    return names[unit_set]


def load_run_firing_rate(run: str, subject: str, unit_set: str) -> tuple[np.ndarray, np.ndarray]:
    path = (
        DATA_ROOT
        / subject
        / "processed/ActPasSynSentence"
        / run
        / "firing_rate_1000Hz"
        / npz_name_for_unit_set(unit_set)
    )
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return data["firing_rate_hz"].astype(np.float32, copy=False), data["unit_ids"]


def make_time_centers(t_min: float, t_max: float, step: float) -> np.ndarray:
    n_steps = int(np.floor((t_max - t_min) / step + 0.5))
    return np.round(t_min + np.arange(n_steps + 1) * step, 10)


def extract_trial_time_tensor(
    firing_rate: np.ndarray,
    events: pd.DataFrame,
    centers_sec: np.ndarray,
    window_sec: float,
    baseline_start_sec: float,
    baseline_stop_sec: float,
    baseline_correct: bool,
) -> np.ndarray:
    n_units, n_time = firing_rate.shape
    n_trials = len(events)
    out = np.full((len(centers_sec), n_trials, n_units), np.nan, dtype=np.float32)
    half_window = int(round(window_sec * 1000 / 2))
    baseline_start = int(round(baseline_start_sec * 1000))
    baseline_stop = int(round(baseline_stop_sec * 1000))

    for trial_i, event in enumerate(events.itertuples(index=False)):
        event_bin = int(event.event_bin_1000Hz)
        baseline = np.zeros(n_units, dtype=np.float32)
        if baseline_correct:
            b0 = max(event_bin + baseline_start, 0)
            b1 = min(event_bin + baseline_stop, n_time)
            if b1 <= b0:
                baseline[:] = np.nan
            else:
                baseline = firing_rate[:, b0:b1].mean(axis=1, dtype=np.float32)
        for time_i, center in enumerate(centers_sec):
            center_offset = int(round(center * 1000))
            w0 = max(event_bin + center_offset - half_window, 0)
            w1 = min(event_bin + center_offset + half_window, n_time)
            if w1 <= w0:
                continue
            values = firing_rate[:, w0:w1].mean(axis=1, dtype=np.float32)
            out[time_i, trial_i, :] = values - baseline
    return out


def variable_unit_mask(x_by_time: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x_by_time).all(axis=(0, 1))
    variable = np.nanstd(x_by_time, axis=(0, 1)) > 0
    return finite & variable


def training_t_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    active = x[y == 1]
    passive = x[y == 0]
    active_var = active.var(axis=0, ddof=1) if active.shape[0] > 1 else np.zeros(x.shape[1])
    passive_var = passive.var(axis=0, ddof=1) if passive.shape[0] > 1 else np.zeros(x.shape[1])
    denom = np.sqrt(active_var / max(active.shape[0], 1) + passive_var / max(passive.shape[0], 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs((active.mean(axis=0) - passive.mean(axis=0)) / denom)
    finite = np.isfinite(x).all(axis=0)
    variable = np.nanstd(x, axis=0) > 0
    scores[~(finite & variable)] = -np.inf
    return np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


def select_top_units(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    scores = training_t_scores(x, y)
    valid = np.where(np.isfinite(scores))[0]
    if len(valid) == 0:
        return np.array([], dtype=int)
    order = valid[np.argsort(scores[valid])[::-1]]
    return order[: min(k, len(order))]


def fit_predict_probs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test_by_time: np.ndarray,
    top_k: int,
    c_value: float,
) -> np.ndarray:
    cols = select_top_units(x_train, y_train, top_k)
    if len(cols) == 0:
        return np.full(x_test_by_time.shape[:2], float(np.mean(y_train)), dtype=float)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
        ),
    )
    model.fit(x_train[:, cols], y_train)
    probs = np.empty(x_test_by_time.shape[:2], dtype=float)
    for test_time_i in range(x_test_by_time.shape[0]):
        probs[test_time_i] = model.predict_proba(x_test_by_time[test_time_i, :, :][:, cols])[:, 1]
    return probs


def cross_temporal_decode(
    x_by_time: np.ndarray,
    y: np.ndarray,
    top_k: int,
    c_value: float,
    cv_repeats: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_times, n_trials, _ = x_by_time.shape
    min_class = int(np.bincount(y, minlength=2).min())
    if min_class < 2:
        raise ValueError("Need at least two trials per class")
    n_splits = min(5, min_class)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=cv_repeats,
        random_state=random_state,
    )
    prob_sum = np.zeros((n_times, n_times, n_trials), dtype=float)
    prob_count = np.zeros((n_times, n_times, n_trials), dtype=float)

    for train_idx, test_idx in cv.split(np.zeros((n_trials, 1)), y):
        x_test_by_time = x_by_time[:, test_idx, :]
        for train_time_i in range(n_times):
            probs = fit_predict_probs(
                x_by_time[train_time_i, train_idx, :],
                y[train_idx],
                x_test_by_time,
                top_k=top_k,
                c_value=c_value,
            )
            prob_sum[train_time_i][:, test_idx] += probs
            prob_count[train_time_i][:, test_idx] += 1

    probabilities = prob_sum / np.maximum(prob_count, 1)
    auc = np.full((n_times, n_times), np.nan, dtype=float)
    balanced_acc_diag = np.full(n_times, np.nan, dtype=float)
    for train_time_i in range(n_times):
        for test_time_i in range(n_times):
            probs = probabilities[train_time_i, test_time_i, :]
            auc[train_time_i, test_time_i] = roc_auc_score(y, probs)
            if train_time_i == test_time_i:
                pred = (probs >= 0.5).astype(int)
                balanced_acc_diag[test_time_i] = balanced_accuracy_score(y, pred)
    return auc, balanced_acc_diag


def summarize_run(
    result: RunResult,
    centers_sec: np.ndarray,
    primary_start: float,
    primary_stop: float,
) -> dict[str, object]:
    diag = np.diag(result.auc_matrix)
    post_mask = (centers_sec >= primary_start) & (centers_sec <= primary_stop)
    peak_idx = int(np.nanargmax(np.where(post_mask, diag, np.nan)))
    return {
        "subject": result.subject,
        "run": result.run,
        "n_trials": result.n_trials,
        "n_active": result.n_active,
        "n_passive": result.n_passive,
        "n_units_input": result.n_units_input,
        "n_units_variable": result.n_units_variable,
        "primary_window_start_sec": primary_start,
        "primary_window_stop_sec": primary_stop,
        "mean_diag_auc_primary_window": float(np.nanmean(diag[post_mask])),
        "max_diag_auc_primary_window": float(diag[peak_idx]),
        "peak_diag_time_sec": float(centers_sec[peak_idx]),
        "mean_cross_temporal_auc_primary_window": float(np.nanmean(result.auc_matrix[np.ix_(post_mask, post_mask)])),
    }


def signed_rank(values: np.ndarray, reference: float = 0.5) -> dict[str, float]:
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


def diagonal_time_stats(diag_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_sec, sub in diag_df.groupby("time_sec"):
        vals = sub["auc"].to_numpy(float)
        p_value = np.nan
        if len(vals) >= 3 and np.any(np.abs(vals - 0.5) > 1e-12):
            p_value = float(stats.wilcoxon(vals - 0.5, alternative="greater").pvalue)
        rows.append(
            {
                "time_sec": float(time_sec),
                "n_runs": int(len(vals)),
                "median_auc": float(np.median(vals)),
                "mean_auc": float(np.mean(vals)),
                "wilcoxon_p_greater": p_value,
            }
        )
    out = pd.DataFrame(rows).sort_values("time_sec")
    p = out["wilcoxon_p_greater"].to_numpy(float)
    valid = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[valid])]
    prev = 1.0
    m = len(order)
    for reverse_rank, i in enumerate(order[::-1], start=1):
        rank = m - reverse_rank + 1
        adjusted = p[i] * m / rank
        prev = min(prev, adjusted)
        q[i] = min(prev, 1.0)
    out["q_bh"] = q
    return out


def build_long_tables(
    results: list[RunResult],
    centers_sec: np.ndarray,
    analysis_mode: str,
    unit_set: str,
    top_k: int,
    c_value: float,
    cv_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for result in results:
        common = {
            "analysis_mode": analysis_mode,
            "unit_set": unit_set,
            "model": f"top{top_k}_l2",
            "logistic_c": c_value,
            "cv_repeats": cv_repeats,
            "subject": result.subject,
            "run": result.run,
            "n_trials": result.n_trials,
            "n_active": result.n_active,
            "n_passive": result.n_passive,
            "n_units_input": result.n_units_input,
            "n_units_variable": result.n_units_variable,
        }
        for train_i, train_time in enumerate(centers_sec):
            for test_i, test_time in enumerate(centers_sec):
                long_rows.append(
                    {
                        **common,
                        "train_time_sec": float(train_time),
                        "test_time_sec": float(test_time),
                        "auc": float(result.auc_matrix[train_i, test_i]),
                    }
                )
        diag = np.diag(result.auc_matrix)
        for time_i, time_sec in enumerate(centers_sec):
            diag_rows.append(
                {
                    **common,
                    "time_sec": float(time_sec),
                    "auc": float(diag[time_i]),
                    "balanced_accuracy": float(result.balanced_accuracy_diag[time_i]),
                }
            )
        summary_rows.append(
            {
                **common,
                **summarize_run(result, centers_sec, primary_start=0.0, primary_stop=0.8),
            }
        )
    return pd.DataFrame(long_rows), pd.DataFrame(diag_rows), pd.DataFrame(summary_rows)


def make_figures(results: list[RunResult], centers_sec: np.ndarray, out_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    if not results:
        return paths

    matrices = np.stack([result.auc_matrix for result in results], axis=0)
    diag = np.stack([np.diag(result.auc_matrix) for result in results], axis=0)
    median_matrix = np.nanmedian(matrices, axis=0)
    median_diag = np.nanmedian(diag, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for result in results:
        ax.plot(centers_sec, np.diag(result.auc_matrix), lw=1.1, alpha=0.45, label=result.run)
    ax.plot(centers_sec, median_diag, color="black", lw=2.4, label="median")
    ax.axhline(0.5, color="0.45", ls="--", lw=1)
    ax.axvline(0, color="0.2", ls=":", lw=1)
    ax.axvspan(0, 0.8, color="0.9", zorder=-2)
    ax.set_xlabel("Time from function-word onset (s)")
    ax.set_ylabel("Cross-validated AUC")
    ax.set_title("Time-resolved active/passive decoding")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    path = out_dir / "fig_time_resolved_decoding_diagonal.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths["time_resolved_diagonal_png"] = str(path)

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(
        median_matrix,
        origin="lower",
        aspect="auto",
        vmin=0.35,
        vmax=0.75,
        cmap="viridis",
        extent=[centers_sec[0], centers_sec[-1], centers_sec[0], centers_sec[-1]],
    )
    ax.plot([centers_sec[0], centers_sec[-1]], [centers_sec[0], centers_sec[-1]], color="white", lw=1, alpha=0.75)
    ax.axvline(0, color="white", ls=":", lw=1)
    ax.axhline(0, color="white", ls=":", lw=1)
    ax.set_xlabel("Test time from function-word onset (s)")
    ax.set_ylabel("Train time from function-word onset (s)")
    ax.set_title("Cross-temporal active/passive decoding, median across runs")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("AUC")
    fig.tight_layout()
    path = out_dir / "fig_cross_temporal_decoding_median_matrix.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    paths["cross_temporal_median_matrix_png"] = str(path)

    n = len(results)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.8 * n_rows), squeeze=False)
    for ax, result in zip(axes.ravel(), results):
        ax.imshow(
            result.auc_matrix,
            origin="lower",
            aspect="auto",
            vmin=0.35,
            vmax=0.75,
            cmap="viridis",
            extent=[centers_sec[0], centers_sec[-1], centers_sec[0], centers_sec[-1]],
        )
        ax.plot([centers_sec[0], centers_sec[-1]], [centers_sec[0], centers_sec[-1]], color="white", lw=0.8, alpha=0.7)
        ax.axvline(0, color="white", ls=":", lw=0.8)
        ax.axhline(0, color="white", ls=":", lw=0.8)
        ax.set_title(f"{result.subject} {result.run.split('_')[-1]}", fontsize=8)
        ax.set_xlabel("test (s)", fontsize=7)
        ax.set_ylabel("train (s)", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Cross-temporal decoding by run", y=1.01)
    fig.tight_layout()
    path = out_dir / "fig_cross_temporal_decoding_by_run.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths["cross_temporal_by_run_png"] = str(path)
    return paths


def markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
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
    result_summaries: pd.DataFrame,
    aggregate: dict[str, object],
    figure_paths: dict[str, str],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Active/passive time-resolved and cross-temporal decoding",
        "",
        "This analysis trains a fixed top-5-unit L2 logistic decoder at each function-word-aligned time bin and tests it at every other time bin.",
        "The primary run set is artifact-clean trials with good units only.",
        "",
        "## Run summary",
        "",
        markdown_table(result_summaries),
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
    lines.extend(
        [
            "",
            "Notes:",
            "- The diagonal is ordinary time-resolved decoding.",
            "- Off-diagonal values test temporal generalization: a stable population code should generalize away from the diagonal.",
            "- Feature selection is nested within each training fold and train-time bin.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    centers_sec = make_time_centers(args.t_min, args.t_max, args.step_sec)

    analysis_mode = "all_trials" if args.artifact_threshold is None else (
        f"artifact_clean_le_{str(args.artifact_threshold).replace('.', 'p')}"
    )
    results: list[RunResult] = []
    skipped_rows: list[dict[str, object]] = []

    for er_dir in find_event_dirs():
        subject = subject_from_er_dir(er_dir)
        events_all = load_event_metadata(er_dir)
        artifact = load_artifact_table(er_dir)
        events_filtered = apply_artifact_filter(events_all, artifact, args.artifact_threshold)
        for run in sorted(events_all["run"].unique()):
            run_events = events_filtered[events_filtered["run"] == run].sort_values("trial_index").copy()
            y = (run_events["condition_simple"].to_numpy() == "active").astype(int)
            common = {
                "analysis_mode": analysis_mode,
                "unit_set": args.unit_set,
                "subject": subject,
                "run": run,
                "n_trials": int(len(y)),
                "n_active": int(np.sum(y == 1)),
                "n_passive": int(np.sum(y == 0)),
            }
            if len(y) < args.min_trials_total or min(common["n_active"], common["n_passive"]) < args.min_trials_per_class:
                skipped_rows.append({**common, "reason": "insufficient artifact-clean active/passive trials"})
                continue
            try:
                firing_rate, unit_ids = load_run_firing_rate(run, subject, args.unit_set)
            except FileNotFoundError as exc:
                skipped_rows.append({**common, "reason": f"missing firing-rate npz: {exc}"})
                continue

            x_by_time = extract_trial_time_tensor(
                firing_rate,
                run_events,
                centers_sec,
                window_sec=args.window_sec,
                baseline_start_sec=args.baseline_start_sec,
                baseline_stop_sec=args.baseline_stop_sec,
                baseline_correct=not args.no_baseline_correct,
            )
            keep_units = variable_unit_mask(x_by_time)
            x_by_time = x_by_time[:, :, keep_units]
            if x_by_time.shape[2] < 1:
                skipped_rows.append({**common, "n_units_input": int(len(unit_ids)), "reason": "no variable units"})
                continue

            auc_matrix, balanced_acc_diag = cross_temporal_decode(
                x_by_time,
                y,
                top_k=args.top_k,
                c_value=args.logistic_c,
                cv_repeats=args.cv_repeats,
                random_state=args.random_state + stable_seed(subject, run, args.unit_set) % 100000,
            )
            results.append(
                RunResult(
                    subject=subject,
                    run=run,
                    n_trials=int(len(y)),
                    n_active=int(np.sum(y == 1)),
                    n_passive=int(np.sum(y == 0)),
                    n_units_input=int(len(unit_ids)),
                    n_units_variable=int(x_by_time.shape[2]),
                    auc_matrix=auc_matrix,
                    balanced_accuracy_diag=balanced_acc_diag,
                )
            )
            print(f"analyzed {subject} {run}: trials={len(y)} units={x_by_time.shape[2]}", flush=True)

    long_df, diag_df, summary_df = build_long_tables(
        results,
        centers_sec,
        analysis_mode=analysis_mode,
        unit_set=args.unit_set,
        top_k=args.top_k,
        c_value=args.logistic_c,
        cv_repeats=args.cv_repeats,
    )
    skipped_df = pd.DataFrame(skipped_rows)

    long_path = OUT_DIR / "active_passive_time_generalization_long.csv"
    diag_path = OUT_DIR / "active_passive_time_resolved_diagonal.csv"
    diag_stats_path = OUT_DIR / "active_passive_time_resolved_diagonal_time_stats.csv"
    summary_path = OUT_DIR / "active_passive_time_generalization_run_summary.csv"
    skipped_path = OUT_DIR / "active_passive_time_generalization_skipped_runs.csv"
    npz_path = OUT_DIR / "active_passive_time_generalization_matrices.npz"
    json_path = OUT_DIR / "active_passive_time_generalization_summary.json"

    long_df.to_csv(long_path, index=False)
    diag_df.to_csv(diag_path, index=False)
    diag_stats = diagonal_time_stats(diag_df) if not diag_df.empty else pd.DataFrame()
    diag_stats.to_csv(diag_stats_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    if results:
        np.savez_compressed(
            npz_path,
            centers_sec=centers_sec,
            auc_matrices=np.stack([result.auc_matrix for result in results], axis=0),
            diag_auc=np.stack([np.diag(result.auc_matrix) for result in results], axis=0),
            subjects=np.array([result.subject for result in results], dtype=object),
            runs=np.array([result.run for result in results], dtype=object),
            n_trials=np.array([result.n_trials for result in results], dtype=int),
            n_units_variable=np.array([result.n_units_variable for result in results], dtype=int),
        )

    figure_paths = make_figures(results, centers_sec, OUT_DIR)
    aggregate: dict[str, object] = {
        "analysis_mode": analysis_mode,
        "unit_set": args.unit_set,
        "model": f"top{args.top_k}_l2",
        "logistic_c": args.logistic_c,
        "cv_repeats": args.cv_repeats,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "time_range_sec": [args.t_min, args.t_max],
        "baseline_corrected": not args.no_baseline_correct,
        "baseline_sec": [args.baseline_start_sec, args.baseline_stop_sec],
        "n_runs_analyzed": len(results),
        "n_runs_skipped": int(len(skipped_df)),
        "outputs": {
            "long_csv": str(long_path),
            "diagonal_csv": str(diag_path),
            "diagonal_time_stats_csv": str(diag_stats_path),
            "run_summary_csv": str(summary_path),
            "skipped_runs_csv": str(skipped_path),
            "matrices_npz": str(npz_path) if results else None,
            "summary_json": str(json_path),
            "report_md": str(REPORT_PATH),
            **figure_paths,
        },
    }
    if results:
        primary_mean = summary_df["mean_diag_auc_primary_window"].to_numpy(float)
        primary_peak = summary_df["max_diag_auc_primary_window"].to_numpy(float)
        primary_cross = summary_df["mean_cross_temporal_auc_primary_window"].to_numpy(float)
        aggregate["primary_window_summary"] = {
            "mean_diag_auc_0_to_0p8": signed_rank(primary_mean, reference=0.5),
            "max_diag_auc_0_to_0p8": signed_rank(primary_peak, reference=0.5),
            "mean_cross_temporal_auc_0_to_0p8": signed_rank(primary_cross, reference=0.5),
        }
        diag_by_time = diag_df.groupby("time_sec")["auc"].median().reset_index()
        peak_row = diag_by_time.loc[diag_by_time["auc"].idxmax()]
        aggregate["median_diagonal_peak"] = {
            "time_sec": float(peak_row["time_sec"]),
            "median_auc": float(peak_row["auc"]),
        }
        if not diag_stats.empty:
            best_time = diag_stats.sort_values("wilcoxon_p_greater").iloc[0]
            aggregate["best_uncorrected_diagonal_time"] = {
                "time_sec": float(best_time["time_sec"]),
                "median_auc": float(best_time["median_auc"]),
                "wilcoxon_p_greater": float(best_time["wilcoxon_p_greater"]),
                "q_bh": float(best_time["q_bh"]),
            }

    json_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(summary_df, aggregate, figure_paths)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-set",
        choices=["good_only", "strict_qc", "usable_lenient", "usable_very_lenient", "all_clusters"],
        default="good_only",
    )
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--t-min", type=float, default=-0.75)
    parser.add_argument("--t-max", type=float, default=1.75)
    parser.add_argument("--step-sec", type=float, default=0.05)
    parser.add_argument("--window-sec", type=float, default=0.2)
    parser.add_argument("--baseline-start-sec", type=float, default=-1.0)
    parser.add_argument("--baseline-stop-sec", type=float, default=0.0)
    parser.add_argument("--no-baseline-correct", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--min-trials-total", type=int, default=10)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
