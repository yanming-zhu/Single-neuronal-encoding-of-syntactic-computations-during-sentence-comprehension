#!/usr/bin/env python3
"""Nested population-size scaling for active/passive Neuropixels decoding.

The primary curve ranks units only within each cross-validation training fold.
The comparison curve samples the same number of units uniformly at random within
the same fold. Runs are summarized within patients before cross-patient summaries.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_active_passive_network_encoding import (
    apply_artifact_filter,
    find_er_dirs,
    load_artifact_table,
    load_metrics,
    make_population_matrix,
    stable_seed,
    subject_from_er_dir,
    training_t_scores,
)


OUT = ROOT / "neuropixel2025" / "data" / "combined_active_passive_population_size_scaling"
REPORT = ROOT / "reports" / "active_passive_population_size_scaling_5patients.md"

K_SPECS: tuple[tuple[str, int | None], ...] = (
    ("1", 1),
    ("2", 2),
    ("5", 5),
    ("10", 10),
    ("25", 25),
    ("all", None),
)

PATIENT_COLORS = {
    "gz001": "#0072B2",
    "gz002": "#D55E00",
    "gz003": "#009E73",
    "gz004": "#CC79A7",
    "gz005": "#E69F00",
}


def make_cv_splits(y: np.ndarray, n_repeats: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    min_class = int(np.bincount(y, minlength=2).min())
    if min_class < 2:
        raise ValueError("Need at least two trials in each class")
    n_splits = min(5, min_class)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    dummy = np.zeros((len(y), 1), dtype=float)
    return [(train.copy(), test.copy()) for train, test in cv.split(dummy, y)]


def fit_fold_probabilities(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    columns: np.ndarray,
    c_value: float,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
        ),
    )
    model.fit(x_train[:, columns], y_train)
    return model.predict_proba(x_test[:, columns])[:, 1]


def aggregate_cv_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    requested_k: int | None,
    c_value: float,
    selection: str,
    random_state: int,
) -> tuple[np.ndarray, float]:
    prob_sum = np.zeros(len(y), dtype=float)
    prob_count = np.zeros(len(y), dtype=float)
    effective_ks: list[int] = []
    rng = np.random.default_rng(random_state)

    for train_idx, test_idx in splits:
        effective_k = x.shape[1] if requested_k is None else min(requested_k, x.shape[1])
        if selection == "ranked":
            if effective_k == x.shape[1]:
                columns = np.arange(x.shape[1], dtype=int)
            else:
                scores = training_t_scores(x[train_idx], y[train_idx])
                columns = np.argsort(scores)[::-1][:effective_k]
        elif selection == "random":
            if effective_k == x.shape[1]:
                columns = np.arange(x.shape[1], dtype=int)
            else:
                columns = np.sort(rng.choice(x.shape[1], size=effective_k, replace=False))
        else:
            raise ValueError(f"Unknown selection rule: {selection}")

        probabilities = fit_fold_probabilities(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            columns,
            c_value,
        )
        prob_sum[test_idx] += probabilities
        prob_count[test_idx] += 1
        effective_ks.append(int(effective_k))

    if np.any(prob_count == 0):
        raise RuntimeError("At least one trial was never held out")
    return prob_sum / prob_count, float(np.median(effective_ks))


def score_predictions(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, (probabilities >= 0.5).astype(int))
        ),
    }


def exact_sign_flip_p(values: np.ndarray, alternative: str = "greater") -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = float(np.mean(values))
    null = np.asarray(
        [np.mean(values * np.asarray(signs, dtype=float)) for signs in itertools.product((-1, 1), repeat=len(values))],
        dtype=float,
    )
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-12))
    if alternative == "two-sided":
        return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    raise ValueError(f"Unknown alternative: {alternative}")


def aggregate_patient_values(
    ranked_runs: pd.DataFrame,
    random_draws: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked_patients = (
        ranked_runs.groupby(["subject", "k_order", "k_label"], as_index=False)["auc"]
        .median()
        .rename(columns={"auc": "ranked_auc"})
    )
    random_patient_draws = (
        random_draws.groupby(["subject", "k_order", "k_label", "draw"], as_index=False)["auc"]
        .median()
        .rename(columns={"auc": "random_auc"})
    )
    random_patients = (
        random_patient_draws.groupby(["subject", "k_order", "k_label"], as_index=False)["random_auc"]
        .agg(
            random_median_auc="median",
            random_q025_auc=lambda values: float(np.quantile(values, 0.025)),
            random_q975_auc=lambda values: float(np.quantile(values, 0.975)),
        )
    )
    patient_values = ranked_patients.merge(
        random_patients,
        on=["subject", "k_order", "k_label"],
        validate="one_to_one",
    )
    patient_values["ranked_minus_random_auc"] = (
        patient_values["ranked_auc"] - patient_values["random_median_auc"]
    )

    rows: list[dict[str, object]] = []
    for (k_order, k_label), frame in patient_values.groupby(["k_order", "k_label"], sort=True):
        ranked = frame["ranked_auc"].to_numpy(float)
        random = frame["random_median_auc"].to_numpy(float)
        difference = ranked - random
        rows.append(
            {
                "k_order": int(k_order),
                "k_label": str(k_label),
                "n_patients": int(frame.subject.nunique()),
                "ranked_patient_median_auc": float(np.median(ranked)),
                "ranked_patient_q25_auc": float(np.quantile(ranked, 0.25)),
                "ranked_patient_q75_auc": float(np.quantile(ranked, 0.75)),
                "random_patient_median_auc": float(np.median(random)),
                "random_patient_q25_auc": float(np.quantile(random, 0.25)),
                "random_patient_q75_auc": float(np.quantile(random, 0.75)),
                "ranked_minus_random_patient_median_auc": float(np.median(difference)),
                "ranked_minus_random_exact_sign_flip_p_greater": exact_sign_flip_p(
                    difference, alternative="greater"
                ),
            }
        )
    return patient_values, random_patient_draws, pd.DataFrame(rows).sort_values("k_order")


def draw_scaling_figure(patient_values: pd.DataFrame, summary: pd.DataFrame) -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(90 / 25.4, 58 / 25.4), facecolor="white")
    for subject, frame in patient_values.groupby("subject", sort=True):
        frame = frame.sort_values("k_order")
        ax.plot(
            frame.k_order,
            frame.ranked_auc,
            color=PATIENT_COLORS[str(subject)],
            marker="o",
            ms=3.2,
            lw=0.8,
            alpha=0.72,
        )
    x = summary.k_order.to_numpy(float)
    ranked = summary.ranked_patient_median_auc.to_numpy(float)
    random = summary.random_patient_median_auc.to_numpy(float)
    ax.fill_between(
        x,
        summary.random_patient_q25_auc,
        summary.random_patient_q75_auc,
        color="#BFC5CB",
        alpha=0.34,
        lw=0,
    )
    ax.plot(x, random, color="#777D82", marker="o", ms=4.1, lw=1.35, label="random K")
    ax.plot(x, ranked, color="#357ABD", marker="o", ms=4.5, lw=1.7, label="train-ranked K")
    ax.axhline(0.5, color="#6E7175", ls="--", lw=0.8)
    ax.set_xticks(x, summary.k_label)
    ax.set_xlabel("number of units retained")
    ax.set_ylabel("cross-validated AUC")
    ax.set_ylim(0.25, 0.86)
    ax.set_title("Population-size dependence", loc="left", fontweight="bold", pad=4)
    ax.legend(loc="lower right", handlelength=1.2, handletextpad=0.4)
    fig.tight_layout(pad=0.45)

    stem = OUT / "fig_population_size_scaling_90mm"
    outputs = {
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
        "tiff": str(stem.with_suffix(".tiff")),
    }
    fig.savefig(outputs["png"], dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(outputs["pdf"], facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(outputs["svg"], facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(
        outputs["tiff"],
        dpi=600,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return outputs


def write_report(summary: pd.DataFrame, metadata: dict[str, object]) -> None:
    lines = [
        "# Active/passive population-size scaling across five patients",
        "",
        "## Analysis contract",
        "",
        "- Primary data: artifact fraction <=0.2 and Kilosort `good` units.",
        "- Response: baseline-corrected firing rate from 0-0.8 s after function-word onset.",
        "- Decoder: standardized L2 logistic regression with fixed C=0.1.",
        "- Cross-validation: repeated stratified fivefold CV; unit ranking is recomputed only within each training fold.",
        "- Random-K baseline: the same number of units is sampled uniformly within the same training fold and evaluated on the untouched test fold.",
        "- Replication level: runs are aggregated within patient; n=5 patients are shown and summarized.",
        "- Status: exploratory scaling analysis; pointwise P values are not used to select K.",
        "",
        "## Patient-level summary",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation boundary",
        "",
        "The curve measures how active/passive construction information changes with the number of retained units. "
        "Because construction and function-word identity/acoustics covary in the current materials, it is not an estimate of abstract voice coding.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, ensure_ascii=False),
        "```",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for er_dir in find_er_dirs():
        subject = subject_from_er_dir(er_dir)
        metrics = load_metrics(er_dir)
        artifact = load_artifact_table(er_dir)
        filtered = apply_artifact_filter(metrics, artifact, args.artifact_threshold)
        for run in sorted(metrics.run.unique()):
            x, y, _, n_units_input, n_units_variable = make_population_matrix(
                filtered, run, "good_only"
            )
            n_active = int(np.sum(y == 1))
            n_passive = int(np.sum(y == 0))
            common = {
                "subject": subject,
                "run": run,
                "n_trials": int(len(y)),
                "n_active": n_active,
                "n_passive": n_passive,
                "n_units_input": int(n_units_input),
                "n_units_variable": int(n_units_variable),
            }
            if (
                len(y) < args.min_trials_total
                or min(n_active, n_passive) < args.min_trials_per_class
                or n_units_variable < 1
            ):
                skipped_rows.append({**common, "reason": "insufficient trials or units"})
                continue

            # Reuse the exact CV seed used by the designated fixed top-5 analysis so
            # K=5 is directly comparable with the existing primary endpoint.
            split_seed = args.random_state
            splits = make_cv_splits(y, args.cv_repeats, split_seed)
            for k_order, (k_label, requested_k) in enumerate(K_SPECS):
                ranked_probs, effective_k = aggregate_cv_probabilities(
                    x,
                    y,
                    splits,
                    requested_k,
                    args.c_value,
                    selection="ranked",
                    random_state=split_seed,
                )
                ranked_score = score_predictions(y, ranked_probs)
                run_rows.append(
                    {
                        **common,
                        "k_order": k_order,
                        "k_label": k_label,
                        "requested_k": requested_k if requested_k is not None else "all",
                        "median_effective_k": effective_k,
                        **ranked_score,
                    }
                )

                n_draws = 1 if requested_k is None or requested_k >= n_units_variable else args.random_draws
                for draw in range(n_draws):
                    random_seed = (
                        args.random_state
                        + stable_seed(subject, run, k_label, "random", draw) % 1000000
                    )
                    random_probs, random_effective_k = aggregate_cv_probabilities(
                        x,
                        y,
                        splits,
                        requested_k,
                        args.c_value,
                        selection="random",
                        random_state=random_seed,
                    )
                    random_score = score_predictions(y, random_probs)
                    random_rows.append(
                        {
                            **common,
                            "k_order": k_order,
                            "k_label": k_label,
                            "requested_k": requested_k if requested_k is not None else "all",
                            "median_effective_k": random_effective_k,
                            "draw": draw,
                            **random_score,
                        }
                    )

    ranked_runs = pd.DataFrame(run_rows).sort_values(["subject", "run", "k_order"])
    random_draws = pd.DataFrame(random_rows).sort_values(["subject", "run", "k_order", "draw"])
    skipped = pd.DataFrame(skipped_rows)
    if ranked_runs.empty or random_draws.empty:
        raise RuntimeError("No runs passed the population-size scaling inclusion criteria")
    if ranked_runs.subject.nunique() != 5:
        raise RuntimeError(f"Expected five patients, found {ranked_runs.subject.nunique()}")

    patient_values, patient_random_draws, summary = aggregate_patient_values(
        ranked_runs, random_draws
    )

    ranked_runs_path = OUT / "population_size_scaling_run_values.csv"
    random_draws_path = OUT / "population_size_scaling_random_run_draws.csv"
    patient_values_path = OUT / "population_size_scaling_patient_values.csv"
    patient_random_draws_path = OUT / "population_size_scaling_random_patient_draws.csv"
    summary_path = OUT / "population_size_scaling_group_summary.csv"
    skipped_path = OUT / "population_size_scaling_skipped_runs.csv"
    ranked_runs.to_csv(ranked_runs_path, index=False)
    random_draws.to_csv(random_draws_path, index=False)
    patient_values.to_csv(patient_values_path, index=False)
    patient_random_draws.to_csv(patient_random_draws_path, index=False)
    summary.to_csv(summary_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    outputs = draw_scaling_figure(patient_values, summary)
    top5 = summary[summary.k_label == "5"].iloc[0]
    peak = summary.loc[summary.ranked_patient_median_auc.idxmax()]
    metadata: dict[str, object] = {
        "analysis": "active/passive population-size scaling",
        "status": "exploratory",
        "analysis_mode": "artifact_clean_le_0p2",
        "unit_set": "good_only",
        "metric": "response_baseline_corrected_hz",
        "decoder": "StandardScaler + L2 logistic regression",
        "c_value": args.c_value,
        "cv_repeats": args.cv_repeats,
        "random_draws": args.random_draws,
        "random_state": args.random_state,
        "n_patients": int(ranked_runs.subject.nunique()),
        "n_runs": int(ranked_runs[["subject", "run"]].drop_duplicates().shape[0]),
        "peak_k_label": str(peak.k_label),
        "peak_ranked_patient_median_auc": float(peak.ranked_patient_median_auc),
        "top5_ranked_patient_median_auc": float(top5.ranked_patient_median_auc),
        "top5_random_patient_median_auc": float(top5.random_patient_median_auc),
        "top5_ranked_minus_random_patient_median_auc": float(
            top5.ranked_minus_random_patient_median_auc
        ),
        "top5_exact_sign_flip_p_greater": float(
            top5.ranked_minus_random_exact_sign_flip_p_greater
        ),
        "source_data": {
            "ranked_run_values": str(ranked_runs_path),
            "random_run_draws": str(random_draws_path),
            "patient_values": str(patient_values_path),
            "random_patient_draws": str(patient_random_draws_path),
            "group_summary": str(summary_path),
            "skipped_runs": str(skipped_path),
        },
        "figure_outputs": outputs,
        "reviewer_risks": [
            "The analysis is exploratory and does not use pointwise tests to choose population size.",
            "Some runs have fewer than 25 variable good units, so requested K is capped at the available unit count.",
            "Active/passive construction and function-word identity/acoustics covary in the current design.",
            "Patients, not runs or resampling draws, are the independent biological units.",
        ],
    }
    metadata_path = OUT / "population_size_scaling_metadata.json"
    metadata["metadata_json"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(summary, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--random-draws", type=int, default=20)
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--min-trials-total", type=int, default=10)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
