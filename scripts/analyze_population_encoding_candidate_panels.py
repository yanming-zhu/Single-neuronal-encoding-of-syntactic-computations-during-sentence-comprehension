#!/usr/bin/env python3
"""Compare three candidate population-encoding panels for revised Fig. 2e.

The candidates are (1) nested top-unit ablation, (2) a condition-preserving
trial-shuffled pseudo-population control, and (3) a cross-validated standardized
centroid distance. Runs are aggregated within patient before cross-patient
summaries; patients are the biological replicates.
"""

from __future__ import annotations

import argparse
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
from analyze_population_size_scaling import (
    aggregate_cv_probabilities,
    exact_sign_flip_p,
    fit_fold_probabilities,
    make_cv_splits,
    score_predictions,
)


OUT = ROOT / "neuropixel2025" / "data" / "combined_active_passive_population_candidate_panels"
REPORT = ROOT / "reports" / "active_passive_population_candidate_panels.md"

REMOVE_SPECS: tuple[tuple[str, int], ...] = (
    ("0", 0),
    ("1", 1),
    ("2", 2),
    ("5", 5),
    ("10", 10),
)
MODEL_SPECS: tuple[tuple[str, str, int | None], ...] = (
    ("top5", "top-5", 5),
    ("all", "all units", None),
)

WIDTH_MM = 90.0
HEIGHT_MM = 58.0
PT = 8.0

BLUE = "#357ABD"
RED = "#C44E52"
ORANGE = "#D98C3A"
GRAY = "#777D82"
LIGHT_GRAY = "#C7CDD3"
DARK = "#222222"

PATIENT_COLORS = {
    "gz001": "#0072B2",
    "gz002": "#D55E00",
    "gz003": "#009E73",
    "gz004": "#CC79A7",
    "gz005": "#E69F00",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": PT,
            "axes.labelsize": PT,
            "axes.titlesize": PT,
            "xtick.labelsize": PT,
            "ytick.labelsize": PT,
            "legend.fontsize": PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def aggregate_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    columns_by_fold: list[np.ndarray],
    c_value: float,
) -> np.ndarray:
    prob_sum = np.zeros(len(y), dtype=float)
    prob_count = np.zeros(len(y), dtype=float)
    if len(columns_by_fold) != len(splits):
        raise ValueError("Need one column set per CV fold")
    for (train_idx, test_idx), columns in zip(splits, columns_by_fold):
        probabilities = fit_fold_probabilities(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            columns,
            c_value,
        )
        prob_sum[test_idx] += probabilities
        prob_count[test_idx] += 1
    if np.any(prob_count == 0):
        raise RuntimeError("At least one trial was never held out")
    return prob_sum / prob_count


def cv_ablation_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    requested_remove: int,
    c_value: float,
    removal: str,
    random_state: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    column_sets: list[np.ndarray] = []
    effective_removed: list[int] = []
    for train_idx, _ in splits:
        n_remove = min(int(requested_remove), max(x.shape[1] - 1, 0))
        if n_remove == 0:
            removed = np.empty(0, dtype=int)
        elif removal == "ranked":
            scores = training_t_scores(x[train_idx], y[train_idx])
            removed = np.argsort(scores)[::-1][:n_remove]
        elif removal == "random":
            removed = np.sort(rng.choice(x.shape[1], size=n_remove, replace=False))
        else:
            raise ValueError(f"Unknown removal rule: {removal}")
        keep = np.setdiff1d(np.arange(x.shape[1], dtype=int), removed, assume_unique=True)
        if len(keep) == 0:
            raise RuntimeError("Ablation removed every unit")
        column_sets.append(keep)
        effective_removed.append(n_remove)
    probabilities = aggregate_probabilities(x, y, splits, column_sets, c_value)
    return probabilities, float(np.median(effective_removed))


def condition_preserving_trial_shuffle(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    shuffled = np.asarray(x, dtype=float).copy()
    for column in range(x.shape[1]):
        for condition in (0, 1):
            indices = np.flatnonzero(y == condition)
            shuffled[indices, column] = x[rng.permutation(indices), column]
    return shuffled


def cv_population_distance(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    requested_k: int | None,
) -> float:
    distances: list[float] = []
    for train_idx, test_idx in splits:
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        center = np.mean(x_train, axis=0)
        scale = np.std(x_train, axis=0, ddof=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        z_train = (x_train - center) / scale
        z_test = (x_test - center) / scale
        if requested_k is None or requested_k >= x.shape[1]:
            columns = np.arange(x.shape[1], dtype=int)
        else:
            scores = training_t_scores(x_train, y_train)
            columns = np.argsort(scores)[::-1][:requested_k]
        delta_train = (
            z_train[y_train == 1][:, columns].mean(axis=0)
            - z_train[y_train == 0][:, columns].mean(axis=0)
        )
        delta_test = (
            z_test[y_test == 1][:, columns].mean(axis=0)
            - z_test[y_test == 0][:, columns].mean(axis=0)
        )
        distances.append(float(np.dot(delta_train, delta_test) / len(columns)))
    return float(np.mean(distances))


def aggregate_ablation(
    ranked_runs: pd.DataFrame,
    random_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked_patients = (
        ranked_runs.groupby(["subject", "remove_order", "remove_label"], as_index=False)["auc"]
        .median()
        .rename(columns={"auc": "ranked_removal_auc"})
    )
    random_patient_draws = (
        random_runs.groupby(
            ["subject", "remove_order", "remove_label", "draw"], as_index=False
        )["auc"]
        .median()
        .rename(columns={"auc": "random_removal_auc"})
    )
    random_patients = (
        random_patient_draws.groupby(
            ["subject", "remove_order", "remove_label"], as_index=False
        )["random_removal_auc"]
        .agg(
            random_removal_median_auc="median",
            random_removal_q025_auc=lambda values: float(np.quantile(values, 0.025)),
            random_removal_q975_auc=lambda values: float(np.quantile(values, 0.975)),
        )
    )
    patient_values = ranked_patients.merge(
        random_patients,
        on=["subject", "remove_order", "remove_label"],
        validate="one_to_one",
    )
    patient_values["random_minus_ranked_removal_auc"] = (
        patient_values.random_removal_median_auc - patient_values.ranked_removal_auc
    )
    rows: list[dict[str, object]] = []
    for (order, label), frame in patient_values.groupby(
        ["remove_order", "remove_label"], sort=True
    ):
        ranked = frame.ranked_removal_auc.to_numpy(float)
        random = frame.random_removal_median_auc.to_numpy(float)
        difference = random - ranked
        rows.append(
            {
                "remove_order": int(order),
                "remove_label": str(label),
                "n_patients": int(frame.subject.nunique()),
                "ranked_removal_patient_median_auc": float(np.median(ranked)),
                "ranked_removal_patient_q25_auc": float(np.quantile(ranked, 0.25)),
                "ranked_removal_patient_q75_auc": float(np.quantile(ranked, 0.75)),
                "random_removal_patient_median_auc": float(np.median(random)),
                "random_removal_patient_q25_auc": float(np.quantile(random, 0.25)),
                "random_removal_patient_q75_auc": float(np.quantile(random, 0.75)),
                "random_minus_ranked_patient_median_auc": float(np.median(difference)),
                "random_minus_ranked_exact_sign_flip_p_greater": exact_sign_flip_p(
                    difference, alternative="greater"
                ),
            }
        )
    return patient_values, random_patient_draws, pd.DataFrame(rows).sort_values("remove_order")


def aggregate_trial_shuffle(
    observed_runs: pd.DataFrame,
    shuffled_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_patients = (
        observed_runs.groupby(["subject", "model", "model_label"], as_index=False)["auc"]
        .median()
        .rename(columns={"auc": "simultaneous_auc"})
    )
    shuffled_patient_draws = (
        shuffled_runs.groupby(
            ["subject", "model", "model_label", "draw"], as_index=False
        )["auc"]
        .median()
        .rename(columns={"auc": "trial_shuffled_auc"})
    )
    shuffled_patients = (
        shuffled_patient_draws.groupby(
            ["subject", "model", "model_label"], as_index=False
        )["trial_shuffled_auc"]
        .agg(
            trial_shuffled_median_auc="median",
            trial_shuffled_q025_auc=lambda values: float(np.quantile(values, 0.025)),
            trial_shuffled_q975_auc=lambda values: float(np.quantile(values, 0.975)),
        )
    )
    patient_values = observed_patients.merge(
        shuffled_patients,
        on=["subject", "model", "model_label"],
        validate="one_to_one",
    )
    patient_values["simultaneous_minus_shuffled_auc"] = (
        patient_values.simultaneous_auc - patient_values.trial_shuffled_median_auc
    )
    rows: list[dict[str, object]] = []
    for (model, label), frame in patient_values.groupby(["model", "model_label"], sort=True):
        simultaneous = frame.simultaneous_auc.to_numpy(float)
        shuffled = frame.trial_shuffled_median_auc.to_numpy(float)
        difference = simultaneous - shuffled
        rows.append(
            {
                "model": model,
                "model_label": label,
                "n_patients": int(frame.subject.nunique()),
                "simultaneous_patient_median_auc": float(np.median(simultaneous)),
                "simultaneous_patient_q25_auc": float(np.quantile(simultaneous, 0.25)),
                "simultaneous_patient_q75_auc": float(np.quantile(simultaneous, 0.75)),
                "trial_shuffled_patient_median_auc": float(np.median(shuffled)),
                "trial_shuffled_patient_q25_auc": float(np.quantile(shuffled, 0.25)),
                "trial_shuffled_patient_q75_auc": float(np.quantile(shuffled, 0.75)),
                "simultaneous_minus_shuffled_patient_median_auc": float(
                    np.median(difference)
                ),
                "simultaneous_minus_shuffled_exact_sign_flip_p_two_sided": exact_sign_flip_p(
                    difference, alternative="two-sided"
                ),
            }
        )
    return patient_values, shuffled_patient_draws, pd.DataFrame(rows)


def aggregate_distance(run_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_values = (
        run_values.groupby(["subject", "model", "model_label"], as_index=False)[
            "cv_population_distance"
        ]
        .median()
        .sort_values(["subject", "model"])
    )
    rows: list[dict[str, object]] = []
    for (model, label), frame in patient_values.groupby(["model", "model_label"], sort=True):
        values = frame.cv_population_distance.to_numpy(float)
        rows.append(
            {
                "model": model,
                "model_label": label,
                "n_patients": int(frame.subject.nunique()),
                "patient_median_cv_population_distance": float(np.median(values)),
                "patient_q25_cv_population_distance": float(np.quantile(values, 0.25)),
                "patient_q75_cv_population_distance": float(np.quantile(values, 0.75)),
                "exact_sign_flip_p_greater_than_zero": exact_sign_flip_p(
                    values, alternative="greater"
                ),
            }
        )
    return patient_values, pd.DataFrame(rows)


def draw_ablation(ax: plt.Axes, patient_values: pd.DataFrame, summary: pd.DataFrame) -> None:
    for subject, frame in patient_values.groupby("subject", sort=True):
        frame = frame.sort_values("remove_order")
        ax.plot(
            frame.remove_order,
            frame.ranked_removal_auc,
            color=PATIENT_COLORS[str(subject)],
            marker="o",
            ms=3.0,
            lw=0.75,
            alpha=0.70,
            zorder=2,
        )
    summary = summary.sort_values("remove_order")
    x = summary.remove_order.to_numpy(float)
    ax.fill_between(
        x,
        summary.random_removal_patient_q25_auc,
        summary.random_removal_patient_q75_auc,
        color=LIGHT_GRAY,
        alpha=0.38,
        lw=0,
        zorder=0,
    )
    ax.plot(
        x,
        summary.random_removal_patient_median_auc,
        color=GRAY,
        marker="o",
        ms=4.0,
        lw=1.45,
        label="random removal (IQR)",
        zorder=3,
    )
    ax.plot(
        x,
        summary.ranked_removal_patient_median_auc,
        color=RED,
        marker="o",
        ms=4.4,
        lw=1.8,
        label="top-ranked removal",
        zorder=4,
    )
    ax.axhline(0.5, color=GRAY, ls="--", lw=0.8)
    ax.set_xticks(x, summary.remove_label.astype(str))
    ax.set_xlim(-0.25, float(x.max()) + 0.25)
    ax.set_ylim(0.15, 0.90)
    ax.set_xlabel("number of units removed")
    ax.set_ylabel("cross-validated AUC")
    ax.set_title("Top-unit ablation", loc="left", fontweight="bold", pad=4)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.29),
        ncol=2,
        handlelength=1.15,
        handletextpad=0.35,
        columnspacing=0.7,
        borderaxespad=0.0,
    )


def draw_trial_shuffle(
    ax: plt.Axes,
    patient_values: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    frame = patient_values[patient_values.model == "top5"].sort_values("subject")
    summary_row = summary[summary.model == "top5"].iloc[0]
    for row in frame.itertuples(index=False):
        color = PATIENT_COLORS[str(row.subject)]
        ax.plot(
            [0, 1],
            [row.simultaneous_auc, row.trial_shuffled_median_auc],
            color="#D6DBE0",
            lw=0.9,
            zorder=1,
        )
        ax.scatter(
            [0, 1],
            [row.simultaneous_auc, row.trial_shuffled_median_auc],
            s=20,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
    medians = [
        float(summary_row.simultaneous_patient_median_auc),
        float(summary_row.trial_shuffled_patient_median_auc),
    ]
    for index, (median, color) in enumerate(zip(medians, (BLUE, ORANGE))):
        ax.plot([index - 0.15, index + 0.15], [median, median], color=DARK, lw=1.2)
        ax.scatter(
            [index],
            [median],
            marker="D",
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
    ax.axhline(0.5, color=GRAY, ls="--", lw=0.8)
    ax.set_xticks([0, 1], ["simultaneous", "condition-wise\ntrial shuffle"])
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(0.25, 0.90)
    ax.set_ylabel("cross-validated AUC")
    ax.set_title("Trial-shuffled pseudo-population", loc="left", fontweight="bold", pad=4)
    ax.text(
        0.50,
        0.98,
        f"top-5: median Δ = {summary_row.simultaneous_minus_shuffled_patient_median_auc:.2f}\n"
        f"two-sided P = {summary_row.simultaneous_minus_shuffled_exact_sign_flip_p_two_sided:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )


def draw_distance(
    ax: plt.Axes,
    patient_values: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    model_order = ["top5", "all"]
    labels = ["top-5", "all units"]
    wide = (
        patient_values.pivot(index="subject", columns="model", values="cv_population_distance")
        .reindex(columns=model_order)
        .sort_index()
    )
    for subject, row in wide.iterrows():
        color = PATIENT_COLORS[str(subject)]
        ax.plot([0, 1], row.to_numpy(float), color="#D6DBE0", lw=0.9, zorder=1)
        ax.scatter(
            [0, 1],
            row.to_numpy(float),
            s=20,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
    summary_by_model = summary.set_index("model")
    medians = [
        float(summary_by_model.loc[model, "patient_median_cv_population_distance"])
        for model in model_order
    ]
    for index, median in enumerate(medians):
        ax.scatter(
            [index],
            [median],
            marker="D",
            s=36,
            color=BLUE if index == 0 else GRAY,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        ax.plot([index - 0.15, index + 0.15], [median, median], color=DARK, lw=1.2)
    all_values = wide.to_numpy(float)
    limit = max(0.25, float(np.nanmax(np.abs(all_values))) * 1.18)
    ax.axhline(0, color=GRAY, ls="--", lw=0.8)
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(-limit, limit)
    ax.set_ylabel("cross-validated distance (a.u.)")
    ax.set_title("Cross-validated population distance", loc="left", fontweight="bold", pad=4)
    top5 = summary_by_model.loc["top5"]
    ax.text(
        0.02,
        0.05,
        f"top-5 median = {top5.patient_median_cv_population_distance:.2f}\n"
        f"one-sided P = {top5.exact_sign_flip_p_greater_than_zero:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )


def save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
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


def draw_candidate_figures(
    ablation_patients: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    shuffle_patients: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    distance_patients: pd.DataFrame,
    distance_summary: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    apply_style()
    outputs: dict[str, dict[str, str]] = {}

    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.32, top=0.88)
    draw_ablation(ax, ablation_patients, ablation_summary)
    outputs["ablation"] = save_figure(fig, OUT / "candidate_e_top_unit_ablation_90mm")

    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.25, top=0.88)
    draw_trial_shuffle(ax, shuffle_patients, shuffle_summary)
    outputs["trial_shuffle"] = save_figure(
        fig, OUT / "candidate_e_trial_shuffled_population_90mm"
    )

    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.25, top=0.88)
    draw_distance(ax, distance_patients, distance_summary)
    outputs["population_distance"] = save_figure(
        fig, OUT / "candidate_e_cv_population_distance_90mm"
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180 / 25.4, 72 / 25.4),
        facecolor="white",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.28, top=0.88, wspace=0.58)
    draw_ablation(axes[0], ablation_patients, ablation_summary)
    draw_trial_shuffle(axes[1], shuffle_patients, shuffle_summary)
    draw_distance(axes[2], distance_patients, distance_summary)
    axes[1].set_title("Trial-shuffled population", loc="left", fontweight="bold", pad=4)
    axes[2].set_title("Cross-validated distance", loc="left", fontweight="bold", pad=4)
    for index, ax in enumerate(axes):
        ax.text(
            -0.22,
            1.08,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontsize=PT,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
    outputs["comparison"] = save_figure(
        fig, OUT / "population_encoding_candidate_panels_180mm"
    )
    return outputs


def write_report(
    metadata: dict[str, object],
    ablation_summary: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Candidate population-encoding panels for revised Fig. 2e",
        "",
        "## Shared analysis contract",
        "",
        "- Artifact fraction <=0.2; Kilosort `good` units only.",
        "- Baseline-corrected 0-0.8 s response after function-word onset.",
        "- L2 logistic regression with C=0.1 for decoder-based candidates.",
        "- Twenty repeats of stratified fivefold cross-validation; all ranking/removal decisions are recomputed within the training fold.",
        "- Runs are aggregated by median within patient; n=5 patients are the biological replicates.",
        "- All three candidates are exploratory and are reported together, including negative results.",
        "",
        "## Candidate 1: top-unit ablation",
        "",
        "The decoder uses all remaining good units after either the most informative training-fold units or the same number of random units are removed. Positive random-minus-ranked differences indicate that removal of top-ranked units is more damaging than random removal. Computational ablation is not causal neuronal perturbation.",
        "",
        ablation_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Candidate 2: condition-preserving trial shuffle",
        "",
        "Each unit is shuffled independently across trials within active and passive conditions. This preserves each unit's marginal condition distribution while destroying trial-level cross-unit covariance. The control uses condition labels by construction and is therefore a covariance diagnostic, not a prospective held-out predictor.",
        "",
        shuffle_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Candidate 3: cross-validated population distance",
        "",
        "Within each fold, units are standardized using training data. The train and test active-minus-passive centroid differences are multiplied and divided by the number of retained units. The signed cross-validated distance is expected to center on zero if separation does not replicate across held-out trials. This is a diagonal standardized distance, not a full-covariance Mahalanobis estimate.",
        "",
        distance_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation boundary",
        "",
        "All candidates quantify construction-linked population information. Active/passive construction, function-word identity and associated acoustics covary in the current materials, so none establishes abstract voice coding.",
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
    ablation_ranked_rows: list[dict[str, object]] = []
    ablation_random_rows: list[dict[str, object]] = []
    shuffle_observed_rows: list[dict[str, object]] = []
    shuffle_draw_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
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
                or n_units_variable < 2
            ):
                skipped_rows.append({**common, "reason": "insufficient trials or units"})
                continue

            splits = make_cv_splits(y, args.cv_repeats, args.random_state)

            for remove_order, (remove_label, requested_remove) in enumerate(REMOVE_SPECS):
                ranked_probs, effective_remove = cv_ablation_probabilities(
                    x,
                    y,
                    splits,
                    requested_remove,
                    args.c_value,
                    removal="ranked",
                    random_state=args.random_state,
                )
                ablation_ranked_rows.append(
                    {
                        **common,
                        "remove_order": remove_order,
                        "remove_label": remove_label,
                        "requested_remove": requested_remove,
                        "median_effective_removed": effective_remove,
                        **score_predictions(y, ranked_probs),
                    }
                )
                n_draws = 1 if requested_remove == 0 else args.random_draws
                for draw in range(n_draws):
                    random_probs, random_effective = cv_ablation_probabilities(
                        x,
                        y,
                        splits,
                        requested_remove,
                        args.c_value,
                        removal="random",
                        random_state=args.random_state
                        + stable_seed(subject, run, remove_label, "ablation", draw) % 1000000,
                    )
                    ablation_random_rows.append(
                        {
                            **common,
                            "remove_order": remove_order,
                            "remove_label": remove_label,
                            "requested_remove": requested_remove,
                            "median_effective_removed": random_effective,
                            "draw": draw,
                            **score_predictions(y, random_probs),
                        }
                    )

            for model, model_label, requested_k in MODEL_SPECS:
                observed_probs, effective_k = aggregate_cv_probabilities(
                    x,
                    y,
                    splits,
                    requested_k,
                    args.c_value,
                    selection="ranked",
                    random_state=args.random_state,
                )
                shuffle_observed_rows.append(
                    {
                        **common,
                        "model": model,
                        "model_label": model_label,
                        "median_effective_k": effective_k,
                        **score_predictions(y, observed_probs),
                    }
                )
                for draw in range(args.shuffle_draws):
                    shuffle_seed = (
                        args.random_state
                        + stable_seed(subject, run, model, "trial_shuffle", draw) % 1000000
                    )
                    shuffled_x = condition_preserving_trial_shuffle(x, y, shuffle_seed)
                    shuffled_probs, shuffled_effective_k = aggregate_cv_probabilities(
                        shuffled_x,
                        y,
                        splits,
                        requested_k,
                        args.c_value,
                        selection="ranked",
                        random_state=args.random_state,
                    )
                    shuffle_draw_rows.append(
                        {
                            **common,
                            "model": model,
                            "model_label": model_label,
                            "median_effective_k": shuffled_effective_k,
                            "draw": draw,
                            **score_predictions(y, shuffled_probs),
                        }
                    )

                distance_rows.append(
                    {
                        **common,
                        "model": model,
                        "model_label": model_label,
                        "median_effective_k": effective_k,
                        "cv_population_distance": cv_population_distance(
                            x, y, splits, requested_k
                        ),
                    }
                )

    ablation_ranked_runs = pd.DataFrame(ablation_ranked_rows).sort_values(
        ["subject", "run", "remove_order"]
    )
    ablation_random_runs = pd.DataFrame(ablation_random_rows).sort_values(
        ["subject", "run", "remove_order", "draw"]
    )
    shuffle_observed_runs = pd.DataFrame(shuffle_observed_rows).sort_values(
        ["subject", "run", "model"]
    )
    shuffle_draw_runs = pd.DataFrame(shuffle_draw_rows).sort_values(
        ["subject", "run", "model", "draw"]
    )
    distance_runs = pd.DataFrame(distance_rows).sort_values(["subject", "run", "model"])
    skipped = pd.DataFrame(skipped_rows)
    if ablation_ranked_runs.empty:
        raise RuntimeError("No eligible runs for candidate analyses")
    if ablation_ranked_runs.subject.nunique() != 5:
        raise RuntimeError(
            f"Expected five patients, found {ablation_ranked_runs.subject.nunique()}"
        )

    ablation_patients, ablation_random_patient_draws, ablation_summary = aggregate_ablation(
        ablation_ranked_runs, ablation_random_runs
    )
    shuffle_patients, shuffle_patient_draws, shuffle_summary = aggregate_trial_shuffle(
        shuffle_observed_runs, shuffle_draw_runs
    )
    distance_patients, distance_summary = aggregate_distance(distance_runs)

    tables = {
        "ablation_ranked_run_values": ablation_ranked_runs,
        "ablation_random_run_draws": ablation_random_runs,
        "ablation_patient_values": ablation_patients,
        "ablation_random_patient_draws": ablation_random_patient_draws,
        "ablation_group_summary": ablation_summary,
        "trial_shuffle_simultaneous_run_values": shuffle_observed_runs,
        "trial_shuffle_run_draws": shuffle_draw_runs,
        "trial_shuffle_patient_values": shuffle_patients,
        "trial_shuffle_patient_draws": shuffle_patient_draws,
        "trial_shuffle_group_summary": shuffle_summary,
        "cv_population_distance_run_values": distance_runs,
        "cv_population_distance_patient_values": distance_patients,
        "cv_population_distance_group_summary": distance_summary,
        "skipped_runs": skipped,
    }
    table_paths: dict[str, str] = {}
    for name, table in tables.items():
        path = OUT / f"{name}.csv"
        table.to_csv(path, index=False)
        table_paths[name] = str(path)

    figure_outputs = draw_candidate_figures(
        ablation_patients,
        ablation_summary,
        shuffle_patients,
        shuffle_summary,
        distance_patients,
        distance_summary,
    )

    ablation_top5 = ablation_summary[ablation_summary.remove_label == "5"].iloc[0]
    shuffle_top5 = shuffle_summary[shuffle_summary.model == "top5"].iloc[0]
    distance_top5 = distance_summary[distance_summary.model == "top5"].iloc[0]
    distance_all = distance_summary[distance_summary.model == "all"].iloc[0]
    metadata: dict[str, object] = {
        "analysis": "candidate population encoding panels for revised Fig. 2e",
        "status": "exploratory",
        "analysis_mode": "artifact_clean_le_0p2",
        "unit_set": "good_only",
        "metric": "response_baseline_corrected_hz",
        "decoder": "StandardScaler + L2 logistic regression",
        "c_value": args.c_value,
        "cv_repeats": args.cv_repeats,
        "random_draws": args.random_draws,
        "shuffle_draws": args.shuffle_draws,
        "random_state": args.random_state,
        "n_patients": int(ablation_ranked_runs.subject.nunique()),
        "n_runs": int(ablation_ranked_runs[["subject", "run"]].drop_duplicates().shape[0]),
        "ablation_remove5_ranked_patient_median_auc": float(
            ablation_top5.ranked_removal_patient_median_auc
        ),
        "ablation_remove5_random_patient_median_auc": float(
            ablation_top5.random_removal_patient_median_auc
        ),
        "ablation_remove5_random_minus_ranked_patient_median_auc": float(
            ablation_top5.random_minus_ranked_patient_median_auc
        ),
        "ablation_remove5_exact_sign_flip_p_greater": float(
            ablation_top5.random_minus_ranked_exact_sign_flip_p_greater
        ),
        "trial_shuffle_top5_simultaneous_patient_median_auc": float(
            shuffle_top5.simultaneous_patient_median_auc
        ),
        "trial_shuffle_top5_shuffled_patient_median_auc": float(
            shuffle_top5.trial_shuffled_patient_median_auc
        ),
        "trial_shuffle_top5_difference_patient_median_auc": float(
            shuffle_top5.simultaneous_minus_shuffled_patient_median_auc
        ),
        "trial_shuffle_top5_exact_sign_flip_p_two_sided": float(
            shuffle_top5.simultaneous_minus_shuffled_exact_sign_flip_p_two_sided
        ),
        "cv_distance_top5_patient_median": float(
            distance_top5.patient_median_cv_population_distance
        ),
        "cv_distance_top5_exact_sign_flip_p_greater": float(
            distance_top5.exact_sign_flip_p_greater_than_zero
        ),
        "cv_distance_all_patient_median": float(
            distance_all.patient_median_cv_population_distance
        ),
        "cv_distance_all_exact_sign_flip_p_greater": float(
            distance_all.exact_sign_flip_p_greater_than_zero
        ),
        "source_data": table_paths,
        "figure_outputs": figure_outputs,
        "reviewer_risks": [
            "All candidate analyses are exploratory and must be reported together rather than selected by significance.",
            "Active/passive construction and function-word identity/acoustics covary in the current design.",
            "Computational ablation tests decoder dependence and is not causal neuronal perturbation.",
            "Condition-wise trial shuffling uses class labels by construction and is a covariance diagnostic.",
            "The cross-validated distance uses diagonal training-fold standardization, not a full covariance estimate.",
            "Patients, not runs, folds or shuffles, are the biological replicates.",
        ],
    }
    metadata_path = OUT / "population_encoding_candidate_panels_metadata.json"
    metadata["metadata_json"] = str(metadata_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(metadata, ablation_summary, shuffle_summary, distance_summary)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--random-draws", type=int, default=20)
    parser.add_argument("--shuffle-draws", type=int, default=20)
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--min-trials-total", type=int, default=10)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
