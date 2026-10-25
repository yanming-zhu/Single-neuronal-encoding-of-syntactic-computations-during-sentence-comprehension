#!/usr/bin/env python3
"""Compare single-unit and population predictive information.

The primary analysis uses one fixed regularization value (C=0.1) for both
readouts. A nested-inner-CV analysis is exported as a calibration sensitivity.
All unit selection, scaling, regularization tuning, and model fitting are
confined to training data. Repeated random-CV predictions are averaged for each
original trial before any information or AUC score is calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    from scripts.analyze_three_paper_borrowed_methods_20260724 import (
        POP_PATH,
        TRIAL_PATH,
        load_response_metrics,
        make_response_matrix,
    )
except ModuleNotFoundError:
    from analyze_three_paper_borrowed_methods_20260724 import (
        POP_PATH,
        TRIAL_PATH,
        load_response_metrics,
        make_response_matrix,
    )


DEFAULT_OUT = (
    ROOT
    / "neuropixel2025"
    / "data"
    / "combined_unit_population_predictive_information"
)
DEFAULT_REPORT = ROOT / "reports" / "unit_population_predictive_information.md"
MODEL_SPECS = (
    ("single_unit", "Single unit", 1),
    ("top5_population", "Top-5 population", 5),
)
VALIDATION_SPECS = (
    ("all_sentences", "All sentences"),
    ("unseen_sentence_pair", "Unseen sentence pairs"),
)
MODEL_COLORS = {
    "single_unit": "#7A7F85",
    "top5_population": "#0072B2",
}
VALIDATION_COLORS = {
    "all_sentences": "#8C8C8C",
    "unseen_sentence_pair": "#009E73",
}


def exact_sign_test(differences: np.ndarray) -> dict[str, int | float]:
    """One-sided exact sign test for independent patient-level differences."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & (~np.isclose(values, 0.0))]
    n_positive = int(np.sum(values > 0))
    p_value = (
        float(stats.binomtest(n_positive, len(values), 0.5, alternative="greater").pvalue)
        if len(values)
        else float("nan")
    )
    return {
        "n_nonzero": int(len(values)),
        "n_positive": n_positive,
        "p_greater": p_value,
    }


def predictive_information_bits(
    y_true: np.ndarray,
    model_probability: np.ndarray,
    null_probability: np.ndarray,
) -> float:
    """Return mean held-out log-likelihood gain over a null model, in bits."""
    y = np.asarray(y_true, dtype=int)
    model_p1 = np.asarray(model_probability, dtype=float)
    null_p1 = np.asarray(null_probability, dtype=float)
    if not (y.shape == model_p1.shape == null_p1.shape):
        raise ValueError("labels and probability arrays must have identical shapes")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    epsilon = np.finfo(float).eps
    model_p1 = np.clip(model_p1, epsilon, 1.0 - epsilon)
    null_p1 = np.clip(null_p1, epsilon, 1.0 - epsilon)
    model_correct = np.where(y == 1, model_p1, 1.0 - model_p1)
    null_correct = np.where(y == 1, null_p1, 1.0 - null_p1)
    return float(np.mean(np.log2(model_correct / null_correct)))


def aggregate_oof_probabilities(
    n_trials: int,
    test_indices: list[np.ndarray],
    probability_batches: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Average repeated out-of-fold probabilities once for each original trial."""
    if len(test_indices) != len(probability_batches):
        raise ValueError("each test-index batch needs one probability batch")
    probability_sum = np.zeros(n_trials, dtype=float)
    counts = np.zeros(n_trials, dtype=int)
    for indices, probabilities in zip(test_indices, probability_batches):
        idx = np.asarray(indices, dtype=int)
        probs = np.asarray(probabilities, dtype=float)
        if idx.shape != probs.shape:
            raise ValueError("test indices and probabilities must have matching shapes")
        np.add.at(probability_sum, idx, probs)
        np.add.at(counts, idx, 1)
    averaged = np.full(n_trials, np.nan, dtype=float)
    covered = counts > 0
    averaged[covered] = probability_sum[covered] / counts[covered]
    return averaged, counts


def leave_one_group_out_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return folds that completely hold out each group exactly once."""
    group_values = np.asarray(groups)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for group in np.unique(group_values):
        test = np.flatnonzero(group_values == group)
        train = np.flatnonzero(group_values != group)
        folds.append((train, test))
    return folds


def training_t_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank units by absolute Welch-style t score using training rows only."""
    active = x[y == 1]
    passive = x[y == 0]
    active_var = active.var(axis=0, ddof=1)
    passive_var = passive.var(axis=0, ddof=1)
    denominator = np.sqrt(active_var / len(active) + passive_var / len(passive))
    difference = np.abs(active.mean(axis=0) - passive.mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = difference / denominator
    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=np.finfo(float).max,
        neginf=0.0,
    )


def _selected_columns(x_train: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(training_t_scores(x_train, y_train), kind="stable")[::-1]
    return order[: min(k, x_train.shape[1])]


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    columns: np.ndarray,
    c_value: float,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            class_weight="balanced",
            max_iter=1500,
            random_state=0,
        ),
    )
    model.fit(x_train[:, columns], y_train)
    return model.predict_proba(x_test[:, columns])[:, 1]


def _choose_regularization(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    candidate_cs: tuple[float, ...],
    random_state: int,
) -> float:
    min_class = int(np.bincount(y_train, minlength=2).min())
    n_splits = min(4, min_class)
    if n_splits < 2:
        return float(candidate_cs[0])
    inner = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = list(inner.split(x_train, y_train))
    scored: list[tuple[float, float]] = []
    for c_value in candidate_cs:
        test_indices: list[np.ndarray] = []
        probability_batches: list[np.ndarray] = []
        null_batches: list[np.ndarray] = []
        for inner_train, inner_test in folds:
            columns = _selected_columns(
                x_train[inner_train], y_train[inner_train], k=k
            )
            probability_batches.append(
                _fit_predict(
                    x_train[inner_train],
                    y_train[inner_train],
                    x_train[inner_test],
                    columns,
                    c_value,
                )
            )
            null_batches.append(
                np.full(len(inner_test), float(np.mean(y_train[inner_train])))
            )
            test_indices.append(inner_test)
        model_probability, _ = aggregate_oof_probabilities(
            len(y_train), test_indices, probability_batches
        )
        null_probability, _ = aggregate_oof_probabilities(
            len(y_train), test_indices, null_batches
        )
        information = predictive_information_bits(
            y_train, model_probability, null_probability
        )
        scored.append((information, float(c_value)))
    best_information = max(value for value, _ in scored)
    return min(c for value, c in scored if np.isclose(value, best_information))


def fit_outer_fold(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
    regularization: str,
    fixed_c: float,
    candidate_cs: tuple[float, ...],
    random_state: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit one outer fold with all selection confined to its training data."""
    if regularization == "fixed":
        chosen_c = float(fixed_c)
    elif regularization == "nested":
        chosen_c = _choose_regularization(
            x_train,
            y_train,
            k=k,
            candidate_cs=candidate_cs,
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown regularization mode: {regularization}")
    selected = _selected_columns(x_train, y_train, k=k)
    probability = _fit_predict(
        x_train,
        y_train,
        x_test,
        selected,
        chosen_c,
    )
    return probability, chosen_c, selected


def cross_validated_run(
    *,
    x: np.ndarray,
    y: np.ndarray,
    pair_ids: np.ndarray,
    k: int,
    validation: str,
    regularization: str,
    fixed_c: float,
    candidate_cs: tuple[float, ...],
    n_repeats: int,
    random_state: int,
) -> dict[str, object]:
    """Return leakage-controlled out-of-fold predictions and information metrics."""
    if validation == "all_sentences":
        min_class = int(np.bincount(y, minlength=2).min())
        outer = RepeatedStratifiedKFold(
            n_splits=min(5, min_class),
            n_repeats=n_repeats,
            random_state=random_state,
        )
        folds = list(outer.split(x, y))
    elif validation == "unseen_sentence_pair":
        folds = leave_one_group_out_folds(pair_ids)
    else:
        raise ValueError(f"unknown validation mode: {validation}")

    test_indices: list[np.ndarray] = []
    model_batches: list[np.ndarray] = []
    null_batches: list[np.ndarray] = []
    chosen_cs: list[float] = []
    selected_units: list[np.ndarray] = []
    for fold_index, (train, test) in enumerate(folds):
        probability, chosen_c, selected = fit_outer_fold(
            x_train=x[train],
            y_train=y[train],
            x_test=x[test],
            k=k,
            regularization=regularization,
            fixed_c=fixed_c,
            candidate_cs=candidate_cs,
            random_state=random_state + fold_index + 1,
        )
        test_indices.append(test)
        model_batches.append(probability)
        null_batches.append(np.full(len(test), float(np.mean(y[train]))))
        chosen_cs.append(chosen_c)
        selected_units.append(selected)

    model_probability, prediction_count = aggregate_oof_probabilities(
        len(y), test_indices, model_batches
    )
    null_probability, null_count = aggregate_oof_probabilities(
        len(y), test_indices, null_batches
    )
    if not np.array_equal(prediction_count, null_count):
        raise RuntimeError("model and null predictions cover different trials")
    if np.any(prediction_count == 0):
        raise RuntimeError("at least one trial lacks an out-of-fold prediction")
    model_correct = np.where(y == 1, model_probability, 1.0 - model_probability)
    null_correct = np.where(y == 1, null_probability, 1.0 - null_probability)
    contribution = np.log2(
        np.clip(model_correct, np.finfo(float).eps, 1.0)
        / np.clip(null_correct, np.finfo(float).eps, 1.0)
    )
    return {
        "predictive_information_bits": float(np.mean(contribution)),
        "auc": float(roc_auc_score(y, model_probability)),
        "model_probability": model_probability,
        "null_probability": null_probability,
        "information_contribution_bits": contribution,
        "prediction_count": prediction_count,
        "chosen_cs": np.asarray(chosen_cs, dtype=float),
        "selected_units": selected_units,
        "outer_folds": folds,
    }


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    random_state: int,
    n_boot: int = 20_000,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(finite), size=(n_boot, len(finite)))
    medians = np.median(finite[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _patient_inference_row(
    values: np.ndarray,
    *,
    regularization: str,
    comparison: str,
    validation: str,
    model: str,
    reference: float,
    random_state: int,
) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    differences = finite - reference
    sign = exact_sign_test(differences)
    ci_low, ci_high = _bootstrap_median_ci(
        finite,
        random_state=random_state,
    )
    return {
        "regularization": regularization,
        "comparison": comparison,
        "validation": validation,
        "model": model,
        "reference": reference,
        "n_patients": int(len(finite)),
        "patient_median": float(np.median(finite)),
        "patient_mean": float(np.mean(finite)),
        "patient_min": float(np.min(finite)),
        "patient_max": float(np.max(finite)),
        "bootstrap_median_ci_low": ci_low,
        "bootstrap_median_ci_high": ci_high,
        "n_nonzero": sign["n_nonzero"],
        "n_positive": sign["n_positive"],
        "exact_sign_p_greater": sign["p_greater"],
    }


def analyze_dataset(
    *,
    cv_repeats: int,
    random_state: int,
    candidate_cs: tuple[float, ...],
    fixed_c: float,
    regularizations: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both validation regimes and aggregate runs within patient."""
    eligible = pd.read_csv(POP_PATH)[["subject", "run"]].drop_duplicates()
    metrics = load_response_metrics()
    trials = pd.read_csv(TRIAL_PATH)
    run_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for run_number, eligible_row in enumerate(eligible.itertuples(index=False)):
        subject = str(eligible_row.subject)
        run = str(eligible_row.run)
        x, y, meta, unit_ids = make_response_matrix(metrics, trials, subject, run)
        pair_ids = meta["pair_id"].to_numpy(int)
        for regularization in regularizations:
            for validation, validation_label in VALIDATION_SPECS:
                seed = random_state + _stable_seed(subject, run, validation) % 100_000
                for model, model_label, k in MODEL_SPECS:
                    result = cross_validated_run(
                        x=x,
                        y=y,
                        pair_ids=pair_ids,
                        k=k,
                        validation=validation,
                        regularization=regularization,
                        fixed_c=fixed_c,
                        candidate_cs=candidate_cs,
                        n_repeats=cv_repeats,
                        random_state=seed,
                    )
                    run_rows.append(
                        {
                            "subject": subject,
                            "run": run,
                            "regularization": regularization,
                            "validation": validation,
                            "validation_label": validation_label,
                            "model": model,
                            "model_label": model_label,
                            "n_trials": int(len(y)),
                            "n_units": int(x.shape[1]),
                            "predictive_information_bits_per_trial": result[
                                "predictive_information_bits"
                            ],
                            "auc": result["auc"],
                            "median_selected_c": float(np.median(result["chosen_cs"])),
                            "min_selected_c": float(np.min(result["chosen_cs"])),
                            "max_selected_c": float(np.max(result["chosen_cs"])),
                            "outer_fold_count": int(len(result["outer_folds"])),
                            "predictions_per_trial": int(
                                np.unique(result["prediction_count"])[0]
                            ),
                        }
                    )
                    for trial_position, trial_meta in enumerate(meta.itertuples(index=False)):
                        trial_rows.append(
                            {
                                "subject": subject,
                                "run": run,
                                "trial_index": int(trial_meta.trial_index),
                                "sentence_pair": int(pair_ids[trial_position]),
                                "condition": "active" if y[trial_position] == 1 else "passive",
                                "regularization": regularization,
                                "validation": validation,
                                "model": model,
                                "model_probability_active": float(
                                    result["model_probability"][trial_position]
                                ),
                                "null_probability_active": float(
                                    result["null_probability"][trial_position]
                                ),
                                "information_contribution_bits": float(
                                    result["information_contribution_bits"][trial_position]
                                ),
                                "prediction_count": int(
                                    result["prediction_count"][trial_position]
                                ),
                            }
                        )
                    for fold_index, ((train, test), c_value, selected) in enumerate(
                        zip(
                            result["outer_folds"],
                            result["chosen_cs"],
                            result["selected_units"],
                        )
                    ):
                        selected_ids = unit_ids[np.asarray(selected, dtype=int)]
                        held_pairs = np.unique(pair_ids[test])
                        fold_rows.append(
                            {
                                "subject": subject,
                                "run": run,
                                "regularization": regularization,
                                "validation": validation,
                                "model": model,
                                "outer_fold": fold_index,
                                "n_train": int(len(train)),
                                "n_test": int(len(test)),
                                "chosen_c": float(c_value),
                                "selected_unit_ids": ";".join(map(str, selected_ids.tolist())),
                                "held_sentence_pair": (
                                    int(held_pairs[0])
                                    if validation == "unseen_sentence_pair"
                                    else np.nan
                                ),
                            }
                        )
        print(
            f"completed {run_number + 1}/{len(eligible)} runs: {subject} {run}",
            flush=True,
        )

    run_values = pd.DataFrame(run_rows).sort_values(
        ["regularization", "validation", "model", "subject", "run"]
    )
    trial_values = pd.DataFrame(trial_rows).sort_values(
        ["regularization", "validation", "model", "subject", "run", "trial_index"]
    )
    fold_values = pd.DataFrame(fold_rows).sort_values(
        ["regularization", "validation", "model", "subject", "run", "outer_fold"]
    )
    patient_values = (
        run_values.groupby(
            [
                "subject",
                "regularization",
                "validation",
                "validation_label",
                "model",
                "model_label",
            ],
            as_index=False,
        )[
            ["predictive_information_bits_per_trial", "auc"]
        ]
        .median()
        .sort_values(["regularization", "validation", "model", "subject"])
    )

    inference_rows: list[dict[str, object]] = []
    for regularization in regularizations:
        reg = patient_values[patient_values.regularization == regularization]
        for validation, _ in VALIDATION_SPECS:
            val = reg[reg.validation == validation]
            for model, _, _ in MODEL_SPECS:
                model_values = val.loc[
                    val.model == model, "predictive_information_bits_per_trial"
                ].to_numpy(float)
                inference_rows.append(
                    _patient_inference_row(
                        model_values,
                        regularization=regularization,
                        comparison="model_vs_train_prior_null",
                        validation=validation,
                        model=model,
                        reference=0.0,
                        random_state=random_state
                        + _stable_seed(regularization, validation, model) % 100_000,
                    )
                )
            wide = val.pivot(
                index="subject",
                columns="model",
                values="predictive_information_bits_per_trial",
            )
            gain = (
                wide["top5_population"] - wide["single_unit"]
            ).to_numpy(float)
            inference_rows.append(
                _patient_inference_row(
                    gain,
                    regularization=regularization,
                    comparison="population_minus_single_unit",
                    validation=validation,
                    model="top5_population_minus_single_unit",
                    reference=0.0,
                    random_state=random_state
                    + _stable_seed(regularization, validation, "gain") % 100_000,
                )
            )

        info_wide = reg.pivot(
            index="subject",
            columns=["validation", "model"],
            values="predictive_information_bits_per_trial",
        )
        for model, _, _ in MODEL_SPECS:
            change = (
                info_wide[("unseen_sentence_pair", model)]
                - info_wide[("all_sentences", model)]
            ).to_numpy(float)
            inference_rows.append(
                _patient_inference_row(
                    change,
                    regularization=regularization,
                    comparison="unseen_minus_all_sentences",
                    validation="unseen_minus_all_sentences",
                    model=model,
                    reference=0.0,
                    random_state=random_state
                    + _stable_seed(regularization, model, "generalization_change")
                    % 100_000,
                )
            )
        interaction = (
            info_wide[("unseen_sentence_pair", "top5_population")]
            - info_wide[("unseen_sentence_pair", "single_unit")]
            - info_wide[("all_sentences", "top5_population")]
            + info_wide[("all_sentences", "single_unit")]
        ).to_numpy(float)
        inference_rows.append(
            _patient_inference_row(
                interaction,
                regularization=regularization,
                comparison="validation_by_model_gain_interaction",
                validation="unseen_minus_all_sentences",
                model="top5_population_minus_single_unit",
                reference=0.0,
                random_state=random_state
                + _stable_seed(regularization, "interaction") % 100_000,
            )
        )

    inference = pd.DataFrame(inference_rows)
    return run_values, patient_values, trial_values, fold_values, inference


def _apply_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_figure(
    patient_values: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    output_dir: Path,
    primary_regularization: str = "fixed",
) -> dict[str, str]:
    """Draw the three-panel patient-resolved primary result."""
    _apply_figure_style()
    primary = patient_values[
        patient_values.regularization == primary_regularization
    ].copy()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(180 / 25.4, 74 / 25.4),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.08], "wspace": 0.48},
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.76, wspace=0.50)
    patient_order = sorted(primary.subject.unique())

    information_values = primary["predictive_information_bits_per_trial"].to_numpy(float)
    information_span = max(float(np.ptp(information_values)), 0.08)
    info_low = min(float(np.min(information_values)) - 0.14 * information_span, -0.025)
    info_high = max(float(np.max(information_values)) + 0.20 * information_span, 0.06)

    for panel_index, (validation, validation_label) in enumerate(VALIDATION_SPECS):
        ax = axes[panel_index]
        sub = primary[primary.validation == validation]
        wide = sub.pivot(
            index="subject",
            columns="model",
            values="predictive_information_bits_per_trial",
        ).reindex(patient_order)
        for subject, row in wide.iterrows():
            ax.plot(
                [0, 1],
                [row.single_unit, row.top5_population],
                color="#C5C9CD",
                lw=0.85,
                zorder=1,
            )
        for x_position, (model, model_label, _) in enumerate(MODEL_SPECS):
            values = wide[model].to_numpy(float)
            ax.scatter(
                np.full(len(values), x_position),
                values,
                s=28,
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            summary = inference[
                (inference.regularization == primary_regularization)
                & (inference.comparison == "model_vs_train_prior_null")
                & (inference.validation == validation)
                & (inference.model == model)
            ].iloc[0]
            ax.errorbar(
                [x_position],
                [summary.patient_median],
                yerr=np.array(
                    [
                        [summary.patient_median - summary.bootstrap_median_ci_low],
                        [summary.bootstrap_median_ci_high - summary.patient_median],
                    ]
                ),
                fmt="D",
                markersize=4.1,
                color="#1E252B",
                markerfacecolor="white",
                markeredgewidth=0.8,
                capsize=2.2,
                lw=0.9,
                zorder=4,
            )
            auc_median = float(
                sub.loc[sub.model == model, "auc"].median()
            )
            ax.text(
                x_position,
                info_high - 0.025 * (info_high - info_low),
                f"I={summary.patient_median:+.3f}\nAUC={auc_median:.2f}",
                ha="center",
                va="top",
                color=MODEL_COLORS[model],
                fontsize=6.7,
            )
        gain = inference[
            (inference.regularization == primary_regularization)
            & (inference.comparison == "population_minus_single_unit")
            & (inference.validation == validation)
        ].iloc[0]
        ax.text(
            0.5,
            info_low + 0.035 * (info_high - info_low),
            f"population gain: {int(gain.n_positive)}/5 positive; "
            f"exact P={gain.exact_sign_p_greater:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#444A50",
        )
        ax.axhline(0, color="#5F666D", ls="--", lw=0.75, zorder=0)
        ax.set_xticks([0, 1], ["Single\nunit", "Top-5\npopulation"])
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(info_low, info_high)
        ax.set_title(
            ("a  " if panel_index == 0 else "b  ") + validation_label,
            loc="left",
            fontweight="bold",
            pad=5,
        )
        if panel_index == 0:
            ax.set_ylabel("Cross-validated predictive information\n(bits/trial)")

    ax = axes[2]
    gain_frame = []
    for validation, validation_label in VALIDATION_SPECS:
        sub = primary[primary.validation == validation]
        wide = sub.pivot(
            index="subject",
            columns="model",
            values="predictive_information_bits_per_trial",
        ).reindex(patient_order)
        for subject in patient_order:
            gain_frame.append(
                {
                    "subject": subject,
                    "validation": validation,
                    "validation_label": validation_label,
                    "gain": float(
                        wide.loc[subject, "top5_population"]
                        - wide.loc[subject, "single_unit"]
                    ),
                }
            )
    gain_values = pd.DataFrame(gain_frame)
    gain_wide = gain_values.pivot(
        index="subject", columns="validation", values="gain"
    ).reindex(patient_order)
    for subject, row in gain_wide.iterrows():
        ax.plot(
            [0, 1],
            [row.all_sentences, row.unseen_sentence_pair],
            color="#C5C9CD",
            lw=0.85,
            zorder=1,
        )
    for x_position, (validation, validation_label) in enumerate(VALIDATION_SPECS):
        values = gain_wide[validation].to_numpy(float)
        summary = inference[
            (inference.regularization == primary_regularization)
            & (inference.comparison == "population_minus_single_unit")
            & (inference.validation == validation)
        ].iloc[0]
        ax.scatter(
            np.full(len(values), x_position),
            values,
            s=28,
            color=VALIDATION_COLORS[validation],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.errorbar(
            [x_position],
            [summary.patient_median],
            yerr=np.array(
                [
                    [summary.patient_median - summary.bootstrap_median_ci_low],
                    [summary.bootstrap_median_ci_high - summary.patient_median],
                ]
            ),
            fmt="D",
            markersize=4.1,
            color="#1E252B",
            markerfacecolor="white",
            markeredgewidth=0.8,
            capsize=2.2,
            lw=0.9,
            zorder=4,
        )
        ax.text(
            x_position,
            float(np.max(gain_wide.to_numpy())) + 0.05 * max(np.ptp(gain_wide.to_numpy()), 0.08),
            f"median {summary.patient_median:+.3f}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color=VALIDATION_COLORS[validation],
        )
    gain_span = max(float(np.ptp(gain_wide.to_numpy())), 0.08)
    gain_low = min(float(np.min(gain_wide.to_numpy())) - 0.15 * gain_span, -0.025)
    gain_high = max(float(np.max(gain_wide.to_numpy())) + 0.22 * gain_span, 0.06)
    ax.axhline(0, color="#5F666D", ls="--", lw=0.75, zorder=0)
    ax.set_xticks([0, 1], ["All\nsentences", "Unseen\nsentence pairs"])
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(gain_low, gain_high)
    ax.set_ylabel("Population information gain, ΔI\n(bits/trial)")
    ax.set_title("c  Population advantage", loc="left", fontweight="bold", pad=5)

    fig.suptitle(
        "Population activity preserves predictive information across unseen sentences",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=9.3,
        fontweight="bold",
    )
    fig.text(
        0.99,
        0.025,
        "All sentences: repeated stratified CV; unseen pairs: leave-one-pair-out CV. "
        "Points: patients (n=5); diamonds: median; bars: bootstrap 95% CI.",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#666C72",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "unit_population_predictive_information_180mm"
    outputs = {
        "png": str(stem.with_suffix(".png")),
        "svg": str(stem.with_suffix(".svg")),
        "pdf": str(stem.with_suffix(".pdf")),
        "tiff": str(stem.with_suffix(".tiff")),
    }
    fig.savefig(outputs["png"], dpi=300, facecolor="white", bbox_inches=None)
    fig.savefig(outputs["svg"], facecolor="white", bbox_inches=None)
    fig.savefig(outputs["pdf"], facecolor="white", bbox_inches=None)
    fig.savefig(
        outputs["tiff"],
        dpi=600,
        facecolor="white",
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    with Image.open(outputs["tiff"]) as source:
        rgb = Image.new("RGB", source.size, "white")
        if source.mode == "RGBA":
            rgb.paste(source, mask=source.getchannel("A"))
        else:
            rgb.paste(source.convert("RGB"))
        rgb.save(outputs["tiff"], dpi=(600, 600), compression="tiff_lzw")
    return outputs


def write_report(
    patient_values: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    report_path: Path,
    outputs: dict[str, str],
    metadata_path: Path,
) -> None:
    primary = patient_values[patient_values.regularization == "fixed"]
    lines = [
        "# Single-unit and population predictive information",
        "",
        "## Analysis definition",
        "",
        "Predictive information is the held-out log-likelihood gain over the training-fold class-prior null: `mean(log2[p_model(y)/p_null(y)])`, in bits/trial. Zero means no improvement over the null; a negative value means out-of-sample probabilities were worse than the null, not that the recorded unit had no physiological response.",
        "",
        "The designated primary analysis uses artifact-unfiltered Kilosort `good` units to match the existing Fig. 2 sentence-pair analysis, a common fixed L2 regularization (`C=0.1`) for both readouts, and training-fold selection of either one unit or five units. This fixed value was inherited from the existing decoder rather than chosen after viewing the information result. The nested-inner-CV regularization result is reported as a sensitivity analysis. Seven runs are summarized within five patients.",
        "",
        "## Primary fixed-regularization results",
        "",
    ]
    for validation, validation_label in VALIDATION_SPECS:
        lines.append(f"### {validation_label}")
        lines.append("")
        for model, model_label, _ in MODEL_SPECS:
            patient = primary[
                (primary.validation == validation) & (primary.model == model)
            ]
            summary = inference[
                (inference.regularization == "fixed")
                & (inference.comparison == "model_vs_train_prior_null")
                & (inference.validation == validation)
                & (inference.model == model)
            ].iloc[0]
            lines.append(
                f"- {model_label}: median predictive information={summary.patient_median:+.3f} bits/trial "
                f"(bootstrap 95% CI {summary.bootstrap_median_ci_low:+.3f} to {summary.bootstrap_median_ci_high:+.3f}); "
                f"median AUC={patient.auc.median():.3f}; {int(summary.n_positive)}/5 patients above zero; "
                f"one-sided exact sign P={summary.exact_sign_p_greater:.4f}."
            )
        gain = inference[
            (inference.regularization == "fixed")
            & (inference.comparison == "population_minus_single_unit")
            & (inference.validation == validation)
        ].iloc[0]
        lines.append(
            f"- Population minus single unit: median ΔI={gain.patient_median:+.3f} bits/trial; "
            f"{int(gain.n_positive)}/5 patients positive; one-sided exact sign P={gain.exact_sign_p_greater:.4f}."
        )
        lines.append("")

    if "nested" in set(patient_values.regularization):
        nested = patient_values[patient_values.regularization == "nested"]
        lines.extend(
            [
                "## Nested-regularization sensitivity",
                "",
            ]
        )
        for validation, validation_label in VALIDATION_SPECS:
            single = inference[
                (inference.regularization == "nested")
                & (inference.comparison == "model_vs_train_prior_null")
                & (inference.validation == validation)
                & (inference.model == "single_unit")
            ].iloc[0]
            population = inference[
                (inference.regularization == "nested")
                & (inference.comparison == "model_vs_train_prior_null")
                & (inference.validation == validation)
                & (inference.model == "top5_population")
            ].iloc[0]
            gain = inference[
                (inference.regularization == "nested")
                & (inference.comparison == "population_minus_single_unit")
                & (inference.validation == validation)
            ].iloc[0]
            nested_auc = nested[nested.validation == validation].groupby("model").auc.median()
            lines.append(
                f"- {validation_label}: single unit {single.patient_median:+.3f} bits/trial "
                f"(AUC={nested_auc['single_unit']:.3f}); population {population.patient_median:+.3f} "
                f"bits/trial (AUC={nested_auc['top5_population']:.3f}); population gain "
                f"{gain.patient_median:+.3f} bits/trial."
            )
        lines.extend(
            [
                "",
                "The held-pair positive information value is therefore sensitive to the regularization strategy: inner tuning on these 16-18-trial training sets often selected stronger shrinkage and did not retain a positive held-pair median information value. The fixed-model panel should be presented as the designated analysis, with this calibration sensitivity disclosed rather than as a universally robust result.",
                "",
            ]
        )

    interaction = inference[
        (inference.regularization == "fixed")
        & (inference.comparison == "validation_by_model_gain_interaction")
    ].iloc[0]
    lines.extend(
        [
            "## Interpretation",
            "",
            "The within-material and held-out-sentence comparisons are reported together so that a negative held-out single-unit value cannot be misread as absence of single-unit information. The scientifically relevant question is whether information survives the stronger sentence-pair generalization test and whether the compact population retains more of it.",
            "",
            f"The validation-by-model gain contrast was {interaction.patient_median:+.3f} bits/trial "
            f"({int(interaction.n_positive)}/5 patients positive; one-sided exact sign P={interaction.exact_sign_p_greater:.4f}). "
            "With only five patients, direction and effect size should be emphasized alongside the exact patient-level P value.",
            "",
            "This analysis measures construction-linked predictive information. Active/passive construction, function-word identity, local context, and associated acoustics covary in the present stimuli, so the result does not isolate an abstract syntax code.",
            "",
            "## Outputs",
            "",
            *[f"- {kind}: `{path}`" for kind, path in outputs.items()],
            f"- metadata: `{metadata_path}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    regularizations = tuple(args.regularizations.split(","))
    candidate_cs = tuple(float(value) for value in args.candidate_cs.split(","))
    run_values, patient_values, trial_values, fold_values, inference = analyze_dataset(
        cv_repeats=args.cv_repeats,
        random_state=args.random_state,
        candidate_cs=candidate_cs,
        fixed_c=args.fixed_c,
        regularizations=regularizations,
    )
    table_paths = {
        "run_values": output_dir / "run_values.csv",
        "patient_values": output_dir / "patient_values.csv",
        "trial_predictions": output_dir / "trial_predictions.csv",
        "outer_fold_audit": output_dir / "outer_fold_audit.csv",
        "patient_inference": output_dir / "patient_inference.csv",
    }
    run_values.to_csv(table_paths["run_values"], index=False)
    patient_values.to_csv(table_paths["patient_values"], index=False)
    trial_values.to_csv(table_paths["trial_predictions"], index=False)
    fold_values.to_csv(table_paths["outer_fold_audit"], index=False)
    inference.to_csv(table_paths["patient_inference"], index=False)
    outputs = draw_figure(
        patient_values,
        inference,
        output_dir=output_dir,
        primary_regularization="fixed",
    )
    metadata: dict[str, object] = {
        "analysis": "single-unit versus top-5-population predictive information",
        "status": "exploratory follow-up to Fig. 2",
        "metric": "mean log2[p_model(y)/p_training_prior_null(y)] in held-out trials",
        "primary_regularization": "fixed common C=0.1",
        "sensitivity_regularization": "inner cross-validation maximizing predictive information",
        "candidate_cs": list(candidate_cs),
        "cv_repeats_all_sentences": args.cv_repeats,
        "n_patients": int(patient_values.subject.nunique()),
        "n_runs": int(run_values[["subject", "run"]].drop_duplicates().shape[0]),
        "n_sentence_pairs": 10,
        "models": {
            "single_unit": "top absolute training-fold t-score",
            "top5_population": "top five absolute training-fold t-scores",
        },
        "data_scope": "artifact-unfiltered Kilosort good units, matching the existing sentence-pair Fig. 2 analysis",
        "aggregation": "out-of-fold probabilities averaged by original trial; runs median-aggregated within patient",
        "independent_inferential_unit": "patient",
        "source_data": {key: str(path) for key, path in table_paths.items()},
        "figure_outputs": outputs,
        "interpretation_boundary": (
            "Construction, function-word identity, local context, and associated acoustics covary; "
            "the metric quantifies construction-linked predictive information, not abstract syntax in isolation."
        ),
    }
    metadata_path = output_dir / "metadata.json"
    metadata["metadata_json"] = str(metadata_path)
    metadata["report"] = str(report_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        patient_values,
        inference,
        report_path=report_path,
        outputs=outputs,
        metadata_path=metadata_path,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=20260903)
    parser.add_argument("--fixed-c", type=float, default=0.1)
    parser.add_argument(
        "--candidate-cs",
        default="0.01,0.03,0.1,0.3,1,3",
    )
    parser.add_argument("--regularizations", default="fixed,nested")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_analysis(parse_args()), ensure_ascii=False, indent=2))
