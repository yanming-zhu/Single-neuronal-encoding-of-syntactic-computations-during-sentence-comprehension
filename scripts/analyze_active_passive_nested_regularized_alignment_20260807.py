#!/usr/bin/env python3
"""Nested comparison of three active/passive latent-alignment estimators.

The analysis is deliberately narrow and post-result exploratory.  It freezes the
target-noun 0--800 ms response, rank 4, held-pair retrieval margin, and the
patient as the inferential unit.  Three estimators share the same outer
leave-two-sentence-pairs-out folds:

1. the previously reported orthogonal Procrustes estimator;
2. Procrustes after training-only baseline-noise whitening;
3. two-view PLS with training-only spectral shrinkage.

Whitening and PLS regularization strengths are selected by an inner
leave-two-pairs-out loop.  Patient-level exact sign flips are single-step
    studentized max-T corrected across the three estimators.  Runs and folds
    remain nested within patients.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_active_passive_geometry_transform_20260806 as base
import analyze_active_passive_manifold_geometry_20260806 as manifold
import analyze_active_passive_shared_latent_geometry_20260806 as shared


ROOT = base.ROOT
DATA_ROOT = base.DATA_ROOT
DEFAULT_REPORT_DIR = ROOT / "reports/active_passive_nested_regularized_alignment_20260807"
DEFAULT_DATA_DIR = DATA_ROOT / "combined_active_passive_nested_regularized_alignment_20260807"

RANK = 4
PRE_PCA_RANK = 8
BASELINE_START_MS = -500
BASELINE_STOP_MS = 0
BASELINE_BIN_MS = 20
METHODS = (
    "original_procrustes",
    "noise_whitened_procrustes",
    "regularized_pls",
)
METHOD_LABELS = {
    "original_procrustes": "Original Procrustes",
    "noise_whitened_procrustes": "Noise-whitened Procrustes",
    "regularized_pls": "Regularized PLS",
}
METHOD_SHORT_LABELS = {
    "original_procrustes": "Original\nProcrustes",
    "noise_whitened_procrustes": "Noise-whitened\nProcrustes",
    "regularized_pls": "Regularized\nPLS",
}
METHOD_COLORS = {
    "original_procrustes": base.OKABE_ITO["purple"],
    "noise_whitened_procrustes": base.OKABE_ITO["blue"],
    "regularized_pls": base.OKABE_ITO["green"],
}
WHITEN_SHRINKAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
PLS_SHRINKAGES = (0.0, 0.1, 0.3, 1.0, 3.0)


def pair_mask(meta: pd.DataFrame, pair_ids: Iterable[int]) -> np.ndarray:
    return meta["pair_id"].astype(int).isin([int(value) for value in pair_ids]).to_numpy()


def load_baseline_residual_tensor(
    run: base.RunData,
    run_metrics: pd.DataFrame,
    unit_set: str,
) -> dict[str, np.ndarray]:
    """Return 20-ms baseline residuals aligned to the run's 20 noun items.

    The returned arrays have shape item x baseline-bin x unit.  Every bin is
    centered by its own trial's -500--0 ms mean, so the covariance represents
    within-baseline cofluctuation rather than across-sentence baseline level.
    """

    archive_path = base.unit_archive_path(run.patient, run.run, unit_set)
    with np.load(archive_path, allow_pickle=True) as archive:
        firing_rate = np.asarray(archive["firing_rate_hz"], dtype=float)
        archive_unit_ids = np.asarray(archive["unit_ids"], dtype=int)
    unit_lookup = {int(unit_id): index for index, unit_id in enumerate(archive_unit_ids)}
    unit_indices = [unit_lookup[int(unit_id)] for unit_id in run.unit_ids]
    firing_rate = firing_rate[unit_indices]

    event_rows = run_metrics.drop_duplicates(
        ["voice", "pair_id", "thematic_role", "noun", "trial_index"]
    )[
        ["voice", "pair_id", "thematic_role", "noun", "trial_index", "event_bin_1000Hz"]
    ]
    n_bins = (BASELINE_STOP_MS - BASELINE_START_MS) // BASELINE_BIN_MS
    output = {
        voice: np.full((len(run.item_meta), n_bins, len(run.unit_ids)), np.nan, dtype=float)
        for voice in base.VOICES
    }
    for voice in base.VOICES:
        for item_index, item in run.item_meta.iterrows():
            row = event_rows[
                (event_rows["voice"] == voice)
                & (event_rows["pair_id"] == int(item["pair_id"]))
                & (event_rows["thematic_role"] == str(item["thematic_role"]))
                & (event_rows["noun"] == str(item["noun"]))
            ]
            if len(row) != 1:
                raise RuntimeError(
                    f"Expected one event for {run.run} {voice} pair={item['pair_id']} "
                    f"role={item['thematic_role']}; found {len(row)}"
                )
            event_bin = int(row.iloc[0]["event_bin_1000Hz"])
            start = event_bin + BASELINE_START_MS
            stop = event_bin + BASELINE_STOP_MS
            if start < 0 or stop > firing_rate.shape[1]:
                raise RuntimeError(f"Baseline out of bounds for {run.run} at {event_bin}")
            baseline = firing_rate[:, start:stop]
            baseline = baseline[:, : n_bins * BASELINE_BIN_MS]
            binned = baseline.reshape(
                len(run.unit_ids), n_bins, BASELINE_BIN_MS
            ).mean(axis=2).T
            output[voice][int(item_index)] = binned - binned.mean(axis=0, keepdims=True)
    if not all(np.isfinite(values).all() for values in output.values()):
        raise RuntimeError(f"Non-finite baseline residual tensor for {run.run}")
    return output


def covariance_whitener(samples: np.ndarray, shrinkage: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    samples = samples - samples.mean(axis=0, keepdims=True)
    dimension = samples.shape[1]
    covariance = samples.T @ samples / max(1, len(samples) - 1)
    scale = float(np.trace(covariance) / max(1, dimension))
    if not np.isfinite(scale) or scale <= 0:
        return np.eye(dimension)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(dimension)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, scale * 1e-6)
    return (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T


def fit_original_procrustes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    return_scores: bool = False,
) -> dict[str, object]:
    active_scores, passive_scores, _ = manifold.fit_procrustes_latent(
        x_train,
        y_train,
        x_test,
        y_test,
        train_meta,
        test_meta,
        RANK,
    )
    result: dict[str, object] = shared.retrieval_metrics(
        active_scores, passive_scores, test_meta
    )
    if return_scores:
        result["active_scores"] = active_scores
        result["passive_scores"] = passive_scores
    return result


def fit_noise_whitened_procrustes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    baseline_x_train: np.ndarray,
    baseline_y_train: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    shrinkage: float,
    return_scores: bool = False,
) -> dict[str, object]:
    xtr, ytr, xte, yte, unit_sd = shared.role_center_pair(
        x_train, y_train, x_test, y_test, train_meta, test_meta
    )
    xp_tr, xp_te, pc_x = shared.pca_project(xtr, xte, RANK)
    yp_tr, yp_te, pc_y = shared.pca_project(ytr, yte, RANK)

    baseline_x_scores = (baseline_x_train / unit_sd[None, None, :]).reshape(
        -1, len(unit_sd)
    ) @ pc_x
    baseline_y_scores = (baseline_y_train / unit_sd[None, None, :]).reshape(
        -1, len(unit_sd)
    ) @ pc_y
    whiten_x = covariance_whitener(baseline_x_scores, shrinkage)
    whiten_y = covariance_whitener(baseline_y_scores, shrinkage)
    xp_tr = xp_tr @ whiten_x
    xp_te = xp_te @ whiten_x
    yp_tr = yp_tr @ whiten_y
    yp_te = yp_te @ whiten_y

    mean_x = xp_tr.mean(axis=0, keepdims=True)
    mean_y = yp_tr.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(
        (yp_tr - mean_y).T @ (xp_tr - mean_x), full_matrices=False
    )
    rotation = u @ vt
    active_scores = xp_te - mean_x
    passive_scores = (yp_te - mean_y) @ rotation
    result: dict[str, object] = shared.retrieval_metrics(
        active_scores, passive_scores, test_meta
    )
    if return_scores:
        result["active_scores"] = active_scores
        result["passive_scores"] = passive_scores
    return result


def fit_regularized_pls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    shrinkage: float,
    return_scores: bool = False,
) -> dict[str, object]:
    """Two-view PLS with shrinkage of weak cross-covariance components."""

    xtr, ytr, xte, yte, _ = shared.role_center_pair(
        x_train, y_train, x_test, y_test, train_meta, test_meta
    )
    available = min(PRE_PCA_RANK, len(xtr) - 2, xtr.shape[1], ytr.shape[1])
    xp_tr, xp_te, _ = shared.pca_project(xtr, xte, available)
    yp_tr, yp_te, _ = shared.pca_project(ytr, yte, available)
    cross = xp_tr.T @ yp_tr / max(1, len(xp_tr) - 1)
    u, singular, vt = np.linalg.svd(cross, full_matrices=False)
    actual_rank = min(RANK, len(singular))
    u = u[:, :actual_rank]
    v = vt.T[:, :actual_rank]
    singular = singular[:actual_rank]
    active_train_scores = xp_tr @ u
    passive_train_scores = yp_tr @ v
    active_scores = xp_te @ u
    passive_scores = yp_te @ v

    sd_x = active_train_scores.std(axis=0, ddof=1)
    sd_y = passive_train_scores.std(axis=0, ddof=1)
    sd_x[~np.isfinite(sd_x) | (sd_x == 0)] = 1.0
    sd_y[~np.isfinite(sd_y) | (sd_y == 0)] = 1.0
    spectral_scale = max(float(singular[0]), 1e-12)
    if np.isclose(shrinkage, 0.0):
        shrink_factor = np.ones_like(singular)
    else:
        shrink_factor = singular / (singular + shrinkage * spectral_scale)
    weights = np.sqrt(np.maximum(singular, 1e-12)) * shrink_factor
    norm = np.linalg.norm(weights)
    weights = weights / norm if norm > 0 else np.ones(actual_rank) / np.sqrt(actual_rank)
    active_scores = active_scores / sd_x * weights
    passive_scores = passive_scores / sd_y * weights
    result: dict[str, object] = shared.retrieval_metrics(
        active_scores, passive_scores, test_meta
    )
    result["singular_values"] = singular
    if return_scores:
        result["active_scores"] = active_scores
        result["passive_scores"] = passive_scores
    return result


def score_method(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    baseline_x_train: np.ndarray,
    baseline_y_train: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    parameter: float | None,
    return_scores: bool = False,
) -> dict[str, object]:
    if method == "original_procrustes":
        return fit_original_procrustes(
            x_train, y_train, x_test, y_test, train_meta, test_meta, return_scores
        )
    if method == "noise_whitened_procrustes":
        if parameter is None:
            raise ValueError("Noise-whitened Procrustes requires shrinkage")
        return fit_noise_whitened_procrustes(
            x_train,
            y_train,
            x_test,
            y_test,
            baseline_x_train,
            baseline_y_train,
            train_meta,
            test_meta,
            parameter,
            return_scores,
        )
    if method == "regularized_pls":
        if parameter is None:
            raise ValueError("Regularized PLS requires shrinkage")
        return fit_regularized_pls(
            x_train,
            y_train,
            x_test,
            y_test,
            train_meta,
            test_meta,
            parameter,
            return_scores,
        )
    raise ValueError(method)


def tune_method(
    method: str,
    x_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    baseline_x_outer_train: np.ndarray,
    baseline_y_outer_train: np.ndarray,
    outer_train_meta: pd.DataFrame,
) -> tuple[float, list[dict[str, float]]]:
    grid = (
        WHITEN_SHRINKAGES
        if method == "noise_whitened_procrustes"
        else PLS_SHRINKAGES
    )
    grid_rows: list[dict[str, float]] = []
    for parameter in grid:
        inner_margins: list[float] = []
        inner_accuracies: list[float] = []
        for _, inner_train, inner_test in base.outer_folds(outer_train_meta):
            inner_train_meta = outer_train_meta.loc[inner_train].reset_index(drop=True)
            inner_test_meta = outer_train_meta.loc[inner_test].reset_index(drop=True)
            result = score_method(
                method,
                x_outer_train[inner_train],
                y_outer_train[inner_train],
                x_outer_train[inner_test],
                y_outer_train[inner_test],
                baseline_x_outer_train[inner_train],
                baseline_y_outer_train[inner_train],
                inner_train_meta,
                inner_test_meta,
                float(parameter),
            )
            inner_margins.append(float(result["retrieval_margin"]))
            inner_accuracies.append(float(result["two_by_two_accuracy"]))
        grid_rows.append(
            {
                "parameter": float(parameter),
                "inner_mean_retrieval_margin": float(np.mean(inner_margins)),
                "inner_mean_accuracy": float(np.mean(inner_accuracies)),
                "n_inner_folds": float(len(inner_margins)),
            }
        )
    # Margin is the frozen primary selection metric.  Exact ties prefer stronger
    # regularization (larger value), which is fixed before inspecting outer data.
    selected = max(
        grid_rows,
        key=lambda row: (row["inner_mean_retrieval_margin"], row["parameter"]),
    )
    return float(selected["parameter"]), grid_rows


def nested_cv_run(
    run: base.RunData,
    baseline: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for held_pairs, outer_train, outer_test in base.outer_folds(run.item_meta):
        train_meta = run.item_meta.loc[outer_train].reset_index(drop=True)
        test_meta = run.item_meta.loc[outer_test].reset_index(drop=True)
        x_train = run.active[outer_train]
        y_train = run.passive[outer_train]
        x_test = run.active[outer_test]
        y_test = run.passive[outer_test]
        baseline_x_train = baseline["active"][outer_train]
        baseline_y_train = baseline["passive"][outer_train]

        baseline_result = score_method(
            "original_procrustes",
            x_train,
            y_train,
            x_test,
            y_test,
            baseline_x_train,
            baseline_y_train,
            train_meta,
            test_meta,
            None,
        )
        fold_rows.append(
            {
                "patient": run.patient,
                "run": run.run,
                "held_pair_1": int(held_pairs[0]),
                "held_pair_2": int(held_pairs[1]),
                "method": "original_procrustes",
                "selected_parameter": np.nan,
                **{key: value for key, value in baseline_result.items() if np.isscalar(value)},
            }
        )

        for method in METHODS[1:]:
            selected_parameter, grid_rows = tune_method(
                method,
                x_train,
                y_train,
                baseline_x_train,
                baseline_y_train,
                train_meta,
            )
            for row in grid_rows:
                tuning_rows.append(
                    {
                        "patient": run.patient,
                        "run": run.run,
                        "held_pair_1": int(held_pairs[0]),
                        "held_pair_2": int(held_pairs[1]),
                        "method": method,
                        **row,
                        "selected": bool(np.isclose(row["parameter"], selected_parameter)),
                    }
                )
            selection_rows.append(
                {
                    "patient": run.patient,
                    "run": run.run,
                    "held_pair_1": int(held_pairs[0]),
                    "held_pair_2": int(held_pairs[1]),
                    "method": method,
                    "selected_parameter": selected_parameter,
                }
            )
            result = score_method(
                method,
                x_train,
                y_train,
                x_test,
                y_test,
                baseline_x_train,
                baseline_y_train,
                train_meta,
                test_meta,
                selected_parameter,
            )
            fold_rows.append(
                {
                    "patient": run.patient,
                    "run": run.run,
                    "held_pair_1": int(held_pairs[0]),
                    "held_pair_2": int(held_pairs[1]),
                    "method": method,
                    "selected_parameter": selected_parameter,
                    **{key: value for key, value in result.items() if np.isscalar(value)},
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(tuning_rows), pd.DataFrame(selection_rows)


def exact_method_maxstat(
    patient_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = patient_summary.pivot(
        index="patient", columns="method", values="retrieval_margin"
    ).reindex(columns=list(METHODS))
    values = pivot.to_numpy(dtype=float)
    observed = values.mean(axis=0)
    observed_t = np.asarray(
        [studentized_one_sample(column) for column in values.T], dtype=float
    )
    null_rows: list[dict[str, object]] = []
    null_max: list[float] = []
    for sign_index, signs in enumerate(
        itertools.product((-1.0, 1.0), repeat=len(pivot))
    ):
        signed_values = values * np.asarray(signs, dtype=float)[:, None]
        signed_means = signed_values.mean(axis=0)
        signed_t = np.asarray(
            [studentized_one_sample(column) for column in signed_values.T], dtype=float
        )
        maximum = float(np.max(signed_t))
        null_max.append(maximum)
        null_rows.append(
            {
                "sign_index": sign_index,
                "signs": json.dumps([int(sign) for sign in signs]),
                "max_studentized_t": maximum,
                **{
                    f"mean_{method}": float(signed_means[index])
                    for index, method in enumerate(METHODS)
                },
                **{
                    f"t_{method}": float(signed_t[index])
                    for index, method in enumerate(METHODS)
                },
            }
        )
    null_max_array = np.asarray(null_max)
    rows = []
    for index, method in enumerate(METHODS):
        method_values = values[:, index]
        sign = base.exact_sign_flip(method_values, 0.0)
        ci = base.bootstrap_patient_ci(
            method_values,
            10000,
            base.stable_seed("nested-regularized-alignment", method),
        )
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "n_patients": len(method_values),
                "patient_mean_retrieval_margin": float(observed[index]),
                "observed_studentized_t": float(observed_t[index]),
                "patient_median_retrieval_margin": float(np.median(method_values)),
                "bootstrap_ci_low": ci[0],
                "bootstrap_ci_high": ci[1],
                "positive_patients": int(np.sum(method_values > 0)),
                "exact_signflip_p_greater_raw": sign["p_greater"],
                "exact_signflip_p_two_sided_raw": sign["p_two_sided"],
                "exact_maxstat_p_fwer_greater": float(
                    np.mean(null_max_array >= observed_t[index] - 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def studentized_one_sample(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    standard_deviation = float(np.std(values, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        mean = float(np.mean(values))
        return 0.0 if np.isclose(mean, 0.0) else float(np.sign(mean) * np.inf)
    return float(np.mean(values) / (standard_deviation / np.sqrt(len(values))))


def method_contrasts(
    patient_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot = patient_summary.pivot(
        index="patient", columns="method", values="retrieval_margin"
    )
    contrasts = {
        "noise_whitened_minus_original": (
            pivot["noise_whitened_procrustes"] - pivot["original_procrustes"]
        ),
        "regularized_pls_minus_original": (
            pivot["regularized_pls"] - pivot["original_procrustes"]
        ),
    }
    patient_rows = []
    for contrast, values in contrasts.items():
        for patient, value in values.items():
            patient_rows.append(
                {"patient": patient, "contrast": contrast, "margin_difference": value}
            )
    observed = np.asarray([values.mean() for values in contrasts.values()], dtype=float)
    matrix = np.column_stack([values.to_numpy(dtype=float) for values in contrasts.values()])
    observed_t = np.asarray(
        [studentized_one_sample(column) for column in matrix.T], dtype=float
    )
    null_rows = []
    null_max = []
    for sign_index, signs in enumerate(
        itertools.product((-1.0, 1.0), repeat=len(pivot))
    ):
        signed_values = matrix * np.asarray(signs)[:, None]
        means = signed_values.mean(axis=0)
        signed_t = np.asarray(
            [studentized_one_sample(column) for column in signed_values.T], dtype=float
        )
        maximum = float(np.max(signed_t))
        null_max.append(maximum)
        null_rows.append(
            {
                "sign_index": sign_index,
                "signs": json.dumps([int(sign) for sign in signs]),
                "max_studentized_t": maximum,
                **{
                    f"mean_{name}": float(means[index])
                    for index, name in enumerate(contrasts)
                },
                **{
                    f"t_{name}": float(signed_t[index])
                    for index, name in enumerate(contrasts)
                },
            }
        )
    null_max = np.asarray(null_max)
    inference_rows = []
    for index, (name, values) in enumerate(contrasts.items()):
        sign = base.exact_sign_flip(values, 0.0)
        inference_rows.append(
            {
                "contrast": name,
                "patient_mean_margin_difference": float(observed[index]),
                "observed_studentized_t": float(observed_t[index]),
                "positive_patients": int(np.sum(values > 0)),
                "exact_signflip_p_greater_raw": sign["p_greater"],
                "exact_signflip_p_two_sided_raw": sign["p_two_sided"],
                "exact_maxstat_p_fwer_greater": float(
                    np.mean(null_max >= observed_t[index] - 1e-12)
                ),
            }
        )
    return pd.DataFrame(patient_rows), pd.DataFrame(inference_rows), pd.DataFrame(null_rows)


def aggregate_folds(
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_summary = folds.groupby(["patient", "run", "method"], as_index=False).agg(
        retrieval_margin=("retrieval_margin", "mean"),
        two_by_two_accuracy=("two_by_two_accuracy", "mean"),
        matched_cosine=("matched_cosine", "mean"),
        matched_angle_deg=("matched_angle_deg", "mean"),
    )
    patient_summary = run_summary.groupby(["patient", "method"], as_index=False).mean(
        numeric_only=True
    )
    return run_summary, patient_summary


def validate_baseline_reproduction(
    folds: pd.DataFrame,
    unit_set: str,
) -> dict[str, object]:
    prior_dir = manifold.DEFAULT_DATA_DIR
    if unit_set == "usable_lenient":
        prior_dir = prior_dir / "sensitivity_usable_lenient"
    prior = pd.read_csv(prior_dir / "static_model_cv_folds.csv")
    prior = prior[(prior["model"] == "Procrustes") & (prior["rank"] == RANK)].copy()
    current = folds[folds["method"] == "original_procrustes"].copy()
    keys = ["patient", "run", "held_pair_1", "held_pair_2"]
    metrics = [
        "retrieval_margin",
        "two_by_two_accuracy",
        "matched_cosine",
        "matched_angle_deg",
    ]
    merged = current.merge(prior[keys + metrics], on=keys, suffixes=("_new", "_prior"))
    differences = {
        metric: float(
            np.max(np.abs(merged[f"{metric}_new"] - merged[f"{metric}_prior"]))
        )
        for metric in metrics
    }
    passed = len(merged) == len(current) == len(prior) and all(
        value <= 1e-10 for value in differences.values()
    )
    return {
        "prior_path": str(prior_dir / "static_model_cv_folds.csv"),
        "n_current_folds": int(len(current)),
        "n_prior_folds": int(len(prior)),
        "n_matched_folds": int(len(merged)),
        "maximum_absolute_differences": differences,
        "passed": bool(passed),
    }


def leakage_audit(
    run: base.RunData,
    baseline: dict[str, np.ndarray],
) -> dict[str, object]:
    held_pairs, train, test = next(iter(base.outer_folds(run.item_meta)))
    train_meta = run.item_meta.loc[train].reset_index(drop=True)
    test_meta = run.item_meta.loc[test].reset_index(drop=True)
    rows = []
    for method, parameter in (
        ("noise_whitened_procrustes", 0.5),
        ("regularized_pls", 0.3),
    ):
        real = score_method(
            method,
            run.active[train],
            run.passive[train],
            run.active[test],
            run.passive[test],
            baseline["active"][train],
            baseline["passive"][train],
            train_meta,
            test_meta,
            parameter,
            return_scores=True,
        )
        altered = score_method(
            method,
            run.active[train],
            run.passive[train],
            run.active[test],
            run.passive[test] * 1_000_000.0,
            baseline["active"][train],
            baseline["passive"][train],
            train_meta,
            test_meta,
            parameter,
            return_scores=True,
        )
        unchanged = bool(
            np.allclose(
                np.asarray(real["active_scores"]),
                np.asarray(altered["active_scores"]),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        rows.append(
            {
                "method": method,
                "active_test_projection_unchanged_after_passive_truth_alteration": unchanged,
            }
        )
    return {
        "run": run.run,
        "held_pairs": [int(value) for value in held_pairs],
        "test_passive_truth_multiplier": 1_000_000,
        "method_checks": rows,
        "passed": bool(
            all(
                row[
                    "active_test_projection_unchanged_after_passive_truth_alteration"
                ]
                for row in rows
            )
        ),
    }


def make_figure(
    report_dir: Path,
    unit_set: str,
    patient: pd.DataFrame,
    group: pd.DataFrame,
    contrasts_patient: pd.DataFrame,
    selections: pd.DataFrame,
    maxstat_null: pd.DataFrame,
) -> None:
    base.configure_style()
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.2), constrained_layout=True)

    pivot = patient.pivot(index="patient", columns="method", values="retrieval_margin")
    ax = axes[0, 0]
    for _, row in pivot.iterrows():
        ax.plot(
            range(len(METHODS)),
            row.reindex(METHODS),
            color=base.OKABE_ITO["light_gray"],
            lw=0.9,
            zorder=1,
        )
    offsets = np.linspace(-0.12, 0.12, len(pivot))
    for method_index, method in enumerate(METHODS):
        values = pivot[method]
        ax.scatter(
            method_index + offsets,
            values,
            s=34,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.scatter(
            method_index,
            values.mean(),
            marker="_",
            s=210,
            color="black",
            linewidth=2.0,
            zorder=4,
        )
        stat = group[group["method"] == method].iloc[0]
        ax.text(
            method_index,
            0.985,
            f"raw {stat.exact_signflip_p_greater_raw:.4f}\nFWER {stat.exact_maxstat_p_fwer_greater:.4f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.5,
        )
    ax.axhline(0, color=base.OKABE_ITO["gray"], lw=0.9, ls="--")
    lower, upper = ax.get_ylim()
    ax.set_ylim(lower, upper + 0.12 * (upper - lower))
    ax.set_xticks(range(len(METHODS)), [METHOD_SHORT_LABELS[m] for m in METHODS])
    ax.set_ylabel("Held-pair retrieval margin")
    ax.set_title("A  Patient-level outer-CV effects")

    ax = axes[0, 1]
    contrast_order = [
        "noise_whitened_minus_original",
        "regularized_pls_minus_original",
    ]
    contrast_labels = ["Whitened − original", "Reg. PLS − original"]
    contrast_pivot = contrasts_patient.pivot(
        index="patient", columns="contrast", values="margin_difference"
    )
    for index, contrast in enumerate(contrast_order):
        values = contrast_pivot[contrast]
        ax.scatter(
            index + offsets,
            values,
            s=34,
            color=METHOD_COLORS[METHODS[index + 1]],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.scatter(index, values.mean(), marker="_", s=210, color="black", linewidth=2)
    ax.axhline(0, color=base.OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set_xticks(range(2), contrast_labels)
    ax.set_ylabel("Within-patient margin difference")
    ax.set_title("B  Improvement over original")

    ax = axes[1, 0]
    all_parameters = sorted(selections["selected_parameter"].unique())
    width = 0.36
    for method_index, method in enumerate(METHODS[1:]):
        sub = selections[selections["method"] == method]
        frequencies = [
            float(np.mean(np.isclose(sub["selected_parameter"], value)))
            for value in all_parameters
        ]
        ax.bar(
            np.arange(len(all_parameters)) + (method_index - 0.5) * width,
            frequencies,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(np.arange(len(all_parameters)), [f"{value:g}" for value in all_parameters])
    ax.set_xlabel("Inner-CV selected shrinkage")
    ax.set_ylabel("Fraction of outer folds")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("C  Hyperparameter stability")

    ax = axes[1, 1]
    null_values = maxstat_null["max_studentized_t"].to_numpy(dtype=float)
    ax.hist(
        null_values,
        bins=min(14, len(np.unique(null_values))),
        color=base.OKABE_ITO["light_gray"],
        edgecolor="white",
    )
    for method in METHODS:
        observed = float(group.loc[group["method"] == method, "observed_studentized_t"].iloc[0])
        ax.axvline(
            observed,
            color=METHOD_COLORS[method],
            lw=1.8,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Maximum studentized patient statistic across 3 methods")
    ax.set_ylabel("Exact sign patterns")
    ax.legend(frameon=False, fontsize=6.4)
    ax.set_title("D  Exact patient max-stat null")

    fig.suptitle(
        f"Nested regularized latent-alignment benchmark ({unit_set})\n"
        "Target noun, 0–800 ms, rank 4; each point is one patient",
        fontsize=10,
    )
    stem = figure_dir / "Fig_6_nested_regularized_alignment_benchmark"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(
    report_dir: Path,
    unit_set: str,
    group: pd.DataFrame,
    contrasts: pd.DataFrame,
    selections: pd.DataFrame,
    validation: dict[str, object],
) -> None:
    table_lines = [
        "| 方法 | 患者均值margin | 正向患者 | 原始单侧P | 三方法max-stat P |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = group[group["method"] == method].iloc[0]
        table_lines.append(
            f"| {METHOD_LABELS[method]} | {row.patient_mean_retrieval_margin:.5f} | "
            f"{int(row.positive_patients)}/5 | {row.exact_signflip_p_greater_raw:.4f} | "
            f"{row.exact_maxstat_p_fwer_greater:.4f} |"
        )
    contrast_lines = [
        "| 对比 | 患者均值差 | 正向患者 | 原始单侧P | 两对比max-stat P |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in contrasts.itertuples(index=False):
        contrast_lines.append(
            f"| {row.contrast} | {row.patient_mean_margin_difference:.5f} | "
            f"{int(row.positive_patients)}/5 | {row.exact_signflip_p_greater_raw:.4f} | "
            f"{row.exact_maxstat_p_fwer_greater:.4f} |"
        )
    selection_lines = []
    for method in METHODS[1:]:
        counts = (
            selections.loc[selections["method"] == method, "selected_parameter"]
            .value_counts()
            .sort_index()
        )
        selection_lines.append(
            f"- {METHOD_LABELS[method]}："
            + "，".join(f"{parameter:g}={count}/{counts.sum()}折" for parameter, count in counts.items())
        )
    best = group.sort_values("patient_mean_retrieval_margin", ascending=False).iloc[0]
    any_fwer = bool((group["exact_maxstat_p_fwer_greater"] < 0.05).any())
    lines = [
        f"# 嵌套正则化隐空间对齐分析（{unit_set}）",
        "",
        "## 怎么读",
        "",
        "三种方法使用完全相同的target-noun 0–800 ms、Rank-4及留两组句对外层测试。"
        "白化强度和PLS谱收缩强度只在外层训练数据的内层交叉验证中选择。"
        "每个run先汇总，再在患者内平均；最终独立推断单位为5位患者。",
        "",
        *table_lines,
        "",
        "## 主要结果",
        "",
        f"- 描述性最高的方法为{best.label}，患者均值margin={best.patient_mean_retrieval_margin:.5f}。",
        f"- 三方法患者层max-stat校正后{'至少一种方法达到P<0.05' if any_fwer else '没有方法达到P<0.05'}。",
        "- 新方法相对原始Procrustes的患者内差异如下；该检验回答的是方法是否改善，而不是每种方法本身是否高于零。",
        "",
        *contrast_lines,
        "",
        "## 内层选择",
        "",
        *selection_lines,
        "",
        "## 验证",
        "",
        f"- 原始Rank-4 Procrustes与冻结旧结果逐折复现：{'通过' if validation['baseline_reproduction']['passed'] else '失败'}。",
        f"- 外层测试真值不影响训练后的active投影：{'通过' if validation['leakage_audit']['passed'] else '失败'}。",
        "- 32种患者符号组合全部枚举；三种方法的患者层t统计量同时进入单步studentized max-T校正。",
        "",
        "## 支持与限制",
        "",
        "- 这是看到既有结果后开展的方法学探索；即使校正P<0.05，也需要新增患者或独立数据确认。",
        "- 刺激置换未在本模块重复运行；本模块的主要问题是患者层一致性，并未把句子/折/run当作独立样本。",
        "- Active/passive仍与构式标记、声学、表面位置和句式共线；结果不能证明抽象语态manifold。",
        "- baseline噪声白化使用名词前−500–0 ms的20-ms残差；它是训练期噪声归一化，不是独立生理噪声测量。",
        "",
    ]
    (report_dir / "COMBINED_REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")

    caption = (
        "Fig. 6 | Nested comparison of rank-4 active/passive latent-alignment estimators. "
        "All methods used target-noun responses averaged from 0 to 800 ms and identical "
        "leave-two-sentence-pairs-out outer folds. Noise-covariance shrinkage and PLS "
        "spectral shrinkage were selected using training-only inner folds. Points denote "
        "patients, grey lines connect estimates from the same patient, and horizontal "
        "bars denote patient means. One-sided exhaustive patient sign-flip tests were "
        "single-step studentized max-T corrected across the three methods. This post-result "
        "exploratory analysis does not isolate voice from marker, acoustic, positional, "
        "or sentence-form differences."
    )
    (report_dir / "Fig6_caption_en.txt").write_text(caption, encoding="utf-8")


def run_spec(
    unit_set: str,
    report_dir: Path,
    data_dir: Path,
    artifact_threshold: float | None,
) -> dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "analysis_date": "2026-08-07",
        "status": "post-result exploratory method benchmark",
        "unit_set": unit_set,
        "artifact_threshold": artifact_threshold,
        "anchor": "target-noun onset",
        "response_window_ms": [0, 800],
        "latent_rank": RANK,
        "primary_endpoint": "held-pair retrieval margin",
        "methods": list(METHODS),
        "outer_cv": "leave two complete sentence pairs out (45 folds per run)",
        "inner_cv": "leave two complete sentence pairs out within each outer training set (28 folds)",
        "whitening_grid": list(WHITEN_SHRINKAGES),
        "pls_spectral_shrinkage_grid": list(PLS_SHRINKAGES),
        "selection_metric": "inner mean retrieval margin",
        "tie_breaker": "larger shrinkage",
        "baseline_noise": "-500 to 0 ms, 20-ms bins, within-trial temporal residuals",
        "patient_inference": "patient-equal exhaustive sign flip",
        "multiple_comparison": "single-step studentized patient max-T across three methods",
        "interpretation_limit": "condition confounded with marker/acoustics/position/form; role with animacy/lexical class",
    }
    (report_dir / "analysis_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (data_dir / "analysis_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics, artifact_runs = base.load_filtered_metrics(unit_set, artifact_threshold)
    runs, audit = base.build_run_data(metrics, artifact_runs, unit_set)
    if len({run.patient for run in runs}) != 5:
        raise RuntimeError(f"Expected five patients:\n{audit.to_string(index=False)}")
    audit.to_csv(report_dir / "run_inclusion_audit.csv", index=False)
    audit.to_csv(data_dir / "run_inclusion_audit.csv", index=False)

    fold_frames = []
    tuning_frames = []
    selection_frames = []
    baseline_by_run: dict[str, dict[str, np.ndarray]] = {}
    for run_index, run in enumerate(runs, start=1):
        run_metrics = metrics[(metrics["patient"] == run.patient) & (metrics["run"] == run.run)]
        baseline = load_baseline_residual_tensor(run, run_metrics, unit_set)
        baseline_by_run[run.run] = baseline
        folds, tuning, selections = nested_cv_run(run, baseline)
        fold_frames.append(folds)
        tuning_frames.append(tuning)
        selection_frames.append(selections)
        print(
            f"Nested alignment {unit_set}: run {run_index}/{len(runs)} ({run.run})",
            flush=True,
        )
    folds = pd.concat(fold_frames, ignore_index=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    run_summary, patient_summary = aggregate_folds(folds)
    group, maxstat_null = exact_method_maxstat(patient_summary)
    contrast_patient, contrast_group, contrast_null = method_contrasts(patient_summary)

    baseline_validation = validate_baseline_reproduction(folds, unit_set)
    leakage = leakage_audit(runs[0], baseline_by_run[runs[0].run])
    validation = {
        "baseline_reproduction": baseline_validation,
        "leakage_audit": leakage,
        "n_outer_folds_per_run": 45,
        "expected_total_method_folds": int(len(runs) * 45 * len(METHODS)),
        "observed_total_method_folds": int(len(folds)),
        "all_finite": bool(
            np.isfinite(folds[["retrieval_margin", "two_by_two_accuracy"]].to_numpy()).all()
        ),
    }
    validation["passed"] = bool(
        baseline_validation["passed"]
        and leakage["passed"]
        and validation["expected_total_method_folds"] == validation["observed_total_method_folds"]
        and validation["all_finite"]
    )
    if not validation["passed"]:
        raise RuntimeError(f"Validation failed: {json.dumps(validation, indent=2)}")

    outputs = {
        "outer_cv_folds.csv": folds,
        "inner_cv_grid_scores.csv": tuning,
        "outer_fold_selected_parameters.csv": selections,
        "run_summary.csv": run_summary,
        "patient_summary.csv": patient_summary,
        "group_inference.csv": group,
        "patient_maxstat_null.csv": maxstat_null,
        "method_contrasts_patient.csv": contrast_patient,
        "method_contrast_inference.csv": contrast_group,
        "method_contrast_maxstat_null.csv": contrast_null,
    }
    for filename, frame in outputs.items():
        frame.to_csv(data_dir / filename, index=False)
    (data_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "analysis_contract": contract,
        "run_inclusion": audit.to_dict(orient="records"),
        "group_inference": group.to_dict(orient="records"),
        "method_contrast_inference": contrast_group.to_dict(orient="records"),
        "selection_counts": (
            selections.groupby(["method", "selected_parameter"]).size().rename("count").reset_index().to_dict(orient="records")
        ),
        "validation": validation,
    }
    for path in (data_dir / "summary.json", report_dir / "summary.json"):
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    make_figure(
        report_dir,
        unit_set,
        patient_summary,
        group,
        contrast_patient,
        selections,
        maxstat_null,
    )
    write_report(report_dir, unit_set, group, contrast_group, selections, validation)
    return {
        "unit_set": unit_set,
        "report_dir": str(report_dir),
        "data_dir": str(data_dir),
        "patient_summary": patient_summary,
        "group_inference": group,
        "contrast_inference": contrast_group,
    }


def make_cross_spec_figure(
    report_dir: Path,
    results: list[dict[str, object]],
) -> None:
    base.configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), constrained_layout=True, sharey=True)
    for ax, result in zip(axes, results):
        patient = result["patient_summary"]
        group = result["group_inference"]
        pivot = patient.pivot(index="patient", columns="method", values="retrieval_margin")
        offsets = np.linspace(-0.12, 0.12, len(pivot))
        for _, row in pivot.iterrows():
            ax.plot(range(len(METHODS)), row.reindex(METHODS), color=base.OKABE_ITO["light_gray"], lw=0.9)
        for method_index, method in enumerate(METHODS):
            values = pivot[method]
            ax.scatter(
                method_index + offsets,
                values,
                s=32,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.scatter(method_index, values.mean(), marker="_", s=200, color="black", linewidth=2)
            stat = group[group["method"] == method].iloc[0]
            ax.text(
                method_index,
                0.98,
                f"{stat.exact_maxstat_p_fwer_greater:.3f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.5,
            )
        ax.axhline(0, color=base.OKABE_ITO["gray"], lw=0.9, ls="--")
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper + 0.10 * (upper - lower))
        ax.set_xticks(range(len(METHODS)), [METHOD_SHORT_LABELS[m] for m in METHODS])
        ax.set_title(str(result["unit_set"]))
    axes[0].set_ylabel("Held-pair retrieval margin")
    fig.suptitle(
        "Rank-4 nested alignment across unit specifications\n"
        "Numbers are patient max-stat FWER P values; specifications are not independent",
        fontsize=9.5,
    )
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "Fig_6_combined_unit_specification_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_cross_spec_report(report_dir: Path, results: list[dict[str, object]]) -> None:
    rows = [
        "| unit规格 | 方法 | 患者均值margin | 正向患者 | 原始P | 三方法max-stat P |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        group = result["group_inference"]
        for method in METHODS:
            row = group[group["method"] == method].iloc[0]
            rows.append(
                f"| {result['unit_set']} | {METHOD_LABELS[method]} | "
                f"{row.patient_mean_retrieval_margin:.5f} | {int(row.positive_patients)}/5 | "
                f"{row.exact_signflip_p_greater_raw:.4f} | {row.exact_maxstat_p_fwer_greater:.4f} |"
            )
    lines = [
        "# 三方法嵌套比较：两套unit规格综合结果",
        "",
        *rows,
        "",
        "两套unit规格高度重叠，只用于敏感性分析，不能当作独立重复。"
        "所有P值首先在各规格内对三种方法做患者max-stat校正；跨规格一致性作描述，未用于增加样本量。",
        "",
    ]
    (report_dir / "CROSS_SPEC_COMBINED_REPORT_zh.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=("good_only", "usable_lenient"), default="good_only")
    parser.add_argument("--all-unit-sets", action="store_true")
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_threshold = None if args.artifact_threshold < 0 else args.artifact_threshold
    unit_sets = ("good_only", "usable_lenient") if args.all_unit_sets else (args.unit_set,)
    results = []
    for unit_set in unit_sets:
        report_dir = args.report_dir
        data_dir = args.data_dir
        if unit_set == "usable_lenient" and args.all_unit_sets:
            report_dir = report_dir / "sensitivity_usable_lenient"
            data_dir = data_dir / "sensitivity_usable_lenient"
        results.append(run_spec(unit_set, report_dir, data_dir, artifact_threshold))
    if len(results) == 2:
        make_cross_spec_figure(args.report_dir, results)
        write_cross_spec_report(args.report_dir, results)
    print(json.dumps({
        "status": "complete",
        "reports": [result["report_dir"] for result in results],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
