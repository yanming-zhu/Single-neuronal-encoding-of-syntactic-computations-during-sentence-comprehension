#!/usr/bin/env python3
"""Intrinsic dimension and manifold alignment for matched active/passive nouns.

This post-result exploratory analysis extends the frozen shared-latent analysis
with three explicitly bounded questions:

1. How concentrated are the active, passive, and cross-view spectra?
2. Can a training-only orthogonal Procrustes map retrieve held-out noun pairs,
   and how does it compare with the more flexible two-view PLS model?
3. Does held-out PLS alignment emerge at a fixed active/passive time lag?

Patients, rather than runs, folds, units, or time cells, are the independent
inferential units.  Active/passive remains confounded with marker, acoustics,
surface position, and sentence form; Agent/Patient remains confounded with
animacy and lexical class.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
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
import analyze_active_passive_shared_latent_geometry_20260806 as shared


ROOT = base.ROOT
DATA_ROOT = base.DATA_ROOT
DEFAULT_REPORT_DIR = ROOT / "reports/active_passive_manifold_geometry_20260806"
DEFAULT_DATA_DIR = DATA_ROOT / "combined_active_passive_manifold_geometry_20260806"

RANKS = (2, 4)
PRIMARY_RANK = 2
MAX_SPECTRUM_RANK = 8
WINDOWS_MS = shared.DYNAMIC_WINDOWS_MS
OKABE_ITO = base.OKABE_ITO


def spectrum_statistics(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    centered = np.asarray(matrix, dtype=float) - np.asarray(matrix, dtype=float).mean(
        axis=0, keepdims=True
    )
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    energy = np.square(singular)
    energy = energy[energy > np.finfo(float).eps]
    if energy.size == 0 or energy.sum() <= 0:
        return np.nan, np.asarray([], dtype=float)
    fraction = energy / energy.sum()
    effective_dimension = float(np.square(energy.sum()) / np.square(energy).sum())
    return effective_dimension, fraction


def cross_view_spectrum(
    active: np.ndarray,
    passive: np.ndarray,
    pre_rank: int = MAX_SPECTRUM_RANK,
) -> tuple[float, np.ndarray]:
    available = min(pre_rank, len(active) - 2, active.shape[1], passive.shape[1])
    active_scores, _, _ = shared.pca_project(active, active, available)
    passive_scores, _, _ = shared.pca_project(passive, passive, available)
    singular = np.linalg.svd(
        active_scores.T @ passive_scores / max(1, len(active_scores) - 1),
        compute_uv=False,
        full_matrices=False,
    )
    energy = np.square(singular)
    energy = energy[energy > np.finfo(float).eps]
    if energy.size == 0 or energy.sum() <= 0:
        return np.nan, np.asarray([], dtype=float)
    fraction = energy / energy.sum()
    effective_dimension = float(np.square(energy.sum()) / np.square(energy).sum())
    return effective_dimension, fraction


def dimension_metrics(
    active: np.ndarray,
    passive: np.ndarray,
    meta: pd.DataFrame,
) -> dict[str, tuple[float, np.ndarray]]:
    active_scaled, passive_scaled = shared.role_center_full(active, passive, meta)
    active_dim, active_fraction = spectrum_statistics(active_scaled)
    passive_dim, passive_fraction = spectrum_statistics(passive_scaled)
    shared_dim, shared_fraction = cross_view_spectrum(active_scaled, passive_scaled)
    return {
        "active": (active_dim, active_fraction),
        "passive": (passive_dim, passive_fraction),
        "shared_cross_covariance": (shared_dim, shared_fraction),
    }


def intrinsic_dimension_analysis(
    runs: list[base.RunData],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spectrum_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for run in runs:
        metrics = dimension_metrics(run.active, run.passive, run.item_meta)
        for view, (effective_dimension, fraction) in metrics.items():
            full_rows.append(
                {
                    "patient": run.patient,
                    "run": run.run,
                    "view": view,
                    "effective_dimension": effective_dimension,
                    "variance_rank2": float(np.sum(fraction[:2])),
                    "variance_rank4": float(np.sum(fraction[:4])),
                    "available_spectrum_rank": int(len(fraction)),
                }
            )
            cumulative = np.cumsum(fraction)
            for index, (value, cum) in enumerate(zip(fraction, cumulative), start=1):
                spectrum_rows.append(
                    {
                        "patient": run.patient,
                        "run": run.run,
                        "view": view,
                        "dimension": index,
                        "spectrum_fraction": value,
                        "cumulative_fraction": cum,
                    }
                )

        rng = np.random.default_rng(base.stable_seed("manifold-dimension-splits", run.run))
        for split_index in range(n_splits):
            order = rng.permutation(len(run.unit_ids))
            halves = np.array_split(order, 2)
            for half_index, unit_idx in enumerate(halves):
                split_metrics = dimension_metrics(
                    run.active[:, unit_idx], run.passive[:, unit_idx], run.item_meta
                )
                for view, (effective_dimension, fraction) in split_metrics.items():
                    split_rows.append(
                        {
                            "patient": run.patient,
                            "run": run.run,
                            "split_index": split_index,
                            "half_index": half_index,
                            "n_units": len(unit_idx),
                            "view": view,
                            "effective_dimension": effective_dimension,
                            "variance_rank2": float(np.sum(fraction[:2])),
                            "variance_rank4": float(np.sum(fraction[:4])),
                        }
                    )

    full = pd.DataFrame(full_rows)
    spectrum = pd.DataFrame(spectrum_rows)
    splits = pd.DataFrame(split_rows)
    split_summary = splits.groupby(["patient", "run", "view"], as_index=False).agg(
        split_median_effective_dimension=("effective_dimension", "median"),
        split_ci_low_effective_dimension=("effective_dimension", lambda x: np.quantile(x, 0.025)),
        split_ci_high_effective_dimension=("effective_dimension", lambda x: np.quantile(x, 0.975)),
        split_median_variance_rank2=("variance_rank2", "median"),
        split_median_variance_rank4=("variance_rank4", "median"),
    )
    run_summary = full.merge(split_summary, on=["patient", "run", "view"], how="left")
    return spectrum, splits, run_summary, full


def rank_limited_dimension(
    spectrum: pd.DataFrame,
    max_rank: int = MAX_SPECTRUM_RANK,
) -> pd.DataFrame:
    """Compute comparable PR after truncating every view to the same rank."""
    rows: list[dict[str, object]] = []
    for (patient, run, view), frame in spectrum.groupby(["patient", "run", "view"]):
        values = frame.sort_values("dimension")
        values = values[values["dimension"] <= max_rank]["spectrum_fraction"].to_numpy()
        captured = float(values.sum())
        if captured <= 0:
            effective_dimension = np.nan
        else:
            normalized = values / captured
            effective_dimension = float(1.0 / np.square(normalized).sum())
        rows.append(
            {
                "patient": patient,
                "run": run,
                "view": view,
                "comparison_rank": max_rank,
                "captured_full_spectrum_fraction": captured,
                "rank_limited_effective_dimension": effective_dimension,
            }
        )
    return pd.DataFrame(rows)


def fit_procrustes_latent(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xtr, ytr, xte, yte, _ = shared.role_center_pair(
        x_train, y_train, x_test, y_test, train_meta, test_meta
    )
    available = min(shared.PRE_PCA_RANK, len(xtr) - 2, xtr.shape[1], ytr.shape[1])
    xp_tr, xp_te, _ = shared.pca_project(xtr, xte, available)
    yp_tr, yp_te, _ = shared.pca_project(ytr, yte, available)
    actual_rank = min(rank, xp_tr.shape[1], yp_tr.shape[1])
    xp_tr = xp_tr[:, :actual_rank]
    yp_tr = yp_tr[:, :actual_rank]
    xp_te = xp_te[:, :actual_rank]
    yp_te = yp_te[:, :actual_rank]
    mean_x = xp_tr.mean(axis=0, keepdims=True)
    mean_y = yp_tr.mean(axis=0, keepdims=True)
    cross = (yp_tr - mean_y).T @ (xp_tr - mean_x)
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    return xp_te - mean_x, (yp_te - mean_y) @ rotation, rotation


def procrustes_cv(
    run: base.RunData,
    ranks: tuple[int, ...] = RANKS,
    passive_pair_permutation: np.ndarray | None = None,
) -> pd.DataFrame:
    passive = (
        run.passive
        if passive_pair_permutation is None
        else base.reorder_passive_by_pair(run.passive, run.item_meta, passive_pair_permutation)
    )
    rows: list[dict[str, object]] = []
    for held_pairs, train, test in base.outer_folds(run.item_meta):
        train_meta = run.item_meta.loc[train].reset_index(drop=True)
        test_meta = run.item_meta.loc[test].reset_index(drop=True)
        for rank in ranks:
            active_scores, passive_scores, _ = fit_procrustes_latent(
                run.active[train],
                passive[train],
                run.active[test],
                passive[test],
                train_meta,
                test_meta,
                rank,
            )
            metrics = shared.retrieval_metrics(active_scores, passive_scores, test_meta)
            rows.append(
                {
                    "patient": run.patient,
                    "run": run.run,
                    "held_pair_1": held_pairs[0],
                    "held_pair_2": held_pairs[1],
                    "rank": rank,
                    "model": "Procrustes",
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_static_models(
    folds: pd.DataFrame,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_summary = folds.groupby(["patient", "run", "rank", "model"], as_index=False).agg(
        retrieval_margin=("retrieval_margin", "mean"),
        two_by_two_accuracy=("two_by_two_accuracy", "mean"),
        matched_cosine=("matched_cosine", "mean"),
        matched_angle_deg=("matched_angle_deg", "mean"),
    )
    patient_summary = run_summary.groupby(["patient", "rank", "model"], as_index=False).mean(
        numeric_only=True
    )
    group_rows: list[dict[str, object]] = []
    patient_rows: list[pd.DataFrame] = []
    for model in ("Procrustes", "PLS"):
        for rank in RANKS:
            frame = run_summary[(run_summary["model"] == model) & (run_summary["rank"] == rank)]
            for column, suffix, reference in (
                ("retrieval_margin", "retrieval_margin", 0.0),
                ("two_by_two_accuracy", "accuracy", 0.5),
            ):
                metric_name = f"{model.lower()}_rank{rank}_{suffix}"
                patient, summary = shared.summarize_metric(
                    frame, column, metric_name, reference, n_bootstrap
                )
                patient["model"] = model
                patient["rank"] = rank
                patient["endpoint"] = suffix
                patient_rows.append(patient)
                group_rows.append(summary)

    contrasts: list[dict[str, object]] = []
    contrast_groups: list[dict[str, object]] = []
    for rank in RANKS:
        frame = patient_summary[patient_summary["rank"] == rank]
        for column, suffix in (
            ("retrieval_margin", "retrieval_margin"),
            ("two_by_two_accuracy", "accuracy"),
        ):
            pivot = frame.pivot(index="patient", columns="model", values=column)
            difference = pivot["PLS"] - pivot["Procrustes"]
            for patient, value in difference.items():
                contrasts.append(
                    {
                        "patient": patient,
                        "rank": rank,
                        "endpoint": suffix,
                        "pls_minus_procrustes": value,
                    }
                )
            sign = base.exact_sign_flip(difference, 0.0)
            ci = base.bootstrap_patient_ci(
                difference,
                n_bootstrap,
                base.stable_seed("manifold-model-contrast", rank, suffix),
            )
            contrast_groups.append(
                {
                    "metric": f"pls_minus_procrustes_rank{rank}_{suffix}",
                    "reference": 0.0,
                    "n_patients": len(difference),
                    "patient_mean": float(difference.mean()),
                    "patient_median": float(difference.median()),
                    "bootstrap_ci_low": ci[0],
                    "bootstrap_ci_high": ci[1],
                    "positive_patients": int((difference > 0).sum()),
                    "exact_signflip_p_greater": sign["p_greater"],
                    "exact_signflip_p_two_sided": sign["p_two_sided"],
                }
            )
    group = pd.DataFrame(group_rows + contrast_groups)
    return run_summary, patient_summary, group, pd.DataFrame(contrasts)


def synchronized_procrustes_permutation(
    runs: list[base.RunData],
    observed_run_summary: pd.DataFrame,
    n_permutations: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    observed: dict[tuple[int, str], float] = {}
    for rank in RANKS:
        frame = observed_run_summary[
            (observed_run_summary["rank"] == rank)
            & (observed_run_summary["model"] == "Procrustes")
        ]
        patient = frame.groupby("patient", as_index=False).mean(numeric_only=True)
        observed[(rank, "margin")] = float(patient["retrieval_margin"].mean())
        observed[(rank, "accuracy")] = float(patient["two_by_two_accuracy"].mean())

    rng = np.random.default_rng(base.stable_seed("manifold-procrustes-synchronized"))
    rows: list[dict[str, object]] = []
    progress_every = max(1, n_permutations // 10)
    for permutation_index in range(n_permutations):
        permutation = rng.permutation(10)
        folds = pd.concat(
            [procrustes_cv(run, passive_pair_permutation=permutation) for run in runs],
            ignore_index=True,
        )
        run_summary = folds.groupby(["patient", "run", "rank"], as_index=False).agg(
            retrieval_margin=("retrieval_margin", "mean"),
            two_by_two_accuracy=("two_by_two_accuracy", "mean"),
        )
        patient = run_summary.groupby(["patient", "rank"], as_index=False).mean(
            numeric_only=True
        )
        for rank in RANKS:
            frame = patient[patient["rank"] == rank]
            rows.append(
                {
                    "permutation_index": permutation_index,
                    "rank": rank,
                    "patient_equal_retrieval_margin": frame["retrieval_margin"].mean(),
                    "patient_equal_accuracy": frame["two_by_two_accuracy"].mean(),
                    "permutation_json": json.dumps(permutation.tolist()),
                }
            )
        if (permutation_index + 1) % progress_every == 0:
            print(
                f"Procrustes permutations: {permutation_index + 1}/{n_permutations}",
                flush=True,
            )
    null = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for rank in RANKS:
        frame = null[null["rank"] == rank]
        for column, suffix in (
            ("patient_equal_retrieval_margin", "margin"),
            ("patient_equal_accuracy", "accuracy"),
        ):
            value = observed[(rank, suffix)]
            summary[f"rank{rank}_{suffix}_observed"] = value
            summary[f"rank{rank}_{suffix}_item_permutation_p_greater"] = float(
                (np.sum(frame[column].to_numpy() >= value - 1e-12) + 1)
                / (n_permutations + 1)
            )
    return null, summary


def model_leave_one_patient_out(patient_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in sorted(patient_summary["model"].unique()):
        for rank in RANKS:
            frame = patient_summary[
                (patient_summary["model"] == model) & (patient_summary["rank"] == rank)
            ]
            all_margin = frame["retrieval_margin"].mean()
            all_accuracy = frame["two_by_two_accuracy"].mean()
            for excluded_patient in sorted(frame["patient"].unique()):
                kept = frame[frame["patient"] != excluded_patient]
                rows.append(
                    {
                        "model": model,
                        "rank": rank,
                        "excluded_patient": excluded_patient,
                        "n_remaining_patients": len(kept),
                        "all_patient_margin": all_margin,
                        "loo_margin": kept["retrieval_margin"].mean(),
                        "delta_margin": kept["retrieval_margin"].mean() - all_margin,
                        "all_patient_accuracy": all_accuracy,
                        "loo_accuracy": kept["two_by_two_accuracy"].mean(),
                        "delta_accuracy": kept["two_by_two_accuracy"].mean() - all_accuracy,
                    }
                )
    return pd.DataFrame(rows)


def dynamic_pls_cv(
    run: base.RunData,
    active: np.ndarray,
    passive: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for active_window_index in range(len(WINDOWS_MS)):
        for passive_window_index in range(len(WINDOWS_MS)):
            x = active[:, active_window_index]
            y = passive[:, passive_window_index]
            for held_pairs, train, test in base.outer_folds(run.item_meta):
                train_meta = run.item_meta.loc[train].reset_index(drop=True)
                test_meta = run.item_meta.loc[test].reset_index(drop=True)
                fit = shared.fit_two_view_pls(
                    x[train],
                    y[train],
                    x[test],
                    y[test],
                    train_meta,
                    test_meta,
                    latent_rank=PRIMARY_RANK,
                )
                metrics = shared.retrieval_metrics(
                    np.asarray(fit["x_test_scores"]),
                    np.asarray(fit["y_test_scores"]),
                    test_meta,
                )
                rows.append(
                    {
                        "patient": run.patient,
                        "run": run.run,
                        "active_window_index": active_window_index,
                        "passive_window_index": passive_window_index,
                        "held_pair_1": held_pairs[0],
                        "held_pair_2": held_pairs[1],
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def exact_maxstat(
    patient_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return shared.exact_patient_max_stat(patient_matrices)


def dynamic_inference(
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    run_summary = folds.groupby(
        ["patient", "run", "active_window_index", "passive_window_index"],
        as_index=False,
    ).agg(
        retrieval_margin=("retrieval_margin", "mean"),
        two_by_two_accuracy=("two_by_two_accuracy", "mean"),
    )
    patient_summary = run_summary.groupby(
        ["patient", "active_window_index", "passive_window_index"],
        as_index=False,
    ).mean(numeric_only=True)
    patients = sorted(patient_summary["patient"].unique())
    margin_matrices = []
    accuracy_excess_matrices = []
    for patient in patients:
        frame = patient_summary[patient_summary["patient"] == patient]
        margin_matrices.append(
            frame.pivot(
                index="active_window_index",
                columns="passive_window_index",
                values="retrieval_margin",
            ).to_numpy()
        )
        accuracy_excess_matrices.append(
            frame.pivot(
                index="active_window_index",
                columns="passive_window_index",
                values="two_by_two_accuracy",
            ).to_numpy()
            - 0.5
        )
    margin_matrices = np.asarray(margin_matrices)
    accuracy_excess_matrices = np.asarray(accuracy_excess_matrices)
    margin_observed, margin_null_max, margin_fwer = exact_maxstat(margin_matrices)
    accuracy_observed_excess, accuracy_null_max, accuracy_fwer = exact_maxstat(
        accuracy_excess_matrices
    )
    cell_rows: list[dict[str, object]] = []
    for active_index in range(len(WINDOWS_MS)):
        for passive_index in range(len(WINDOWS_MS)):
            margin_values = margin_matrices[:, active_index, passive_index]
            accuracy_values = accuracy_excess_matrices[:, active_index, passive_index]
            margin_sign = base.exact_sign_flip(margin_values, 0.0)
            accuracy_sign = base.exact_sign_flip(accuracy_values, 0.0)
            cell_rows.append(
                {
                    "active_window_index": active_index,
                    "passive_window_index": passive_index,
                    "active_window_ms": f"{WINDOWS_MS[active_index][0]}-{WINDOWS_MS[active_index][1]}",
                    "passive_window_ms": f"{WINDOWS_MS[passive_index][0]}-{WINDOWS_MS[passive_index][1]}",
                    "patient_mean_retrieval_margin": margin_observed[active_index, passive_index],
                    "margin_positive_patients": int(np.sum(margin_values > 0)),
                    "margin_exact_signflip_p_greater": margin_sign["p_greater"],
                    "margin_exact_maxstat_p_fwer_greater": margin_fwer[
                        active_index, passive_index
                    ],
                    "patient_mean_accuracy": accuracy_observed_excess[
                        active_index, passive_index
                    ]
                    + 0.5,
                    "accuracy_above_chance_patients": int(np.sum(accuracy_values > 0)),
                    "accuracy_exact_signflip_p_greater": accuracy_sign["p_greater"],
                    "accuracy_exact_maxstat_p_fwer_greater": accuracy_fwer[
                        active_index, passive_index
                    ],
                }
            )
    matrices = {
        "patients": np.asarray(patients, dtype=object),
        "margin_patient_matrices": margin_matrices,
        "accuracy_excess_patient_matrices": accuracy_excess_matrices,
        "margin_observed": margin_observed,
        "margin_null_max": margin_null_max,
        "margin_fwer": margin_fwer,
        "accuracy_observed_excess": accuracy_observed_excess,
        "accuracy_null_max": accuracy_null_max,
        "accuracy_fwer": accuracy_fwer,
    }
    return run_summary, patient_summary, pd.DataFrame(cell_rows), matrices


def dynamic_lag_analysis(
    matrices: dict[str, np.ndarray],
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = {
        "active_lag_100ms": [(index + 1, index) for index in range(len(WINDOWS_MS) - 1)],
        "aligned": [(index, index) for index in range(len(WINDOWS_MS))],
        "passive_lag_100ms": [(index, index + 1) for index in range(len(WINDOWS_MS) - 1)],
    }
    rows: list[dict[str, object]] = []
    for patient_index, patient in enumerate(matrices["patients"]):
        for lag, cells in definitions.items():
            margin = float(
                np.mean(
                    [
                        matrices["margin_patient_matrices"][patient_index, active, passive]
                        for active, passive in cells
                    ]
                )
            )
            accuracy = float(
                np.mean(
                    [
                        matrices["accuracy_excess_patient_matrices"][
                            patient_index, active, passive
                        ]
                        + 0.5
                        for active, passive in cells
                    ]
                )
            )
            rows.append(
                {
                    "patient": patient,
                    "lag": lag,
                    "retrieval_margin": margin,
                    "two_by_two_accuracy": accuracy,
                }
            )
    patient = pd.DataFrame(rows)
    group_rows: list[dict[str, object]] = []
    for lag in definitions:
        frame = patient[patient["lag"] == lag]
        for column, suffix, reference in (
            ("retrieval_margin", "margin", 0.0),
            ("two_by_two_accuracy", "accuracy", 0.5),
        ):
            _, summary = shared.summarize_metric(
                frame,
                column,
                f"dynamic_{lag}_{suffix}",
                reference,
                n_bootstrap,
            )
            group_rows.append(summary)
    return patient, pd.DataFrame(group_rows)


def leakage_audit(run: base.RunData) -> dict[str, object]:
    held_pairs, train, test = next(iter(base.outer_folds(run.item_meta)))
    train_meta = run.item_meta.loc[train].reset_index(drop=True)
    test_meta = run.item_meta.loc[test].reset_index(drop=True)
    x1, _, rotation1 = fit_procrustes_latent(
        run.active[train],
        run.passive[train],
        run.active[test],
        run.passive[test],
        train_meta,
        test_meta,
        PRIMARY_RANK,
    )
    x2, _, rotation2 = fit_procrustes_latent(
        run.active[train],
        run.passive[train],
        run.active[test],
        run.passive[test] * 1_000_000.0,
        train_meta,
        test_meta,
        PRIMARY_RANK,
    )
    rotation_unchanged = bool(np.allclose(rotation1, rotation2, atol=1e-12, rtol=1e-12))
    active_test_unchanged = bool(np.allclose(x1, x2, atol=1e-12, rtol=1e-12))
    return {
        "run": run.run,
        "held_pairs": [int(pair) for pair in held_pairs],
        "test_passive_truth_multiplier": 1_000_000,
        "rotation_unchanged": rotation_unchanged,
        "active_test_projection_unchanged": active_test_unchanged,
        "passed": rotation_unchanged and active_test_unchanged,
    }


def patient_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    categories: list[str],
    category_col: str,
    value_col: str,
    colors: dict[str, str],
    reference: float | None = None,
) -> None:
    patients = sorted(frame["patient"].unique())
    offsets = np.linspace(-0.13, 0.13, len(patients))
    for category_index, category in enumerate(categories):
        sub = frame[frame[category_col] == category]
        for offset, patient in zip(offsets, patients):
            values = sub.loc[sub["patient"] == patient, value_col]
            if not values.empty:
                ax.scatter(
                    category_index + offset,
                    values.iloc[0],
                    color=colors[category],
                    edgecolor="white",
                    linewidth=0.4,
                    s=28,
                    alpha=0.9,
                )
        if not sub.empty:
            ax.scatter(
                category_index,
                sub[value_col].mean(),
                marker="_",
                s=190,
                color="black",
                linewidth=2,
            )
    if reference is not None:
        ax.axhline(reference, color=OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set_xticks(range(len(categories)), categories)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def make_dimension_alignment_figure(
    report_dir: Path,
    spectrum: pd.DataFrame,
    dimension_patient: pd.DataFrame,
    model_patient: pd.DataFrame,
    model_group: pd.DataFrame,
    contrasts: pd.DataFrame,
    permutation_null: pd.DataFrame,
    permutation_summary: dict[str, float],
) -> None:
    shared.configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 6.5))

    ax = axes[0, 0]
    spectrum_patient = spectrum.groupby(["patient", "view", "dimension"], as_index=False)[
        "cumulative_fraction"
    ].mean()
    view_style = {
        "active": (OKABE_ITO["blue"], "Active", "-"),
        "passive": (OKABE_ITO["orange"], "Passive", "-"),
        "shared_cross_covariance": (OKABE_ITO["green"], "Cross-view (r8)", "--"),
    }
    for view, (color, label, linestyle) in view_style.items():
        sub = spectrum_patient[
            (spectrum_patient["view"] == view)
            & (spectrum_patient["dimension"] <= MAX_SPECTRUM_RANK)
        ]
        pivot = sub.pivot(index="patient", columns="dimension", values="cumulative_fraction")
        for _, values in pivot.iterrows():
            ax.plot(
                pivot.columns,
                values,
                color=color,
                alpha=0.18,
                lw=0.8,
                linestyle=linestyle,
            )
        ax.plot(
            pivot.columns,
            pivot.mean(axis=0),
            color=color,
            lw=2,
            linestyle=linestyle,
            label=label,
        )
    ax.axvline(2, color=OKABE_ITO["gray"], ls=":", lw=0.8)
    ax.axvline(4, color=OKABE_ITO["gray"], ls=":", lw=0.8)
    ax.set(xlabel="Retained dimensions", ylabel="Cumulative spectral fraction", ylim=(0, 1.03))
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Descriptive spectral concentration")
    panel_label(ax, "A")

    ax = axes[0, 1]
    dimension_plot = dimension_patient.copy()
    labels = ["active", "passive", "shared_cross_covariance"]
    patient_scatter(
        ax,
        dimension_plot,
        labels,
        "view",
        "rank_limited_effective_dimension",
        {
            "active": OKABE_ITO["blue"],
            "passive": OKABE_ITO["orange"],
            "shared_cross_covariance": OKABE_ITO["green"],
        },
    )
    ax.set_xticklabels(["Active", "Passive", "Cross-view\nspectrum"])
    ax.set_ylabel("Rank-8 effective dimension (PR)")
    ax.set_title("Common-rank effective dimension")
    panel_label(ax, "B")

    ax = axes[0, 2]
    plot_rows = []
    for row in model_patient.itertuples(index=False):
        plot_rows.append(
            {
                "patient": row.patient,
                "category": f"{row.model} r{int(row.rank)}",
                "retrieval_margin": row.retrieval_margin,
            }
        )
    model_plot = pd.DataFrame(plot_rows)
    categories = ["Procrustes r2", "PLS r2", "Procrustes r4", "PLS r4"]
    patient_scatter(
        ax,
        model_plot,
        categories,
        "category",
        "retrieval_margin",
        {
            "Procrustes r2": OKABE_ITO["purple"],
            "PLS r2": OKABE_ITO["green"],
            "Procrustes r4": OKABE_ITO["purple"],
            "PLS r4": OKABE_ITO["green"],
        },
        0.0,
    )
    ax.set_xticklabels(["Proc.\nr2", "PLS\nr2", "Proc.\nr4", "PLS\nr4"])
    ax.set_ylabel("Held-pair retrieval margin")
    ax.set_title("Rigid versus flexible alignment")
    panel_label(ax, "C")

    ax = axes[1, 0]
    null_rank2 = permutation_null[permutation_null["rank"] == PRIMARY_RANK][
        "patient_equal_retrieval_margin"
    ]
    null_rank4 = permutation_null[permutation_null["rank"] == 4][
        "patient_equal_retrieval_margin"
    ]
    bins = np.linspace(
        min(null_rank2.min(), null_rank4.min()),
        max(null_rank2.max(), null_rank4.max()),
        26,
    )
    ax.hist(
        null_rank2,
        bins=bins,
        color=OKABE_ITO["light_gray"],
        edgecolor="white",
        alpha=0.8,
        label="Rank-2 null",
    )
    ax.hist(
        null_rank4,
        bins=bins,
        histtype="step",
        color=OKABE_ITO["orange"],
        linewidth=1.2,
        label="Rank-4 null",
    )
    ax.axvline(
        permutation_summary["rank2_margin_observed"],
        color=OKABE_ITO["purple"],
        lw=2.0,
        label="Rank-2 matched",
    )
    ax.axvline(
        permutation_summary["rank4_margin_observed"],
        color=OKABE_ITO["orange"],
        lw=2.0,
        label="Rank-4 matched",
    )
    ax.axvline(0, color=OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set(xlabel="Patient-equal Procrustes margin", ylabel="Synchronized permutations")
    ax.legend(frameon=False)
    ax.set_title("Item-level Procrustes test")
    panel_label(ax, "D")

    ax = axes[1, 1]
    contrast_plot = contrasts[contrasts["rank"] == PRIMARY_RANK].copy()
    contrast_plot["category"] = contrast_plot["endpoint"]
    patient_scatter(
        ax,
        contrast_plot,
        ["retrieval_margin", "accuracy"],
        "category",
        "pls_minus_procrustes",
        {
            "retrieval_margin": OKABE_ITO["green"],
            "accuracy": OKABE_ITO["blue"],
        },
        0.0,
    )
    ax.set_xticklabels(["Margin", "Accuracy"])
    ax.set_ylabel("PLS minus Procrustes")
    ax.set_title("Model flexibility advantage")
    panel_label(ax, "E")

    ax = axes[1, 2]
    ax.axis("off")
    lookup = model_group.set_index("metric")
    proc = lookup.loc["procrustes_rank2_retrieval_margin"]
    proc_rank4 = lookup.loc["procrustes_rank4_retrieval_margin"]
    pls = lookup.loc["pls_rank2_retrieval_margin"]
    contrast = lookup.loc["pls_minus_procrustes_rank2_retrieval_margin"]
    text = (
        "Interpretation gate\n\n"
        f"Procrustes r2 margin={proc.patient_mean:.3f}\n"
        f"patient P={proc.exact_signflip_p_greater:.4f}; "
        f"item P={permutation_summary['rank2_margin_item_permutation_p_greater']:.4f}\n\n"
        f"Procrustes r4 margin={proc_rank4.patient_mean:.3f}\n"
        f"patient P={proc_rank4.exact_signflip_p_greater:.4f}; "
        f"item P={permutation_summary['rank4_margin_item_permutation_p_greater']:.4f}\n\n"
        f"PLS r2 margin={pls.patient_mean:.3f}\n"
        f"patient P={pls.exact_signflip_p_greater:.4f}\n\n"
        f"PLS-Proc. margin={contrast.patient_mean:.3f}\n"
        f"paired patient P={contrast.exact_signflip_p_two_sided:.4f}\n\n"
        "Spectral PR is descriptive, not a test\n"
        "of a generative neural manifold."
    )
    ax.text(0, 1, text, va="top", fontsize=7.8, linespacing=1.28)
    panel_label(ax, "F")

    fig.suptitle(
        "Intrinsic dimensionality and cross-validated manifold alignment",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0, w_pad=1.8)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"Fig_3_intrinsic_dimension_manifold_alignment.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    p_fwer: np.ndarray,
    center: float,
    title: str,
    colorbar_label: str,
) -> None:
    centered = values - center
    limit = max(0.05, float(np.nanmax(np.abs(centered))))
    image = ax.imshow(
        centered,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    labels = [f"{start}-{stop}" for start, stop in WINDOWS_MS]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Passive noun window (ms)")
    ax.set_ylabel("Active noun window (ms)")
    ax.set_title(title)
    for active in range(len(labels)):
        for passive in range(len(labels)):
            display = values[active, passive]
            marker = "*" if p_fwer[active, passive] < 0.05 else ""
            ax.text(
                passive,
                active,
                f"{display:.2f}{marker}",
                ha="center",
                va="center",
                fontsize=5.7,
                color="black",
            )
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def make_dynamic_figure(
    report_dir: Path,
    cell_stats: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    lag_patient: pd.DataFrame,
    lag_group: pd.DataFrame,
) -> None:
    shared.configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))
    n_windows = len(WINDOWS_MS)
    margin = cell_stats.pivot(
        index="active_window_index",
        columns="passive_window_index",
        values="patient_mean_retrieval_margin",
    ).to_numpy()
    margin_fwer = cell_stats.pivot(
        index="active_window_index",
        columns="passive_window_index",
        values="margin_exact_maxstat_p_fwer_greater",
    ).to_numpy()
    accuracy = cell_stats.pivot(
        index="active_window_index",
        columns="passive_window_index",
        values="patient_mean_accuracy",
    ).to_numpy()
    accuracy_fwer = cell_stats.pivot(
        index="active_window_index",
        columns="passive_window_index",
        values="accuracy_exact_maxstat_p_fwer_greater",
    ).to_numpy()

    heatmap(
        axes[0, 0],
        margin,
        margin_fwer,
        0.0,
        "Held-pair latent retrieval margin",
        "Margin",
    )
    panel_label(axes[0, 0], "A")
    heatmap(
        axes[0, 1],
        accuracy,
        accuracy_fwer,
        0.5,
        "Held-pair latent retrieval accuracy",
        "Accuracy minus 0.5",
    )
    panel_label(axes[0, 1], "B")

    ax = axes[1, 0]
    patient_matrices = matrices["margin_patient_matrices"]
    x = np.arange(n_windows)
    labels = [f"{start}-{stop}" for start, stop in WINDOWS_MS]
    for patient_matrix in patient_matrices:
        ax.plot(x, np.diag(patient_matrix), color=OKABE_ITO["light_gray"], lw=1)
    ax.plot(
        x,
        np.mean([np.diag(matrix) for matrix in patient_matrices], axis=0),
        color=OKABE_ITO["purple"],
        marker="o",
        lw=2,
    )
    ax.axhline(0, color=OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set(xlabel="Matched noun window (ms)", ylabel="Retrieval margin")
    ax.set_title("Diagonal latent alignment")
    panel_label(ax, "C")

    ax = axes[1, 1]
    categories = ["active_lag_100ms", "aligned", "passive_lag_100ms"]
    patient_scatter(
        ax,
        lag_patient,
        categories,
        "lag",
        "retrieval_margin",
        {
            "active_lag_100ms": OKABE_ITO["blue"],
            "aligned": OKABE_ITO["purple"],
            "passive_lag_100ms": OKABE_ITO["orange"],
        },
        0.0,
    )
    ax.set_xticklabels(["Active\n+100 ms", "Aligned", "Passive\n+100 ms"])
    ax.set_ylabel("Retrieval margin")
    lookup = lag_group.set_index("metric")
    aligned_p = lookup.loc["dynamic_aligned_margin", "exact_signflip_p_greater"]
    passive_lag_p = lookup.loc[
        "dynamic_passive_lag_100ms_margin", "exact_signflip_p_greater"
    ]
    ax.set_title(f"Prespecified lag summaries\naligned P={aligned_p:.3f}; passive lag P={passive_lag_p:.3f}")
    panel_label(ax, "D")

    fig.suptitle(
        "Time-resolved held-out latent-manifold alignment",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0, w_pad=2.0)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"Fig_4_dynamic_latent_manifold_alignment.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def write_report(
    report_dir: Path,
    contract: dict[str, object],
    audit: pd.DataFrame,
    dimension_patient: pd.DataFrame,
    model_group: pd.DataFrame,
    permutation_summary: dict[str, float],
    loo: pd.DataFrame,
    cell_stats: pd.DataFrame,
    lag_group: pd.DataFrame,
    leakage: dict[str, object],
) -> None:
    group = model_group.set_index("metric")
    dimension = dimension_patient.set_index("view")
    proc_margin = group.loc["procrustes_rank2_retrieval_margin"]
    proc_accuracy = group.loc["procrustes_rank2_accuracy"]
    proc_rank4_margin = group.loc["procrustes_rank4_retrieval_margin"]
    proc_rank4_accuracy = group.loc["procrustes_rank4_accuracy"]
    pls_margin = group.loc["pls_rank2_retrieval_margin"]
    pls_accuracy = group.loc["pls_rank2_accuracy"]
    contrast = group.loc["pls_minus_procrustes_rank2_retrieval_margin"]
    gz002_proc = loo[
        (loo["model"] == "Procrustes")
        & (loo["rank"] == PRIMARY_RANK)
        & (loo["excluded_patient"] == "gz002")
    ].iloc[0]
    gz002_pls = loo[
        (loo["model"] == "PLS")
        & (loo["rank"] == PRIMARY_RANK)
        & (loo["excluded_patient"] == "gz002")
    ].iloc[0]
    gz002_proc_rank4 = loo[
        (loo["model"] == "Procrustes")
        & (loo["rank"] == 4)
        & (loo["excluded_patient"] == "gz002")
    ].iloc[0]
    margin_sig = int((cell_stats["margin_exact_maxstat_p_fwer_greater"] < 0.05).sum())
    accuracy_sig = int((cell_stats["accuracy_exact_maxstat_p_fwer_greater"] < 0.05).sum())
    best_margin = cell_stats.sort_values(
        "patient_mean_retrieval_margin", ascending=False
    ).iloc[0]
    lag = lag_group.set_index("metric")

    lines = [
        "# 主动/被动隐空间与manifold分析",
        "",
        "## 一句话结论",
        "",
        "光谱显示群体响应具有有限维度集中，但严格的刚性Procrustes映射能否跨句式推广，必须由患者层和同步刺激置换共同判断；动态隐空间结果则以64格患者max-stat为最终门槛。",
        "",
        "## 冻结规格",
        "",
        f"- unit set：`{contract['unit_set']}`；伪迹阈值：`{contract['artifact_threshold']}`。",
        f"- 纳入{int(audit['included'].sum())} runs、{audit.loc[audit['included'], 'patient'].nunique()} patients；患者等权。",
        "- Procrustes与PLS均完整留出两个sentence pairs；role centering、unit scaling、PCA和映射只在训练折拟合。",
        "- Procrustes只允许正交旋转；PLS允许两种句式使用不同读出轴，因此更灵活。",
        "- 内在维度为full-data描述性participation ratio；unit-split区间不作为独立样本置信区间。",
        "- 动态分析固定Rank 2和8×8个100-ms窗，患者exact max-stat控制64格FWER。",
        "",
        "## 1. 内在维度与光谱集中",
        "",
        f"- Active全谱PR={dimension.loc['active', 'effective_dimension']:.2f}；统一Rank-8截断PR={dimension.loc['active', 'rank_limited_effective_dimension']:.2f}；前2/4维全谱占比={dimension.loc['active', 'variance_rank2']:.3f}/{dimension.loc['active', 'variance_rank4']:.3f}。",
        f"- Passive全谱PR={dimension.loc['passive', 'effective_dimension']:.2f}；统一Rank-8截断PR={dimension.loc['passive', 'rank_limited_effective_dimension']:.2f}；前2/4维全谱占比={dimension.loc['passive', 'variance_rank2']:.3f}/{dimension.loc['passive', 'variance_rank4']:.3f}。",
        f"- Cross-view covariance为预设Rank-8 spectrum：PR={dimension.loc['shared_cross_covariance', 'rank_limited_effective_dimension']:.2f}；前2/4维占比={dimension.loc['shared_cross_covariance', 'variance_rank2']:.3f}/{dimension.loc['shared_cross_covariance', 'variance_rank4']:.3f}。",
        "- 只有统一Rank-8 PR可在三种view间直接比较；这些值仅说明谱集中程度，不等价于已识别连续、可泛化的生成流形。",
        "",
        "## 2. Cross-validated Procrustes与PLS",
        "",
        f"- Procrustes Rank-2 margin={proc_margin.patient_mean:.3f}，患者P={proc_margin.exact_signflip_p_greater:.4f}，同步刺激置换P={permutation_summary['rank2_margin_item_permutation_p_greater']:.4f}。",
        f"- Procrustes Rank-2 accuracy={proc_accuracy.patient_mean:.3f}，患者P={proc_accuracy.exact_signflip_p_greater:.4f}，刺激P={permutation_summary['rank2_accuracy_item_permutation_p_greater']:.4f}。",
        f"- 预设Rank-4敏感性：Procrustes margin={proc_rank4_margin.patient_mean:.3f}，患者单侧P={proc_rank4_margin.exact_signflip_p_greater:.4f}、双侧P={proc_rank4_margin.exact_signflip_p_two_sided:.4f}，刺激P={permutation_summary['rank4_margin_item_permutation_p_greater']:.4f}；accuracy={proc_rank4_accuracy.patient_mean:.3f}，患者单侧P={proc_rank4_accuracy.exact_signflip_p_greater:.4f}，刺激P={permutation_summary['rank4_accuracy_item_permutation_p_greater']:.4f}。",
        f"- PLS Rank-2 margin={pls_margin.patient_mean:.3f}，患者P={pls_margin.exact_signflip_p_greater:.4f}；accuracy={pls_accuracy.patient_mean:.3f}，患者P={pls_accuracy.exact_signflip_p_greater:.4f}。",
        f"- PLS-Procrustes margin差={contrast.patient_mean:.3f}，患者配对双侧P={contrast.exact_signflip_p_two_sided:.4f}。",
        f"- 排除gz002后，Procrustes margin/accuracy={gz002_proc.loo_margin:.3f}/{gz002_proc.loo_accuracy:.3f}；PLS={gz002_pls.loo_margin:.3f}/{gz002_pls.loo_accuracy:.3f}。",
        f"- Rank-4 Procrustes排除gz002后margin/accuracy={gz002_proc_rank4.loo_margin:.3f}/{gz002_proc_rank4.loo_accuracy:.3f}。",
        "- 若PLS优于Procrustes，只能说明灵活的不同读出轴优于刚性旋转；若两者患者层均失败，则不能确认共享manifold。",
        "",
        "## 3. 时间分辨latent-manifold alignment",
        "",
        f"- 64格中margin通过患者max-stat FWER的cells={margin_sig}/64；accuracy={accuracy_sig}/64。",
        f"- 描述性最大margin位于active {best_margin.active_window_ms} ms、passive {best_margin.passive_window_ms} ms：margin={best_margin.patient_mean_retrieval_margin:.3f}，FWER P={best_margin.margin_exact_maxstat_p_fwer_greater:.4f}。",
        f"- 预设aligned-window margin={lag.loc['dynamic_aligned_margin', 'patient_mean']:.3f}，患者P={lag.loc['dynamic_aligned_margin', 'exact_signflip_p_greater']:.4f}。",
        f"- passive延迟100 ms的margin={lag.loc['dynamic_passive_lag_100ms_margin', 'patient_mean']:.3f}，患者P={lag.loc['dynamic_passive_lag_100ms_margin', 'exact_signflip_p_greater']:.4f}。",
        "- 最佳时间格只作描述；只有max-stat通过时才能称为时间特异性隐空间对齐。",
        "",
        "## 审计与解释边界",
        "",
        f"- 测试truth放大一百万倍后，训练Procrustes rotation与active测试投影保持不变：`passed={leakage['passed']}`。",
        "- active/passive与功能词、声学、表面位置和句式共同变化，不能把对齐解释为抽象语态流形。",
        "- Agent/Patient与生命性和词汇类别混杂；role centering不能解除设计层面混杂。",
        "- 本分析在查看既有结果后开展，属于post-result exploratory analysis。",
        "",
        "## 主要输出",
        "",
        "- `intrinsic_spectrum.csv`, `intrinsic_dimension_run_summary.csv`, `intrinsic_dimension_rank8_comparison.csv`, `intrinsic_dimension_unit_splits.csv`",
        "- `static_model_cv_folds.csv`, `static_model_patient_summary.csv`, `static_model_group_inference.csv`",
        "- `procrustes_synchronized_pair_permutation_null.csv`, `static_model_leave_one_patient_out.csv`",
        "- `dynamic_latent_cv_folds.csv`, `dynamic_latent_cell_stats.csv`, `dynamic_lag_group_inference.csv`",
        "- `figures/Fig_3_intrinsic_dimension_manifold_alignment.png/.pdf`",
        "- `figures/Fig_4_dynamic_latent_manifold_alignment.png/.pdf`",
        "",
    ]
    (report_dir / "README_zh.md").write_text("\n".join(lines), encoding="utf-8")


def refresh_manifest(report_dir: Path, data_dir: Path) -> None:
    rows = []
    for path in sorted(list(report_dir.rglob("*")) + list(data_dir.glob("*"))):
        if path.is_file() and path.name != "delivery_manifest.csv":
            rows.append({"path": str(path), "size_bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(report_dir / "delivery_manifest.csv", index=False)


def write_combined_if_available(main_report_dir: Path, main_data_dir: Path) -> None:
    specifications = [
        ("主规格", main_report_dir, main_data_dir),
        (
            "宽松unit敏感性",
            main_report_dir / "sensitivity_usable_lenient",
            main_data_dir / "sensitivity_usable_lenient",
        ),
    ]
    rows = []
    for label, report_dir, data_dir in specifications:
        summary_path = data_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        group = pd.DataFrame(summary["static_model_group_inference"]).set_index("metric")
        dimension = pd.DataFrame(summary["intrinsic_dimension_patient_summary"]).groupby(
            "view"
        ).mean(numeric_only=True)
        if "rank_limited_effective_dimension" not in dimension.columns:
            continue
        loo = pd.DataFrame(summary["static_model_leave_one_patient_out"])
        proc_loo = loo[
            (loo["model"] == "Procrustes")
            & (loo["rank"] == PRIMARY_RANK)
            & (loo["excluded_patient"] == "gz002")
        ].iloc[0]
        proc_rank4_loo = loo[
            (loo["model"] == "Procrustes")
            & (loo["rank"] == 4)
            & (loo["excluded_patient"] == "gz002")
        ].iloc[0]
        rows.append(
            {
                "specification": label,
                "unit_set": summary["analysis_contract"]["unit_set"],
                "active_effective_dimension": dimension.loc["active", "effective_dimension"],
                "passive_effective_dimension": dimension.loc["passive", "effective_dimension"],
                "active_rank8_effective_dimension": dimension.loc[
                    "active", "rank_limited_effective_dimension"
                ],
                "passive_rank8_effective_dimension": dimension.loc[
                    "passive", "rank_limited_effective_dimension"
                ],
                "cross_view_rank8_effective_dimension": dimension.loc[
                    "shared_cross_covariance", "rank_limited_effective_dimension"
                ],
                "procrustes_rank2_margin": group.loc[
                    "procrustes_rank2_retrieval_margin", "patient_mean"
                ],
                "procrustes_rank2_margin_patient_p": group.loc[
                    "procrustes_rank2_retrieval_margin", "exact_signflip_p_greater"
                ],
                "procrustes_rank2_margin_item_p": summary["procrustes_permutation"][
                    "rank2_margin_item_permutation_p_greater"
                ],
                "procrustes_rank2_accuracy": group.loc[
                    "procrustes_rank2_accuracy", "patient_mean"
                ],
                "procrustes_rank4_margin": group.loc[
                    "procrustes_rank4_retrieval_margin", "patient_mean"
                ],
                "procrustes_rank4_margin_patient_p_greater": group.loc[
                    "procrustes_rank4_retrieval_margin", "exact_signflip_p_greater"
                ],
                "procrustes_rank4_margin_patient_p_two_sided": group.loc[
                    "procrustes_rank4_retrieval_margin", "exact_signflip_p_two_sided"
                ],
                "procrustes_rank4_margin_item_p": summary["procrustes_permutation"][
                    "rank4_margin_item_permutation_p_greater"
                ],
                "procrustes_rank4_accuracy": group.loc[
                    "procrustes_rank4_accuracy", "patient_mean"
                ],
                "procrustes_rank4_accuracy_patient_p_greater": group.loc[
                    "procrustes_rank4_accuracy", "exact_signflip_p_greater"
                ],
                "procrustes_rank4_accuracy_item_p": summary["procrustes_permutation"][
                    "rank4_accuracy_item_permutation_p_greater"
                ],
                "pls_rank2_margin": group.loc["pls_rank2_retrieval_margin", "patient_mean"],
                "pls_rank2_margin_patient_p": group.loc[
                    "pls_rank2_retrieval_margin", "exact_signflip_p_greater"
                ],
                "pls_minus_procrustes_margin": group.loc[
                    "pls_minus_procrustes_rank2_retrieval_margin", "patient_mean"
                ],
                "pls_minus_procrustes_margin_two_sided_p": group.loc[
                    "pls_minus_procrustes_rank2_retrieval_margin",
                    "exact_signflip_p_two_sided",
                ],
                "loo_gz002_procrustes_margin": proc_loo["loo_margin"],
                "loo_gz002_procrustes_accuracy": proc_loo["loo_accuracy"],
                "loo_gz002_procrustes_rank4_margin": proc_rank4_loo["loo_margin"],
                "loo_gz002_procrustes_rank4_accuracy": proc_rank4_loo["loo_accuracy"],
                "dynamic_margin_fwer_cells": summary["dynamic_margin_fwer_cells"],
                "dynamic_accuracy_fwer_cells": summary["dynamic_accuracy_fwer_cells"],
            }
        )
    comparison = pd.DataFrame(rows)
    if len(comparison) != 2:
        return
    comparison.to_csv(main_report_dir / "specification_comparison.csv", index=False)
    indexed = comparison.set_index("specification")
    main = indexed.loc["主规格"]
    sensitivity = indexed.loc["宽松unit敏感性"]
    lines = [
        "# 隐空间与manifold分析：跨规格综合结论",
        "",
        "## 结果表",
        "",
        "| 指标 | 主规格 good_only | 宽松规格 usable_lenient |",
        "|---|---:|---:|",
        f"| Active / Passive全谱PR | {main.active_effective_dimension:.2f} / {main.passive_effective_dimension:.2f} | {sensitivity.active_effective_dimension:.2f} / {sensitivity.passive_effective_dimension:.2f} |",
        f"| 统一Rank-8 Active / Passive / Cross-view PR | {main.active_rank8_effective_dimension:.2f} / {main.passive_rank8_effective_dimension:.2f} / {main.cross_view_rank8_effective_dimension:.2f} | {sensitivity.active_rank8_effective_dimension:.2f} / {sensitivity.passive_rank8_effective_dimension:.2f} / {sensitivity.cross_view_rank8_effective_dimension:.2f} |",
        f"| Procrustes Rank-2 margin | {main.procrustes_rank2_margin:.3f}；患者P={main.procrustes_rank2_margin_patient_p:.4f}；刺激P={main.procrustes_rank2_margin_item_p:.4f} | {sensitivity.procrustes_rank2_margin:.3f}；患者P={sensitivity.procrustes_rank2_margin_patient_p:.4f}；刺激P={sensitivity.procrustes_rank2_margin_item_p:.4f} |",
        f"| Procrustes Rank-4 margin | {main.procrustes_rank4_margin:.3f}；患者单侧/双侧P={main.procrustes_rank4_margin_patient_p_greater:.4f}/{main.procrustes_rank4_margin_patient_p_two_sided:.4f}；刺激P={main.procrustes_rank4_margin_item_p:.4f} | {sensitivity.procrustes_rank4_margin:.3f}；患者单侧/双侧P={sensitivity.procrustes_rank4_margin_patient_p_greater:.4f}/{sensitivity.procrustes_rank4_margin_patient_p_two_sided:.4f}；刺激P={sensitivity.procrustes_rank4_margin_item_p:.4f} |",
        f"| Procrustes Rank-4 accuracy | {main.procrustes_rank4_accuracy:.3f}；患者P={main.procrustes_rank4_accuracy_patient_p_greater:.4f}；刺激P={main.procrustes_rank4_accuracy_item_p:.4f} | {sensitivity.procrustes_rank4_accuracy:.3f}；患者P={sensitivity.procrustes_rank4_accuracy_patient_p_greater:.4f}；刺激P={sensitivity.procrustes_rank4_accuracy_item_p:.4f} |",
        f"| PLS Rank-2 margin | {main.pls_rank2_margin:.3f}；患者P={main.pls_rank2_margin_patient_p:.4f} | {sensitivity.pls_rank2_margin:.3f}；患者P={sensitivity.pls_rank2_margin_patient_p:.4f} |",
        f"| PLS-Procrustes margin | {main.pls_minus_procrustes_margin:.3f}；双侧P={main.pls_minus_procrustes_margin_two_sided_p:.4f} | {sensitivity.pls_minus_procrustes_margin:.3f}；双侧P={sensitivity.pls_minus_procrustes_margin_two_sided_p:.4f} |",
        f"| 排除gz002后的Procrustes margin / accuracy | {main.loo_gz002_procrustes_margin:.3f} / {main.loo_gz002_procrustes_accuracy:.3f} | {sensitivity.loo_gz002_procrustes_margin:.3f} / {sensitivity.loo_gz002_procrustes_accuracy:.3f} |",
        f"| 排除gz002后的Rank-4 Procrustes margin / accuracy | {main.loo_gz002_procrustes_rank4_margin:.3f} / {main.loo_gz002_procrustes_rank4_accuracy:.3f} | {sensitivity.loo_gz002_procrustes_rank4_margin:.3f} / {sensitivity.loo_gz002_procrustes_rank4_accuracy:.3f} |",
        f"| 动态margin / accuracy FWER cells | {int(main.dynamic_margin_fwer_cells)}/64 / {int(main.dynamic_accuracy_fwer_cells)}/64 | {int(sensitivity.dynamic_margin_fwer_cells)}/64 / {int(sensitivity.dynamic_accuracy_fwer_cells)}/64 |",
        "",
        "## 判读原则",
        "",
        "- 最稳定的新模式是Rank-4 Procrustes margin：两套规格的同步刺激置换均达到P<0.05，患者单侧P均为0.0625、双侧P均为0.125。这是跨规格一致的方向性证据，但尚未达到患者层确认标准。",
        "- 两套unit规格高度重叠，属于敏感性分析而非独立重复；跨规格一致不能替代新患者或独立数据验证。",
        "- 有限PR只能说明光谱集中，不能单独证明连续manifold。",
        "- Procrustes成功意味着更强的‘同一几何、不同正交坐标系’；只有PLS成功则更符合不同读出轴下的灵活线性对应。",
        "- 患者层、同步刺激置换、unit规格敏感性和gz002留一诊断必须同时考虑。",
        "- 动态热图中未经max-stat通过的最佳格只能作描述。",
        "",
    ]
    (main_report_dir / "COMBINED_REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


def shared_latent_data_dir(unit_set: str) -> Path:
    if unit_set == "good_only":
        return shared.DEFAULT_DATA_DIR
    return shared.DEFAULT_DATA_DIR / "sensitivity_usable_lenient"


def run_analysis(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    data_dir = Path(args.data_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_threshold = None if args.artifact_threshold < 0 else args.artifact_threshold
    contract = {
        "analysis_date": "2026-08-06",
        "status": "post-result exploratory literature-motivated analysis",
        "unit_set": args.unit_set,
        "artifact_threshold": artifact_threshold,
        "intrinsic_dimension": "participation ratio of role-centered full-data spectra",
        "shared_spectrum": "squared singular-value spectrum of rank-8 cross-view covariance",
        "dimension_unit_splits": args.dimension_unit_splits,
        "static_models": ["orthogonal Procrustes", "two-view PLS"],
        "latent_ranks": list(RANKS),
        "outer_cv": "leave two complete sentence pairs out",
        "procrustes_permutations": args.procrustes_permutations,
        "dynamic_model": "two-view PLS rank 2",
        "dynamic_windows_ms": [list(window) for window in WINDOWS_MS],
        "dynamic_multiple_comparison": "patient exact max-stat across 64 cells",
        "patient_inference": "patient-equal exhaustive sign flip",
        "interpretation_limit": "condition confounded with marker/acoustics/position/form; role with animacy/lexical class",
    }
    (report_dir / "analysis_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics, available_artifact_runs = base.load_filtered_metrics(args.unit_set, artifact_threshold)
    runs, audit = base.build_run_data(metrics, available_artifact_runs, args.unit_set)
    if len(runs) < 5 or len({run.patient for run in runs}) < 5:
        raise RuntimeError(f"Insufficient complete data:\n{audit.to_string(index=False)}")
    audit.to_csv(data_dir / "run_inclusion_audit.csv", index=False)
    audit.to_csv(report_dir / "run_inclusion_audit.csv", index=False)

    leakage = leakage_audit(runs[0])
    if not leakage["passed"]:
        raise RuntimeError(f"Procrustes leakage audit failed: {leakage}")
    (data_dir / "leakage_audit.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    spectrum, dimension_splits, dimension_run, _ = intrinsic_dimension_analysis(
        runs, args.dimension_unit_splits
    )
    spectrum.to_csv(data_dir / "intrinsic_spectrum.csv", index=False)
    dimension_splits.to_csv(data_dir / "intrinsic_dimension_unit_splits.csv", index=False)
    dimension_run.to_csv(data_dir / "intrinsic_dimension_run_summary.csv", index=False)
    rank8_dimension_run = rank_limited_dimension(spectrum)
    rank8_dimension_run.to_csv(
        data_dir / "intrinsic_dimension_rank8_comparison.csv", index=False
    )
    dimension_patient = dimension_run.groupby(["patient", "view"], as_index=False).agg(
        effective_dimension=("effective_dimension", "mean"),
        variance_rank2=("variance_rank2", "mean"),
        variance_rank4=("variance_rank4", "mean"),
        split_median_effective_dimension=("split_median_effective_dimension", "mean"),
    )
    rank8_dimension_patient = rank8_dimension_run.groupby(
        ["patient", "view"], as_index=False
    ).agg(
        rank_limited_effective_dimension=("rank_limited_effective_dimension", "mean"),
        captured_full_spectrum_fraction=("captured_full_spectrum_fraction", "mean"),
    )
    dimension_patient = dimension_patient.merge(
        rank8_dimension_patient, on=["patient", "view"], how="left"
    )
    dimension_patient.to_csv(data_dir / "intrinsic_dimension_patient_summary.csv", index=False)
    dimension_group = dimension_patient.groupby("view", as_index=False).mean(numeric_only=True)
    dimension_group.to_csv(data_dir / "intrinsic_dimension_group_summary.csv", index=False)

    procrustes_folds = pd.concat([procrustes_cv(run) for run in runs], ignore_index=True)
    source_dir = shared_latent_data_dir(args.unit_set)
    pls_path = source_dir / "pls_cv_folds.csv"
    if not pls_path.exists():
        raise FileNotFoundError(f"Frozen PLS folds not found: {pls_path}")
    pls_folds = pd.read_csv(pls_path)
    pls_folds["model"] = "PLS"
    common_columns = [
        "patient",
        "run",
        "held_pair_1",
        "held_pair_2",
        "rank",
        "model",
        "retrieval_margin",
        "two_by_two_accuracy",
        "matched_cosine",
        "matched_angle_deg",
    ]
    static_folds = pd.concat(
        [procrustes_folds[common_columns], pls_folds[common_columns]], ignore_index=True
    )
    static_folds.to_csv(data_dir / "static_model_cv_folds.csv", index=False)
    static_run, static_patient, static_group, contrasts = summarize_static_models(
        static_folds, args.n_bootstrap
    )
    static_run.to_csv(data_dir / "static_model_run_summary.csv", index=False)
    static_patient.to_csv(data_dir / "static_model_patient_summary.csv", index=False)
    contrasts.to_csv(data_dir / "static_model_contrasts_patient.csv", index=False)

    permutation_null, permutation_summary = synchronized_procrustes_permutation(
        runs,
        static_run,
        args.procrustes_permutations,
    )
    permutation_null.to_csv(
        data_dir / "procrustes_synchronized_pair_permutation_null.csv", index=False
    )
    for rank in RANKS:
        for suffix, metric_name in (
            ("margin", f"procrustes_rank{rank}_retrieval_margin"),
            ("accuracy", f"procrustes_rank{rank}_accuracy"),
        ):
            static_group.loc[
                static_group["metric"] == metric_name, "item_permutation_p_greater"
            ] = permutation_summary[f"rank{rank}_{suffix}_item_permutation_p_greater"]
    prior_group = pd.read_csv(source_dir / "group_inference.csv").set_index("metric")
    for rank in (PRIMARY_RANK,):
        for suffix, prior_metric, new_metric in (
            ("margin", f"pls_rank{rank}_retrieval_margin", f"pls_rank{rank}_retrieval_margin"),
            ("accuracy", f"pls_rank{rank}_accuracy", f"pls_rank{rank}_accuracy"),
        ):
            static_group.loc[
                static_group["metric"] == new_metric, "item_permutation_p_greater"
            ] = prior_group.loc[prior_metric, "item_permutation_p_greater"]
    static_group.to_csv(data_dir / "static_model_group_inference.csv", index=False)
    loo = model_leave_one_patient_out(static_patient)
    loo.to_csv(data_dir / "static_model_leave_one_patient_out.csv", index=False)

    dynamic_frames = []
    for run_index, run in enumerate(runs, start=1):
        frame = metrics[(metrics["patient"] == run.patient) & (metrics["run"] == run.run)]
        active_dynamic, passive_dynamic = shared.extract_dynamic_tensor(run, frame, args.unit_set)
        dynamic_frames.append(dynamic_pls_cv(run, active_dynamic, passive_dynamic))
        print(f"Dynamic latent CV: {run_index}/{len(runs)} runs", flush=True)
    dynamic_folds = pd.concat(dynamic_frames, ignore_index=True)
    dynamic_folds.to_csv(data_dir / "dynamic_latent_cv_folds.csv", index=False)
    dynamic_run, dynamic_patient, cell_stats, matrices = dynamic_inference(dynamic_folds)
    dynamic_run.to_csv(data_dir / "dynamic_latent_run_summary.csv", index=False)
    dynamic_patient.to_csv(data_dir / "dynamic_latent_patient_summary.csv", index=False)
    cell_stats.to_csv(data_dir / "dynamic_latent_cell_stats.csv", index=False)
    np.savez_compressed(data_dir / "dynamic_latent_patient_matrices.npz", **matrices)
    lag_patient, lag_group = dynamic_lag_analysis(matrices, args.n_bootstrap)
    lag_patient.to_csv(data_dir / "dynamic_lag_patient_summary.csv", index=False)
    lag_group.to_csv(data_dir / "dynamic_lag_group_inference.csv", index=False)

    make_dimension_alignment_figure(
        report_dir,
        spectrum,
        dimension_patient,
        static_patient,
        static_group,
        contrasts,
        permutation_null,
        permutation_summary,
    )
    make_dynamic_figure(report_dir, cell_stats, matrices, lag_patient, lag_group)

    summary_payload = {
        "analysis_contract": contract,
        "run_inclusion": audit.to_dict(orient="records"),
        "leakage_audit": leakage,
        "intrinsic_dimension_patient_summary": dimension_patient.to_dict(orient="records"),
        "static_model_group_inference": static_group.to_dict(orient="records"),
        "static_model_leave_one_patient_out": loo.to_dict(orient="records"),
        "procrustes_permutation": permutation_summary,
        "dynamic_margin_fwer_cells": int(
            (cell_stats["margin_exact_maxstat_p_fwer_greater"] < 0.05).sum()
        ),
        "dynamic_accuracy_fwer_cells": int(
            (cell_stats["accuracy_exact_maxstat_p_fwer_greater"] < 0.05).sum()
        ),
        "dynamic_lag_group_inference": lag_group.to_dict(orient="records"),
    }
    for path in (data_dir / "summary.json", report_dir / "summary.json"):
        path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        report_dir,
        contract,
        audit,
        dimension_group,
        static_group,
        permutation_summary,
        loo,
        cell_stats,
        lag_group,
        leakage,
    )
    refresh_manifest(report_dir, data_dir)

    if args.unit_set == "good_only":
        main_report_dir, main_data_dir = report_dir, data_dir
    else:
        main_report_dir, main_data_dir = report_dir.parent, data_dir.parent
    write_combined_if_available(main_report_dir, main_data_dir)
    if main_report_dir.exists() and main_data_dir.exists():
        refresh_manifest(main_report_dir, main_data_dir)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


def postprocess_existing(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    data_dir = Path(args.data_dir)
    required = [
        report_dir / "analysis_contract.json",
        data_dir / "summary.json",
        data_dir / "intrinsic_spectrum.csv",
        data_dir / "static_model_group_inference.csv",
        data_dir / "dynamic_latent_patient_matrices.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing existing outputs: {missing}")
    contract = json.loads((report_dir / "analysis_contract.json").read_text(encoding="utf-8"))
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(data_dir / "run_inclusion_audit.csv")
    leakage = json.loads((data_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    spectrum = pd.read_csv(data_dir / "intrinsic_spectrum.csv")
    dimension_patient = pd.read_csv(data_dir / "intrinsic_dimension_patient_summary.csv")
    dimension_patient = dimension_patient.drop(
        columns=["rank_limited_effective_dimension", "captured_full_spectrum_fraction"],
        errors="ignore",
    )
    rank8_dimension_run = rank_limited_dimension(spectrum)
    rank8_dimension_run.to_csv(
        data_dir / "intrinsic_dimension_rank8_comparison.csv", index=False
    )
    rank8_dimension_patient = rank8_dimension_run.groupby(
        ["patient", "view"], as_index=False
    ).agg(
        rank_limited_effective_dimension=("rank_limited_effective_dimension", "mean"),
        captured_full_spectrum_fraction=("captured_full_spectrum_fraction", "mean"),
    )
    dimension_patient = dimension_patient.merge(
        rank8_dimension_patient, on=["patient", "view"], how="left"
    )
    dimension_patient.to_csv(data_dir / "intrinsic_dimension_patient_summary.csv", index=False)
    dimension_group = dimension_patient.groupby("view", as_index=False).mean(numeric_only=True)
    dimension_group.to_csv(data_dir / "intrinsic_dimension_group_summary.csv", index=False)
    summary["intrinsic_dimension_patient_summary"] = dimension_patient.to_dict(orient="records")
    for path in (data_dir / "summary.json", report_dir / "summary.json"):
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    static_patient = pd.read_csv(data_dir / "static_model_patient_summary.csv")
    static_group = pd.read_csv(data_dir / "static_model_group_inference.csv")
    contrasts = pd.read_csv(data_dir / "static_model_contrasts_patient.csv")
    permutation_null = pd.read_csv(
        data_dir / "procrustes_synchronized_pair_permutation_null.csv"
    )
    permutation_summary = summary["procrustes_permutation"]
    loo = pd.read_csv(data_dir / "static_model_leave_one_patient_out.csv")
    cell_stats = pd.read_csv(data_dir / "dynamic_latent_cell_stats.csv")
    with np.load(data_dir / "dynamic_latent_patient_matrices.npz", allow_pickle=True) as archive:
        matrices = {key: np.asarray(archive[key]) for key in archive.files}
    lag_patient = pd.read_csv(data_dir / "dynamic_lag_patient_summary.csv")
    lag_group = pd.read_csv(data_dir / "dynamic_lag_group_inference.csv")

    make_dimension_alignment_figure(
        report_dir,
        spectrum,
        dimension_patient,
        static_patient,
        static_group,
        contrasts,
        permutation_null,
        permutation_summary,
    )
    make_dynamic_figure(report_dir, cell_stats, matrices, lag_patient, lag_group)
    write_report(
        report_dir,
        contract,
        audit,
        dimension_group,
        static_group,
        permutation_summary,
        loo,
        cell_stats,
        lag_group,
        leakage,
    )
    refresh_manifest(report_dir, data_dir)
    if args.unit_set == "good_only":
        main_report_dir, main_data_dir = report_dir, data_dir
    else:
        main_report_dir, main_data_dir = report_dir.parent, data_dir.parent
    write_combined_if_available(main_report_dir, main_data_dir)
    refresh_manifest(main_report_dir, main_data_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=("good_only", "usable_lenient"), default="good_only")
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--dimension-unit-splits", type=int, default=500)
    parser.add_argument("--procrustes-permutations", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="Regenerate figures, reports, comparison tables, and manifests from saved outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.postprocess_existing:
        postprocess_existing(parsed_args)
    else:
        run_analysis(parsed_args)
