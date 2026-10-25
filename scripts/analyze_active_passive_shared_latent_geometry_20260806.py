#!/usr/bin/env python3
"""Shared-latent and local geometry analyses for matched active/passive nouns.

This analysis follows the evidence chain in the bilingual hippocampal-neuron
study without assuming that a positive result must exist:

1. cross-construction single-unit tuning correlations;
2. cross-validated two-view PLS axes that may differ by construction while
   defining a common held-out latent space;
3. local neighborhood/triplet preservation when a global rigid map fails;
4. unit-sampling stability and limited cross-run reliability estimates;
5. fixed 8 x 8 time-resolved RDM analysis with exact patient max statistics.

All inferential endpoints are first reduced to one value per patient.  The
active/passive contrast remains fully confounded with marker identity,
acoustics, surface position, and sentence form.  Agent/Patient also remains
confounded with animacy and lexical class.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import analyze_active_passive_geometry_transform_20260806 as base


ROOT = base.ROOT
DATA_ROOT = base.DATA_ROOT
DEFAULT_REPORT_DIR = ROOT / "reports/active_passive_shared_latent_geometry_20260806"
DEFAULT_DATA_DIR = DATA_ROOT / "combined_active_passive_shared_latent_geometry_20260806"

PRIMARY_RANK = 2
SENSITIVITY_RANK = 4
LATENT_RANKS = (PRIMARY_RANK, SENSITIVITY_RANK)
PRE_PCA_RANK = 8
LOCAL_K = 2
DYNAMIC_WINDOWS_MS = tuple((start, start + 100) for start in range(0, 800, 100))

OKABE_ITO = base.OKABE_ITO


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size != x.size or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    denom = an * bn
    denom[denom == 0] = 1.0
    return np.sum(a * b, axis=1) / denom


def role_center_pair(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Training-only condition-specific role centering with pooled unit scale."""
    xtr = np.asarray(x_train, dtype=float).copy()
    ytr = np.asarray(y_train, dtype=float).copy()
    xte = np.asarray(x_test, dtype=float).copy()
    yte = np.asarray(y_test, dtype=float).copy()
    train_roles = train_meta["thematic_role"].astype(str).to_numpy()
    test_roles = test_meta["thematic_role"].astype(str).to_numpy()
    for role in base.ROLES:
        train_idx = train_roles == role
        test_idx = test_roles == role
        mean_x = xtr[train_idx].mean(axis=0, keepdims=True)
        mean_y = ytr[train_idx].mean(axis=0, keepdims=True)
        xtr[train_idx] -= mean_x
        ytr[train_idx] -= mean_y
        xte[test_idx] -= mean_x
        yte[test_idx] -= mean_y
    pooled = np.vstack([xtr, ytr])
    unit_sd = pooled.std(axis=0, ddof=1)
    unit_sd[~np.isfinite(unit_sd) | (unit_sd == 0)] = 1.0
    return xtr / unit_sd, ytr / unit_sd, xte / unit_sd, yte / unit_sd, unit_sd


def pca_project(
    train: np.ndarray,
    test: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
    actual = min(rank, vt.shape[0], vt.shape[1])
    components = vt[:actual].T
    return (train - mean) @ components, (test - mean) @ components, components


def fit_two_view_pls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    latent_rank: int,
    pre_pca_rank: int = PRE_PCA_RANK,
) -> dict[str, np.ndarray | float]:
    """Fit separate active/passive readout axes using training pairs only."""
    xtr, ytr, xte, yte, unit_sd = role_center_pair(
        x_train, y_train, x_test, y_test, train_meta, test_meta
    )
    available_pre_rank = min(pre_pca_rank, len(xtr) - 2, xtr.shape[1], ytr.shape[1])
    xp_tr, xp_te, pc_x = pca_project(xtr, xte, available_pre_rank)
    yp_tr, yp_te, pc_y = pca_project(ytr, yte, available_pre_rank)
    cross = xp_tr.T @ yp_tr / max(1, len(xp_tr) - 1)
    u, singular, vt = np.linalg.svd(cross, full_matrices=False)
    actual_rank = min(latent_rank, len(singular))
    u = u[:, :actual_rank]
    v = vt.T[:, :actual_rank]
    singular = singular[:actual_rank]
    sx_tr = xp_tr @ u
    sy_tr = yp_tr @ v
    sx_te = xp_te @ u
    sy_te = yp_te @ v
    sd_x = sx_tr.std(axis=0, ddof=1)
    sd_y = sy_tr.std(axis=0, ddof=1)
    sd_x[~np.isfinite(sd_x) | (sd_x == 0)] = 1.0
    sd_y[~np.isfinite(sd_y) | (sd_y == 0)] = 1.0
    weights = np.sqrt(np.maximum(singular, 1e-12))
    weights /= np.linalg.norm(weights) if np.linalg.norm(weights) > 0 else 1.0
    sx_tr = sx_tr / sd_x * weights
    sy_tr = sy_tr / sd_y * weights
    sx_te = sx_te / sd_x * weights
    sy_te = sy_te / sd_y * weights
    loading_x = (pc_x @ u) / unit_sd[:, None]
    loading_y = (pc_y @ v) / unit_sd[:, None]
    qx, _ = np.linalg.qr(loading_x)
    qy, _ = np.linalg.qr(loading_y)
    axis_cosines = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return {
        "x_train_scores": sx_tr,
        "y_train_scores": sy_tr,
        "x_test_scores": sx_te,
        "y_test_scores": sy_te,
        "loading_x": loading_x,
        "loading_y": loading_y,
        "singular_values": singular,
        "mean_principal_angle_deg": float(
            np.degrees(np.mean(np.arccos(np.clip(axis_cosines, -1.0, 1.0))))
        ),
        "axis_alignment": float(np.mean(np.square(axis_cosines))),
    }


def retrieval_metrics(
    active_scores: np.ndarray,
    passive_scores: np.ndarray,
    meta: pd.DataFrame,
) -> dict[str, float]:
    accuracy = base.within_role_two_by_two(active_scores, passive_scores, meta)
    margins = []
    matched_cosines = []
    for role in base.ROLES:
        idx = base.role_indices(meta, role)
        distance = base.cosine_distance_matrix(active_scores[idx], passive_scores[idx])
        if len(idx) == 2:
            correct = distance[0, 0] + distance[1, 1]
            swapped = distance[0, 1] + distance[1, 0]
            margins.append((swapped - correct) / 2.0)
        matched_cosines.extend(cosine_rows(active_scores[idx], passive_scores[idx]).tolist())
    return {
        "two_by_two_accuracy": float(accuracy),
        "retrieval_margin": float(np.mean(margins)) if margins else np.nan,
        "matched_cosine": float(np.mean(matched_cosines)),
        "matched_angle_deg": float(np.degrees(np.arccos(np.clip(np.mean(matched_cosines), -1.0, 1.0)))),
    }


def pls_cv(
    run: base.RunData,
    ranks: Iterable[int] = LATENT_RANKS,
    passive_pair_permutation: np.ndarray | None = None,
) -> pd.DataFrame:
    passive = (
        run.passive
        if passive_pair_permutation is None
        else base.reorder_passive_by_pair(run.passive, run.item_meta, passive_pair_permutation)
    )
    rows = []
    for held_pairs, train, test in base.outer_folds(run.item_meta):
        train_meta = run.item_meta.loc[train].reset_index(drop=True)
        test_meta = run.item_meta.loc[test].reset_index(drop=True)
        for rank in ranks:
            fit = fit_two_view_pls(
                run.active[train],
                passive[train],
                run.active[test],
                passive[test],
                train_meta,
                test_meta,
                latent_rank=rank,
            )
            metrics = retrieval_metrics(
                np.asarray(fit["x_test_scores"]),
                np.asarray(fit["y_test_scores"]),
                test_meta,
            )
            rows.append(
                {
                    "patient": run.patient,
                    "run": run.run,
                    "held_pair_1": held_pairs[0],
                    "held_pair_2": held_pairs[1],
                    "rank": rank,
                    **metrics,
                    "mean_principal_angle_deg": fit["mean_principal_angle_deg"],
                    "axis_alignment": fit["axis_alignment"],
                }
            )
    return pd.DataFrame(rows)


def synchronized_pls_permutation(
    runs: list[base.RunData],
    n_perm: int,
    rank: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    real_rows = []
    for run in runs:
        frame = pls_cv(run, ranks=(rank,))
        real_rows.append(
            {
                "patient": run.patient,
                "run": run.run,
                "margin": frame["retrieval_margin"].mean(),
                "accuracy": frame["two_by_two_accuracy"].mean(),
            }
        )
    real = pd.DataFrame(real_rows)
    observed_margin = float(real.groupby("patient")["margin"].mean().mean())
    observed_accuracy = float(real.groupby("patient")["accuracy"].mean().mean())
    rng = np.random.default_rng(seed)
    null_rows = []
    for perm_idx in range(n_perm):
        permutation = rng.permutation(10)
        perm_rows = []
        for run in runs:
            frame = pls_cv(run, ranks=(rank,), passive_pair_permutation=permutation)
            perm_rows.append(
                {
                    "patient": run.patient,
                    "margin": frame["retrieval_margin"].mean(),
                    "accuracy": frame["two_by_two_accuracy"].mean(),
                }
            )
        perm_frame = pd.DataFrame(perm_rows)
        null_rows.append(
            {
                "permutation_index": perm_idx,
                "retrieval_margin": perm_frame.groupby("patient")["margin"].mean().mean(),
                "two_by_two_accuracy": perm_frame.groupby("patient")["accuracy"].mean().mean(),
            }
        )
    null = pd.DataFrame(null_rows)
    summary = {
        "observed_retrieval_margin": observed_margin,
        "observed_two_by_two_accuracy": observed_accuracy,
        "margin_item_permutation_p_greater": float(
            (np.sum(null["retrieval_margin"] >= observed_margin) + 1) / (n_perm + 1)
        ),
        "accuracy_item_permutation_p_greater": float(
            (np.sum(null["two_by_two_accuracy"] >= observed_accuracy) + 1) / (n_perm + 1)
        ),
    }
    return null, summary


def rank_columns(values: np.ndarray) -> np.ndarray:
    ranked = stats.rankdata(values, axis=0)
    ranked -= ranked.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(np.square(ranked), axis=0, keepdims=True))
    norm[norm == 0] = 1.0
    return ranked / norm


def tuning_correlations(
    run: base.RunData,
    n_perm: int,
) -> pd.DataFrame:
    role_active = {}
    role_passive = {}
    observed = {}
    for role in base.ROLES:
        idx = base.role_indices(run.item_meta, role)
        role_active[role] = rank_columns(run.active[idx])
        role_passive[role] = rank_columns(run.passive[idx])
        observed[role] = np.sum(role_active[role] * role_passive[role], axis=0)
    observed_z = np.mean(
        [np.arctanh(np.clip(observed[role], -0.999999, 0.999999)) for role in base.ROLES],
        axis=0,
    )
    null = np.empty((n_perm, len(run.unit_ids)), dtype=float)
    rng = np.random.default_rng(base.stable_seed("shared-latent-unit-tuning", run.run))
    for perm_idx in range(n_perm):
        permutation = rng.permutation(10)
        role_null = []
        for role in base.ROLES:
            rho = np.sum(
                role_active[role] * role_passive[role][permutation],
                axis=0,
            )
            role_null.append(np.arctanh(np.clip(rho, -0.999999, 0.999999)))
        null[perm_idx] = np.mean(role_null, axis=0)
    p_greater = (np.sum(null >= observed_z[None, :], axis=0) + 1) / (n_perm + 1)
    return pd.DataFrame(
        {
            "patient": run.patient,
            "run": run.run,
            "unit_id": run.unit_ids,
            "agent_rho": observed["agent"],
            "patient_rho": observed["patient"],
            "role_equal_fisher_z": observed_z,
            "role_equal_rho": np.tanh(observed_z),
            "positive_both_roles": (observed["agent"] > 0) & (observed["patient"] > 0),
            "permutation_p_greater": p_greater,
        }
    )


def fdr_unit_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["q_bh_global"] = base.bh_fdr(out["permutation_p_greater"])
    out["q_bh_within_run"] = np.nan
    for run, idx in out.groupby("run").groups.items():
        out.loc[idx, "q_bh_within_run"] = base.bh_fdr(out.loc[idx, "permutation_p_greater"])
    return out


def role_center_full(
    active: np.ndarray,
    passive: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(active, dtype=float).copy()
    passive = np.asarray(passive, dtype=float).copy()
    for role in base.ROLES:
        idx = base.role_indices(meta, role)
        active[idx] -= active[idx].mean(axis=0, keepdims=True)
        passive[idx] -= passive[idx].mean(axis=0, keepdims=True)
    pooled = np.vstack([active, passive])
    sd = pooled.std(axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return active / sd, passive / sd


def triplet_consistency(distance_a: np.ndarray, distance_b: np.ndarray) -> float:
    scores = []
    n = distance_a.shape[0]
    for anchor in range(n):
        others = [idx for idx in range(n) if idx != anchor]
        for left, right in itertools.combinations(others, 2):
            delta_a = distance_a[anchor, left] - distance_a[anchor, right]
            delta_b = distance_b[anchor, left] - distance_b[anchor, right]
            if np.isclose(delta_a, 0) or np.isclose(delta_b, 0):
                scores.append(0.5)
            else:
                scores.append(float(np.sign(delta_a) == np.sign(delta_b)))
    return float(np.mean(scores))


def knn_overlap(distance_a: np.ndarray, distance_b: np.ndarray, k: int) -> float:
    overlaps = []
    for idx in range(distance_a.shape[0]):
        nn_a = set(np.argsort(distance_a[idx])[1 : k + 1].tolist())
        nn_b = set(np.argsort(distance_b[idx])[1 : k + 1].tolist())
        overlaps.append(len(nn_a & nn_b) / k)
    return float(np.mean(overlaps))


def local_geometry_endpoint(
    run: base.RunData,
    passive_pair_permutation: np.ndarray | None = None,
    k: int = LOCAL_K,
) -> dict[str, float]:
    active, passive = role_center_full(run.active, run.passive, run.item_meta)
    if passive_pair_permutation is not None:
        passive = base.reorder_passive_by_pair(passive, run.item_meta, passive_pair_permutation)
    triplets = []
    overlaps = []
    for role in base.ROLES:
        idx = base.role_indices(run.item_meta, role)
        rdm_a = base.cosine_rdm(active[idx])
        rdm_p = base.cosine_rdm(passive[idx])
        triplets.append(triplet_consistency(rdm_a, rdm_p))
        overlaps.append(knn_overlap(rdm_a, rdm_p, k))
    return {
        "triplet_consistency": float(np.mean(triplets)),
        "knn_overlap": float(np.mean(overlaps)),
    }


def synchronized_local_permutation(
    runs: list[base.RunData],
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    real_rows = []
    for run in runs:
        real_rows.append({"patient": run.patient, **local_geometry_endpoint(run)})
    real = pd.DataFrame(real_rows)
    observed_triplet = float(real.groupby("patient")["triplet_consistency"].mean().mean())
    observed_knn = float(real.groupby("patient")["knn_overlap"].mean().mean())
    rng = np.random.default_rng(seed)
    rows = []
    for perm_idx in range(n_perm):
        permutation = rng.permutation(10)
        values = []
        for run in runs:
            values.append(
                {
                    "patient": run.patient,
                    **local_geometry_endpoint(run, passive_pair_permutation=permutation),
                }
            )
        frame = pd.DataFrame(values).groupby("patient").mean(numeric_only=True)
        rows.append(
            {
                "permutation_index": perm_idx,
                "triplet_consistency": frame["triplet_consistency"].mean(),
                "knn_overlap": frame["knn_overlap"].mean(),
            }
        )
    null = pd.DataFrame(rows)
    summary = {
        "observed_triplet_consistency": observed_triplet,
        "triplet_item_permutation_p_greater": float(
            (np.sum(null["triplet_consistency"] >= observed_triplet) + 1) / (n_perm + 1)
        ),
        "observed_knn_overlap": observed_knn,
        "knn_item_permutation_p_greater": float(
            (np.sum(null["knn_overlap"] >= observed_knn) + 1) / (n_perm + 1)
        ),
    }
    return null, summary


def within_condition_split_rdm(
    matrix: np.ndarray,
    meta: pd.DataFrame,
    left_units: np.ndarray,
    right_units: np.ndarray,
) -> float:
    role_z = []
    for role in base.ROLES:
        idx = base.role_indices(meta, role)
        rho = base.spearman_r(
            base.rdm_vector(base.cosine_rdm(matrix[idx][:, left_units])),
            base.rdm_vector(base.cosine_rdm(matrix[idx][:, right_units])),
        )
        role_z.append(base.fisher_z(rho))
    return float(np.nanmean(role_z))


def unit_split_reliability(run: base.RunData, n_splits: int) -> pd.DataFrame:
    active, passive = role_center_full(run.active, run.passive, run.item_meta)
    rng = np.random.default_rng(base.stable_seed("unit-split-reliability", run.run))
    rows = []
    for split_idx in range(n_splits):
        order = rng.permutation(len(run.unit_ids))
        half = len(order) // 2
        left = order[:half]
        right = order[half : 2 * half]
        active_z = within_condition_split_rdm(active, run.item_meta, left, right)
        passive_z = within_condition_split_rdm(passive, run.item_meta, left, right)
        rows.append(
            {
                "patient": run.patient,
                "run": run.run,
                "split_index": split_idx,
                "active_fisher_z": active_z,
                "passive_fisher_z": passive_z,
                "role_condition_equal_fisher_z": float(np.nanmean([active_z, passive_z])),
            }
        )
    return pd.DataFrame(rows)


def cross_run_reliability(runs: list[base.RunData]) -> pd.DataFrame:
    rows = []
    for patient in sorted({run.patient for run in runs}):
        patient_runs = [run for run in runs if run.patient == patient]
        for left, right in itertools.combinations(patient_runs, 2):
            for voice in base.VOICES:
                left_matrix = left.active if voice == "active" else left.passive
                right_matrix = right.active if voice == "active" else right.passive
                left_centered, _ = role_center_full(left_matrix, left_matrix.copy(), left.item_meta)
                right_centered, _ = role_center_full(right_matrix, right_matrix.copy(), right.item_meta)
                role_z = []
                for role in base.ROLES:
                    idx_l = base.role_indices(left.item_meta, role)
                    idx_r = base.role_indices(right.item_meta, role)
                    rho = base.spearman_r(
                        base.rdm_vector(base.cosine_rdm(left_centered[idx_l])),
                        base.rdm_vector(base.cosine_rdm(right_centered[idx_r])),
                    )
                    role_z.append(base.fisher_z(rho))
                rows.append(
                    {
                        "patient": patient,
                        "run_left": left.run,
                        "run_right": right.run,
                        "voice": voice,
                        "role_equal_fisher_z": float(np.mean(role_z)),
                        "role_equal_rho": base.inv_fisher(float(np.mean(role_z))),
                    }
                )
    return pd.DataFrame(rows)


def full_readout_axes(
    run: base.RunData,
    rank: int,
    n_perm: int,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray]:
    fit = fit_two_view_pls(
        run.active,
        run.passive,
        run.active,
        run.passive,
        run.item_meta,
        run.item_meta,
        latent_rank=rank,
    )
    loading_x = np.asarray(fit["loading_x"])
    loading_y = np.asarray(fit["loading_y"])
    qx, _ = np.linalg.qr(loading_x)
    qy, _ = np.linalg.qr(loading_y)
    participation_x = np.sum(np.square(loading_x), axis=1)
    participation_y = np.sum(np.square(loading_y), axis=1)
    participation_rho = base.spearman_r(participation_x, participation_y)
    alignment = float(fit["axis_alignment"])
    rng = np.random.default_rng(base.stable_seed("shared-latent-axis-null", run.run, rank))
    null_alignment = np.empty(n_perm, dtype=float)
    null_participation = np.empty(n_perm, dtype=float)
    for perm_idx in range(n_perm):
        permutation = rng.permutation(len(run.unit_ids))
        singular = np.linalg.svd(qx.T @ qy[permutation], compute_uv=False)
        null_alignment[perm_idx] = np.mean(np.square(singular))
        null_participation[perm_idx] = base.spearman_r(
            participation_x, participation_y[permutation]
        )
    null_sd = null_alignment.std(ddof=1)
    summary = {
        "patient": run.patient,
        "run": run.run,
        "rank": rank,
        "mean_principal_angle_deg": fit["mean_principal_angle_deg"],
        "axis_alignment": alignment,
        "axis_alignment_null_mean": float(null_alignment.mean()),
        "axis_alignment_null_z": float((alignment - null_alignment.mean()) / null_sd)
        if null_sd > 0
        else np.nan,
        "axis_alignment_permutation_p": float(
            (np.sum(null_alignment >= alignment) + 1) / (n_perm + 1)
        ),
        "participation_rho": participation_rho,
        "participation_permutation_p_two_sided": float(
            (np.sum(np.abs(null_participation) >= abs(participation_rho)) + 1) / (n_perm + 1)
        ),
    }
    units = pd.DataFrame(
        {
            "patient": run.patient,
            "run": run.run,
            "unit_id": run.unit_ids,
            "active_axis_participation": participation_x,
            "passive_axis_participation": participation_y,
        }
    )
    return (
        summary,
        units,
        np.asarray(fit["x_test_scores"]),
        np.asarray(fit["y_test_scores"]),
    )


def extract_dynamic_tensor(
    run: base.RunData,
    frame: pd.DataFrame,
    unit_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    archive_path = base.unit_archive_path(run.patient, run.run, unit_set)
    with np.load(archive_path, allow_pickle=True) as archive:
        firing_rate = np.asarray(archive["firing_rate_hz"], dtype=float)
        archive_unit_ids = np.asarray(archive["unit_ids"], dtype=int)
    lookup = {int(unit_id): idx for idx, unit_id in enumerate(archive_unit_ids)}
    firing_rate = firing_rate[[lookup[int(unit_id)] for unit_id in run.unit_ids]]
    event_rows = frame.drop_duplicates(
        ["voice", "pair_id", "thematic_role", "noun", "trial_index"]
    )[["voice", "pair_id", "thematic_role", "noun", "trial_index", "event_bin_1000Hz"]]
    output = {
        voice: np.full(
            (len(run.item_meta), len(DYNAMIC_WINDOWS_MS), len(run.unit_ids)),
            np.nan,
            dtype=float,
        )
        for voice in base.VOICES
    }
    for voice in base.VOICES:
        for _, item in run.item_meta.iterrows():
            row = event_rows[
                (event_rows["voice"] == voice)
                & (event_rows["pair_id"] == int(item["pair_id"]))
                & (event_rows["thematic_role"] == item["thematic_role"])
            ]
            if len(row) != 1:
                continue
            event_bin = int(row.iloc[0]["event_bin_1000Hz"])
            baseline = firing_rate[:, event_bin - 500 : event_bin].mean(axis=1)
            for window_idx, (start, stop) in enumerate(DYNAMIC_WINDOWS_MS):
                response = firing_rate[:, event_bin + start : event_bin + stop].mean(axis=1)
                output[voice][int(item["item_index"]), window_idx] = response - baseline
    if not np.isfinite(output["active"]).all() or not np.isfinite(output["passive"]).all():
        raise RuntimeError(f"Dynamic extraction produced missing values for {run.run}")
    return output["active"], output["passive"]


def dynamic_rdm_matrix(
    run: base.RunData,
    dynamic_active: np.ndarray,
    dynamic_passive: np.ndarray,
) -> np.ndarray:
    matrix = np.full((len(DYNAMIC_WINDOWS_MS), len(DYNAMIC_WINDOWS_MS)), np.nan)
    for idx_a in range(len(DYNAMIC_WINDOWS_MS)):
        for idx_p in range(len(DYNAMIC_WINDOWS_MS)):
            role_z = []
            for role in base.ROLES:
                idx = base.role_indices(run.item_meta, role)
                active = dynamic_active[idx, idx_a].copy()
                passive = dynamic_passive[idx, idx_p].copy()
                active -= active.mean(axis=0, keepdims=True)
                passive -= passive.mean(axis=0, keepdims=True)
                pooled = np.vstack([active, passive])
                sd = pooled.std(axis=0, ddof=1)
                sd[~np.isfinite(sd) | (sd == 0)] = 1.0
                rho = base.spearman_r(
                    base.rdm_vector(base.cosine_rdm(active / sd)),
                    base.rdm_vector(base.cosine_rdm(passive / sd)),
                )
                role_z.append(base.fisher_z(rho))
            matrix[idx_a, idx_p] = float(np.mean(role_z))
    return matrix


def exact_patient_max_stat(patient_matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = patient_matrices.mean(axis=0)
    null_max = []
    for signs in itertools.product((-1.0, 1.0), repeat=patient_matrices.shape[0]):
        signed = patient_matrices * np.asarray(signs)[:, None, None]
        null_max.append(np.max(signed.mean(axis=0)))
    null_max = np.asarray(null_max)
    p_fwer = np.asarray(
        [np.mean(null_max >= value - 1e-12) for value in observed.ravel()]
    ).reshape(observed.shape)
    return observed, null_max, p_fwer


def summarize_metric(
    frame: pd.DataFrame,
    value_col: str,
    metric: str,
    reference: float,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    patient, summary = base.summarize_patient_metric(
        frame, value_col, metric, reference, n_bootstrap
    )
    return patient, summary


def compute_pls_leave_one_patient_out(pls_patient: pd.DataFrame) -> pd.DataFrame:
    """Quantify patient influence without treating LOO rows as independent tests."""
    rows = []
    for rank in sorted(pls_patient["rank"].unique()):
        frame = pls_patient[pls_patient["rank"] == rank].copy()
        all_margin = float(frame["retrieval_margin"].mean())
        all_accuracy = float(frame["two_by_two_accuracy"].mean())
        for excluded_patient in sorted(frame["patient"].unique()):
            kept = frame[frame["patient"] != excluded_patient]
            margin_values = kept["retrieval_margin"].to_numpy(dtype=float)
            accuracy_excess = kept["two_by_two_accuracy"].to_numpy(dtype=float) - 0.5
            margin_test = base.exact_sign_flip(margin_values, 0.0)
            accuracy_test = base.exact_sign_flip(accuracy_excess, 0.0)
            loo_margin = float(np.mean(margin_values))
            loo_accuracy = float(np.mean(accuracy_excess) + 0.5)
            rows.append(
                {
                    "rank": int(rank),
                    "excluded_patient": excluded_patient,
                    "n_remaining_patients": int(len(kept)),
                    "all_patient_mean_retrieval_margin": all_margin,
                    "loo_mean_retrieval_margin": loo_margin,
                    "delta_loo_minus_all_retrieval_margin": loo_margin - all_margin,
                    "loo_positive_margin_patients": int(np.sum(margin_values > 0)),
                    "loo_exact_signflip_p_greater_margin": margin_test["p_greater"],
                    "all_patient_mean_accuracy": all_accuracy,
                    "loo_mean_accuracy": loo_accuracy,
                    "delta_loo_minus_all_accuracy": loo_accuracy - all_accuracy,
                    "loo_above_chance_accuracy_patients": int(np.sum(accuracy_excess > 0)),
                    "loo_exact_signflip_p_greater_accuracy": accuracy_test["p_greater"],
                }
            )
    return pd.DataFrame(rows)


def configure_style() -> None:
    base.configure_style()
    plt.rcParams["savefig.dpi"] = 300


def patient_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    categories: list[str],
    category_col: str,
    value_col: str,
    colors: dict[str, str],
    reference: float | None,
) -> None:
    patients = sorted(frame["patient"].unique())
    offsets = np.linspace(-0.14, 0.14, len(patients))
    for x, category in enumerate(categories):
        sub = frame[frame[category_col] == category]
        for offset, patient in zip(offsets, patients):
            value = sub.loc[sub["patient"] == patient, value_col]
            if not value.empty:
                ax.scatter(
                    x + offset,
                    value.iloc[0],
                    s=28,
                    color=colors[category],
                    edgecolor="white",
                    linewidth=0.4,
                    alpha=0.9,
                )
        if not sub.empty:
            ax.scatter(x, sub[value_col].mean(), marker="_", s=190, color="black", linewidth=2)
    if reference is not None and np.isfinite(reference):
        ax.axhline(reference, color=OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set_xticks(range(len(categories)), categories)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def make_figures(
    report_dir: Path,
    reliability_patient: pd.DataFrame,
    unit_tuning: pd.DataFrame,
    tuning_patient: pd.DataFrame,
    pls_patient: pd.DataFrame,
    pls_null: pd.DataFrame,
    pls_summary: dict[str, float],
    axes_patient: pd.DataFrame,
    selected_run: base.RunData,
    selected_scores_active: np.ndarray,
    selected_scores_passive: np.ndarray,
    local_patient: pd.DataFrame,
    local_null: pd.DataFrame,
    local_summary: dict[str, float],
    dynamic_observed: np.ndarray,
    dynamic_fwer: np.ndarray,
    dynamic_patient_matrices: np.ndarray,
) -> None:
    configure_style()
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.0, 6.9))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.43)

    ax = fig.add_subplot(gs[0, 0])
    values = np.tanh(reliability_patient["value"])
    ax.scatter(np.zeros(len(values)), values, s=34, color=OKABE_ITO["blue"], edgecolor="white", linewidth=0.4)
    ax.scatter(0, values.mean(), marker="_", s=210, color="black", linewidth=2)
    ax.axhline(0, color=OKABE_ITO["gray"], ls="--", lw=0.9)
    ax.set_xticks([0], ["Repeated unit splits"])
    ax.set_ylabel("Within-condition RDM rho")
    ax.set_title("Unit-sampling stability")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-1, 1, 25)
    ax.hist(unit_tuning["role_equal_rho"], bins=bins, color=OKABE_ITO["light_gray"], edgecolor="white")
    ax.axvline(0, color=OKABE_ITO["gray"], ls="--", lw=0.9)
    for value in np.tanh(tuning_patient["value"]):
        ax.plot([value, value], [0, ax.get_ylim()[1] * 0.09], color=OKABE_ITO["purple"], lw=1.3)
    n_global = int((unit_tuning["q_bh_global"] < 0.05).sum())
    ax.text(0.03, 0.96, f"Global BH-FDR: {n_global}/{len(unit_tuning)} units", transform=ax.transAxes, va="top")
    ax.set_xlabel("Role-equal active-passive tuning rho")
    ax.set_ylabel("Units")
    ax.set_title("Single-unit tuning overlap")
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    rank_frame = pls_patient.copy()
    rank_frame["rank_label"] = rank_frame["rank"].map({2: "PLS rank 2", 4: "PLS rank 4"})
    patient_scatter(
        ax,
        rank_frame,
        ["PLS rank 2", "PLS rank 4"],
        "rank_label",
        "two_by_two_accuracy",
        {"PLS rank 2": OKABE_ITO["green"], "PLS rank 4": OKABE_ITO["orange"]},
        0.5,
    )
    ax.set_ylim(0.25, 0.80)
    ax.set_ylabel("Within-role 2-vs-2 accuracy")
    ax.set_title("Held-pair shared latent retrieval")
    panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(pls_null["retrieval_margin"], bins=28, color=OKABE_ITO["light_gray"], edgecolor="white")
    ax.axvline(pls_summary["observed_retrieval_margin"], color=OKABE_ITO["green"], lw=2, label="Matched")
    ax.axvline(0, color=OKABE_ITO["gray"], lw=0.9, ls="--", label="Zero margin")
    ax.set_xlabel("Patient-equal held-pair margin")
    ax.set_ylabel("Synchronized permutations")
    ax.set_title("Item-level PLS test")
    ax.legend(frameon=False)
    panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1])
    colors = {"agent": OKABE_ITO["blue"], "patient": OKABE_ITO["orange"]}
    for idx, row in selected_run.item_meta.iterrows():
        role = str(row["thematic_role"])
        ax.plot(
            [selected_scores_active[idx, 0], selected_scores_passive[idx, 0]],
            [selected_scores_active[idx, 1], selected_scores_passive[idx, 1]],
            color=colors[role],
            alpha=0.35,
            lw=0.8,
        )
        ax.scatter(selected_scores_active[idx, 0], selected_scores_active[idx, 1], s=20, marker="o", facecolor="none", edgecolor=colors[role], linewidth=0.8)
        ax.scatter(selected_scores_passive[idx, 0], selected_scores_passive[idx, 1], s=22, marker="^", color=colors[role], edgecolor="white", linewidth=0.3)
    ax.axhline(0, color=OKABE_ITO["light_gray"], lw=0.7)
    ax.axvline(0, color=OKABE_ITO["light_gray"], lw=0.7)
    ax.set_xlabel("Shared latent dimension 1")
    ax.set_ylabel("Shared latent dimension 2")
    ax.set_title("Descriptive full-data latent map")
    ax.text(0.02, 0.02, "circle=active; triangle=passive\nselected by CV margin; descriptive", transform=ax.transAxes, fontsize=6.5, va="bottom")
    panel_label(ax, "E")

    ax = fig.add_subplot(gs[1, 2])
    patient_scatter(
        ax,
        axes_patient,
        ["axis_alignment_null_z", "participation_rho"],
        "metric",
        "value",
        {"axis_alignment_null_z": OKABE_ITO["orange"], "participation_rho": OKABE_ITO["blue"]},
        0.0,
    )
    ax.set_xticklabels(["Axis alignment\nnull z", "Same-unit\nparticipation rho"])
    ax.set_ylabel("Patient endpoint")
    ax.set_title("Common units, distinct axes")
    panel_label(ax, "F")

    fig.suptitle("Cross-construction tuning and shared latent readout axes", fontsize=12, fontweight="bold", y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"Fig_1_shared_latent_evidence.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10.8, 6.7))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    patient_scatter(
        ax,
        local_patient,
        ["triplet_consistency", "knn_overlap"],
        "metric",
        "value",
        {"triplet_consistency": OKABE_ITO["purple"], "knn_overlap": OKABE_ITO["green"]},
        None,
    )
    ax.plot([-0.35, 0.35], [0.5, 0.5], color=OKABE_ITO["gray"], ls="--", lw=0.9)
    ax.plot([0.65, 1.35], [LOCAL_K / 9.0, LOCAL_K / 9.0], color=OKABE_ITO["gray"], ls="--", lw=0.9)
    ax.set_xticklabels(["Triplet\nconsistency", f"{LOCAL_K}-NN\noverlap"])
    ax.set_ylabel("Patient endpoint")
    ax.set_title("Local geometry preservation")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(local_null["triplet_consistency"], bins=28, color=OKABE_ITO["light_gray"], edgecolor="white")
    ax.axvline(local_summary["observed_triplet_consistency"], color=OKABE_ITO["purple"], lw=2, label="Matched")
    ax.set_xlabel("Patient-equal triplet consistency")
    ax.set_ylabel("Synchronized permutations")
    ax.set_title("Item-level local-geometry test")
    ax.legend(frameon=False)
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[:, 2])
    im = ax.imshow(np.tanh(dynamic_observed), cmap="RdBu_r", vmin=-0.35, vmax=0.35, origin="lower")
    labels = [f"{start}-{stop}" for start, stop in DYNAMIC_WINDOWS_MS]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Passive noun window (ms)")
    ax.set_ylabel("Active noun window (ms)")
    ax.set_title("Time-resolved RDM rho")
    for idx_a in range(len(labels)):
        for idx_p in range(len(labels)):
            marker = "*" if dynamic_fwer[idx_a, idx_p] < 0.05 else ""
            ax.text(idx_p, idx_a, f"{math.tanh(dynamic_observed[idx_a, idx_p]):.2f}{marker}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.62, pad=0.03, label="rho")
    panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    diag_patient = np.tanh(np.diagonal(dynamic_patient_matrices, axis1=1, axis2=2))
    for row in diag_patient:
        ax.plot(range(len(labels)), row, color=OKABE_ITO["light_gray"], lw=0.8)
    ax.plot(range(len(labels)), diag_patient.mean(axis=0), color=OKABE_ITO["purple"], marker="o", lw=2)
    ax.axhline(0, color=OKABE_ITO["gray"], ls="--", lw=0.9)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_xlabel("Matched noun window (ms)")
    ax.set_ylabel("RDM rho")
    ax.set_title("Diagonal temporal geometry")
    panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    text = (
        "Interpretation gate\n\n"
        "Reliable + PLS succeeds:\nshared latent geometry on distinct axes.\n\n"
        "Only local geometry succeeds:\nnon-rigid neighborhood preservation.\n\n"
        "Reliable + all fail:\ncontext-dependent remapping is supported.\n\n"
        "Low reliability + all fail:\ninconclusive measurement ceiling."
    )
    ax.text(0, 1, text, va="top", fontsize=8.2, linespacing=1.35)
    panel_label(ax, "E")

    fig.suptitle("Local and time-resolved cross-construction geometry", fontsize=12, fontweight="bold", y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"Fig_2_local_dynamic_geometry.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_dir: Path,
    contract: dict[str, object],
    audit: pd.DataFrame,
    group_inference: pd.DataFrame,
    pls_summary: dict[str, float],
    local_summary: dict[str, float],
    unit_tuning: pd.DataFrame,
    cross_run: pd.DataFrame,
    dynamic_fwer: np.ndarray,
    selected_run: str,
    pls_loo: pd.DataFrame,
) -> None:
    lookup = group_inference.set_index("metric").to_dict("index")
    reliability = lookup["unit_split_reliability_fisher_z"]
    tuning = lookup["unit_tuning_median_fisher_z"]
    pls_margin = lookup["pls_rank2_retrieval_margin"]
    pls_acc = lookup["pls_rank2_accuracy"]
    rank4_acc = lookup["pls_rank4_accuracy"]
    triplet = lookup["triplet_consistency"]
    knn = lookup["knn_overlap"]
    axis = lookup["axis_alignment_null_z"]
    participation = lookup["axis_participation_rho"]
    n_global = int((unit_tuning["q_bh_global"] < 0.05).sum())
    n_run = int((unit_tuning["q_bh_within_run"] < 0.05).sum())
    n_time = int(np.sum(dynamic_fwer < 0.05))
    rank2_loo = pls_loo[pls_loo["rank"] == PRIMARY_RANK].copy()
    rank2_loo["absolute_margin_change"] = rank2_loo[
        "delta_loo_minus_all_retrieval_margin"
    ].abs()
    influential = rank2_loo.sort_values("absolute_margin_change", ascending=False).iloc[0]

    def p_text(value: float) -> str:
        return f"{value:.4f}" if np.isfinite(value) else "NA"

    if (
        pls_margin["exact_signflip_p_greater"] < 0.05
        and pls_summary["margin_item_permutation_p_greater"] < 0.05
    ):
        headline = "严格留出PLS支持主动/被动在不同读出轴上共享低维关系结构。"
    else:
        headline = "严格留出PLS尚未同时通过患者与刺激两个推广检验，不能确认共享潜在几何。"
    if (
        triplet["exact_signflip_p_greater"] < 0.05
        and local_summary["triplet_item_permutation_p_greater"] < 0.05
    ):
        local_line = "局部名词距离次序得到独立支持。"
    else:
        local_line = "局部几何也未同时通过患者与刺激检验。"

    cross_run_text = "无可用双run患者。"
    if not cross_run.empty:
        cross_run_text = ", ".join(
            f"{row.patient}-{row.voice}: rho={row.role_equal_rho:.3f}"
            for row in cross_run.itertuples(index=False)
        )

    lines = [
        "# 主动/被动共享潜在几何：跨句式单元、读出轴与局部结构",
        "",
        "## 一句话结果",
        "",
        f"{headline}{local_line}",
        "",
        "## 冻结规格",
        "",
        f"- unit set：`{contract['unit_set']}`；伪迹阈值：`{contract['artifact_threshold']}`。",
        f"- 纳入 {int(audit['included'].sum())} runs、{audit.loc[audit['included'], 'patient'].nunique()} patients；患者等权。",
        "- PLS主rank=2，rank=4敏感性；每折完整留出两个sentence pairs。",
        "- 所有role centering、unit scaling、PCA和PLS axes只用训练项目。",
        "- 置换始终用同一个10-pair shuffle同步作用于所有患者、runs和两个roles。",
        "- 单神经元、triplets、RDM distances和CV folds均不作为独立组水平样本。",
        "",
        "## 1. 几何是否具有可测可靠性",
        "",
        f"- 重复unit-split的患者等权RDM rho={math.tanh(reliability['patient_mean']):.3f}，95% bootstrap CI（Fisher-z）[{reliability['bootstrap_ci_low']:.3f}, {reliability['bootstrap_ci_high']:.3f}]，单侧exact P={p_text(reliability['exact_signflip_p_greater'])}。",
        f"- 双run同条件RDM仅作描述：{cross_run_text}",
        "- unit-split衡量population sampling stability，不是重复试次noise ceiling。",
        "",
        "## 2. 是否存在cross-construction单神经元",
        "",
        f"- 患者等权unit-median tuning rho={math.tanh(tuning['patient_mean']):.3f}，单侧exact P={p_text(tuning['exact_signflip_p_greater'])}。",
        f"- 全局BH-FDR q<0.05：{n_global}/{len(unit_tuning)} units；run内BH-FDR：{n_run}/{len(unit_tuning)} units。",
        "- 每个noun-condition在单run仅一次呈现；单元结果即使过FDR也只能称候选cross-construction tuning。",
        "",
        "## 3. 不同读出轴上是否存在共享潜在空间",
        "",
        f"- Rank-2 held-pair retrieval margin={pls_margin['patient_mean']:.3f}，患者exact P={p_text(pls_margin['exact_signflip_p_greater'])}，同步句对置换P={p_text(pls_summary['margin_item_permutation_p_greater'])}。",
        f"- Rank-2 2-vs-2 accuracy={pls_acc['patient_mean']:.3f}，95% CI [{pls_acc['bootstrap_ci_low']:.3f}, {pls_acc['bootstrap_ci_high']:.3f}]，刺激置换P={p_text(pls_summary['accuracy_item_permutation_p_greater'])}。",
        f"- Rank-4 accuracy={rank4_acc['patient_mean']:.3f}，患者exact P={p_text(rank4_acc['exact_signflip_p_greater'])}。",
        f"- LOO影响诊断中，排除{influential['excluded_patient']}对Rank-2 margin影响最大：margin={influential['loo_mean_retrieval_margin']:.3f}（全体={influential['all_patient_mean_retrieval_margin']:.3f}），accuracy={influential['loo_mean_accuracy']:.3f}（全体={influential['all_patient_mean_accuracy']:.3f}）。",
        "- 刺激层置换结果不能替代患者层推广；LOO仅用于识别影响来源，不作为新增确认性检验。",
        f"- Full-data readout-axis alignment相对unit shuffle的患者平均z={axis['patient_mean']:.3f}；same-unit participation rho={participation['patient_mean']:.3f}。",
        f"- Fig.1E选择CV margin最高的run（{selected_run}）作描述性坐标图，不参与推断。",
        "",
        "## 4. 局部与动态几何",
        "",
        f"- Triplet-order consistency={triplet['patient_mean']:.3f}；患者P={p_text(triplet['exact_signflip_p_greater'])}；同步句对置换P={p_text(local_summary['triplet_item_permutation_p_greater'])}。",
        f"- {LOCAL_K}-NN overlap={knn['patient_mean']:.3f}；患者P={p_text(knn['exact_signflip_p_greater'])}；同步句对置换P={p_text(local_summary['knn_item_permutation_p_greater'])}。",
        f"- 8x8固定100-ms时间矩阵中，患者exact max-stat FWER<0.05的cells={n_time}/64。",
        "",
        "## 解释边界",
        "",
        "1. active/passive与功能词、声学上下文、表面位置和句式共同变化，不能识别抽象语态轴。",
        "2. Agent/Patient与生命性和词汇类别混杂；训练折role centering防止两类均值直接驱动PLS，但不能解除设计混杂。",
        "3. 两视图PLS比刚性旋转更灵活，因此只有held-pair和同步刺激置换通过时才允许称为共享潜在几何。",
        "4. 若unit-sampling stability低，PLS/RDM阴性优先解释为测量上限不足，而非已证明remapping。",
        "5. 本分析由参考论文启发且在查看既有结果后开展，属于post-result exploratory analysis。",
        "",
        "## 主要输出",
        "",
        "- `unit_tuning_correlations.csv`, `unit_tuning_run_summary.csv`",
        "- `pls_cv_folds.csv`, `pls_run_summary.csv`, `pls_patient_summary.csv`",
        "- `pls_leave_one_patient_out.csv`",
        "- `local_geometry_run_summary.csv`, `unit_split_reliability.csv`",
        "- `readout_axis_run_summary.csv`, `dynamic_time_cell_stats.csv`",
        "- `group_inference.csv`, `summary.json`, `analysis_contract.json`",
        "- `figures/Fig_1_shared_latent_evidence.png/.pdf`",
        "- `figures/Fig_2_local_dynamic_geometry.png/.pdf`",
        "",
    ]
    (report_dir / "README_zh.md").write_text("\n".join(lines), encoding="utf-8")


def refresh_manifest(report_dir: Path, data_dir: Path) -> None:
    manifest_rows = []
    for path in sorted(list(report_dir.rglob("*")) + list(data_dir.glob("*"))):
        if path.is_file() and path.name != "delivery_manifest.csv":
            manifest_rows.append({"path": str(path), "size_bytes": path.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(report_dir / "delivery_manifest.csv", index=False)


def build_specification_comparison(
    specifications: list[tuple[str, Path, Path]],
) -> pd.DataFrame:
    rows = []
    for label, report_dir, data_dir in specifications:
        group_path = data_dir / "group_inference.csv"
        summary_path = data_dir / "summary.json"
        loo_path = data_dir / "pls_leave_one_patient_out.csv"
        if not (group_path.exists() and summary_path.exists() and loo_path.exists()):
            continue
        group = pd.read_csv(group_path).set_index("metric")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        loo = pd.read_csv(loo_path)
        gz002 = loo[(loo["rank"] == PRIMARY_RANK) & (loo["excluded_patient"] == "gz002")].iloc[0]

        def metric(name: str, column: str) -> float:
            return float(group.loc[name, column])

        rows.append(
            {
                "specification": label,
                "unit_set": summary["analysis_contract"]["unit_set"],
                "n_runs": int(sum(row["included"] for row in summary["run_inclusion"])),
                "n_patients": int(group.loc["pls_rank2_retrieval_margin", "n_patients"]),
                "unit_split_reliability_rho": math.tanh(
                    metric("unit_split_reliability_fisher_z", "patient_mean")
                ),
                "unit_split_reliability_positive_patients": int(
                    metric("unit_split_reliability_fisher_z", "positive_patients")
                ),
                "unit_split_reliability_patient_p_greater": metric(
                    "unit_split_reliability_fisher_z", "exact_signflip_p_greater"
                ),
                "unit_tuning_median_rho": math.tanh(
                    metric("unit_tuning_median_fisher_z", "patient_mean")
                ),
                "unit_tuning_global_fdr_count": int(summary["unit_tuning_global_fdr_count"]),
                "unit_tuning_within_run_fdr_count": int(
                    summary["unit_tuning_within_run_fdr_count"]
                ),
                "pls_rank2_margin": metric("pls_rank2_retrieval_margin", "patient_mean"),
                "pls_rank2_margin_positive_patients": int(
                    metric("pls_rank2_retrieval_margin", "positive_patients")
                ),
                "pls_rank2_margin_patient_p_greater": metric(
                    "pls_rank2_retrieval_margin", "exact_signflip_p_greater"
                ),
                "pls_rank2_margin_item_p_greater": metric(
                    "pls_rank2_retrieval_margin", "item_permutation_p_greater"
                ),
                "pls_rank2_accuracy": metric("pls_rank2_accuracy", "patient_mean"),
                "pls_rank2_accuracy_patient_p_greater": metric(
                    "pls_rank2_accuracy", "exact_signflip_p_greater"
                ),
                "pls_rank2_accuracy_item_p_greater": metric(
                    "pls_rank2_accuracy", "item_permutation_p_greater"
                ),
                "pls_rank4_accuracy": metric("pls_rank4_accuracy", "patient_mean"),
                "pls_rank4_accuracy_patient_p_greater": metric(
                    "pls_rank4_accuracy", "exact_signflip_p_greater"
                ),
                "loo_gz002_rank2_margin": float(gz002["loo_mean_retrieval_margin"]),
                "loo_gz002_rank2_accuracy": float(gz002["loo_mean_accuracy"]),
                "triplet_consistency": metric("triplet_consistency", "patient_mean"),
                "triplet_patient_p_greater": metric(
                    "triplet_consistency", "exact_signflip_p_greater"
                ),
                "triplet_item_p_greater": metric(
                    "triplet_consistency", "item_permutation_p_greater"
                ),
                "axis_alignment_null_z": metric("axis_alignment_null_z", "patient_mean"),
                "axis_alignment_positive_patients": int(
                    metric("axis_alignment_null_z", "positive_patients")
                ),
                "axis_participation_rho": metric("axis_participation_rho", "patient_mean"),
                "axis_participation_positive_patients": int(
                    metric("axis_participation_rho", "positive_patients")
                ),
                "axis_participation_patient_p_greater": metric(
                    "axis_participation_rho", "exact_signflip_p_greater"
                ),
                "axis_participation_patient_p_two_sided": metric(
                    "axis_participation_rho", "exact_signflip_p_two_sided"
                ),
                "dynamic_maxstat_significant_cells": int(
                    summary["dynamic_maxstat_significant_cells"]
                ),
            }
        )
    return pd.DataFrame(rows)


def write_combined_comparison(main_report_dir: Path, main_data_dir: Path) -> None:
    specifications = [
        ("主规格", main_report_dir, main_data_dir),
        (
            "宽松unit敏感性",
            main_report_dir / "sensitivity_usable_lenient",
            main_data_dir / "sensitivity_usable_lenient",
        ),
    ]
    comparison = build_specification_comparison(specifications)
    if len(comparison) != 2:
        return
    comparison.to_csv(main_report_dir / "specification_comparison.csv", index=False)
    main = comparison.set_index("specification").loc["主规格"]
    sensitivity = comparison.set_index("specification").loc["宽松unit敏感性"]

    def cell(row: pd.Series, value: str, patient_p: str | None = None, item_p: str | None = None) -> str:
        text = f"{row[value]:.3f}"
        if patient_p is not None:
            text += f"；患者P={row[patient_p]:.4f}"
        if item_p is not None:
            text += f"；刺激P={row[item_p]:.4f}"
        return text

    lines = [
        "# 主动/被动共享潜在几何：主规格与敏感性综合结论",
        "",
        "## 结论",
        "",
        "两套unit规格都出现Rank-2 held-pair margin的刺激层置换信号，但患者层检验均未通过；排除gz002后，两套规格的accuracy都降至接近机会水平。因此，当前结果支持‘存在值得复核的分布式低维关系信号’，但不支持‘五位患者中可推广的主动/被动通用神经几何’。",
        "",
        "## 跨规格结果",
        "",
        "| 指标 | 主规格 good_only | 宽松规格 usable_lenient |",
        "|---|---:|---:|",
        f"| Unit-split reliability rho | {cell(main, 'unit_split_reliability_rho', 'unit_split_reliability_patient_p_greater')} | {cell(sensitivity, 'unit_split_reliability_rho', 'unit_split_reliability_patient_p_greater')} |",
        f"| Rank-2 held-pair margin | {cell(main, 'pls_rank2_margin', 'pls_rank2_margin_patient_p_greater', 'pls_rank2_margin_item_p_greater')} | {cell(sensitivity, 'pls_rank2_margin', 'pls_rank2_margin_patient_p_greater', 'pls_rank2_margin_item_p_greater')} |",
        f"| Rank-2 accuracy | {cell(main, 'pls_rank2_accuracy', 'pls_rank2_accuracy_patient_p_greater', 'pls_rank2_accuracy_item_p_greater')} | {cell(sensitivity, 'pls_rank2_accuracy', 'pls_rank2_accuracy_patient_p_greater', 'pls_rank2_accuracy_item_p_greater')} |",
        f"| 排除gz002后的Rank-2 margin / accuracy | {main['loo_gz002_rank2_margin']:.3f} / {main['loo_gz002_rank2_accuracy']:.3f} | {sensitivity['loo_gz002_rank2_margin']:.3f} / {sensitivity['loo_gz002_rank2_accuracy']:.3f} |",
        f"| Same-unit participation rho | {main['axis_participation_rho']:.3f}；{int(main['axis_participation_positive_patients'])}/5正向；双侧P={main['axis_participation_patient_p_two_sided']:.4f} | {sensitivity['axis_participation_rho']:.3f}；{int(sensitivity['axis_participation_positive_patients'])}/5正向；双侧P={sensitivity['axis_participation_patient_p_two_sided']:.4f} |",
        f"| Triplet consistency | {cell(main, 'triplet_consistency', 'triplet_patient_p_greater', 'triplet_item_p_greater')} | {cell(sensitivity, 'triplet_consistency', 'triplet_patient_p_greater', 'triplet_item_p_greater')} |",
        f"| 单神经元全局FDR候选 | {int(main['unit_tuning_global_fdr_count'])} | {int(sensitivity['unit_tuning_global_fdr_count'])} |",
        f"| 动态RDM max-stat FWER cells | {int(main['dynamic_maxstat_significant_cells'])}/64 | {int(sensitivity['dynamic_maxstat_significant_cells'])}/64 |",
        "",
        "## 证据链怎么读",
        "",
        "1. **刺激层**：Rank-2 margin在两套规格的同步句对置换中均达到约P=0.048；accuracy只在宽松规格达到P<0.05，主规格为P=0.0579。这个层面说明项目配对关系不是完全随机，但不能建立患者推广性。",
        "2. **患者层**：Rank-2 margin与accuracy均未通过五患者exact sign-flip。主规格4/5患者margin为正，宽松规格也是4/5，但效应大小不均衡。",
        "3. **影响诊断**：gz002对PLS效应贡献最大。排除后主规格margin/accuracy为0.027/0.511，宽松规格为0.031/0.510，提示刺激层阳性主要由单一患者推动。",
        "4. **共同单元、不同轴**：same-unit participation在两套规格均为5/5正向，单侧P=0.03125；但在n=5下双侧exact P的最小非零值为0.0625，因此应写成一致方向性证据，而不是已确认效应。",
        "5. **分布层级**：单神经元全局与run内FDR均无候选，动态RDM无FWER显著cell。局部triplet只在主规格的刺激层阳性，未在宽松unit规格复现。",
        "",
        "## 可以写进论文的最强表述",
        "",
        "在严格的句对留出与训练折内降维条件下，主动和被动句中的同词同角色表征显示出探索性的低维跨句式对应关系。该对应在同步刺激置换中可检测，并伴随两种句式读出轴对相同神经元参与度的一致招募；然而，患者水平检验未达阈值，且效应明显受gz002驱动。因此，这些结果提示分布式群体关系结构的可能性，但尚不足以证明跨患者可推广的通用神经几何。",
        "",
        "## 不可越过的解释边界",
        "",
        "- active/passive与功能词、声学上下文、表面位置和句式共线，不能将结果归因于抽象语态本身。",
        "- Agent/Patient与生命性和词汇类别混杂，role centering不能解除实验设计层面的混杂。",
        "- 本分析受参考论文启发并在查看既有数据后开展，应明确标为post-result exploratory。",
        "",
        "完整数值见 `specification_comparison.csv`；各规格的患者留一结果见对应数据目录中的 `pls_leave_one_patient_out.csv`。",
        "",
    ]
    (main_report_dir / "COMBINED_REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


def postprocess_existing(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    data_dir = Path(args.data_dir)
    required = [
        report_dir / "analysis_contract.json",
        data_dir / "summary.json",
        data_dir / "pls_patient_summary.csv",
        data_dir / "group_inference.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing existing outputs: {missing}")

    pls_patient = pd.read_csv(data_dir / "pls_patient_summary.csv")
    pls_loo = compute_pls_leave_one_patient_out(pls_patient)
    pls_loo.to_csv(data_dir / "pls_leave_one_patient_out.csv", index=False)
    summary_payload = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    summary_payload["pls_leave_one_patient_out"] = pls_loo.to_dict(orient="records")
    for path in (data_dir / "summary.json", report_dir / "summary.json"):
        path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    contract = json.loads((report_dir / "analysis_contract.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(data_dir / "run_inclusion_audit.csv")
    group_inference = pd.read_csv(data_dir / "group_inference.csv")
    unit_tuning = pd.read_csv(data_dir / "unit_tuning_correlations.csv")
    cross_run = pd.read_csv(data_dir / "cross_run_same_condition_reliability.csv")
    dynamic_fwer = pd.read_csv(data_dir / "dynamic_time_cell_stats.csv")[
        "exact_maxstat_p_fwer_greater"
    ].to_numpy()
    write_report(
        report_dir,
        contract,
        audit,
        group_inference,
        summary_payload["pls_primary"],
        summary_payload["local_geometry"],
        unit_tuning,
        cross_run,
        dynamic_fwer,
        summary_payload["selected_descriptive_run"],
        pls_loo,
    )
    refresh_manifest(report_dir, data_dir)

    if args.unit_set == "good_only":
        main_report_dir, main_data_dir = report_dir, data_dir
    else:
        main_report_dir, main_data_dir = report_dir.parent, data_dir.parent
    write_combined_comparison(main_report_dir, main_data_dir)
    if main_report_dir.exists() and main_data_dir.exists():
        refresh_manifest(main_report_dir, main_data_dir)


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
        "estimand": "same-noun same-role active/passive shared latent and local geometry",
        "primary_latent_rank": PRIMARY_RANK,
        "sensitivity_latent_rank": SENSITIVITY_RANK,
        "pre_pca_rank": PRE_PCA_RANK,
        "outer_cv": "leave two complete sentence pairs out",
        "primary_pls_endpoint": "within-role held-pair retrieval margin",
        "local_endpoints": ["triplet_consistency", f"{LOCAL_K}-nearest-neighbor overlap"],
        "dynamic_windows_ms": [list(window) for window in DYNAMIC_WINDOWS_MS],
        "n_pls_permutations": args.pls_permutations,
        "n_local_permutations": args.local_permutations,
        "n_unit_tuning_permutations": args.unit_tuning_permutations,
        "n_axis_permutations": args.axis_permutations,
        "n_unit_splits": args.unit_splits,
        "n_bootstrap": args.n_bootstrap,
        "patient_inference": "patient-equal exhaustive sign flip",
        "item_inference": "synchronized 10-pair permutation across all runs/patients/roles",
        "interpretation_limit": "condition is confounded with marker/acoustics/position/form; role with animacy/lexical class",
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

    reliability_splits = pd.concat(
        [unit_split_reliability(run, args.unit_splits) for run in runs], ignore_index=True
    )
    reliability_splits.to_csv(data_dir / "unit_split_reliability.csv", index=False)
    reliability_run = reliability_splits.groupby(["patient", "run"], as_index=False).agg(
        active_fisher_z=("active_fisher_z", "median"),
        passive_fisher_z=("passive_fisher_z", "median"),
        role_condition_equal_fisher_z=("role_condition_equal_fisher_z", "median"),
    )
    reliability_run.to_csv(data_dir / "unit_split_reliability_run_summary.csv", index=False)
    cross_run = cross_run_reliability(runs)
    cross_run.to_csv(data_dir / "cross_run_same_condition_reliability.csv", index=False)

    unit_tuning = pd.concat(
        [tuning_correlations(run, args.unit_tuning_permutations) for run in runs],
        ignore_index=True,
    )
    unit_tuning = fdr_unit_table(unit_tuning)
    unit_tuning.to_csv(data_dir / "unit_tuning_correlations.csv", index=False)
    tuning_run = unit_tuning.groupby(["patient", "run"], as_index=False).agg(
        median_role_equal_z=("role_equal_fisher_z", "median"),
        mean_role_equal_z=("role_equal_fisher_z", "mean"),
        fraction_positive_both=("positive_both_roles", "mean"),
        n_units=("unit_id", "size"),
        n_q_global=("q_bh_global", lambda x: int((x < 0.05).sum())),
        n_q_within_run=("q_bh_within_run", lambda x: int((x < 0.05).sum())),
    )
    tuning_run.to_csv(data_dir / "unit_tuning_run_summary.csv", index=False)

    pls_folds = pd.concat([pls_cv(run) for run in runs], ignore_index=True)
    pls_folds.to_csv(data_dir / "pls_cv_folds.csv", index=False)
    pls_run = pls_folds.groupby(["patient", "run", "rank"], as_index=False).agg(
        retrieval_margin=("retrieval_margin", "mean"),
        two_by_two_accuracy=("two_by_two_accuracy", "mean"),
        matched_cosine=("matched_cosine", "mean"),
        matched_angle_deg=("matched_angle_deg", "mean"),
        mean_principal_angle_deg=("mean_principal_angle_deg", "mean"),
        axis_alignment=("axis_alignment", "mean"),
    )
    pls_run.to_csv(data_dir / "pls_run_summary.csv", index=False)
    pls_patient = pls_run.groupby(["patient", "rank"], as_index=False).mean(numeric_only=True)
    pls_patient.to_csv(data_dir / "pls_patient_summary.csv", index=False)
    pls_loo = compute_pls_leave_one_patient_out(pls_patient)
    pls_loo.to_csv(data_dir / "pls_leave_one_patient_out.csv", index=False)
    pls_null, pls_summary = synchronized_pls_permutation(
        runs,
        args.pls_permutations,
        PRIMARY_RANK,
        base.stable_seed("shared-latent-pls-synchronized"),
    )
    pls_null.to_csv(data_dir / "pls_synchronized_pair_permutation_null.csv", index=False)

    local_rows = []
    for run in runs:
        local_rows.append({"patient": run.patient, "run": run.run, **local_geometry_endpoint(run)})
    local_run = pd.DataFrame(local_rows)
    local_run.to_csv(data_dir / "local_geometry_run_summary.csv", index=False)
    local_null, local_summary = synchronized_local_permutation(
        runs,
        args.local_permutations,
        base.stable_seed("shared-latent-local-synchronized"),
    )
    local_null.to_csv(data_dir / "local_synchronized_pair_permutation_null.csv", index=False)

    axis_rows = []
    unit_axis_rows = []
    full_scores = {}
    for run in runs:
        summary, units, scores_active, scores_passive = full_readout_axes(
            run, PRIMARY_RANK, args.axis_permutations
        )
        axis_rows.append(summary)
        unit_axis_rows.append(units)
        full_scores[run.run] = (scores_active, scores_passive)
    axis_run = pd.DataFrame(axis_rows)
    axis_run.to_csv(data_dir / "readout_axis_run_summary.csv", index=False)
    pd.concat(unit_axis_rows, ignore_index=True).to_csv(
        data_dir / "readout_axis_unit_participation.csv", index=False
    )

    dynamic_rows = []
    dynamic_run_matrices = []
    for run in runs:
        frame = metrics[(metrics["patient"] == run.patient) & (metrics["run"] == run.run)]
        dynamic_active, dynamic_passive = extract_dynamic_tensor(run, frame, args.unit_set)
        matrix = dynamic_rdm_matrix(run, dynamic_active, dynamic_passive)
        dynamic_run_matrices.append(matrix)
        for idx_a, (start_a, stop_a) in enumerate(DYNAMIC_WINDOWS_MS):
            for idx_p, (start_p, stop_p) in enumerate(DYNAMIC_WINDOWS_MS):
                dynamic_rows.append(
                    {
                        "patient": run.patient,
                        "run": run.run,
                        "active_window_index": idx_a,
                        "passive_window_index": idx_p,
                        "active_window_ms": f"{start_a}-{stop_a}",
                        "passive_window_ms": f"{start_p}-{stop_p}",
                        "role_equal_fisher_z": matrix[idx_a, idx_p],
                        "role_equal_rho": base.inv_fisher(matrix[idx_a, idx_p]),
                    }
                )
    dynamic_run = pd.DataFrame(dynamic_rows)
    dynamic_run.to_csv(data_dir / "dynamic_time_run_cells.csv", index=False)
    patients = sorted(dynamic_run["patient"].unique())
    patient_matrices = []
    for patient in patients:
        frame = dynamic_run[dynamic_run["patient"] == patient]
        matrix = frame.groupby(["active_window_index", "passive_window_index"])[
            "role_equal_fisher_z"
        ].mean().unstack().to_numpy()
        patient_matrices.append(matrix)
    patient_matrices = np.asarray(patient_matrices)
    dynamic_observed, dynamic_null_max, dynamic_fwer = exact_patient_max_stat(patient_matrices)
    dynamic_stats = []
    for idx_a in range(len(DYNAMIC_WINDOWS_MS)):
        for idx_p in range(len(DYNAMIC_WINDOWS_MS)):
            values = patient_matrices[:, idx_a, idx_p]
            sign = base.exact_sign_flip(values, 0.0)
            dynamic_stats.append(
                {
                    "active_window_index": idx_a,
                    "passive_window_index": idx_p,
                    "patient_mean_fisher_z": dynamic_observed[idx_a, idx_p],
                    "patient_mean_rho": base.inv_fisher(dynamic_observed[idx_a, idx_p]),
                    "positive_patients": int((values > 0).sum()),
                    "exact_signflip_p_greater": sign["p_greater"],
                    "exact_maxstat_p_fwer_greater": dynamic_fwer[idx_a, idx_p],
                    "patient_values_json": json.dumps(dict(zip(patients, values)), ensure_ascii=False),
                }
            )
    dynamic_stats = pd.DataFrame(dynamic_stats)
    dynamic_stats["q_bh"] = base.bh_fdr(dynamic_stats["exact_signflip_p_greater"])
    dynamic_stats.to_csv(data_dir / "dynamic_time_cell_stats.csv", index=False)
    np.savez_compressed(
        data_dir / "dynamic_time_patient_matrices.npz",
        patient_matrices=patient_matrices,
        patients=np.asarray(patients, dtype=object),
        observed=dynamic_observed,
        null_max=dynamic_null_max,
        p_fwer=dynamic_fwer,
        windows_ms=np.asarray(DYNAMIC_WINDOWS_MS),
    )

    patient_tables = []
    group_rows = []
    patient, summary = summarize_metric(
        reliability_run,
        "role_condition_equal_fisher_z",
        "unit_split_reliability_fisher_z",
        0.0,
        args.n_bootstrap,
    )
    patient_tables.append(patient)
    group_rows.append(summary)
    patient, summary = summarize_metric(
        tuning_run,
        "median_role_equal_z",
        "unit_tuning_median_fisher_z",
        0.0,
        args.n_bootstrap,
    )
    patient_tables.append(patient)
    group_rows.append(summary)
    for rank in LATENT_RANKS:
        frame = pls_run[pls_run["rank"] == rank]
        for column, suffix, reference in (
            ("retrieval_margin", "retrieval_margin", 0.0),
            ("two_by_two_accuracy", "accuracy", 0.5),
        ):
            patient, summary = summarize_metric(
                frame,
                column,
                f"pls_rank{rank}_{suffix}",
                reference,
                args.n_bootstrap,
            )
            patient_tables.append(patient)
            group_rows.append(summary)
    for column, metric, reference in (
        ("triplet_consistency", "triplet_consistency", 0.5),
        ("knn_overlap", "knn_overlap", LOCAL_K / 9.0),
    ):
        patient, summary = summarize_metric(
            local_run, column, metric, reference, args.n_bootstrap
        )
        patient_tables.append(patient)
        group_rows.append(summary)
    for column, metric in (
        ("axis_alignment_null_z", "axis_alignment_null_z"),
        ("participation_rho", "axis_participation_rho"),
    ):
        patient, summary = summarize_metric(
            axis_run, column, metric, 0.0, args.n_bootstrap
        )
        patient_tables.append(patient)
        group_rows.append(summary)
    all_patient = pd.concat(patient_tables, ignore_index=True)
    all_patient.to_csv(data_dir / "all_patient_endpoints.csv", index=False)
    group_inference = pd.DataFrame(group_rows)
    group_inference.loc[
        group_inference["metric"] == "pls_rank2_retrieval_margin",
        "item_permutation_p_greater",
    ] = pls_summary["margin_item_permutation_p_greater"]
    group_inference.loc[
        group_inference["metric"] == "pls_rank2_accuracy",
        "item_permutation_p_greater",
    ] = pls_summary["accuracy_item_permutation_p_greater"]
    group_inference.loc[
        group_inference["metric"] == "triplet_consistency",
        "item_permutation_p_greater",
    ] = local_summary["triplet_item_permutation_p_greater"]
    group_inference.loc[
        group_inference["metric"] == "knn_overlap",
        "item_permutation_p_greater",
    ] = local_summary["knn_item_permutation_p_greater"]
    group_inference.to_csv(data_dir / "group_inference.csv", index=False)

    reliability_patient = all_patient[
        all_patient["metric"] == "unit_split_reliability_fisher_z"
    ][["patient", "value"]]
    tuning_patient = all_patient[
        all_patient["metric"] == "unit_tuning_median_fisher_z"
    ][["patient", "value"]]
    axes_patient_rows = []
    for metric in ("axis_alignment_null_z", "axis_participation_rho"):
        short = "participation_rho" if metric == "axis_participation_rho" else metric
        frame = all_patient[all_patient["metric"] == metric]
        for row in frame.itertuples(index=False):
            axes_patient_rows.append({"patient": row.patient, "metric": short, "value": row.value})
    axes_patient = pd.DataFrame(axes_patient_rows)
    local_patient_rows = []
    for metric in ("triplet_consistency", "knn_overlap"):
        frame = all_patient[all_patient["metric"] == metric]
        for row in frame.itertuples(index=False):
            local_patient_rows.append({"patient": row.patient, "metric": metric, "value": row.value})
    local_patient = pd.DataFrame(local_patient_rows)

    selected_run_name = (
        pls_run[pls_run["rank"] == PRIMARY_RANK]
        .sort_values("retrieval_margin", ascending=False)
        .iloc[0]["run"]
    )
    selected_run = next(run for run in runs if run.run == selected_run_name)
    selected_scores_active, selected_scores_passive = full_scores[selected_run_name]
    make_figures(
        report_dir,
        reliability_patient,
        unit_tuning,
        tuning_patient,
        pls_patient,
        pls_null,
        pls_summary,
        axes_patient,
        selected_run,
        selected_scores_active,
        selected_scores_passive,
        local_patient,
        local_null,
        local_summary,
        dynamic_observed,
        dynamic_fwer,
        patient_matrices,
    )

    summary_payload = {
        "analysis_contract": contract,
        "run_inclusion": audit.to_dict(orient="records"),
        "group_inference": group_inference.to_dict(orient="records"),
        "pls_primary": pls_summary,
        "local_geometry": local_summary,
        "unit_tuning_global_fdr_count": int((unit_tuning["q_bh_global"] < 0.05).sum()),
        "unit_tuning_within_run_fdr_count": int((unit_tuning["q_bh_within_run"] < 0.05).sum()),
        "dynamic_maxstat_significant_cells": int(np.sum(dynamic_fwer < 0.05)),
        "selected_descriptive_run": selected_run_name,
        "pls_leave_one_patient_out": pls_loo.to_dict(orient="records"),
    }
    for path in (data_dir / "summary.json", report_dir / "summary.json"):
        path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        report_dir,
        contract,
        audit,
        group_inference,
        pls_summary,
        local_summary,
        unit_tuning,
        cross_run,
        dynamic_fwer,
        selected_run_name,
        pls_loo,
    )
    refresh_manifest(report_dir, data_dir)
    if args.unit_set == "good_only":
        main_report_dir, main_data_dir = report_dir, data_dir
    else:
        main_report_dir, main_data_dir = report_dir.parent, data_dir.parent
    write_combined_comparison(main_report_dir, main_data_dir)
    if main_report_dir.exists() and main_data_dir.exists():
        refresh_manifest(main_report_dir, main_data_dir)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=("good_only", "usable_lenient"), default="good_only")
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--pls-permutations", type=int, default=500)
    parser.add_argument("--local-permutations", type=int, default=5000)
    parser.add_argument("--unit-tuning-permutations", type=int, default=5000)
    parser.add_argument("--axis-permutations", type=int, default=5000)
    parser.add_argument("--unit-splits", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="Regenerate LOO diagnostics, reports, and manifests from saved outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.postprocess_existing:
        postprocess_existing(parsed_args)
    else:
        run_analysis(parsed_args)
