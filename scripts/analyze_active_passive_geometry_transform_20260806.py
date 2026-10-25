#!/usr/bin/env python3
"""Active-to-passive noun geometry: translation, rotation, or remapping.

The independent inferential unit is the patient.  The analysis uses the same
20 nouns in active and passive sentences, with thematic role held fixed while
surface position, function word, acoustics, and sentence form change.

Primary tests
-------------
1. Role-stratified active/passive RDM preservation.
2. Strict leave-two-sentence-pairs-out comparison of identity, translation,
   orthogonal rotation, and ridge-affine maps.
3. Patient-level exact sign-flip inference plus synchronized sentence-pair
   label permutations for the primary RDM and rotation endpoints.

Secondary tests
---------------
4. A fixed 4 x 4 time-resolved geometry matrix (four 200-ms noun windows).
5. Fixed-rank unit-space subspace alignment and same-unit participation.
6. Training-fold condition-unit ablation versus matched random ablation.
7. Leave-one-patient-out consensus RDMs, which do not require shared units.

This is a post-result, literature-motivated analysis and must be reported as
exploratory.  It cannot identify abstract voice independently of marker and
acoustic identity.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025/data"
NOUN_DIR = DATA_ROOT / "combined_noun_role_structure_encoding"
METRICS_PATH = NOUN_DIR / "noun_role_unit_trial_response_metrics.csv"
REGISTRY_PATH = ROOT / "reports/noun_firstchar_llm_mapping_20260725/stimulus_registry.csv"
DEFAULT_REPORT_DIR = ROOT / "reports/active_passive_geometry_transform_20260806"
DEFAULT_DATA_DIR = DATA_ROOT / "combined_active_passive_geometry_transform_20260806"

METRIC = "response_baseline_corrected_hz"
ROLES = ("agent", "patient")
VOICES = ("active", "passive")
WINDOWS_MS = ((0, 200), (200, 400), (400, 600), (600, 800))
RANKS = (2, 4)
MODELS = ("identity", "translation", "rotation", "affine")
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#8E44AD",
    "sky": "#56B4E9",
    "black": "#111111",
    "gray": "#777777",
    "light_gray": "#D9D9D9",
}


@dataclass
class RunData:
    patient: str
    run: str
    unit_ids: np.ndarray
    item_meta: pd.DataFrame
    active: np.ndarray
    passive: np.ndarray
    time_active: np.ndarray
    time_passive: np.ndarray
    artifact_annotations_available: bool


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def inv_fisher(z: float) -> float:
    return float(np.tanh(z)) if np.isfinite(z) else np.nan


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return np.nan
    return float(stats.spearmanr(x[keep], y[keep]).correlation)


def cosine_rdm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms == 0)] = 1.0
    xn = x / norms
    return 1.0 - np.clip(xn @ xn.T, -1.0, 1.0)


def rdm_vector(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)[np.tril_indices(x.shape[0], k=-1)]


def standardize_joint(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joined = np.vstack([a, b]).astype(float)
    mean = joined.mean(axis=0, keepdims=True)
    sd = joined.std(axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (a - mean) / sd, (b - mean) / sd


def exact_sign_flip(values: Iterable[float], reference: float = 0.0) -> dict[str, float]:
    x = np.asarray(list(values), dtype=float) - reference
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "observed_mean": np.nan, "p_greater": np.nan, "p_two_sided": np.nan}
    observed = float(x.mean())
    null = np.asarray(
        [np.mean(x * np.asarray(signs, dtype=float)) for signs in itertools.product((-1.0, 1.0), repeat=x.size)]
    )
    tol = 1e-12
    return {
        "n": int(x.size),
        "observed_mean": observed,
        "p_greater": float(np.mean(null >= observed - tol)),
        "p_two_sided": float(np.mean(np.abs(null) >= abs(observed) - tol)),
    }


def bootstrap_patient_ci(values: Iterable[float], n_boot: int, seed: int) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        draws[idx] = rng.choice(x, size=x.size, replace=True).mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return q
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * valid.size / np.arange(1, valid.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def pair_id_from_sound_file(value: object) -> int:
    stem = Path(str(value)).stem
    digits = "".join(char for char in stem if char.isdigit())
    if not digits:
        raise ValueError(f"Cannot infer pair id from {value}")
    return int(digits)


def load_registry() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY_PATH)
    registry = registry[registry["condition"].isin(VOICES)].copy()
    registry["pair_id"] = registry["pair_id"].astype(int)
    registry["surface_position"] = registry["surface_position"].astype(str)
    keep = ["sound_file", "pair_id", "condition", "noun", "thematic_role", "surface_position"]
    return registry[keep].drop_duplicates(["sound_file", "noun"])


def load_artifacts() -> tuple[pd.DataFrame, set[str]]:
    frames: list[pd.DataFrame] = []
    available_runs: set[str] = set()
    pattern = "gz*/processed/ActPasSynSentence/event_related_active_passive/trial_artifact_overlap_minus1_to_plus2s.csv"
    for path in sorted(DATA_ROOT.glob(pattern)):
        frame = pd.read_csv(path)
        frame = frame[frame["condition"].astype(str).str.startswith(VOICES)].copy()
        if frame.empty:
            continue
        available_runs.update(frame["run"].astype(str).unique())
        frames.append(
            frame[["run", "trial_index", "artifact_fraction_minus1_to_plus2s"]].drop_duplicates(
                ["run", "trial_index"]
            )
        )
    if not frames:
        return pd.DataFrame(columns=["run", "trial_index", "artifact_fraction_minus1_to_plus2s"]), available_runs
    return pd.concat(frames, ignore_index=True), available_runs


def unit_archive_path(patient: str, run: str, unit_set: str) -> Path:
    names = {
        "good_only": "firing_rate_1000Hz_good_clusters.npz",
        "usable_lenient": "firing_rate_1000Hz_usable_lenient_clusters.npz",
    }
    return (
        DATA_ROOT
        / patient
        / "processed/ActPasSynSentence"
        / run
        / "firing_rate_1000Hz"
        / names[unit_set]
    )


def load_filtered_metrics(unit_set: str, artifact_threshold: float | None) -> tuple[pd.DataFrame, set[str]]:
    metrics = pd.read_csv(METRICS_PATH)
    metrics["pair_id"] = metrics["sound_file"].map(pair_id_from_sound_file).astype(int)
    metrics["voice"] = metrics["sentence_condition"].astype(str)
    if unit_set == "good_only":
        metrics = metrics[metrics["ks_label"].astype(str).eq("good")].copy()
    elif unit_set != "usable_lenient":
        raise ValueError(f"Unsupported unit set: {unit_set}")
    registry = load_registry()
    metrics = metrics.merge(
        registry,
        left_on=["sound_file", "pair_id", "voice", "noun"],
        right_on=["sound_file", "pair_id", "condition", "noun"],
        how="left",
        validate="many_to_one",
    )
    if metrics["thematic_role"].isna().any():
        missing = metrics.loc[metrics["thematic_role"].isna(), ["sound_file", "noun"]].drop_duplicates()
        raise RuntimeError(f"Registry join failed:\n{missing.to_string(index=False)}")
    artifacts, available_runs = load_artifacts()
    metrics = metrics.merge(artifacts, on=["run", "trial_index"], how="left")
    if artifact_threshold is not None:
        has_table = metrics["run"].isin(available_runs)
        keep = (~has_table) | (
            metrics["artifact_fraction_minus1_to_plus2s"].notna()
            & (metrics["artifact_fraction_minus1_to_plus2s"] <= artifact_threshold)
        )
        metrics = metrics.loc[keep].copy()
    return metrics, available_runs


def sorted_item_meta(frame: pd.DataFrame) -> pd.DataFrame:
    meta = frame[
        ["pair_id", "noun", "thematic_role"]
    ].drop_duplicates().sort_values(["thematic_role", "pair_id"]).reset_index(drop=True)
    meta["item_index"] = np.arange(len(meta), dtype=int)
    return meta


def raw_time_tensor(
    patient: str,
    run: str,
    unit_set: str,
    unit_ids: np.ndarray,
    event_rows: pd.DataFrame,
    item_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    archive_path = unit_archive_path(patient, run, unit_set)
    with np.load(archive_path, allow_pickle=True) as archive:
        firing_rate = np.asarray(archive["firing_rate_hz"], dtype=float)
        archive_unit_ids = np.asarray(archive["unit_ids"], dtype=int)
    lookup = {int(unit_id): idx for idx, unit_id in enumerate(archive_unit_ids)}
    indices = [lookup[int(unit_id)] for unit_id in unit_ids]
    firing_rate = firing_rate[indices]
    out: dict[str, np.ndarray] = {
        voice: np.full((len(item_meta), len(WINDOWS_MS), len(unit_ids)), np.nan, dtype=float)
        for voice in VOICES
    }
    for voice in VOICES:
        for _, item in item_meta.iterrows():
            row = event_rows[
                (event_rows["voice"] == voice)
                & (event_rows["pair_id"] == int(item["pair_id"]))
                & (event_rows["thematic_role"] == item["thematic_role"])
            ]
            if len(row) != 1:
                continue
            event_bin = int(row.iloc[0]["event_bin_1000Hz"])
            if event_bin - 500 < 0 or event_bin + 800 > firing_rate.shape[1]:
                continue
            baseline = firing_rate[:, event_bin - 500 : event_bin].mean(axis=1)
            for win_idx, (start, stop) in enumerate(WINDOWS_MS):
                response = firing_rate[:, event_bin + start : event_bin + stop].mean(axis=1)
                out[voice][int(item["item_index"]), win_idx] = response - baseline
    return out["active"], out["passive"]


def build_run_data(
    metrics: pd.DataFrame,
    available_artifact_runs: set[str],
    unit_set: str,
) -> tuple[list[RunData], pd.DataFrame]:
    runs: list[RunData] = []
    audit_rows: list[dict[str, object]] = []
    for (patient, run), frame in metrics.groupby(["patient", "run"], sort=True):
        item_meta = sorted_item_meta(frame)
        complete_items = len(item_meta) == 20 and all(
            len(frame[(frame["voice"] == voice) & (frame["thematic_role"] == role)][["pair_id", "noun"]].drop_duplicates()) == 10
            for voice in VOICES
            for role in ROLES
        )
        if not complete_items:
            audit_rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "included": False,
                    "reason": "incomplete matched 20-noun active/passive set after filtering",
                    "n_items": len(item_meta),
                    "n_units": frame["unit_id"].nunique(),
                    "artifact_annotations_available": run in available_artifact_runs,
                }
            )
            continue
        unit_ids = np.asarray(sorted(frame["unit_id"].astype(int).unique()), dtype=int)
        matrices: dict[str, np.ndarray] = {}
        valid_units = np.ones(len(unit_ids), dtype=bool)
        for voice in VOICES:
            pivot = frame[frame["voice"] == voice].pivot_table(
                index=["thematic_role", "pair_id", "noun"], columns="unit_id", values=METRIC, aggfunc="mean"
            )
            order = pd.MultiIndex.from_frame(item_meta[["thematic_role", "pair_id", "noun"]])
            pivot = pivot.reindex(index=order, columns=unit_ids)
            arr = pivot.to_numpy(dtype=float)
            matrices[voice] = arr
            valid_units &= np.isfinite(arr).all(axis=0) & (np.std(arr, axis=0) > 0)
        unit_ids = unit_ids[valid_units]
        active = matrices["active"][:, valid_units]
        passive = matrices["passive"][:, valid_units]
        if len(unit_ids) < 4:
            audit_rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "included": False,
                    "reason": "fewer than four finite variable units",
                    "n_items": 20,
                    "n_units": len(unit_ids),
                    "artifact_annotations_available": run in available_artifact_runs,
                }
            )
            continue
        event_rows = frame.drop_duplicates(
            ["voice", "pair_id", "thematic_role", "noun", "trial_index"]
        )[
            ["voice", "pair_id", "thematic_role", "noun", "trial_index", "event_bin_1000Hz"]
        ]
        time_active, time_passive = raw_time_tensor(
            patient, run, unit_set, unit_ids, event_rows, item_meta
        )
        if not np.isfinite(time_active).all() or not np.isfinite(time_passive).all():
            audit_rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "included": False,
                    "reason": "time-window extraction produced missing values",
                    "n_items": 20,
                    "n_units": len(unit_ids),
                    "artifact_annotations_available": run in available_artifact_runs,
                }
            )
            continue
        runs.append(
            RunData(
                patient=str(patient),
                run=str(run),
                unit_ids=unit_ids,
                item_meta=item_meta,
                active=active,
                passive=passive,
                time_active=time_active,
                time_passive=time_passive,
                artifact_annotations_available=run in available_artifact_runs,
            )
        )
        audit_rows.append(
            {
                "patient": patient,
                "run": run,
                "included": True,
                "reason": "included",
                "n_items": 20,
                "n_units": len(unit_ids),
                "artifact_annotations_available": run in available_artifact_runs,
            }
        )
    return runs, pd.DataFrame(audit_rows)


def role_indices(meta: pd.DataFrame, role: str) -> np.ndarray:
    return np.flatnonzero(meta["thematic_role"].astype(str).to_numpy() == role)


def reorder_passive_by_pair(passive: np.ndarray, meta: pd.DataFrame, permutation: np.ndarray) -> np.ndarray:
    out = passive.copy()
    for role in ROLES:
        idx = role_indices(meta, role)
        pair_order = meta.iloc[idx]["pair_id"].to_numpy(dtype=int)
        if not np.array_equal(pair_order, np.arange(1, 11)):
            raise RuntimeError(f"Unexpected pair order for {role}: {pair_order}")
        out[idx] = passive[idx][permutation]
    return out


def geometry_endpoint(
    active: np.ndarray,
    passive: np.ndarray,
    meta: pd.DataFrame,
    permutation: np.ndarray | None = None,
) -> dict[str, float]:
    active_z, passive_z = standardize_joint(active, passive)
    if permutation is not None:
        passive_z = reorder_passive_by_pair(passive_z, meta, permutation)
    role_rhos: dict[str, float] = {}
    for role in ROLES:
        idx = role_indices(meta, role)
        role_rhos[role] = spearman_r(
            rdm_vector(cosine_rdm(active_z[idx])),
            rdm_vector(cosine_rdm(passive_z[idx])),
        )
    z_mean = float(np.nanmean([fisher_z(role_rhos[role]) for role in ROLES]))
    return {
        "agent_rho": role_rhos["agent"],
        "patient_rho": role_rhos["patient"],
        "role_equal_fisher_z": z_mean,
        "role_equal_rho": inv_fisher(z_mean),
    }


def patient_average(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return frame.groupby("patient", as_index=False)[value_col].mean().rename(columns={value_col: "value"})


def synchronized_rdm_permutation(
    runs: list[RunData],
    n_perm: int,
    seed: int,
) -> tuple[np.ndarray, float, dict[str, float]]:
    real_rows = []
    for run in runs:
        endpoint = geometry_endpoint(run.active, run.passive, run.item_meta)
        real_rows.append({"patient": run.patient, "run": run.run, "z": endpoint["role_equal_fisher_z"]})
    real = pd.DataFrame(real_rows)
    observed = float(real.groupby("patient")["z"].mean().mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for perm_idx in range(n_perm):
        permutation = rng.permutation(10)
        rows = []
        for run in runs:
            endpoint = geometry_endpoint(run.active, run.passive, run.item_meta, permutation)
            rows.append((run.patient, endpoint["role_equal_fisher_z"]))
        perm_frame = pd.DataFrame(rows, columns=["patient", "z"])
        null[perm_idx] = perm_frame.groupby("patient")["z"].mean().mean()
    p_greater = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    p_two = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return null, observed, {"p_greater": p_greater, "p_two_sided": p_two}


def fit_outer_latent(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_joint = np.vstack([x_train, y_train]).astype(float)
    unit_mean = train_joint.mean(axis=0, keepdims=True)
    unit_sd = train_joint.std(axis=0, ddof=1, keepdims=True)
    unit_sd[~np.isfinite(unit_sd) | (unit_sd == 0)] = 1.0

    def scale(x: np.ndarray) -> np.ndarray:
        return (x - unit_mean) / unit_sd

    xtr = scale(x_train)
    ytr = scale(y_train)
    xte = scale(x_test)
    yte = scale(y_test)
    pca_train = np.vstack([xtr, ytr])
    pca_mean = pca_train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pca_train - pca_mean, full_matrices=False)
    actual_rank = min(rank, vt.shape[0], vt.shape[1])
    components = vt[:actual_rank].T

    def project(x: np.ndarray) -> np.ndarray:
        return (x - pca_mean) @ components

    return project(xtr), project(ytr), project(xte), project(yte)


def fit_rotation(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_x = x.mean(axis=0, keepdims=True)
    mean_y = y.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd((x - mean_x).T @ (y - mean_y), full_matrices=False)
    return u @ vt, mean_x, mean_y


def fit_affine(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_x = x.mean(axis=0, keepdims=True)
    mean_y = y.mean(axis=0, keepdims=True)
    xc = x - mean_x
    yc = y - mean_y
    ridge = xc.T @ xc + alpha * np.eye(x.shape[1])
    weights = np.linalg.solve(ridge, xc.T @ yc)
    return weights, mean_x, mean_y


def row_angles(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_norm = np.linalg.norm(pred, axis=1)
    truth_norm = np.linalg.norm(truth, axis=1)
    denom = pred_norm * truth_norm
    denom[denom == 0] = 1.0
    cosine = np.sum(pred * truth, axis=1) / denom
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.where(np.linalg.norm(a, axis=1, keepdims=True) == 0, 1.0, np.linalg.norm(a, axis=1, keepdims=True))
    bn = b / np.where(np.linalg.norm(b, axis=1, keepdims=True) == 0, 1.0, np.linalg.norm(b, axis=1, keepdims=True))
    return 1.0 - np.clip(an @ bn.T, -1.0, 1.0)


def within_role_two_by_two(pred: np.ndarray, truth: np.ndarray, test_meta: pd.DataFrame) -> float:
    scores: list[float] = []
    for role in ROLES:
        idx = np.flatnonzero(test_meta["thematic_role"].astype(str).to_numpy() == role)
        if len(idx) != 2:
            continue
        distance = cosine_distance_matrix(pred[idx], truth[idx])
        correct = distance[0, 0] + distance[1, 1]
        swapped = distance[0, 1] + distance[1, 0]
        scores.append(1.0 if correct < swapped else (0.5 if np.isclose(correct, swapped) else 0.0))
    return float(np.mean(scores)) if scores else np.nan


def select_ridge_alpha(x: np.ndarray, y: np.ndarray, train_meta: pd.DataFrame) -> float:
    losses = {alpha: [] for alpha in RIDGE_ALPHAS}
    for pair_id in sorted(train_meta["pair_id"].unique()):
        valid = train_meta["pair_id"].to_numpy() == pair_id
        if valid.sum() != 2 or (~valid).sum() < 6:
            continue
        for alpha in RIDGE_ALPHAS:
            weights, mean_x, mean_y = fit_affine(x[~valid], y[~valid], alpha)
            pred = (x[valid] - mean_x) @ weights + mean_y
            losses[alpha].append(float(np.mean(row_angles(pred, y[valid]))))
    return min(RIDGE_ALPHAS, key=lambda alpha: (np.mean(losses[alpha]) if losses[alpha] else np.inf, alpha))


def predict_model(
    model: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    train_meta: pd.DataFrame,
) -> tuple[np.ndarray, float | None]:
    if model == "identity":
        return x_test.copy(), None
    if model == "translation":
        return x_test + np.mean(y_train - x_train, axis=0, keepdims=True), None
    if model == "rotation":
        rotation, mean_x, mean_y = fit_rotation(x_train, y_train)
        return (x_test - mean_x) @ rotation + mean_y, None
    if model == "affine":
        alpha = select_ridge_alpha(x_train, y_train, train_meta)
        weights, mean_x, mean_y = fit_affine(x_train, y_train, alpha)
        return (x_test - mean_x) @ weights + mean_y, alpha
    raise ValueError(model)


def outer_folds(meta: pd.DataFrame) -> Iterable[tuple[tuple[int, int], np.ndarray, np.ndarray]]:
    pair_values = sorted(meta["pair_id"].astype(int).unique())
    pair_array = meta["pair_id"].astype(int).to_numpy()
    for held_pairs in itertools.combinations(pair_values, 2):
        test = np.isin(pair_array, held_pairs)
        yield held_pairs, ~test, test


def transform_cv(
    run: RunData,
    ranks: Iterable[int] = RANKS,
    models: Iterable[str] = MODELS,
    passive_pair_permutation: np.ndarray | None = None,
) -> pd.DataFrame:
    y = run.passive if passive_pair_permutation is None else reorder_passive_by_pair(
        run.passive, run.item_meta, passive_pair_permutation
    )
    rows: list[dict[str, object]] = []
    for held_pairs, train, test in outer_folds(run.item_meta):
        train_meta = run.item_meta.loc[train].reset_index(drop=True)
        test_meta = run.item_meta.loc[test].reset_index(drop=True)
        for rank in ranks:
            xtr, ytr, xte, yte = fit_outer_latent(
                run.active[train], y[train], run.active[test], y[test], rank
            )
            for model in models:
                pred, alpha = predict_model(model, xtr, ytr, xte, train_meta)
                rows.append(
                    {
                        "patient": run.patient,
                        "run": run.run,
                        "held_pair_1": held_pairs[0],
                        "held_pair_2": held_pairs[1],
                        "rank": rank,
                        "model": model,
                        "mean_angle_deg": float(np.mean(row_angles(pred, yte))),
                        "two_by_two_accuracy": within_role_two_by_two(pred, yte, test_meta),
                        "ridge_alpha": alpha,
                    }
                )
    return pd.DataFrame(rows)


def condition_unit_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    delta = y - x
    sd = delta.std(axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.abs(delta.mean(axis=0) / sd)


def evaluate_rotation_subset(
    run: RunData,
    train: np.ndarray,
    test: np.ndarray,
    keep_units: np.ndarray,
    rank: int,
) -> tuple[float, float]:
    train_meta = run.item_meta.loc[train].reset_index(drop=True)
    test_meta = run.item_meta.loc[test].reset_index(drop=True)
    xtr, ytr, xte, yte = fit_outer_latent(
        run.active[train][:, keep_units],
        run.passive[train][:, keep_units],
        run.active[test][:, keep_units],
        run.passive[test][:, keep_units],
        rank,
    )
    pred, _ = predict_model("rotation", xtr, ytr, xte, train_meta)
    return float(np.mean(row_angles(pred, yte))), within_role_two_by_two(pred, yte, test_meta)


def ablation_cv(run: RunData, rank: int, random_repeats: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for held_pairs, train, test in outer_folds(run.item_meta):
        n_units = run.active.shape[1]
        n_remove = min(max(1, int(round(0.20 * n_units))), n_units - max(4, rank))
        scores = condition_unit_scores(run.active[train], run.passive[train])
        remove_top = np.argsort(scores)[-n_remove:]
        keep_top = np.ones(n_units, dtype=bool)
        keep_top[remove_top] = False
        top_angle, top_acc = evaluate_rotation_subset(run, train, test, keep_top, rank)
        rng = np.random.default_rng(stable_seed("ablation", run.run, held_pairs))
        random_angles: list[float] = []
        random_accs: list[float] = []
        for _ in range(random_repeats):
            remove_random = rng.choice(n_units, size=n_remove, replace=False)
            keep_random = np.ones(n_units, dtype=bool)
            keep_random[remove_random] = False
            angle, acc = evaluate_rotation_subset(run, train, test, keep_random, rank)
            random_angles.append(angle)
            random_accs.append(acc)
        rows.append(
            {
                "patient": run.patient,
                "run": run.run,
                "held_pair_1": held_pairs[0],
                "held_pair_2": held_pairs[1],
                "rank": rank,
                "n_units": n_units,
                "n_removed": n_remove,
                "top_condition_removed_angle_deg": top_angle,
                "top_condition_removed_accuracy": top_acc,
                "random_removed_mean_angle_deg": float(np.mean(random_angles)),
                "random_removed_mean_accuracy": float(np.mean(random_accs)),
                "angle_delta_top_minus_random": top_angle - float(np.mean(random_angles)),
                "accuracy_delta_top_minus_random": top_acc - float(np.mean(random_accs)),
            }
        )
    return pd.DataFrame(rows)


def synchronized_transform_permutation(
    runs: list[RunData], n_perm: int, rank: int, seed: int
) -> tuple[np.ndarray, float, dict[str, float]]:
    real_rows = []
    for run in runs:
        result = transform_cv(run, ranks=(rank,), models=("rotation",))
        real_rows.append((run.patient, float(result["two_by_two_accuracy"].mean())))
    real = pd.DataFrame(real_rows, columns=["patient", "accuracy"])
    observed = float(real.groupby("patient")["accuracy"].mean().mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for perm_idx in range(n_perm):
        permutation = rng.permutation(10)
        rows = []
        for run in runs:
            result = transform_cv(
                run,
                ranks=(rank,),
                models=("rotation",),
                passive_pair_permutation=permutation,
            )
            rows.append((run.patient, float(result["two_by_two_accuracy"].mean())))
        frame = pd.DataFrame(rows, columns=["patient", "accuracy"])
        null[perm_idx] = frame.groupby("patient")["accuracy"].mean().mean()
    p_greater = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    p_two = float((np.sum(np.abs(null - 0.5) >= abs(observed - 0.5)) + 1) / (n_perm + 1))
    return null, observed, {"p_greater": p_greater, "p_two_sided": p_two}


def time_geometry(run: RunData) -> np.ndarray:
    joined = np.concatenate([run.time_active, run.time_passive], axis=0)
    mean = joined.reshape(-1, joined.shape[-1]).mean(axis=0, keepdims=True)
    sd = joined.reshape(-1, joined.shape[-1]).std(axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    active = (run.time_active - mean) / sd
    passive = (run.time_passive - mean) / sd
    matrix = np.full((len(WINDOWS_MS), len(WINDOWS_MS)), np.nan, dtype=float)
    for idx_a in range(len(WINDOWS_MS)):
        for idx_p in range(len(WINDOWS_MS)):
            role_z: list[float] = []
            for role in ROLES:
                idx = role_indices(run.item_meta, role)
                rho = spearman_r(
                    rdm_vector(cosine_rdm(active[idx, idx_a])),
                    rdm_vector(cosine_rdm(passive[idx, idx_p])),
                )
                role_z.append(fisher_z(rho))
            matrix[idx_a, idx_p] = float(np.mean(role_z))
    return matrix


def subspace_analysis(run: RunData, rank: int, n_perm: int) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    active, passive = standardize_joint(run.active, run.passive)
    active = active - active.mean(axis=0, keepdims=True)
    passive = passive - passive.mean(axis=0, keepdims=True)
    _, _, vt_a = np.linalg.svd(active, full_matrices=False)
    _, _, vt_p = np.linalg.svd(passive, full_matrices=False)
    k = min(rank, vt_a.shape[0], vt_p.shape[0])
    basis_a = vt_a[:k].T
    basis_p = vt_p[:k].T
    singular = np.linalg.svd(basis_a.T @ basis_p, compute_uv=False)
    alignment = float(np.mean(np.square(singular)))
    participation_a = np.sum(np.square(basis_a), axis=1)
    participation_p = np.sum(np.square(basis_p), axis=1)
    participation_rho = spearman_r(participation_a, participation_p)
    rng = np.random.default_rng(stable_seed("subspace", run.run, rank))
    null_alignment = np.empty(n_perm, dtype=float)
    null_participation = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        perm = rng.permutation(len(run.unit_ids))
        singular_null = np.linalg.svd(basis_a.T @ basis_p[perm], compute_uv=False)
        null_alignment[idx] = np.mean(np.square(singular_null))
        null_participation[idx] = spearman_r(participation_a, participation_p[perm])
    align_sd = null_alignment.std(ddof=1)
    result = {
        "subspace_alignment": alignment,
        "subspace_null_mean": float(null_alignment.mean()),
        "subspace_null_z": float((alignment - null_alignment.mean()) / align_sd) if align_sd > 0 else np.nan,
        "subspace_permutation_p": float((np.sum(null_alignment >= alignment) + 1) / (n_perm + 1)),
        "participation_rho": participation_rho,
        "participation_permutation_p": float(
            (np.sum(np.abs(null_participation) >= abs(participation_rho)) + 1) / (n_perm + 1)
        ),
        "mean_principal_angle_deg": float(np.degrees(np.mean(np.arccos(np.clip(singular, -1.0, 1.0))))),
    }
    units = pd.DataFrame(
        {
            "patient": run.patient,
            "run": run.run,
            "unit_id": run.unit_ids,
            "active_participation": participation_a,
            "passive_participation": participation_p,
        }
    )
    return result, units, null_alignment


def run_rdm_payload(run: RunData) -> dict[str, dict[str, np.ndarray]]:
    active, passive = standardize_joint(run.active, run.passive)
    payload: dict[str, dict[str, np.ndarray]] = {voice: {} for voice in VOICES}
    for role in ROLES:
        idx = role_indices(run.item_meta, role)
        payload["active"][role] = rdm_vector(cosine_rdm(active[idx]))
        payload["passive"][role] = rdm_vector(cosine_rdm(passive[idx]))
    return payload


def cross_patient_consensus(runs: list[RunData]) -> pd.DataFrame:
    patient_payload: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for patient in sorted({run.patient for run in runs}):
        patient_runs = [run for run in runs if run.patient == patient]
        patient_payload[patient] = {voice: {} for voice in VOICES}
        for voice in VOICES:
            for role in ROLES:
                values = [run_rdm_payload(run)[voice][role] for run in patient_runs]
                patient_payload[patient][voice][role] = np.mean(values, axis=0)
    rows: list[dict[str, object]] = []
    patients = sorted(patient_payload)
    comparisons = (
        ("active_to_active", "active", "active"),
        ("passive_to_passive", "passive", "passive"),
        ("active_to_other_passive", "active", "passive"),
        ("passive_to_other_active", "passive", "active"),
    )
    for patient in patients:
        others = [other for other in patients if other != patient]
        for label, target_voice, consensus_voice in comparisons:
            role_rhos = []
            for role in ROLES:
                target = patient_payload[patient][target_voice][role]
                consensus = np.mean(
                    [patient_payload[other][consensus_voice][role] for other in others], axis=0
                )
                role_rhos.append(spearman_r(target, consensus))
            z = float(np.mean([fisher_z(rho) for rho in role_rhos]))
            rows.append(
                {
                    "patient": patient,
                    "comparison": label,
                    "agent_rho": role_rhos[0],
                    "patient_rho": role_rhos[1],
                    "role_equal_fisher_z": z,
                    "role_equal_rho": inv_fisher(z),
                }
            )
    return pd.DataFrame(rows)


def summarize_patient_metric(
    frame: pd.DataFrame,
    value_col: str,
    metric: str,
    reference: float,
    n_boot: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    patient = frame.groupby("patient", as_index=False)[value_col].mean().rename(columns={value_col: "value"})
    patient["metric"] = metric
    sign = exact_sign_flip(patient["value"], reference)
    ci = bootstrap_patient_ci(patient["value"], n_boot, stable_seed("bootstrap", metric))
    summary = {
        "metric": metric,
        "reference": reference,
        "n_patients": int(len(patient)),
        "patient_mean": float(patient["value"].mean()),
        "patient_median": float(patient["value"].median()),
        "bootstrap_ci_low": ci[0],
        "bootstrap_ci_high": ci[1],
        "positive_patients": int(np.sum(patient["value"] > reference)),
        "exact_signflip_p_greater": sign["p_greater"],
        "exact_signflip_p_two_sided": sign["p_two_sided"],
    }
    return patient, summary


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def plot_patient_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: list[str],
    colors: dict[str, str],
    reference: float | None = None,
) -> None:
    patient_order = sorted(frame["patient"].unique())
    offsets = np.linspace(-0.14, 0.14, len(patient_order))
    for x_idx, label in enumerate(order):
        sub = frame[frame[x_col] == label]
        for offset, patient in zip(offsets, patient_order):
            value = sub.loc[sub["patient"] == patient, y_col]
            if not value.empty:
                ax.scatter(x_idx + offset, value.iloc[0], color=colors[label], s=28, alpha=0.85, edgecolor="white", linewidth=0.4)
        if not sub.empty:
            ax.scatter(x_idx, sub[y_col].mean(), color="black", marker="_", s=180, linewidth=2, zorder=5)
    if reference is not None:
        ax.axhline(reference, color=OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)


def make_figures(
    report_dir: Path,
    run_geometry: pd.DataFrame,
    patient_geometry: pd.DataFrame,
    transform_patient: pd.DataFrame,
    transform_run: pd.DataFrame,
    transform_null: np.ndarray,
    transform_observed: float,
    time_group_z: np.ndarray,
    time_stats: pd.DataFrame,
    subspace_patient: pd.DataFrame,
    ablation_patient: pd.DataFrame,
    consensus: pd.DataFrame,
) -> None:
    configure_style()
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.8, 6.7))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(np.zeros(len(patient_geometry)), np.tanh(patient_geometry["value"]), s=34, color=OKABE_ITO["purple"], edgecolor="white", linewidth=0.4)
    ax.scatter(0, np.tanh(patient_geometry["value"].mean()), marker="_", s=220, color="black", linewidth=2.2)
    ax.axhline(0, color=OKABE_ITO["gray"], ls="--", lw=0.9)
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([0])
    ax.set_xticklabels(["Role-stratified\nactive-passive RDM"])
    ax.set_ylabel("Patient endpoint (rho)")
    ax.set_title("Geometry preservation")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    model_order = list(MODELS)
    model_colors = {
        "identity": OKABE_ITO["gray"],
        "translation": OKABE_ITO["blue"],
        "rotation": OKABE_ITO["green"],
        "affine": OKABE_ITO["orange"],
    }
    plot_patient_points(ax, transform_patient, "model", "accuracy", model_order, model_colors, 0.5)
    ax.set_ylim(0.25, 0.82)
    ax.set_ylabel("Within-role 2-vs-2 accuracy")
    ax.set_title("Leave-two-pairs-out prediction (rank 4)")
    ax.tick_params(axis="x", rotation=18)
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(transform_null, bins=28, color=OKABE_ITO["light_gray"], edgecolor="white")
    ax.axvline(transform_observed, color=OKABE_ITO["green"], lw=2, label="Matched")
    ax.axvline(0.5, color=OKABE_ITO["gray"], lw=0.9, ls="--", label="Chance")
    ax.set_xlabel("Patient-equal rotation accuracy")
    ax.set_ylabel("Synchronized permutations")
    ax.set_title("Item-level rotation test")
    ax.legend(frameon=False)
    panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    consensus_order = ["active_to_active", "passive_to_passive", "active_to_other_passive"]
    consensus_colors = {
        "active_to_active": OKABE_ITO["blue"],
        "passive_to_passive": OKABE_ITO["vermillion"],
        "active_to_other_passive": OKABE_ITO["purple"],
    }
    plot_patient_points(ax, consensus, "comparison", "role_equal_rho", consensus_order, consensus_colors, 0.0)
    ax.set_xticklabels(["Active\nwithin voice", "Passive\nwithin voice", "Active to\nother-passive"])
    ax.set_ylabel("LOPO consensus rho")
    ax.set_title("Cross-patient common geometry")
    panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 1])
    run_plot = transform_run[(transform_run["rank"] == 4) & (transform_run["model"].isin(["translation", "rotation", "affine"]))]
    for model in ["translation", "rotation", "affine"]:
        sub = run_plot[run_plot["model"] == model]
        ax.scatter(sub["mean_angle_deg"], sub["two_by_two_accuracy"], s=32, color=model_colors[model], label=model, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color=OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set_xlabel("Held-out angular error (degrees)")
    ax.set_ylabel("Within-role 2-vs-2 accuracy")
    ax.set_title("Run-level model diagnostics")
    ax.legend(frameon=False)
    panel_label(ax, "E")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    text = (
        "Interpretation gate\n\n"
        "RDM > 0 + translation succeeds:\ncommon geometry with a common shift.\n\n"
        "RDM > 0 + rotation improves:\nshared geometry with reoriented readout.\n\n"
        "RDM near 0 + all maps fail:\nremapping or insufficient reliability.\n\n"
        "Voice remains confounded with marker identity."
    )
    ax.text(0.0, 1.0, text, va="top", fontsize=8.5, linespacing=1.35)
    panel_label(ax, "F")

    fig.suptitle("Active-to-passive noun geometry: translation, rotation, or remapping", fontsize=12, fontweight="bold", y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"Fig_1_geometry_transform_models.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10.8, 6.5))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(np.tanh(time_group_z), cmap="RdBu_r", vmin=-0.35, vmax=0.35, origin="lower")
    labels = [f"{start}-{stop}" for start, stop in WINDOWS_MS]
    ax.set_xticks(range(4), labels, rotation=30, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Passive noun window (ms)")
    ax.set_ylabel("Active noun window (ms)")
    ax.set_title("Time-resolved RDM")
    for row in time_stats.itertuples(index=False):
        marker = "*" if row.q_bh < 0.05 else ""
        ax.text(row.passive_window_index, row.active_window_index, f"{row.patient_mean_rho:.2f}{marker}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.58, pad=0.04, label="rho")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    sub_order = ["subspace_null_z", "participation_rho"]
    sub_colors = {"subspace_null_z": OKABE_ITO["orange"], "participation_rho": OKABE_ITO["blue"]}
    plot_patient_points(ax, subspace_patient, "metric", "value", sub_order, sub_colors, 0.0)
    ax.set_xticklabels(["Subspace\nalignment z", "Same-unit\nparticipation rho"])
    ax.set_ylabel("Patient endpoint")
    ax.set_title("Shared units and readout axes")
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(np.zeros(len(ablation_patient)), ablation_patient["angle_delta_top_minus_random"], s=34, color=OKABE_ITO["vermillion"], edgecolor="white", linewidth=0.4)
    ax.scatter(0, ablation_patient["angle_delta_top_minus_random"].mean(), marker="_", s=220, color="black", linewidth=2.2)
    ax.axhline(0, color=OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set_xticks([0], ["Top condition units\nminus random removal"])
    ax.set_ylabel("Angular-error change (degrees)")
    ax.set_title("Training-fold unit ablation")
    panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    diag = time_stats[time_stats["active_window_index"] == time_stats["passive_window_index"]]
    patient_diag_rows = []
    for row in diag.itertuples(index=False):
        values = json.loads(row.patient_values_json)
        for patient, value in values.items():
            patient_diag_rows.append({"window": row.active_window_index, "patient": patient, "rho": math.tanh(value)})
    patient_diag = pd.DataFrame(patient_diag_rows)
    for patient, sub in patient_diag.groupby("patient"):
        ax.plot(sub["window"], sub["rho"], color=OKABE_ITO["light_gray"], lw=0.9, alpha=0.9)
    means = patient_diag.groupby("window")["rho"].mean()
    ax.plot(means.index, means.values, color=OKABE_ITO["purple"], marker="o", lw=2)
    ax.axhline(0, color=OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set_xticks(range(4), labels, rotation=25, ha="right")
    ax.set_xlabel("Matched active/passive noun window (ms)")
    ax.set_ylabel("RDM rho")
    ax.set_title("Diagonal time course")
    panel_label(ax, "D")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    notes = (
        "Fixed before inspection\n\n"
        "- Four 200-ms noun windows\n"
        "- Rank 4 primary; rank 2 sensitivity\n"
        "- Leave two complete sentence pairs out\n"
        "- Patient is the independent unit\n"
        "- Pair permutations are synchronized\n"
        "  across patients and runs\n\n"
        "Asterisks: BH-FDR q < 0.05 across 16 cells."
    )
    ax.text(0.0, 1.0, notes, va="top", fontsize=8.5, linespacing=1.4)
    panel_label(ax, "E")

    fig.suptitle("Temporal geometry, subspace organization, and unit ablation", fontsize=12, fontweight="bold", y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"Fig_2_time_subspace_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_dir: Path,
    contract: dict[str, object],
    summaries: pd.DataFrame,
    transform_patient: pd.DataFrame,
    transform_run: pd.DataFrame,
    rdm_item_p: dict[str, float],
    transform_item_p: dict[str, float],
    time_stats: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    lookup = summaries.set_index("metric").to_dict("index")
    rdm = lookup["rdm_role_equal_fisher_z"]
    rotation = lookup["rotation_rank4_accuracy"]
    translation = lookup["translation_rank4_accuracy"]
    affine = lookup["affine_rank4_accuracy"]
    participation = lookup["participation_rho"]
    subspace = lookup["subspace_null_z"]
    ablation = lookup["ablation_angle_top_minus_random"]
    rotation_vs_translation_accuracy = lookup["rotation_vs_translation_accuracy_delta"]
    rotation_vs_translation_angle = lookup["rotation_vs_translation_angle_improvement_deg"]
    model_angles = transform_patient.groupby("model")["angle_deg"].mean().to_dict()
    rank2_rotation = (
        transform_run[(transform_run["rank"] == 2) & (transform_run["model"] == "rotation")]
        .groupby("patient")["two_by_two_accuracy"]
        .mean()
    )
    rank2_rotation_sign = exact_sign_flip(rank2_rotation, 0.5)
    significant_time = int((time_stats["q_bh"] < 0.05).sum())

    def p_text(value: float) -> str:
        return f"{value:.4f}" if np.isfinite(value) else "NA"

    if rdm["patient_mean"] > 0 and rdm_item_p["p_greater"] < 0.05:
        geometry_line = "跨句式名词关系几何在刺激块置换检验下得到正向支持。"
    elif rdm["patient_mean"] > 0 and rdm_item_p["p_greater"] < 0.10:
        geometry_line = "跨句式名词关系几何呈正向趋势，但未通过刺激块置换门槛。"
    else:
        geometry_line = "跨句式名词关系几何未显示稳定的正向保留。"
    if rotation["patient_mean"] > 0.5 and transform_item_p["p_greater"] < 0.05:
        transform_line = "正交旋转能够泛化预测未见句对中的被动句名词向量。"
    else:
        transform_line = "正交旋转尚不能在患者级与刺激级同时证明对未见句对的泛化预测。"

    lines = [
        "# 主动到被动的名词神经几何：平移、旋转还是重映射",
        "",
        "## 一句话结果",
        "",
        f"{geometry_line}{transform_line}",
        "",
        "## 冻结的分析问题",
        "",
        "同一个名词在主动和被动句中保持 thematic role 不变，但表面位置、功能词、声学上下文和句式发生变化。",
        "本分析检验这种变化能否由共同平移、正交旋转或一般仿射变换描述；它不把主动/被动解释为独立于功能词的抽象语态。",
        "",
        "## 数据与推断",
        "",
        f"- 主规格：`{contract['unit_set']}`；伪迹阈值：`{contract['artifact_threshold']}`（仅在有标注的 run 应用）。",
        f"- 纳入 {int(audit['included'].sum())} 个 run、{audit.loc[audit['included'], 'patient'].nunique()} 位患者；患者等权。",
        "- 20个同词 active/passive 配对；Agent 与 Patient 各自计算 RDM，再以 Fisher-z 等权平均。",
        "- 变换模型完整留出两个 sentence pairs；PCA、unit scaling、变换和 ridge 选择均不接触外层测试项目。",
        "- 患者层 exact sign-flip 与跨患者同步的句对块置换分别约束患者和刺激两个推广维度。",
        "",
        "## 1. 关系几何是否保留",
        "",
        f"- 患者等权平均 Fisher-z={rdm['patient_mean']:.3f}，对应 rho={math.tanh(rdm['patient_mean']):.3f}。",
        f"- 95% patient bootstrap CI（Fisher-z）[{rdm['bootstrap_ci_low']:.3f}, {rdm['bootstrap_ci_high']:.3f}]。",
        f"- 5患者单侧 exact sign-flip P={p_text(rdm['exact_signflip_p_greater'])}；同步句对块置换 P={p_text(rdm_item_p['p_greater'])}。",
        "",
        "## 2. 平移、旋转和仿射模型的严格留出预测",
        "",
        f"- 恒等模型 patient-equal accuracy={lookup['identity_rank4_accuracy']['patient_mean']:.3f}，angular error={model_angles['identity']:.2f} degrees。",
        f"- 平移模型 accuracy={translation['patient_mean']:.3f}，angular error={model_angles['translation']:.2f} degrees。",
        f"- 正交旋转模型 accuracy={rotation['patient_mean']:.3f}，angular error={model_angles['rotation']:.2f} degrees；患者 sign-flip P={p_text(rotation['exact_signflip_p_greater'])}；同步刺激置换 P={p_text(transform_item_p['p_greater'])}。",
        f"- 正则化仿射模型 accuracy={affine['patient_mean']:.3f}，angular error={model_angles['affine']:.2f} degrees。",
        f"- 旋转相对平移的 accuracy 差={rotation_vs_translation_accuracy['patient_mean']:.3f}（单侧 paired sign-flip P={p_text(rotation_vs_translation_accuracy['exact_signflip_p_greater'])}）；angular-error 改善={rotation_vs_translation_angle['patient_mean']:.2f} degrees（P={p_text(rotation_vs_translation_angle['exact_signflip_p_greater'])}）。",
        f"- Rank-2 敏感性：旋转 accuracy={rank2_rotation.mean():.3f}，患者单侧 sign-flip P={p_text(rank2_rotation_sign['p_greater'])}。",
        "- 模型差值为患者内配对比较；accuracy 差或 angular-error 改善为正时，表示前一个模型更好。不能根据训练拟合或挑选最佳 rank 下结论。",
        "",
        "## 3. 时间、子空间与单元参与",
        "",
        f"- 4x4 固定时间矩阵中 BH-FDR q<0.05 的 cell 数={significant_time}/16。",
        f"- 主动/被动 unit-space 子空间相对 unit-identity shuffle 的患者平均 z={subspace['patient_mean']:.3f}。",
        f"- 同一 units 的 participation-strength 患者平均 rho={participation['patient_mean']:.3f}；5患者双侧 exact sign-flip P={p_text(participation['exact_signflip_p_two_sided'])}。",
        f"- 移除训练折内最能区分句式的20% units，相比匹配随机移除使 angular error 改变 {ablation['patient_mean']:.3f} degrees（单侧 P={p_text(ablation['exact_signflip_p_greater'])}）。",
        "",
        "## 解释边界",
        "",
        "1. active/passive 与功能词类别完全共线，故结果只能称为跨句式/标记/表面位置变化的关系几何。",
        "2. Agent 与生命性、Patient 与非生命性混杂；主分析通过角色内 RDM 和角色内检索避免由两类均值直接驱动，但不能识别独立的 thematic-role axis。",
        "3. 每个名词-句式在单个 run 中只有一次观察，不能据此确认单神经元的稳定跨情境 tuning。",
        "4. 该分析由新文献启发并在既有结果之后开展，应标为 exploratory；所有阴性结果同样保留。",
        "",
        "## 输出",
        "",
        "- `run_geometry_endpoints.csv`, `patient_geometry_endpoints.csv`",
        "- `transform_cv_folds.csv`, `transform_run_summary.csv`, `transform_patient_summary.csv`, `transform_model_comparisons_patient.csv`",
        "- `time_geometry_cell_stats.csv`, `subspace_run_summary.csv`, `ablation_cv_folds.csv`",
        "- `cross_patient_consensus.csv`, `group_inference.csv`, `summary.json`",
        "- `figures/Fig_1_geometry_transform_models.png/.pdf`",
        "- `figures/Fig_2_time_subspace_ablation.png/.pdf`",
        "",
    ]
    (report_dir / "README_zh.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    data_dir = Path(args.data_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    artifact_threshold = None if args.artifact_threshold < 0 else args.artifact_threshold
    contract = {
        "analysis_date": "2026-08-06",
        "status": "post-result exploratory literature-motivated analysis",
        "estimand": "same-noun active-to-passive geometry with thematic role held fixed",
        "unit_set": args.unit_set,
        "metric": METRIC,
        "artifact_threshold": artifact_threshold,
        "artifact_policy": "threshold applied only to runs with available annotations",
        "roles": list(ROLES),
        "windows_ms": [list(window) for window in WINDOWS_MS],
        "primary_rank": 4,
        "sensitivity_rank": 2,
        "outer_cv": "leave two complete sentence pairs out",
        "primary_geometry": "role-stratified cosine RDM Spearman correlation, Fisher-z role equal",
        "primary_transform": "rank-4 orthogonal Procrustes, within-role 2-vs-2 retrieval",
        "patient_inference": "patient-equal exhaustive exact sign flip",
        "item_inference": "synchronized sentence-pair block permutation across runs and patients",
        "n_rdm_permutations": args.n_permutations,
        "n_transform_permutations": args.transform_null_permutations,
        "n_bootstrap": args.n_bootstrap,
        "interpretation_limit": "voice is confounded with function-word identity and acoustics; role is confounded with animacy and lexical class",
    }
    (report_dir / "analysis_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics, available_artifact_runs = load_filtered_metrics(args.unit_set, artifact_threshold)
    runs, audit = build_run_data(metrics, available_artifact_runs, args.unit_set)
    if len(runs) < 5 or len({run.patient for run in runs}) < 5:
        raise RuntimeError(f"Insufficient complete data after filtering:\n{audit.to_string(index=False)}")
    audit.to_csv(data_dir / "run_inclusion_audit.csv", index=False)
    audit.to_csv(report_dir / "run_inclusion_audit.csv", index=False)

    geometry_rows = []
    for run in runs:
        endpoint = geometry_endpoint(run.active, run.passive, run.item_meta)
        geometry_rows.append(
            {
                "patient": run.patient,
                "run": run.run,
                "n_units": len(run.unit_ids),
                "artifact_annotations_available": run.artifact_annotations_available,
                **endpoint,
            }
        )
    run_geometry = pd.DataFrame(geometry_rows)
    run_geometry.to_csv(data_dir / "run_geometry_endpoints.csv", index=False)
    patient_geometry = run_geometry.groupby("patient", as_index=False)["role_equal_fisher_z"].mean().rename(columns={"role_equal_fisher_z": "value"})
    patient_geometry["role_equal_rho"] = np.tanh(patient_geometry["value"])
    patient_geometry.to_csv(data_dir / "patient_geometry_endpoints.csv", index=False)

    rdm_null, rdm_observed, rdm_item_p = synchronized_rdm_permutation(
        runs, args.n_permutations, stable_seed("synchronized-rdm")
    )
    np.save(data_dir / "primary_rdm_synchronized_pair_permutation_null.npy", rdm_null)

    transform_folds = pd.concat([transform_cv(run) for run in runs], ignore_index=True)
    transform_folds.to_csv(data_dir / "transform_cv_folds.csv", index=False)
    transform_run = transform_folds.groupby(["patient", "run", "rank", "model"], as_index=False).agg(
        mean_angle_deg=("mean_angle_deg", "mean"),
        two_by_two_accuracy=("two_by_two_accuracy", "mean"),
    )
    transform_run.to_csv(data_dir / "transform_run_summary.csv", index=False)
    transform_patient = transform_run[transform_run["rank"] == 4].groupby(["patient", "model"], as_index=False).agg(
        accuracy=("two_by_two_accuracy", "mean"), angle_deg=("mean_angle_deg", "mean")
    )
    transform_patient.to_csv(data_dir / "transform_patient_summary.csv", index=False)
    transform_wide = transform_patient.pivot(index="patient", columns="model", values=["accuracy", "angle_deg"])
    comparison_specs = (
        ("translation_vs_identity", "translation", "identity"),
        ("rotation_vs_identity", "rotation", "identity"),
        ("rotation_vs_translation", "rotation", "translation"),
        ("affine_vs_rotation", "affine", "rotation"),
    )
    model_comparison_rows = []
    for label, model_a, model_b in comparison_specs:
        for patient in transform_wide.index:
            model_comparison_rows.append(
                {
                    "patient": patient,
                    "comparison": label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "accuracy_delta": float(
                        transform_wide.loc[patient, ("accuracy", model_a)]
                        - transform_wide.loc[patient, ("accuracy", model_b)]
                    ),
                    "angle_improvement_deg": float(
                        transform_wide.loc[patient, ("angle_deg", model_b)]
                        - transform_wide.loc[patient, ("angle_deg", model_a)]
                    ),
                }
            )
    model_comparisons = pd.DataFrame(model_comparison_rows)
    model_comparisons.to_csv(data_dir / "transform_model_comparisons_patient.csv", index=False)

    transform_null, transform_observed, transform_item_p = synchronized_transform_permutation(
        runs,
        args.transform_null_permutations,
        rank=4,
        seed=stable_seed("synchronized-transform"),
    )
    np.save(data_dir / "primary_rotation_synchronized_pair_permutation_null.npy", transform_null)

    time_matrices = []
    time_run_rows = []
    for run in runs:
        matrix = time_geometry(run)
        time_matrices.append(matrix)
        for idx_a, (start_a, stop_a) in enumerate(WINDOWS_MS):
            for idx_p, (start_p, stop_p) in enumerate(WINDOWS_MS):
                time_run_rows.append(
                    {
                        "patient": run.patient,
                        "run": run.run,
                        "active_window_index": idx_a,
                        "passive_window_index": idx_p,
                        "active_window_ms": f"{start_a}-{stop_a}",
                        "passive_window_ms": f"{start_p}-{stop_p}",
                        "role_equal_fisher_z": matrix[idx_a, idx_p],
                        "role_equal_rho": inv_fisher(matrix[idx_a, idx_p]),
                    }
                )
    np.savez_compressed(
        data_dir / "time_geometry_run_matrices.npz",
        matrices=np.asarray(time_matrices),
        patients=np.asarray([run.patient for run in runs], dtype=object),
        runs=np.asarray([run.run for run in runs], dtype=object),
        windows_ms=np.asarray(WINDOWS_MS),
    )
    time_run = pd.DataFrame(time_run_rows)
    time_run.to_csv(data_dir / "time_geometry_run_cells.csv", index=False)
    time_stat_rows = []
    for (idx_a, idx_p), frame in time_run.groupby(["active_window_index", "passive_window_index"], sort=True):
        patient_values = frame.groupby("patient")["role_equal_fisher_z"].mean()
        sign = exact_sign_flip(patient_values, 0.0)
        time_stat_rows.append(
            {
                "active_window_index": idx_a,
                "passive_window_index": idx_p,
                "patient_mean_fisher_z": float(patient_values.mean()),
                "patient_mean_rho": inv_fisher(float(patient_values.mean())),
                "positive_patients": int((patient_values > 0).sum()),
                "exact_signflip_p_greater": sign["p_greater"],
                "patient_values_json": json.dumps(patient_values.to_dict(), ensure_ascii=False),
            }
        )
    time_stats = pd.DataFrame(time_stat_rows)
    time_stats["q_bh"] = bh_fdr(time_stats["exact_signflip_p_greater"])
    time_stats.to_csv(data_dir / "time_geometry_cell_stats.csv", index=False)
    time_group_z = time_stats.pivot(index="active_window_index", columns="passive_window_index", values="patient_mean_fisher_z").to_numpy()

    subspace_rows = []
    unit_participation = []
    subspace_nulls = []
    for run in runs:
        result, units, null = subspace_analysis(run, rank=4, n_perm=args.n_permutations)
        subspace_rows.append({"patient": run.patient, "run": run.run, "rank": 4, **result})
        unit_participation.append(units)
        for value in null:
            subspace_nulls.append({"patient": run.patient, "run": run.run, "null_alignment": value})
    subspace_run = pd.DataFrame(subspace_rows)
    subspace_run.to_csv(data_dir / "subspace_run_summary.csv", index=False)
    pd.concat(unit_participation, ignore_index=True).to_csv(data_dir / "unit_participation.csv", index=False)
    pd.DataFrame(subspace_nulls).to_csv(data_dir / "subspace_alignment_nulls.csv", index=False)

    ablation_folds = pd.concat(
        [ablation_cv(run, rank=4, random_repeats=args.ablation_random_repeats) for run in runs],
        ignore_index=True,
    )
    ablation_folds.to_csv(data_dir / "ablation_cv_folds.csv", index=False)
    ablation_run = ablation_folds.groupby(["patient", "run"], as_index=False).agg(
        angle_delta_top_minus_random=("angle_delta_top_minus_random", "mean"),
        accuracy_delta_top_minus_random=("accuracy_delta_top_minus_random", "mean"),
    )
    ablation_run.to_csv(data_dir / "ablation_run_summary.csv", index=False)
    ablation_patient = ablation_run.groupby("patient", as_index=False).mean(numeric_only=True)
    ablation_patient.to_csv(data_dir / "ablation_patient_summary.csv", index=False)

    consensus = cross_patient_consensus(runs)
    consensus.to_csv(data_dir / "cross_patient_consensus.csv", index=False)

    patient_tables = []
    group_summaries = []
    patient, summary = summarize_patient_metric(
        run_geometry, "role_equal_fisher_z", "rdm_role_equal_fisher_z", 0.0, args.n_bootstrap
    )
    patient_tables.append(patient)
    group_summaries.append(summary)
    for model in MODELS:
        frame = transform_run[(transform_run["rank"] == 4) & (transform_run["model"] == model)]
        patient, summary = summarize_patient_metric(
            frame, "two_by_two_accuracy", f"{model}_rank4_accuracy", 0.5, args.n_bootstrap
        )
        patient_tables.append(patient)
        group_summaries.append(summary)
    for comparison in model_comparisons["comparison"].unique():
        frame = model_comparisons[model_comparisons["comparison"] == comparison]
        for column, suffix in (
            ("accuracy_delta", "accuracy_delta"),
            ("angle_improvement_deg", "angle_improvement_deg"),
        ):
            patient, summary = summarize_patient_metric(
                frame,
                column,
                f"{comparison}_{suffix}",
                0.0,
                args.n_bootstrap,
            )
            patient_tables.append(patient)
            group_summaries.append(summary)
    for metric, col, reference in (
        ("subspace_null_z", "subspace_null_z", 0.0),
        ("participation_rho", "participation_rho", 0.0),
    ):
        patient, summary = summarize_patient_metric(subspace_run, col, metric, reference, args.n_bootstrap)
        patient_tables.append(patient)
        group_summaries.append(summary)
    patient, summary = summarize_patient_metric(
        ablation_run,
        "angle_delta_top_minus_random",
        "ablation_angle_top_minus_random",
        0.0,
        args.n_bootstrap,
    )
    patient_tables.append(patient)
    group_summaries.append(summary)
    for comparison in consensus["comparison"].unique():
        frame = consensus[consensus["comparison"] == comparison].copy()
        patient, summary = summarize_patient_metric(
            frame,
            "role_equal_fisher_z",
            f"consensus_{comparison}_fisher_z",
            0.0,
            args.n_bootstrap,
        )
        patient_tables.append(patient)
        group_summaries.append(summary)

    patient_endpoints = pd.concat(patient_tables, ignore_index=True)
    patient_endpoints.to_csv(data_dir / "all_patient_endpoints.csv", index=False)
    group_inference = pd.DataFrame(group_summaries)
    group_inference.loc[group_inference["metric"] == "rdm_role_equal_fisher_z", "item_permutation_p_greater"] = rdm_item_p["p_greater"]
    group_inference.loc[group_inference["metric"] == "rotation_rank4_accuracy", "item_permutation_p_greater"] = transform_item_p["p_greater"]
    group_inference.to_csv(data_dir / "group_inference.csv", index=False)

    subspace_patient_rows = []
    for metric in ["subspace_null_z", "participation_rho"]:
        frame = patient_endpoints[patient_endpoints["metric"] == metric]
        for row in frame.itertuples(index=False):
            subspace_patient_rows.append({"patient": row.patient, "metric": metric, "value": row.value})
    subspace_patient = pd.DataFrame(subspace_patient_rows)

    make_figures(
        report_dir,
        run_geometry,
        patient_geometry,
        transform_patient,
        transform_run,
        transform_null,
        transform_observed,
        time_group_z,
        time_stats,
        subspace_patient,
        ablation_patient,
        consensus,
    )

    summary_payload = {
        "analysis_contract": contract,
        "run_inclusion": audit.to_dict(orient="records"),
        "group_inference": group_inference.to_dict(orient="records"),
        "primary_rdm": {"observed_group_mean_fisher_z": rdm_observed, **rdm_item_p},
        "primary_rotation": {"observed_group_accuracy": transform_observed, **transform_item_p},
        "time_fdr_significant_cells": int((time_stats["q_bh"] < 0.05).sum()),
    }
    (data_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        report_dir,
        contract,
        group_inference,
        transform_patient,
        transform_run,
        rdm_item_p,
        transform_item_p,
        time_stats,
        audit,
    )

    manifest_rows = []
    for path in sorted(list(report_dir.rglob("*")) + list(data_dir.glob("*"))):
        if path.is_file():
            manifest_rows.append({"path": str(path), "size_bytes": path.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(report_dir / "delivery_manifest.csv", index=False)

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=("good_only", "usable_lenient"), default="good_only")
    parser.add_argument("--artifact-threshold", type=float, default=0.2, help="Use -1 to disable filtering")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--transform-null-permutations", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--ablation-random-repeats", type=int, default=20)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
