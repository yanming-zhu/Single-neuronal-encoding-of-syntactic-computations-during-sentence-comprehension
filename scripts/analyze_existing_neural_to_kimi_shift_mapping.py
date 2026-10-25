#!/usr/bin/env python3
"""Test whether existing noun responses map onto open-weight Kimi states.

Primary estimand
-----------------
For each of the 20 repeated nouns, compute the passive-minus-active change
vector in neural activity and in Kimi.  The thematic role of the noun is held
constant while voice, construction marker, and surface position change.

The primary test is a patient-level, role- and pair-controlled RSA using all
four fixed post-onset windows and three fixed Kimi depth bands jointly.  A
strict nested leave-two-sentence-pairs-out PCA+ridge analysis is secondary.
All preprocessing and hyperparameter selection in the mapping analysis are
fit inside the training fold.

This analysis cannot identify an abstract Agent/Patient code because noun
identity, animacy, and thematic role are collinear in the stimulus set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

from scripts.analyze_noun_firstchar_llm_mapping import (  # noqa: E402
    EVENTS_PATH,
    KIMI_ID,
    REGISTRY_PATH,
    REP_DIR,
    RUNS,
    WINDOWS_MS,
    neural_windows,
    run_npz_path,
)
from scripts.analyze_noun_role_structure_encoding import (  # noqa: E402
    PATIENTS,
    setup_matplotlib,
)


BASE_OUTPUT = ROOT / "reports/existing_neural_to_kimi_lowdim_mapping_20260730"
OUTPUT = BASE_OUTPUT
FIGURES = OUTPUT / "figures"
CURRENT_UNIT_SET = "usable_lenient"
FULL_KIMI = (
    ROOT
    / "reports/multirun_grouped_go_20260713/llm"
    / "exploratory_kimi_vl_a3b_4bit_20260723"
    / "representation_cache/representations.npz"
)
FIRSTCHAR_KIMI = REP_DIR / "kimi_firstchar.npz"

SEED = 20260730
PRIMARY_READOUT = "full_noun_all_depth"
NEURAL_RANKS = (2, 4, 8)
KIMI_RANKS = (2, 4)
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
WINDOW_LABELS = tuple(f"{left}–{right}" for left, right in WINDOWS_MS)


@dataclass(frozen=True)
class TargetSet:
    name: str
    observation_states: np.ndarray
    observation_meta: pd.DataFrame
    shift_states: np.ndarray
    noun_meta: pd.DataFrame
    band_shift_states: dict[str, np.ndarray]


@dataclass(frozen=True)
class LatentFit:
    train_scores: np.ndarray
    test_scores: np.ndarray
    components: np.ndarray
    mean: np.ndarray


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def ensure_dirs() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.where(norms > 1e-12, norms, 1.0)


def fixed_depth_bands(hidden: np.ndarray) -> dict[str, np.ndarray]:
    """Return three jointly prespecified depth bands without selecting a layer."""
    hidden = normalize_rows(np.asarray(hidden, dtype=np.float64))
    indices = np.array_split(np.arange(hidden.shape[1]), 3)
    names = ("early", "middle", "late")
    output: dict[str, np.ndarray] = {}
    for name, band_indices in zip(names, indices):
        band = hidden[:, band_indices].mean(axis=1)
        output[name] = normalize_rows(band)
    return output


def integrated_depth_state(bands: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([bands[name] for name in ("early", "middle", "late")], axis=1) / math.sqrt(3.0)


def noun_meta_from_registry() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY_PATH)
    grammar = registry.loc[registry["condition"].isin(["active", "passive"])].copy()
    rows: list[dict[str, object]] = []
    for noun, cell in grammar.groupby("noun", sort=False):
        if set(cell["condition"].astype(str)) != {"active", "passive"} or len(cell) != 2:
            raise RuntimeError(f"Noun does not have one active and one passive item: {noun}")
        row = cell.iloc[0]
        rows.append(
            {
                "noun": str(noun),
                "pair_id": int(row["pair_id"]),
                "thematic_role": str(row["thematic_role"]),
                "first_char": str(row["first_char"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["role_order"] = frame["thematic_role"].map({"agent": 0, "patient": 1})
    frame = frame.sort_values(["role_order", "pair_id"]).drop(columns="role_order").reset_index(drop=True)
    if len(frame) != 20 or frame["pair_id"].nunique() != 10:
        raise RuntimeError("Expected 20 nouns in 10 sentence pairs")
    return frame


def build_target(
    *,
    name: str,
    hidden: np.ndarray,
    observation_meta: pd.DataFrame,
    noun_meta: pd.DataFrame,
) -> TargetSet:
    bands = fixed_depth_bands(hidden)
    integrated = integrated_depth_state(bands)
    lookup = {
        (str(row.noun), str(row.voice)): index
        for index, row in observation_meta.reset_index(drop=True).iterrows()
    }
    shifts: list[np.ndarray] = []
    band_shifts: dict[str, list[np.ndarray]] = {name: [] for name in bands}
    for row in noun_meta.itertuples(index=False):
        active = lookup[(str(row.noun), "active")]
        passive = lookup[(str(row.noun), "passive")]
        shifts.append(integrated[passive] - integrated[active])
        for band_name, values in bands.items():
            band_shifts[band_name].append(values[passive] - values[active])
    return TargetSet(
        name=name,
        observation_states=integrated,
        observation_meta=observation_meta.reset_index(drop=True),
        shift_states=np.stack(shifts),
        noun_meta=noun_meta.copy(),
        band_shift_states={key: np.stack(value) for key, value in band_shifts.items()},
    )


def load_targets() -> dict[str, TargetSet]:
    noun_meta = noun_meta_from_registry()
    with np.load(FULL_KIMI) as archive:
        noun_rows = np.asarray(archive["constituent_ids"]).astype(str) != "verb"
        hidden = np.asarray(archive["hidden_last"], dtype=np.float64)[noun_rows]
        full_meta = pd.DataFrame(
            {
                "sentence_id": np.asarray(archive["sentence_ids"]).astype(str)[noun_rows],
                "pair_id": np.asarray(archive["sentence_pair_ids"]).astype(int)[noun_rows],
                "voice": np.asarray(archive["voices"]).astype(str)[noun_rows],
                "noun": np.asarray(archive["target_text"]).astype(str)[noun_rows],
                "constituent": np.asarray(archive["constituent_ids"]).astype(str)[noun_rows],
            }
        )
    full_meta["thematic_role"] = full_meta["noun"].map(
        noun_meta.set_index("noun")["thematic_role"]
    )
    full = build_target(
        name=PRIMARY_READOUT,
        hidden=hidden,
        observation_meta=full_meta,
        noun_meta=noun_meta,
    )

    with np.load(FIRSTCHAR_KIMI) as archive:
        grammar_rows = np.isin(np.asarray(archive["conditions"]).astype(str), ["active", "passive"])
        first_hidden = np.asarray(archive["hidden_first_char"], dtype=np.float64)[grammar_rows]
        first_meta = pd.DataFrame(
            {
                "sentence_id": np.asarray(archive["sentence_ids"]).astype(str)[grammar_rows],
                "pair_id": np.asarray(archive["pair_ids"]).astype(int)[grammar_rows],
                "voice": np.asarray(archive["conditions"]).astype(str)[grammar_rows],
                "noun": np.asarray(archive["nouns"]).astype(str)[grammar_rows],
                "thematic_role": np.asarray(archive["thematic_roles"]).astype(str)[grammar_rows],
            }
        )
    first = build_target(
        name="first_character_all_depth",
        hidden=first_hidden,
        observation_meta=first_meta,
        noun_meta=noun_meta,
    )
    return {full.name: full, first.name: first}


def grammar_neural_patterns(
    patient: str,
    run: str,
    events: pd.DataFrame,
    registry: pd.DataFrame,
    unit_set: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    run_events = events.loc[events["patient"].eq(patient) & events["run"].eq(run)]
    if unit_set == "usable_lenient":
        all_values = neural_windows(patient, run, run_events, registry)
    else:
        with np.load(run_npz_path(patient, run, unit_set), allow_pickle=True) as archive:
            firing = np.asarray(archive["firing_rate_hz"], dtype=np.float32)
        event_lookup = run_events.set_index("observation_id")
        all_values = np.empty(
            (len(registry), len(WINDOWS_MS), firing.shape[0]), dtype=np.float32
        )
        for observation_index, observation_id in enumerate(
            registry["observation_id"].astype(str)
        ):
            row = event_lookup.loc[observation_id]
            event_bin = int(row.event_bin_1000Hz)
            baseline = firing[:, event_bin - 200 : event_bin].mean(axis=1)
            for window_index, (left, right) in enumerate(WINDOWS_MS):
                all_values[observation_index, window_index] = (
                    firing[:, event_bin + left : event_bin + right].mean(axis=1)
                    - baseline
                )
    grammar_rows = registry["condition"].isin(["active", "passive"]).to_numpy()
    values = all_values[grammar_rows].reshape(int(grammar_rows.sum()), -1).astype(np.float64)
    meta = registry.loc[grammar_rows, ["noun", "pair_id", "condition", "thematic_role"]].copy()
    meta = meta.rename(columns={"condition": "voice"}).reset_index(drop=True)
    return values, meta


def neural_shift(
    observation_values: np.ndarray,
    observation_meta: pd.DataFrame,
    noun_meta: pd.DataFrame,
    *,
    scale_for_geometry: bool,
) -> np.ndarray:
    values = np.asarray(observation_values, dtype=np.float64)
    if scale_for_geometry:
        scale = values.std(axis=0)
        values = values / np.where(scale > 1e-8, scale, 1.0)
    lookup = {
        (str(row.noun), str(row.voice)): index
        for index, row in observation_meta.reset_index(drop=True).iterrows()
    }
    return np.stack(
        [
            values[lookup[(str(row.noun), "passive")]]
            - values[lookup[(str(row.noun), "active")]]
            for row in noun_meta.itertuples(index=False)
        ]
    )


def correlation_rdm(patterns: np.ndarray) -> np.ndarray:
    patterns = np.asarray(patterns, dtype=np.float64)
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    normalized = normalize_rows(centered)
    rdm = 1.0 - normalized @ normalized.T
    np.fill_diagonal(rdm, 0.0)
    return rdm


def control_design(meta: pd.DataFrame) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    triangle = np.triu_indices(len(meta), 1)
    left, right = triangle
    roles = meta["thematic_role"].astype(str).to_numpy()
    pairs = meta["pair_id"].astype(int).to_numpy()
    design = np.column_stack(
        [
            np.ones(len(left)),
            (roles[left] == roles[right]).astype(float),
            (pairs[left] == pairs[right]).astype(float),
        ]
    )
    return design, triangle


def residualized_rank_rdm(
    rdm: np.ndarray,
    design: np.ndarray,
    triangle: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    values = rankdata(rdm[triangle])
    residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    residual -= residual.mean()
    norm = np.linalg.norm(residual)
    return residual / norm if norm > 1e-12 else residual


def controlled_rsa(
    neural_rdm: np.ndarray,
    model_rdm: np.ndarray,
    meta: pd.DataFrame,
) -> float:
    design, triangle = control_design(meta)
    neural = residualized_rank_rdm(neural_rdm, design, triangle)
    model = residualized_rank_rdm(model_rdm, design, triangle)
    return float(neural @ model)


def exact_signflip(values: Iterable[float], null: float = 0.0) -> float:
    centered = np.asarray(list(values), dtype=float) - null
    observed = float(centered.mean())
    null_values = []
    for mask in range(1 << len(centered)):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(len(centered))]
        )
        null_values.append(float(np.mean(centered * signs)))
    return float(np.mean(np.asarray(null_values) >= observed - 1e-15))


def geometry_analysis(
    targets: dict[str, TargetSet],
    neural_by_run: dict[tuple[str, str], tuple[np.ndarray, pd.DataFrame]],
    permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    run_rows: list[dict[str, object]] = []
    neural_rdms: dict[tuple[str, str], np.ndarray] = {}
    meta = targets[PRIMARY_READOUT].noun_meta
    for (patient, run), (observation_values, observation_meta) in neural_by_run.items():
        shift = neural_shift(
            observation_values,
            observation_meta,
            meta,
            scale_for_geometry=True,
        )
        neural_rdm = correlation_rdm(shift)
        neural_rdms[(patient, run)] = neural_rdm
        for target_name, target in targets.items():
            readouts = {"all_depth": target.shift_states, **target.band_shift_states}
            for depth_readout, target_shift in readouts.items():
                rho = controlled_rsa(neural_rdm, correlation_rdm(target_shift), meta)
                run_rows.append(
                    {
                        "patient": patient,
                        "run": run,
                        "target": target_name,
                        "depth_readout": depth_readout,
                        "controlled_spearman_rho": rho,
                    }
                )
    run_frame = pd.DataFrame(run_rows)

    patient_rdm: dict[str, np.ndarray] = {}
    for patient in PATIENTS:
        matrices = [matrix for (owner, _run), matrix in neural_rdms.items() if owner == patient]
        if not matrices:
            raise RuntimeError(f"No neural RDM for {patient}")
        patient_rdm[patient] = np.mean(matrices, axis=0)

    patient_rows: list[dict[str, object]] = []
    for target_name, target in targets.items():
        readouts = {"all_depth": target.shift_states, **target.band_shift_states}
        for depth_readout, target_shift in readouts.items():
            model_rdm = correlation_rdm(target_shift)
            for patient in PATIENTS:
                patient_rows.append(
                    {
                        "patient": patient,
                        "target": target_name,
                        "depth_readout": depth_readout,
                        "controlled_spearman_rho": controlled_rsa(
                            patient_rdm[patient], model_rdm, meta
                        ),
                    }
                )
    patient_frame = pd.DataFrame(patient_rows)

    inference_rows: list[dict[str, object]] = []
    for (target_name, depth_readout), cell in patient_frame.groupby(
        ["target", "depth_readout"], sort=False
    ):
        values = cell.set_index("patient").loc[list(PATIENTS), "controlled_spearman_rho"].to_numpy(float)
        inference_rows.append(
            {
                "target": target_name,
                "depth_readout": depth_readout,
                "patient_mean_rho": float(values.mean()),
                "patient_median_rho": float(np.median(values)),
                "patient_min_rho": float(values.min()),
                "patient_max_rho": float(values.max()),
                "exact_patient_signflip_p_one_sided": exact_signflip(values),
                "primary": bool(target_name == PRIMARY_READOUT and depth_readout == "all_depth"),
            }
        )
    inference = pd.DataFrame(inference_rows)
    inference["pair_block_permutation_count"] = np.nan
    inference["pair_block_permutation_p_one_sided"] = np.nan
    inference["passes_both_patient_and_item_0p05"] = pd.Series(
        [pd.NA] * len(inference), dtype="boolean"
    )

    # Stimulus-label randomization for the one frozen primary cell.  Permute
    # sentence pairs as blocks, preserving role and the Agent/Patient pairing.
    design, triangle = control_design(meta)
    neural_vectors = {
        patient: residualized_rank_rdm(rdm, design, triangle)
        for patient, rdm in patient_rdm.items()
    }
    target = targets[PRIMARY_READOUT]
    observed_model = residualized_rank_rdm(
        correlation_rdm(target.shift_states), design, triangle
    )
    observed = float(np.mean([vector @ observed_model for vector in neural_vectors.values()]))
    rng = np.random.default_rng(SEED)
    pairs = np.sort(meta["pair_id"].unique())
    pair_to_indices = {
        int(pair): {
            role: int(
                meta.index[
                    meta["pair_id"].eq(pair) & meta["thematic_role"].eq(role)
                ][0]
            )
            for role in ("agent", "patient")
        }
        for pair in pairs
    }
    null = np.empty(permutations, dtype=float)
    for permutation_index in range(permutations):
        source_pairs = rng.permutation(pairs)
        row_permutation = np.arange(len(meta))
        for destination_pair, source_pair in zip(pairs, source_pairs):
            for role in ("agent", "patient"):
                destination = pair_to_indices[int(destination_pair)][role]
                source = pair_to_indices[int(source_pair)][role]
                row_permutation[destination] = source
        permuted = target.shift_states[row_permutation]
        model_vector = residualized_rank_rdm(correlation_rdm(permuted), design, triangle)
        null[permutation_index] = np.mean(
            [vector @ model_vector for vector in neural_vectors.values()]
        )
    permutation_p = float((1 + np.sum(null >= observed - 1e-15)) / (permutations + 1))
    primary_mask = inference["primary"]
    inference.loc[primary_mask, "pair_block_permutation_count"] = permutations
    inference.loc[primary_mask, "pair_block_permutation_p_one_sided"] = permutation_p
    inference.loc[primary_mask, "passes_both_patient_and_item_0p05"] = bool(
        inference.loc[primary_mask, "exact_patient_signflip_p_one_sided"].iloc[0] < 0.05
        and permutation_p < 0.05
    )
    return run_frame, patient_frame, inference, null


def standardize_train_test(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def fit_dual_pca(
    train: np.ndarray,
    test: np.ndarray,
    max_rank: int,
) -> LatentFit:
    mean = train.mean(axis=0)
    centered = train - mean
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    positive = np.flatnonzero(eigenvalues > 1e-10)
    rank = min(max_rank, len(positive))
    if rank < 1:
        raise RuntimeError("PCA training data have zero rank")
    singular = np.sqrt(eigenvalues[:rank])
    components = (eigenvectors[:, :rank].T @ centered) / singular[:, None]
    train_scores = centered @ components.T
    test_scores = (test - mean) @ components.T
    return LatentFit(train_scores, test_scores, components, mean)


def fit_ridge_scores(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def reconstruct(scores: np.ndarray, fit: LatentFit, rank: int) -> np.ndarray:
    return fit.mean + scores @ fit.components[:rank]


def cosine_matrix(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return normalize_rows(predicted) @ normalize_rows(actual).T


def mean_true_cosine(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.sum(normalize_rows(predicted) * normalize_rows(actual), axis=1)))


def role_two_by_two(
    predicted: np.ndarray,
    actual: np.ndarray,
    test_meta: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ("agent", "patient"):
        indices = np.flatnonzero(test_meta["thematic_role"].astype(str).to_numpy() == role)
        if len(indices) != 2:
            raise RuntimeError("Outer test fold must contain two nouns per role")
        scores = cosine_matrix(predicted[indices], actual[indices])
        diagonal = float(scores[0, 0] + scores[1, 1])
        off_diagonal = float(scores[0, 1] + scores[1, 0])
        if diagonal > off_diagonal + 1e-12:
            correct = 1.0
        elif diagonal < off_diagonal - 1e-12:
            correct = 0.0
        else:
            correct = 0.5
        rows.append(
            {
                "thematic_role": role,
                "two_by_two_correct": correct,
                "cosine_assignment_margin": 0.5 * (diagonal - off_diagonal),
                "mean_true_cosine": float(np.mean(np.diag(scores))),
            }
        )
    return rows


def absolute_cell_two_by_two(
    predicted: np.ndarray,
    actual: np.ndarray,
    test_meta: pd.DataFrame,
) -> list[dict[str, object]]:
    """Two-vs-two retrieval within each fixed role-by-voice test cell."""
    rows: list[dict[str, object]] = []
    for role in ("agent", "patient"):
        for voice in ("active", "passive"):
            indices = np.flatnonzero(
                test_meta["thematic_role"].astype(str).eq(role).to_numpy()
                & test_meta["voice"].astype(str).eq(voice).to_numpy()
            )
            if len(indices) != 2:
                raise RuntimeError("Absolute-state test cell must contain two nouns")
            scores = cosine_matrix(predicted[indices], actual[indices])
            diagonal = float(scores[0, 0] + scores[1, 1])
            off_diagonal = float(scores[0, 1] + scores[1, 0])
            if diagonal > off_diagonal + 1e-12:
                correct = 1.0
            elif diagonal < off_diagonal - 1e-12:
                correct = 0.0
            else:
                correct = 0.5
            rows.append(
                {
                    "thematic_role": role,
                    "voice": voice,
                    "two_by_two_correct": correct,
                    "cosine_assignment_margin": 0.5 * (diagonal - off_diagonal),
                    "mean_true_cosine": float(np.mean(np.diag(scores))),
                }
            )
    return rows


def latent_for_split(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    max_x_rank: int,
    max_y_rank: int,
) -> tuple[LatentFit, LatentFit]:
    x_train, x_test = standardize_train_test(x[train_indices], x[test_indices])
    x_fit = fit_dual_pca(x_train, x_test, max_x_rank)
    y_fit = fit_dual_pca(y[train_indices], y[test_indices], max_y_rank)
    return x_fit, y_fit


def predict_latent(
    x_fit: LatentFit,
    y_fit: LatentFit,
    *,
    x_rank: int,
    y_rank: int,
    alpha: float,
) -> np.ndarray:
    weights = fit_ridge_scores(
        x_fit.train_scores[:, :x_rank],
        y_fit.train_scores[:, :y_rank],
        alpha,
    )
    predicted_scores = x_fit.test_scores[:, :x_rank] @ weights
    return reconstruct(predicted_scores, y_fit, y_rank)


def choose_hyperparameters(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    outer_train_pairs: np.ndarray,
) -> tuple[int, int, float, float]:
    scores: dict[tuple[int, int, float], list[float]] = {
        (x_rank, y_rank, alpha): []
        for x_rank in NEURAL_RANKS
        for y_rank in KIMI_RANKS
        for alpha in RIDGE_ALPHAS
    }
    for validation_pair in outer_train_pairs:
        train_pairs = outer_train_pairs[outer_train_pairs != validation_pair]
        train_indices = np.flatnonzero(meta["pair_id"].isin(train_pairs).to_numpy())
        validation_indices = np.flatnonzero(meta["pair_id"].eq(validation_pair).to_numpy())
        x_fit, y_fit = latent_for_split(
            x,
            y,
            train_indices,
            validation_indices,
            max(NEURAL_RANKS),
            max(KIMI_RANKS),
        )
        actual = y[validation_indices]
        for parameters in scores:
            x_rank, y_rank, alpha = parameters
            predicted = predict_latent(
                x_fit,
                y_fit,
                x_rank=x_rank,
                y_rank=y_rank,
                alpha=alpha,
            )
            scores[parameters].append(mean_true_cosine(predicted, actual))
    means = {parameters: float(np.mean(values)) for parameters, values in scores.items()}
    # Deterministic tie-breaking favors the simpler latent model, then stronger
    # shrinkage.  All choices use training data only.
    best = max(
        means,
        key=lambda parameters: (
            means[parameters],
            -(parameters[0] + parameters[1]),
            parameters[2],
        ),
    )
    return best[0], best[1], best[2], means[best]


def nested_mapping_run(
    *,
    patient: str,
    run: str,
    x: np.ndarray,
    target: TargetSet,
) -> pd.DataFrame:
    meta = target.noun_meta.reset_index(drop=True)
    y = target.shift_states
    pair_ids = np.sort(meta["pair_id"].unique())
    rows: list[dict[str, object]] = []
    for fold_index, held_pairs_tuple in enumerate(combinations(pair_ids, 2), start=1):
        held_pairs = np.asarray(held_pairs_tuple, dtype=int)
        outer_train_pairs = pair_ids[~np.isin(pair_ids, held_pairs)]
        train_indices = np.flatnonzero(meta["pair_id"].isin(outer_train_pairs).to_numpy())
        test_indices = np.flatnonzero(meta["pair_id"].isin(held_pairs).to_numpy())
        x_rank, y_rank, alpha, inner_cosine = choose_hyperparameters(
            x,
            y,
            meta,
            outer_train_pairs,
        )
        x_fit, y_fit = latent_for_split(
            x,
            y,
            train_indices,
            test_indices,
            max(NEURAL_RANKS),
            max(KIMI_RANKS),
        )
        predicted = predict_latent(
            x_fit,
            y_fit,
            x_rank=x_rank,
            y_rank=y_rank,
            alpha=alpha,
        )
        test_meta = meta.iloc[test_indices].reset_index(drop=True)
        actual = y[test_indices]
        for role_result in role_two_by_two(predicted, actual, test_meta):
            rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "target": target.name,
                    "outer_fold": fold_index,
                    "held_pair_1": int(held_pairs[0]),
                    "held_pair_2": int(held_pairs[1]),
                    "neural_pca_rank": x_rank,
                    "kimi_pca_rank": y_rank,
                    "ridge_alpha": alpha,
                    "inner_mean_true_cosine": inner_cosine,
                    **role_result,
                }
            )
    return pd.DataFrame(rows)


def nested_absolute_mapping_run(
    *,
    patient: str,
    run: str,
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Exploratory absolute-state mapping with role and voice matched at test."""
    meta = meta.reset_index(drop=True)
    pair_ids = np.sort(meta["pair_id"].unique())
    rows: list[dict[str, object]] = []
    for fold_index, held_pairs_tuple in enumerate(combinations(pair_ids, 2), start=1):
        held_pairs = np.asarray(held_pairs_tuple, dtype=int)
        outer_train_pairs = pair_ids[~np.isin(pair_ids, held_pairs)]
        train_indices = np.flatnonzero(meta["pair_id"].isin(outer_train_pairs).to_numpy())
        test_indices = np.flatnonzero(meta["pair_id"].isin(held_pairs).to_numpy())
        x_rank, y_rank, alpha, inner_cosine = choose_hyperparameters(
            x,
            y,
            meta,
            outer_train_pairs,
        )
        x_fit, y_fit = latent_for_split(
            x,
            y,
            train_indices,
            test_indices,
            max(NEURAL_RANKS),
            max(KIMI_RANKS),
        )
        predicted = predict_latent(
            x_fit,
            y_fit,
            x_rank=x_rank,
            y_rank=y_rank,
            alpha=alpha,
        )
        test_meta = meta.iloc[test_indices].reset_index(drop=True)
        actual = y[test_indices]
        for result in absolute_cell_two_by_two(predicted, actual, test_meta):
            rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "target": "full_noun_absolute_state_exploratory",
                    "outer_fold": fold_index,
                    "held_pair_1": int(held_pairs[0]),
                    "held_pair_2": int(held_pairs[1]),
                    "neural_pca_rank": x_rank,
                    "kimi_pca_rank": y_rank,
                    "ridge_alpha": alpha,
                    "inner_mean_true_cosine": inner_cosine,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def summarize_mapping(
    outer: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_scores = (
        outer.groupby(["patient", "run", "target"], as_index=False)
        .agg(
            two_by_two_accuracy=("two_by_two_correct", "mean"),
            cosine_assignment_margin=("cosine_assignment_margin", "mean"),
            mean_true_cosine=("mean_true_cosine", "mean"),
            outer_cell_decisions=("two_by_two_correct", "size"),
        )
    )
    patient_scores = (
        run_scores.groupby(["patient", "target"], as_index=False)
        .agg(
            two_by_two_accuracy=("two_by_two_accuracy", "mean"),
            cosine_assignment_margin=("cosine_assignment_margin", "mean"),
            mean_true_cosine=("mean_true_cosine", "mean"),
            run_count=("run", "nunique"),
        )
    )
    inference_rows: list[dict[str, object]] = []
    for target_name, cell in patient_scores.groupby("target", sort=False):
        values = cell.set_index("patient").loc[list(PATIENTS), "two_by_two_accuracy"].to_numpy(float)
        margins = cell.set_index("patient").loc[list(PATIENTS), "cosine_assignment_margin"].to_numpy(float)
        inference_rows.extend(
            [
                {
                    "target": target_name,
                    "metric": "within_cell_two_by_two_accuracy",
                    "chance": 0.5,
                    "patient_mean": float(values.mean()),
                    "patient_median": float(np.median(values)),
                    "exact_patient_signflip_p_one_sided": exact_signflip(values, null=0.5),
                },
                {
                    "target": target_name,
                    "metric": "cosine_assignment_margin",
                    "chance": 0.0,
                    "patient_mean": float(margins.mean()),
                    "patient_median": float(np.median(margins)),
                    "exact_patient_signflip_p_one_sided": exact_signflip(margins, null=0.0),
                },
            ]
        )
    return run_scores, patient_scores, pd.DataFrame(inference_rows)


def mapping_analysis(
    target: TargetSet,
    neural_by_run: dict[tuple[str, str], tuple[np.ndarray, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_frames: list[pd.DataFrame] = []
    for (patient, run), (observation_values, observation_meta) in neural_by_run.items():
        x = neural_shift(
            observation_values,
            observation_meta,
            target.noun_meta,
            scale_for_geometry=False,
        )
        outer_frames.append(
            nested_mapping_run(patient=patient, run=run, x=x, target=target)
        )
        print(stable_json({"stage": "mapping", "patient": patient, "run": run}), flush=True)
    outer = pd.concat(outer_frames, ignore_index=True)
    run_scores, patient_scores, inference = summarize_mapping(outer)
    inference["metric"] = inference["metric"].replace(
        {"within_cell_two_by_two_accuracy": "within_role_two_by_two_accuracy"}
    )
    return outer, run_scores, patient_scores, inference


def absolute_mapping_analysis(
    target: TargetSet,
    neural_by_run: dict[tuple[str, str], tuple[np.ndarray, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    target_lookup = {
        (str(row.noun), str(row.voice)): index
        for index, row in target.observation_meta.reset_index(drop=True).iterrows()
    }
    for (patient, run), (x, neural_meta) in neural_by_run.items():
        target_indices = np.asarray(
            [
                target_lookup[(str(row.noun), str(row.voice))]
                for row in neural_meta.itertuples(index=False)
            ],
            dtype=int,
        )
        y = target.observation_states[target_indices]
        meta = neural_meta.copy().reset_index(drop=True)
        frames.append(
            nested_absolute_mapping_run(
                patient=patient,
                run=run,
                x=x,
                y=y,
                meta=meta,
            )
        )
        print(stable_json({"stage": "absolute_mapping", "patient": patient, "run": run}), flush=True)
    outer = pd.concat(frames, ignore_index=True)
    run_scores, patient_scores, inference = summarize_mapping(outer)
    return outer, run_scores, patient_scores, inference


def nearest_centroid_role_probe(target: TargetSet) -> pd.DataFrame:
    meta = target.observation_meta.reset_index(drop=True)
    values = target.observation_states
    pairs = np.sort(meta["pair_id"].unique())
    rows: list[dict[str, object]] = []
    for held_pair in pairs:
        training = meta["pair_id"].ne(held_pair).to_numpy()
        testing = ~training
        centroids = {
            role: normalize_rows(
                values[training & meta["thematic_role"].eq(role).to_numpy()].mean(axis=0, keepdims=True)
            )[0]
            for role in ("agent", "patient")
        }
        test_values = normalize_rows(values[testing])
        test_meta = meta.loc[testing].reset_index(drop=True)
        for vector, row in zip(test_values, test_meta.itertuples(index=False)):
            scores = {role: float(vector @ centroid) for role, centroid in centroids.items()}
            prediction = max(scores, key=scores.get)
            rows.append(
                {
                    "target": target.name,
                    "held_pair": int(held_pair),
                    "voice": str(row.voice),
                    "noun": str(row.noun),
                    "true_role": str(row.thematic_role),
                    "predicted_role": prediction,
                    "correct": int(prediction == str(row.thematic_role)),
                    "interpretation_limit": "role_is_confounded_with_animacy_and_lexical_identity",
                }
            )
    return pd.DataFrame(rows)


def plot_geometry(
    targets: dict[str, TargetSet],
    neural_by_run: dict[tuple[str, str], tuple[np.ndarray, pd.DataFrame]],
    patient_geometry: pd.DataFrame,
    inference: pd.DataFrame,
    null: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    setup_matplotlib()
    target = targets[PRIMARY_READOUT]
    meta = target.noun_meta
    model_rdm = correlation_rdm(target.shift_states)
    neural_rdms_by_patient: list[np.ndarray] = []
    for patient in PATIENTS:
        matrices = []
        for (owner, _run), (values, observation_meta) in neural_by_run.items():
            if owner != patient:
                continue
            shift = neural_shift(values, observation_meta, meta, scale_for_geometry=True)
            matrices.append(correlation_rdm(shift))
        neural_rdms_by_patient.append(np.mean(matrices, axis=0))
    neural_rdm = np.mean(neural_rdms_by_patient, axis=0)
    primary = inference.loc[inference["primary"]].iloc[0]
    patient = patient_geometry.loc[
        patient_geometry["target"].eq(PRIMARY_READOUT)
        & patient_geometry["depth_readout"].eq("all_depth")
    ].set_index("patient").loc[list(PATIENTS)]

    figure = plt.figure(figsize=(13.2, 7.4), facecolor="#F7F8FA")
    grid = GridSpec(2, 4, figure=figure, width_ratios=(1.05, 1.0, 1.0, 1.1), height_ratios=(0.72, 1.0))
    axis_design = figure.add_subplot(grid[0, :])
    axis_model = figure.add_subplot(grid[1, 0])
    axis_neural = figure.add_subplot(grid[1, 1])
    axis_patient = figure.add_subplot(grid[1, 2])
    axis_null = figure.add_subplot(grid[1, 3])

    axis_design.set_axis_off()
    axis_design.text(0.01, 0.88, "A  同一名词的上下文位移", fontsize=14, fontweight="bold", transform=axis_design.transAxes)
    boxes = [
        (0.02, "主动句\n虫子｜句首｜Agent", "#DDEEFF"),
        (0.22, "被动句\n虫子｜标记后｜Agent", "#E8E1FF"),
        (0.46, "ΔKimi\n三段深度联合", "#E3F3EA"),
        (0.66, "ΔNeural\n0–800 ms 四窗联合", "#FFF0D8"),
        (0.86, "比较关系几何\n控制角色与句对", "#FBE2E5"),
    ]
    for x, label, color in boxes:
        axis_design.text(
            x,
            0.34,
            label,
            ha="center",
            va="center",
            fontsize=10,
            transform=axis_design.transAxes,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "#607080", "linewidth": 1.1},
        )
    for left, right in ((0.08, 0.18), (0.29, 0.40), (0.53, 0.60), (0.73, 0.80)):
        axis_design.annotate(
            "",
            xy=(right, 0.34),
            xytext=(left, 0.34),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#405060", "lw": 1.4},
        )
    axis_design.text(
        0.01,
        0.02,
        "Δ = passive − active；词义和语义角色不变，改变的是语态、功能词与表面位置。",
        fontsize=9.5,
        color="#3D4A58",
        transform=axis_design.transAxes,
    )

    vmax = float(max(np.quantile(model_rdm, 0.95), np.quantile(neural_rdm, 0.95)))
    for axis, matrix, title in (
        (axis_model, model_rdm, "B  Kimi 位移几何"),
        (axis_neural, neural_rdm, "C  神经位移几何"),
    ):
        image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        axis.axhline(9.5, color="white", lw=0.8)
        axis.axvline(9.5, color="white", lw=0.8)
        axis.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
        axis.set_xticks(range(20), meta["noun"], rotation=90, fontsize=6)
        axis.set_yticks(range(20), meta["noun"], fontsize=6)
        axis.tick_params(length=0)
    colorbar = figure.colorbar(image, ax=[axis_model, axis_neural], fraction=0.026, pad=0.015)
    colorbar.ax.tick_params(labelsize=7)

    values = patient["controlled_spearman_rho"].to_numpy(float)
    axis_patient.axhline(0, color="#88939F", lw=1, ls="--")
    axis_patient.plot(np.arange(5), values, "o", color="#365D8D", ms=7)
    axis_patient.plot([-0.3, 4.3], [values.mean(), values.mean()], color="#D1495B", lw=2.3)
    label_offsets = {"gz001": -0.014, "gz002": -0.014, "gz003": 0.010, "gz004": 0.010, "gz005": -0.014}
    for index, (name, value) in enumerate(zip(PATIENTS, values)):
        axis_patient.text(index, value + label_offsets[name], name, ha="center", fontsize=7)
    axis_patient.set_xlim(-0.6, 4.6)
    axis_patient.set_xticks([])
    axis_patient.set_ylabel("controlled Spearman ρ")
    axis_patient.set_title("D  患者等权端点", loc="left", fontsize=11.5, fontweight="bold")
    axis_patient.text(
        0.04,
        0.26,
        f"mean ρ = {float(primary.patient_mean_rho):.3f}\nexact sign-flip P = {float(primary.exact_patient_signflip_p_one_sided):.4f}",
        transform=axis_patient.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCD2D8"},
    )
    axis_patient.spines[["top", "right"]].set_visible(False)

    axis_null.hist(null, bins=38, color="#B8C4D1", edgecolor="white", linewidth=0.5)
    axis_null.axvline(float(primary.patient_mean_rho), color="#D1495B", lw=2.5)
    axis_null.set_title("E  句对块置换零分布", loc="left", fontsize=11.5, fontweight="bold")
    axis_null.set_xlabel("patient-equal controlled ρ")
    axis_null.set_ylabel("count")
    p_perm = float(primary.pair_block_permutation_p_one_sided)
    axis_null.text(
        0.04,
        0.96,
        f"Pperm = {p_perm:.4f}\n保留 Agent/Patient 配对结构",
        transform=axis_null.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCD2D8"},
    )
    axis_null.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "现有神经数据能否映射到开放权重 Kimi：同词跨语态的上下文位移",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.008,
        f"Primary feature set in this run: {CURRENT_UNIT_SET}; four fixed 200-ms windows; Kimi full-noun last-token states; three fixed depth bands jointly; five patients equally weighted.",
        fontsize=7.5,
        color="#576574",
    )
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.10, top=0.93, wspace=0.42, hspace=0.24)
    figure.savefig(FIGURES / "Fig_1_neural_kimi_context_shift_geometry.png", dpi=300, facecolor=figure.get_facecolor())
    figure.savefig(FIGURES / "Fig_1_neural_kimi_context_shift_geometry.pdf", facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_mapping(
    patient_scores: pd.DataFrame,
    run_scores: pd.DataFrame,
    inference: pd.DataFrame,
    outer: pd.DataFrame,
    role_probe: pd.DataFrame,
    absolute_patient_scores: pd.DataFrame,
    absolute_inference: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    setup_matplotlib()
    figure = plt.figure(figsize=(13.2, 7.4), facecolor="#F7F8FA")
    grid = GridSpec(2, 3, figure=figure, width_ratios=(1.05, 1.15, 1.0), height_ratios=(0.7, 1.0))
    axis_cv = figure.add_subplot(grid[0, :2])
    axis_message = figure.add_subplot(grid[0, 2])
    axis_patient = figure.add_subplot(grid[1, 0])
    axis_run = figure.add_subplot(grid[1, 1])
    axis_control = figure.add_subplot(grid[1, 2])

    axis_cv.set_axis_off()
    axis_cv.set_title("A  严格的外层泛化测试", loc="left", fontsize=12, fontweight="bold")
    steps = [
        (0.02, "外层留出两个句对\n4 个未见名词"),
        (0.29, "训练折内标准化\nPCA + ridge"),
        (0.56, "内层 LOPO 选择\n维度与正则化"),
        (0.83, "同角色 2-vs-2\n机会水平 50%"),
    ]
    for x, label in steps:
        axis_cv.text(
            x,
            0.42,
            label,
            transform=axis_cv.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "#6C7B8A"},
        )
    for left, right in ((0.10, 0.21), (0.37, 0.48), (0.64, 0.75)):
        axis_cv.annotate(
            "",
            xy=(right, 0.42),
            xytext=(left, 0.42),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#536579"},
        )
    axis_cv.text(
        0.02,
        0.04,
        "关键：PCA 从未看到测试名词；检索只在两个 Agent 之间或两个 Patient 之间进行，因此角色/生命性均值本身只能得到 50%。",
        transform=axis_cv.transAxes,
        fontsize=9,
        color="#3C4A57",
    )

    accuracy_row = inference.loc[inference["metric"].eq("within_role_two_by_two_accuracy")].iloc[0]
    absolute_accuracy_row = absolute_inference.loc[
        absolute_inference["metric"].eq("within_cell_two_by_two_accuracy")
    ].iloc[0]
    passed = bool(float(accuracy_row.exact_patient_signflip_p_one_sided) < 0.05)
    axis_message.set_axis_off()
    axis_message.set_title("B  预测性结论", loc="left", fontsize=12, fontweight="bold")
    axis_message.text(
        0.04,
        0.55,
        "通过患者级检验" if passed else "尚未通过患者级检验",
        fontsize=17,
        fontweight="bold",
        color="#24734D" if passed else "#B04A5A",
        transform=axis_message.transAxes,
        va="center",
    )
    axis_message.text(
        0.04,
        0.24,
        f"patient-equal accuracy = {float(accuracy_row.patient_mean):.3f}\nexact P = {float(accuracy_row.exact_patient_signflip_p_one_sided):.4f}",
        fontsize=10,
        transform=axis_message.transAxes,
        va="center",
    )

    patient = patient_scores.set_index("patient").loc[list(PATIENTS)]
    values = patient["two_by_two_accuracy"].to_numpy(float)
    axis_patient.axhline(0.5, color="#88939F", lw=1.2, ls="--", label="chance")
    axis_patient.plot(range(5), values, "o", ms=8, color="#365D8D")
    axis_patient.plot([-0.35, 4.35], [values.mean(), values.mean()], color="#D1495B", lw=2.4)
    axis_patient.set_xticks(range(5), PATIENTS, rotation=35, ha="right")
    axis_patient.set_ylim(max(0.25, values.min() - 0.08), min(0.9, values.max() + 0.08))
    axis_patient.set_ylabel("within-role 2-vs-2 accuracy")
    axis_patient.set_title("C  五位患者等权", loc="left", fontsize=11.5, fontweight="bold")
    axis_patient.spines[["top", "right"]].set_visible(False)

    patient_colors = dict(zip(PATIENTS, ["#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B"]))
    run_order = run_scores["run"].astype(str).tolist()
    run_position = {name: index for index, name in enumerate(run_order)}
    for patient_name, cell in run_scores.groupby("patient"):
        axis_run.scatter(
            [run_position[str(name)] for name in cell["run"]],
            cell["two_by_two_accuracy"],
            s=52,
            color=patient_colors[patient_name],
            label=patient_name,
            alpha=0.9,
        )
    axis_run.axhline(0.5, color="#88939F", lw=1.2, ls="--")
    axis_run.set_xticks(range(len(run_scores)), [f"R{i + 1}" for i in range(len(run_scores))])
    axis_run.set_ylabel("2-vs-2 accuracy")
    axis_run.set_title("D  8 个 run 的一致性", loc="left", fontsize=11.5, fontweight="bold")
    axis_run.legend(frameon=False, ncol=3, fontsize=7, loc="best")
    axis_run.spines[["top", "right"]].set_visible(False)

    role_accuracy = float(role_probe["correct"].mean())
    absolute_accuracy = float(absolute_accuracy_row.patient_mean)
    map_accuracy = float(accuracy_row.patient_mean)
    axis_control.bar(
        [0, 1, 2],
        [role_accuracy, absolute_accuracy, map_accuracy],
        color=["#8B5CF6", "#3B82C4", "#2E7D6B"],
        width=0.62,
    )
    axis_control.axhline(0.5, color="#88939F", lw=1.2, ls="--")
    axis_control.set_xticks(
        [0, 1, 2],
        ["Kimi role\nprobe", "absolute noun\nmap*", "grammar-shift\nmap"],
    )
    axis_control.set_ylim(0.35, 1.02)
    axis_control.set_ylabel("accuracy")
    axis_control.set_title("E  必须区分两件事", loc="left", fontsize=11.5, fontweight="bold")
    axis_control.text(0, role_accuracy + 0.025, f"{role_accuracy:.1%}", ha="center", fontsize=10)
    axis_control.text(1, absolute_accuracy + 0.025, f"{absolute_accuracy:.1%}", ha="center", fontsize=10)
    axis_control.text(2, map_accuracy + 0.025, f"{map_accuracy:.1%}", ha="center", fontsize=10)
    axis_control.spines[["top", "right"]].set_visible(False)

    selection = (
        outer.groupby(["neural_pca_rank", "kimi_pca_rank", "ridge_alpha"]).size().sort_values(ascending=False).head(1)
    )
    top_parameters = selection.index[0]
    figure.suptitle(
        "神经活动到 Kimi 的低维预测：完全留出未见句对与名词",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.008,
        f"Most frequent inner-CV choice: neural rank {top_parameters[0]}, Kimi rank {top_parameters[1]}, alpha {top_parameters[2]:g}. *Absolute noun mapping was added after the shift result and is exploratory.",
        fontsize=7.5,
        color="#576574",
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.11, top=0.92, wspace=0.34, hspace=0.32)
    figure.savefig(FIGURES / "Fig_2_nested_neural_to_kimi_mapping.png", dpi=300, facecolor=figure.get_facecolor())
    figure.savefig(FIGURES / "Fig_2_nested_neural_to_kimi_mapping.pdf", facecolor=figure.get_facecolor())
    plt.close(figure)


def write_report(
    geometry_inference: pd.DataFrame,
    mapping_inference: pd.DataFrame,
    role_probe: pd.DataFrame,
    run_scores: pd.DataFrame,
    absolute_mapping_inference: pd.DataFrame,
) -> None:
    geometry = geometry_inference.loc[geometry_inference["primary"]].iloc[0]
    accuracy = mapping_inference.loc[
        mapping_inference["metric"].eq("within_role_two_by_two_accuracy")
    ].iloc[0]
    role_accuracy = float(role_probe["correct"].mean())
    absolute_accuracy = absolute_mapping_inference.loc[
        absolute_mapping_inference["metric"].eq("within_cell_two_by_two_accuracy")
    ].iloc[0]
    geometry_pass = bool(geometry.get("passes_both_patient_and_item_0p05", False))
    mapping_pass = bool(float(accuracy.exact_patient_signflip_p_one_sided) < 0.05)
    conclusion = []
    if geometry_pass:
        conclusion.append("同词跨语态的神经位移与 Kimi 位移具有通过双重检验的关系几何对应")
    else:
        conclusion.append("关系几何尚未同时通过患者级与刺激级检验")
    if mapping_pass:
        conclusion.append("严格留出未见句对的低维预测高于机会水平")
    else:
        conclusion.append("严格留出未见句对的低维预测尚未通过患者级检验")
    text = f"""# 现有神经数据到开放权重 Kimi 的映射

## 一句话结果

{'；'.join(conclusion)}。

## 冻结的主问题

对每个名词计算 `passive − active`。例如“虫子”在两句话中都仍是 Agent，变化的是语态、功能词和表面位置。我们比较 20 个同词位移在神经空间和 Kimi 空间中的关系几何，而不是从 28 层中挑一个最好看的 layer。

- 神经：名词起始后 0–800 ms，四个预先固定的 200 ms 窗联合；`{CURRENT_UNIT_SET}` units。
- Kimi：开放权重 `mlx-community/Kimi-VL-A3B-Thinking-4bit`，完整名词末 token；early/middle/late 三个固定深度段联合。
- 推断：先把同一患者的多个 run 合并，再让 5 位患者等权。
- 主 RSA 同时控制“是否同角色”和“是否同句对”；句对块置换保留 Agent/Patient 配对结构。

## 主结果：位移几何

- 患者等权 controlled Spearman ρ = **{float(geometry.patient_mean_rho):.4f}**。
- 5 患者 exact sign-flip P = **{float(geometry.exact_patient_signflip_p_one_sided):.4f}**。
- 句对块置换 P = **{float(geometry.pair_block_permutation_p_one_sided):.4f}**。
- 双重门槛（两者均 <0.05）：**{'通过' if geometry_pass else '未通过'}**。

## 辅助结果：严格 neural→Kimi 预测

外层每次完整留出两个句对，即 4 个未见名词；PCA、维度选择和 ridge 正则化全部只在训练折内完成。测试时只在两个 Agent 之间或两个 Patient 之间做 2-vs-2，因此单纯依赖角色/生命性均值的模型只能达到 50%。

- 患者等权 2-vs-2 accuracy = **{float(accuracy.patient_mean):.4f}**（chance=0.5）。
- exact patient sign-flip P = **{float(accuracy.exact_patient_signflip_p_one_sided):.4f}**。
- 8 run 范围 = **{float(run_scores.two_by_two_accuracy.min()):.4f}–{float(run_scores.two_by_two_accuracy.max()):.4f}**。

## 看见主分析后增加的探索性检验：绝对名词状态

为了区分“没有语法位移映射”与“完全没有跨模态映射”，我们另做了绝对名词状态预测。外层仍完整留出两个句对；测试检索进一步限制在相同角色、相同语态的两个未见名词之间，因此角色或语态均值本身只能得到 50%。这一检验是在看到主位移分析结果后增加的，必须标为 exploratory。

- patient-equal 2-vs-2 accuracy = **{float(absolute_accuracy.patient_mean):.4f}**。
- exact patient sign-flip P = **{float(absolute_accuracy.exact_patient_signflip_p_one_sided):.4f}**。

## Kimi 是否含有 Agent/Patient 信息

Kimi 的留一句对最近质心 probe accuracy = **{role_accuracy:.4f}**。这说明 Kimi 表征含有可解码的 Agent/Patient-associated 信息，但由于所有 Agent 名词均有生、所有 Patient 名词均无生，它不能证明抽象、独立于词义和生命性的语义角色轴。

## 应该如何解释

1. 若几何通过而预测不通过：可以说“神经与 Kimi 对上下文变化的关系结构相似”，不能说“神经活动可逐项解码成 Kimi hidden state”。
2. 若预测也通过：可以进一步说“从神经位移学到的低维线性映射能泛化到未见句对中的同角色名词”。
3. 无论结果如何，都不能说同一个名词从 Agent 变成 Patient；本实验中语义角色保持不变。
4. Kimi role probe、neural→Kimi mapping、抽象角色可识别性是三个不同问题。

## 文件

- `figures/Fig_1_neural_kimi_context_shift_geometry.png/.pdf`
- `figures/Fig_2_nested_neural_to_kimi_mapping.png/.pdf`
- `geometry_patient_scores.csv`, `geometry_inference.csv`
- `mapping_outer_folds.csv`, `mapping_patient_scores.csv`, `mapping_inference.csv`
- `absolute_mapping_outer_folds_exploratory.csv`, `absolute_mapping_inference_exploratory.csv`
- `kimi_role_associated_probe.csv`
"""
    (OUTPUT / "README_zh.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Pair-block permutations for the primary geometry test.",
    )
    parser.add_argument(
        "--smoke-run",
        type=str,
        default=None,
        help="Run only one named run and skip patient-level inference/figures.",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="Run geometry only.",
    )
    parser.add_argument(
        "--unit-set",
        choices=("usable_lenient", "all_clusters", "good_clusters"),
        default="usable_lenient",
        help="Neural unit set. Non-primary sets are written as sensitivities.",
    )
    return parser.parse_args()


def main() -> None:
    global OUTPUT, FIGURES, CURRENT_UNIT_SET
    args = parse_args()
    CURRENT_UNIT_SET = args.unit_set
    if args.unit_set == "usable_lenient":
        OUTPUT = BASE_OUTPUT
    else:
        OUTPUT = BASE_OUTPUT / f"sensitivity_{args.unit_set}"
    FIGURES = OUTPUT / "figures"
    ensure_dirs()
    targets = load_targets()
    registry = pd.read_csv(REGISTRY_PATH)
    events = pd.read_csv(EVENTS_PATH)
    selected_runs = list(RUNS)
    if args.smoke_run:
        selected_runs = [value for value in RUNS if value[1] == args.smoke_run]
        if len(selected_runs) != 1:
            raise RuntimeError(f"Unknown or duplicated run name: {args.smoke_run}")
    neural_by_run = {
        (patient, run): grammar_neural_patterns(
            patient, run, events, registry, args.unit_set
        )
        for patient, run in selected_runs
    }
    if args.smoke_run:
        patient, run = selected_runs[0]
        values, meta = neural_by_run[(patient, run)]
        shift = neural_shift(values, meta, targets[PRIMARY_READOUT].noun_meta, scale_for_geometry=False)
        print(
            stable_json(
                {
                    "stage": "smoke",
                    "patient": patient,
                    "run": run,
                    "neural_observations": list(values.shape),
                    "neural_shift": list(shift.shape),
                    "kimi_shift": list(targets[PRIMARY_READOUT].shift_states.shape),
                }
            )
        )
        if not args.skip_mapping:
            smoke = nested_mapping_run(patient=patient, run=run, x=shift, target=targets[PRIMARY_READOUT])
            print(
                stable_json(
                    {
                        "stage": "smoke_mapping",
                        "rows": len(smoke),
                        "accuracy": float(smoke["two_by_two_correct"].mean()),
                    }
                )
            )
        return

    geometry_run, geometry_patient, geometry_inference, null = geometry_analysis(
        targets,
        neural_by_run,
        args.permutations,
    )
    geometry_run.to_csv(OUTPUT / "geometry_run_scores.csv", index=False)
    geometry_patient.to_csv(OUTPUT / "geometry_patient_scores.csv", index=False)
    geometry_inference.to_csv(OUTPUT / "geometry_inference.csv", index=False)
    np.save(OUTPUT / "primary_pair_block_permutation_null.npy", null)

    role_probe = nearest_centroid_role_probe(targets[PRIMARY_READOUT])
    role_probe.to_csv(OUTPUT / "kimi_role_associated_probe.csv", index=False)

    if args.skip_mapping:
        print(stable_json({"stage": "geometry_done", "output": str(OUTPUT)}))
        return

    outer, mapping_run, mapping_patient, mapping_inference = mapping_analysis(
        targets[PRIMARY_READOUT],
        neural_by_run,
    )
    outer.to_csv(OUTPUT / "mapping_outer_folds.csv", index=False)
    mapping_run.to_csv(OUTPUT / "mapping_run_scores.csv", index=False)
    mapping_patient.to_csv(OUTPUT / "mapping_patient_scores.csv", index=False)
    mapping_inference.to_csv(OUTPUT / "mapping_inference.csv", index=False)

    absolute_outer, absolute_run, absolute_patient, absolute_inference = absolute_mapping_analysis(
        targets[PRIMARY_READOUT],
        neural_by_run,
    )
    absolute_outer.to_csv(OUTPUT / "absolute_mapping_outer_folds_exploratory.csv", index=False)
    absolute_run.to_csv(OUTPUT / "absolute_mapping_run_scores_exploratory.csv", index=False)
    absolute_patient.to_csv(OUTPUT / "absolute_mapping_patient_scores_exploratory.csv", index=False)
    absolute_inference.to_csv(OUTPUT / "absolute_mapping_inference_exploratory.csv", index=False)

    specification = {
        "analysis_date": "2026-07-30",
        "model_id": KIMI_ID,
        "model_readout": "full noun last token; normalized fixed early/middle/late band means concatenated",
        "neural_readout": f"{args.unit_set} units; baseline-subtracted 0-200/200-400/400-600/600-800 ms windows concatenated",
        "unit_set": args.unit_set,
        "estimand": "passive_minus_active shift for the same noun with thematic role held fixed",
        "primary_metric": "role-and-pair-controlled Spearman RSA",
        "primary_inference": "five-patient exact sign flip plus sentence-pair block label permutation",
        "secondary_mapping": "nested leave-two-pairs-out PCA+ridge; within-role 2-vs-2",
        "post_result_exploratory_mapping": "absolute noun states; nested leave-two-pairs-out; within-role-and-voice 2-vs-2",
        "neural_pca_ranks": list(NEURAL_RANKS),
        "kimi_pca_ranks": list(KIMI_RANKS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "interpretation_limit": "Agent/Patient is confounded with animacy and lexical identity",
    }
    (OUTPUT / "analysis_specification.json").write_text(
        json.dumps(specification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plot_geometry(targets, neural_by_run, geometry_patient, geometry_inference, null)
    plot_mapping(
        mapping_patient,
        mapping_run,
        mapping_inference,
        outer,
        role_probe,
        absolute_patient,
        absolute_inference,
    )
    write_report(
        geometry_inference,
        mapping_inference,
        role_probe,
        mapping_run,
        absolute_inference,
    )
    print(
        stable_json(
            {
                "stage": "done",
                "output": str(OUTPUT),
                "geometry": geometry_inference.loc[geometry_inference["primary"]].iloc[0].to_dict(),
                "mapping": mapping_inference.to_dict(orient="records"),
                "absolute_mapping_exploratory": absolute_inference.to_dict(orient="records"),
                "kimi_role_probe_accuracy": float(role_probe["correct"].mean()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
