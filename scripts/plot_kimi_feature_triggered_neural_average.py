#!/usr/bin/env python3
"""Create a Kimi-feature-triggered neural average (KFTA) figure.

The figure is an explanatory follow-up to the observed Kimi surface-position
alignment map.  It fixes Kimi hidden-state level 1 and a 300--500 ms neural
weighting window from the previously observed peak, so it is explicitly
post-selection descriptive and receives no new inferential p value.

Within that constraint, noun items are cross-fitted: for each held-out
semantic pair, the Kimi surface-position axis and neural population weights
are estimated using the other nine pairs, then applied to the four unseen
noun events in the held-out pair.  Runs are summarized within patient before
the five patients are equally weighted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

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
    REGISTRY_PATH,
    RUNS,
    run_npz_path,
)
from scripts.analyze_noun_role_structure_encoding import (  # noqa: E402
    PATIENTS,
    setup_matplotlib,
)


REPORT = ROOT / "reports/existing_neural_to_kimi_lowdim_mapping_20260730"
FIGURES = REPORT / "figures"
OUTPUT = REPORT / "kimi_feature_triggered_average"
KIMI_CACHE = (
    ROOT
    / "reports/multirun_grouped_go_20260713/llm"
    / "exploratory_kimi_vl_a3b_4bit_20260723"
    / "representation_cache/representations.npz"
)

KIMI_LEVEL = 1
UNIT_SET = "all_clusters"
PRE_MS = 200
POST_MS = 800
BIN_MS = 10
TIMES_MS = np.arange(-PRE_MS + BIN_MS / 2, POST_MS, BIN_MS)
WEIGHT_WINDOW_MS = (300, 500)
SEED = 20260730


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms > 1e-12, norms, 1.0)


def load_kimi_and_meta() -> tuple[pd.DataFrame, np.ndarray]:
    registry = pd.read_csv(REGISTRY_PATH)
    meta = registry.loc[
        registry["condition"].isin(["active", "passive"]),
        [
            "observation_id",
            "sentence_id",
            "pair_id",
            "condition",
            "noun",
            "thematic_role",
            "surface_position",
        ],
    ].copy()
    meta = meta.rename(columns={"condition": "voice"}).reset_index(drop=True)

    with np.load(KIMI_CACHE) as archive:
        noun_rows = np.asarray(archive["constituent_ids"]).astype(str) != "verb"
        cache_meta = pd.DataFrame(
            {
                "noun": np.asarray(archive["target_text"]).astype(str)[noun_rows],
                "voice": np.asarray(archive["voices"]).astype(str)[noun_rows],
                "pair_id": np.asarray(archive["sentence_pair_ids"]).astype(int)[noun_rows],
            }
        )
        hidden = np.asarray(archive["hidden_last"], dtype=np.float64)[
            noun_rows, KIMI_LEVEL
        ]
    hidden = normalize_rows(hidden)
    lookup = {
        (str(row.noun), str(row.voice)): index
        for index, row in cache_meta.iterrows()
    }
    indices = np.asarray(
        [
            lookup[(str(row.noun), str(row.voice))]
            for row in meta.itertuples(index=False)
        ],
        dtype=int,
    )
    aligned = hidden[indices]
    if aligned.shape != (40, 2048):
        raise RuntimeError(f"Unexpected aligned Kimi shape: {aligned.shape}")
    return meta, aligned


def load_all_kimi_levels(meta: pd.DataFrame) -> np.ndarray:
    with np.load(KIMI_CACHE) as archive:
        noun_rows = np.asarray(archive["constituent_ids"]).astype(str) != "verb"
        cache_meta = pd.DataFrame(
            {
                "noun": np.asarray(archive["target_text"]).astype(str)[noun_rows],
                "voice": np.asarray(archive["voices"]).astype(str)[noun_rows],
            }
        )
        hidden = np.asarray(archive["hidden_last"], dtype=np.float64)[noun_rows]
    lookup = {
        (str(row.noun), str(row.voice)): index
        for index, row in cache_meta.iterrows()
    }
    indices = np.asarray(
        [lookup[(str(row.noun), str(row.voice))] for row in meta.itertuples(index=False)],
        dtype=int,
    )
    aligned = hidden[indices]
    if aligned.shape[:2] != (40, 28):
        raise RuntimeError(f"Unexpected all-level Kimi shape: {aligned.shape}")
    return aligned


def center_within_noun(scores: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    centered = np.asarray(scores, dtype=np.float64).copy()
    for noun, indices in meta.groupby("noun").groups.items():
        noun_indices = np.asarray(list(indices), dtype=int)
        if len(noun_indices) != 2:
            raise RuntimeError(f"Expected two positions for noun {noun}")
        centered[noun_indices] -= centered[noun_indices].mean()
    return centered


def build_crossfit_kimi_scores(
    meta: pd.DataFrame,
    hidden: np.ndarray,
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    pair_ids = np.sort(meta["pair_id"].unique())
    heldout_scores = np.full(len(meta), np.nan, dtype=np.float64)
    fold_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for held_pair in pair_ids:
        training = meta["pair_id"].ne(held_pair).to_numpy()
        testing = ~training
        post = training & meta["surface_position"].eq("post_marker").to_numpy()
        initial = training & meta["surface_position"].eq("initial").to_numpy()
        axis = hidden[post].mean(axis=0) - hidden[initial].mean(axis=0)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            raise RuntimeError(f"Kimi position axis is zero in fold {held_pair}")
        axis /= axis_norm
        raw = hidden @ axis
        train_meta = meta.loc[training].reset_index(drop=True)
        test_meta = meta.loc[testing].reset_index(drop=True)
        train_scores = center_within_noun(raw[training], train_meta)
        test_scores = center_within_noun(raw[testing], test_meta)
        scale = float(train_scores.std())
        scale = scale if scale > 1e-8 else 1.0
        train_scores /= scale
        test_scores /= scale
        train_indices = np.flatnonzero(training)
        test_indices = np.flatnonzero(testing)
        heldout_scores[test_indices] = test_scores
        fold_scores[int(held_pair)] = (train_indices, test_indices, np.concatenate([train_scores, test_scores]))
    if not np.isfinite(heldout_scores).all():
        raise RuntimeError("Cross-fitted Kimi scores contain non-finite values")

    # Store train/test scores separately but in a compact fold tuple.  The
    # concatenated vector is split by train length when used below.
    return heldout_scores, fold_scores


def load_run_neural(
    patient: str,
    run: str,
    meta: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(run_npz_path(patient, run, UNIT_SET), allow_pickle=True) as archive:
        firing = np.asarray(archive["firing_rate_hz"], dtype=np.float32)
        unit_ids = np.asarray(archive["unit_ids"])
    run_events = events.loc[
        events["patient"].eq(patient) & events["run"].eq(run)
    ].set_index("observation_id")
    bin_count = int((PRE_MS + POST_MS) / BIN_MS)
    values = np.empty((len(meta), firing.shape[0], bin_count), dtype=np.float32)
    for observation_index, observation_id in enumerate(meta["observation_id"].astype(str)):
        row = run_events.loc[observation_id]
        event_bin = int(row.event_bin_1000Hz)
        segment = firing[:, event_bin - PRE_MS : event_bin + POST_MS]
        if segment.shape[1] != PRE_MS + POST_MS:
            raise RuntimeError(f"Epoch out of bounds: {patient}/{run}/{observation_id}")
        binned = segment.reshape(firing.shape[0], bin_count, BIN_MS).mean(axis=2)
        baseline = binned[:, : PRE_MS // BIN_MS].mean(axis=1, keepdims=True)
        values[observation_index] = binned - baseline
    return values.astype(np.float64), unit_ids


def smooth_time(values: np.ndarray, sigma_bins: float = 1.2) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(values, sigma=sigma_bins, axis=-1, mode="nearest")


def compute_run_kfta(
    neural: np.ndarray,
    meta: pd.DataFrame,
    hidden: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    pair_ids = np.sort(meta["pair_id"].unique())
    fold_run_curves: list[np.ndarray] = []
    unit_curve_sum = np.zeros((neural.shape[1], neural.shape[2]), dtype=np.float64)
    unit_curve_count = np.zeros(neural.shape[1], dtype=np.int64)
    unit_importance_sum = np.zeros(neural.shape[1], dtype=np.float64)
    unit_importance_count = np.zeros(neural.shape[1], dtype=np.int64)
    fold_rows: list[dict[str, object]] = []
    item_rows: list[dict[str, object]] = []
    response_mask = (TIMES_MS >= 0) & (TIMES_MS < POST_MS)
    weight_mask = (TIMES_MS >= WEIGHT_WINDOW_MS[0]) & (TIMES_MS < WEIGHT_WINDOW_MS[1])

    for held_pair in pair_ids:
        training = meta["pair_id"].ne(held_pair).to_numpy()
        testing = ~training

        post = training & meta["surface_position"].eq("post_marker").to_numpy()
        initial = training & meta["surface_position"].eq("initial").to_numpy()
        axis = hidden[post].mean(axis=0) - hidden[initial].mean(axis=0)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        raw_scores = hidden @ axis
        train_meta = meta.loc[training].reset_index(drop=True)
        test_meta = meta.loc[testing].reset_index(drop=True)
        train_scores = center_within_noun(raw_scores[training], train_meta)
        test_scores = center_within_noun(raw_scores[testing], test_meta)
        score_scale = float(train_scores.std())
        score_scale = score_scale if score_scale > 1e-8 else 1.0
        train_scores /= score_scale
        test_scores /= score_scale

        train_neural = neural[training]
        test_neural = neural[testing]
        unit_scale = train_neural[:, :, response_mask].std(axis=(0, 2))
        valid_units = unit_scale > 1e-8
        unit_scale = np.where(valid_units, unit_scale, 1.0)
        train_neural = train_neural / unit_scale[None, :, None]
        test_neural = test_neural / unit_scale[None, :, None]

        train_summary = train_neural[:, :, weight_mask].mean(axis=2)
        train_denominator = float(train_scores @ train_scores)
        test_denominator = float(test_scores @ test_scores)
        beta_train = (train_scores @ train_summary) / max(train_denominator, 1e-12)
        beta_test = np.tensordot(test_scores, test_neural, axes=(0, 0)) / max(
            test_denominator, 1e-12
        )
        signs = np.where(beta_train >= 0, 1.0, -1.0)
        aligned = signs[:, None] * beta_test
        aligned[~valid_units] = np.nan
        weights = np.abs(beta_train)
        weights[~valid_units] = 0.0
        if float(weights.sum()) <= 1e-12:
            raise RuntimeError(f"No weighted units in fold {held_pair}")
        fold_curve = np.nansum(aligned * weights[:, None], axis=0) / weights.sum()
        item_projection = np.tensordot(
            test_neural,
            np.where(valid_units, beta_train, 0.0),
            axes=(1, 0),
        ) / weights.sum()
        fold_run_curves.append(fold_curve)
        finite_units = np.isfinite(aligned).all(axis=1)
        unit_curve_sum[finite_units] += aligned[finite_units]
        unit_curve_count[finite_units] += 1
        unit_importance_sum[valid_units] += np.abs(beta_train[valid_units])
        unit_importance_count[valid_units] += 1
        fold_rows.append(
            {
                "held_pair": int(held_pair),
                "valid_units": int(valid_units.sum()),
                "mean_abs_training_weight": float(np.mean(np.abs(beta_train[valid_units]))),
                "heldout_kimi_score_post_minus_initial": float(
                    test_scores[test_meta["surface_position"].eq("post_marker")].mean()
                    - test_scores[test_meta["surface_position"].eq("initial")].mean()
                ),
                "heldout_kfta_300_500": float(fold_curve[weight_mask].mean()),
            }
        )
        for local_index, source in enumerate(test_meta.itertuples(index=False)):
            item_rows.append(
                {
                    "held_pair": int(held_pair),
                    "observation_id": str(source.observation_id),
                    "noun": str(source.noun),
                    "voice": str(source.voice),
                    "surface_position": str(source.surface_position),
                    "thematic_role": str(source.thematic_role),
                    "kimi_position_score": float(test_scores[local_index]),
                    "neural_projection_300_500": float(
                        item_projection[local_index, weight_mask].mean()
                    ),
                }
            )

    run_curve = np.mean(np.stack(fold_run_curves), axis=0)
    unit_curves = unit_curve_sum / np.where(unit_curve_count[:, None] > 0, unit_curve_count[:, None], 1)
    unit_curves[unit_curve_count == 0] = np.nan
    unit_importance = unit_importance_sum / np.where(
        unit_importance_count > 0, unit_importance_count, 1
    )
    unit_importance[unit_importance_count == 0] = np.nan
    return run_curve, unit_curves, unit_importance, fold_rows, item_rows


def compute_kimi_layer_profile(meta: pd.DataFrame) -> pd.DataFrame:
    all_levels = load_all_kimi_levels(meta)
    rows: list[dict[str, object]] = []
    for level in range(all_levels.shape[1]):
        hidden = normalize_rows(all_levels[:, level])
        try:
            scores, _ = build_crossfit_kimi_scores(meta, hidden)
        except RuntimeError as error:
            # A complete leave-one-pair-out profile requires a non-zero
            # position axis in every fold.  Mark a layer as non-estimable
            # instead of silently inserting an arbitrary score.
            rows.append(
                {
                    "kimi_level": level,
                    "heldout_post_minus_initial_mean": np.nan,
                    "heldout_post_minus_initial_median": np.nan,
                    "noun_positive_fraction": np.nan,
                    "valid_crossfit_layer": False,
                    "note": str(error),
                }
            )
            continue
        score_frame = meta[["noun", "surface_position"]].copy()
        score_frame["score"] = scores
        differences = []
        for noun, cell in score_frame.groupby("noun", sort=False):
            lookup = cell.set_index("surface_position")["score"]
            differences.append(float(lookup["post_marker"] - lookup["initial"]))
        differences = np.asarray(differences, dtype=float)
        rows.append(
            {
                "kimi_level": level,
                "heldout_post_minus_initial_mean": float(differences.mean()),
                "heldout_post_minus_initial_median": float(np.median(differences)),
                "noun_positive_fraction": float(np.mean(differences > 0)),
                "valid_crossfit_layer": True,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(values.mean())
    null = []
    for mask in range(1 << len(values)):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(len(values))]
        )
        null.append(float(np.mean(values * signs)))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def item_comparison_correlations(
    frame: pd.DataFrame,
    neural_column: str,
) -> dict[str, float]:
    """Return overall and position-controlled item-level correspondences."""
    from scipy.stats import spearmanr

    x_values = frame["kimi_position_score"].astype(float)
    y_values = frame[neural_column].astype(float)
    x_residual = x_values - frame.groupby("surface_position")[
        "kimi_position_score"
    ].transform("mean")
    y_residual = y_values - frame.groupby("surface_position")[neural_column].transform(
        "mean"
    )
    paired = frame.pivot(
        index=["noun", "thematic_role"],
        columns="surface_position",
        values=["kimi_position_score", neural_column],
    )
    delta_x = (
        paired[("kimi_position_score", "post_marker")]
        - paired[("kimi_position_score", "initial")]
    )
    delta_y = (
        paired[(neural_column, "post_marker")]
        - paired[(neural_column, "initial")]
    )
    return {
        "spearman_rho": float(spearmanr(x_values, y_values).statistic),
        "position_controlled_spearman_rho": float(
            spearmanr(x_residual, y_residual).statistic
        ),
        "paired_delta_spearman_rho": float(spearmanr(delta_x, delta_y).statistic),
    }


def analyze() -> dict[str, object]:
    ensure_dirs()
    meta, hidden = load_kimi_and_meta()
    crossfit_scores, _fold_scores = build_crossfit_kimi_scores(meta, hidden)
    meta = meta.copy()
    meta["crossfit_kimi_position_score"] = crossfit_scores

    events = pd.read_csv(EVENTS_PATH)
    run_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    item_rows: list[dict[str, object]] = []
    unit_frames: list[pd.DataFrame] = []
    run_curves: dict[tuple[str, str], np.ndarray] = {}
    for patient, run in RUNS:
        neural, unit_ids = load_run_neural(patient, run, meta, events)
        run_curve, unit_curves, unit_importance, folds, items = compute_run_kfta(
            neural, meta, hidden
        )
        run_curves[(patient, run)] = run_curve
        for time_ms, value in zip(TIMES_MS, run_curve):
            run_rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "time_ms": float(time_ms),
                    "kfta": float(value),
                }
            )
        for row in folds:
            fold_rows.append({"patient": patient, "run": run, **row})
        for row in items:
            item_rows.append({"patient": patient, "run": run, **row})
        unit_frame = pd.DataFrame(unit_curves, columns=[f"t_{int(value)}" for value in TIMES_MS])
        unit_frame.insert(0, "training_importance", unit_importance)
        unit_frame.insert(0, "valid_kfta_unit", np.isfinite(unit_importance))
        unit_frame.insert(0, "unit_id", unit_ids.astype(str))
        unit_frame.insert(0, "run", run)
        unit_frame.insert(0, "patient", patient)
        unit_frames.append(unit_frame)
        print(
            json.dumps(
                {
                    "stage": "run",
                    "patient": patient,
                    "run": run,
                    "units": int(neural.shape[1]),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    patient_rows: list[dict[str, object]] = []
    patient_curves: dict[str, np.ndarray] = {}
    for patient in PATIENTS:
        curves = [curve for (owner, _run), curve in run_curves.items() if owner == patient]
        patient_curve = np.mean(np.stack(curves), axis=0)
        patient_curves[patient] = patient_curve
        for time_ms, value in zip(TIMES_MS, patient_curve):
            patient_rows.append(
                {
                    "patient": patient,
                    "time_ms": float(time_ms),
                    "kfta": float(value),
                }
            )
    group_curve = np.mean(np.stack(list(patient_curves.values())), axis=0)
    weight_mask = (TIMES_MS >= WEIGHT_WINDOW_MS[0]) & (TIMES_MS < WEIGHT_WINDOW_MS[1])
    endpoints = np.asarray(
        [patient_curves[patient][weight_mask].mean() for patient in PATIENTS],
        dtype=float,
    )

    meta.to_csv(OUTPUT / "crossfit_kimi_position_scores.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUTPUT / "run_kfta_timecourses.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(OUTPUT / "patient_kfta_timecourses.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "fold_audit.csv", index=False)
    unit_source = pd.concat(unit_frames, ignore_index=True)
    unit_source.to_csv(OUTPUT / "unit_crossfit_aligned_kfta.csv.gz", index=False, compression="gzip")
    endpoint_frame = pd.DataFrame(
        {
            "patient": list(PATIENTS),
            "kfta_300_500": endpoints,
            "post_selection_descriptive": True,
        }
    )
    endpoint_frame.to_csv(OUTPUT / "patient_endpoints.csv", index=False)

    item_frame = pd.DataFrame(item_rows)
    item_frame.to_csv(OUTPUT / "heldout_item_activation_neural_projection.csv", index=False)
    layer_profile = compute_kimi_layer_profile(meta)
    layer_profile.to_csv(OUTPUT / "kimi_activation_layer_profile.csv", index=False)

    patient_item = (
        item_frame.groupby(
            [
                "patient",
                "observation_id",
                "noun",
                "voice",
                "surface_position",
                "thematic_role",
            ],
            as_index=False,
        )
        .agg(
            kimi_position_score=("kimi_position_score", "mean"),
            neural_projection_300_500=("neural_projection_300_500", "mean"),
        )
    )
    patient_item["neural_projection_z"] = patient_item.groupby("patient")[
        "neural_projection_300_500"
    ].transform(
        lambda values: (values - values.mean())
        / (values.std(ddof=0) if values.std(ddof=0) > 1e-12 else 1.0)
    )
    patient_item.to_csv(OUTPUT / "patient_item_activation_neural_projection.csv", index=False)
    group_item = (
        patient_item.groupby(
            ["observation_id", "noun", "voice", "surface_position", "thematic_role"],
            as_index=False,
        )
        .agg(
            kimi_position_score=("kimi_position_score", "mean"),
            patient_equal_neural_projection_z=("neural_projection_z", "mean"),
        )
    )
    group_item.to_csv(OUTPUT / "patient_equal_item_comparison.csv", index=False)

    summary = {
        "analysis_role": "post_selection_descriptive_visualization",
        "kimi_level": KIMI_LEVEL,
        "unit_set": UNIT_SET,
        "run_count": len(RUNS),
        "patient_count": len(PATIENTS),
        "run_unit_count": int(len(unit_source)),
        "valid_kfta_run_unit_count": int(unit_source["valid_kfta_unit"].sum()),
        "stimulus_crossfit": "leave_one_semantic_pair_out",
        "neural_weight_window_ms": list(WEIGHT_WINDOW_MS),
        "patient_equal_endpoint_mean": float(endpoints.mean()),
        "patient_equal_endpoint_median": float(np.median(endpoints)),
        "patient_endpoint_min": float(endpoints.min()),
        "patient_endpoint_max": float(endpoints.max()),
        "exact_signflip_p_descriptive_only": exact_signflip_p(endpoints),
        "group_curve_peak_time_ms": float(TIMES_MS[int(np.argmax(group_curve))]),
        "group_curve_peak": float(np.max(group_curve)),
        "inference_warning": (
            "Kimi level 1 and the 300-500 ms neural weighting window were chosen "
            "after observing the existing layer-time map; no new independent "
            "significance claim is permitted."
        ),
    }
    patient_correlations = []
    for patient, cell in patient_item.groupby("patient", sort=False):
        patient_correlations.append(
            {
                "patient": patient,
                **item_comparison_correlations(cell, "neural_projection_z"),
            }
        )
    correlation_frame = pd.DataFrame(patient_correlations)
    correlation_frame.to_csv(OUTPUT / "patient_item_correlations.csv", index=False)
    group_correlations = item_comparison_correlations(
        group_item,
        "patient_equal_neural_projection_z",
    )
    summary["patient_item_spearman_mean"] = float(correlation_frame["spearman_rho"].mean())
    summary["patient_item_spearman_range"] = [
        float(correlation_frame["spearman_rho"].min()),
        float(correlation_frame["spearman_rho"].max()),
    ]
    summary["patient_position_controlled_spearman_mean"] = float(
        correlation_frame["position_controlled_spearman_rho"].mean()
    )
    summary["patient_position_controlled_spearman_range"] = [
        float(correlation_frame["position_controlled_spearman_rho"].min()),
        float(correlation_frame["position_controlled_spearman_rho"].max()),
    ]
    summary["patient_paired_delta_spearman_mean"] = float(
        correlation_frame["paired_delta_spearman_rho"].mean()
    )
    summary["patient_equal_item_spearman_rho"] = group_correlations["spearman_rho"]
    summary["patient_equal_position_controlled_spearman_rho"] = group_correlations[
        "position_controlled_spearman_rho"
    ]
    summary["patient_equal_paired_delta_spearman_rho"] = group_correlations[
        "paired_delta_spearman_rho"
    ]
    summary["valid_kimi_profile_level_count"] = int(
        layer_profile["valid_crossfit_layer"].sum()
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "meta": meta,
        "patient_curves": patient_curves,
        "group_curve": group_curve,
        "endpoints": endpoints,
        "unit_source": unit_source,
        "summary": summary,
        "layer_profile": layer_profile,
        "patient_item": patient_item,
        "group_item": group_item,
        "patient_correlations": correlation_frame,
    }


def plot(results: dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.colors import TwoSlopeNorm

    setup_matplotlib()
    meta = results["meta"]
    patient_curves = results["patient_curves"]
    group_curve = np.asarray(results["group_curve"], dtype=float)
    endpoints = np.asarray(results["endpoints"], dtype=float)
    unit_source = results["unit_source"]
    summary = results["summary"]

    valid_source = unit_source.loc[unit_source["valid_kfta_unit"].astype(bool)].copy()
    time_columns = [column for column in valid_source.columns if column.startswith("t_")]
    unit_matrix = valid_source[time_columns].to_numpy(float)
    unit_matrix = smooth_time(unit_matrix)
    order = np.argsort(valid_source["training_importance"].to_numpy(float))[::-1]
    unit_matrix = unit_matrix[order]
    finite = unit_matrix[np.isfinite(unit_matrix)]
    limit = float(np.quantile(np.abs(finite), 0.985)) if finite.size else 1.0
    limit = max(limit, 1e-6)

    role_colors = {"agent": "#2B6CB0", "patient": "#D97732"}
    patient_colors = dict(
        zip(PATIENTS, ["#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B"])
    )

    figure = plt.figure(figsize=(14.2, 8.2), facecolor="#F7F8FA")
    grid = GridSpec(
        2,
        12,
        figure=figure,
        height_ratios=(0.92, 1.08),
        hspace=0.42,
        wspace=0.8,
    )
    axis_kimi = figure.add_subplot(grid[0, :4])
    axis_heat = figure.add_subplot(grid[0, 4:])
    axis_curve = figure.add_subplot(grid[1, :8])
    axis_patient = figure.add_subplot(grid[1, 8:])

    # A: cross-fitted Kimi position scores, same noun connected.
    for noun, cell in meta.groupby("noun", sort=False):
        cell = cell.sort_values("surface_position", key=lambda s: s.map({"initial": 0, "post_marker": 1}))
        if list(cell["surface_position"]) != ["initial", "post_marker"]:
            raise RuntimeError(f"Unexpected positions for {noun}")
        role = str(cell["thematic_role"].iloc[0])
        values = cell["crossfit_kimi_position_score"].to_numpy(float)
        axis_kimi.plot([0, 1], values, color=role_colors[role], alpha=0.38, lw=1.0)
        axis_kimi.scatter([0, 1], values, color=role_colors[role], s=20, alpha=0.8, zorder=3)
    axis_kimi.axhline(0, color="#8B96A3", lw=0.9, ls="--")
    axis_kimi.set_xticks([0, 1], ["句首", "标记词后"])
    axis_kimi.set_ylabel("cross-fitted Kimi position score")
    axis_kimi.set_title("A  Kimi Layer 1 的表面位置轴", loc="left", fontweight="bold", fontsize=12)
    axis_kimi.legend(
        handles=[
            Line2D([0], [0], color=role_colors["agent"], marker="o", lw=1, label="Agent nouns"),
            Line2D([0], [0], color=role_colors["patient"], marker="o", lw=1, label="Patient nouns"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    axis_kimi.spines[["top", "right"]].set_visible(False)

    # B: all-unit cross-fitted, sign-aligned KFTA heatmap.
    image = axis_heat.imshow(
        unit_matrix,
        aspect="auto",
        interpolation="nearest",
        extent=[TIMES_MS[0] - BIN_MS / 2, TIMES_MS[-1] + BIN_MS / 2, len(unit_matrix), 0],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        rasterized=True,
    )
    axis_heat.axvline(0, color="#242B33", lw=1.0)
    axis_heat.axvspan(
        WEIGHT_WINDOW_MS[0],
        WEIGHT_WINDOW_MS[1],
        color="#F4C95D",
        alpha=0.14,
        lw=0,
    )
    axis_heat.axvline(385, color="#E0A800", lw=1.0, ls="--")
    axis_heat.set_xlabel("名词起始后的时间 (ms)")
    axis_heat.set_ylabel(f"{len(unit_matrix):,} valid run-units\n按训练折Kimi权重排序")
    axis_heat.set_title("B  留一句对交叉拟合的 unit-level KFTA", loc="left", fontweight="bold", fontsize=12)
    colorbar = figure.colorbar(image, ax=axis_heat, fraction=0.025, pad=0.015)
    colorbar.set_label("held-out aligned KFTA", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    # C: patient curves and patient-equal mean.
    for patient in PATIENTS:
        axis_curve.plot(
            TIMES_MS,
            smooth_time(np.asarray(patient_curves[patient])[None, :])[0],
            color=patient_colors[patient],
            alpha=0.7,
            lw=1.35,
            label=patient,
        )
    axis_curve.plot(
        TIMES_MS,
        smooth_time(group_curve[None, :])[0],
        color="#17212B",
        lw=3.0,
        label="patient-equal mean",
    )
    axis_curve.axhline(0, color="#8B96A3", lw=0.9, ls="--")
    axis_curve.axvline(0, color="#242B33", lw=1.0)
    axis_curve.axvspan(
        WEIGHT_WINDOW_MS[0],
        WEIGHT_WINDOW_MS[1],
        color="#F4C95D",
        alpha=0.16,
        label="post-selected 300–500 ms",
    )
    axis_curve.axvline(385, color="#E0A800", lw=1.2, ls="--", label="original peak 385 ms")
    axis_curve.set_xlim(-200, 800)
    axis_curve.set_xlabel("名词起始后的时间 (ms)")
    axis_curve.set_ylabel("Kimi-feature-triggered neural average")
    axis_curve.set_title("C  五位患者的KFTA时间过程", loc="left", fontweight="bold", fontsize=12)
    axis_curve.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper right")
    axis_curve.spines[["top", "right"]].set_visible(False)

    # D: patient-equal endpoint, explicitly descriptive.
    rng = np.random.default_rng(SEED)
    jitter = rng.uniform(-0.05, 0.05, size=len(PATIENTS))
    axis_patient.axhline(0, color="#8B96A3", lw=0.9, ls="--")
    for index, patient in enumerate(PATIENTS):
        axis_patient.scatter(
            jitter[index],
            endpoints[index],
            s=70,
            color=patient_colors[patient],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis_patient.text(
            jitter[index] + 0.025,
            endpoints[index],
            patient,
            fontsize=8,
            va="center",
            color="#2C3640",
        )
    axis_patient.plot([-0.16, 0.16], [endpoints.mean(), endpoints.mean()], color="#D1495B", lw=2.5)
    axis_patient.set_xlim(-0.22, 0.28)
    axis_patient.set_xticks([])
    axis_patient.set_ylabel("mean KFTA, 300–500 ms")
    axis_patient.set_title("D  患者等权端点", loc="left", fontweight="bold", fontsize=12)
    axis_patient.text(
        0.03,
        0.97,
        f"mean = {endpoints.mean():.3f}\nrange = {endpoints.min():.3f} to {endpoints.max():.3f}",
        transform=axis_patient.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD2D9"},
    )
    axis_patient.text(
        0.03,
        0.04,
        "描述性图：Layer 1与300–500 ms\n来自既有热图，不报告新的独立P值",
        transform=axis_patient.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#A13D4A",
    )
    axis_patient.spines[["top", "right", "bottom"]].set_visible(False)

    figure.suptitle(
        "Kimi Layer 1 表面位置特征触发的神经群体平均（KFTA）",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.955,
        "模型轴与神经权重均在其他9个语义句对中估计，再应用于完全留出的句对；run内汇总后五位患者等权。",
        fontsize=9.5,
        color="#4D5A67",
    )
    figure.text(
        0.04,
        0.012,
        f"Kimi-VL-A3B-Thinking 4-bit · full-noun last-token state · level {KIMI_LEVEL} · {summary['valid_kfta_run_unit_count']:,} valid / {summary['run_unit_count']:,} registered run-units · 8 runs / 5 patients",
        fontsize=7.8,
        color="#596775",
    )
    figure.subplots_adjust(left=0.065, right=0.98, top=0.90, bottom=0.09)
    for suffix in ("png", "pdf", "svg"):
        path = FIGURES / f"Fig_3_Kimi_feature_triggered_neural_average.{suffix}"
        figure.savefig(
            path,
            dpi=300 if suffix == "png" else None,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_activation_neural_comparison(results: dict[str, object]) -> None:
    """Compare Kimi activation and neural projections for the same held-out items."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    setup_matplotlib()
    layer_profile = results["layer_profile"].copy()
    patient_curves = results["patient_curves"]
    group_item = results["group_item"].copy()
    patient_correlations = results["patient_correlations"].copy()
    summary = results["summary"]

    patient_colors = dict(
        zip(PATIENTS, ["#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B"])
    )
    position_colors = {"initial": "#2B6CB0", "post_marker": "#E69F00"}
    role_markers = {"agent": "o", "patient": "s"}

    figure = plt.figure(figsize=(14.2, 8.2), facecolor="#F7F8FA")
    grid = GridSpec(
        2,
        12,
        figure=figure,
        height_ratios=(0.92, 1.08),
        hspace=0.44,
        wspace=1.05,
    )
    axis_layers = figure.add_subplot(grid[0, :5])
    axis_time = figure.add_subplot(grid[0, 5:])
    axis_items = figure.add_subplot(grid[1, :8])
    axis_patients = figure.add_subplot(grid[1, 8:])

    # A: the actual Kimi activation profile across all cached levels.
    valid_profile = layer_profile.loc[
        layer_profile["valid_crossfit_layer"].astype(bool)
        & np.isfinite(layer_profile["heldout_post_minus_initial_mean"])
    ]
    axis_layers.plot(
        valid_profile["kimi_level"],
        valid_profile["heldout_post_minus_initial_mean"],
        color="#34495E",
        lw=1.6,
        marker="o",
        markersize=3.8,
    )
    selected = layer_profile.loc[layer_profile["kimi_level"].eq(KIMI_LEVEL)].iloc[0]
    if bool(selected["valid_crossfit_layer"]):
        axis_layers.scatter(
            [KIMI_LEVEL],
            [selected["heldout_post_minus_initial_mean"]],
            s=90,
            color="#D55E00",
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
            label=f"selected Level {KIMI_LEVEL}",
        )
    invalid_levels = layer_profile.loc[
        ~layer_profile["valid_crossfit_layer"].astype(bool), "kimi_level"
    ].astype(int).tolist()
    axis_layers.axhline(0, color="#8B96A3", lw=0.9, ls="--")
    axis_layers.set_xlim(-0.5, int(layer_profile["kimi_level"].max()) + 0.5)
    axis_layers.set_xlabel("Kimi hidden-state level")
    axis_layers.set_ylabel("held-out position effect\n(post-marker − initial)")
    axis_layers.set_title("A  Kimi activation 的跨层位置效应", loc="left", fontweight="bold", fontsize=12)
    axis_layers.legend(frameon=False, fontsize=8, loc="best")
    if invalid_levels:
        axis_layers.text(
            0.02,
            0.04,
            "不可估计层：" + ", ".join(map(str, invalid_levels)) + "（某折位置轴为0）",
            transform=axis_layers.transAxes,
            fontsize=7.5,
            color="#6B7785",
            va="bottom",
        )
    axis_layers.spines[["top", "right"]].set_visible(False)

    # B: the neural time axis remains milliseconds and is never equated with layers.
    smoothed_patient = np.stack(
        [smooth_time(np.asarray(patient_curves[p], dtype=float)[None, :])[0] for p in PATIENTS]
    )
    neural_mean = smoothed_patient.mean(axis=0)
    neural_sem = smoothed_patient.std(axis=0, ddof=1) / np.sqrt(len(PATIENTS))
    for index, patient in enumerate(PATIENTS):
        axis_time.plot(
            TIMES_MS,
            smoothed_patient[index],
            color=patient_colors[patient],
            lw=1.0,
            alpha=0.43,
        )
    axis_time.fill_between(
        TIMES_MS,
        neural_mean - neural_sem,
        neural_mean + neural_sem,
        color="#17212B",
        alpha=0.13,
        lw=0,
        label="patient SEM",
    )
    axis_time.plot(TIMES_MS, neural_mean, color="#17212B", lw=2.8, label="patient-equal mean")
    axis_time.axhline(0, color="#8B96A3", lw=0.9, ls="--")
    axis_time.axvline(0, color="#242B33", lw=1.0)
    axis_time.axvspan(300, 500, color="#F4C95D", alpha=0.17, label="300–500 ms readout")
    axis_time.axvline(385, color="#D55E00", lw=1.2, ls="--", label="existing peak 385 ms")
    axis_time.set_xlim(-200, 800)
    axis_time.set_xlabel("Time from noun onset (ms)")
    axis_time.set_ylabel("Kimi-weighted neural projection")
    axis_time.set_title("B  同一 Kimi 特征对应的神经时间过程", loc="left", fontweight="bold", fontsize=12)
    axis_time.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    axis_time.spines[["top", "right"]].set_visible(False)

    # C: direct item-wise comparison of the same 40 noun events.
    for noun, cell in group_item.groupby("noun", sort=False):
        if len(cell) == 2:
            axis_items.plot(
                cell["kimi_position_score"],
                cell["patient_equal_neural_projection_z"],
                color="#A8B0B8",
                lw=0.75,
                alpha=0.48,
                zorder=1,
            )
    for role, marker in role_markers.items():
        for position, color in position_colors.items():
            cell = group_item.loc[
                group_item["thematic_role"].eq(role)
                & group_item["surface_position"].eq(position)
            ]
            axis_items.scatter(
                cell["kimi_position_score"],
                cell["patient_equal_neural_projection_z"],
                s=48,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.92,
                zorder=3,
            )
    x_values = group_item["kimi_position_score"].to_numpy(float)
    y_values = group_item["patient_equal_neural_projection_z"].to_numpy(float)
    if float(np.std(x_values)) > 1e-12:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(x_values.min(), x_values.max(), 100)
        axis_items.plot(
            x_line,
            intercept + slope * x_line,
            color="#3B434B",
            lw=1.5,
            ls="--",
            alpha=0.8,
            zorder=2,
        )
    axis_items.axhline(0, color="#8B96A3", lw=0.8, ls=":")
    axis_items.axvline(0, color="#8B96A3", lw=0.8, ls=":")
    axis_items.set_xlabel("held-out Kimi activation position score")
    axis_items.set_ylabel("held-out neural projection, 300–500 ms\n(patient-equal z score)")
    axis_items.set_title("C  逐事件直接比较：Kimi activation vs neural activity", loc="left", fontweight="bold", fontsize=12)
    axis_items.text(
        0.03,
        0.96,
        (
            f"40 held-out noun events: overall ρ = {summary['patient_equal_item_spearman_rho']:.2f}\n"
            f"control position: ρ = {summary['patient_equal_position_controlled_spearman_rho']:.2f}\n"
            f"20 noun-pair Δ: ρ = {summary['patient_equal_paired_delta_spearman_rho']:.2f}"
        ),
        transform=axis_items.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD2D9"},
    )
    axis_items.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=position_colors["initial"], markeredgecolor="white", label="句首"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=position_colors["post_marker"], markeredgecolor="white", label="标记词后"),
            Line2D([0], [0], marker="o", color="#5C6670", markerfacecolor="none", lw=0, label="Agent noun"),
            Line2D([0], [0], marker="s", color="#5C6670", markerfacecolor="none", lw=0, label="Patient noun"),
        ],
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower right",
    )
    axis_items.spines[["top", "right"]].set_visible(False)

    # D: patient is the replication unit; separate overall position separation
    # from graded item correspondence after controlling surface position.
    patient_correlations["patient"] = pd.Categorical(
        patient_correlations["patient"], categories=list(PATIENTS), ordered=True
    )
    patient_correlations = patient_correlations.sort_values("patient")
    x_positions = np.arange(len(patient_correlations))
    axis_patients.axhline(0, color="#8B96A3", lw=0.9, ls="--")
    for x_index, row in enumerate(patient_correlations.itertuples(index=False)):
        patient = str(row.patient)
        axis_patients.plot(
            [x_index - 0.08, x_index + 0.08],
            [float(row.spearman_rho), float(row.position_controlled_spearman_rho)],
            color="#A8B0B8",
            lw=1.0,
            zorder=1,
        )
        axis_patients.scatter(
            [x_index - 0.08],
            [float(row.spearman_rho)],
            s=75,
            color=patient_colors[patient],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis_patients.scatter(
            [x_index + 0.08],
            [float(row.position_controlled_spearman_rho)],
            s=62,
            marker="D",
            facecolor="white",
            edgecolor=patient_colors[patient],
            linewidth=1.5,
            zorder=3,
        )
    rho_mean = float(patient_correlations["spearman_rho"].mean())
    controlled_mean = float(
        patient_correlations["position_controlled_spearman_rho"].mean()
    )
    axis_patients.axhline(rho_mean, color="#D1495B", lw=1.5, ls="-", alpha=0.85)
    axis_patients.axhline(controlled_mean, color="#3B6E8F", lw=1.3, ls=":", alpha=0.9)
    axis_patients.set_xticks(x_positions, [str(value) for value in patient_correlations["patient"]], rotation=25, ha="right")
    axis_patients.set_ylim(-1.0, 1.0)
    axis_patients.set_ylabel("item-level Spearman ρ")
    axis_patients.set_title("D  位置分离 vs 连续强度", loc="left", fontweight="bold", fontsize=12)
    axis_patients.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#6D597A", markeredgecolor="white", label=f"overall, mean ρ={rho_mean:.2f}"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="#3B6E8F", label=f"control position, mean ρ={controlled_mean:.2f}"),
        ],
        frameon=False,
        fontsize=7.5,
        loc="upper left",
    )
    axis_patients.text(
        0.03,
        0.04,
        "每位患者 n=40 事件\n仅作post-selection描述",
        transform=axis_patients.transAxes,
        fontsize=8.5,
        color="#A13D4A",
        va="bottom",
    )
    axis_patients.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Kimi activation 与神经活动的逐事件对应",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.955,
        "Layer 是模型深度，毫秒是神经时间；二者通过同一批40个留出名词事件的数值投影连接，而不是直接对齐两条轴。",
        fontsize=9.5,
        color="#4D5A67",
    )
    figure.text(
        0.04,
        0.012,
        "Kimi Level 1与300–500 ms窗口来自既有layer×time结果；所有面板用于解释该结果，不构成新的独立显著性检验。",
        fontsize=7.8,
        color="#596775",
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.10)
    for suffix in ("png", "pdf", "svg"):
        path = FIGURES / f"Fig_4_Kimi_activation_vs_neural_projection.{suffix}"
        figure.savefig(
            path,
            dpi=300 if suffix == "png" else None,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
    plt.close(figure)


def write_methods(results: dict[str, object]) -> None:
    summary = results["summary"]
    endpoints = np.asarray(results["endpoints"], dtype=float)
    text = f"""# Kimi feature-triggered neural average（KFTA）

## 图的目的

把原来的 Kimi layer×time RSA 热图转成类似 spike-triggered average 的直观时间过程。传统 STA 以 spike 为触发量；本图以 Kimi Layer 1 的名词表面位置分数作为连续触发特征。

第二张直接比较图不把模型layer等同于神经毫秒，而是在同一批40个留出名词事件上比较：横轴是Kimi activation position score，纵轴是该事件300–500 ms神经群体投影。

## 计算

1. 固定既有热图的描述性峰值层：Kimi Level {KIMI_LEVEL}。
2. 每次留出一个语义句对；用其余9个句对估计 `post-marker − initial` 的 Kimi位置轴。
3. 将训练和测试名词状态投影到该轴，并在每个名词内中心化，减少稳定词身份成分。
4. 在训练句对的300–500 ms神经响应中估计每个unit对Kimi分数的方向和权重。
5. 把权重应用于完全留出的4个名词事件，计算held-out KFTA时间过程。
6. 10折先在run内平均；同患者多个run再平均；最后5位患者等权。

## 数据

- 开放权重模型：`mlx-community/Kimi-VL-A3B-Thinking-4bit`。
- Kimi读出：完整名词最后token，Level {KIMI_LEVEL}。
- 神经单元：`{UNIT_SET}`，共 {summary['run_unit_count']:,} 个注册run-units；其中 {summary['valid_kfta_run_unit_count']:,} 个在训练折中有非零方差并进入KFTA。
- 时间：名词起始前200 ms至起始后800 ms，10 ms bin；每个事件用−200至0 ms作基线。

## 患者端点

- 300–500 ms patient-equal mean KFTA：{endpoints.mean():.4f}。
- 患者范围：{endpoints.min():.4f} 至 {endpoints.max():.4f}。
- 描述性exact sign-flip P：{summary['exact_signflip_p_descriptive_only']:.4f}，不可作为新独立检验。
- 40个事件的患者等权整体Spearman ρ：{summary['patient_equal_item_spearman_rho']:.3f}。
- 控制句首/标记词后表面位置后的Spearman ρ：{summary['patient_equal_position_controlled_spearman_rho']:.3f}。
- 20个名词的 `ΔKimi(post−initial)` 对 `Δneural(post−initial)` Spearman ρ：{summary['patient_equal_paired_delta_spearman_rho']:.3f}。
- 五位患者逐项Spearman ρ均值：{summary['patient_item_spearman_mean']:.3f}，范围 {summary['patient_item_spearman_range'][0]:.3f} 至 {summary['patient_item_spearman_range'][1]:.3f}。

## 统计边界

Layer 1和300–500 ms窗口来自已经观察过的layer×time热图。因此，尽管刺激句对和神经权重在图内做了交叉拟合，这仍是post-selection descriptive visualization。它可以解释既有显著簇的方向与时间形状，不能再提供一个新的独立显著性结论。

整体逐项相关同时包含了“句首 vs 标记词后”的二分类分离；只有控制位置后的相关或名词内差值相关，才能支持Kimi activation连续强度被神经活动逐项追踪。当前结果中这两个控制相关接近0，因此准确结论是共享表面位置分离，而不是精细的连续activation幅度映射。
"""
    (OUTPUT / "METHODS_AND_INTERPRETATION_zh.md").write_text(text, encoding="utf-8")


def main() -> None:
    results = analyze()
    plot(results)
    plot_activation_neural_comparison(results)
    write_methods(results)
    print(
        json.dumps(
            {
                "stage": "done",
                "figures": [
                    str(FIGURES / "Fig_3_Kimi_feature_triggered_neural_average.png"),
                    str(FIGURES / "Fig_4_Kimi_activation_vs_neural_projection.png"),
                ],
                "summary": results["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
