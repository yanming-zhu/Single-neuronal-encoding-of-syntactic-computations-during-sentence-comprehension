#!/usr/bin/env python3
"""Build Extended Data Fig. 4 controls for noun identity and position.

The main Fig. 4 remains unchanged.  This companion figure adds direct Kimi
layer decoding, real-input perturbations, neural metric sensitivity,
descriptive time-resolved decoding, and a permutation calibration for the
negative cross-position noun-identity decoder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import analyze_noun_role_structure_encoding as noun_analysis
from scripts import decode_noun_role_multifactor as neural_decode
from scripts import make_fig4_noun_identity_position_kimi_20260806 as fig4


DATE_TAG = "20260807"
OUT = ROOT / f"reports/fig4_extended_data_{DATE_TAG}"
FIG_DIR = OUT / "figures"
SOURCE_DIR = OUT / "source_data"
QA_DIR = OUT / "qa"
MAIN_SOURCE = ROOT / "reports/fig4_noun_identity_position_kimi_20260806/source_data"
NEURAL_DIR = ROOT / "neuropixel2025/data/combined_noun_role_structure_encoding"
EVENTS_PATH = NEURAL_DIR / "noun_role_event_metadata.csv"
NOUN_PREDICTIONS_PATH = NEURAL_DIR / "noun_role_multifactor_noun_predictions.csv"
KIMI_PERTURBATION_CACHE = SOURCE_DIR / "kimi_perturbations/representations.npz"
ORIGINAL_KIMI_CACHE = (
    ROOT
    / "reports/multirun_grouped_go_20260713/llm"
    / "exploratory_kimi_vl_a3b_4bit_20260723/representation_cache/representations.npz"
)

FONT_PT = 8.0
NOUN = fig4.NOUN
POSITION = fig4.POSITION
POSITION_DARK = fig4.POSITION_DARK
NEUTRAL = fig4.NEUTRAL
LIGHT = fig4.LIGHT
BLACK = fig4.BLACK
PATIENT_COLORS = fig4.PATIENT_COLORS
PURPLE = "#7C6BA8"
GREEN = "#4D9078"
GOLD = "#C88B2D"

WINDOW_WIDTH_MS = 100
WINDOW_STEP_MS = 50
WINDOW_STARTS_MS = np.arange(-200, 751, WINDOW_STEP_MS, dtype=int)
WINDOW_CENTERS_MS = WINDOW_STARTS_MS + WINDOW_WIDTH_MS / 2
N_PERMUTATIONS = 20_000
RNG_SEED = 20260807


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": FONT_PT,
            "axes.labelsize": FONT_PT,
            "axes.titlesize": FONT_PT,
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
            "figure.titlesize": FONT_PT,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_dirs() -> None:
    for path in (OUT, FIG_DIR, SOURCE_DIR, QA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_PT,
        fontweight="bold",
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms > 1e-12, norms, 1.0)


def original_kimi_cache_validation() -> dict[str, object]:
    if not KIMI_PERTURBATION_CACHE.exists() or not ORIGINAL_KIMI_CACHE.exists():
        return {"exact_match": False, "max_abs_difference": None, "n_rows": 0}
    with np.load(KIMI_PERTURBATION_CACHE, allow_pickle=False) as archive:
        keep = np.asarray(archive["perturbations"]).astype(str) == "original"
        new_hidden = np.asarray(archive["hidden_last"])[keep]
        new_nouns = np.asarray(archive["source_nouns"]).astype(str)[keep]
        new_voices = np.asarray(archive["voices"]).astype(str)[keep]
    with np.load(ORIGINAL_KIMI_CACHE, allow_pickle=False) as archive:
        keep = np.asarray(archive["constituent_ids"]).astype(str) != "verb"
        old_hidden = np.asarray(archive["hidden_last"])[keep]
        old_nouns = np.asarray(archive["target_text"]).astype(str)[keep]
        old_voices = np.asarray(archive["voices"]).astype(str)[keep]
    lookup = {(noun, voice): index for index, (noun, voice) in enumerate(zip(old_nouns, old_voices))}
    indices = np.asarray([lookup[(noun, voice)] for noun, voice in zip(new_nouns, new_voices)], dtype=int)
    difference = np.abs(new_hidden - old_hidden[indices])
    return {
        "exact_match": bool(np.array_equal(new_hidden, old_hidden[indices])),
        "max_abs_difference": float(difference.max()),
        "n_rows": int(len(new_hidden)),
    }


def cross_role_position_summary(meta: pd.DataFrame, x: np.ndarray) -> tuple[int, int, float]:
    truth: list[str] = []
    prediction: list[str] = []
    for train_semantic, test_semantic in (("agent", "patient"), ("patient", "agent")):
        for held_pair in sorted(meta["pair_id"].unique()):
            train = meta["thematic_role"].eq(train_semantic) & meta["pair_id"].ne(held_pair)
            test = meta["thematic_role"].eq(test_semantic) & meta["pair_id"].eq(held_pair)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RNG_SEED,
                ),
            )
            model.fit(x[train], meta.loc[train, "surface_position"])
            truth.extend(meta.loc[test, "surface_position"].astype(str))
            prediction.extend(model.predict(x[test]).astype(str))
    truth_array = np.asarray(truth)
    prediction_array = np.asarray(prediction)
    correct = int(np.sum(truth_array == prediction_array))
    tests = int(len(truth_array))
    return correct, tests, correct / tests


def analyze_kimi_perturbations() -> pd.DataFrame:
    if not KIMI_PERTURBATION_CACHE.exists():
        raise FileNotFoundError(
            f"Kimi perturbation cache is missing: {KIMI_PERTURBATION_CACHE}. "
            "Run extract_kimi_position_perturbations_20260807.py first."
        )
    with np.load(KIMI_PERTURBATION_CACHE, allow_pickle=False) as archive:
        hidden = np.asarray(archive["hidden_last"], dtype=np.float64)
        meta = pd.DataFrame(
            {
                "perturbation": np.asarray(archive["perturbations"]).astype(str),
                "observation_id": np.asarray(archive["observation_ids"]).astype(str),
                "pair_id": np.asarray(archive["pair_ids"]).astype(int),
                "voice": np.asarray(archive["voices"]).astype(str),
                "thematic_role": np.asarray(archive["thematic_roles"]).astype(str),
                "surface_position": np.asarray(archive["surface_positions"]).astype(str),
                "source_noun": np.asarray(archive["source_nouns"]).astype(str),
                "target_text": np.asarray(archive["target_text"]).astype(str),
            }
        )
        layer_ids = np.asarray(archive["layer_ids"], dtype=int)
    if hidden.shape != (200, 28, 2048):
        raise RuntimeError(f"Unexpected perturbation hidden-state shape: {hidden.shape}")
    rows: list[dict[str, object]] = []
    for perturbation in meta["perturbation"].drop_duplicates():
        keep = meta["perturbation"].eq(perturbation).to_numpy()
        cell_meta = meta.loc[keep].reset_index(drop=True)
        for level in layer_ids:
            x = normalize_rows(hidden[keep, int(level)])
            correct, tests, accuracy = cross_role_position_summary(cell_meta, x)
            rows.append(
                {
                    "perturbation": perturbation,
                    "level": int(level),
                    "n_correct": correct,
                    "n_tests": tests,
                    "accuracy": accuracy,
                    "chance": 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(SOURCE_DIR / "kimi_perturbation_cross_role_position_decoding.csv", index=False)
    return frame


def time_resolved_neural_decoding(recompute: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_path = SOURCE_DIR / "time_resolved_run_decoding.csv"
    patient_path = SOURCE_DIR / "time_resolved_patient_decoding.csv"
    if run_path.exists() and patient_path.exists() and not recompute:
        return pd.read_csv(run_path), pd.read_csv(patient_path)

    events = pd.read_csv(EVENTS_PATH)
    run_rows: list[dict[str, object]] = []
    run_order = events[["patient", "run"]].drop_duplicates().itertuples(index=False, name=None)
    for patient, run in run_order:
        run_events = (
            events.loc[events["patient"].eq(patient) & events["run"].eq(run)]
            .sort_values(["trial_index", "role"])
            .reset_index(drop=True)
        )
        meta = run_events[
            [
                "patient",
                "run",
                "trial_index",
                "sound_file",
                "sentence_condition",
                "role",
                "noun",
                "counterpart_noun",
                "event_bin_1000Hz",
            ]
        ].copy()
        meta["pair_id"] = meta["sound_file"].map(neural_decode.parse_pair_id)
        with np.load(
            noun_analysis.run_npz_path(str(patient), str(run), "usable_lenient"),
            allow_pickle=True,
        ) as archive:
            firing = np.asarray(archive["firing_rate_hz"], dtype=np.float32)

        baselines = np.stack(
            [
                firing[:, int(row.event_bin_1000Hz) - 500 : int(row.event_bin_1000Hz)].mean(axis=1)
                for row in run_events.itertuples(index=False)
            ]
        )
        for start_ms, center_ms in zip(WINDOW_STARTS_MS, WINDOW_CENTERS_MS):
            responses = np.stack(
                [
                    firing[
                        :,
                        int(row.event_bin_1000Hz) + int(start_ms) :
                        int(row.event_bin_1000Hz) + int(start_ms) + WINDOW_WIDTH_MS,
                    ].mean(axis=1)
                    for row in run_events.itertuples(index=False)
                ]
            )
            x = responses - baselines
            cross_predictions = fig4.cross_role_position_predictions(meta, x, seed=RNG_SEED)
            noun_predictions, _noun_summary = neural_decode.decode_noun_cross_role(meta, x)
            for target, predictions, chance in (
                ("cross-role position", cross_predictions, 0.5),
                ("noun identity across position", noun_predictions, 0.05),
            ):
                run_rows.append(
                    {
                        "patient": patient,
                        "run": run,
                        "window_start_ms": int(start_ms),
                        "window_stop_ms": int(start_ms + WINDOW_WIDTH_MS),
                        "window_center_ms": float(center_ms),
                        "target": target,
                        "n_correct": int(predictions["correct"].sum()),
                        "n_tests": int(len(predictions)),
                        "accuracy": float(predictions["correct"].mean()),
                        "chance": chance,
                    }
                )
        print(json.dumps({"time_resolved_complete": f"{patient}/{run}"}, ensure_ascii=False), flush=True)
    run_frame = pd.DataFrame(run_rows)
    patient_frame = (
        run_frame.groupby(
            ["patient", "window_start_ms", "window_stop_ms", "window_center_ms", "target", "chance"],
            as_index=False,
        )
        .agg(n_correct=("n_correct", "sum"), n_tests=("n_tests", "sum"))
    )
    patient_frame["accuracy"] = patient_frame["n_correct"] / patient_frame["n_tests"]
    patient_frame["accuracy_minus_chance"] = patient_frame["accuracy"] - patient_frame["chance"]
    run_frame.to_csv(run_path, index=False)
    patient_frame.to_csv(patient_path, index=False)
    return run_frame, patient_frame


def noun_identity_permutation_null(recompute: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    null_path = SOURCE_DIR / "noun_identity_patient_equal_permutation_null.csv"
    summary_path = SOURCE_DIR / "noun_identity_permutation_summary.json"
    if null_path.exists() and summary_path.exists() and not recompute:
        return pd.read_csv(null_path), json.loads(summary_path.read_text(encoding="utf-8"))

    predictions = pd.read_csv(NOUN_PREDICTIONS_PATH)
    predictions = predictions.loc[predictions["metric"].eq("bc")].copy()
    patients = sorted(predictions["patient"].unique())
    patient_index = {patient: index for index, patient in enumerate(patients)}
    patient_tests = predictions.groupby("patient").size().reindex(patients).to_numpy(int)
    observed_patient = (
        predictions.groupby("patient")["correct"].mean().reindex(patients).to_numpy(float)
    )
    observed = float(observed_patient.mean())

    rng = np.random.default_rng(RNG_SEED)
    null_correct = np.zeros((N_PERMUTATIONS, len(patients)), dtype=np.int16)
    group_cols = ["patient", "run", "train_role", "test_role"]
    for keys, cell in predictions.groupby(group_cols, sort=True):
        patient = str(keys[0])
        truth = cell["noun"].astype(str).to_numpy()
        predicted = cell["predicted_noun"].astype(str).to_numpy()
        random_order = np.argsort(rng.random((N_PERMUTATIONS, len(cell))), axis=1)
        permuted_truth = truth[random_order]
        null_correct[:, patient_index[patient]] += np.sum(permuted_truth == predicted, axis=1).astype(np.int16)
    null_accuracy = null_correct / patient_tests[None, :]
    null_patient_equal = null_accuracy.mean(axis=1)
    p_above = float((1 + np.sum(null_patient_equal >= observed - 1e-15)) / (N_PERMUTATIONS + 1))

    bootstrap_index = rng.integers(0, len(patients), size=(N_PERMUTATIONS, len(patients)))
    bootstrap_means = observed_patient[bootstrap_index].mean(axis=1)
    bootstrap_ci = np.quantile(bootstrap_means, [0.025, 0.975])
    null_ci = np.quantile(null_patient_equal, [0.025, 0.975])
    null_frame = pd.DataFrame(
        {
            "permutation": np.arange(1, N_PERMUTATIONS + 1),
            "patient_equal_accuracy": null_patient_equal,
        }
    )
    null_frame.to_csv(null_path, index=False)
    summary = {
        "n_permutations": N_PERMUTATIONS,
        "patients": patients,
        "patient_accuracies": {patient: float(value) for patient, value in zip(patients, observed_patient)},
        "observed_patient_equal_accuracy": observed,
        "chance": 0.05,
        "permutation_null_mean": float(null_patient_equal.mean()),
        "permutation_null_95_interval": [float(value) for value in null_ci],
        "permutation_p_one_sided_above_chance": p_above,
        "patient_bootstrap_95_interval": [float(value) for value in bootstrap_ci],
        "interpretation": "Calibration only; failure to exceed the null is not proof that noun information is absent.",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return null_frame, summary


def load_existing_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    layer = pd.read_csv(MAIN_SOURCE / "supplementary_kimi_layer_decoding.csv")
    sensitivity = pd.read_csv(MAIN_SOURCE / "cross_role_position_metric_sensitivity.csv")
    layer.to_csv(SOURCE_DIR / "kimi_layer_decoding.csv", index=False)
    sensitivity.to_csv(SOURCE_DIR / "neural_metric_sensitivity.csv", index=False)
    return layer, sensitivity


def metric_statistics(sensitivity: pd.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for metric, cell in sensitivity.groupby("metric", sort=False):
        values = cell.sort_values("patient")["accuracy"].to_numpy(float)
        output[str(metric)] = {
            "patient_equal_accuracy": float(values.mean()),
            "patient_sd": float(values.std(ddof=1)),
            "exact_patient_signflip_p_one_sided": fig4.exact_signflip(values - 0.5),
        }
    return output


def time_statistics(patient_time: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for (target, center), cell in patient_time.groupby(["target", "window_center_ms"], sort=True):
        values = cell.sort_values("patient")["accuracy_minus_chance"].to_numpy(float)
        rows.append(
            {
                "target": target,
                "window_center_ms": float(center),
                "patient_equal_accuracy_minus_chance": float(values.mean()),
                "patient_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
                "exact_patient_signflip_p_one_sided": fig4.exact_signflip(values),
            }
        )
    frame = pd.DataFrame(rows)
    frame["bonferroni_p_within_target"] = (
        frame.groupby("target")["exact_patient_signflip_p_one_sided"]
        .transform(lambda values: np.minimum(1.0, values * len(values)))
    )
    frame.to_csv(SOURCE_DIR / "time_resolved_patient_equal_summary.csv", index=False)
    peak_rows = []
    for target, cell in frame.groupby("target", sort=False):
        peak = cell.loc[cell["patient_equal_accuracy_minus_chance"].idxmax()]
        pre_noun = cell.loc[cell["window_center_ms"].lt(0)]
        pre_peak = pre_noun.loc[pre_noun["patient_equal_accuracy_minus_chance"].idxmax()]
        peak_rows.append(
            {
                "target": target,
                "peak_center_ms": float(peak["window_center_ms"]),
                "peak_accuracy_minus_chance": float(peak["patient_equal_accuracy_minus_chance"]),
                "peak_unadjusted_p": float(peak["exact_patient_signflip_p_one_sided"]),
                "pre_noun_peak_center_ms": float(pre_peak["window_center_ms"]),
                "pre_noun_peak_accuracy_minus_chance": float(pre_peak["patient_equal_accuracy_minus_chance"]),
                "pre_noun_peak_unadjusted_p": float(pre_peak["exact_patient_signflip_p_one_sided"]),
                "minimum_bonferroni_p": float(cell["bonferroni_p_within_target"].min()),
            }
        )
    return {"timepoint_family_size_per_target": int(frame["window_center_ms"].nunique()), "peaks": peak_rows}


def plot_panel_a(ax: plt.Axes, layer: pd.DataFrame) -> None:
    specs = (
        ("surface position", POSITION, "-", "position"),
        ("position across semantic roles", POSITION_DARK, "--", "cross-role position"),
        ("noun identity across position", NOUN, ":", "noun identity across position"),
    )
    for target, color, linestyle, label in specs:
        cell = layer.loc[layer["target"].eq(target)]
        ax.plot(cell["level"], 100 * cell["accuracy"], color=color, lw=1.4, ls=linestyle, label=label)
    ax.axhline(50, color=NEUTRAL, lw=0.7, ls="--")
    ax.axhline(5, color=NOUN, lw=0.6, ls=":", alpha=0.75)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 104)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax.set_xlabel("Kimi hidden-state level")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title("Kimi layer decoding", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.30))
    add_panel_label(ax, "a")


def plot_panel_b(ax: plt.Axes, perturbation: pd.DataFrame) -> None:
    specs = (
        ("original", POSITION_DARK, "Original"),
        ("noun_substituted", NOUN, "Noun substituted"),
        ("target_masked", GOLD, "Target masked"),
        ("marker_neutralized", GREEN, "Marker neutralized"),
        ("target_only", NEUTRAL, "Target only"),
    )
    for condition, color, label in specs:
        cell = perturbation.loc[perturbation["perturbation"].eq(condition)]
        ax.plot(cell["level"], 100 * cell["accuracy"], color=color, lw=1.2, label=label)
    ax.axhline(50, color=NEUTRAL, lw=0.7, ls="--")
    ax.set_xlim(0, 27)
    ax.set_ylim(35, 104)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax.set_xlabel("Kimi hidden-state level")
    ax.set_ylabel("Cross-role position accuracy (%)")
    ax.set_title("Input perturbation", loc="left")
    ax.legend(loc="center right")
    add_panel_label(ax, "b")


def plot_panel_c(ax: plt.Axes, sensitivity: pd.DataFrame, metric_stats: dict[str, dict[str, float]]) -> None:
    order = ["pre-noun baseline", "raw response", "baseline-corrected"]
    labels = ["Pre-noun\nbaseline", "Raw\nresponse", "Baseline-\ncorrected"]
    x = np.arange(len(order))
    wide = sensitivity.pivot(index="patient", columns="metric", values="accuracy").reindex(columns=order)
    for patient, row in wide.iterrows():
        values = 100 * row.to_numpy(float)
        color = PATIENT_COLORS.get(str(patient), LIGHT)
        ax.plot(x, values, color=color, lw=0.8, alpha=0.8)
        ax.scatter(x, values, s=16, color=color, edgecolor="white", linewidth=0.35, zorder=3)
    means = 100 * wide.mean(axis=0).to_numpy(float)
    ax.plot(x, means, color=BLACK, lw=1.5, zorder=4)
    for index, (metric, value) in enumerate(zip(order, means)):
        ax.plot([index - 0.16, index + 0.16], [value, value], color=POSITION_DARK, lw=2.6, zorder=5)
        p_value = metric_stats[metric]["exact_patient_signflip_p_one_sided"]
        ax.text(index, value + 2.5, f"{value:.1f}%\nP={p_value:.3f}", ha="center", va="bottom")
    ax.axhline(50, color=NEUTRAL, lw=0.7, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylim(30, 91)
    ax.set_ylabel("Cross-role position accuracy (%)")
    ax.set_title("Neural response controls", loc="left")
    add_panel_label(ax, "c")


def plot_panel_d(ax: plt.Axes, patient_time: pd.DataFrame) -> None:
    specs = (
        ("cross-role position", POSITION_DARK, "Cross-role position"),
        ("noun identity across position", NOUN, "Noun identity across position"),
    )
    ax.axvspan(0, 800, color="#F2F5F8", zorder=0)
    for target, color, label in specs:
        cell = patient_time.loc[patient_time["target"].eq(target)]
        summary = (
            cell.groupby("window_center_ms")["accuracy_minus_chance"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"])
        x = summary["window_center_ms"].to_numpy(float)
        y = 100 * summary["mean"].to_numpy(float)
        sem = 100 * summary["sem"].to_numpy(float)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, y, color=color, lw=1.4, label=label)
    ax.axhline(0, color=NEUTRAL, lw=0.7, ls="--")
    ax.axvline(0, color=BLACK, lw=0.7)
    ax.set_xlim(-200, 850)
    ax.set_xticks([-200, 0, 200, 400, 600, 800])
    ax.set_xlabel("Time from noun onset (ms)")
    ax.set_ylabel("Accuracy − chance (pp)")
    ax.set_title("Time-resolved neural decoding", loc="left")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.04, "100 ms / 50 ms; descriptive", transform=ax.transAxes, ha="right", color=NEUTRAL)
    add_panel_label(ax, "d", x=-0.16, y=1.16)


def make_panel_d_45mm(patient_time: pd.DataFrame) -> None:
    """Export panel d as a narrow, standalone 45-mm-wide figure."""
    configure_style()
    width_mm = 45.0
    height_mm = 55.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    # Preserve the original 13.5-mm bottom and 8.5-mm top margins, assigning
    # the full 5-mm height increase to the plotting area (28 -> 33 mm).
    ax = fig.add_axes([0.28, 13.5 / height_mm, 0.69, 33.0 / height_mm])
    specs = (
        ("cross-role position", POSITION_DARK, "Position"),
        ("noun identity across position", NOUN, "Noun ID"),
    )
    ax.axvspan(0, 800, color="#F2F5F8", zorder=0)
    for target, color, label in specs:
        cell = patient_time.loc[patient_time["target"].eq(target)]
        summary = (
            cell.groupby("window_center_ms")["accuracy_minus_chance"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"])
        x = summary["window_center_ms"].to_numpy(float)
        y = 100 * summary["mean"].to_numpy(float)
        sem = 100 * summary["sem"].to_numpy(float)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, y, color=color, lw=1.25, label=label)
    ax.axhline(0, color=NEUTRAL, lw=0.7, ls="--")
    ax.axvline(0, color=BLACK, lw=0.7)
    ax.set_xlim(-200, 850)
    ax.set_ylim(-10, 20)
    ax.set_xticks([-200, 0, 400, 800])
    ax.set_yticks([-10, 0, 10, 20])
    ax.set_xlabel("Time from noun onset\n(ms)", labelpad=3)
    ax.set_ylabel("Accuracy − chance (pp)", labelpad=2)
    ax.set_title("Time-resolved decoding", loc="left", pad=4)
    ax.legend(
        loc="upper left",
        handlelength=1.25,
        handletextpad=0.45,
        borderaxespad=0.25,
        labelspacing=0.25,
    )
    ax.text(
        0.98,
        0.04,
        "100/50 ms; descriptive",
        transform=ax.transAxes,
        ha="right",
        color=NEUTRAL,
    )
    fig.text(0.035, 0.96, "d", ha="left", va="top", fontsize=FONT_PT, fontweight="bold")
    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(
            FIG_DIR / f"ExtendedDataFig4d_time_resolved_45mm.{suffix}",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def plot_panel_e(ax: plt.Axes, null_frame: pd.DataFrame, summary: dict[str, object]) -> None:
    values = 100 * null_frame["patient_equal_accuracy"].to_numpy(float)
    observed = 100 * float(summary["observed_patient_equal_accuracy"])
    ax.hist(values, bins=np.arange(values.min() - 0.1, values.max() + 0.35, 0.25), density=True, color=LIGHT, edgecolor="white", linewidth=0.25)
    ax.axvline(5, color=NEUTRAL, lw=0.8, ls="--")
    ax.axvline(observed, color=NOUN, lw=1.6)
    ax.text(0.02, 0.94, f"observed = {observed:.2f}%", transform=ax.transAxes, color=NOUN, ha="left", va="top")
    ax.text(
        0.02,
        0.80,
        f"permutation P={summary['permutation_p_one_sided_above_chance']:.3f}",
        transform=ax.transAxes,
        ha="left",
    )
    ax.set_xlabel("Noun accuracy (%)")
    ax.set_ylabel("Permutation density")
    ax.set_title("Noun-identity null", loc="left")
    add_panel_label(ax, "e")


def make_figure(
    layer: pd.DataFrame,
    perturbation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    metric_stats: dict[str, dict[str, float]],
    patient_time: pd.DataFrame,
    null_frame: pd.DataFrame,
    null_summary: dict[str, object],
) -> None:
    configure_style()
    width_mm = 180.0
    height_mm = 125.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.94,
        hspace=0.58,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=(105, 75), wspace=0.38)
    bottom = outer[1].subgridspec(1, 3, width_ratios=(60, 75, 45), wspace=0.58)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])
    ax_e = fig.add_subplot(bottom[0, 2])
    plot_panel_a(ax_a, layer)
    plot_panel_b(ax_b, perturbation)
    plot_panel_c(ax_c, sensitivity, metric_stats)
    plot_panel_d(ax_d, patient_time)
    plot_panel_e(ax_e, null_frame, null_summary)
    fig.text(
        0.075,
        0.025,
        "Neural inference uses patients (n=5); Kimi item/layer and perturbation analyses are descriptive.",
        fontsize=FONT_PT,
        color=NEUTRAL,
    )
    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(
            FIG_DIR / f"ExtendedDataFig4_position_controls.{suffix}",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def write_outputs(
    layer: pd.DataFrame,
    perturbation: pd.DataFrame,
    metric_stats: dict[str, dict[str, float]],
    time_stats: dict[str, object],
    null_summary: dict[str, object],
) -> None:
    perturbation_level1 = {
        str(condition): float(cell.loc[cell["level"].eq(1), "accuracy"].iloc[0])
        for condition, cell in perturbation.groupby("perturbation", sort=False)
    }
    perturbation_deep_mean = {
        str(condition): float(cell.loc[cell["level"].ge(4), "accuracy"].mean())
        for condition, cell in perturbation.groupby("perturbation", sort=False)
    }
    level1_original = perturbation_level1["original"]
    level1_target_only = perturbation_level1["target_only"]
    noun_layer = layer.loc[layer["target"].eq("noun identity across position")]
    deep_noun_mean = float(noun_layer.loc[noun_layer["level"].ge(4), "accuracy"].mean())
    position_layer = layer.loc[layer["target"].eq("position across semantic roles")]
    deep_position_mean = float(position_layer.loc[position_layer["level"].ge(4), "accuracy"].mean())

    summary = {
        "figure_contract": {
            "core_conclusion": "Kimi retains noun identity in its embedding/earliest states, but deeper task-related structure and neural-Kimi correspondence are dominated by surface position/local prefix context; neural noun identity remains at chance.",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "final_size_mm": [180, 125],
            "font_size_pt": FONT_PT,
            "biological_replicates": "5 patients; runs and units nested within patient",
        },
        "metric_sensitivity": metric_stats,
        "time_resolved": time_stats,
        "noun_identity_permutation": null_summary,
        "kimi_layer_summary": {
            "noun_identity_embedding_accuracy": float(noun_layer.loc[noun_layer["level"].eq(0), "accuracy"].iloc[0]),
            "noun_identity_level1_accuracy": float(noun_layer.loc[noun_layer["level"].eq(1), "accuracy"].iloc[0]),
            "deep_levels_4_27_noun_identity_mean_accuracy": deep_noun_mean,
            "deep_levels_4_27_crossrole_position_mean_accuracy": deep_position_mean,
        },
        "kimi_perturbation_summary": {
            "level1_crossrole_position_accuracy": perturbation_level1,
            "deep_levels_4_27_mean_crossrole_position_accuracy": perturbation_deep_mean,
            "interpretation": "Target-only removes serial position and local prefix context together; it does not isolate syntax from sequence context.",
        },
        "interpretation_boundaries": [
            "Kimi contains strong noun identity information at the embedding and first hidden level; do not write that the model contains no lexical or semantic information.",
            "The deeper Kimi readout is position/local-context dominated for this task; this is not a claim that the whole model learned only position.",
            "Neural position remains confounded with serial position, preceding marker/local context, baseline state, and acoustic timing.",
            "Time-resolved windows and Kimi perturbations are exploratory/descriptive; no timepoint survives within-target Bonferroni correction with n=5.",
            "A chance-level noun decoder is absence of detected generalizable information, not proof of semantic absence.",
        ],
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    caption = (
        "Extended Data Fig. 4 | Layer, perturbation and neural controls for noun identity and surface position. "
        "a, Held-out Kimi decoding across the embedding and 27 hidden-state levels. Noun identity generalized across positions at the embedding and first hidden level but approached its 5% chance level in deeper states, whereas surface position and cross-role position approached ceiling. "
        "b, Cross-role position decoding after controlled input perturbations. Noun substitution, target masking and marker neutralization preserve different portions of the original prefix; the target-only condition removes all preceding serial and local context. These perturbations are descriptive and do not identify hierarchical syntax. "
        "c, Cross-role neural position decoding from pre-noun baseline, raw response and baseline-corrected response for each patient (n=5 patients). Exact one-sided patient sign-flip tests are shown without correction. "
        "d, Patient-equal time-resolved decoding with fixed 100-ms windows advanced in 50-ms steps. Lines are patient means and shading is s.e.m.; time-resolved results are descriptive, and no window survives Bonferroni correction within either target. "
        f"e, Permutation calibration of cross-position noun-identity decoding ({null_summary['n_permutations']:,} within-run/direction label permutations). The observed patient-equal accuracy was {100*null_summary['observed_patient_equal_accuracy']:.2f}% (chance 5%; one-sided permutation P={null_summary['permutation_p_one_sided_above_chance']:.3f}). "
        "Together, these controls show that Kimi contains early lexical identity information, while the deeper task-related and shared neural-Kimi component is dominated by surface position/local prefix context rather than stable noun-specific alignment. Source data are provided."
    )
    (OUT / "ExtendedDataFig4_caption.txt").write_text(caption + "\n", encoding="utf-8")

    position_time = next(item for item in time_stats["peaks"] if item["target"] == "cross-role position")
    zh = f"""# Extended Data Fig. 4 结果与解释

## 一句话结论

Kimi 在 embedding 和第 1 层保留了很强的名词身份信息，但在更深层中，可跨 Agent/Patient 泛化的任务表征主要是句首与标记词后位置/局部前缀上下文；神经数据仍未检出可跨位置泛化的名词身份编码。

## 新增结果

- Kimi 的名词身份跨位置解码在 embedding 为 {100*summary['kimi_layer_summary']['noun_identity_embedding_accuracy']:.1f}%，第 1 层为 {100*summary['kimi_layer_summary']['noun_identity_level1_accuracy']:.1f}%；第 4–27 层平均为 {100*deep_noun_mean:.2f}%（机会水平 5%）。同一深层范围的跨语义角色位置解码平均为 {100*deep_position_mean:.2f}%。
- Kimi 扰动分析中，第 1 层的跨角色位置解码为：原始输入 {100*perturbation_level1['original']:.1f}%，同语义角色内循环替换名词 {100*perturbation_level1['noun_substituted']:.1f}%，所有目标名词统一替换为“东西” {100*perturbation_level1['target_masked']:.1f}%，标记词统一为“把” {100*perturbation_level1['marker_neutralized']:.1f}%，仅保留目标名词 {100*perturbation_level1['target_only']:.1f}%。第 4–27 层的相应平均值为 {100*perturbation_deep_mean['original']:.1f}%、{100*perturbation_deep_mean['noun_substituted']:.1f}%、{100*perturbation_deep_mean['target_masked']:.1f}%、{100*perturbation_deep_mean['marker_neutralized']:.1f}% 和 {100*perturbation_deep_mean['target_only']:.1f}%。“仅目标名词”对照同时删除了串行位置和局部上下文，不能用来单独识别层级句法。
- 跨角色神经位置解码的患者等权准确率为：名词前基线 {100*metric_stats['pre-noun baseline']['patient_equal_accuracy']:.2f}%（P={metric_stats['pre-noun baseline']['exact_patient_signflip_p_one_sided']:.4f}），原始响应 {100*metric_stats['raw response']['patient_equal_accuracy']:.2f}%（P={metric_stats['raw response']['exact_patient_signflip_p_one_sided']:.4f}），基线校正响应 {100*metric_stats['baseline-corrected']['patient_equal_accuracy']:.2f}%（P={metric_stats['baseline-corrected']['exact_patient_signflip_p_one_sided']:.4f}）。
- 时间分辨分析采用固定 100-ms 窗、50-ms 步长，仅作描述性补充。跨角色位置解码在名词前 {position_time['pre_noun_peak_center_ms']:.0f} ms 窗中已达到 {100*(0.5+position_time['pre_noun_peak_accuracy_minus_chance']):.1f}%（未校正 P={position_time['pre_noun_peak_unadjusted_p']:.4f}），名词后峰值为 {position_time['peak_center_ms']:.0f} ms、{100*(0.5+position_time['peak_accuracy_minus_chance']):.1f}%（未校正 P={position_time['peak_unadjusted_p']:.4f}）。在 n=5 的 exact sign-flip 分辨率下，两个 target 均无任何时窗通过 target 内 Bonferroni 校正（跨角色位置的最小校正 P={position_time['minimum_bonferroni_p']:.3f}）。
- 名词身份跨位置解码的患者等权准确率为 {100*null_summary['observed_patient_equal_accuracy']:.2f}%，位于置换零分布内（单侧 P={null_summary['permutation_p_one_sided_above_chance']:.4f}）。

## 表述边界

1. 请写“Kimi 的早层保留名词身份信息，而该任务的深层可泛化读出主要由位置/局部上下文主导”，不要写“LLM 没有语义”或“LLM 只学到位置”。
2. 位置仍与串行位置、前导标记词、局部前缀和基线状态共变，不能称为纯层级句法编码。
3. 置换零结果支持“未检出超过机会水平的可泛化名词身份信息”，不能证明语义信息完全不存在。
"""
    (OUT / "analysis_and_interpretation_zh.md").write_text(zh, encoding="utf-8")


def write_qa() -> None:
    outputs = [
        FIG_DIR / "ExtendedDataFig4_position_controls.svg",
        FIG_DIR / "ExtendedDataFig4_position_controls.pdf",
        FIG_DIR / "ExtendedDataFig4_position_controls.png",
        FIG_DIR / "ExtendedDataFig4_position_controls.tiff",
        OUT / "ExtendedDataFig4_caption.txt",
        OUT / "analysis_summary.json",
        OUT / "analysis_and_interpretation_zh.md",
    ]
    qa: dict[str, object] = {
        "backend_exclusive": "Python/matplotlib",
        "outputs_exist": {path.name: path.exists() and path.stat().st_size > 0 for path in outputs},
        "figure_size_mm": [180, 125],
        "font_size_pt": FONT_PT,
        "panel_labels": ["a", "b", "c", "d", "e"],
        "svg_editable_text": False,
        "svg_all_text_8pt": False,
        "svg_font_sizes_pt": [],
        "svg_size_mm": [],
        "pdf_truetype_fonttype": mpl.rcParams["pdf.fonttype"] == 42,
        "statistics_note": "Patients (n=5) are biological replicates; time windows and Kimi analyses are exploratory/descriptive.",
        "image_integrity": "No microscopy or image manipulation; all panels are programmatic plots from tabular/model source data.",
        "kimi_original_cache_validation": original_kimi_cache_validation(),
        "source_data_expected_rows": {
            "kimi_perturbation_readouts": 200,
            "time_resolved_run_rows": 320,
            "time_resolved_patient_rows": 200,
            "noun_permutations": N_PERMUTATIONS,
        },
    }
    svg = FIG_DIR / "ExtendedDataFig4_position_controls.svg"
    if svg.exists():
        text = svg.read_text(encoding="utf-8", errors="ignore")
        qa["svg_editable_text"] = "<text" in text and ("font-family" in text or "font:" in text)
        font_sizes = sorted({float(value) for value in re.findall(r"font:\s+(?:\d+\s+)?([0-9.]+)px", text)})
        qa["svg_font_sizes_pt"] = font_sizes
        qa["svg_all_text_8pt"] = font_sizes == [FONT_PT]
        size_match = re.search(r'<svg[^>]+width="([0-9.]+)pt" height="([0-9.]+)pt"', text)
        if size_match:
            qa["svg_size_mm"] = [
                float(size_match.group(1)) / 72.0 * 25.4,
                float(size_match.group(2)) / 72.0 * 25.4,
            ]
    qa["pass"] = (
        all(qa["outputs_exist"].values())
        and bool(qa["svg_editable_text"])
        and bool(qa["svg_all_text_8pt"])
        and bool(qa["pdf_truetype_fonttype"])
        and bool(qa["kimi_original_cache_validation"]["exact_match"])
    )
    (QA_DIR / "qa_report.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_panel_d_qa() -> None:
    stem = FIG_DIR / "ExtendedDataFig4d_time_resolved_45mm"
    outputs = [stem.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png", ".tiff")]
    qa: dict[str, object] = {
        "backend_exclusive": "Python/matplotlib",
        "outputs_exist": {path.name: path.exists() and path.stat().st_size > 0 for path in outputs},
        "requested_size_mm": [45, 55],
        "font_size_pt": FONT_PT,
        "panel_label": "d",
        "svg_editable_text": False,
        "svg_all_text_8pt": False,
        "svg_font_sizes_pt": [],
        "svg_size_mm": [],
        "pdf_truetype_fonttype": mpl.rcParams["pdf.fonttype"] == 42,
        "raster_dpi": {},
        "data_provenance": "Real 1000-Hz neural firing rates from five patients; no simulated data.",
        "statistics_note": "Patient-equal means and across-patient SEM; exploratory/descriptive time windows.",
    }
    svg = stem.with_suffix(".svg")
    if svg.exists():
        text = svg.read_text(encoding="utf-8", errors="ignore")
        qa["svg_editable_text"] = "<text" in text and ("font-family" in text or "font:" in text)
        font_sizes = sorted({float(value) for value in re.findall(r"font:\s+(?:\d+\s+)?([0-9.]+)px", text)})
        qa["svg_font_sizes_pt"] = font_sizes
        qa["svg_all_text_8pt"] = font_sizes == [FONT_PT]
        size_match = re.search(r'<svg[^>]+width="([0-9.]+)pt" height="([0-9.]+)pt"', text)
        if size_match:
            qa["svg_size_mm"] = [
                float(size_match.group(1)) / 72.0 * 25.4,
                float(size_match.group(2)) / 72.0 * 25.4,
            ]
    try:
        from PIL import Image

        for suffix in (".png", ".tiff"):
            with Image.open(stem.with_suffix(suffix)) as image:
                dpi = image.info.get("dpi", (None, None))
                qa["raster_dpi"][suffix.lstrip(".")] = [
                    None if value is None else float(value) for value in dpi
                ]
    except (ImportError, OSError):
        pass
    size_ok = (
        len(qa["svg_size_mm"]) == 2
        and abs(qa["svg_size_mm"][0] - 45.0) < 0.01
        and abs(qa["svg_size_mm"][1] - 55.0) < 0.01
    )
    dpi_ok = all(
        len(values) == 2 and all(value is not None and abs(value - 600) < 1 for value in values)
        for values in qa["raster_dpi"].values()
    ) and len(qa["raster_dpi"]) == 2
    qa["pass"] = (
        all(qa["outputs_exist"].values())
        and bool(qa["svg_editable_text"])
        and bool(qa["svg_all_text_8pt"])
        and bool(qa["pdf_truetype_fonttype"])
        and size_ok
        and dpi_ok
    )
    (QA_DIR / "qa_panel_d_45mm.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-time", action="store_true")
    parser.add_argument("--recompute-permutation", action="store_true")
    parser.add_argument("--panel-d-only", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.panel_d_only:
        _run_time, patient_time = time_resolved_neural_decoding(False)
        make_panel_d_45mm(patient_time)
        write_panel_d_qa()
        print(
            json.dumps(
                {
                    "panel_d": str(FIG_DIR / "ExtendedDataFig4d_time_resolved_45mm.png"),
                    "qa": str(QA_DIR / "qa_panel_d_45mm.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    layer, sensitivity = load_existing_sources()
    perturbation = analyze_kimi_perturbations()
    _run_time, patient_time = time_resolved_neural_decoding(args.recompute_time)
    null_frame, null_summary = noun_identity_permutation_null(args.recompute_permutation)
    metric_stats = metric_statistics(sensitivity)
    time_stats = time_statistics(patient_time)
    make_figure(layer, perturbation, sensitivity, metric_stats, patient_time, null_frame, null_summary)
    make_panel_d_45mm(patient_time)
    write_outputs(layer, perturbation, metric_stats, time_stats, null_summary)
    write_qa()
    write_panel_d_qa()
    print(
        json.dumps(
            {
                "figure": str(FIG_DIR / "ExtendedDataFig4_position_controls.png"),
                "caption": str(OUT / "ExtendedDataFig4_caption.txt"),
                "summary": str(OUT / "analysis_and_interpretation_zh.md"),
                "qa": str(QA_DIR / "qa_report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
