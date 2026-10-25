#!/usr/bin/env python3
"""Build Fig. 4: noun identity, surface position and Kimi alignment.

The figure separates three questions that are easy to conflate:

1. whether single units show noun-identity or surface-position enrichment;
2. whether neural population activity generalizes a position decoder across
   Agent and Patient noun classes; and
3. whether the item-wise neural--Kimi correspondence remains after surface
   position is controlled.

All biological inference is summarized with patients (n=5) as the independent
units. Pooled unit counts are retained only as descriptive screening results.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import decode_noun_role_multifactor as neural_decode
from scripts import plot_kimi_feature_triggered_neural_average as kfta


DATE_TAG = "20260806"
OUT = ROOT / f"reports/fig4_noun_identity_position_kimi_{DATE_TAG}"
FIG_DIR = OUT / "figures"
SOURCE_DIR = OUT / "source_data"
QA_DIR = OUT / "qa"
KFTA_DIR = SOURCE_DIR / "kfta_usable_lenient"

NEURAL_DIR = ROOT / "neuropixel2025/data/combined_noun_role_structure_encoding"
CLASSIFIED_PATH = NEURAL_DIR / "noun_role_semantic_vs_syntax_unit_classification.csv"
RUN_DECODE_PATH = NEURAL_DIR / "noun_role_multifactor_decoding_run_summary.csv"
METRICS_PATH = NEURAL_DIR / "noun_role_unit_trial_response_metrics.csv"
KIMI_REPORT = ROOT / "reports/existing_neural_to_kimi_lowdim_mapping_20260730"

NOUN = "#D55E00"
POSITION = "#1479B8"
POSITION_DARK = "#075985"
VOICE = "#7B8794"
INITIAL = "#9EB7CC"
POST = "#1479B8"
NEUTRAL = "#6B7280"
LIGHT = "#D9DEE5"
BLACK = "#1F2933"
PATIENT_COLORS = {
    "gz001": "#355070",
    "gz002": "#6D597A",
    "gz003": "#B56576",
    "gz004": "#E56B6F",
    "gz005": "#EAAC8B",
}
FONT_PT = 8.0


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": FONT_PT,
            "axes.labelsize": FONT_PT,
            "axes.titlesize": FONT_PT,
            "axes.titleweight": "normal",
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
            "figure.titlesize": FONT_PT,
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
    for path in (OUT, FIG_DIR, SOURCE_DIR, QA_DIR, KFTA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def exact_signflip(values: np.ndarray, alternative: str = "greater") -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = float(values.mean())
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(float(np.mean(values * np.asarray(signs))))
    null = np.asarray(null)
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-15))
    if alternative == "two-sided":
        return float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    raise ValueError(alternative)


def exact_binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def p_text(value: float) -> str:
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=FONT_PT,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def load_unit_screen() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    frame = pd.read_csv(CLASSIFIED_PATH)
    pooled_rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []
    for metric in ("bc", "raw"):
        cell = frame.loc[frame["metric"].eq(metric)].copy()
        for factor, column in (
            ("noun identity", "semantic_noun_sig_uncorrected"),
            ("surface position", "syntax_role_sig_uncorrected"),
        ):
            k = int(cell[column].sum())
            n = int(len(cell))
            low, high = exact_binomial_ci(k, n)
            pooled_rows.append(
                {
                    "metric": metric,
                    "factor": factor,
                    "n_units": n,
                    "n_p_lt_0p05": k,
                    "rate": k / n,
                    "ci95_low": low,
                    "ci95_high": high,
                    "binomial_enrichment_p_vs_0p05": float(stats.binom.sf(k - 1, n, 0.05)),
                }
            )
        for patient, patient_cell in cell.groupby("patient", sort=True):
            for factor, column in (
                ("noun identity", "semantic_noun_sig_uncorrected"),
                ("surface position", "syntax_role_sig_uncorrected"),
            ):
                patient_rows.append(
                    {
                        "patient": patient,
                        "metric": metric,
                        "factor": factor,
                        "n_units": int(len(patient_cell)),
                        "n_p_lt_0p05": int(patient_cell[column].sum()),
                        "rate": float(patient_cell[column].mean()),
                    }
                )
    pooled = pd.DataFrame(pooled_rows)
    patient = pd.DataFrame(patient_rows)
    bc = patient.loc[patient["metric"].eq("bc")].pivot(
        index="patient", columns="factor", values="rate"
    )
    differences = (bc["surface position"] - bc["noun identity"]).to_numpy(float)
    stats_out = {
        "patient_equal_noun_rate": float(bc["noun identity"].mean()),
        "patient_equal_position_rate": float(bc["surface position"].mean()),
        "patient_paired_position_minus_noun_rate": float(differences.mean()),
        "patient_exact_signflip_p_one_sided": exact_signflip(differences),
    }
    pooled.to_csv(SOURCE_DIR / "panel_a_pooled_unit_screen.csv", index=False)
    patient.to_csv(SOURCE_DIR / "panel_b_patient_unit_rates.csv", index=False)
    return pooled, patient, stats_out


def semantic_role(meta: pd.DataFrame) -> np.ndarray:
    agent = (
        meta["sentence_condition"].eq("active") & meta["role"].eq("subject")
    ) | (
        meta["sentence_condition"].eq("passive") & meta["role"].eq("predicate")
    )
    return np.where(agent, "agent", "patient")


def cross_role_position_predictions(
    meta: pd.DataFrame,
    x: np.ndarray,
    seed: int = 20260806,
) -> pd.DataFrame:
    meta = meta.copy()
    meta["semantic_role"] = semantic_role(meta)
    rows: list[dict[str, object]] = []
    for train_semantic, test_semantic in (("agent", "patient"), ("patient", "agent")):
        for held_pair in sorted(meta["pair_id"].unique()):
            train = meta["semantic_role"].eq(train_semantic) & meta["pair_id"].ne(held_pair)
            test = meta["semantic_role"].eq(test_semantic) & meta["pair_id"].eq(held_pair)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                ),
            )
            model.fit(x[train], meta.loc[train, "role"])
            prediction = model.predict(x[test])
            test_meta = meta.loc[test].reset_index(drop=True)
            for source, predicted in zip(test_meta.itertuples(index=False), prediction):
                rows.append(
                    {
                        "held_pair": str(held_pair),
                        "train_semantic_role": train_semantic,
                        "test_semantic_role": test_semantic,
                        "noun": str(source.noun),
                        "voice": str(source.sentence_condition),
                        "true_position": str(source.role),
                        "predicted_position": str(predicted),
                        "correct": bool(source.role == predicted),
                    }
                )
    return pd.DataFrame(rows)


def analyze_neural_decoding() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    run_summary = pd.read_csv(RUN_DECODE_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    prediction_tables: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []
    metric_columns = {
        "baseline-corrected": "response_baseline_corrected_hz",
        "raw response": "response_hz",
        "pre-noun baseline": "baseline_hz",
    }
    run_order = list(metrics[["patient", "run"]].drop_duplicates().itertuples(index=False, name=None))
    for metric_label, metric_column in metric_columns.items():
        for patient, run in run_order:
            meta, x = neural_decode.event_matrix(metrics, metric_column, patient, run)
            predictions = cross_role_position_predictions(meta, x)
            predictions.insert(0, "metric", metric_label)
            predictions.insert(0, "run", run)
            predictions.insert(0, "patient", patient)
            prediction_tables.append(predictions)
            run_rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "metric": metric_label,
                    "n_tests": int(len(predictions)),
                    "n_correct": int(predictions["correct"].sum()),
                    "accuracy": float(predictions["correct"].mean()),
                }
            )
    cross_predictions = pd.concat(prediction_tables, ignore_index=True)
    cross_run = pd.DataFrame(run_rows)
    cross_predictions.to_csv(SOURCE_DIR / "panel_c_cross_semantic_role_predictions.csv", index=False)
    cross_run.to_csv(SOURCE_DIR / "panel_c_cross_semantic_role_run_summary.csv", index=False)

    standard = (
        run_summary.loc[run_summary["metric"].eq("bc")]
        .groupby("patient", as_index=False)
        .agg(
            noun_correct=("noun_correct", "sum"),
            noun_tests=("noun_tests", "sum"),
            voice_correct=("voice_correct", "sum"),
            voice_tests=("voice_tests", "sum"),
            position_correct=("role_correct", "sum"),
            position_tests=("role_tests", "sum"),
        )
    )
    cross_patient = (
        cross_predictions.loc[cross_predictions["metric"].eq("baseline-corrected")]
        .groupby("patient", as_index=False)
        .agg(
            crossrole_position_correct=("correct", "sum"),
            crossrole_position_tests=("correct", "size"),
        )
    )
    standard = standard.merge(cross_patient, on="patient", validate="one_to_one")
    long_rows: list[dict[str, object]] = []
    target_specs = (
        ("noun identity", "noun", 0.05),
        ("voice", "voice", 0.5),
        ("surface position", "position", 0.5),
        ("position across semantic roles", "crossrole_position", 0.5),
    )
    for row in standard.itertuples(index=False):
        for target, prefix, chance in target_specs:
            correct = int(getattr(row, f"{prefix}_correct"))
            tests = int(getattr(row, f"{prefix}_tests"))
            accuracy = correct / tests
            long_rows.append(
                {
                    "patient": row.patient,
                    "target": target,
                    "n_correct": correct,
                    "n_tests": tests,
                    "accuracy": accuracy,
                    "chance": chance,
                    "accuracy_minus_chance": accuracy - chance,
                }
            )
    patient_long = pd.DataFrame(long_rows)
    patient_long.to_csv(SOURCE_DIR / "panel_c_patient_decoding_endpoints.csv", index=False)

    sensitivity = (
        cross_predictions.groupby(["patient", "metric"], as_index=False)
        .agg(n_correct=("correct", "sum"), n_tests=("correct", "size"))
    )
    sensitivity["accuracy"] = sensitivity["n_correct"] / sensitivity["n_tests"]
    sensitivity["chance"] = 0.5
    sensitivity.to_csv(SOURCE_DIR / "cross_role_position_metric_sensitivity.csv", index=False)

    def endpoint(target: str) -> np.ndarray:
        return patient_long.loc[patient_long["target"].eq(target), "accuracy_minus_chance"].to_numpy(float)

    stats_out = {
        "noun_identity_patient_equal_accuracy": float(
            patient_long.loc[patient_long["target"].eq("noun identity"), "accuracy"].mean()
        ),
        "surface_position_patient_equal_accuracy": float(
            patient_long.loc[patient_long["target"].eq("surface position"), "accuracy"].mean()
        ),
        "crossrole_position_patient_equal_accuracy": float(
            patient_long.loc[
                patient_long["target"].eq("position across semantic roles"), "accuracy"
            ].mean()
        ),
        "crossrole_position_exact_patient_signflip_p_one_sided": exact_signflip(
            endpoint("position across semantic roles")
        ),
        "surface_position_exact_patient_signflip_p_one_sided": exact_signflip(
            endpoint("surface position")
        ),
        "noun_identity_exact_patient_signflip_p_one_sided": exact_signflip(
            endpoint("noun identity")
        ),
    }
    for metric_label, cell in sensitivity.groupby("metric", sort=False):
        values = cell["accuracy"].to_numpy(float)
        key = metric_label.replace("-", "_").replace(" ", "_")
        stats_out[f"crossrole_position_{key}_patient_equal_accuracy"] = float(values.mean())
        stats_out[f"crossrole_position_{key}_exact_patient_signflip_p_one_sided"] = exact_signflip(
            values - 0.5
        )
    return patient_long, sensitivity, stats_out


def supplementary_kimi_layer_decoding() -> pd.DataFrame:
    meta, _ = kfta.load_kimi_and_meta()
    levels = kfta.load_all_kimi_levels(meta)
    rows: list[dict[str, object]] = []
    for level in range(levels.shape[1]):
        x = np.asarray(levels[:, level], dtype=float)
        if np.nanstd(x) < 1e-12:
            rows.extend(
                [
                    {"level": level, "target": target, "n_correct": np.nan, "n_tests": np.nan, "accuracy": np.nan, "chance": chance}
                    for target, chance in (
                        ("surface position", 0.5),
                        ("position across semantic roles", 0.5),
                        ("noun identity across position", 0.05),
                    )
                ]
            )
            continue

        for mode in ("surface position", "position across semantic roles"):
            truth: list[str] = []
            prediction: list[str] = []
            for held_pair in sorted(meta["pair_id"].unique()):
                directions = [(None, None)] if mode == "surface position" else [
                    ("agent", "patient"),
                    ("patient", "agent"),
                ]
                for train_semantic, test_semantic in directions:
                    train = meta["pair_id"].ne(held_pair)
                    test = meta["pair_id"].eq(held_pair)
                    if train_semantic is not None:
                        train &= meta["thematic_role"].eq(train_semantic)
                        test &= meta["thematic_role"].eq(test_semantic)
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty="l2",
                            C=1.0,
                            solver="liblinear",
                            max_iter=2000,
                            random_state=20260806,
                        ),
                    )
                    model.fit(x[train], meta.loc[train, "surface_position"])
                    truth.extend(meta.loc[test, "surface_position"].astype(str))
                    prediction.extend(model.predict(x[test]).astype(str))
            correct = int(np.sum(np.asarray(truth) == np.asarray(prediction)))
            rows.append(
                {
                    "level": level,
                    "target": mode,
                    "n_correct": correct,
                    "n_tests": len(truth),
                    "accuracy": correct / len(truth),
                    "chance": 0.5,
                }
            )

        noun_correct: list[bool] = []
        for train_position, test_position in (("initial", "post_marker"), ("post_marker", "initial")):
            train = meta["surface_position"].eq(train_position).to_numpy()
            test = meta["surface_position"].eq(test_position).to_numpy()
            predicted = neural_decode.nearest_template_predictions(
                x[train], meta.loc[train, "noun"].to_numpy(str), x[test]
            )
            noun_correct.extend(predicted == meta.loc[test, "noun"].to_numpy(str))
        rows.append(
            {
                "level": level,
                "target": "noun identity across position",
                "n_correct": int(np.sum(noun_correct)),
                "n_tests": len(noun_correct),
                "accuracy": float(np.mean(noun_correct)),
                "chance": 0.05,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(SOURCE_DIR / "supplementary_kimi_layer_decoding.csv", index=False)
    return frame


def ensure_usable_kfta(recompute: bool) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    summary_path = KFTA_DIR / "summary.json"
    required = [
        summary_path,
        KFTA_DIR / "patient_equal_item_comparison.csv",
        KFTA_DIR / "patient_item_correlations.csv",
    ]
    if recompute or not all(path.exists() for path in required):
        kfta.UNIT_SET = "usable_lenient"
        kfta.OUTPUT = KFTA_DIR
        kfta.analyze()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    group_item = pd.read_csv(KFTA_DIR / "patient_equal_item_comparison.csv")
    patient_corr = pd.read_csv(KFTA_DIR / "patient_item_correlations.csv")
    return summary, group_item, patient_corr


def load_strict_mapping_checks() -> pd.DataFrame:
    geometry = pd.read_csv(KIMI_REPORT / "geometry_inference.csv")
    geometry = geometry.loc[geometry["primary"].astype(bool)].iloc[0]
    mapping = pd.read_csv(KIMI_REPORT / "mapping_inference.csv").iloc[0]
    absolute = pd.read_csv(KIMI_REPORT / "absolute_mapping_inference_exploratory.csv").iloc[0]
    frame = pd.DataFrame(
        [
            {
                "analysis": "controlled grammar-shift geometry",
                "estimate": float(geometry["patient_mean_rho"]),
                "chance": 0.0,
                "patient_exact_p_one_sided": float(geometry["exact_patient_signflip_p_one_sided"]),
                "primary": True,
            },
            {
                "analysis": "nested neural-to-Kimi shift mapping",
                "estimate": float(mapping["patient_mean"]),
                "chance": float(mapping["chance"]),
                "patient_exact_p_one_sided": float(mapping["exact_patient_signflip_p_one_sided"]),
                "primary": True,
            },
            {
                "analysis": "absolute noun-state mapping",
                "estimate": float(absolute["patient_mean"]),
                "chance": float(absolute["chance"]),
                "patient_exact_p_one_sided": float(absolute["exact_patient_signflip_p_one_sided"]),
                "primary": False,
            },
        ]
    )
    frame.to_csv(SOURCE_DIR / "strict_neural_kimi_mapping_checks.csv", index=False)
    return frame


def plot_panel_a(ax: plt.Axes, pooled: pd.DataFrame) -> None:
    x_centers = {"bc": 0.0, "raw": 1.35}
    offsets = {"noun identity": -0.18, "surface position": 0.18}
    colors = {"noun identity": NOUN, "surface position": POSITION}
    for metric in ("bc", "raw"):
        for factor in ("noun identity", "surface position"):
            row = pooled.loc[pooled["metric"].eq(metric) & pooled["factor"].eq(factor)].iloc[0]
            x = x_centers[metric] + offsets[factor]
            y = 100 * float(row["rate"])
            low = 100 * float(row["ci95_low"])
            high = 100 * float(row["ci95_high"])
            ax.bar(x, y, width=0.31, color=colors[factor], edgecolor=BLACK, linewidth=0.45)
            ax.errorbar(x, y, yerr=np.asarray([[y - low], [high - y]]), fmt="none", color=BLACK, lw=0.55, capsize=1.8)
            ax.text(x, high + 0.7, f"{int(row['n_p_lt_0p05'])}/{int(row['n_units'])}", ha="center", va="bottom", fontsize=FONT_PT)
    ax.axhline(5, color=NEUTRAL, lw=0.8, ls="--")
    ax.text(0.80, 5.30, "5% null", ha="center", va="bottom", color=NEUTRAL, fontsize=FONT_PT)
    ax.set_xticks([0, 1.35], ["Baseline-\ncorrected", "Raw\nresponse"])
    ax.set_xlim(-0.55, 1.80)
    ax.set_ylim(0, 19.5)
    ax.set_ylabel("Units with factor P < 0.05 (%)")
    ax.set_title("Unit-level factor screen", loc="left")
    ax.legend(
        handles=[
            Line2D([0], [0], color=NOUN, lw=5, label="noun ID"),
            Line2D([0], [0], color=POSITION, lw=5, label="position"),
        ],
        loc="upper right",
    )


def plot_panel_b(ax: plt.Axes, patient: pd.DataFrame, stats_out: dict[str, float]) -> None:
    cell = patient.loc[patient["metric"].eq("bc")].pivot(index="patient", columns="factor", values="rate")
    patients = list(cell.index)
    x = np.arange(len(patients))
    noun = 100 * cell["noun identity"].to_numpy(float)
    position = 100 * cell["surface position"].to_numpy(float)
    for index, (left, right) in enumerate(zip(noun, position)):
        ax.plot([index - 0.12, index + 0.12], [left, right], color=LIGHT, lw=0.9, zorder=1)
    ax.scatter(x - 0.12, noun, s=18, color=NOUN, edgecolor=BLACK, linewidth=0.35, zorder=3)
    ax.scatter(x + 0.12, position, s=18, color=POSITION, edgecolor=BLACK, linewidth=0.35, zorder=3)
    ax.axhline(5, color=NEUTRAL, lw=0.8, ls="--")
    ax.set_xticks(x, patients, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 30)
    ax.set_ylabel("P < 0.05 units (%)")
    ax.set_title("Patient-level rates", loc="left")
    ax.text(
        1.45,
        29.2,
        f"5/5 position > noun\nP = {stats_out['patient_exact_signflip_p_one_sided']:.3f}",
        ha="left",
        va="top",
        fontsize=FONT_PT,
    )


def plot_panel_c(ax: plt.Axes, patient_decode: pd.DataFrame, stats_out: dict[str, float]) -> None:
    targets = ["noun identity", "voice", "surface position", "position across semantic roles"]
    labels = ["Noun\nID", "Voice", "Surf.\npos.", "Cross-\nrole\npos."]
    colors = [NOUN, VOICE, POSITION, POSITION_DARK]
    rng = np.random.default_rng(20260806)
    for index, (target, color) in enumerate(zip(targets, colors)):
        values = 100 * patient_decode.loc[patient_decode["target"].eq(target), "accuracy_minus_chance"].to_numpy(float)
        jitter = rng.uniform(-0.075, 0.075, len(values))
        ax.scatter(index + jitter, values, s=14, facecolor="white", edgecolor=color, linewidth=0.8, zorder=3)
        ax.plot([index - 0.18, index + 0.18], [values.mean(), values.mean()], color=color, lw=2.2, zorder=4)
    ax.axhline(0, color=NEUTRAL, lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(targets)), labels)
    ax.set_ylim(-17, 49)
    ax.set_ylabel("Accuracy − chance (percentage points)")
    ax.set_title("Population decoding", loc="left")
    ax.text(
        0.02,
        0.95,
        f"P = {stats_out['crossrole_position_exact_patient_signflip_p_one_sided']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_PT,
        color=POSITION_DARK,
    )
    ax.text(
        0.98,
        0.03,
        "Agent ↔ Patient",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_PT,
        color=NEUTRAL,
    )


def plot_supplementary_kimi_layer_decoding(ax: plt.Axes, layer_decode: pd.DataFrame) -> None:
    specs = [
        ("surface position", POSITION, "-", "position, held-out pair"),
        ("position across semantic roles", POSITION_DARK, "--", "position, cross-role"),
        ("noun identity across position", NOUN, ":", "noun identity, cross-position"),
    ]
    for target, color, linestyle, label in specs:
        cell = layer_decode.loc[layer_decode["target"].eq(target)]
        ax.plot(cell["level"], 100 * cell["accuracy"], color=color, lw=1.4, ls=linestyle, marker="o", ms=2.0, label=label)
    ax.axhline(50, color=NEUTRAL, lw=0.7, ls="--")
    ax.axhline(5, color=NOUN, lw=0.6, ls=":", alpha=0.7)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 103)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax.set_xlabel("Kimi hidden-state level")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title("Kimi position decoding", loc="left")
    ax.legend(loc="center right", fontsize=FONT_PT)


def plot_panel_d(ax: plt.Axes, group_item: pd.DataFrame, summary: dict[str, object]) -> None:
    order = {"initial": 0, "post_marker": 1}
    for _, cell in group_item.groupby("noun", sort=False):
        cell = cell.sort_values("surface_position", key=lambda values: values.map(order))
        ax.plot(cell["kimi_position_score"], cell["patient_equal_neural_projection_z"], color=LIGHT, lw=0.55, zorder=1)
    markers = {"agent": "o", "patient": "s"}
    colors = {"initial": INITIAL, "post_marker": POST}
    for semantic, marker in markers.items():
        for position, color in colors.items():
            cell = group_item.loc[
                group_item["thematic_role"].eq(semantic) & group_item["surface_position"].eq(position)
            ]
            ax.scatter(
                cell["kimi_position_score"],
                cell["patient_equal_neural_projection_z"],
                s=24,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
    x = group_item["kimi_position_score"].to_numpy(float)
    y = group_item["patient_equal_neural_projection_z"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, intercept + slope * xx, color=BLACK, lw=1.2, ls="--")
    ax.axhline(0, color=NEUTRAL, lw=0.6, ls=":")
    ax.axvline(0, color=NEUTRAL, lw=0.6, ls=":")
    ax.set_xlabel("Held-out Kimi position score")
    ax.set_ylabel("Held-out neural projection, 300–500 ms\n(patient-equal z score)")
    ax.set_title("Held-out Kimi–neural items", loc="left")
    ax.text(
        0.02,
        0.98,
        f"40 events: ρ = {summary['patient_equal_item_spearman_rho']:.2f}\ncontrol position: ρ = {summary['patient_equal_position_controlled_spearman_rho']:.2f}\n20 noun Δ: ρ = {summary['patient_equal_paired_delta_spearman_rho']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_PT,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": LIGHT},
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=BLACK, label="Agent noun"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=BLACK, label="Patient noun"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=INITIAL, markeredgecolor="white", label="sentence-initial"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=POST, markeredgecolor="white", label="post-marker"),
        ],
        loc="lower right",
        ncol=2,
        fontsize=FONT_PT,
    )


def plot_panel_e(
    ax: plt.Axes,
    patient_corr: pd.DataFrame,
    _strict_checks: pd.DataFrame,
) -> None:
    metrics = ["spearman_rho", "position_controlled_spearman_rho", "paired_delta_spearman_rho"]
    labels = ["Overall", "Control\npos.", "Within-\nnoun Δ"]
    colors = [POSITION_DARK, NEUTRAL, NOUN]
    x = np.arange(3)
    for patient, cell in patient_corr.groupby("patient", sort=True):
        row = cell.iloc[0]
        values = [float(row[metric]) for metric in metrics]
        ax.plot(x, values, color=PATIENT_COLORS.get(patient, LIGHT), lw=0.75, alpha=0.7)
        ax.scatter(x, values, s=13, color=PATIENT_COLORS.get(patient, LIGHT), edgecolor="white", linewidth=0.35, zorder=3)
    means = patient_corr[metrics].mean().to_numpy(float)
    for index, (value, color) in enumerate(zip(means, colors)):
        ax.plot([index - 0.16, index + 0.16], [value, value], color=color, lw=2.4)
        ax.text(index, value + 0.055, f"{value:.2f}", ha="center", fontsize=FONT_PT, color=color)
    ax.axhline(0, color=NEUTRAL, lw=0.7, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.32, 0.80)
    ax.set_ylabel("Item-level Spearman ρ")
    ax.set_title("Position-control summary", loc="left")

    ax.text(0.98, 0.04, "*exploratory; chance=50%", transform=ax.transAxes, ha="right", fontsize=FONT_PT, color=NEUTRAL)


def make_figure(
    pooled: pd.DataFrame,
    patient_unit: pd.DataFrame,
    unit_stats: dict[str, float],
    patient_decode: pd.DataFrame,
    neural_stats: dict[str, float],
    kfta_summary: dict[str, object],
    group_item: pd.DataFrame,
    patient_corr: pd.DataFrame,
    strict_checks: pd.DataFrame,
) -> None:
    configure_style()
    width_mm = 180.0
    height_mm = 113.5
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")

    # Physical-width contract: top-row panel allocations are exactly
    # 45 + 15 + 45 + 15 + 45 + 15 mm. Each panel keeps the left edge of the
    # previous 60-mm module and uses the final 15 mm as a readability gutter.
    # The lower row is 90 + 30 + 45 + 15 mm: panel d, a central gutter,
    # panel e aligned under panel c, and a trailing readability gutter.
    # Subfigures keep these allocations independent of the internal axis
    # margins required by labels and tick text.
    # Relative to the previous 143.5-mm layout, both physical panel rows are
    # exactly 15 mm shorter while the 5.519-mm inter-row gap is unchanged.
    top_height_mm = 63.47115384615384 - 15.0
    bottom_height_mm = 74.50961538461539 - 15.0
    row_gap_mm = 5.519230769230766
    row_hspace = row_gap_mm / ((top_height_mm + bottom_height_mm) / 2.0)
    top_row, bottom_row = fig.subfigures(
        2,
        1,
        height_ratios=(top_height_mm, bottom_height_mm),
        hspace=row_hspace,
    )
    top_cells = top_row.subfigures(
        1,
        6,
        width_ratios=(45, 15, 45, 15, 45, 15),
        wspace=0.0,
    )
    top_panels = (top_cells[0], top_cells[2], top_cells[4])
    top_widths_mm = [subfig.bbox.width / fig.dpi * 25.4 for subfig in top_panels]
    if not np.allclose(top_widths_mm, [45.0, 45.0, 45.0], atol=0.01):
        raise RuntimeError(f"Top-row panel widths are not 45 mm: {top_widths_mm}")
    bottom_cells = bottom_row.subfigures(
        1,
        4,
        width_ratios=(90, 30, 45, 15),
        wspace=0.0,
    )
    bottom_panels = (bottom_cells[0], bottom_cells[2])
    bottom_widths_mm = [subfig.bbox.width / fig.dpi * 25.4 for subfig in bottom_panels]
    if not np.allclose(bottom_widths_mm, [90.0, 45.0], atol=0.01):
        raise RuntimeError(f"Bottom-row panel widths are not 90/45 mm: {bottom_widths_mm}")

    ax_a = top_panels[0].subplots()
    ax_b = top_panels[1].subplots()
    ax_c = top_panels[2].subplots()
    ax_d = bottom_panels[0].subplots()
    ax_e = bottom_panels[1].subplots()
    for subfig in top_panels:
        subfig.subplots_adjust(left=0.24, right=0.96, top=0.86, bottom=0.20)
    top_panels[2].subplots_adjust(left=0.24, right=0.96, top=0.86, bottom=0.24)
    bottom_panels[0].subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.20)
    bottom_panels[1].subplots_adjust(left=0.29, right=0.95, top=0.86, bottom=0.20)

    plot_panel_a(ax_a, pooled)
    plot_panel_b(ax_b, patient_unit, unit_stats)
    plot_panel_c(ax_c, patient_decode, neural_stats)
    plot_panel_d(ax_d, group_item, kfta_summary)
    plot_panel_e(ax_e, patient_corr, strict_checks)
    for label, subfig, label_x in zip(
        ("a", "b", "c", "d", "e"),
        (top_panels[0], top_panels[1], top_panels[2], bottom_panels[0], bottom_panels[1]),
        (0.02, 0.02, -0.08, 0.02, 0.02),
    ):
        subfig.text(
            label_x,
            0.98,
            label,
            fontsize=FONT_PT,
            fontweight="bold",
            va="top",
            ha="left",
        )
    fig.text(
        0.015,
        0.012,
        "Neural analyses: usable-lenient units, 8 runs nested within 5 patients. Kimi–neural item panels are post-selected descriptive analyses.",
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
            FIG_DIR / f"Fig4_noun_identity_position_kimi.{suffix}",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def write_caption(
    unit_stats: dict[str, float],
    neural_stats: dict[str, float],
    kfta_summary: dict[str, object],
) -> None:
    caption = f"""Fig. 4 | Neural and Kimi correspondence is dominated by noun position rather than noun identity. a, Unit-level screening of noun identity and surface position in baseline-corrected and raw responses. Bars show uncorrected P < 0.05 rates, exact 95% binomial confidence intervals and the 5% null. Units are nested; pooled counts are descriptive. b, Patient-level baseline-corrected rates (n = 5 patients). Position exceeded noun identity in all patients (exact paired sign-flip, one-sided P = {unit_stats['patient_exact_signflip_p_one_sided']:.3f}). c, Patient decoding relative to chance (5% for 20-way noun identity; 50% for binary targets). Noun identity remained at chance. Position generalized between Agent and Patient noun classes with each semantic pair held out (patient-equal accuracy {100*neural_stats['crossrole_position_patient_equal_accuracy']:.1f}%; exact patient sign-flip, one-sided P = {neural_stats['crossrole_position_exact_patient_signflip_p_one_sided']:.3f}). Tests in b and c are unadjusted and exploratory (Bonferroni-adjusted P = 0.0625 each as one two-test family). d, Forty held-out noun events comparing Kimi position score with the 300–500-ms neural projection; lines join the same noun. Overall ρ = {kfta_summary['patient_equal_item_spearman_rho']:.2f}, position-controlled ρ = {kfta_summary['patient_equal_position_controlled_spearman_rho']:.2f} and within-noun position-difference ρ = {kfta_summary['patient_equal_paired_delta_spearman_rho']:.2f}. e, Patient-wise decomposition and strict held-out mapping checks. The Kimi level and neural window in d–e were selected from a previous map; these panels are descriptive, not independent significance tests. Together, the data support a shared surface-position/context component rather than reliable noun-specific neural–LLM mapping. Source data are provided in the accompanying files."""
    (OUT / "Fig4_caption.txt").write_text(caption + "\n", encoding="utf-8")


def write_summary(
    unit_stats: dict[str, float],
    neural_stats: dict[str, float],
    kfta_summary: dict[str, object],
    strict_checks: pd.DataFrame,
) -> None:
    result = {
        "figure_contract": {
            "core_conclusion": "Reliable neural and neural-Kimi effects are dominated by sentence-initial versus post-marker noun position; no reliable noun-identity enrichment or cross-position decoding was detected.",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "final_size_mm": [180, 113.5],
            "panel_allocations_mm": {"a": 45, "spacer_after_a": 15, "b": 45, "spacer_after_b": 15, "c": 45, "spacer_after_c": 15, "d": 90, "spacer_d_e": 30, "e": 45, "spacer_after_e": 15},
            "panel_row_heights_mm": {"a-c": 48.47115384615384, "d-e": 59.50961538461539},
            "font_size_pt": FONT_PT,
            "biological_replicates": "5 patients; 8 runs nested within patient",
            "unit_set": "usable_lenient",
        },
        "unit_screen": unit_stats,
        "neural_decoding": neural_stats,
        "kfta_usable_lenient": kfta_summary,
        "strict_mapping_checks": strict_checks.to_dict(orient="records"),
        "interpretation_boundary": [
            "Noun identity is the operational lexical/semantic identity test; failure to detect it is not proof that semantic information is absent.",
            "The position contrast remains confounded with serial position and preceding local context/function-word environment.",
            "Kimi contains lexical information, especially in early levels; the claim is that the shared neural-Kimi component is position-dominated, not that the language model contains only position.",
            "KFTA level 1 and the 300-500 ms window are post-selected and descriptive.",
            "The two new exact patient tests are exploratory and unadjusted; Bonferroni adjustment across them gives P=0.0625 for each.",
        ],
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checks = strict_checks.set_index("analysis")
    text = f"""# Fig. 4 结果与解释

## 一句话结论

本数据没有检出可靠的名词身份（词汇/语义身份）富集或跨位置解码；可泛化的神经信号主要区分“句首名词”与“标记词后名词”。Kimi 与神经活动的表面相关也主要由这一位置分离驱动，不支持稳定的名词逐项映射。

## 主结果

- 基线校正后，患者等权的名词身份单元筛查率为 {100*unit_stats['patient_equal_noun_rate']:.2f}%，位置率为 {100*unit_stats['patient_equal_position_rate']:.2f}%。五位患者均为位置高于名词身份，exact sign-flip 单侧 P={unit_stats['patient_exact_signflip_p_one_sided']:.4f}。
- 名词身份跨位置解码的患者等权准确率为 {100*neural_stats['noun_identity_patient_equal_accuracy']:.2f}%（机会水平 5%）。
- 位置解码的患者等权准确率为 {100*neural_stats['surface_position_patient_equal_accuracy']:.2f}%；训练 Agent 、测试 Patient 并反向交换的留一句对解码为 {100*neural_stats['crossrole_position_patient_equal_accuracy']:.2f}%（机会水平 50%，exact patient sign-flip 单侧 P={neural_stats['crossrole_position_exact_patient_signflip_p_one_sided']:.4f}）。
- 跨语义角色位置解码的另两种口径为：原始响应 {100*neural_stats['crossrole_position_raw_response_patient_equal_accuracy']:.2f}%（P={neural_stats['crossrole_position_raw_response_exact_patient_signflip_p_one_sided']:.4f}），名词前基线 {100*neural_stats['crossrole_position_pre_noun_baseline_patient_equal_accuracy']:.2f}%（P={neural_stats['crossrole_position_pre_noun_baseline_exact_patient_signflip_p_one_sided']:.4f}）。这表明位置信号不完全由单一基线校正步骤产生，但局部基线/上下文仍有贡献。
- usable-lenient 单元中，Kimi–neural 逐事件整体 Spearman ρ={kfta_summary['patient_equal_item_spearman_rho']:.3f}；控制位置后 ρ={kfta_summary['patient_equal_position_controlled_spearman_rho']:.3f}；20 个名词的位置差值相关 ρ={kfta_summary['patient_equal_paired_delta_spearman_rho']:.3f}。后两者是事后选择层和时间窗后的描述性结果。
- 更严格的预先定义检验未通过：控制后语法位移几何 ρ={checks.loc['controlled grammar-shift geometry','estimate']:.3f}, P={checks.loc['controlled grammar-shift geometry','patient_exact_p_one_sided']:.4f}；嵌套 neural→Kimi 位移映射准确率 {100*checks.loc['nested neural-to-Kimi shift mapping','estimate']:.2f}%, P={checks.loc['nested neural-to-Kimi shift mapping','patient_exact_p_one_sided']:.4f}。

## 论文中建议的表述

“Noun identity did not show reliable enrichment or cross-position decoding, whereas a population signal generalized sentence-initial versus post-marker noun position across Agent and Patient noun classes. Kimi hidden states robustly represented the same position contrast; however, item-wise neural–Kimi correspondence was substantially reduced after position control, indicating a shared position/context component rather than stable noun-specific alignment.”

## 边界

1. 请写“未检出可靠的名词身份编码”，不要写“语义完全不存在”。
2. 位置仍与串行位置、前导标记词以及局部上下文共变，不能直接称为纯层级句法编码。
3. Kimi 早层仍含有名词身份信息；准确说法是“与神经活动共享的任务相关成分主要是位置”，而不是“LLM 只学会了位置”。
4. 面板 b 和 c 的两个 exact P 为本轮新增的未校正探索性检验；若视为同一两检验家族，Bonferroni 校正后均为 0.0625。
"""
    (OUT / "analysis_and_interpretation_zh.md").write_text(text, encoding="utf-8")


def write_qa() -> None:
    outputs = [
        FIG_DIR / "Fig4_noun_identity_position_kimi.svg",
        FIG_DIR / "Fig4_noun_identity_position_kimi.pdf",
        FIG_DIR / "Fig4_noun_identity_position_kimi.png",
        FIG_DIR / "Fig4_noun_identity_position_kimi.tiff",
        OUT / "Fig4_caption.txt",
        OUT / "analysis_summary.json",
    ]
    qa = {
        "backend_exclusive": "Python/matplotlib",
        "outputs_exist": {path.name: path.exists() and path.stat().st_size > 0 for path in outputs},
        "svg_editable_text": False,
        "svg_all_text_8pt": False,
        "panel_e_middle_text_removed": False,
        "panel_e_ylim": [-0.32, 0.80],
        "svg_font_sizes_pt": [],
        "svg_size_mm": [],
        "pdf_truetype_fonttype": mpl.rcParams["pdf.fonttype"] == 42,
        "panel_labels": ["a", "b", "c", "d", "e"],
        "panel_allocations_mm": {"a": 45, "spacer_after_a": 15, "b": 45, "spacer_after_b": 15, "c": 45, "spacer_after_c": 15, "d": 90, "spacer_d_e": 30, "e": 45, "spacer_after_e": 15},
        "panel_row_heights_mm": {"a-c": 48.47115384615384, "d-e": 59.50961538461539},
        "panel_height_reduction_mm": 15,
        "top_panel_width_runtime_assertion": "passed at 45 mm for a, b and c",
        "bottom_panel_width_runtime_assertion": "passed at 90 mm for d and 45 mm for e",
        "font_size_pt": FONT_PT,
        "figure_size_inches": [180 / 25.4, 113.5 / 25.4],
        "figure_size_mm": [180, 113.5],
        "statistics_note": "Patients are biological replicates; pooled units are descriptive only.",
        "image_integrity": "No microscopy/image manipulation; all panels are programmatic plots from tabular source data.",
    }
    svg = FIG_DIR / "Fig4_noun_identity_position_kimi.svg"
    if svg.exists():
        text = svg.read_text(encoding="utf-8", errors="ignore")
        qa["svg_editable_text"] = "<text" in text and ("font-family" in text or "font:" in text)
        qa["panel_e_middle_text_removed"] = "Strict held-out checks" not in text
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
        and qa["svg_editable_text"]
        and qa["svg_all_text_8pt"]
        and qa["panel_e_middle_text_removed"]
        and qa["pdf_truetype_fonttype"]
    )
    (QA_DIR / "qa_report.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute-kfta",
        action="store_true",
        help="Recompute the usable-lenient Kimi-feature-triggered neural analysis.",
    )
    args = parser.parse_args()
    ensure_dirs()
    pooled, patient_unit, unit_stats = load_unit_screen()
    patient_decode, _sensitivity, neural_stats = analyze_neural_decoding()
    kfta_summary, group_item, patient_corr = ensure_usable_kfta(args.recompute_kfta)
    strict_checks = load_strict_mapping_checks()
    make_figure(
        pooled,
        patient_unit,
        unit_stats,
        patient_decode,
        neural_stats,
        kfta_summary,
        group_item,
        patient_corr,
        strict_checks,
    )
    write_caption(unit_stats, neural_stats, kfta_summary)
    write_summary(unit_stats, neural_stats, kfta_summary, strict_checks)
    write_qa()
    print(
        json.dumps(
            {
                "figure": str(FIG_DIR / "Fig4_noun_identity_position_kimi.png"),
                "caption": str(OUT / "Fig4_caption.txt"),
                "summary": str(OUT / "analysis_and_interpretation_zh.md"),
                "qa": str(QA_DIR / "qa_report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
