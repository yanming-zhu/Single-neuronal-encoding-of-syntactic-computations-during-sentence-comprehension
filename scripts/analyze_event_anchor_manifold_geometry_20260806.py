#!/usr/bin/env python3
"""Compare noun-, construction-marker-, and verb-aligned Rank-4 geometry.

This exploratory analysis preserves the frozen held-sentence-pair folds,
training-only role centering/scaling/PCA/orthogonal Procrustes fitting, and
patient-level exact inference.  It adds two event anchors:

* construction marker: 把/给/将 for active and 被/让/叫 for passive;
* lexical verb: onset of the matched sentence-final verb.

The marker- and verb-aligned responses are sentence-pair-level observations.
They are duplicated across the two thematic-role rows only to preserve the
frozen fold and retrieval implementation; this duplication does not add
independent observations or alter patient-level inference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
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
import make_integrated_manifold_fig4_20260806 as integrated


ROOT = base.ROOT
DATA_ROOT = base.DATA_ROOT
REPORT_ROOT = ROOT / "reports/event_anchor_manifold_geometry_20260806"
OUTPUT_ROOT = DATA_ROOT / "combined_event_anchor_manifold_geometry_20260806"
MANIFOLD_DATA_ROOT = DATA_ROOT / "combined_active_passive_manifold_geometry_20260806"
NOUN_EVENTS_PATH = (
    DATA_ROOT / "combined_noun_role_structure_encoding/noun_role_event_metadata.csv"
)
VERB_EVENTS_PATH = (
    DATA_ROOT / "combined_verb_active_passive_unit_activity/verb_event_metadata.csv"
)
MARKER_ONSETS_PATH = (
    DATA_ROOT
    / "gz001/processed/ActPasSynSentence/event_related_active_passive"
    / "active_passive_function_word_audio_onsets.csv"
)

ANCHORS = ("noun", "marker", "verb")
NEW_ANCHORS = ("marker", "verb")
RANK = 4
WINDOWS_MS = shared.DYNAMIC_WINDOWS_MS
ANCHOR_LABELS = {
    "noun": "Noun",
    "marker": "Marker",
    "verb": "Verb",
}
ANCHOR_COLORS = {
    "noun": "#7651A8",
    "marker": "#D88C17",
    "verb": "#2B9A77",
}


def event_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.18,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def specification_paths(unit_set: str) -> tuple[Path, Path, Path]:
    if unit_set == "good_only":
        return REPORT_ROOT, OUTPUT_ROOT, MANIFOLD_DATA_ROOT
    return (
        REPORT_ROOT / "sensitivity_usable_lenient",
        OUTPUT_ROOT / "sensitivity_usable_lenient",
        MANIFOLD_DATA_ROOT / "sensitivity_usable_lenient",
    )


def pair_id_from_sound_file(value: str) -> int:
    return base.pair_id_from_sound_file(value)


def build_marker_event_table() -> pd.DataFrame:
    noun_events = pd.read_csv(NOUN_EVENTS_PATH)
    sentence_start = noun_events[noun_events["role"] == "subject"].copy()
    offsets = pd.read_csv(MARKER_ONSETS_PATH)[
        [
            "sound_file",
            "condition",
            "target_word",
            "target_onset_sec_from_sentence_onset",
            "onset_source",
        ]
    ].copy()
    offsets = offsets.rename(
        columns={
            "condition": "sentence_condition",
            "target_word": "anchor_word",
            "onset_source": "anchor_onset_source",
        }
    )
    event = sentence_start.merge(
        offsets,
        on=["sound_file", "sentence_condition"],
        how="left",
        validate="many_to_one",
    )
    if event["target_onset_sec_from_sentence_onset"].isna().any():
        raise RuntimeError("Marker onset merge produced missing values")
    event["voice"] = event["sentence_condition"]
    event["pair_id"] = event["sound_file"].map(pair_id_from_sound_file).astype(int)
    event["anchor"] = "marker"
    event["anchor_event_bin_1000Hz"] = event["event_bin_1000Hz"].astype(int) + np.rint(
        event["target_onset_sec_from_sentence_onset"].astype(float) * 1000
    ).astype(int)
    event["anchor_onset_sec_from_sentence_onset"] = event[
        "target_onset_sec_from_sentence_onset"
    ].astype(float)
    return event[
        [
            "patient",
            "run",
            "trial_index",
            "sound_file",
            "voice",
            "pair_id",
            "anchor",
            "anchor_word",
            "anchor_event_bin_1000Hz",
            "anchor_onset_sec_from_sentence_onset",
            "anchor_onset_source",
        ]
    ].drop_duplicates()


def build_verb_event_table() -> pd.DataFrame:
    event = pd.read_csv(VERB_EVENTS_PATH).copy()
    event["voice"] = event["sentence_condition"]
    event["pair_id"] = event["verb_id"].astype(int)
    event["anchor"] = "verb"
    event["anchor_word"] = event["verb"].astype(str)
    event["anchor_event_bin_1000Hz"] = event["event_bin_1000Hz"].astype(int)
    event["anchor_onset_sec_from_sentence_onset"] = event[
        "verb_onset_sec_from_sentence_onset"
    ].astype(float)
    event["anchor_onset_source"] = event["onset_source"].astype(str)
    return event[
        [
            "patient",
            "run",
            "trial_index",
            "sound_file",
            "voice",
            "pair_id",
            "anchor",
            "anchor_word",
            "anchor_event_bin_1000Hz",
            "anchor_onset_sec_from_sentence_onset",
            "anchor_onset_source",
        ]
    ].drop_duplicates()


def event_tables() -> dict[str, pd.DataFrame]:
    tables = {
        "marker": build_marker_event_table(),
        "verb": build_verb_event_table(),
    }
    for anchor, table in tables.items():
        counts = table.groupby(["run", "voice"])["pair_id"].nunique()
        if not (counts == 10).all():
            raise RuntimeError(f"Incomplete {anchor} event table:\n{counts.to_string()}")
    return tables


def extract_anchor_tensors(
    run: base.RunData,
    event_table: pd.DataFrame,
    unit_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return half-open 100-ms bins and inclusive cumulative 0--T windows."""
    archive_path = base.unit_archive_path(run.patient, run.run, unit_set)
    with np.load(archive_path, allow_pickle=True) as archive:
        firing_rate = np.asarray(archive["firing_rate_hz"], dtype=float)
        archive_unit_ids = np.asarray(archive["unit_ids"], dtype=int)
    lookup = {int(unit_id): index for index, unit_id in enumerate(archive_unit_ids)}
    firing_rate = firing_rate[[lookup[int(unit_id)] for unit_id in run.unit_ids]]

    events = event_table[event_table["run"] == run.run].copy()
    if len(events) != 20:
        raise RuntimeError(f"Expected 20 anchor events for {run.run}, found {len(events)}")
    dynamic = {
        voice: np.full(
            (len(run.item_meta), len(WINDOWS_MS), len(run.unit_ids)), np.nan, dtype=float
        )
        for voice in base.VOICES
    }
    cumulative = {
        voice: np.full(
            (len(run.item_meta), len(WINDOWS_MS), len(run.unit_ids)), np.nan, dtype=float
        )
        for voice in base.VOICES
    }
    audit_rows: list[dict[str, object]] = []
    for voice in base.VOICES:
        for _, item in run.item_meta.iterrows():
            row = events[
                (events["voice"] == voice)
                & (events["pair_id"] == int(item["pair_id"]))
            ]
            if len(row) != 1:
                raise RuntimeError(
                    f"Expected one {voice} pair {item['pair_id']} event for {run.run}"
                )
            event = row.iloc[0]
            event_bin = int(event["anchor_event_bin_1000Hz"])
            if event_bin - 500 < 0 or event_bin + 801 > firing_rate.shape[1]:
                raise RuntimeError(f"Anchor window out of bounds for {run.run}: {event_bin}")
            baseline = firing_rate[:, event_bin - 500 : event_bin].mean(axis=1)
            for window_index, (start, stop) in enumerate(WINDOWS_MS):
                response = firing_rate[:, event_bin + start : event_bin + stop].mean(axis=1)
                dynamic[voice][int(item["item_index"]), window_index] = response - baseline
                cumulative_response = firing_rate[:, event_bin : event_bin + stop + 1].mean(
                    axis=1
                )
                cumulative[voice][int(item["item_index"]), window_index] = (
                    cumulative_response - baseline
                )
            audit_rows.append(
                {
                    "patient": run.patient,
                    "run": run.run,
                    "voice": voice,
                    "pair_id": int(item["pair_id"]),
                    "thematic_role": item["thematic_role"],
                    "anchor": event["anchor"],
                    "anchor_word": event["anchor_word"],
                    "anchor_event_bin_1000Hz": event_bin,
                    "anchor_onset_sec_from_sentence_onset": event[
                        "anchor_onset_sec_from_sentence_onset"
                    ],
                    "anchor_onset_source": event["anchor_onset_source"],
                    "local_baseline_start_ms": -500,
                    "local_baseline_stop_ms": 0,
                }
            )
    arrays = (
        dynamic["active"],
        dynamic["passive"],
        cumulative["active"],
        cumulative["passive"],
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError(f"Non-finite anchor tensor for {run.run}")
    return (*arrays, pd.DataFrame(audit_rows))


def static_rank4_analysis(
    runs: list[base.RunData],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    folds = pd.concat(
        [manifold.procrustes_cv(run, ranks=(RANK,)) for run in runs], ignore_index=True
    )
    run_summary = folds.groupby(["patient", "run"], as_index=False).agg(
        retrieval_margin=("retrieval_margin", "mean"),
        two_by_two_accuracy=("two_by_two_accuracy", "mean"),
    )
    patient_summary = run_summary.groupby("patient", as_index=False).mean(numeric_only=True)
    stats: dict[str, dict[str, float]] = {}
    for column, endpoint, reference in (
        ("retrieval_margin", "margin", 0.0),
        ("two_by_two_accuracy", "accuracy", 0.5),
    ):
        values = patient_summary[column].to_numpy()
        sign = base.exact_sign_flip(values, reference)
        ci = base.bootstrap_patient_ci(
            values, 10000, base.stable_seed("event-anchor-static", endpoint)
        )
        stats[endpoint] = {
            "patient_mean": float(values.mean()),
            "patient_median": float(np.median(values)),
            "bootstrap_ci_low": ci[0],
            "bootstrap_ci_high": ci[1],
            "positive_patients": int(np.sum(values > reference)),
            "exact_signflip_p_greater": sign["p_greater"],
            "exact_signflip_p_two_sided": sign["p_two_sided"],
        }
    return folds, run_summary, patient_summary, stats


def frozen_permutations(manifold_data_dir: Path) -> list[np.ndarray]:
    null = pd.read_csv(
        manifold_data_dir / "procrustes_synchronized_pair_permutation_null.csv"
    )
    null = null[null["rank"] == RANK].sort_values("permutation_index")
    return [np.asarray(json.loads(value), dtype=int) for value in null["permutation_json"]]


def item_permutation_analysis(
    runs: list[base.RunData],
    permutations: list[np.ndarray],
    observed_stats: dict[str, dict[str, float]],
    anchor: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    progress_every = max(1, len(permutations) // 10)
    for permutation_index, permutation in enumerate(permutations):
        folds = pd.concat(
            [
                manifold.procrustes_cv(
                    run, ranks=(RANK,), passive_pair_permutation=permutation
                )
                for run in runs
            ],
            ignore_index=True,
        )
        run_summary = folds.groupby(["patient", "run"], as_index=False).agg(
            retrieval_margin=("retrieval_margin", "mean"),
            two_by_two_accuracy=("two_by_two_accuracy", "mean"),
        )
        patient = run_summary.groupby("patient", as_index=False).mean(numeric_only=True)
        rows.append(
            {
                "anchor": anchor,
                "permutation_index": permutation_index,
                "patient_equal_retrieval_margin": patient["retrieval_margin"].mean(),
                "patient_equal_accuracy": patient["two_by_two_accuracy"].mean(),
                "permutation_json": json.dumps(permutation.tolist()),
            }
        )
        if (permutation_index + 1) % progress_every == 0:
            print(
                f"{anchor} synchronized permutations: {permutation_index + 1}/{len(permutations)}",
                flush=True,
            )
    null = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for column, endpoint in (
        ("patient_equal_retrieval_margin", "margin"),
        ("patient_equal_accuracy", "accuracy"),
    ):
        observed = observed_stats[endpoint]["patient_mean"]
        summary[f"{endpoint}_observed"] = observed
        summary[f"{endpoint}_item_permutation_p_greater"] = float(
            (np.sum(null[column].to_numpy() >= observed - 1e-12) + 1)
            / (len(null) + 1)
        )
    return null, summary


def exact_anchor_family_patient_p(
    patient_all: pd.DataFrame,
    value_col: str,
    reference: float,
) -> dict[str, float]:
    pivot = patient_all.pivot(index="patient", columns="anchor", values=value_col).reindex(
        columns=ANCHORS
    )
    values = pivot.to_numpy() - reference
    observed = values.mean(axis=0)
    null_max = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(pivot)):
        null_max.append(np.max((values * np.asarray(signs)[:, None]).mean(axis=0)))
    null_max = np.asarray(null_max)
    return {
        anchor: float(np.mean(null_max >= observed[index] - 1e-12))
        for index, anchor in enumerate(ANCHORS)
    }


def add_joint_time_correction(
    matrices_by_anchor: dict[str, dict[str, np.ndarray]],
    cell_stats_by_anchor: dict[str, pd.DataFrame],
) -> None:
    patient_order = [str(value) for value in matrices_by_anchor["noun"]["patients"]]
    for anchor in ANCHORS:
        if [str(value) for value in matrices_by_anchor[anchor]["patients"]] != patient_order:
            raise RuntimeError(f"Patient order mismatch for {anchor}")
    margin = np.stack(
        [matrices_by_anchor[anchor]["margin_patient_matrices"] for anchor in ANCHORS],
        axis=1,
    )
    accuracy = np.stack(
        [
            matrices_by_anchor[anchor]["accuracy_excess_patient_matrices"]
            for anchor in ANCHORS
        ],
        axis=1,
    )
    _, _, margin_p = shared.exact_patient_max_stat(
        margin.reshape(margin.shape[0], len(ANCHORS) * len(WINDOWS_MS), len(WINDOWS_MS))
    )
    _, _, accuracy_p = shared.exact_patient_max_stat(
        accuracy.reshape(
            accuracy.shape[0], len(ANCHORS) * len(WINDOWS_MS), len(WINDOWS_MS)
        )
    )
    margin_p = margin_p.reshape(len(ANCHORS), len(WINDOWS_MS), len(WINDOWS_MS))
    accuracy_p = accuracy_p.reshape(len(ANCHORS), len(WINDOWS_MS), len(WINDOWS_MS))
    for anchor_index, anchor in enumerate(ANCHORS):
        frame = cell_stats_by_anchor[anchor].sort_values(
            ["active_window_index", "passive_window_index"]
        ).copy()
        frame["margin_exact_anchor_time_maxstat_p_fwer_greater"] = margin_p[
            anchor_index
        ].ravel()
        frame["accuracy_exact_anchor_time_maxstat_p_fwer_greater"] = accuracy_p[
            anchor_index
        ].ravel()
        cell_stats_by_anchor[anchor] = frame


def add_joint_cumulative_correction(
    matrices_by_anchor: dict[str, dict[str, np.ndarray]],
    stats_by_anchor: dict[str, pd.DataFrame],
) -> None:
    patient_order = [str(value) for value in matrices_by_anchor["noun"]["patients"]]
    for anchor in ANCHORS:
        if [str(value) for value in matrices_by_anchor[anchor]["patients"]] != patient_order:
            raise RuntimeError(f"Cumulative patient order mismatch for {anchor}")
    margin = np.stack(
        [matrices_by_anchor[anchor]["margin_patient_matrix"] for anchor in ANCHORS],
        axis=1,
    )
    accuracy = np.stack(
        [
            matrices_by_anchor[anchor]["accuracy_excess_patient_matrix"]
            for anchor in ANCHORS
        ],
        axis=1,
    )
    _, _, margin_p = shared.exact_patient_max_stat(
        margin.reshape(margin.shape[0], len(ANCHORS) * len(WINDOWS_MS), 1)
    )
    _, _, accuracy_p = shared.exact_patient_max_stat(
        accuracy.reshape(accuracy.shape[0], len(ANCHORS) * len(WINDOWS_MS), 1)
    )
    margin_p = margin_p.reshape(len(ANCHORS), len(WINDOWS_MS))
    accuracy_p = accuracy_p.reshape(len(ANCHORS), len(WINDOWS_MS))
    for anchor_index, anchor in enumerate(ANCHORS):
        frame = stats_by_anchor[anchor].sort_values("cumulative_window_index").copy()
        frame["margin_exact_anchor_time_maxstat_p_fwer_greater"] = margin_p[anchor_index]
        frame["accuracy_exact_anchor_time_maxstat_p_fwer_greater"] = accuracy_p[
            anchor_index
        ]
        stats_by_anchor[anchor] = frame


def load_noun_results(
    manifold_data_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
]:
    cells = pd.read_csv(manifold_data_dir / "fig4_rank4_dynamic_cell_stats.csv")
    with np.load(
        manifold_data_dir / "fig4_rank4_dynamic_patient_matrices.npz", allow_pickle=True
    ) as archive:
        dynamic_matrices = {key: np.asarray(archive[key]) for key in archive.files}
    cumulative_stats = pd.read_csv(
        manifold_data_dir / "fig4_rank4_cumulative_window_stats.csv"
    )
    with np.load(
        manifold_data_dir / "fig4_rank4_cumulative_patient_matrices.npz",
        allow_pickle=True,
    ) as archive:
        cumulative_matrices = {key: np.asarray(archive[key]) for key in archive.files}
    static_patient = pd.read_csv(manifold_data_dir / "static_model_patient_summary.csv")
    static_patient = static_patient[
        (static_patient["model"] == "Procrustes") & (static_patient["rank"] == RANK)
    ][["patient", "retrieval_margin", "two_by_two_accuracy"]].copy()
    static_patient["anchor"] = "noun"
    null = pd.read_csv(
        manifold_data_dir / "procrustes_synchronized_pair_permutation_null.csv"
    )
    null = null[null["rank"] == RANK][
        [
            "permutation_index",
            "patient_equal_retrieval_margin",
            "patient_equal_accuracy",
            "permutation_json",
        ]
    ].copy()
    null["anchor"] = "noun"
    return (
        cells,
        dynamic_matrices,
        cumulative_stats,
        cumulative_matrices,
        static_patient,
        null,
    )


def add_joint_item_correction(
    null_all: pd.DataFrame,
    static_summary: pd.DataFrame,
) -> pd.DataFrame:
    output = static_summary.copy()
    for column, endpoint, observed_col in (
        ("patient_equal_retrieval_margin", "margin", "patient_mean_retrieval_margin"),
        ("patient_equal_accuracy", "accuracy", "patient_mean_accuracy"),
    ):
        pivot = null_all.pivot(index="permutation_index", columns="anchor", values=column)
        pivot = pivot.reindex(columns=ANCHORS)
        null_max = pivot.max(axis=1).to_numpy()
        p_values = {}
        for anchor in ANCHORS:
            observed = float(output.loc[output["anchor"] == anchor, observed_col].iloc[0])
            p_values[anchor] = float(
                (np.sum(null_max >= observed - 1e-12) + 1) / (len(null_max) + 1)
            )
        output[f"{endpoint}_item_anchor_maxstat_p_fwer_greater"] = output["anchor"].map(
            p_values
        )
    return output


def static_comparison_table(
    patient_all: pd.DataFrame,
    permutation_summaries: dict[str, dict[str, float]],
) -> pd.DataFrame:
    margin_family = exact_anchor_family_patient_p(
        patient_all, "retrieval_margin", 0.0
    )
    accuracy_family = exact_anchor_family_patient_p(
        patient_all, "two_by_two_accuracy", 0.5
    )
    rows: list[dict[str, object]] = []
    for anchor in ANCHORS:
        frame = patient_all[patient_all["anchor"] == anchor]
        margin_sign = base.exact_sign_flip(frame["retrieval_margin"], 0.0)
        accuracy_sign = base.exact_sign_flip(frame["two_by_two_accuracy"], 0.5)
        rows.append(
            {
                "anchor": anchor,
                "patient_mean_retrieval_margin": frame["retrieval_margin"].mean(),
                "margin_positive_patients": int(np.sum(frame["retrieval_margin"] > 0)),
                "margin_patient_p_greater": margin_sign["p_greater"],
                "margin_patient_p_two_sided": margin_sign["p_two_sided"],
                "margin_patient_anchor_maxstat_p_fwer_greater": margin_family[anchor],
                "margin_item_permutation_p_greater": permutation_summaries[anchor][
                    "margin_item_permutation_p_greater"
                ],
                "patient_mean_accuracy": frame["two_by_two_accuracy"].mean(),
                "accuracy_above_chance_patients": int(
                    np.sum(frame["two_by_two_accuracy"] > 0.5)
                ),
                "accuracy_patient_p_greater": accuracy_sign["p_greater"],
                "accuracy_patient_p_two_sided": accuracy_sign["p_two_sided"],
                "accuracy_patient_anchor_maxstat_p_fwer_greater": accuracy_family[
                    anchor
                ],
                "accuracy_item_permutation_p_greater": permutation_summaries[anchor][
                    "accuracy_item_permutation_p_greater"
                ],
            }
        )
    return pd.DataFrame(rows)


def patient_anchor_plot(
    ax: plt.Axes,
    patient_all: pd.DataFrame,
    summary: pd.DataFrame,
    value_col: str,
    reference: float,
    ylabel: str,
    panel: str,
) -> None:
    patients = sorted(patient_all["patient"].unique())
    for patient in patients:
        frame = patient_all[patient_all["patient"] == patient].set_index("anchor").reindex(
            ANCHORS
        )
        values = frame[value_col].to_numpy() - reference
        color = integrated.PATIENT_COLORS.get(patient, "#777777")
        ax.plot(range(len(ANCHORS)), values, color=color, alpha=0.55, lw=0.8)
        ax.scatter(range(len(ANCHORS)), values, color=color, s=24, edgecolor="white", lw=0.4)
    means = [
        patient_all[patient_all["anchor"] == anchor][value_col].mean() - reference
        for anchor in ANCHORS
    ]
    ax.scatter(range(len(ANCHORS)), means, marker="_", s=210, color="black", lw=2.0)
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    labels = []
    for anchor in ANCHORS:
        row = summary[summary["anchor"] == anchor].iloc[0]
        p = row[
            "margin_patient_p_greater"
            if value_col == "retrieval_margin"
            else "accuracy_patient_p_greater"
        ]
        labels.append(f"{ANCHOR_LABELS[anchor]}\nP={p:.3f}")
    ax.set_xticks(range(len(ANCHORS)), labels)
    ax.set_ylabel(ylabel)
    ax.set_title("Static 0–800 ms patient endpoints")
    event_panel_label(ax, panel)


def item_null_plot(
    ax: plt.Axes,
    null_all: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    for anchor in ANCHORS:
        values = null_all[null_all["anchor"] == anchor][
            "patient_equal_retrieval_margin"
        ]
        observed = summary.loc[
            summary["anchor"] == anchor, "patient_mean_retrieval_margin"
        ].iloc[0]
        p = summary.loc[
            summary["anchor"] == anchor, "margin_item_permutation_p_greater"
        ].iloc[0]
        ax.hist(
            values,
            bins=30,
            histtype="step",
            density=True,
            lw=1.1,
            color=ANCHOR_COLORS[anchor],
            alpha=0.85,
        )
        ax.axvline(
            observed,
            color=ANCHOR_COLORS[anchor],
            lw=1.8,
            label=f"{ANCHOR_LABELS[anchor]} P={p:.3f}",
        )
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set(xlabel="Patient-equal Rank-4 margin", ylabel="Permutation density")
    ax.set_title("Synchronized item-label controls")
    ax.legend(frameon=False, loc="upper left")
    event_panel_label(ax, "c")


def anchor_heatmap(
    ax: plt.Axes,
    stats: pd.DataFrame,
    endpoint: str,
    anchor: str,
    panel: str,
) -> None:
    if endpoint == "margin":
        value_col = "patient_mean_retrieval_margin"
        within_col = "margin_exact_maxstat_p_fwer_greater"
        joint_col = "margin_exact_anchor_time_maxstat_p_fwer_greater"
        center = 0.0
        colorbar_label = "Margin"
    else:
        value_col = "patient_mean_accuracy"
        within_col = "accuracy_exact_maxstat_p_fwer_greater"
        joint_col = "accuracy_exact_anchor_time_maxstat_p_fwer_greater"
        center = 0.5
        colorbar_label = "Accuracy minus 0.5"
    values = stats.pivot(
        index="active_window_index", columns="passive_window_index", values=value_col
    ).to_numpy()
    p_within = stats.pivot(
        index="active_window_index", columns="passive_window_index", values=within_col
    ).to_numpy()
    p_joint = stats.pivot(
        index="active_window_index", columns="passive_window_index", values=joint_col
    ).to_numpy()
    centered = values - center
    limit = max(0.03, float(np.max(np.abs(centered))))
    image = ax.imshow(
        centered,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    labels = [f"{start}-{stop}" for start, stop in WINDOWS_MS]
    ax.set_xticks(range(8), labels, rotation=45, ha="right")
    ax.set_yticks(range(8), labels)
    ax.set_xlabel("Passive window (ms)")
    ax.set_ylabel("Active window (ms)")
    n_within = int(np.sum(p_within < 0.05))
    n_joint = int(np.sum(p_joint < 0.05))
    ax.set_title(
        f"{ANCHOR_LABELS[anchor]}-aligned {endpoint}\nwithin/joint FWER: {n_within}/64, {n_joint}/192"
    )
    for active_index in range(8):
        for passive_index in range(8):
            marker = "*" if p_joint[active_index, passive_index] < 0.05 else ""
            ax.text(
                passive_index,
                active_index,
                f"{values[active_index, passive_index]:.2f}{marker}",
                ha="center",
                va="center",
                fontsize=3.7,
            )
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(labelsize=5.6)
    event_panel_label(ax, panel)


def anchor_temporal_plot(
    ax: plt.Axes,
    dynamic_matrices: dict[str, np.ndarray],
    cumulative_matrices: dict[str, np.ndarray],
    cumulative_stats: pd.DataFrame,
    anchor: str,
    panel: str,
) -> None:
    diagonal = np.vstack(
        [np.diag(matrix) for matrix in dynamic_matrices["margin_patient_matrices"]]
    )
    cumulative = cumulative_matrices["margin_patient_matrix"]
    x = np.arange(100, 801, 100)
    patients = [str(value) for value in dynamic_matrices["patients"]]
    for patient_index, patient in enumerate(patients):
        color = integrated.PATIENT_COLORS.get(patient, "#777777")
        ax.plot(x, diagonal[patient_index], color=color, alpha=0.17, lw=0.7, ls="--")
        ax.plot(x, cumulative[patient_index], color=color, alpha=0.24, lw=0.8)
    ax.plot(
        x,
        diagonal.mean(axis=0),
        color="#D88C17",
        marker="o",
        ms=2.8,
        lw=1.6,
        ls="--",
        label="Matched 100-ms bins",
    )
    ax.plot(
        x,
        cumulative.mean(axis=0),
        color="#7651A8",
        marker="o",
        ms=2.8,
        lw=1.8,
        label="Cumulative 0–T",
    )
    ax.axhline(0, color="#777777", lw=0.8, ls=":")
    within = int(np.sum(cumulative_stats["margin_exact_maxstat_p_fwer_greater"] < 0.05))
    joint = int(
        np.sum(
            cumulative_stats["margin_exact_anchor_time_maxstat_p_fwer_greater"] < 0.05
        )
    )
    ax.set_xticks(x, [str(value) for value in x])
    ax.set(xlabel="Window end T (ms)", ylabel="Rank-4 retrieval margin")
    ax.set_title(
        f"{ANCHOR_LABELS[anchor]} short versus cumulative\nwithin/joint cumulative FWER: {within}/8, {joint}/24"
    )
    ax.legend(frameon=False, loc="best")
    event_panel_label(ax, panel)


def make_figure(
    report_dir: Path,
    static_patient_all: pd.DataFrame,
    static_summary: pd.DataFrame,
    null_all: pd.DataFrame,
    cell_stats: dict[str, pd.DataFrame],
    dynamic_matrices: dict[str, dict[str, np.ndarray]],
    cumulative_stats: dict[str, pd.DataFrame],
    cumulative_matrices: dict[str, dict[str, np.ndarray]],
) -> None:
    integrated.configure_style()
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 8.65))
    patient_anchor_plot(
        axes[0, 0],
        static_patient_all,
        static_summary,
        "retrieval_margin",
        0.0,
        "Rank-4 retrieval margin",
        "a",
    )
    patient_anchor_plot(
        axes[0, 1],
        static_patient_all,
        static_summary,
        "two_by_two_accuracy",
        0.5,
        "Accuracy minus 0.5",
        "b",
    )
    item_null_plot(axes[0, 2], null_all, static_summary)
    anchor_heatmap(axes[1, 0], cell_stats["marker"], "margin", "marker", "d")
    anchor_heatmap(axes[1, 1], cell_stats["marker"], "accuracy", "marker", "e")
    anchor_temporal_plot(
        axes[1, 2],
        dynamic_matrices["marker"],
        cumulative_matrices["marker"],
        cumulative_stats["marker"],
        "marker",
        "f",
    )
    anchor_heatmap(axes[2, 0], cell_stats["verb"], "margin", "verb", "g")
    anchor_heatmap(axes[2, 1], cell_stats["verb"], "accuracy", "verb", "h")
    anchor_temporal_plot(
        axes[2, 2],
        dynamic_matrices["verb"],
        cumulative_matrices["verb"],
        cumulative_stats["verb"],
        "verb",
        "i",
    )
    fig.suptitle(
        "Rank-4 geometry across noun, construction-marker and verb anchors",
        fontsize=9.2,
        fontweight="bold",
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.978), h_pad=2.2, w_pad=1.9)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "Fig_5_event_anchor_rank4_manifold_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(
    report_dir: Path,
    unit_set: str,
    static_summary: pd.DataFrame,
    cell_stats: dict[str, pd.DataFrame],
    cumulative_stats: dict[str, pd.DataFrame],
    onset_audit: pd.DataFrame,
) -> None:
    rows = []
    for anchor in ANCHORS:
        static = static_summary[static_summary["anchor"] == anchor].iloc[0]
        cells = cell_stats[anchor]
        cumulative = cumulative_stats[anchor]
        best = cumulative.loc[cumulative["patient_mean_retrieval_margin"].idxmax()]
        rows.append(
            {
                "anchor": anchor,
                "static_margin": static.patient_mean_retrieval_margin,
                "margin_patient_p": static.margin_patient_p_greater,
                "margin_patient_anchor_fwer_p": static.margin_patient_anchor_maxstat_p_fwer_greater,
                "margin_item_p": static.margin_item_permutation_p_greater,
                "margin_item_anchor_fwer_p": static.margin_item_anchor_maxstat_p_fwer_greater,
                "static_accuracy": static.patient_mean_accuracy,
                "accuracy_patient_p": static.accuracy_patient_p_greater,
                "accuracy_patient_anchor_fwer_p": static.accuracy_patient_anchor_maxstat_p_fwer_greater,
                "accuracy_item_p": static.accuracy_item_permutation_p_greater,
                "accuracy_item_anchor_fwer_p": static.accuracy_item_anchor_maxstat_p_fwer_greater,
                "dynamic_margin_within_fwer_cells": int(
                    np.sum(cells["margin_exact_maxstat_p_fwer_greater"] < 0.05)
                ),
                "dynamic_margin_joint_fwer_cells": int(
                    np.sum(
                        cells["margin_exact_anchor_time_maxstat_p_fwer_greater"] < 0.05
                    )
                ),
                "dynamic_accuracy_within_fwer_cells": int(
                    np.sum(cells["accuracy_exact_maxstat_p_fwer_greater"] < 0.05)
                ),
                "dynamic_accuracy_joint_fwer_cells": int(
                    np.sum(
                        cells["accuracy_exact_anchor_time_maxstat_p_fwer_greater"] < 0.05
                    )
                ),
                "cumulative_margin_within_fwer_windows": int(
                    np.sum(cumulative["margin_exact_maxstat_p_fwer_greater"] < 0.05)
                ),
                "cumulative_margin_joint_fwer_windows": int(
                    np.sum(
                        cumulative[
                            "margin_exact_anchor_time_maxstat_p_fwer_greater"
                        ]
                        < 0.05
                    )
                ),
                "best_cumulative_window_ms": best.cumulative_window_ms,
                "best_cumulative_margin": best.patient_mean_retrieval_margin,
                "best_cumulative_margin_within_fwer_p": best.margin_exact_maxstat_p_fwer_greater,
                "best_cumulative_margin_joint_fwer_p": best.margin_exact_anchor_time_maxstat_p_fwer_greater,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(report_dir / "event_anchor_summary.csv", index=False)

    table_lines = [
        "| 对齐点 | 静态margin | 患者P / 三anchor校正P | 刺激P / 三anchor校正P | 静态accuracy | 动态margin/accuracy通过格 | 累积最佳窗口 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        table_lines.append(
            f"| {ANCHOR_LABELS[row.anchor]} | {row.static_margin:.4f} | {row.margin_patient_p:.4f} / {row.margin_patient_anchor_fwer_p:.4f} | {row.margin_item_p:.4f} / {row.margin_item_anchor_fwer_p:.4f} | {row.static_accuracy:.3f} | {row.dynamic_margin_within_fwer_cells}/64 / {row.dynamic_accuracy_within_fwer_cells}/64 | {row.best_cumulative_window_ms}，margin={row.best_cumulative_margin:.4f}，FWER P={row.best_cumulative_margin_within_fwer_p:.4f} |"
        )
    marker_words = ", ".join(
        sorted(onset_audit[onset_audit["anchor"] == "marker"]["anchor_word"].unique())
    )
    verb_sources = (
        onset_audit[onset_audit["anchor"] == "verb"]
        .drop_duplicates(["run", "voice", "pair_id", "anchor"])[
            "anchor_onset_source"
        ]
        .value_counts()
    )
    verb_source_text = "; ".join(
        f"{source}={count}" for source, count in verb_sources.items()
    )
    report = f"""# 构式标记与动词对齐的Rank-4 manifold分析

## 结果

{chr(10).join(table_lines)}

## 方法口径

- unit规格：`{unit_set}`；7 runs、5 patients，患者为独立推断单位。
- 三个对齐点均使用Rank-4正交Procrustes、完整留出两个sentence pairs和训练折内预处理。
- 动态分析为8×8个100 ms半开窗口；每个anchor内部控制64格，并额外对三个anchors合计192格做患者exact max-stat。
- 累积分析使用0–T闭区间；每个anchor内部控制8个窗口，并额外控制三个anchors合计24个窗口。
- 三个anchor使用同一组500次同步句对置换，因此可做跨anchor的刺激层max-stat校正。
- Marker实际包括：{marker_words}。只取字面“把/被”会仅剩4个sentence pairs，无法维持当前Rank-4设计。
- Verb onset来源：{verb_source_text}。

## 解释边界

- Marker和Verb对齐后，每个句子的Agent/Patient两行共享同一个群体响应；角色行的重复只是为了保持冻结的sentence-pair折与检索实现，不增加样本量。相应指标是sentence-pair retrieval，不是该时刻的名词身份检索。
- Marker后0–800 ms包含后续名词，并可能延伸至动词；长窗结果不能称为“把/被神经反应”。
- Verb前−500–0 ms基线包含此前的名词和构式标记；结果表示相对于局部句子上下文的变化。
- 主动/被动与标记词身份、声学和句式共线。即使marker对齐出现效应，也不能识别抽象语态。
"""
    (report_dir / "REPORT_zh.md").write_text(report, encoding="utf-8")
    caption = (
        "Fig. 5 | Event-anchor dependence of cross-construction Rank-4 geometry. "
        "a,b, Patient-level held-pair retrieval margin and two-by-two accuracy above chance "
        "for 0–800 ms responses aligned to noun, construction-marker or lexical-verb onset. "
        "Coloured points and lines denote patients (n = 5); black bars denote patient means. "
        "c, Observed margins relative to 500 synchronized sentence-pair permutations. "
        "d,e, Marker-aligned Rank-4 margin and accuracy time-generalization matrices. "
        "f, Marker-aligned matched 100-ms and cumulative 0-to-T margins. g–i, Corresponding "
        "verb-aligned analyses. Each map was fit only on training sentence pairs. Within-anchor "
        "max-stat corrections cover 64 time cells or eight cumulative windows; joint corrections "
        "cover three anchors and 192 cells or 24 cumulative windows. Construction markers comprise "
        "把/给/将 in active sentences and 被/让/叫 in passive sentences. Marker and verb responses "
        "are sentence-pair-level observations and do not constitute noun-identity retrieval at those anchors."
    )
    (report_dir / "Fig5_caption_en.txt").write_text(caption + "\n", encoding="utf-8")


def validate(
    static_folds: dict[str, pd.DataFrame],
    dynamic_folds: dict[str, pd.DataFrame],
    cumulative_folds: dict[str, pd.DataFrame],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for anchor in NEW_ANCHORS:
        checks = {
            "static_rows": int(len(static_folds[anchor])),
            "static_rows_expected": 7 * 45,
            "dynamic_rows": int(len(dynamic_folds[anchor])),
            "dynamic_rows_expected": 7 * 64 * 45,
            "cumulative_rows": int(len(cumulative_folds[anchor])),
            "cumulative_rows_expected": 7 * 8 * 45,
            "all_finite": bool(
                np.isfinite(
                    pd.concat(
                        [
                            static_folds[anchor][
                                ["retrieval_margin", "two_by_two_accuracy"]
                            ],
                            dynamic_folds[anchor][
                                ["retrieval_margin", "two_by_two_accuracy"]
                            ],
                            cumulative_folds[anchor][
                                ["retrieval_margin", "two_by_two_accuracy"]
                            ],
                        ],
                        ignore_index=True,
                    ).to_numpy()
                ).all()
            ),
        }
        checks["passed"] = bool(
            checks["static_rows"] == checks["static_rows_expected"]
            and checks["dynamic_rows"] == checks["dynamic_rows_expected"]
            and checks["cumulative_rows"] == checks["cumulative_rows_expected"]
            and checks["all_finite"]
        )
        if not checks["passed"]:
            raise RuntimeError(f"Validation failed for {anchor}: {checks}")
        payload[anchor] = checks
    payload["passed"] = True
    return payload


def run_specification(unit_set: str, artifact_threshold: float | None) -> None:
    report_dir, output_dir, manifold_data_dir = specification_paths(unit_set)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, available_artifact_runs = base.load_filtered_metrics(unit_set, artifact_threshold)
    base_runs, audit = base.build_run_data(metrics, available_artifact_runs, unit_set)
    if len(base_runs) != 7 or len({run.patient for run in base_runs}) != 5:
        raise RuntimeError(f"Unexpected included data:\n{audit.to_string(index=False)}")
    tables = event_tables()
    permutations = frozen_permutations(manifold_data_dir)
    if len(permutations) != 500:
        raise RuntimeError(f"Expected 500 frozen permutations, found {len(permutations)}")

    static_folds: dict[str, pd.DataFrame] = {}
    static_patients: dict[str, pd.DataFrame] = {}
    static_stats: dict[str, dict[str, dict[str, float]]] = {}
    permutation_nulls: dict[str, pd.DataFrame] = {}
    permutation_summaries: dict[str, dict[str, float]] = {}
    dynamic_folds: dict[str, pd.DataFrame] = {}
    cell_stats: dict[str, pd.DataFrame] = {}
    dynamic_matrices: dict[str, dict[str, np.ndarray]] = {}
    cumulative_folds: dict[str, pd.DataFrame] = {}
    cumulative_stats: dict[str, pd.DataFrame] = {}
    cumulative_matrices: dict[str, dict[str, np.ndarray]] = {}
    onset_audits: list[pd.DataFrame] = []

    for anchor in NEW_ANCHORS:
        anchor_runs: list[base.RunData] = []
        dynamic_parts: list[pd.DataFrame] = []
        cumulative_parts: list[pd.DataFrame] = []
        for run_index, run in enumerate(base_runs, start=1):
            active_dynamic, passive_dynamic, active_cumulative, passive_cumulative, onset = (
                extract_anchor_tensors(run, tables[anchor], unit_set)
            )
            onset_audits.append(onset)
            anchor_run = replace(
                run,
                active=active_cumulative[:, -1],
                passive=passive_cumulative[:, -1],
            )
            anchor_runs.append(anchor_run)
            dynamic_parts.append(
                integrated.dynamic_procrustes_cv(run, active_dynamic, passive_dynamic)
            )
            cumulative_parts.append(
                integrated.cumulative_procrustes_cv(
                    run, active_cumulative, passive_cumulative
                )
            )
            print(
                f"{unit_set} {anchor}: static/dynamic/cumulative {run_index}/{len(base_runs)}",
                flush=True,
            )
        folds, run_summary, patient_summary, stats = static_rank4_analysis(anchor_runs)
        static_folds[anchor] = folds
        static_patients[anchor] = patient_summary.assign(anchor=anchor)
        static_stats[anchor] = stats
        null, perm_summary = item_permutation_analysis(
            anchor_runs, permutations, stats, anchor
        )
        permutation_nulls[anchor] = null
        permutation_summaries[anchor] = perm_summary

        dynamic_folds[anchor] = pd.concat(dynamic_parts, ignore_index=True)
        dynamic_run, dynamic_patient, cells, matrices = manifold.dynamic_inference(
            dynamic_folds[anchor]
        )
        cell_stats[anchor] = cells
        dynamic_matrices[anchor] = matrices
        cumulative_folds[anchor] = pd.concat(cumulative_parts, ignore_index=True)
        cumulative_run, cumulative_patient, cumulative_window, cumulative_matrix = (
            integrated.summarize_cumulative(cumulative_folds[anchor])
        )
        cumulative_stats[anchor] = cumulative_window
        cumulative_matrices[anchor] = cumulative_matrix

        anchor_dir = output_dir / anchor
        anchor_dir.mkdir(parents=True, exist_ok=True)
        folds.to_csv(anchor_dir / "static_rank4_cv_folds.csv", index=False)
        run_summary.to_csv(anchor_dir / "static_rank4_run_summary.csv", index=False)
        patient_summary.to_csv(anchor_dir / "static_rank4_patient_summary.csv", index=False)
        null.to_csv(anchor_dir / "static_rank4_synchronized_pair_null.csv", index=False)
        dynamic_folds[anchor].to_csv(anchor_dir / "dynamic_rank4_cv_folds.csv", index=False)
        dynamic_run.to_csv(anchor_dir / "dynamic_rank4_run_summary.csv", index=False)
        dynamic_patient.to_csv(anchor_dir / "dynamic_rank4_patient_summary.csv", index=False)
        cumulative_folds[anchor].to_csv(
            anchor_dir / "cumulative_rank4_cv_folds.csv", index=False
        )
        cumulative_run.to_csv(anchor_dir / "cumulative_rank4_run_summary.csv", index=False)
        cumulative_patient.to_csv(
            anchor_dir / "cumulative_rank4_patient_summary.csv", index=False
        )

    (
        noun_cells,
        noun_dynamic_matrices,
        noun_cumulative_stats,
        noun_cumulative_matrices,
        noun_static_patient,
        noun_null,
    ) = load_noun_results(manifold_data_dir)
    cell_stats["noun"] = noun_cells
    dynamic_matrices["noun"] = noun_dynamic_matrices
    cumulative_stats["noun"] = noun_cumulative_stats
    cumulative_matrices["noun"] = noun_cumulative_matrices
    static_patients["noun"] = noun_static_patient
    permutation_nulls["noun"] = noun_null
    manifold_summary = json.loads((manifold_data_dir / "summary.json").read_text(encoding="utf-8"))
    permutation_summaries["noun"] = {
        "margin_observed": manifold_summary["procrustes_permutation"][
            "rank4_margin_observed"
        ],
        "margin_item_permutation_p_greater": manifold_summary[
            "procrustes_permutation"
        ]["rank4_margin_item_permutation_p_greater"],
        "accuracy_observed": manifold_summary["procrustes_permutation"][
            "rank4_accuracy_observed"
        ],
        "accuracy_item_permutation_p_greater": manifold_summary[
            "procrustes_permutation"
        ]["rank4_accuracy_item_permutation_p_greater"],
    }

    add_joint_time_correction(dynamic_matrices, cell_stats)
    add_joint_cumulative_correction(cumulative_matrices, cumulative_stats)
    static_patient_all = pd.concat(
        [static_patients[anchor] for anchor in ANCHORS], ignore_index=True
    )
    static_summary = static_comparison_table(
        static_patient_all, permutation_summaries
    )
    null_all = pd.concat([permutation_nulls[anchor] for anchor in ANCHORS], ignore_index=True)
    static_summary = add_joint_item_correction(null_all, static_summary)

    onset_audit = pd.concat(onset_audits, ignore_index=True)
    onset_audit.to_csv(output_dir / "event_anchor_inclusion_audit.csv", index=False)
    static_patient_all.to_csv(output_dir / "static_rank4_patient_anchor_comparison.csv", index=False)
    static_summary.to_csv(output_dir / "static_rank4_anchor_inference.csv", index=False)
    null_all.to_csv(output_dir / "static_rank4_anchor_permutation_null.csv", index=False)
    for anchor in ANCHORS:
        anchor_dir = output_dir / anchor
        anchor_dir.mkdir(parents=True, exist_ok=True)
        cell_stats[anchor].to_csv(anchor_dir / "dynamic_rank4_cell_stats.csv", index=False)
        cumulative_stats[anchor].to_csv(
            anchor_dir / "cumulative_rank4_window_stats.csv", index=False
        )
        np.savez_compressed(
            anchor_dir / "dynamic_rank4_patient_matrices.npz",
            **dynamic_matrices[anchor],
        )
        np.savez_compressed(
            anchor_dir / "cumulative_rank4_patient_matrices.npz",
            **cumulative_matrices[anchor],
        )

    validation = validate(static_folds, dynamic_folds, cumulative_folds)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure(
        report_dir,
        static_patient_all,
        static_summary,
        null_all,
        cell_stats,
        dynamic_matrices,
        cumulative_stats,
        cumulative_matrices,
    )
    write_report(
        report_dir,
        unit_set,
        static_summary,
        cell_stats,
        cumulative_stats,
        onset_audit,
    )


def postprocess_specification(unit_set: str) -> None:
    report_dir, output_dir, _ = specification_paths(unit_set)
    static_patient_all = pd.read_csv(
        output_dir / "static_rank4_patient_anchor_comparison.csv"
    )
    static_summary = pd.read_csv(output_dir / "static_rank4_anchor_inference.csv")
    null_all = pd.read_csv(output_dir / "static_rank4_anchor_permutation_null.csv")
    onset_audit = pd.read_csv(output_dir / "event_anchor_inclusion_audit.csv")
    cell_stats: dict[str, pd.DataFrame] = {}
    dynamic_matrices: dict[str, dict[str, np.ndarray]] = {}
    cumulative_stats: dict[str, pd.DataFrame] = {}
    cumulative_matrices: dict[str, dict[str, np.ndarray]] = {}
    for anchor in ANCHORS:
        anchor_dir = output_dir / anchor
        cell_stats[anchor] = pd.read_csv(anchor_dir / "dynamic_rank4_cell_stats.csv")
        cumulative_stats[anchor] = pd.read_csv(
            anchor_dir / "cumulative_rank4_window_stats.csv"
        )
        with np.load(anchor_dir / "dynamic_rank4_patient_matrices.npz", allow_pickle=True) as archive:
            dynamic_matrices[anchor] = {
                key: np.asarray(archive[key]) for key in archive.files
            }
        with np.load(
            anchor_dir / "cumulative_rank4_patient_matrices.npz", allow_pickle=True
        ) as archive:
            cumulative_matrices[anchor] = {
                key: np.asarray(archive[key]) for key in archive.files
            }
    make_figure(
        report_dir,
        static_patient_all,
        static_summary,
        null_all,
        cell_stats,
        dynamic_matrices,
        cumulative_stats,
        cumulative_matrices,
    )
    write_report(
        report_dir,
        unit_set,
        static_summary,
        cell_stats,
        cumulative_stats,
        onset_audit,
    )


def write_combined_report() -> None:
    frames = []
    for unit_set in ("good_only", "usable_lenient"):
        report_dir, _, _ = specification_paths(unit_set)
        path = report_dir / "event_anchor_summary.csv"
        if not path.exists():
            return
        frame = pd.read_csv(path)
        frame["unit_set"] = unit_set
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_ROOT / "event_anchor_specification_comparison.csv", index=False)
    lines = [
        "# Event-anchor manifold：两套unit规格综合结果",
        "",
        "| unit规格 | anchor | 静态margin | 患者P / anchor校正P | 刺激P / anchor校正P | accuracy | 动态margin/accuracy通过格 | 累积最佳窗口 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined.itertuples(index=False):
        lines.append(
            f"| {row.unit_set} | {ANCHOR_LABELS[row.anchor]} | {row.static_margin:.4f} | {row.margin_patient_p:.4f} / {row.margin_patient_anchor_fwer_p:.4f} | {row.margin_item_p:.4f} / {row.margin_item_anchor_fwer_p:.4f} | {row.static_accuracy:.3f} | {int(row.dynamic_margin_within_fwer_cells)}/64 / {int(row.dynamic_accuracy_within_fwer_cells)}/64 | {row.best_cumulative_window_ms}，P={row.best_cumulative_margin_within_fwer_p:.4f} |"
        )
    lines.extend(
        [
            "",
            "三种anchor的unit规格结果来自高度重叠的unit集合，属于敏感性分析，不是独立重复。最终解释应同时满足患者层、刺激置换、跨anchor校正和unit规格一致性。",
        ]
    )
    (REPORT_ROOT / "COMBINED_REPORT_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-set",
        choices=("good_only", "usable_lenient", "both"),
        default="both",
    )
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--postprocess-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unit_sets = ("good_only", "usable_lenient") if args.unit_set == "both" else (args.unit_set,)
    artifact_threshold = None if args.artifact_threshold < 0 else args.artifact_threshold
    for unit_set in unit_sets:
        if args.postprocess_existing:
            postprocess_specification(unit_set)
        else:
            run_specification(unit_set, artifact_threshold)
    write_combined_report()


if __name__ == "__main__":
    main()
