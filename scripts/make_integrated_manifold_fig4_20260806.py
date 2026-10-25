#!/usr/bin/env python3
"""Integrate static Rank-4 manifold evidence with time-resolved Fig. 4.

The integrated figure uses the same training-fold-only orthogonal Procrustes
pipeline for both the 0--800 ms averaged response and the dynamic analyses.
Patients are the independent inferential units.  The 8 x 8 time-generalization
grid and the eight cumulative 0--T endpoints are controlled separately with
exhaustive patient-level max-stat sign flips.
"""

from __future__ import annotations

import argparse
import json
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
import analyze_active_passive_manifold_geometry_20260806 as manifold
import analyze_active_passive_shared_latent_geometry_20260806 as shared


ROOT = base.ROOT
DATA_ROOT = base.DATA_ROOT
REPORT_ROOT = ROOT / "reports/active_passive_manifold_geometry_20260806"
DATA_OUTPUT_ROOT = DATA_ROOT / "combined_active_passive_manifold_geometry_20260806"
RANK = 4
WINDOWS_MS = shared.DYNAMIC_WINDOWS_MS
PATIENT_COLORS = {
    "gz001": "#4C78A8",
    "gz002": "#F58518",
    "gz003": "#54A24B",
    "gz004": "#B279A2",
    "gz005": "#E45756",
}
PURPLE = "#7651A8"
ORANGE = "#D88C17"
BLUE = "#3E78B2"
GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"


def specification_paths(unit_set: str) -> tuple[Path, Path]:
    if unit_set == "good_only":
        return REPORT_ROOT, DATA_OUTPUT_ROOT
    return REPORT_ROOT / "sensitivity_usable_lenient", DATA_OUTPUT_ROOT / "sensitivity_usable_lenient"


def dynamic_procrustes_cv(
    run: base.RunData,
    active: np.ndarray,
    passive: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all 8 x 8 window pairs with a fold-trained Rank-4 map."""
    rows: list[dict[str, object]] = []
    for active_index in range(len(WINDOWS_MS)):
        for passive_index in range(len(WINDOWS_MS)):
            x = active[:, active_index]
            y = passive[:, passive_index]
            for held_pairs, train, test in base.outer_folds(run.item_meta):
                train_meta = run.item_meta.loc[train].reset_index(drop=True)
                test_meta = run.item_meta.loc[test].reset_index(drop=True)
                x_test, y_test, _ = manifold.fit_procrustes_latent(
                    x[train],
                    y[train],
                    x[test],
                    y[test],
                    train_meta,
                    test_meta,
                    RANK,
                )
                rows.append(
                    {
                        "patient": run.patient,
                        "run": run.run,
                        "rank": RANK,
                        "model": "orthogonal Procrustes",
                        "active_window_index": active_index,
                        "passive_window_index": passive_index,
                        "held_pair_1": held_pairs[0],
                        "held_pair_2": held_pairs[1],
                        **shared.retrieval_metrics(x_test, y_test, test_meta),
                    }
                )
    return pd.DataFrame(rows)


def extract_cumulative_tensor(
    run: base.RunData,
    frame: pd.DataFrame,
    unit_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract cumulative 0--T responses using the frozen static convention.

    The source metric defines the response mask as 0 <= t <= T.  Inclusive
    upper endpoints are therefore required for the 0--800 ms cumulative value
    to reproduce the static matrix exactly.  The adjacent 100 ms dynamic bins
    remain half-open and are not repurposed for this calculation.
    """
    archive_path = base.unit_archive_path(run.patient, run.run, unit_set)
    with np.load(archive_path, allow_pickle=True) as archive:
        firing_rate = np.asarray(archive["firing_rate_hz"], dtype=float)
        archive_unit_ids = np.asarray(archive["unit_ids"], dtype=int)
    lookup = {int(unit_id): index for index, unit_id in enumerate(archive_unit_ids)}
    firing_rate = firing_rate[[lookup[int(unit_id)] for unit_id in run.unit_ids]]
    event_rows = frame.drop_duplicates(
        ["voice", "pair_id", "thematic_role", "noun", "trial_index"]
    )[["voice", "pair_id", "thematic_role", "noun", "trial_index", "event_bin_1000Hz"]]
    output = {
        voice: np.full(
            (len(run.item_meta), len(WINDOWS_MS), len(run.unit_ids)),
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
            for window_index, (_, stop) in enumerate(WINDOWS_MS):
                response = firing_rate[:, event_bin : event_bin + stop + 1].mean(axis=1)
                output[voice][int(item["item_index"]), window_index] = response - baseline
    if not np.isfinite(output["active"]).all() or not np.isfinite(output["passive"]).all():
        raise RuntimeError(f"Cumulative extraction produced missing values for {run.run}")
    if not np.allclose(output["active"][:, -1], run.active, atol=1e-4, rtol=1e-7):
        raise RuntimeError(f"Cumulative/static active mismatch for {run.run}")
    if not np.allclose(output["passive"][:, -1], run.passive, atol=1e-4, rtol=1e-7):
        raise RuntimeError(f"Cumulative/static passive mismatch for {run.run}")
    # Use the frozen CSV-backed matrices at 800 ms after confirming equivalence;
    # this makes the final cumulative endpoint numerically identical to Fig. 3.
    output["active"][:, -1] = run.active
    output["passive"][:, -1] = run.passive
    return output["active"], output["passive"]


def cumulative_procrustes_cv(
    run: base.RunData,
    active_cumulative: np.ndarray,
    passive_cumulative: np.ndarray,
) -> pd.DataFrame:
    """Evaluate matched cumulative 0--T windows with Rank-4 Procrustes."""
    rows: list[dict[str, object]] = []
    for window_index, (_, stop) in enumerate(WINDOWS_MS):
        x = active_cumulative[:, window_index]
        y = passive_cumulative[:, window_index]
        for held_pairs, train, test in base.outer_folds(run.item_meta):
            train_meta = run.item_meta.loc[train].reset_index(drop=True)
            test_meta = run.item_meta.loc[test].reset_index(drop=True)
            x_test, y_test, _ = manifold.fit_procrustes_latent(
                x[train],
                y[train],
                x[test],
                y[test],
                train_meta,
                test_meta,
                RANK,
            )
            rows.append(
                {
                    "patient": run.patient,
                    "run": run.run,
                    "rank": RANK,
                    "model": "orthogonal Procrustes",
                    "cumulative_window_index": window_index,
                    "cumulative_start_ms": 0,
                    "cumulative_stop_ms": stop,
                    "held_pair_1": held_pairs[0],
                    "held_pair_2": held_pairs[1],
                    **shared.retrieval_metrics(x_test, y_test, test_meta),
                }
            )
    return pd.DataFrame(rows)


def summarize_cumulative(
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    run_summary = folds.groupby(
        ["patient", "run", "cumulative_window_index", "cumulative_start_ms", "cumulative_stop_ms"],
        as_index=False,
    ).agg(
        retrieval_margin=("retrieval_margin", "mean"),
        two_by_two_accuracy=("two_by_two_accuracy", "mean"),
    )
    patient_summary = run_summary.groupby(
        ["patient", "cumulative_window_index", "cumulative_start_ms", "cumulative_stop_ms"],
        as_index=False,
    ).mean(numeric_only=True)
    patients = sorted(patient_summary["patient"].unique())
    margin = np.vstack(
        [
            patient_summary[patient_summary["patient"] == patient]
            .sort_values("cumulative_window_index")["retrieval_margin"]
            .to_numpy()
            for patient in patients
        ]
    )
    accuracy_excess = np.vstack(
        [
            patient_summary[patient_summary["patient"] == patient]
            .sort_values("cumulative_window_index")["two_by_two_accuracy"]
            .to_numpy()
            - 0.5
            for patient in patients
        ]
    )
    margin_observed, margin_null_max, margin_fwer = shared.exact_patient_max_stat(
        margin[:, :, None]
    )
    accuracy_observed, accuracy_null_max, accuracy_fwer = shared.exact_patient_max_stat(
        accuracy_excess[:, :, None]
    )
    rows: list[dict[str, object]] = []
    for window_index, (_, stop) in enumerate(WINDOWS_MS):
        margin_sign = base.exact_sign_flip(margin[:, window_index], 0.0)
        accuracy_sign = base.exact_sign_flip(accuracy_excess[:, window_index], 0.0)
        rows.append(
            {
                "cumulative_window_index": window_index,
                "cumulative_window_ms": f"0-{stop}",
                "patient_mean_retrieval_margin": margin_observed[window_index, 0],
                "margin_positive_patients": int(np.sum(margin[:, window_index] > 0)),
                "margin_exact_signflip_p_greater": margin_sign["p_greater"],
                "margin_exact_maxstat_p_fwer_greater": margin_fwer[window_index, 0],
                "patient_mean_accuracy": accuracy_observed[window_index, 0] + 0.5,
                "accuracy_above_chance_patients": int(
                    np.sum(accuracy_excess[:, window_index] > 0)
                ),
                "accuracy_exact_signflip_p_greater": accuracy_sign["p_greater"],
                "accuracy_exact_maxstat_p_fwer_greater": accuracy_fwer[window_index, 0],
            }
        )
    matrices = {
        "patients": np.asarray(patients, dtype=object),
        "margin_patient_matrix": margin,
        "accuracy_excess_patient_matrix": accuracy_excess,
        "margin_observed": margin_observed[:, 0],
        "margin_null_max": margin_null_max,
        "margin_fwer": margin_fwer[:, 0],
        "accuracy_observed_excess": accuracy_observed[:, 0],
        "accuracy_null_max": accuracy_null_max,
        "accuracy_fwer": accuracy_fwer[:, 0],
    }
    return run_summary, patient_summary, pd.DataFrame(rows), matrices


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.17,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def patient_endpoint_plot(
    ax: plt.Axes,
    static_patient: pd.DataFrame,
    static_group: pd.DataFrame,
) -> None:
    rank4 = static_patient[
        (static_patient["model"] == "Procrustes") & (static_patient["rank"] == RANK)
    ].copy()
    patients = sorted(rank4["patient"].unique())
    endpoint_values = {
        "Margin": rank4.set_index("patient")["retrieval_margin"],
        "Accuracy\nminus 0.5": rank4.set_index("patient")["two_by_two_accuracy"] - 0.5,
    }
    for x, (label, values) in enumerate(endpoint_values.items()):
        ordered = values.reindex(patients)
        for offset, patient, value in zip(
            np.linspace(-0.12, 0.12, len(patients)), patients, ordered
        ):
            ax.scatter(
                x + offset,
                value,
                s=26,
                color=PATIENT_COLORS.get(patient, PURPLE),
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        ax.scatter(x, ordered.mean(), marker="_", s=210, color="black", linewidth=2.0, zorder=4)
    ax.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_xticks([0, 1], endpoint_values.keys())
    ax.set_ylabel("Held-pair effect above chance")
    group = static_group.set_index("metric")
    p_margin = group.loc["procrustes_rank4_retrieval_margin", "exact_signflip_p_greater"]
    p_accuracy = group.loc["procrustes_rank4_accuracy", "exact_signflip_p_greater"]
    ax.set_title(f"Static Rank-4 rigid alignment\npatient P={p_margin:.3f} / {p_accuracy:.3f}")
    panel_label(ax, "b")


def spectrum_plot(ax: plt.Axes, spectrum: pd.DataFrame) -> None:
    patient = spectrum.groupby(["patient", "view", "dimension"], as_index=False)[
        "cumulative_fraction"
    ].mean()
    styles = {
        "active": (BLUE, "Active", "-"),
        "passive": (ORANGE, "Passive", "-"),
        "shared_cross_covariance": ("#2B9A77", "Cross-view", "--"),
    }
    for view, (color, label, linestyle) in styles.items():
        sub = patient[(patient["view"] == view) & (patient["dimension"] <= 8)]
        pivot = sub.pivot(index="patient", columns="dimension", values="cumulative_fraction")
        for _, row in pivot.iterrows():
            ax.plot(pivot.columns, row, color=color, alpha=0.16, lw=0.7, ls=linestyle)
        ax.plot(pivot.columns, pivot.mean(axis=0), color=color, lw=1.8, ls=linestyle, label=label)
    cross_rank4 = patient[
        (patient["view"] == "shared_cross_covariance") & (patient["dimension"] == 4)
    ]["cumulative_fraction"].mean()
    ax.axvline(4, color=GRAY, ls=":", lw=0.8)
    ax.set(xlabel="Retained dimensions", ylabel="Cumulative spectral fraction", ylim=(0, 1.03))
    ax.set_xticks([1, 2, 4, 6, 8])
    ax.legend(frameon=False, loc="lower right", handlelength=2.2)
    ax.set_title(f"Low-rank cross-view spectrum\nfirst 4 dimensions = {cross_rank4:.3f}")
    panel_label(ax, "a")


def item_null_plot(
    ax: plt.Axes,
    permutation_null: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    null = permutation_null[permutation_null["rank"] == RANK][
        "patient_equal_retrieval_margin"
    ]
    observed = float(summary["procrustes_permutation"]["rank4_margin_observed"])
    p_item = float(
        summary["procrustes_permutation"]["rank4_margin_item_permutation_p_greater"]
    )
    ax.hist(null, bins=26, color=LIGHT_GRAY, edgecolor="white", linewidth=0.4)
    ax.axvline(observed, color=PURPLE, lw=2.0, label="Matched pairs")
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    ax.set(xlabel="Patient-equal Rank-4 margin", ylabel="Synchronized permutations")
    ax.set_title(f"Item-label control\npermutation P={p_item:.3f}")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "c")


def dynamic_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    p_fwer: np.ndarray,
    center: float,
    title: str,
    colorbar_label: str,
    panel: str,
) -> None:
    centered = values - center
    limit = max(0.03, float(np.nanmax(np.abs(centered))))
    image = ax.imshow(
        centered,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
        interpolation="nearest",
    )
    labels = [f"{start}-{stop}" for start, stop in WINDOWS_MS]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Passive noun window (ms)")
    ax.set_ylabel("Active noun window (ms)")
    n_sig = int(np.sum(p_fwer < 0.05))
    ax.set_title(f"{title}\nFWER-positive cells: {n_sig}/64")
    for active_index in range(len(labels)):
        for passive_index in range(len(labels)):
            marker = "*" if p_fwer[active_index, passive_index] < 0.05 else ""
            ax.text(
                passive_index,
                active_index,
                f"{values[active_index, passive_index]:.2f}{marker}",
                ha="center",
                va="center",
                fontsize=4.0,
                color="black",
            )
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(labelsize=5.8)
    panel_label(ax, panel)


def temporal_summary_plot(
    ax: plt.Axes,
    dynamic_matrices: dict[str, np.ndarray],
    cumulative_matrices: dict[str, np.ndarray],
    cumulative_stats: pd.DataFrame,
) -> None:
    patients = [str(value) for value in dynamic_matrices["patients"]]
    diagonal = np.vstack(
        [np.diag(matrix) for matrix in dynamic_matrices["margin_patient_matrices"]]
    )
    cumulative = cumulative_matrices["margin_patient_matrix"]
    x = np.arange(100, 801, 100)
    for patient_index, patient in enumerate(patients):
        color = PATIENT_COLORS.get(patient, GRAY)
        ax.plot(x, diagonal[patient_index], color=color, alpha=0.20, lw=0.75, ls="--")
        ax.plot(x, cumulative[patient_index], color=color, alpha=0.28, lw=0.85)
    ax.plot(
        x,
        diagonal.mean(axis=0),
        color=ORANGE,
        marker="o",
        ms=3.0,
        lw=1.7,
        ls="--",
        label="Matched 100-ms bins",
    )
    ax.plot(
        x,
        cumulative.mean(axis=0),
        color=PURPLE,
        marker="o",
        ms=3.0,
        lw=2.0,
        label="Cumulative 0-T",
    )
    ax.axhline(0, color=GRAY, lw=0.8, ls=":")
    for row in cumulative_stats.itertuples(index=False):
        if row.margin_exact_maxstat_p_fwer_greater < 0.05:
            ax.text(
                int(str(row.cumulative_window_ms).split("-")[1]),
                row.patient_mean_retrieval_margin,
                "*",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    n_cumulative_sig = int(
        np.sum(cumulative_stats["margin_exact_maxstat_p_fwer_greater"] < 0.05)
    )
    ax.set_xticks(x, [str(value) for value in x])
    ax.set(xlabel="Window end T after noun onset (ms)", ylabel="Rank-4 retrieval margin")
    ax.set_title(f"Short-window versus accumulated signal\ncumulative FWER: {n_cumulative_sig}/8")
    ax.legend(frameon=False, loc="best")
    panel_label(ax, "f")


def make_figure(
    report_dir: Path,
    data_dir: Path,
    cell_stats: pd.DataFrame,
    dynamic_matrices: dict[str, np.ndarray],
    cumulative_stats: pd.DataFrame,
    cumulative_matrices: dict[str, np.ndarray],
) -> None:
    configure_style()
    spectrum = pd.read_csv(data_dir / "intrinsic_spectrum.csv")
    static_patient = pd.read_csv(data_dir / "static_model_patient_summary.csv")
    static_group = pd.read_csv(data_dir / "static_model_group_inference.csv")
    permutation_null = pd.read_csv(data_dir / "procrustes_synchronized_pair_permutation_null.csv")
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.15))
    spectrum_plot(axes[0, 0], spectrum)
    patient_endpoint_plot(axes[0, 1], static_patient, static_group)
    item_null_plot(axes[0, 2], permutation_null, summary)

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
    dynamic_heatmap(
        axes[1, 0],
        margin,
        margin_fwer,
        0.0,
        "Rank-4 dynamic margin",
        "Margin",
        "d",
    )
    dynamic_heatmap(
        axes[1, 1],
        accuracy,
        accuracy_fwer,
        0.5,
        "Rank-4 dynamic accuracy",
        "Accuracy minus 0.5",
        "e",
    )
    temporal_summary_plot(
        axes[1, 2], dynamic_matrices, cumulative_matrices, cumulative_stats
    )
    fig.suptitle(
        "A candidate low-dimensional rigid correspondence lacks a reproducible short-window locus",
        fontsize=9.2,
        fontweight="bold",
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.966), h_pad=2.4, w_pad=2.0)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "Fig_4_integrated_static_dynamic_manifold_alignment"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def write_caption_and_report(
    report_dir: Path,
    data_dir: Path,
    unit_set: str,
    cell_stats: pd.DataFrame,
    cumulative_stats: pd.DataFrame,
) -> None:
    static_group = pd.read_csv(data_dir / "static_model_group_inference.csv").set_index("metric")
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    margin = static_group.loc["procrustes_rank4_retrieval_margin"]
    accuracy = static_group.loc["procrustes_rank4_accuracy"]
    item_p_margin = summary["procrustes_permutation"][
        "rank4_margin_item_permutation_p_greater"
    ]
    item_p_accuracy = summary["procrustes_permutation"][
        "rank4_accuracy_item_permutation_p_greater"
    ]
    dynamic_margin_sig = int(
        np.sum(cell_stats["margin_exact_maxstat_p_fwer_greater"] < 0.05)
    )
    dynamic_accuracy_sig = int(
        np.sum(cell_stats["accuracy_exact_maxstat_p_fwer_greater"] < 0.05)
    )
    cumulative_margin_sig = int(
        np.sum(cumulative_stats["margin_exact_maxstat_p_fwer_greater"] < 0.05)
    )
    cumulative_accuracy_sig = int(
        np.sum(cumulative_stats["accuracy_exact_maxstat_p_fwer_greater"] < 0.05)
    )
    best_cumulative = cumulative_stats.iloc[
        int(cumulative_stats["patient_mean_retrieval_margin"].argmax())
    ]

    caption = (
        "Fig. 4 | Static and time-resolved low-dimensional alignment across active and passive constructions. "
        "a, Patient-level cumulative spectra of role-centred population responses; the cross-view spectrum is "
        "computed at a common maximum rank of eight. b, Held-pair Rank-4 orthogonal Procrustes retrieval margin "
        "and accuracy above chance. Coloured points denote patients (n = 5), and black bars denote patient means. "
        "c, Patient-equal observed Rank-4 margin relative to 500 synchronized sentence-pair permutations. "
        "d,e, Rank-4 Procrustes time-generalization matrices for retrieval margin and two-by-two accuracy; each "
        "map is fit only on training sentence pairs, and no cell is marked unless it survives an exhaustive "
        "patient-level max-stat correction across 64 cells. f, Matched 100-ms-bin margins and cumulative 0-to-T "
        "margins. Cumulative endpoints are separately controlled by patient-level max-stat correction across "
        "eight windows. Active/passive construction remains confounded with marker identity, acoustics, surface "
        "position and sentence form; therefore these data do not establish an abstract voice manifold."
    )
    (report_dir / "Fig4_integrated_caption_en.txt").write_text(caption + "\n", encoding="utf-8")

    report = f"""# 整合版Fig.4：静态manifold与时间来源

## 图的中心结论

0–800 ms平均反应中出现约四维的跨构式刚性对齐候选，但当前数据未将它可靠定位到某个100 ms短时窗；因此图中区分“平均反应可检测”与“时间来源已定位”。

## 当前规格

- unit规格：`{unit_set}`；7 runs、5 patients，患者为独立推断单位。
- 静态与动态均使用Rank-4正交Procrustes；每折完整留出两个sentence pairs，所有预处理和rotation只在训练折拟合。
- 静态margin={margin.patient_mean:.5f}，患者单侧P={margin.exact_signflip_p_greater:.5f}、双侧P={margin.exact_signflip_p_two_sided:.5f}，同步刺激置换P={item_p_margin:.5f}。
- 静态accuracy={accuracy.patient_mean:.5f}，患者单侧P={accuracy.exact_signflip_p_greater:.5f}，同步刺激置换P={item_p_accuracy:.5f}。
- 8×8动态矩阵：margin/accuracy分别有{dynamic_margin_sig}/64和{dynamic_accuracy_sig}/64格通过患者exact max-stat FWER。
- 累积0–T分析：margin/accuracy分别有{cumulative_margin_sig}/8和{cumulative_accuracy_sig}/8个终点通过患者exact max-stat FWER。
- 累积margin最大的终点为{best_cumulative.cumulative_window_ms} ms：mean={best_cumulative.patient_mean_retrieval_margin:.5f}，未校正单侧P={best_cumulative.margin_exact_signflip_p_greater:.5f}，max-stat FWER P={best_cumulative.margin_exact_maxstat_p_fwer_greater:.5f}。

## 讲故事边界

- 可以说：`The 0–800 ms averaged population response contains a candidate low-dimensional rigid correspondence across constructions, without a reproducible short-window temporal locus.`
- 若累积曲线未通过校正，不写“temporally integrated signal”；只能写“detectable in the 0–800 ms averaged response”。
- Rank-4的刺激层置换结果不能替代患者层确认；两套unit集合是重叠的敏感性分析，不是独立重复。
- active/passive与功能词、声学、表面位置和句式共线；Agent/Patient与生命性和词汇类别混杂，不能称为抽象语态manifold。
"""
    (report_dir / "FIG4_INTEGRATED_REPORT_zh.md").write_text(report, encoding="utf-8")

    integrated_summary = {
        "unit_set": unit_set,
        "rank": RANK,
        "patients": 5,
        "runs": 7,
        "static_margin": float(margin.patient_mean),
        "static_margin_patient_p_greater": float(margin.exact_signflip_p_greater),
        "static_margin_patient_p_two_sided": float(margin.exact_signflip_p_two_sided),
        "static_margin_item_permutation_p_greater": float(item_p_margin),
        "static_accuracy": float(accuracy.patient_mean),
        "static_accuracy_patient_p_greater": float(accuracy.exact_signflip_p_greater),
        "static_accuracy_item_permutation_p_greater": float(item_p_accuracy),
        "dynamic_margin_fwer_cells": dynamic_margin_sig,
        "dynamic_accuracy_fwer_cells": dynamic_accuracy_sig,
        "cumulative_margin_fwer_windows": cumulative_margin_sig,
        "cumulative_accuracy_fwer_windows": cumulative_accuracy_sig,
        "best_cumulative_margin_window_ms": best_cumulative.cumulative_window_ms,
        "best_cumulative_margin": float(best_cumulative.patient_mean_retrieval_margin),
        "best_cumulative_margin_p_uncorrected": float(
            best_cumulative.margin_exact_signflip_p_greater
        ),
        "best_cumulative_margin_p_fwer": float(
            best_cumulative.margin_exact_maxstat_p_fwer_greater
        ),
    }
    (data_dir / "fig4_integrated_summary.json").write_text(
        json.dumps(integrated_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def validate_outputs(
    dynamic_folds: pd.DataFrame,
    cumulative_folds: pd.DataFrame,
    cell_stats: pd.DataFrame,
    cumulative_stats: pd.DataFrame,
) -> dict[str, object]:
    expected_dynamic = 7 * 64 * 45
    expected_cumulative = 7 * 8 * 45
    checks = {
        "dynamic_rows": int(len(dynamic_folds)),
        "dynamic_rows_expected": expected_dynamic,
        "cumulative_rows": int(len(cumulative_folds)),
        "cumulative_rows_expected": expected_cumulative,
        "dynamic_cells": int(len(cell_stats)),
        "cumulative_windows": int(len(cumulative_stats)),
        "dynamic_all_finite": bool(
            np.isfinite(
                dynamic_folds[["retrieval_margin", "two_by_two_accuracy"]].to_numpy()
            ).all()
        ),
        "cumulative_all_finite": bool(
            np.isfinite(
                cumulative_folds[["retrieval_margin", "two_by_two_accuracy"]].to_numpy()
            ).all()
        ),
    }
    checks["passed"] = bool(
        checks["dynamic_rows"] == expected_dynamic
        and checks["cumulative_rows"] == expected_cumulative
        and checks["dynamic_cells"] == 64
        and checks["cumulative_windows"] == 8
        and checks["dynamic_all_finite"]
        and checks["cumulative_all_finite"]
    )
    if not checks["passed"]:
        raise RuntimeError(f"Integrated Fig. 4 validation failed: {checks}")
    return checks


def run_specification(unit_set: str, artifact_threshold: float | None) -> None:
    report_dir, data_dir = specification_paths(unit_set)
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics, available_artifact_runs = base.load_filtered_metrics(unit_set, artifact_threshold)
    runs, audit = base.build_run_data(metrics, available_artifact_runs, unit_set)
    if len(runs) != 7 or len({run.patient for run in runs}) != 5:
        raise RuntimeError(f"Unexpected included data for {unit_set}:\n{audit.to_string(index=False)}")

    dynamic_parts: list[pd.DataFrame] = []
    cumulative_parts: list[pd.DataFrame] = []
    for run_index, run in enumerate(runs, start=1):
        frame = metrics[(metrics["patient"] == run.patient) & (metrics["run"] == run.run)]
        active, passive = shared.extract_dynamic_tensor(run, frame, unit_set)
        dynamic_parts.append(dynamic_procrustes_cv(run, active, passive))
        active_cumulative, passive_cumulative = extract_cumulative_tensor(
            run, frame, unit_set
        )
        cumulative_parts.append(
            cumulative_procrustes_cv(run, active_cumulative, passive_cumulative)
        )
        print(f"{unit_set}: Rank-4 dynamic and cumulative CV {run_index}/{len(runs)}", flush=True)

    dynamic_folds = pd.concat(dynamic_parts, ignore_index=True)
    cumulative_folds = pd.concat(cumulative_parts, ignore_index=True)
    dynamic_run, dynamic_patient, cell_stats, dynamic_matrices = manifold.dynamic_inference(
        dynamic_folds
    )
    cumulative_run, cumulative_patient, cumulative_stats, cumulative_matrices = (
        summarize_cumulative(cumulative_folds)
    )
    validation = validate_outputs(
        dynamic_folds, cumulative_folds, cell_stats, cumulative_stats
    )

    dynamic_folds.to_csv(data_dir / "fig4_rank4_dynamic_cv_folds.csv", index=False)
    dynamic_run.to_csv(data_dir / "fig4_rank4_dynamic_run_summary.csv", index=False)
    dynamic_patient.to_csv(data_dir / "fig4_rank4_dynamic_patient_summary.csv", index=False)
    cell_stats.to_csv(data_dir / "fig4_rank4_dynamic_cell_stats.csv", index=False)
    np.savez_compressed(
        data_dir / "fig4_rank4_dynamic_patient_matrices.npz", **dynamic_matrices
    )
    cumulative_folds.to_csv(data_dir / "fig4_rank4_cumulative_cv_folds.csv", index=False)
    cumulative_run.to_csv(data_dir / "fig4_rank4_cumulative_run_summary.csv", index=False)
    cumulative_patient.to_csv(
        data_dir / "fig4_rank4_cumulative_patient_summary.csv", index=False
    )
    cumulative_stats.to_csv(data_dir / "fig4_rank4_cumulative_window_stats.csv", index=False)
    np.savez_compressed(
        data_dir / "fig4_rank4_cumulative_patient_matrices.npz", **cumulative_matrices
    )
    (data_dir / "fig4_integrated_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    make_figure(
        report_dir,
        data_dir,
        cell_stats,
        dynamic_matrices,
        cumulative_stats,
        cumulative_matrices,
    )
    write_caption_and_report(report_dir, data_dir, unit_set, cell_stats, cumulative_stats)


def postprocess_specification(unit_set: str) -> None:
    report_dir, data_dir = specification_paths(unit_set)
    required = [
        data_dir / "fig4_rank4_dynamic_cell_stats.csv",
        data_dir / "fig4_rank4_dynamic_patient_matrices.npz",
        data_dir / "fig4_rank4_cumulative_window_stats.csv",
        data_dir / "fig4_rank4_cumulative_patient_matrices.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing integrated Fig. 4 outputs: {missing}")
    cell_stats = pd.read_csv(required[0])
    with np.load(required[1], allow_pickle=True) as archive:
        dynamic_matrices = {key: np.asarray(archive[key]) for key in archive.files}
    cumulative_stats = pd.read_csv(required[2])
    with np.load(required[3], allow_pickle=True) as archive:
        cumulative_matrices = {key: np.asarray(archive[key]) for key in archive.files}
    make_figure(
        report_dir,
        data_dir,
        cell_stats,
        dynamic_matrices,
        cumulative_stats,
        cumulative_matrices,
    )
    write_caption_and_report(report_dir, data_dir, unit_set, cell_stats, cumulative_stats)


def write_combined_report() -> None:
    records: list[dict[str, object]] = []
    for unit_set in ("good_only", "usable_lenient"):
        _, data_dir = specification_paths(unit_set)
        summary_path = data_dir / "fig4_integrated_summary.json"
        if not summary_path.exists():
            return
        records.append(json.loads(summary_path.read_text(encoding="utf-8")))
    comparison = pd.DataFrame(records)
    comparison.to_csv(
        DATA_OUTPUT_ROOT / "fig4_integrated_specification_comparison.csv", index=False
    )
    main = comparison.set_index("unit_set").loc["good_only"]
    sensitivity = comparison.set_index("unit_set").loc["usable_lenient"]
    report = f"""# 整合版Fig.4：两套unit规格综合判读

| 指标 | good_only主规格 | usable_lenient敏感性 |
|---|---:|---:|
| 静态Rank-4 margin | {main.static_margin:.5f}；患者P={main.static_margin_patient_p_greater:.4f}；刺激P={main.static_margin_item_permutation_p_greater:.4f} | {sensitivity.static_margin:.5f}；患者P={sensitivity.static_margin_patient_p_greater:.4f}；刺激P={sensitivity.static_margin_item_permutation_p_greater:.4f} |
| 静态Rank-4 accuracy | {main.static_accuracy:.5f}；患者P={main.static_accuracy_patient_p_greater:.4f}；刺激P={main.static_accuracy_item_permutation_p_greater:.4f} | {sensitivity.static_accuracy:.5f}；患者P={sensitivity.static_accuracy_patient_p_greater:.4f}；刺激P={sensitivity.static_accuracy_item_permutation_p_greater:.4f} |
| 动态margin / accuracy FWER格 | {int(main.dynamic_margin_fwer_cells)}/64 / {int(main.dynamic_accuracy_fwer_cells)}/64 | {int(sensitivity.dynamic_margin_fwer_cells)}/64 / {int(sensitivity.dynamic_accuracy_fwer_cells)}/64 |
| 累积margin / accuracy FWER窗口 | {int(main.cumulative_margin_fwer_windows)}/8 / {int(main.cumulative_accuracy_fwer_windows)}/8 | {int(sensitivity.cumulative_margin_fwer_windows)}/8 / {int(sensitivity.cumulative_accuracy_fwer_windows)}/8 |
| 最大累积margin | {main.best_cumulative_margin_window_ms} ms，{main.best_cumulative_margin:.5f}，FWER P={main.best_cumulative_margin_p_fwer:.4f} | {sensitivity.best_cumulative_margin_window_ms} ms，{sensitivity.best_cumulative_margin:.5f}，FWER P={sensitivity.best_cumulative_margin_p_fwer:.4f} |

## 结论

- 整合后的静态和动态面板现在都使用Rank-4正交Procrustes，方法链一致。
- 两套规格均保留静态margin的同方向结果及刺激层置换信号，但患者层仍未达到P<0.05。
- 两套规格的8×8短时窗矩阵均为0/64格通过校正；累积分析也均为0/8窗口通过校正。
- 最大累积margin的时间终点在两套规格间不同（主规格为{main.best_cumulative_margin_window_ms} ms，宽松规格为{sensitivity.best_cumulative_margin_window_ms} ms），因此不能写成稳定的时间积累轨迹。
- 最稳妥主线是：0–800 ms平均反应中有低维刚性对应候选，但当前数据不能确定它来自哪个短时窗，也不能确认稳定的时间积累机制。

## 端点一致性审计

累积0–800 ms终点使用原静态分析的闭区间定义，并在每个run内与冻结的静态矩阵逐值核对；最终患者margin和accuracy与静态Rank-4结果完全一致。半开100 ms动态窗仅用于时间泛化矩阵，没有被混入该静态端点。
"""
    (REPORT_ROOT / "FIG4_INTEGRATED_COMBINED_REPORT_zh.md").write_text(
        report, encoding="utf-8"
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
