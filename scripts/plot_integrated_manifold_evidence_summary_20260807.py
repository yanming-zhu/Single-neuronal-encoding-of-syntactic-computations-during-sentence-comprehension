#!/usr/bin/env python3
"""Create an evidence-balanced summary of the active/passive manifold results.

The figure deliberately combines descriptive low-dimensional structure,
patient-level candidate effects, robustness checks, and corrected negative
results. Patients, rather than runs, folds, or units, are the inferential
replicates (n=5).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025" / "data"
MANIFOLD_DIR = DATA_ROOT / "combined_active_passive_manifold_geometry_20260806"
NESTED_DIR = DATA_ROOT / "combined_active_passive_nested_regularized_alignment_20260807"
SHARED_DIR = DATA_ROOT / "combined_active_passive_shared_latent_geometry_20260806"
ANCHOR_DIR = DATA_ROOT / "combined_event_anchor_manifold_geometry_20260806"
REPORT_DIR = ROOT / "reports" / "active_passive_manifold_evidence_summary_20260807"
FIGURE_DIR = REPORT_DIR / "figures"
FIGURE_STEM = FIGURE_DIR / "Fig_8_integrated_manifold_evidence_summary"


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#8E44AD",
    "sky": "#56B4E9",
    "black": "#111111",
    "gray": "#777777",
    "light_gray": "#D9D9D9",
    "very_light_gray": "#EEEEEE",
}
PATIENT_COLORS = {
    "gz001": "#4C78A8",
    "gz002": "#F58518",
    "gz003": "#54A24B",
    "gz004": "#B279A2",
    "gz005": "#E45756",
}
PATIENTS = list(PATIENT_COLORS)


def configure_style() -> None:
    """Use a compact, vector-friendly publication style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 7.7,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.17,
        1.11,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def zero_line(ax: plt.Axes) -> None:
    ax.axhline(0, color=COLORS["gray"], lw=0.85, ls="--", zorder=0)


def mean_bar(ax: plt.Axes, x: float, values: np.ndarray) -> None:
    mean = float(np.mean(values))
    ax.plot([x - 0.16, x + 0.16], [mean, mean], color=COLORS["black"], lw=2.2, zorder=5)


def load_inputs() -> dict[str, pd.DataFrame]:
    paths = {
        "spectrum": MANIFOLD_DIR / "intrinsic_spectrum.csv",
        "dimension_group": MANIFOLD_DIR / "intrinsic_dimension_group_summary.csv",
        "good_patient": NESTED_DIR / "patient_summary.csv",
        "good_group": NESTED_DIR / "group_inference.csv",
        "lenient_patient": NESTED_DIR / "sensitivity_usable_lenient" / "patient_summary.csv",
        "lenient_group": NESTED_DIR / "sensitivity_usable_lenient" / "group_inference.csv",
        "participation_run": SHARED_DIR / "readout_axis_run_summary.csv",
        "shared_group": SHARED_DIR / "group_inference.csv",
        "anchor_patient": ANCHOR_DIR / "static_rank4_patient_anchor_comparison.csv",
        "anchor_inference": ANCHOR_DIR / "static_rank4_anchor_inference.csv",
        "dynamic": MANIFOLD_DIR / "fig4_rank4_dynamic_cell_stats.csv",
        "cumulative_patient": MANIFOLD_DIR / "fig4_rank4_cumulative_patient_summary.csv",
        "cumulative_stats": MANIFOLD_DIR / "fig4_rank4_cumulative_window_stats.csv",
        "tests": NESTED_DIR / "good_only_statistical_test_sensitivity.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))
    return {name: pd.read_csv(path) for name, path in paths.items()}


def check_patient_set(frame: pd.DataFrame, label: str) -> None:
    observed = sorted(frame["patient"].dropna().astype(str).unique())
    if observed != sorted(PATIENTS):
        raise RuntimeError(f"{label}: expected patients {PATIENTS}, observed {observed}")


def plot_spectrum(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    patient = (
        data["spectrum"]
        .groupby(["patient", "view", "dimension"], as_index=False)["cumulative_fraction"]
        .mean()
    )
    styles = {
        "active": (COLORS["blue"], "Active", "-", "o"),
        "passive": (COLORS["orange"], "Passive", "-.", "s"),
        "shared_cross_covariance": (COLORS["green"], "Cross-view", "--", "^"),
    }
    for view, (color, label, linestyle, marker) in styles.items():
        sub = patient[(patient["view"] == view) & (patient["dimension"] <= 8)]
        pivot = sub.pivot(index="patient", columns="dimension", values="cumulative_fraction")
        for _, row in pivot.iterrows():
            ax.plot(
                pivot.columns,
                row,
                color=color,
                alpha=0.14,
                lw=0.7,
                ls=linestyle,
            )
        ax.plot(
            pivot.columns,
            pivot.mean(axis=0),
            color=color,
            lw=1.9,
            marker=marker,
            ms=2.4,
            ls=linestyle,
            label=label,
        )
    group = data["dimension_group"].set_index("view")
    cross_rank4 = float(group.loc["shared_cross_covariance", "variance_rank4"])
    pr_values = [
        float(group.loc[view, "rank_limited_effective_dimension"])
        for view in ("active", "passive", "shared_cross_covariance")
    ]
    ax.axvline(4, color=COLORS["gray"], ls=":", lw=0.85)
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(0.15, 1.035)
    ax.set_xticks([1, 2, 4, 6, 8])
    ax.set_xlabel("Retained dimensions")
    ax.set_ylabel("Cumulative spectral fraction")
    ax.legend(frameon=False, loc="lower right", handlelength=2.2)
    ax.set_title("Low-dimensional cross-view spectrum\n(descriptive)")
    ax.text(
        0.03,
        0.97,
        f"Cross-view rank 4 = {cross_rank4:.3f}\nRank-8 PR = {pr_values[0]:.2f} / {pr_values[1]:.2f} / {pr_values[2]:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    panel_label(ax, "A")


def original_method(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[frame["method"] == "original_procrustes"]
        .sort_values("patient")
        .reset_index(drop=True)
    )


def plot_unit_rule_sensitivity(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    good = original_method(data["good_patient"])[["patient", "retrieval_margin"]].rename(
        columns={"retrieval_margin": "good_only"}
    )
    lenient = original_method(data["lenient_patient"])[["patient", "retrieval_margin"]].rename(
        columns={"retrieval_margin": "usable_lenient"}
    )
    paired = good.merge(lenient, on="patient", validate="one_to_one")
    for row in paired.itertuples(index=False):
        color = PATIENT_COLORS[row.patient]
        values = [row.good_only, row.usable_lenient]
        ax.plot([0, 1], values, color=color, alpha=0.52, lw=1.0, zorder=1)
        ax.scatter([0, 1], values, color=color, s=24, edgecolor="white", linewidth=0.45, zorder=3)
    good_values = paired["good_only"].to_numpy(dtype=float)
    lenient_values = paired["usable_lenient"].to_numpy(dtype=float)
    mean_bar(ax, 0, good_values)
    mean_bar(ax, 1, lenient_values)
    zero_line(ax)
    good_p = float(
        data["good_group"].set_index("method").loc[
            "original_procrustes", "exact_signflip_p_greater_raw"
        ]
    )
    lenient_p = float(
        data["lenient_group"].set_index("method").loc[
            "original_procrustes", "exact_signflip_p_greater_raw"
        ]
    )
    ax.set_xticks([0, 1], ["Good only\n(primary)", "Usable units\n(lenient)"])
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylabel("Held-pair retrieval margin")
    ax.set_title("Static correspondence across\nunit-inclusion rules")
    ax.text(
        0.03,
        0.97,
        f"Means = {good_values.mean():.4f} / {lenient_values.mean():.4f}\nExact P = {good_p:.4f} / {lenient_p:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    panel_label(ax, "B")


def plot_participation(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    patient = (
        data["participation_run"]
        .groupby("patient", as_index=False)["participation_rho"]
        .mean()
        .sort_values("patient")
    )
    check_patient_set(patient, "participation")
    offsets = np.linspace(-0.12, 0.12, len(patient))
    for offset, row in zip(offsets, patient.itertuples(index=False)):
        ax.scatter(
            offset,
            row.participation_rho,
            color=PATIENT_COLORS[row.patient],
            s=31,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    values = patient["participation_rho"].to_numpy(dtype=float)
    mean_bar(ax, 0, values)
    zero_line(ax)
    inference = data["shared_group"].set_index("metric").loc["axis_participation_rho"]
    ax.set_xlim(-0.32, 0.32)
    ax.set_ylim(-0.04, 0.68)
    ax.set_xticks([0], ["Active/passive\nparticipation"])
    ax.set_ylabel("Spearman rho")
    ax.set_title("Concordant population\nparticipation")
    ax.text(
        0.03,
        0.97,
        f"Mean = {values.mean():.3f}; positive = {int((values > 0).sum())}/5\nOne-sided exact P = {float(inference['exact_signflip_p_greater']):.5f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    panel_label(ax, "C")


def plot_anchor_comparison(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    order = ["noun", "marker", "verb"]
    labels = ["Target noun", "Marker\n(ba/bei)", "Main verb"]
    patient = data["anchor_patient"].copy()
    check_patient_set(patient, "anchor comparison")
    pivot = patient.pivot(index="patient", columns="anchor", values="retrieval_margin").loc[PATIENTS, order]
    for patient_name, row in pivot.iterrows():
        ax.plot(range(3), row.to_numpy(), color=PATIENT_COLORS[patient_name], alpha=0.45, lw=0.9)
        ax.scatter(
            range(3),
            row.to_numpy(),
            color=PATIENT_COLORS[patient_name],
            s=22,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    for index, anchor in enumerate(order):
        mean_bar(ax, index, pivot[anchor].to_numpy(dtype=float))
    zero_line(ax)
    inference = data["anchor_inference"].set_index("anchor")
    p_values = [
        float(inference.loc[anchor, "margin_patient_anchor_maxstat_p_fwer_greater"])
        for anchor in order
    ]
    ax.set_xticks(range(3), labels, rotation=18, ha="right")
    ax.set_xlim(-0.42, 2.42)
    ax.set_ylim(-0.245, 0.225)
    ax.set_ylabel("Held-pair retrieval margin")
    ax.set_title("Event-anchor dependence is\nnot statistically established")
    for index, p_value in enumerate(p_values):
        ax.text(index, 0.207, f"P={p_value:.3g}", ha="center", va="top", fontsize=6.0)
    ax.text(
        0.03,
        0.03,
        "Anchor max-stat correction;\nno endpoint passes 0.05",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=6.1,
    )
    panel_label(ax, "D")


def plot_dynamic_heatmap(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    dynamic = data["dynamic"]
    matrix = dynamic.pivot(
        index="active_window_index",
        columns="passive_window_index",
        values="patient_mean_retrieval_margin",
    ).to_numpy()
    p_fwer = dynamic.pivot(
        index="active_window_index",
        columns="passive_window_index",
        values="margin_exact_maxstat_p_fwer_greater",
    ).to_numpy()
    limit = float(np.ceil(np.nanmax(np.abs(matrix)) * 100) / 100)
    image = ax.imshow(
        matrix,
        origin="lower",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="equal",
        interpolation="nearest",
    )
    labels = [f"{start}–{start + 100}" for start in range(0, 800, 100)]
    ax.set_xticks(range(8), labels, rotation=47, ha="right")
    ax.set_yticks(range(8), labels)
    ax.set_xlabel("Passive noun window (ms)")
    ax.set_ylabel("Active noun window (ms)")
    n_sig = int(np.sum(p_fwer < 0.05))
    n_raw = int(np.sum(dynamic["margin_exact_signflip_p_greater"] < 0.05))
    ax.set_title("Dynamic Rank-4 margin\n0/64 cells survive correction")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Patient-mean margin", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.2)
    ax.text(
        0.02,
        0.02,
        f"Raw P<.05: {n_raw}/64\nMax-stat FWER: {n_sig}/64",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=5.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )
    panel_label(ax, "E")


def plot_cumulative(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    patient = data["cumulative_patient"].copy()
    check_patient_set(patient, "cumulative")
    x = np.arange(100, 801, 100)
    for patient_name in PATIENTS:
        sub = patient[patient["patient"] == patient_name].sort_values("cumulative_stop_ms")
        ax.plot(
            x,
            sub["retrieval_margin"],
            color=PATIENT_COLORS[patient_name],
            alpha=0.28,
            lw=0.9,
        )
    stats = data["cumulative_stats"].sort_values("cumulative_window_index")
    ax.plot(
        x,
        stats["patient_mean_retrieval_margin"],
        color=COLORS["purple"],
        marker="o",
        ms=3.2,
        lw=2.0,
        zorder=4,
    )
    zero_line(ax)
    n_sig = int(np.sum(stats["margin_exact_maxstat_p_fwer_greater"] < 0.05))
    final = stats.iloc[-1]
    ax.annotate(
        f"0–800 ms\nP$_{{FWER}}$={final.margin_exact_maxstat_p_fwer_greater:.4f}",
        xy=(800, final.patient_mean_retrieval_margin),
        xytext=(-42, 25),
        textcoords="offset points",
        fontsize=6.2,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": COLORS["gray"], "lw": 0.65},
    )
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_xlim(72, 835)
    ax.set_xlabel("Window end T after noun onset (ms)")
    ax.set_ylabel("Rank-4 retrieval margin")
    ax.set_title(f"Cumulative 0–T trajectory\n{n_sig}/8 endpoints survive correction")
    panel_label(ax, "F")


def plot_method_robustness(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    order = ["original_procrustes", "noise_whitened_procrustes", "regularized_pls"]
    labels = ["Original\nProcrustes", "Noise-whitened\nProcrustes", "Regularized\nPLS"]
    patient = data["good_patient"].copy()
    check_patient_set(patient, "model robustness")
    pivot = patient.pivot(index="patient", columns="method", values="retrieval_margin").loc[PATIENTS, order]
    for patient_name, row in pivot.iterrows():
        ax.plot(range(3), row.to_numpy(), color=PATIENT_COLORS[patient_name], alpha=0.46, lw=0.9)
        ax.scatter(
            range(3),
            row.to_numpy(),
            color=PATIENT_COLORS[patient_name],
            s=22,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    for index, method in enumerate(order):
        mean_bar(ax, index, pivot[method].to_numpy(dtype=float))
    zero_line(ax)
    group = data["good_group"].set_index("method")
    p_values = [float(group.loc[method, "exact_maxstat_p_fwer_greater"]) for method in order]
    means = [float(group.loc[method, "patient_mean_retrieval_margin"]) for method in order]
    ax.set_xticks(range(3), labels)
    ax.set_xlim(-0.42, 2.42)
    ax.set_ylabel("Held-pair retrieval margin")
    ax.set_title("Alternative alignment models\ndo not improve the effect")
    ax.text(
        0.03,
        0.97,
        "Means: " + " / ".join(f"{value:.3f}" for value in means) + "\nFWER P: " + " / ".join(f"{value:.4f}" for value in p_values),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.1},
    )
    panel_label(ax, "G")


def plot_test_sensitivity(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    name_map = {
        "Exact sign-flip (mean)": "Exact sign-flip",
        "Studentized sign-flip": "Studentized flip",
        "Exact Wilcoxon": "Exact Wilcoxon",
        "Binomial sign test": "Binomial sign",
        "Parametric one-sample t": "One-sample t",
        "Parametric t + Holm (3 methods)": "t + Holm (3)",
    }
    order = list(name_map)
    tests = data["tests"].set_index("test").loc[order].reset_index()
    y = np.arange(len(tests))[::-1]
    colors = [
        COLORS["blue"] if family == "distribution-free" else COLORS["vermillion"]
        for family in tests["family"]
    ]
    markers = ["o" if family == "distribution-free" else "s" for family in tests["family"]]
    ax.hlines(y, 0, tests["p_value"], color=colors, lw=1.8)
    for p_value, y_value, color, marker in zip(tests["p_value"], y, colors, markers):
        ax.scatter(
            p_value,
            y_value,
            color=color,
            marker=marker,
            s=28,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    ax.axvline(0.05, color=COLORS["gray"], lw=0.9, ls="--")
    for y_value, p_value in zip(y, tests["p_value"]):
        ax.text(p_value + 0.005, y_value, f"{p_value:.4f}", va="center", fontsize=6.0)
    ax.set_yticks(y, [name_map[name] for name in order])
    ax.set_xlim(0, 0.22)
    ax.set_xlabel("One-sided P value")
    ax.set_title("Inference is sensitive to\ndistributional assumptions")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markeredgecolor="none", label="Distribution-free"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["vermillion"], markeredgecolor="none", label="Normal-model dependent"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=5.9, handletextpad=0.25)
    panel_label(ax, "H")


def source_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    spectrum_patient = (
        data["spectrum"]
        .groupby(["patient", "view", "dimension"], as_index=False)["cumulative_fraction"]
        .mean()
    )
    for row in spectrum_patient[spectrum_patient["dimension"] <= 8].itertuples(index=False):
        rows.append(
            {
                "panel": "A",
                "analysis": "cumulative_spectrum",
                "patient": row.patient,
                "condition": row.view,
                "x": row.dimension,
                "value": row.cumulative_fraction,
            }
        )

    for unit_set, frame in (("good_only", data["good_patient"]), ("usable_lenient", data["lenient_patient"])):
        for row in original_method(frame).itertuples(index=False):
            rows.append(
                {
                    "panel": "B",
                    "analysis": "static_unit_rule_sensitivity",
                    "patient": row.patient,
                    "condition": unit_set,
                    "value": row.retrieval_margin,
                }
            )

    participation = data["participation_run"].groupby("patient", as_index=False)["participation_rho"].mean()
    for row in participation.itertuples(index=False):
        rows.append(
            {
                "panel": "C",
                "analysis": "readout_axis_participation",
                "patient": row.patient,
                "condition": "active_passive",
                "value": row.participation_rho,
            }
        )

    for row in data["anchor_patient"].itertuples(index=False):
        rows.append(
            {
                "panel": "D",
                "analysis": "event_anchor_comparison",
                "patient": row.patient,
                "condition": row.anchor,
                "value": row.retrieval_margin,
            }
        )

    for row in data["dynamic"].itertuples(index=False):
        rows.append(
            {
                "panel": "E",
                "analysis": "dynamic_rank4_margin",
                "condition": f"active_{row.active_window_ms}__passive_{row.passive_window_ms}",
                "x": row.passive_window_index,
                "y": row.active_window_index,
                "value": row.patient_mean_retrieval_margin,
                "p_value": row.margin_exact_maxstat_p_fwer_greater,
                "p_type": "patient_maxstat_fwer_greater",
            }
        )

    for row in data["cumulative_patient"].itertuples(index=False):
        stat = data["cumulative_stats"].loc[
            data["cumulative_stats"]["cumulative_window_index"] == row.cumulative_window_index
        ].iloc[0]
        rows.append(
            {
                "panel": "F",
                "analysis": "cumulative_rank4_margin",
                "patient": row.patient,
                "condition": f"0-{row.cumulative_stop_ms}_ms",
                "x": row.cumulative_stop_ms,
                "value": row.retrieval_margin,
                "p_value": stat.margin_exact_maxstat_p_fwer_greater,
                "p_type": "patient_maxstat_fwer_greater",
            }
        )

    for row in data["good_patient"].itertuples(index=False):
        rows.append(
            {
                "panel": "G",
                "analysis": "alignment_method_robustness",
                "patient": row.patient,
                "condition": row.method,
                "value": row.retrieval_margin,
            }
        )

    for row in data["tests"].itertuples(index=False):
        rows.append(
            {
                "panel": "H",
                "analysis": "statistical_test_sensitivity",
                "condition": row.test,
                "value": row.statistic,
                "p_value": row.p_value,
                "p_type": row.family,
            }
        )

    columns = ["panel", "analysis", "patient", "condition", "x", "y", "value", "p_value", "p_type"]
    return pd.DataFrame(rows).reindex(columns=columns)


def make_figure(data: dict[str, pd.DataFrame]) -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.15))
    plot_spectrum(axes[0, 0], data)
    plot_unit_rule_sensitivity(axes[0, 1], data)
    plot_participation(axes[0, 2], data)
    plot_anchor_comparison(axes[0, 3], data)
    plot_dynamic_heatmap(axes[1, 0], data)
    plot_cumulative(axes[1, 1], data)
    plot_method_robustness(axes[1, 2], data)
    plot_test_sensitivity(axes[1, 3], data)
    fig.suptitle(
        "Candidate low-dimensional cross-construction correspondence after target-noun onset",
        fontsize=12.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        "Patient is the inferential unit (n=5); candidate effects are shown together with corrected temporal, anchor, and model checks.",
        ha="center",
        va="top",
        fontsize=7.6,
        color="#333333",
    )
    fig.text(
        0.5,
        0.018,
        "Margins are held-sentence-pair cross-validated. P values are one-sided; FWER values use exhaustive patient-level max-stat sign flips. Colors encode patients or analysis families, not significance.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.105, top=0.895, wspace=0.50, hspace=0.52)
    for suffix in ("png", "pdf", "svg", "tiff"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(FIGURE_STEM.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def write_caption(data: dict[str, pd.DataFrame]) -> None:
    good_group = data["good_group"].set_index("method")
    participation = data["shared_group"].set_index("metric").loc["axis_participation_rho"]
    anchor = data["anchor_inference"].set_index("anchor")
    dynamic = data["dynamic"]
    cumulative = data["cumulative_stats"].sort_values("cumulative_window_index")
    tests = data["tests"].set_index("test")
    caption = f"""Figure 8 | Evidence-balanced summary of candidate low-dimensional active/passive correspondence after target-noun onset. (A) Patient-averaged cumulative spectra for active, passive, and shared cross-covariance views. The first four shared dimensions captured {float(data['dimension_group'].set_index('view').loc['shared_cross_covariance', 'variance_rank4']):.3f} of the cross-view spectrum; this is a descriptive compactness result. Thin lines show patients and thick lines show patient means. (B) Held-sentence-pair Rank-4 Procrustes retrieval margins under the primary good-unit and lenient usable-unit specifications. Colored points and lines denote patients; black horizontal bars denote means. The primary mean was {float(good_group.loc['original_procrustes', 'patient_mean_retrieval_margin']):.4f} (4/5 positive; one-sided exact sign-flip P={float(good_group.loc['original_procrustes', 'exact_signflip_p_greater_raw']):.4f}). (C) Across-run patient means of the Spearman correlation between active and passive readout-axis participation weights (mean rho={float(participation['patient_mean']):.3f}; 5/5 positive; one-sided exact P={float(participation['exact_signflip_p_greater']):.5f}). This population-level concordance does not imply identical tuning of individual neurons. (D) Static Rank-4 margins after alignment to target noun, 把/被 marker, or main verb. Anchor-family max-stat FWER P values were {float(anchor.loc['noun', 'margin_patient_anchor_maxstat_p_fwer_greater']):.5f}, {float(anchor.loc['marker', 'margin_patient_anchor_maxstat_p_fwer_greater']):.5f}, and {float(anchor.loc['verb', 'margin_patient_anchor_maxstat_p_fwer_greater']):.5f}, respectively; therefore noun specificity was not established. (E) Patient-mean Rank-4 retrieval margin for all active-by-passive 100-ms window pairs. No cell survived exhaustive patient-level max-stat correction ({int((dynamic['margin_exact_maxstat_p_fwer_greater'] < 0.05).sum())}/64). Colors encode effect size, not significance. (F) Patient trajectories and means for cumulative 0–T target-noun windows. No endpoint survived correction ({int((cumulative['margin_exact_maxstat_p_fwer_greater'] < 0.05).sum())}/8); the 0–800-ms endpoint was {float(cumulative.iloc[-1]['patient_mean_retrieval_margin']):.4f} (FWER P={float(cumulative.iloc[-1]['margin_exact_maxstat_p_fwer_greater']):.4f}). (G) Paired patient results for original Procrustes, noise-whitened Procrustes, and nested regularized PLS. The alternative models did not improve the patient-level evidence. (H) Statistical-test sensitivity for the primary good-unit Original Procrustes endpoint. Distribution-free exact tests did not cross 0.05, whereas the one-sample t test did (P={float(tests.loc['Parametric one-sample t', 'p_value']):.5f}) and was borderline after Holm correction across three methods (P={float(tests.loc['Parametric t + Holm (3 methods)', 'p_value']):.5f}). Patients were the independent inferential units (n=5); runs, folds, and neurons were not treated as independent replicates."""
    (REPORT_DIR / "Fig8_caption_en.txt").write_text(caption + "\n", encoding="utf-8")


def write_chinese_report(data: dict[str, pd.DataFrame]) -> None:
    good_group = data["good_group"].set_index("method")
    lenient_group = data["lenient_group"].set_index("method")
    shared = data["shared_group"].set_index("metric").loc["axis_participation_rho"]
    anchor = data["anchor_inference"].set_index("anchor")
    cumulative = data["cumulative_stats"].sort_values("cumulative_window_index")
    tests = data["tests"].set_index("test")
    report = f"""# 主动/被动低维几何结果整合图：中文解读

## 一句话结论

在 **target-noun onset 后 0–800 ms 的平均群体反应**中，存在一个值得继续检验的低维跨构式几何对应：跨视图谱较紧凑，Rank-4 Procrustes 的患者均值 margin 为 {float(good_group.loc['original_procrustes', 'patient_mean_retrieval_margin']):.4f}，4/5 患者为正；但患者层精确检验为 P={float(good_group.loc['original_procrustes', 'exact_signflip_p_greater_raw']):.4f}，时间格点、累积窗口和不同事件对齐点均未通过相应的多重校正。因此当前结果应称为“候选低维跨构式对应”，不能称为已确认的抽象语态 manifold。

## 各面板怎么读

- **A，低维谱：** active 和 passive 各自需要较多维度，而 shared cross-covariance 的前 4 维已解释约 {float(data['dimension_group'].set_index('view').loc['shared_cross_covariance', 'variance_rank4']):.3f}。这说明两种构式间的共同变化可被较少维度概括，但谱压缩本身不等于存在生成性的神经 manifold。
- **B，unit 选择敏感性：** good-only 主规格均值为 {float(good_group.loc['original_procrustes', 'patient_mean_retrieval_margin']):.4f}；宽松 usable-unit 规格均值为 {float(lenient_group.loc['original_procrustes', 'patient_mean_retrieval_margin']):.4f}。两者方向相近，患者精确 P 均为 0.0625，说明结论不是只由 unit 筛选产生，但证据仍不足以在 n=5 下越过 0.05。
- **C，共同参与：** active/passive readout 轴上的 unit participation 在患者层 5/5 为正，均值 rho={float(shared['patient_mean']):.3f}，单侧精确 P={float(shared['exact_signflip_p_greater']):.5f}。它支持“同一群体以相关权重参与两种构式”，但不等于每个神经元具有相同调谐，也不等于发现了语法选择性单元。
- **D，事件对齐：** target noun 的 margin 最大；marker 与 verb 接近或低于 0。但把 noun、marker、verb 作为一个 anchor family 校正后，noun 的 P_FWER={float(anchor.loc['noun', 'margin_patient_anchor_maxstat_p_fwer_greater']):.5f}，未达到 0.05，所以不能下“noun-specific”结论。
- **E，动态 8×8：** 色彩仅表示患者平均 margin 的方向和大小，不表示显著性。64 个 active/passive 时间窗组合中 0 个通过患者 max-stat 校正，说明当前数据没有定位出可重复的短时窗几何对应。
- **F，累积 0–T：** 曲线不是稳定单调累积，0–500 ms 甚至出现负均值；8 个累积终点 0 个通过校正。0–800 ms 是最强终点之一（margin={float(cumulative.iloc[-1]['patient_mean_retrieval_margin']):.4f}，P_FWER={float(cumulative.iloc[-1]['margin_exact_maxstat_p_fwer_greater']):.4f}），更像跨较长时间平均后显现的弱而分布式信号。
- **G，模型稳健性：** noise-whitened Procrustes 与原模型几乎相同，nested regularized PLS 反而更弱；更复杂的映射没有提高患者层证据。
- **H，统计敏感性：** 分布无关的精确检验均不低于 0.0625；参数 t 检验为 P={float(tests.loc['Parametric one-sample t', 'p_value']):.5f}，三方法 Holm 校正后为 P={float(tests.loc['Parametric t + Holm (3 methods)', 'p_value']):.5f}。在 n=5 下，这一“显著”依赖正态模型假设，只能作为敏感性补充，不能替代主精确检验。

## 与 Fig. 4 的故事线

Fig. 4 可以先回答“主动/被动句的神经响应是否具有可比较的群体结构”；本图进一步回答“这种结构能否被描述为低维对应，以及它在时间、对齐点和模型选择上是否稳健”。合起来最稳妥的叙述是：**target-noun onset 后的平均群体活动包含候选的低维跨构式对应，但它尚未表现为经过患者层校正确认的、时间局部化或 noun-specific 的 manifold。**

## 解释边界

患者是独立统计单位（n=5）。主动/被动条件同时伴随功能词、声学、表面位置和句式差异；因此本图不能证明抽象语态编码、跨患者通用几何或语法选择性神经元。
"""
    (REPORT_DIR / "RESULTS_GUIDE_zh.md").write_text(report, encoding="utf-8")


def validate(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    good = original_method(data["good_patient"])
    participation = data["participation_run"].groupby("patient", as_index=False)["participation_rho"].mean()
    dynamic = data["dynamic"]
    cumulative = data["cumulative_stats"]
    checks = {
        "n_primary_patients": int(good["patient"].nunique()),
        "primary_mean_margin": float(good["retrieval_margin"].mean()),
        "primary_positive_patients": int((good["retrieval_margin"] > 0).sum()),
        "participation_mean_rho": float(participation["participation_rho"].mean()),
        "participation_positive_patients": int((participation["participation_rho"] > 0).sum()),
        "dynamic_cells": int(len(dynamic)),
        "dynamic_fwer_positive_cells": int((dynamic["margin_exact_maxstat_p_fwer_greater"] < 0.05).sum()),
        "cumulative_endpoints": int(len(cumulative)),
        "cumulative_fwer_positive_endpoints": int((cumulative["margin_exact_maxstat_p_fwer_greater"] < 0.05).sum()),
        "anchor_count": int(data["anchor_inference"]["anchor"].nunique()),
        "method_count": int(data["good_group"]["method"].nunique()),
        "test_count": int(len(data["tests"])),
    }
    checks["passed"] = bool(
        checks["n_primary_patients"] == 5
        and np.isclose(checks["primary_mean_margin"], 0.08664908179277483)
        and checks["primary_positive_patients"] == 4
        and np.isclose(checks["participation_mean_rho"], 0.39035573778428817)
        and checks["participation_positive_patients"] == 5
        and checks["dynamic_cells"] == 64
        and checks["dynamic_fwer_positive_cells"] == 0
        and checks["cumulative_endpoints"] == 8
        and checks["cumulative_fwer_positive_endpoints"] == 0
        and checks["anchor_count"] == 3
        and checks["method_count"] == 3
        and checks["test_count"] == 6
    )
    return checks


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    for key in ("good_patient", "lenient_patient", "participation_run", "anchor_patient", "cumulative_patient"):
        check_patient_set(data[key], key)
    make_figure(data)
    source_data(data).to_csv(REPORT_DIR / "Fig8_source_data.csv", index=False)
    write_caption(data)
    write_chinese_report(data)
    checks = validate(data)
    checks["outputs"] = {
        suffix: str(FIGURE_STEM.with_suffix(f".{suffix}"))
        for suffix in ("png", "pdf", "svg", "tiff")
    }
    checks["all_outputs_exist"] = all(Path(path).exists() for path in checks["outputs"].values())
    checks["passed"] = bool(checks["passed"] and checks["all_outputs_exist"])
    (REPORT_DIR / "validation.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not checks["passed"]:
        raise RuntimeError(f"Validation failed: {checks}")
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
