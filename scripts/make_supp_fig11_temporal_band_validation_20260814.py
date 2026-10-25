#!/usr/bin/env python3
"""Build the proposed expanded Supplementary Fig. 11 for main Fig. 2f.

Figure contract
---------------
Core conclusion:
    The fixed early/middle/late bands used for the descriptive Fig. 2f
    archetypes provide a transparent, low-complexity summary of peak timing
    that closely matches a patient-robust three-component GMM, although pooled
    BIC alone prefers a boundary-adjacent fourth component.
Archetype:
    Quantitative validation grid.
Panel map:
    a, Peak-latency distribution, fixed-band definitions and GMM component means.
    b, Pooled GMM model comparison by delta BIC.
    c, Leave-one-patient-out reproducibility for K=3 versus K=4.
    d, Concordance of the fixed Fig. 2f bands with pooled K=3 assignments.
    e, Cross-session composition of the six direction-by-latency archetypes.
Reviewer risk:
    The analysis does not establish three natural neuronal types or processing
    stages. Full temporal envelopes lack stable discrete clustering; K=3 is a
    descriptive patient-robust summary and K=4 remains the pooled-BIC sensitivity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle


REPORT = ROOT / "reports" / "active_passive_unsupervised_time_clustering_20260724"
ARCHETYPE = ROOT / "reports" / "active_passive_go_style_interpretable_20260723"
OUT = ROOT / "manuscript" / "Supplementary_Figures_20260814"
SOURCE = OUT / "source_data"

WIDTH_MM = 180.0
HEIGHT_MM = 148.0
PT = 7.0

INK = "#202326"
GRAY = "#6E7175"
LIGHT_GRAY = "#D8DDE3"
K3 = "#D7A640"
K4 = "#2C7FB8"
EARLY = "#C6DBEF"
MIDDLE = "#6BAED6"
LATE = "#2171B5"
ACTIVE_COLORS = {"A-E": "#C6DBEF", "A-M": "#6BAED6", "A-L": "#2171B5"}
PASSIVE_COLORS = {"P-E": "#FCAE91", "P-M": "#FB6A4A", "P-L": "#CB181D"}
ARCHETYPE_COLORS = {**ACTIVE_COLORS, **PASSIVE_COLORS}
CLUSTER_ORDER = ["A-E", "A-M", "A-L", "P-E", "P-M", "P-L"]


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": PT,
        "axes.labelsize": PT,
        "axes.titlesize": PT,
        "xtick.labelsize": PT - 0.3,
        "ytick.labelsize": PT - 0.3,
        "legend.fontsize": PT - 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PT + 1.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "assignments": pd.read_csv(REPORT / "time_cluster_assignments.csv"),
        "selection": pd.read_csv(REPORT / "peak_latency_gmm_selection.csv"),
        "k3_components": pd.read_csv(REPORT / "peak_latency_gmm_components.csv"),
        "k4_components": pd.read_csv(REPORT / "pooled_bic_k4_components.csv"),
        "lopo": pd.read_csv(REPORT / "leave_one_patient_out_all_k.csv"),
        "composition": pd.read_csv(ARCHETYPE / "interpretable_cluster_session_composition.csv"),
        "full_envelope": pd.read_csv(REPORT / "time_cluster_model_selection.csv"),
    }
    assignments = data["assignments"]
    if len(assignments) != 172 or assignments["subject"].nunique() != 5:
        raise ValueError("Expected 172 units from five patients")
    if set(assignments["peak_latency_class"]) != {"Early", "Middle", "Late"}:
        raise ValueError("Unexpected fixed latency labels")
    if set(assignments["time_cluster_label"]) != {"T1", "T2", "T3"}:
        raise ValueError("Unexpected K=3 GMM labels")
    return data


def draw_peak_distribution(
    ax: plt.Axes,
    assignments: pd.DataFrame,
    k3_components: pd.DataFrame,
    k4_components: pd.DataFrame,
) -> None:
    peaks = assignments["qualifying_peak_ms"].to_numpy(float)
    bins = np.arange(0, 826, 25)
    counts, edges = np.histogram(peaks, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [EARLY if x < 267 else MIDDLE if x < 533 else LATE for x in centers]
    ax.bar(centers, counts, width=22, color=colors, edgecolor="white", linewidth=0.35)

    ax.axvspan(0, 267, color=EARLY, alpha=0.14, lw=0)
    ax.axvspan(267, 533, color=MIDDLE, alpha=0.10, lw=0)
    ax.axvspan(533, 800, color=LATE, alpha=0.08, lw=0)
    for boundary in (267, 533):
        ax.axvline(boundary, color=GRAY, lw=0.75, ls=":", zorder=1)

    ymax = max(counts) * 1.23
    ax.set_ylim(0, ymax)
    fixed_counts = assignments["peak_latency_class"].value_counts()
    for x, label in [(133, "Early"), (400, "Middle"), (666, "Late")]:
        ax.text(
            x,
            ymax * 0.94,
            f"{label}\nn={int(fixed_counts[label])}",
            ha="center",
            va="top",
            color=INK,
            fontsize=PT - 0.2,
        )

    for row in k3_components.itertuples(index=False):
        mean = float(row.component_mean_ms)
        ax.axvline(mean, color=K3, lw=1.0, ls="--", zorder=3)
        ax.plot(mean, ymax * 0.72, marker="v", color=K3, ms=3.8, clip_on=False)
        ax.text(
            mean,
            ymax * 0.67,
            f"{mean:.0f}",
            color="#9B741C",
            ha="center",
            va="top",
            fontsize=PT - 0.4,
        )

    edge = k4_components.sort_values("component_mean_ms").iloc[-1]
    ax.annotate(
        f"K=4 adds a {edge.component_mean_ms:.0f}-ms\nedge component (n=12)",
        xy=(edge.component_mean_ms, ymax * 0.30),
        xytext=(585, ymax * 0.54),
        ha="left",
        va="center",
        color=K4,
        fontsize=PT - 0.5,
        arrowprops={"arrowstyle": "-", "color": K4, "lw": 0.8},
    )
    ax.set_xlim(0, 800)
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlabel("Peak latency after function-word onset (ms)")
    ax.set_ylabel("Units")
    ax.set_title("Fixed bands and patient-robust K=3 means", loc="left", fontweight="bold")
    ax.text(
        0.0,
        -0.30,
        "Dashed lines: K=3 GMM means (156, 393 and 656 ms).",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=GRAY,
        fontsize=PT - 0.5,
    )


def draw_bic(ax: plt.Axes, selection: pd.DataFrame) -> None:
    ks = selection["k"].to_numpy(int)
    delta = selection["delta_bic"].to_numpy(float)
    colors = [K3 if k == 3 else K4 if k == 4 else "#899694" for k in ks]
    ax.axhspan(0, 2, color="#DCEDE8", zorder=0)
    ax.axhspan(2, 6, color="#F3E9CE", zorder=0)
    ax.axhspan(6, 10, color="#F6DECF", zorder=0)
    ax.axhspan(10, max(22, delta.max() + 1), color="#F3D3D0", zorder=0)
    for k, value, color in zip(ks, delta, colors):
        ax.vlines(k, 0, value, color=color, lw=1.8)
        ax.scatter(k, value, s=30, facecolor=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(
            k,
            value + 0.72,
            f"{value:.2f}",
            color=color,
            ha="center",
            va="bottom",
            fontsize=PT - 0.6,
            fontweight="bold" if k in (3, 4) else "normal",
        )
    ax.set_xlim(0.6, 6.4)
    ax.set_ylim(0, 22.5)
    ax.set_xticks(ks)
    ax.set_xlabel("GMM components, K")
    ax.set_ylabel(r"$\Delta$BIC (lower is better)")
    ax.set_title("Pooled BIC prefers K=4", loc="left", fontweight="bold")
    ax.text(
        0.06,
        0.95,
        "K=4 weight 81.4%\nK=3 weight 4.5%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=PT - 0.5,
    )


def draw_lopo(ax: plt.Axes, selection: pd.DataFrame, lopo: pd.DataFrame) -> None:
    rows = selection.set_index("k")
    metrics = [
        ("Agreement", "agreement_fraction", "lopo_overall_agreement"),
        ("Adjusted\nRand index", "ari_vs_full_solution", "lopo_overall_ari"),
    ]
    width = 0.28
    x = np.arange(len(metrics), dtype=float)
    jitter = np.linspace(-0.045, 0.045, 5)
    for offset, k, color in [(-width / 2, 3, K3), (width / 2, 4, K4)]:
        overall = np.array([float(rows.loc[k, aggregate]) for _, _, aggregate in metrics])
        ax.bar(x + offset, overall, width=width, color=color, alpha=0.88, zorder=1, label=f"K={k}")
        folds = lopo[lopo["k"].eq(k)].sort_values("held_out_subject")
        for index, (_, fold_col, _) in enumerate(metrics):
            values = folds[fold_col].to_numpy(float)
            ax.scatter(
                np.full(values.size, x[index] + offset) + jitter[: values.size],
                values,
                s=13,
                facecolor="white",
                edgecolor=color,
                linewidth=0.75,
                zorder=3,
            )
            display_value = f"{100 * overall[index]:.1f}%" if index == 0 else f"{overall[index]:.3f}"
            ax.text(
                x[index] + offset,
                max(0.07, overall[index] - 0.10),
                display_value,
                ha="center",
                va="center",
                color="white",
                fontsize=PT - 1.0,
                fontweight="bold",
            )
    ax.set_ylim(0, 1.13)
    ax.set_xlim(-0.45, 1.48)
    ax.set_xticks(x, [label for label, _, _ in metrics])
    ax.set_ylabel("Leave-one-patient-out score")
    ax.set_title("K=3 is more patient-robust", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.96,
        "K=3",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=K3,
        fontsize=PT - 0.4,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.88,
        "K=4",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=K4,
        fontsize=PT - 0.4,
        fontweight="bold",
    )


def draw_concordance(ax: plt.Axes, assignments: pd.DataFrame) -> float:
    fixed_order = ["Early", "Middle", "Late"]
    gmm_order = ["T1", "T2", "T3"]
    table = pd.crosstab(assignments["peak_latency_class"], assignments["time_cluster_label"])
    table = table.reindex(index=fixed_order, columns=gmm_order, fill_value=0)
    matrix = table.to_numpy(int)
    concordance = float(np.trace(matrix) / matrix.sum())

    cmap = LinearSegmentedColormap.from_list("soft_blue", ["#F5F7F7", "#2C7FB8"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(matrix[i, j] / matrix.max()),
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() * 0.45 else INK,
                fontsize=PT + 0.4,
                fontweight="bold" if i == j else "normal",
            )
    ax.set_xticks(range(3), ["T1", "T2", "T3"])
    ax.set_yticks(range(3), fixed_order)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("K=3 GMM assignment")
    ax.set_ylabel("Fixed band")
    ax.set_title(
        f"Fixed bands agree with K=3 ({100 * concordance:.1f}%)",
        loc="left",
        fontweight="bold",
    )
    return concordance


def draw_session_composition(ax: plt.Axes, composition: pd.DataFrame) -> None:
    session_order = list(dict.fromkeys(composition["run"].tolist()))
    labels = {
        "gz001t_g0_actpas_run1": "P1 run 1",
        "gz002_g0_actpas_run2": "P2 run 2",
        "gz003t2_g0_actpas_run1": "P3 run 1",
        "gz003t2_g0_actpas_run4": "P3 run 4",
        "gz004t1_g0_actpas_run0": "P4 run 0",
        "gz004t1_g3_actpas_run3": "P4 run 3",
        "gz005t0_g0_actpas_run0": "P5 run 0",
        "gz005t0_g3_actpas_run3": "P5 run 3",
    }
    matrix = (
        composition.pivot_table(index="run", columns="cluster", values="fraction", fill_value=0.0)
        .reindex(index=session_order, columns=CLUSTER_ORDER, fill_value=0.0)
    )
    counts = composition.groupby("run", sort=False)["n_units"].sum().reindex(session_order).astype(int)
    y = np.arange(len(session_order))
    left = np.zeros(len(session_order), dtype=float)
    for cluster in CLUSTER_ORDER:
        values = matrix[cluster].to_numpy(float)
        ax.barh(y, values, left=left, color=ARCHETYPE_COLORS[cluster], height=0.70, edgecolor="white", linewidth=0.25)
        left += values
    for index, count in enumerate(counts):
        ax.text(1.025, index, f"n={count}", ha="left", va="center", color=GRAY, fontsize=PT - 0.5)
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax.set_xlabel("Units within session (%)")
    ax.set_yticks(y, [labels[session] for session in session_order])
    ax.invert_yaxis()
    ax.set_title("Multiple timing bands recur across recordings", loc="left", fontweight="bold")
    handles = [Patch(facecolor=ARCHETYPE_COLORS[c], label=c.replace("A-", "A>P ").replace("P-", "A<P ").replace("E", "early").replace("M", "middle").replace("L", "late")) for c in CLUSTER_ORDER]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.48, -0.32), ncols=6, columnspacing=0.65, handlelength=0.9)


def export_source_data(data: dict[str, pd.DataFrame], concordance: float) -> None:
    assignments = data["assignments"]
    assignments[
        [
            "subject",
            "run",
            "unit_id",
            "qualifying_peak_ms",
            "peak_latency_class",
            "time_cluster_label",
            "gmm_max_posterior",
        ]
    ].to_csv(SOURCE / "panel_a_d_peak_latency_assignments.csv", index=False)
    data["selection"].to_csv(SOURCE / "panel_b_gmm_model_selection.csv", index=False)
    data["lopo"].query("k in [3, 4]").to_csv(SOURCE / "panel_c_leave_one_patient_out.csv", index=False)
    data["composition"].to_csv(SOURCE / "panel_e_session_composition.csv", index=False)
    metadata = {
        "figure": "Proposed expanded Supplementary Figure 11",
        "n_units": 172,
        "n_patients": 5,
        "main_figure_fixed_bands_ms": {"Early": "0-266", "Middle": "267-532", "Late": "533-799"},
        "fixed_band_counts": assignments["peak_latency_class"].value_counts().to_dict(),
        "fixed_vs_k3_gmm_exact_concordance": concordance,
        "pooled_bic_best_k": 4,
        "k3_delta_bic": float(data["selection"].set_index("k").loc[3, "delta_bic"]),
        "k3_lopo_agreement": float(data["selection"].set_index("k").loc[3, "lopo_overall_agreement"]),
        "k4_lopo_agreement": float(data["selection"].set_index("k").loc[4, "lopo_overall_agreement"]),
        "k3_lopo_ari": float(data["selection"].set_index("k").loc[3, "lopo_overall_ari"]),
        "k4_lopo_ari": float(data["selection"].set_index("k").loc[4, "lopo_overall_ari"]),
        "full_envelope_max_silhouette": float(data["full_envelope"]["silhouette"].max()),
        "full_envelope_max_bootstrap_ari": float(data["full_envelope"]["bootstrap_ari_mean"].max()),
        "interpretation": "The three bands are a transparent descriptive summary, not three natural neuronal types.",
    }
    (SOURCE / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    data = load_data()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        4,
        left=0.065,
        right=0.975,
        bottom=0.14,
        top=0.94,
        wspace=0.78,
        hspace=0.64,
        height_ratios=[1.0, 1.05],
    )
    ax_a = fig.add_subplot(grid[0, 0:2])
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[0, 3])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1:4])

    draw_peak_distribution(ax_a, data["assignments"], data["k3_components"], data["k4_components"])
    draw_bic(ax_b, data["selection"])
    draw_lopo(ax_c, data["selection"], data["lopo"])
    concordance = draw_concordance(ax_d, data["assignments"])
    draw_session_composition(ax_e, data["composition"])

    for label, ax in zip("abcde", [ax_a, ax_b, ax_c, ax_d, ax_e]):
        panel_label(ax, label)

    fig.text(
        0.065,
        0.018,
        "Descriptive only: full 160-bin temporal envelopes did not form stable discrete clusters\n"
        "(maximum silhouette = 0.187; maximum bootstrap ARI = 0.449).",
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=PT - 0.2,
    )

    stem = OUT / "Fig_S11_temporal_band_validation"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    export_source_data(data, concordance)


if __name__ == "__main__":
    main()
