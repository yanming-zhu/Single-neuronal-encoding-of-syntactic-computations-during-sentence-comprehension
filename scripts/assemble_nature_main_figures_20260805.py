from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import make_nature_science_manuscript_package as base


OUT = ROOT / "reports" / "nature_science_draft" / "assembled_figures_20260805"
OUT.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def save_bundle(fig: plt.Figure, png_path: Path) -> None:
    """Save a review PNG plus editable vector versions."""
    stem = png_path.with_suffix("")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def letter(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def ptxt(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def fig1_distributed_population() -> Path:
    primary = base.get_primary_scores()
    best = pd.read_csv(base.NET / "active_passive_best_network_models.csv")
    best = best[
        (best.analysis_mode == "artifact_clean_le_0p2")
        & (best.unit_set == "good_only")
    ].copy()
    leave = pd.read_csv(base.NET / "active_passive_leave_pair_out_decoding.csv")
    leave = leave[leave.unit_set == "good_only"].copy()
    noun_units = pd.read_csv(base.NOUN / "noun_role_structure_encoding_units.csv")

    model_order = [
        "nested_single_unit",
        "network_top5_l2",
        "network_top10_l2",
        "network_top25_l2",
        "network_all_l2",
        "network_pca2_l2",
        "network_pca4_l2",
        "network_pca8_l2",
    ]
    model_labels = ["single", "top5", "top10", "top25", "all", "PCA2", "PCA4", "PCA8"]
    pivot = (
        primary.pivot_table(index="run", columns="model", values="auc", aggfunc="mean")
        .reindex(columns=model_order)
        .sort_index()
    )
    best_aligned = best.set_index("run").reindex(pivot.index).dropna(subset=["auc"])
    top5 = (
        primary[primary.model == "network_top5_l2"]
        .set_index("run")
        .reindex(best_aligned.index)["auc"]
    )
    leave_aligned = (
        leave.set_index("run")
        .reindex(best_aligned.index)["auc_leave_pair_out"]
    )

    fig = plt.figure(figsize=(9.0, 5.8))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.95, 1.62, 1.05],
        height_ratios=[1.0, 1.0],
        wspace=0.58,
        hspace=0.62,
    )
    ax_task = fig.add_subplot(gs[0, 0])
    ax_sample = fig.add_subplot(gs[1, 0])
    ax_models = fig.add_subplot(gs[:, 1])
    ax_gain = fig.add_subplot(gs[0, 2])
    ax_checks = fig.add_subplot(gs[1, 2])

    letter(ax_task, "a")
    ax_task.axis("off")
    ax_task.set_title("Role-reversal sentence assay", loc="left", fontsize=10)
    ax_task.text(0.02, 0.82, "Active", color=base.COLORS["blue"], fontweight="bold", fontsize=9)
    ax_task.text(0.02, 0.69, "N1   ba/gei/jiang   N2   V", family="DejaVu Sans Mono", fontsize=7.2)
    ax_task.text(0.02, 0.56, "N1 agent-like; N2 patient-like", color=base.COLORS["gray"], fontsize=6.7)
    ax_task.text(0.02, 0.38, "Passive", color=base.COLORS["red"], fontweight="bold", fontsize=9)
    ax_task.text(0.02, 0.25, "N2   bei/rang/jiao   N1   V", family="DejaVu Sans Mono", fontsize=7.2)
    ax_task.text(0.02, 0.12, "roles reversed", color=base.COLORS["gray"], fontsize=6.7)
    ax_task.text(
        0.02,
        -0.03,
        "Construction and function-word identity covary.",
        color=base.COLORS["orange"],
        fontsize=6.2,
    )

    letter(ax_sample, "b")
    by_patient = noun_units.groupby(["patient", "ks_label"]).size().unstack(fill_value=0)
    for col in ("good", "mua"):
        if col not in by_patient:
            by_patient[col] = 0
    x = np.arange(len(by_patient))
    ax_sample.bar(
        x,
        by_patient["good"],
        color=base.COLORS["blue"],
        label="good (461)",
    )
    ax_sample.bar(
        x,
        by_patient["mua"],
        bottom=by_patient["good"],
        color=base.COLORS["orange"],
        label="MUA (258)",
    )
    ax_sample.set_xticks(x)
    ax_sample.set_xticklabels(by_patient.index, rotation=30, ha="right", fontsize=6.5)
    ax_sample.set_ylabel("unit entries")
    ax_sample.set_ylim(0, 310)
    ax_sample.set_title("Recordings: 5 patients, 8 runs", loc="left", fontsize=10)
    ax_sample.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")

    letter(ax_models, "c")
    image = ax_models.imshow(
        pivot.values,
        vmin=0.25,
        vmax=0.85,
        cmap="viridis",
        aspect="auto",
    )
    ax_models.set_xticks(np.arange(len(model_order)))
    ax_models.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=7)
    ax_models.set_yticks(np.arange(pivot.shape[0]))
    ax_models.set_yticklabels([base.short_run(run) for run in pivot.index], fontsize=7)
    ax_models.set_title("Cross-validated decoding across runs", loc="left", fontsize=10)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.values[row, col]
            if np.isfinite(value):
                color = "white" if value > 0.64 or value < 0.42 else "black"
                ax_models.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.4, color=color)
    colorbar = fig.colorbar(image, ax=ax_models, shrink=0.62, pad=0.025)
    colorbar.set_label("AUC")
    ax_models.text(
        0.02,
        -0.18,
        "Artifact-clean good units; chance AUC = 0.5",
        transform=ax_models.transAxes,
        fontsize=6.5,
        color=base.COLORS["gray"],
    )

    letter(ax_gain, "d")
    for _, row in best_aligned.iterrows():
        ax_gain.plot(
            [0, 1],
            [row.nested_single_unit_auc, row.auc],
            color=base.COLORS["light_gray"],
            lw=1.1,
            zorder=1,
        )
    ax_gain.scatter(
        np.zeros(len(best_aligned)),
        best_aligned.nested_single_unit_auc,
        facecolor="white",
        edgecolor=base.COLORS["gray"],
        s=28,
        zorder=2,
    )
    ax_gain.scatter(
        np.ones(len(best_aligned)),
        best_aligned.auc,
        color=base.COLORS["blue"],
        s=28,
        zorder=2,
    )
    ax_gain.axhline(0.5, color=base.COLORS["gray"], ls="--", lw=0.8)
    ax_gain.set_xticks([0, 1])
    ax_gain.set_xticklabels(["nested\nsingle", "descriptive best\npopulation"], fontsize=7)
    ax_gain.set_ylim(0.25, 0.88)
    ax_gain.set_ylabel("AUC")
    ax_gain.set_title("Population readout gain", loc="left", fontsize=10)
    p_best = wilcoxon(best_aligned.auc - 0.5, alternative="greater").pvalue
    p_gain = wilcoxon(best_aligned.auc_gain_vs_nested_single_unit, alternative="greater").pvalue
    ax_gain.text(
        0.03,
        0.95,
        f"median AUC = {best_aligned.auc.median():.2f}, P = {ptxt(p_best)}\n"
        f"median gain = {best_aligned.auc_gain_vs_nested_single_unit.median():.2f}, P = {ptxt(p_gain)}",
        transform=ax_gain.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
    )

    letter(ax_checks, "e")
    checks = pd.DataFrame(
        {
            "fixed top5": top5,
            "descriptive best": best_aligned.auc,
            "leave-pair-out": leave_aligned,
        }
    )
    for _, row in checks.iterrows():
        ax_checks.plot(
            np.arange(3),
            row.values.astype(float),
            color=base.COLORS["light_gray"],
            lw=0.8,
            alpha=0.9,
        )
    point_colors = [base.COLORS["purple"], base.COLORS["blue"], base.COLORS["orange"]]
    for idx, col in enumerate(checks.columns):
        vals = checks[col].dropna().astype(float)
        jitter = np.linspace(-0.06, 0.06, len(vals))
        ax_checks.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            s=22,
            color=point_colors[idx],
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
        median = vals.median()
        ax_checks.plot([idx - 0.16, idx + 0.16], [median, median], color=base.COLORS["dark"], lw=1.4)
        ax_checks.text(idx, 0.82, f"{median:.2f}", ha="center", fontsize=7)
    ax_checks.axhline(0.5, color=base.COLORS["gray"], ls="--", lw=0.8)
    ax_checks.set_xticks(range(3))
    ax_checks.set_xticklabels(["fixed\ntop5", "best\nmodel", "leave\npair out"], fontsize=7)
    ax_checks.set_ylim(0.25, 0.86)
    ax_checks.set_ylabel("AUC")
    ax_checks.set_title("Fixed and held-out checks", loc="left", fontsize=10)

    fig.suptitle(
        "Distributed population activity distinguishes role-reversal constructions",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.88, wspace=0.62, hspace=0.68)
    path = OUT / "fig1_distributed_population.png"
    save_bundle(fig, path)
    return path


def fig2_temporal_dynamics() -> Path:
    summary = json.loads(
        (base.TIME / "active_passive_time_generalization_summary.json").read_text()
    )
    run_summary = pd.read_csv(
        base.TIME / "active_passive_time_generalization_run_summary.csv"
    ).sort_values("run")
    arrays = np.load(
        base.TIME / "active_passive_time_generalization_matrices.npz",
        allow_pickle=True,
    )
    centers = arrays["centers_sec"]
    diagonal = arrays["diag_auc"]
    median_diagonal = np.nanmedian(diagonal, axis=0)
    median_matrix = np.nanmedian(arrays["auc_matrices"], axis=0)

    fig = plt.figure(figsize=(8.5, 6.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.38, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.48,
        hspace=0.58,
    )
    ax_time = fig.add_subplot(gs[0, 0])
    ax_peaks = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[1, 0])
    ax_summary = fig.add_subplot(gs[1, 1])

    letter(ax_time, "a")
    for row in diagonal:
        ax_time.plot(centers, row, color=base.COLORS["light_gray"], lw=0.8, alpha=0.8)
    ax_time.plot(centers, median_diagonal, color=base.COLORS["blue"], lw=2.2, label="median")
    ax_time.axhline(0.5, color=base.COLORS["gray"], ls="--", lw=0.8)
    ax_time.axvspan(0, 0.8, color="#BFD3EA", alpha=0.23)
    ax_time.set_xlim(-0.75, 1.75)
    ax_time.set_ylim(0.25, 0.95)
    ax_time.set_xlabel("time from function word (s)")
    ax_time.set_ylabel("AUC")
    ax_time.set_title("Time-resolved fixed-top5 decoding", loc="left", fontsize=10)
    ax_time.text(
        0.02,
        0.95,
        f"descriptive median peak = {summary['median_diagonal_peak']['median_auc']:.2f} "
        f"at {summary['median_diagonal_peak']['time_sec']:.2f} s",
        transform=ax_time.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
    )

    letter(ax_peaks, "b")
    ax_peaks.scatter(
        run_summary.peak_diag_time_sec,
        run_summary.max_diag_auc_primary_window,
        s=34,
        color=base.COLORS["blue"],
        edgecolor="white",
        linewidth=0.4,
    )
    label_offsets = [(4, 5), (4, -8), (4, 5), (-34, 5), (4, -8), (-36, 5), (-35, -8)]
    for (_, row), offset in zip(run_summary.iterrows(), label_offsets):
        ax_peaks.annotate(
            base.short_run(row.run),
            (row.peak_diag_time_sec, row.max_diag_auc_primary_window),
            xytext=offset,
            textcoords="offset points",
            fontsize=6,
            color=base.COLORS["dark"],
        )
    ax_peaks.axhline(0.5, color=base.COLORS["gray"], ls="--", lw=0.8)
    ax_peaks.set_xlim(-0.03, 0.86)
    ax_peaks.set_ylim(0.62, 0.93)
    ax_peaks.set_xlabel("within-window peak time (s)")
    ax_peaks.set_ylabel("peak AUC")
    ax_peaks.set_title("Run-level peak timing", loc="left", fontsize=10)
    ax_peaks.text(
        0.03,
        0.05,
        "Peak summaries are descriptive.",
        transform=ax_peaks.transAxes,
        fontsize=6.5,
        color=base.COLORS["gray"],
    )

    letter(ax_matrix, "c")
    matrix_image = ax_matrix.imshow(
        median_matrix,
        extent=[centers[0], centers[-1], centers[-1], centers[0]],
        cmap="viridis",
        vmin=0.35,
        vmax=0.75,
        aspect="auto",
    )
    ax_matrix.plot(
        [centers[0], centers[-1]],
        [centers[0], centers[-1]],
        color="white",
        lw=0.8,
        alpha=0.9,
    )
    ax_matrix.set_xlabel("test time (s)")
    ax_matrix.set_ylabel("train time (s)")
    ax_matrix.set_title("Median cross-temporal generalization", loc="left", fontsize=10)
    colorbar = fig.colorbar(matrix_image, ax=ax_matrix, shrink=0.72, pad=0.025)
    colorbar.set_label("AUC")

    letter(ax_summary, "d")
    diag_mean = run_summary.mean_diag_auc_primary_window.astype(float)
    cross_mean = run_summary.mean_cross_temporal_auc_primary_window.astype(float)
    for left, right in zip(diag_mean, cross_mean):
        ax_summary.plot([0, 1], [left, right], color=base.COLORS["light_gray"], lw=1.0)
    ax_summary.scatter(
        np.zeros(len(diag_mean)),
        diag_mean,
        s=28,
        color=base.COLORS["blue"],
        edgecolor="white",
        linewidth=0.3,
    )
    ax_summary.scatter(
        np.ones(len(cross_mean)),
        cross_mean,
        s=28,
        color=base.COLORS["purple"],
        edgecolor="white",
        linewidth=0.3,
    )
    ax_summary.axhline(0.5, color=base.COLORS["gray"], ls="--", lw=0.8)
    ax_summary.set_xticks([0, 1])
    ax_summary.set_xticklabels(["mean diagonal", "mean cross-time"], fontsize=7)
    ax_summary.set_ylim(0.35, 0.70)
    ax_summary.set_ylabel("AUC in 0–0.8 s window")
    ax_summary.set_title("Limited window-averaged stability", loc="left", fontsize=10)
    diag_stats = summary["primary_window_summary"]["mean_diag_auc_0_to_0p8"]
    cross_stats = summary["primary_window_summary"]["mean_cross_temporal_auc_0_to_0p8"]
    ax_summary.text(
        0.02,
        0.04,
        f"diagonal median = {diag_stats['median']:.2f}, P = {ptxt(diag_stats['wilcoxon_p_greater'])}\n"
        f"cross-time median = {cross_stats['median']:.2f}, P = {ptxt(cross_stats['wilcoxon_p_greater'])}",
        transform=ax_summary.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
    )

    fig.suptitle(
        "Construction-linked population decoding is transient and weakly stable",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.10, top=0.88, wspace=0.48, hspace=0.62)
    path = OUT / "fig2_temporal_dynamics.png"
    save_bundle(fig, path)
    return path


def assemble_fig3_and_fig4() -> tuple[Path, Path]:
    base.FIG = OUT
    base.savefig = save_bundle
    fig3 = base.fig3_role_state_geometry()
    fig4 = base.fig4_specificity_controls()
    return fig3, fig4


def make_montage(paths: list[Path]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    for index, (ax, path) in enumerate(zip(axes.ravel(), paths), start=1):
        ax.imshow(mpimg.imread(path))
        ax.set_title(f"Figure {index}", loc="left", fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Nature manuscript main-figure set | assembled preview", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    montage_path = OUT / "fig1_to_fig4_montage.png"
    fig.savefig(montage_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return montage_path


def main() -> None:
    fig1 = fig1_distributed_population()
    fig2 = fig2_temporal_dynamics()
    fig3, fig4 = assemble_fig3_and_fig4()
    montage = make_montage([fig1, fig2, fig3, fig4])
    outputs = {
        "fig1": str(fig1),
        "fig2": str(fig2),
        "fig3": str(fig3),
        "fig4": str(fig4),
        "montage": str(montage),
    }
    (OUT / "assembled_figure_manifest.json").write_text(
        json.dumps(outputs, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
