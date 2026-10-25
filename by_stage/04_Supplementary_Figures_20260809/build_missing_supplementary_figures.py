#!/usr/bin/env python3
"""Build four panels for the historical supplementary base layout.

This retains the August 2026 layout and its original panel names; it is not a
final manuscript figure-numbering entry point. Use
manuscript/revision_20260921_s6s8/audit/build_supplementary_discriminability.py
for the later S6-S8 discriminability panels and
scripts/make_supp_fig11_temporal_band_validation_20260814.py for current
S10/S12 sensitivity components. See the package README for the figure map.

The script is deliberately source-table driven. It does not rerun neural
analyses; it assembles already-audited patient/run/stimulus endpoints into
submission-grade figures and writes a compact source-data table for each one.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "new_figures"
DATA_DIR = HERE / "source_data"

SESSION = ROOT / "reports/active_passive_unit_deck_20260722/tables/session_unit_summary.csv"
FIG2_SOURCE = ROOT / "reports/nature_science_draft/assembled_figures_20260805/source_data/fig2_population_180mm_revised"
SPEC = ROOT / "reports/three_paper_borrowed_analyses_20260724/tables/specification_curve.csv"
SPEC_PATIENT = ROOT / "reports/three_paper_borrowed_analyses_20260724/tables/specification_curve_patient_values.csv"
POP = ROOT / "neuropixel2025/data/combined_active_passive_population_candidate_panels"
ACOUSTIC = ROOT / "reports/intro_driven_structural_transformation_20260807/source_data/stimulus_acoustic_features.csv"
VARIANCE = ROOT / "reports/nature_science_draft/supplementary/tables/supp_table_5_variance_partitioning.csv"

MM = 1 / 25.4
FIGSIZE = (180 * MM, 130 * MM)
INK = "#16312C"
DEEP = "#102A26"
TEAL = "#147D70"
TEAL_PALE = "#DDEEEA"
CORAL = "#D9644A"
GOLD = "#D9A01C"
BLUE = "#437A9B"
GRAY = "#6D7773"
GRAY2 = "#AAB3AF"
LIGHT = "#E7ECE9"
WHITE = "#FFFFFF"
PATIENT_COLORS = {
    "gz001": "#2C7FB8",
    "gz002": "#41AE76",
    "gz003": "#D9A01C",
    "gz004": "#D9644A",
    "gz005": "#7A5AA6",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY2)
    ax.spines["bottom"].set_color(GRAY2)
    ax.tick_params(color=GRAY2, labelcolor=DEEP)
    if grid:
        ax.grid(axis=grid, color=LIGHT, linewidth=0.55, zorder=0)


def finish_figure(fig: plt.Figure, stem: str, metadata: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        FIG_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    metadata = {
        **metadata,
        "final_size_mm": [180, 130],
        "formats": ["png", "svg", "pdf", "tiff"],
        "tiff_dpi": 600,
    }
    (DATA_DIR / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plt.close(fig)


def draw_patient_median(ax: plt.Axes, x: float, values: np.ndarray, color: str = INK) -> None:
    med = float(np.median(values))
    ax.plot([x - 0.16, x + 0.16], [med, med], color=color, lw=2.2, zorder=5)


def build_recording_yield() -> None:
    frame = pd.read_csv(SESSION)
    frame["short_run"] = frame["display"].str.replace(" · ", "\n", regex=False)
    frame["primary_decoder_included"] = ~(
        (frame["patient"] == "gz003") & frame["run"].str.endswith("run1")
    )
    frame["primary_decoder_exclusion_reason"] = np.where(
        frame["primary_decoder_included"],
        "",
        "insufficient artifact-clean trials for fixed cross-validation (7 total; 2 active, 5 passive)",
    )
    total_units = int(frame["units"].sum())
    total_good = int(frame["good"].sum())
    total_mua = int(frame["mua"].sum())
    decoder_good = int(frame.loc[frame["primary_decoder_included"], "good"].sum())

    source = frame.copy()
    source["overall_registered_total"] = total_units
    source["overall_good_total"] = total_good
    source["overall_mua_total"] = total_mua
    source["primary_decoder_good_total"] = decoder_good
    source.to_csv(DATA_DIR / "S2_recording_yield_and_inclusion_source_data.csv", index=False)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1.0, 1.05],
        left=0.08,
        right=0.97,
        top=0.94,
        bottom=0.12,
        wspace=0.34,
        hspace=0.46,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    y = np.arange(len(frame))[::-1]
    ax_a.barh(y, frame["mua"], color="#C8D1CD", height=0.67, label="MUA")
    ax_a.barh(y, frame["good"], left=frame["mua"], color=TEAL, height=0.67, label="good")
    for yi, total in zip(y, frame["units"]):
        ax_a.text(total + 6, yi, f"{int(total)}", va="center", fontsize=5.8, color=GRAY)
    ax_a.set_yticks(y, frame["short_run"])
    ax_a.set_xlabel("Registered unit/run entries")
    ax_a.set_title("Session-level cluster yield", loc="left", fontweight="bold", color=INK, pad=4)
    ax_a.set_xlim(0, frame["units"].max() * 1.15)
    clean_axis(ax_a, "x")
    ax_a.legend(frameon=False, ncol=2, loc="lower right", handlelength=1.2, columnspacing=1.0)
    panel_label(ax_a, "a")

    panel_label(ax_b, "b")
    ax_b.set_title("Analysis inclusion funnel", loc="left", fontweight="bold", color=INK, pad=4)
    funnel = [
        ("registered", total_units, 0.90, "#DDE3E0", INK),
        ("Kilosort good", total_good, 0.64, TEAL_PALE, TEAL),
        ("primary decoder\n7 sessions", decoder_good, 0.47, "#F6EBCB", "#8A6515"),
    ]
    for idx, (label, value, width, fill, edge) in enumerate(funnel):
        x0 = 0.5 - width / 2
        y0 = 0.72 - idx * 0.30
        ax_b.add_patch(
            FancyBboxPatch(
                (x0, y0), width, 0.20,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                transform=ax_b.transAxes,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.8,
            )
        )
        ax_b.text(0.5, y0 + 0.125, f"{value:,}", transform=ax_b.transAxes,
                  ha="center", va="center", fontsize=10, fontweight="bold", color=edge)
        ax_b.text(0.5, y0 + 0.045, label, transform=ax_b.transAxes,
                  ha="center", va="center", fontsize=6, color=DEEP, linespacing=0.95)
    ax_b.text(0.5, 0.015, "unit/run entries; not independent biological replicates",
              transform=ax_b.transAxes, ha="center", va="bottom", fontsize=5.5, color=GRAY)
    ax_b.set_axis_off()

    panel_label(ax_c, "c")
    ax_c.set_title("Primary decoder session map", loc="left", fontweight="bold", color=INK, pad=4)
    patients = ["gz001", "gz002", "gz003", "gz004", "gz005"]
    ax_c.set_xlim(-0.3, 2.35)
    ax_c.set_ylim(-0.6, 4.6)
    for row, patient in enumerate(patients[::-1]):
        sub = frame[frame["patient"] == patient].reset_index(drop=True)
        for col, item in sub.iterrows():
            included = bool(item["primary_decoder_included"])
            ax_c.scatter(
                col,
                row,
                s=48,
                marker="s",
                facecolor=TEAL if included else WHITE,
                edgecolor=TEAL if included else CORAL,
                linewidth=1.0,
                zorder=3,
            )
            run_label = item["display"].split("run")[-1]
            ax_c.text(col, row, run_label, ha="center", va="center", fontsize=5.4,
                      color=WHITE if included else CORAL, fontweight="bold")
            ax_c.text(col + 0.22, row, f"{int(item['good'])} good", ha="left", va="center",
                      fontsize=5.7, color=DEEP)
    ax_c.set_yticks(range(5), patients[::-1])
    ax_c.set_xticks([])
    ax_c.spines[:].set_visible(False)
    ax_c.tick_params(length=0)
    ax_c.text(0.02, -0.17, "filled: included", transform=ax_c.transAxes, fontsize=5.7, color=TEAL)
    ax_c.text(0.54, -0.17, "open: excluded", transform=ax_c.transAxes, fontsize=5.7, color=CORAL)

    finish_figure(
        fig,
        "S2_recording_yield_and_inclusion",
        {
            "title": "Recording yield and analysis inclusion",
            "sources": [str(SESSION)],
            "biological_replicates": "five patients",
            "nested_observations": "eight recording sessions and unit/run entries",
            "key_values": {
                "registered_unit_run_entries": total_units,
                "good_entries": total_good,
                "mua_entries": total_mua,
                "primary_decoder_sessions": 7,
                "primary_decoder_good_entries": decoder_good,
            },
        },
    )


def build_decoder_sensitivity() -> None:
    spec = pd.read_csv(SPEC).sort_values("curve_order")
    spec_patient = pd.read_csv(SPEC_PATIENT)
    patient_medians = pd.read_csv(FIG2_SOURCE / "panels_a_b_patient_medians.csv")
    held = pd.read_csv(FIG2_SOURCE / "panel_b_sentence_pair_transfer_patient_values.csv")
    held_inf = pd.read_csv(FIG2_SOURCE / "panel_b_sentence_pair_transfer_inference.csv")

    spec_export = spec.assign(record_type="specification")
    primary_export = patient_medians.assign(record_type="primary_patient_endpoint")
    held_export = held.assign(record_type="held_pair_patient_endpoint")
    inference_export = held_inf.assign(record_type="held_pair_inference")
    pd.concat([spec_export, primary_export, held_export, inference_export], ignore_index=True, sort=False).to_csv(
        DATA_DIR / "S5_decoder_specification_and_inference_source_data.csv", index=False
    )

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.62, 1.0],
        left=0.08,
        right=0.97,
        top=0.94,
        bottom=0.13,
        wspace=0.34,
        hspace=0.50,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    combo_colors = {
        ("all_trials", "all_clusters"): GRAY2,
        ("all_trials", "good_only"): BLUE,
        ("artifact_clean_le_0p2", "all_clusters"): CORAL,
        ("artifact_clean_le_0p2", "good_only"): TEAL,
    }
    for _, row in spec.iterrows():
        x = int(row["curve_order"])
        primary = bool(row["is_primary_fixed"])
        color = GOLD if primary else combo_colors[(row["analysis_mode"], row["unit_set"])]
        marker = "*" if primary else "o"
        size = 68 if primary else 18
        ax_a.vlines(x, row["bootstrap_ci_low"], row["bootstrap_ci_high"], color=color,
                    lw=0.75, alpha=0.75, zorder=1)
        ax_a.scatter(x, row["patient_median_auc"], s=size, marker=marker, color=color,
                     edgecolor=WHITE if primary else color, linewidth=0.45, zorder=3)
    ax_a.axhline(0.5, color=INK, ls=(0, (3, 2)), lw=0.8)
    ax_a.set_xlim(0.2, 32.8)
    ax_a.set_ylim(0.05, 0.88)
    ax_a.set_xticks([1, 8, 16, 24, 32])
    ax_a.set_xlabel("Correlated specifications ordered by patient-median AUC")
    ax_a.set_ylabel("Patient-median AUC (bootstrap 95% CI)")
    ax_a.set_title("32-specification sensitivity curve", loc="left", fontweight="bold", color=INK, pad=4)
    clean_axis(ax_a, "y")
    ax_a.annotate(
        "prespecified fixed top-5\nmedian 0.69",
        xy=(32, 0.69),
        xytext=(24.1, 0.80),
        fontsize=6,
        color="#8A6515",
        arrowprops={"arrowstyle": "-", "color": GOLD, "lw": 0.8},
        ha="left",
    )
    ax_a.text(0.98, 0.02, "range 0.46-0.69; 27/32 medians > 0.5",
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=5.8, color=GRAY)
    handles = [
        mpl.lines.Line2D([], [], color=color, marker="o", lw=0, markersize=4, label=label)
        for color, label in [
            (GRAY2, "all trials · all clusters"),
            (BLUE, "all trials · good only"),
            (CORAL, "artifact-clean · all clusters"),
            (TEAL, "artifact-clean · good only"),
            (GOLD, "primary fixed top-5"),
        ]
    ]
    ax_a.legend(
        handles=handles,
        frameon=True,
        facecolor=WHITE,
        edgecolor="none",
        framealpha=0.95,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.04),
        handletextpad=0.35,
        labelspacing=0.35,
    )
    panel_label(ax_a, "a")

    panel_label(ax_b, "b")
    ax_b.set_title("Primary patient endpoints", loc="left", fontweight="bold", color=INK, pad=4)
    x_positions = [0, 1]
    cols = ["nested_single_unit_auc", "fixed_top5_auc"]
    for _, row in patient_medians.iterrows():
        vals = row[cols].to_numpy(dtype=float)
        ax_b.plot(x_positions, vals, color="#B6C0BC", lw=0.75, zorder=1)
        ax_b.scatter(x_positions, vals, s=22, color=BLUE, edgecolor=WHITE, linewidth=0.4, zorder=3)
    for x, col in zip(x_positions, cols):
        draw_patient_median(ax_b, x, patient_medians[col].to_numpy(dtype=float))
    ax_b.axhline(0.5, color=INK, ls=(0, (3, 2)), lw=0.7)
    ax_b.set_xticks(x_positions, ["nested\nsingle unit", "fixed\ntop-5"])
    ax_b.set_ylabel("AUC")
    ax_b.set_ylim(0.28, 0.88)
    clean_axis(ax_b, "y")
    ax_b.text(0.98, 0.03, "top-5 median 0.69\nexact sign P = 0.1875",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=5.8, color=TEAL)

    panel_label(ax_c, "c")
    ax_c.set_title("Held-sentence-pair calibration", loc="left", fontweight="bold", color=INK, pad=4)
    held_pivot = held.pivot(index="subject", columns="model", values="auc").loc[
        ["gz001", "gz002", "gz003", "gz004", "gz005"]
    ]
    held_cols = ["nested_single_unit", "fixed_top5_population"]
    for subject, row in held_pivot.iterrows():
        vals = row[held_cols].to_numpy(dtype=float)
        ax_c.plot(x_positions, vals, color="#B6C0BC", lw=0.75, zorder=1)
        ax_c.scatter(x_positions, vals, s=22, color=BLUE,
                     edgecolor=WHITE, linewidth=0.4, zorder=3)
    for x, col in zip(x_positions, held_cols):
        draw_patient_median(ax_c, x, held_pivot[col].to_numpy(dtype=float))
    ax_c.axhline(0.5, color=INK, ls=(0, (3, 2)), lw=0.7)
    ax_c.set_xticks(x_positions, ["nested\nsingle unit", "fixed\ntop-5"])
    ax_c.set_ylabel("Held-pair AUC")
    ax_c.set_ylim(0.28, 0.92)
    clean_axis(ax_c, "y")
    ax_c.text(
        0.98,
        0.03,
        "top-5 median 0.66\npooled permutation P = 0.024\nexact patient sign P = 0.1875",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color=TEAL,
    )

    primary = spec.loc[spec["is_primary_fixed"].astype(bool)].iloc[0]
    finish_figure(
        fig,
        "S5_decoder_specification_and_inference",
        {
            "title": "Population-decoder specification and inference sensitivity",
            "sources": [str(SPEC), str(SPEC_PATIENT), str(FIG2_SOURCE)],
            "independent_inferential_units": "five patients",
            "key_values": {
                "specification_range_patient_median_auc": [
                    float(spec["patient_median_auc"].min()), float(spec["patient_median_auc"].max())
                ],
                "specifications_above_chance_median": int((spec["patient_median_auc"] > 0.5).sum()),
                "primary_fixed_top5_patient_median_auc": float(primary["patient_median_auc"]),
                "primary_exact_sign_p_greater": float(primary["exact_sign_p_greater"]),
                "held_pair_top5_patient_median_auc": 0.66,
                "held_pair_pooled_permutation_p": 0.024,
                "held_pair_exact_patient_sign_p": 0.1875,
            },
        },
    )


def build_population_diagnostics() -> None:
    ablation = pd.read_csv(POP / "ablation_group_summary.csv")
    ablation_pat = pd.read_csv(POP / "ablation_patient_values.csv")
    shuffle = pd.read_csv(POP / "trial_shuffle_patient_values.csv")
    shuffle_summary = pd.read_csv(POP / "trial_shuffle_group_summary.csv")
    distance = pd.read_csv(POP / "cv_population_distance_patient_values.csv")
    distance_summary = pd.read_csv(POP / "cv_population_distance_group_summary.csv")

    pd.concat(
        [
            ablation.assign(record_type="ablation_group"),
            ablation_pat.assign(record_type="ablation_patient"),
            shuffle.assign(record_type="condition_preserving_shuffle_patient"),
            shuffle_summary.assign(record_type="condition_preserving_shuffle_group"),
            distance.assign(record_type="cv_distance_patient"),
            distance_summary.assign(record_type="cv_distance_group"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(DATA_DIR / "S6_population_information_diagnostics_source_data.csv", index=False)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        left=0.08,
        right=0.97,
        top=0.94,
        bottom=0.13,
        wspace=0.35,
        hspace=0.50,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    x = ablation["remove_order"].to_numpy(dtype=float)
    ax_a.fill_between(
        x,
        ablation["ranked_removal_patient_q25_auc"].to_numpy(dtype=float),
        ablation["ranked_removal_patient_q75_auc"].to_numpy(dtype=float),
        color=CORAL,
        alpha=0.14,
        linewidth=0,
    )
    ax_a.fill_between(
        x,
        ablation["random_removal_patient_q25_auc"].to_numpy(dtype=float),
        ablation["random_removal_patient_q75_auc"].to_numpy(dtype=float),
        color=GRAY2,
        alpha=0.18,
        linewidth=0,
    )
    ax_a.plot(x, ablation["ranked_removal_patient_median_auc"], "o-", color=CORAL,
              lw=1.35, ms=4, label="remove train-ranked units")
    ax_a.plot(x, ablation["random_removal_patient_median_auc"], "o-", color=GRAY,
              lw=1.2, ms=3.7, label="remove random units")
    ax_a.axhline(0.5, color=INK, ls=(0, (3, 2)), lw=0.7)
    ax_a.set_xticks(x, ablation["remove_label"].astype(str))
    ax_a.set_xlabel("Units removed within each training fold")
    ax_a.set_ylabel("Patient-median AUC (IQR)")
    ax_a.set_ylim(0.25, 0.66)
    ax_a.set_title("Ranked versus random unit ablation", loc="left", fontweight="bold", color=INK, pad=4)
    clean_axis(ax_a, "y")
    ax_a.legend(frameon=False, loc="lower left")
    ax_a.text(0.98, 0.04, "ranked-removal disadvantage\nexact P = 0.03125 at 1, 2 and 5 units",
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=5.8, color=CORAL)
    panel_label(ax_a, "a")

    panel_label(ax_b, "b")
    ax_b.set_title("Condition-preserving trial shuffle", loc="left", fontweight="bold", color=INK, pad=4)
    model_positions = {"all": (0, 1), "top5": (2.4, 3.4)}
    for model, (x0, x1) in model_positions.items():
        sub = shuffle[shuffle["model"] == model]
        for _, row in sub.iterrows():
            vals = [row["simultaneous_auc"], row["trial_shuffled_median_auc"]]
            ax_b.plot([x0, x1], vals, color="#B6C0BC", lw=0.7, zorder=1)
            ax_b.scatter([x0, x1], vals, s=19, color=BLUE, edgecolor=WHITE, linewidth=0.35, zorder=3)
        draw_patient_median(ax_b, x0, sub["simultaneous_auc"].to_numpy(dtype=float))
        draw_patient_median(ax_b, x1, sub["trial_shuffled_median_auc"].to_numpy(dtype=float))
    ax_b.axhline(0.5, color=INK, ls=(0, (3, 2)), lw=0.7)
    ax_b.set_xticks([0, 1, 2.4, 3.4], ["simult.\nall", "shuffled\nall", "simult.\ntop-5", "shuffled\ntop-5"])
    ax_b.set_ylabel("AUC")
    ax_b.set_ylim(0.25, 0.88)
    clean_axis(ax_b, "y")
    ax_b.text(0.98, 0.03, "top-5 median Δ = 0.005\ntwo-sided exact P = 1.0",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=5.7, color=TEAL)

    panel_label(ax_c, "c")
    ax_c.set_title("Cross-validated population distance", loc="left", fontweight="bold", color=INK, pad=4)
    rng = np.random.default_rng(20260809)
    for x0, model in enumerate(["all", "top5"]):
        sub = distance[distance["model"] == model]
        jitter = rng.uniform(-0.07, 0.07, len(sub))
        for j, (_, row) in enumerate(sub.iterrows()):
            ax_c.scatter(x0 + jitter[j], row["cv_population_distance"], s=25,
                         color=BLUE, edgecolor=WHITE,
                         linewidth=0.4, zorder=3)
        draw_patient_median(ax_c, x0, sub["cv_population_distance"].to_numpy(dtype=float))
    ax_c.axhline(0, color=INK, ls=(0, (3, 2)), lw=0.7)
    ax_c.set_xticks([0, 1], ["all units", "top-5"])
    ax_c.set_ylabel("Cross-validated distance")
    ax_c.set_ylim(-0.18, 0.57)
    clean_axis(ax_c, "y")
    ax_c.text(0.98, 0.03, "top-5 median 0.148\none-sided exact P = 0.0625",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=5.7, color=TEAL)

    finish_figure(
        fig,
        "S6_population_information_diagnostics",
        {
            "title": "Population-information diagnostics",
            "sources": [str(POP)],
            "independent_inferential_units": "five patients",
            "interpretation_boundary": (
                "Train-fold ablation is computational rather than causal; the condition-preserving "
                "shuffle is a covariance diagnostic rather than a prospective label null."
            ),
            "key_values": {
                "top5_simultaneous_minus_shuffled_patient_median_auc": 0.005,
                "top5_shuffle_exact_two_sided_p": 1.0,
                "top5_cv_distance_patient_median": 0.1480967789103958,
                "top5_cv_distance_exact_p_greater": 0.0625,
            },
        },
    )


def build_stimulus_identifiability() -> None:
    acoustic = pd.read_csv(ACOUSTIC)
    variance = pd.read_csv(VARIANCE)
    sentences = acoustic[["sound_file", "voice", "function_word"]].drop_duplicates()
    marker_counts = (
        sentences.groupby(["function_word", "voice"]).size().rename("n_sentences").reset_index()
    )
    marker_order = ["把", "给", "将", "被", "让", "叫"]
    marker_labels = {"把": "BA", "给": "GEI", "将": "JIANG", "被": "BEI", "让": "RANG", "叫": "JIAO"}
    pivot = marker_counts.pivot(index="function_word", columns="voice", values="n_sentences").fillna(0)
    pivot = pivot.reindex(marker_order).reindex(columns=["active", "passive"], fill_value=0)

    numeric_features = [
        "log_rms",
        "spectral_centroid_hz",
        "zero_crossing_rate",
        "f0_hz",
        "voiced_fraction",
        "envelope_chunk_duration_sec",
    ]
    z = acoustic.copy()
    for col in numeric_features:
        z[col] = (z[col] - z[col].mean()) / z[col].std(ddof=0)
    paired = []
    for noun, sub in z.groupby("noun"):
        if set(sub["surface_position"]) != {"initial", "post_marker"}:
            continue
        initial = sub[sub["surface_position"] == "initial"].iloc[0]
        post = sub[sub["surface_position"] == "post_marker"].iloc[0]
        item = {"noun": noun}
        for col in numeric_features:
            item[f"delta_{col}"] = float(post[col] - initial[col])
        paired.append(item)
    acoustic_diff = pd.DataFrame(paired).sort_values("noun")

    onset_export = acoustic[["noun", "surface_position", "serial_onset_sec", "preceding_marker_present"]].copy()
    marker_export = marker_counts.assign(record_type="marker_by_voice")
    onset_export = onset_export.assign(record_type="noun_serial_onset")
    diff_export = acoustic_diff.assign(record_type="same_noun_standardized_acoustic_difference")
    var_export = variance[
        ["subject", "run", "n_units", "median_syntax_candidate_no_word", "median_syntax_after_word_identity"]
    ].assign(record_type="run_level_condition_candidate_cv_r2")
    pd.concat([marker_export, onset_export, diff_export, var_export], ignore_index=True, sort=False).to_csv(
        DATA_DIR / "S14_stimulus_acoustic_identifiability_source_data.csv", index=False
    )

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.88, 1.65],
        height_ratios=[0.90, 1.10],
        left=0.09,
        right=0.97,
        top=0.94,
        bottom=0.13,
        wspace=0.35,
        hspace=0.50,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    matrix = pivot.to_numpy(dtype=float)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("neutral_teal", [WHITE, TEAL_PALE, TEAL])
    ax_a.imshow(matrix, cmap=cmap, vmin=0, vmax=max(1, matrix.max()), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_a.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                      fontsize=7, color=WHITE if matrix[i, j] >= matrix.max() * 0.65 else INK,
                      fontweight="bold")
    ax_a.set_xticks([0, 1], ["active", "passive"])
    ax_a.set_yticks(range(6), [marker_labels[x] for x in marker_order])
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.set_title("Deterministic marker-class mapping", loc="left", fontweight="bold", color=INK, pad=4)
    ax_a.text(0.5, -0.19, "20 sentences; Cramer's V = 1.00", transform=ax_a.transAxes,
              ha="center", va="top", fontsize=5.7, color=GRAY)
    panel_label(ax_a, "a")

    panel_label(ax_b, "b")
    ax_b.set_title("Position is coupled to sentence-relative onset", loc="left", fontweight="bold", color=INK, pad=4)
    onset_pivot = acoustic.pivot_table(index="noun", columns="surface_position", values="serial_onset_sec", aggfunc="first")
    for noun, row in onset_pivot.iterrows():
        ax_b.plot([0, 1], [row["initial"], row["post_marker"]], color="#C4CCC8", lw=0.65, zorder=1)
        ax_b.scatter([0, 1], [row["initial"], row["post_marker"]], s=13,
                     color=[TEAL, CORAL], edgecolor=WHITE, linewidth=0.3, zorder=2)
    ax_b.set_xticks([0, 1], ["sentence-\ninitial", "post-marker"])
    ax_b.set_ylabel("Noun onset from sentence start (s)")
    ax_b.set_xlim(-0.35, 1.35)
    ax_b.set_ylim(-0.06, max(1.05, float(onset_pivot["post_marker"].max()) + 0.08))
    clean_axis(ax_b, "y")
    ax_b.text(0.98, 0.03, "20 matched nouns\npreceding marker: 0/20 vs 20/20",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=5.7, color=GRAY)

    panel_label(ax_c, "c")
    ax_c.set_title("Same-noun acoustic differences across positions", loc="left", fontweight="bold", color=INK, pad=4)
    labels = ["log RMS", "spectral\ncentroid", "zero-crossing\nrate", "F0", "voiced\nfraction", "chunk\nduration"]
    delta_cols = [f"delta_{x}" for x in numeric_features]
    rng = np.random.default_rng(20260809)
    for idx, col in enumerate(delta_cols):
        values = acoustic_diff[col].to_numpy(dtype=float)
        jitter = rng.uniform(-0.11, 0.11, len(values))
        ax_c.scatter(idx + jitter, values, s=9, color=TEAL, alpha=0.58,
                     edgecolor=WHITE, linewidth=0.2, zorder=2)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax_c.plot([idx, idx], [q1, q3], color=INK, lw=2.2, solid_capstyle="round", zorder=3)
        ax_c.plot([idx - 0.12, idx + 0.12], [med, med], color=GOLD, lw=1.4, zorder=4)
    ax_c.axhline(0, color=INK, ls=(0, (3, 2)), lw=0.7)
    ax_c.set_xticks(range(6), labels)
    ax_c.set_ylabel("Post-marker minus initial (pooled SD units)")
    ax_c.set_xlim(-0.5, 5.5)
    clean_axis(ax_c, "y")
    ax_c.text(0.99, 0.03, "dots: 20 stimulus nouns; bars: IQR; gold: median",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=5.5, color=GRAY)

    panel_label(ax_d, "d")
    ax_d.set_title("Active/passive is non-identifiable after marker identity", loc="left",
                   fontweight="bold", color=INK, pad=4)
    for _, row in variance.iterrows():
        vals = [row["median_syntax_candidate_no_word"], row["median_syntax_after_word_identity"]]
        ax_d.plot([0, 1], vals, color="#B6C0BC", lw=0.7, zorder=1)
        ax_d.scatter([0, 1], vals, s=22, color=BLUE, edgecolor=WHITE, linewidth=0.4, zorder=3)
    before = variance["median_syntax_candidate_no_word"].to_numpy(dtype=float)
    after = variance["median_syntax_after_word_identity"].to_numpy(dtype=float)
    draw_patient_median(ax_d, 0, before)
    draw_patient_median(ax_d, 1, after)
    ax_d.axhline(0, color=INK, ls=(0, (3, 2)), lw=0.7)
    ax_d.set_xticks([0, 1], ["active/passive candidate\nwithout marker identity", "active/passive candidate\nafter marker identity"])
    ax_d.set_ylabel("Run-median unique cross-validated R²")
    ax_d.set_xlim(-0.35, 1.35)
    ax_d.set_ylim(-0.45, 0.075)
    clean_axis(ax_d, "y")
    ax_d.text(0.98, 0.03, "7 runs nested in 5 patients\nmedians: -0.179 and -0.030",
              transform=ax_d.transAxes, ha="right", va="bottom", fontsize=5.7, color=GRAY)

    finish_figure(
        fig,
        "S14_stimulus_acoustic_identifiability",
        {
            "title": "Stimulus acoustics and design identifiability",
            "sources": [str(ACOUSTIC), str(VARIANCE)],
            "stimulus_units": "20 sentences and 20 matched nouns in panels a-c",
            "neural_summary_units": "seven runs nested within five patients in panel d",
            "identifiability_boundary": (
                "Voice is perfectly coupled to marker family; noun position is coupled to serial time "
                "and preceding-marker presence. The design does not isolate abstract voice or hierarchy."
            ),
            "key_values": {
                "marker_voice_cramers_v": 1.0,
                "n_unique_sentences": int(sentences.shape[0]),
                "n_matched_nouns": int(acoustic["noun"].nunique()),
                "median_condition_candidate_no_word_cv_r2": float(np.median(before)),
                "median_condition_candidate_after_word_identity_cv_r2": float(np.median(after)),
            },
        },
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_recording_yield()
    build_decoder_sensitivity()
    build_population_diagnostics()
    build_stimulus_identifiability()
    print(FIG_DIR)


if __name__ == "__main__":
    main()
