#!/usr/bin/env python3
"""Plot patient-level noun-role structure encoding summary."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
cache_dir = ROOT / ".cache"
(cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scripts.analyze_noun_role_structure_encoding as analysis


def label_positive_bars(ax: plt.Axes, bars, dy: float = 0.4) -> None:
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    analysis.setup_matplotlib()
    summary_path = analysis.OUT_DIR / "noun_role_structure_encoding_summary_by_patient.csv"
    summary = pd.read_csv(summary_path)
    fig_dir = analysis.OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_noun_role_patient_stats_summary.png"
    pdf_path = fig_dir / "fig_noun_role_patient_stats_summary.pdf"

    patients = summary["patient"].tolist()
    x = np.arange(len(patients))
    gz002_i = patients.index("gz002") if "gz002" in patients else None

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    ax_direction, ax_metric, ax_fdr, ax_effect = axes.ravel()

    def shade_gz002(ax: plt.Axes) -> None:
        if gz002_i is not None:
            ax.axvspan(gz002_i - 0.5, gz002_i + 0.5, color="#F4E6D2", alpha=0.55, zorder=0)

    y = np.arange(len(patients))
    subject = summary["bc_uncorrected_subject_pref"].to_numpy()
    predicate = summary["bc_uncorrected_predicate_pref"].to_numpy()
    ax_direction.barh(y, subject, color=analysis.ACTIVE, label="subject-pref")
    ax_direction.barh(y, -predicate, color=analysis.PREDICATE, label="predicate-pref")
    for yi, subj, pred in zip(y, subject, predicate):
        if subj:
            ax_direction.text(subj + 0.7, yi, str(int(subj)), va="center", fontsize=8)
        if pred:
            ax_direction.text(-pred - 0.7, yi, str(int(pred)), va="center", ha="right", fontsize=8)
    ax_direction.axvline(0, color="0.35", lw=0.9)
    ax_direction.set_yticks(y)
    ax_direction.set_yticklabels(patients)
    ax_direction.set_xlabel("baseline-corrected uncorrected p<0.05 units")
    ax_direction.set_title("A. Candidate direction by patient", loc="left", weight="bold")
    ax_direction.legend(frameon=False, loc="lower right")
    ax_direction.grid(axis="x", color="0.9")

    shade_gz002(ax_metric)
    width = 0.22
    bc_bars = ax_metric.bar(
        x - width,
        summary["bc_uncorrected_p05"],
        width=width,
        color="#4C78A8",
        label="baseline-corrected",
    )
    raw_bars = ax_metric.bar(
        x,
        summary["raw_response_p05"],
        width=width,
        color="#72B7B2",
        label="raw response",
    )
    base_bars = ax_metric.bar(
        x + width,
        summary["baseline_p05"],
        width=width,
        color="#F58518",
        label="local baseline",
    )
    ax_metric.scatter(
        x - width,
        summary["bc_expected_5pct"],
        marker="x",
        color="black",
        s=34,
        label="5% null expectation",
        zorder=4,
    )
    label_positive_bars(ax_metric, bc_bars)
    label_positive_bars(ax_metric, raw_bars)
    label_positive_bars(ax_metric, base_bars)
    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels(patients)
    ax_metric.set_ylabel("p<0.05 units")
    ax_metric.set_title("B. Metric choice / baseline check", loc="left", weight="bold")
    ax_metric.legend(frameon=False, fontsize=7)
    ax_metric.grid(axis="y", color="0.9")

    shade_gz002(ax_fdr)
    width = 0.23
    patient_bars = ax_fdr.bar(
        x - width,
        summary["bc_patient_fdr05"],
        width=width,
        color="#59A14F",
        label="patient-FDR",
    )
    run_bars = ax_fdr.bar(
        x,
        summary["bc_run_fdr05"],
        width=width,
        color="#B279A2",
        label="run-FDR",
    )
    global_bars = ax_fdr.bar(
        x + width,
        summary["bc_global_fdr05"],
        width=width,
        color="#9C755F",
        label="global-FDR",
    )
    label_positive_bars(ax_fdr, patient_bars, dy=0.25)
    label_positive_bars(ax_fdr, run_bars, dy=0.25)
    label_positive_bars(ax_fdr, global_bars, dy=0.25)
    ax_fdr.set_xticks(x)
    ax_fdr.set_xticklabels(patients)
    ax_fdr.set_ylabel("FDR<0.05 units")
    ax_fdr.set_title("C. FDR-surviving units", loc="left", weight="bold")
    ax_fdr.legend(frameon=False)
    ax_fdr.grid(axis="y", color="0.9")

    shade_gz002(ax_effect)
    effects = summary["bc_median_effect_hz"].to_numpy()
    effect_colors = np.where(effects >= 0, analysis.ACTIVE, analysis.PREDICATE)
    ax_effect.bar(x, effects, color=effect_colors, alpha=0.85, label="median effect")
    ax_effect.axhline(0, color="0.35", lw=0.9)
    ax_effect.set_xticks(x)
    ax_effect.set_xticklabels(patients)
    ax_effect.set_ylabel("median subject - predicate effect (Hz)")
    ax_effect.set_title("D. Effect direction and prevalence", loc="left", weight="bold")
    ax_effect.grid(axis="y", color="0.9")
    ax_effect2 = ax_effect.twinx()
    ax_effect2.plot(
        x,
        summary["bc_positive_frac"] * 100,
        color="#333333",
        marker="o",
        lw=1.6,
        label="positive-effect units",
    )
    ax_effect2.set_ylabel("units with subject > predicate (%)")
    ax_effect2.set_ylim(0, 100)
    lines, labels = ax_effect.get_legend_handles_labels()
    lines2, labels2 = ax_effect2.get_legend_handles_labels()
    ax_effect.legend(lines + lines2, labels + labels2, frameon=False, loc="lower right")

    fig.suptitle("Patient-level noun structural-role statistics", fontsize=15, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Shaded column marks gz002: many baseline-corrected candidates, but raw-response counts are near the 5% null expectation while local-baseline differences are high.",
        ha="center",
        fontsize=8.5,
        color="0.25",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
