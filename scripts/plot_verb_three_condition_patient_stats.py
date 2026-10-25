#!/usr/bin/env python3
"""Plot patient-level stats for active/passive/scrambled verb responses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analyze_verb_three_condition_unit_activity import (  # noqa: E402
    CONDITION_COLORS,
    POINTWISE_ALPHA,
    PRIMARY_SIGNIFICANCE_COL,
)


CONDITIONS = ("active", "passive", "scrambled")


from verb_three_condition_versioning import input_metadata, parse_downstream_args, resolve_input

VERB_VERSION, OUT_DIR = resolve_input(validate=False)

def patient_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient, group in results.groupby("patient", sort=True):
        screened = group[group[PRIMARY_SIGNIFICANCE_COL].astype(bool)]
        counts = screened["timepoint_preferred_condition"].value_counts()
        rows.append(
            {
                "patient": patient,
                "n_runs": int(group["run"].nunique()),
                "n_units": int(len(group)),
                "time_gt5ms_p_condition": int(len(screened)),
                "aggregate_friedman_p05": int(group["aggregate_friedman_p05"].sum()),
                "active_pref_time": int(counts.get("active", 0)),
                "passive_pref_time": int(counts.get("passive", 0)),
                "scrambled_pref_time": int(counts.get("scrambled", 0)),
                "best_min_pointwise_p": float(group["min_pointwise_p"].min()),
                "max_longest_segment_ms": int(group["longest_segment_ms"].max()),
                "median_effect_hz": float(group["max_minus_min_bc_hz"].median()),
                "p90_effect_hz": float(group["max_minus_min_bc_hz"].quantile(0.9)),
                "max_effect_hz": float(group["max_minus_min_bc_hz"].max()),
            }
        )
    return pd.DataFrame(rows)


def annotate_bars(ax: plt.Axes, values: np.ndarray, x: np.ndarray, y_offset: float = 3.0) -> None:
    for xi, value in zip(x, values):
        ax.text(xi, value + y_offset, str(int(value)), ha="center", va="bottom", fontsize=8, color="0.25")


def plot_patient_stats(results: pd.DataFrame, summary: pd.DataFrame) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_verb_three_condition_patient_stats.png"
    pdf_path = fig_dir / "fig_verb_three_condition_patient_stats.pdf"

    patients = summary["patient"].tolist()
    x = np.arange(len(patients))

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
    ax_units = fig.add_subplot(gs[0, 0])
    ax_counts = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 0])
    ax_effect = fig.add_subplot(gs[1, 1])

    n_units = summary["n_units"].to_numpy(int)
    bars = ax_units.bar(x, n_units, color="#6A9C89", width=0.65)
    annotate_bars(ax_units, n_units, x, y_offset=max(n_units) * 0.015)
    ax_units.set_xticks(x)
    ax_units.set_xticklabels(patients)
    ax_units.set_ylabel("units analyzed")
    ax_units.set_title("Units per patient", loc="left", fontsize=12, weight="bold")
    ax_units.grid(axis="y", color="0.9", lw=0.7)
    ax_units.bar_label(
        bars,
        labels=[f"{int(run)} run{'s' if int(run) > 1 else ''}" for run in summary["n_runs"]],
        label_type="center",
        color="white",
        fontsize=8,
    )

    bottom = np.zeros(len(patients))
    count_cols = {
        "active": "active_pref_time",
        "passive": "passive_pref_time",
        "scrambled": "scrambled_pref_time",
    }
    for condition in CONDITIONS:
        vals = summary[count_cols[condition]].to_numpy(int)
        ax_counts.bar(
            x,
            vals,
            bottom=bottom,
            color=CONDITION_COLORS[condition],
            width=0.65,
            label=condition,
        )
        for xi, btm, val in zip(x, bottom, vals):
            if val > 0:
                ax_counts.text(xi, btm + val / 2, str(int(val)), ha="center", va="center", fontsize=8, color="white")
        bottom += vals
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(patients)
    ax_counts.set_ylabel("time-screen units")
    ax_counts.set_title(f"p<{POINTWISE_ALPHA:.4f} for >5 ms by preference", loc="left", fontsize=12, weight="bold")
    ax_counts.legend(frameon=False, loc="upper right", ncol=3, fontsize=8)
    ax_counts.grid(axis="y", color="0.9", lw=0.7)

    best_p = np.clip(summary["best_min_pointwise_p"].to_numpy(float), 1e-12, 1.0)
    log_best_p = -np.log10(best_p)
    ax_p.plot(x, log_best_p, marker="o", lw=1.8, color="#4C78A8", label="best min pointwise p")
    ax_p.axhline(-np.log10(POINTWISE_ALPHA), color="0.45", linestyle="--", lw=0.9, label=f"0.05/{len(CONDITIONS)}")
    y_top = float(log_best_p.max() + 0.55)
    ax_p.set_ylim(bottom=max(0.0, float(log_best_p.min() - 0.45)), top=y_top)
    for xi, y_value, longest in zip(x, log_best_p, summary["max_longest_segment_ms"].to_numpy(int)):
        near_top = y_value > y_top - 0.35
        ax_p.text(
            xi,
            y_value - 0.18 if near_top else y_value + 0.16,
            f"{longest} ms",
            ha="center",
            va="top" if near_top else "bottom",
            fontsize=8,
            color="0.25",
        )
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(patients)
    ax_p.set_ylabel("-log10 value")
    ax_p.set_title("Best timepoint p and longest segment", loc="left", fontsize=12, weight="bold")
    ax_p.legend(frameon=False, loc="upper right", fontsize=8)
    ax_p.grid(color="0.9", lw=0.7)

    distributions = [
        results.loc[results["patient"] == patient, "max_minus_min_bc_hz"].to_numpy(float)
        for patient in patients
    ]
    bp = ax_effect.boxplot(distributions, tick_labels=patients, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#DCE8F2")
        patch.set_edgecolor("0.35")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    for i, patient in enumerate(patients, start=1):
        cand = results[
            (results["patient"] == patient)
            & results[PRIMARY_SIGNIFICANCE_COL].astype(bool)
        ].copy()
        if cand.empty:
            continue
        jitter = np.linspace(-0.13, 0.13, len(cand)) if len(cand) > 1 else np.array([0.0])
        colors = [CONDITION_COLORS.get(str(c), "#BDBDBD") for c in cand["timepoint_preferred_condition"]]
        ax_effect.scatter(
            np.full(len(cand), i) + jitter,
            cand["max_minus_min_bc_hz"],
            c=colors,
            s=26,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    ax_effect.set_ylabel("max - min condition response (Hz)")
    ax_effect.set_title("Condition-effect distributions", loc="left", fontsize=12, weight="bold")
    ax_effect.grid(axis="y", color="0.9", lw=0.7)

    fig.suptitle("Patient-level time screen: active/passive/scrambled verb responses", y=0.995, fontsize=15, weight="bold")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> None:
    global VERB_VERSION, OUT_DIR
    args = parse_downstream_args(__doc__, unit_set=False)
    VERB_VERSION, OUT_DIR = args.verb_version, args.verb_input_dir
    results_path = OUT_DIR / "verb_three_condition_unit_results.csv"
    results = pd.read_csv(results_path)
    summary = patient_summary(results)
    summary_path = OUT_DIR / "verb_three_condition_summary_by_patient.csv"
    summary.to_csv(summary_path, index=False)
    paths = plot_patient_stats(results, summary)
    payload = {
        **input_metadata(VERB_VERSION, OUT_DIR),
        "summary_csv": str(summary_path),
        **paths,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
