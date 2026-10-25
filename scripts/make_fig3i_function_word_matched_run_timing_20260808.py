#!/usr/bin/env python3
"""Matched-by-run timing comparison for active- and passive-preferring units.

Each recording run contributes one pair: the median latency across its
active-preferring units and the median latency across its passive-preferring
units.  The directional test is an exact paired Wilcoxon signed-rank test of
active minus passive < 0.  Runs, rather than individual units, are the paired
observations; runs remain nested within patients, so this is exploratory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from make_fig3i_function_word_ba_bei_timing_20260806 import load_timing_data


OUT_ROOT = (
    ROOT
    / "reports"
    / "nature_science_draft"
    / "assembled_figures_20260805"
)
OUT = OUT_ROOT / "subplots"
SOURCE = OUT_ROOT / "source_data" / "fig3i_function_word_matched_run_timing_60x40mm"
STEM = OUT / "fig3i_function_word_matched_run_timing_60x40mm"

WIDTH_MM = 60.0
HEIGHT_MM = 40.0
PT = 8.0
ACTIVE = "#357ABD"
PASSIVE = "#D95F4C"
GRAY = "#A9AFB5"
INK = "#202426"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": PT,
            "axes.labelsize": PT,
            "axes.titlesize": PT,
            "xtick.labelsize": PT,
            "ytick.labelsize": PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_pairs(table: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        table.groupby(["subject", "run", "metric", "direction"], sort=False)[
            "value_ms"
        ]
        .median()
        .unstack("direction")
        .reset_index()
        .rename(
            columns={
                "active_gt_passive": "active_preferring_median_ms",
                "passive_gt_active": "passive_preferring_median_ms",
            }
        )
    )
    pairs = pairs.dropna(
        subset=["active_preferring_median_ms", "passive_preferring_median_ms"]
    ).copy()
    pairs["active_minus_passive_ms"] = (
        pairs["active_preferring_median_ms"]
        - pairs["passive_preferring_median_ms"]
    )
    expected_runs = 7
    counts = pairs.groupby("metric").size().to_dict()
    if counts != {"onset": expected_runs, "peak": expected_runs}:
        raise ValueError(f"Expected seven matched runs per metric, found {counts}")
    return pairs


def matched_tests(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("onset", "peak"):
        part = pairs.loc[pairs["metric"].eq(metric)]
        active = part["active_preferring_median_ms"].to_numpy(float)
        passive = part["passive_preferring_median_ms"].to_numpy(float)
        difference = active - passive
        wilcoxon = stats.wilcoxon(
            active,
            passive,
            alternative="less",
            zero_method="wilcox",
            method="exact",
        )
        paired_t = stats.ttest_rel(active, passive, alternative="less")
        rows.append(
            {
                "metric": metric,
                "matching_unit": "recording run",
                "within_run_summary": "median across screen-positive units",
                "n_matched_runs": int(len(part)),
                "n_runs_active_earlier": int(np.sum(difference < 0)),
                "run_level_active_median_ms": float(np.median(active)),
                "run_level_passive_median_ms": float(np.median(passive)),
                "median_paired_difference_ms": float(np.median(difference)),
                "mean_paired_difference_ms": float(np.mean(difference)),
                "test": "exact one-sided paired Wilcoxon signed-rank",
                "alternative": "active-preferring run median < passive-preferring run median",
                "wilcoxon_w": float(wilcoxon.statistic),
                "wilcoxon_one_sided_p_value": float(wilcoxon.pvalue),
                "paired_t": float(paired_t.statistic),
                "paired_t_df": int(len(part) - 1),
                "paired_t_one_sided_p_value": float(paired_t.pvalue),
                "inference_note": "exploratory; seven runs are nested within five patients",
            }
        )
    return pd.DataFrame(rows)


def draw_panel(pairs: pd.DataFrame, tests: pd.DataFrame) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        sharey=True,
        facecolor="white",
    )
    for ax, metric in zip(axes, ("onset", "peak")):
        part = pairs.loc[pairs["metric"].eq(metric)].copy()
        active = part["active_preferring_median_ms"].to_numpy(float)
        passive = part["passive_preferring_median_ms"].to_numpy(float)
        for active_value, passive_value in zip(active, passive):
            ax.plot(
                [0, 1],
                [active_value, passive_value],
                color=GRAY,
                lw=0.7,
                alpha=0.85,
                zorder=1,
            )
        ax.scatter(
            np.zeros_like(active),
            active,
            s=16,
            color=ACTIVE,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            np.ones_like(passive),
            passive,
            s=16,
            color=PASSIVE,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        for x, values in ((0, active), (1, passive)):
            median = float(np.median(values))
            ax.plot([x - 0.16, x + 0.16], [median, median], color=INK, lw=1.2, zorder=4)
        p_value = float(
            tests.loc[
                tests["metric"].eq(metric), "wilcoxon_one_sided_p_value"
            ].iloc[0]
        )
        ax.set_title(
            f"{metric.capitalize()}\nP = {p_value:.3f}",
            fontweight="bold",
            pad=3,
        )
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1], ["A-pref.", "P-pref."])
        ax.set_ylim(0, 820)
        ax.set_yticks([0, 200, 400, 600, 800])
        ax.grid(axis="y", color="#EEF0F2", lw=0.55)
    axes[0].set_ylabel("Run median latency (ms)")
    axes[0].text(
        -0.58,
        1.16,
        "i",
        transform=axes[0].transAxes,
        fontsize=PT,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.22, top=0.73, wspace=0.38)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    table = load_timing_data()
    pairs = make_pairs(table)
    tests = matched_tests(pairs)

    pairs.to_csv(SOURCE / "fig3i_function_word_timing_matched_by_run.csv", index=False)
    tests.to_csv(SOURCE / "fig3i_function_word_timing_matched_tests.csv", index=False)

    fig = draw_panel(pairs, tests)
    fig.savefig(f"{STEM}.pdf", facecolor="white")
    fig.savefig(f"{STEM}.svg", facecolor="white")
    fig.savefig(f"{STEM}.png", dpi=600, facecolor="white")
    fig.savefig(
        f"{STEM}.tiff",
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    onset = tests.loc[tests["metric"].eq("onset")].iloc[0]
    peak = tests.loc[tests["metric"].eq("peak")].iloc[0]
    metadata = {
        "figure_contract": {
            "core_conclusion": "Five of seven runs show earlier active-preferring run-median latencies, but the matched run-level differences are not significant.",
            "archetype": "paired quantitative comparison",
            "backend": "Python/matplotlib",
            "final_size_mm": [WIDTH_MM, HEIGHT_MM],
            "font_size_pt": PT,
        },
        "matching": {
            "unit": "recording run",
            "within_run_endpoint": "median latency among screen-positive units in each preference direction",
            "n_matched_runs": 7,
            "n_patients": 5,
        },
        "matched_tests": tests.to_dict(orient="records"),
        "reviewer_risks": [
            "Active- and passive-preferring units are different neurons and cannot be paired one-to-one without an arbitrary matching rule.",
            "Runs are nested within five patients, so the run-paired analysis is exploratory rather than patient-level population inference.",
            "BA/BEI marker family is fully collinear with active/passive construction in the current stimuli.",
            "Latency is conditional on selection as a screen-positive unit.",
        ],
    }
    (SOURCE / "fig3i_function_word_timing_matched_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    legend = (
        "**Fig. 3i | Run-matched construction-preference timing.** "
        "Within each recording run, points show the median onset or direction-consistent effect-peak latency across active-preferring and passive-preferring screen-positive unit entries; lines connect values from the same run and black bars denote medians across runs. "
        "P values are from exact one-sided paired Wilcoxon signed-rank tests of the directional hypothesis that active-preferring run medians are earlier "
        f"(onset W = {onset['wilcoxon_w']:.0f}, P = {onset['wilcoxon_one_sided_p_value']:.3f}; "
        f"peak W = {peak['wilcoxon_w']:.0f}, P = {peak['wilcoxon_one_sided_p_value']:.3f}; n = 7 matched runs). "
        "Five of seven runs showed negative active-minus-passive latency differences for both endpoints. Runs are nested within five patients, and BA/BEI marker family is collinear with active/passive construction; this matched-run analysis is therefore exploratory. Source data are provided with the figure.\n"
    )
    (SOURCE / "fig3i_function_word_timing_matched_legend.md").write_text(
        legend, encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
