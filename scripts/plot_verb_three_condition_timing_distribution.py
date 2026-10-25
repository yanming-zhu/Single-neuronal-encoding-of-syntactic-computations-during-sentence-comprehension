#!/usr/bin/env python3
"""Plot timing distributions for active/passive/scrambled-preferring verb units."""

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

from analyze_verb_active_passive_unit_activity import setup_matplotlib  # noqa: E402
from analyze_verb_three_condition_unit_activity import (  # noqa: E402
    CONDITIONS,
    CONDITION_COLORS,
)


METRICS = (
    ("first_significant_onset_ms", "差异开始", "first significant onset"),
    ("preferred_peak_ms", "条件峰值", "condition peak"),
    ("effect_peak_ms", "差异峰值", "effect peak"),
)

CONDITION_LABELS_ZH = {
    "active": "主动句最高",
    "passive": "被动句最高",
    "scrambled": "无意义句最高",
}


from verb_three_condition_versioning import input_metadata, parse_downstream_args, resolve_input

VERB_VERSION, OUT_DIR = resolve_input(validate=False)

def fmt_ms(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return str(int(np.floor(float(value) + 0.5)))


def finite_quantile(values: pd.Series, q: float) -> float:
    vals = values.dropna().astype(float)
    return float(vals.quantile(q)) if len(vals) else float("nan")


def build_stats(timing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, zh_label, en_label in METRICS:
        for condition in CONDITIONS:
            values = timing.loc[timing["preferred_condition"] == condition, metric].dropna().astype(float)
            rows.append(
                {
                    "metric": metric,
                    "metric_zh": zh_label,
                    "metric_en": en_label,
                    "preferred_condition": condition,
                    "preferred_condition_zh": CONDITION_LABELS_ZH[condition],
                    "n_units": int(len(values)),
                    "median_ms": float(values.median()) if len(values) else float("nan"),
                    "q25_ms": finite_quantile(values, 0.25),
                    "q75_ms": finite_quantile(values, 0.75),
                    "min_ms": float(values.min()) if len(values) else float("nan"),
                    "max_ms": float(values.max()) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_timing_distribution(timing: pd.DataFrame, stats_table: pd.DataFrame) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_verb_three_condition_timing_distribution.png"
    pdf_path = fig_dir / "fig_verb_three_condition_timing_distribution.pdf"
    font_prop = setup_matplotlib()

    fig, axes = plt.subplots(3, 1, figsize=(14.2, 8.0), sharex=True)
    rng = np.random.default_rng(13)
    y_positions = np.arange(len(CONDITIONS))[::-1]
    x_ticks = [0, 200, 400, 600, 800]

    for ax, (metric, zh_label, en_label) in zip(axes, METRICS):
        values_by_condition = [
            timing.loc[timing["preferred_condition"] == condition, metric].dropna().astype(float).to_numpy()
            for condition in CONDITIONS
        ]
        bp = ax.boxplot(
            values_by_condition,
            vert=False,
            positions=y_positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "0.45", "linewidth": 1.0},
            capprops={"color": "0.45", "linewidth": 1.0},
        )
        for patch, condition in zip(bp["boxes"], CONDITIONS):
            patch.set_facecolor(CONDITION_COLORS[condition])
            patch.set_alpha(0.15)
            patch.set_edgecolor(CONDITION_COLORS[condition])
            patch.set_linewidth(1.2)

        for y, condition, values in zip(y_positions, CONDITIONS, values_by_condition):
            jitter = rng.uniform(-0.16, 0.16, size=len(values))
            ax.scatter(
                values,
                y + jitter,
                s=28,
                color=CONDITION_COLORS[condition],
                alpha=0.72,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
            stat_row = stats_table[
                (stats_table["metric"] == metric) & (stats_table["preferred_condition"] == condition)
            ].iloc[0]
            median = float(stat_row["median_ms"])
            q25 = float(stat_row["q25_ms"])
            q75 = float(stat_row["q75_ms"])
            ax.plot([median], [y], marker="D", markersize=5.8, color=CONDITION_COLORS[condition], zorder=4)
            ax.text(
                827,
                y,
                f"{fmt_ms(median)} ms\n[{fmt_ms(q25)}, {fmt_ms(q75)}]",
                va="center",
                ha="left",
                fontsize=8,
                color=CONDITION_COLORS[condition],
                fontproperties=font_prop,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [
                f"{CONDITION_LABELS_ZH[c]} (n={int(stats_table[(stats_table['metric'] == metric) & (stats_table['preferred_condition'] == c)].iloc[0]['n_units'])})"
                for c in CONDITIONS
            ],
            fontproperties=font_prop,
        )
        ax.set_xlim(-25, 930)
        ax.set_xticks(x_ticks)
        ax.axvline(0, color="0.35", lw=0.9)
        ax.axvline(800, color="0.75", lw=0.9, linestyle="--")
        ax.grid(axis="x", color="0.88", lw=0.8)
        ax.set_title(f"{zh_label} ({en_label})", loc="left", fontsize=12, weight="bold", fontproperties=font_prop)
        ax.text(
            827,
            2.63,
            "median\n[IQR]",
            ha="left",
            va="top",
            fontsize=7.5,
            color="0.35",
            fontproperties=font_prop,
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[-1].set_xlabel("ms after verb onset", fontsize=10)
    fig.suptitle(
        "时间分布：每个点 = 1 个显著 time-screen unit",
        x=0.05,
        y=0.985,
        ha="left",
        fontsize=16,
        weight="bold",
        fontproperties=font_prop,
    )
    fig.text(
        0.05,
        0.945,
        "分组按该 unit 的最高条件；筛选标准：p < 0.05/3，连续显著时间 >5 ms，不做 FDR。",
        ha="left",
        va="top",
        fontsize=10,
        color="0.35",
        fontproperties=font_prop,
    )
    fig.text(
        0.94,
        0.05,
        "灰虚线：800 ms response window end",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.45",
    )
    fig.subplots_adjust(left=0.14, right=0.88, top=0.88, bottom=0.1, hspace=0.42)
    fig.savefig(png_path, dpi=190)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> None:
    global VERB_VERSION, OUT_DIR
    args = parse_downstream_args(__doc__, unit_set=False)
    VERB_VERSION, OUT_DIR = args.verb_version, args.verb_input_dir
    timing = pd.read_csv(OUT_DIR / "verb_three_condition_population_timing_by_unit.csv")
    timing = timing[timing["preferred_condition"].isin(CONDITIONS)].copy()
    stats_table = build_stats(timing)
    stats_path = OUT_DIR / "verb_three_condition_timing_distribution_stats.csv"
    stats_table.to_csv(stats_path, index=False)
    paths = plot_timing_distribution(timing, stats_table)
    payload = {
        **input_metadata(VERB_VERSION, OUT_DIR),
        "n_units": int(len(timing)),
        "stats_csv": str(stats_path),
        "outputs": paths,
    }
    (OUT_DIR / "verb_three_condition_timing_distribution_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
