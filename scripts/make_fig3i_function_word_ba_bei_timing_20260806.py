#!/usr/bin/env python3
"""Create a Fig. 3c/f-style timing panel aligned to BA/BEI-family markers.

The analysis reuses the audited corrected function-word-onset screen.  It does
not rescreen units or change the locked threshold.  For every directional
screen-positive unit/run entry, the panel shows:

* onset: start of the first qualifying no-gap segment;
* peak: the largest direction-consistent active-minus-passive effect within
  qualifying segments.

Active trials contain the BA-family markers BA/GEI/JIANG; passive trials
contain the BEI-family markers BEI/RANG/JIAO.  Unit/run entries remain nested
within five patients, so the Mann-Whitney comparisons are descriptive.
"""

from __future__ import annotations

import json
import os
from itertools import product
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


INPUT = ROOT / "outputs" / "active_passive_corrected_timepoint_screen_20260722"
OUT = (
    ROOT
    / "reports"
    / "nature_science_draft"
    / "assembled_figures_20260805"
    / "subplots"
)
SOURCE = (
    ROOT
    / "reports"
    / "nature_science_draft"
    / "assembled_figures_20260805"
    / "source_data"
    / "fig3i_function_word_ba_bei_timing_ttest_anova_60x40mm"
)
STEM = OUT / "fig3i_function_word_ba_bei_timing_ttest_anova_60x40mm"

WIDTH_MM = 60.0
HEIGHT_MM = 40.0
PT = 8.0
ACTIVE = "#357ABD"
PASSIVE = "#D95F4C"
INK = "#202426"
GRAY = "#70757A"
LIGHT_GRAY = "#D8DDE3"
PALE_GRAY = "#F0F2F3"
PATIENTS = ("gz001", "gz002", "gz003", "gz004", "gz005")
DIRECTIONS = ("active_gt_passive", "passive_gt_active")


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


def load_timing_data() -> pd.DataFrame:
    units = pd.read_csv(INPUT / "corrected_unit_results.csv")
    points = pd.read_csv(INPUT / "pointwise_passing_bins.csv")

    required_units = {
        "subject",
        "run",
        "unit_id",
        "ks_label",
        "has_qualifying_segment",
        "qualifying_direction",
        "first_segment_onset_ms",
    }
    required_points = {
        "subject",
        "run",
        "unit_id",
        "alignment",
        "time_ms",
        "active_minus_passive_hz",
        "in_qualifying_segment",
    }
    if not required_units.issubset(units.columns):
        raise ValueError(f"Unit table lacks {sorted(required_units - set(units.columns))}")
    if not required_points.issubset(points.columns):
        raise ValueError(f"Point table lacks {sorted(required_points - set(points.columns))}")

    selected = units.loc[
        units["has_qualifying_segment"].astype(bool)
        & units["qualifying_direction"].isin(DIRECTIONS),
        [
            "subject",
            "run",
            "unit_id",
            "ks_label",
            "qualifying_direction",
            "first_segment_onset_ms",
        ],
    ].copy()
    if len(selected) != 172:
        raise ValueError(f"Expected 172 directional screen-positive entries, found {len(selected)}")
    counts = selected["qualifying_direction"].value_counts().to_dict()
    if counts != {"passive_gt_active": 92, "active_gt_passive": 80}:
        raise ValueError(f"Unexpected direction counts: {counts}")
    if tuple(sorted(selected["subject"].unique())) != PATIENTS:
        raise ValueError("Patient set does not match gz001-gz005")

    qualifying = points.loc[
        points["alignment"].eq("corrected")
        & points["in_qualifying_segment"].astype(bool)
    ].merge(
        selected[["subject", "run", "unit_id", "qualifying_direction"]],
        on=["subject", "run", "unit_id"],
        how="inner",
        validate="many_to_one",
    )
    qualifying["directional_effect_hz"] = np.where(
        qualifying["qualifying_direction"].eq("active_gt_passive"),
        qualifying["active_minus_passive_hz"],
        -qualifying["active_minus_passive_hz"],
    )
    peak_indices = qualifying.groupby(
        ["subject", "run", "unit_id"], sort=False
    )["directional_effect_hz"].idxmax()
    peaks = qualifying.loc[
        peak_indices,
        ["subject", "run", "unit_id", "time_ms", "directional_effect_hz"],
    ].rename(
        columns={
            "time_ms": "effect_peak_ms",
            "directional_effect_hz": "effect_peak_directional_hz",
        }
    )
    selected = selected.merge(
        peaks,
        on=["subject", "run", "unit_id"],
        how="left",
        validate="one_to_one",
    )
    if selected[["first_segment_onset_ms", "effect_peak_ms"]].isna().any().any():
        raise ValueError("Missing onset or peak for a selected entry")
    if not selected["first_segment_onset_ms"].between(0, 799).all():
        raise ValueError("Onset latency outside the registered 0-799 ms window")
    if not selected["effect_peak_ms"].between(0, 799).all():
        raise ValueError("Peak latency outside the registered 0-799 ms window")

    rows: list[pd.DataFrame] = []
    for metric, source in (
        ("onset", "first_segment_onset_ms"),
        ("peak", "effect_peak_ms"),
    ):
        part = selected[
            [
                "subject",
                "run",
                "unit_id",
                "ks_label",
                "qualifying_direction",
                source,
                "effect_peak_directional_hz",
            ]
        ].rename(
            columns={
                "qualifying_direction": "direction",
                source: "value_ms",
            }
        )
        part["metric"] = metric
        rows.append(part)
    return pd.concat(rows, ignore_index=True)[
        [
            "subject",
            "run",
            "unit_id",
            "ks_label",
            "direction",
            "metric",
            "value_ms",
            "effect_peak_directional_hz",
        ]
    ]


def timing_tests(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("onset", "peak"):
        ba = table.loc[
            table["direction"].eq("active_gt_passive")
            & table["metric"].eq(metric),
            "value_ms",
        ].to_numpy(float)
        bei = table.loc[
            table["direction"].eq("passive_gt_active")
            & table["metric"].eq(metric),
            "value_ms",
        ].to_numpy(float)
        welch = stats.ttest_ind(ba, bei, equal_var=False, alternative="less")
        anova = stats.f_oneway(ba, bei)
        levene = stats.levene(ba, bei, center="median")
        shapiro_ba = stats.shapiro(ba)
        shapiro_bei = stats.shapiro(bei)
        n_ba, n_bei = len(ba), len(bei)
        var_ba, var_bei = np.var(ba, ddof=1), np.var(bei, ddof=1)
        welch_se = np.sqrt(var_ba / n_ba + var_bei / n_bei)
        welch_df = (var_ba / n_ba + var_bei / n_bei) ** 2 / (
            (var_ba / n_ba) ** 2 / (n_ba - 1)
            + (var_bei / n_bei) ** 2 / (n_bei - 1)
        )
        mean_difference = float(np.mean(ba) - np.mean(bei))
        ci_half = float(stats.t.ppf(0.975, welch_df) * welch_se)
        pooled_sd = np.sqrt(
            ((n_ba - 1) * var_ba + (n_bei - 1) * var_bei)
            / (n_ba + n_bei - 2)
        )
        hedges_correction = 1.0 - 3.0 / (4.0 * (n_ba + n_bei) - 9.0)
        hedges_g = float(hedges_correction * mean_difference / pooled_sd)
        eta_squared = float(
            anova.statistic / (anova.statistic + n_ba + n_bei - 2)
        )
        rows.append(
            {
                "metric": metric,
                "test": "one-sided pooled Welch t-test and two-sided one-way ANOVA",
                "comparison": "BA-family-high versus BEI-family-high",
                "alternative": "BA-family-preferring latency < BEI-family-preferring latency",
                "n_ba_family_high": int(n_ba),
                "n_bei_family_high": int(n_bei),
                "mean_ba_family_high_ms": float(np.mean(ba)),
                "sd_ba_family_high_ms": float(np.std(ba, ddof=1)),
                "mean_bei_family_high_ms": float(np.mean(bei)),
                "sd_bei_family_high_ms": float(np.std(bei, ddof=1)),
                "mean_difference_ba_minus_bei_ms": mean_difference,
                "mean_difference_95ci_lower_ms": mean_difference - ci_half,
                "mean_difference_95ci_upper_ms": mean_difference + ci_half,
                "median_ba_family_high_ms": float(np.median(ba)),
                "median_bei_family_high_ms": float(np.median(bei)),
                "median_difference_ba_minus_bei_ms": float(np.median(ba) - np.median(bei)),
                "welch_t": float(welch.statistic),
                "welch_df": float(welch_df),
                "one_sided_p_value": float(welch.pvalue),
                "hedges_g": hedges_g,
                "anova_f": float(anova.statistic),
                "anova_df1": 1,
                "anova_df2": int(n_ba + n_bei - 2),
                "anova_p_value": float(anova.pvalue),
                "eta_squared": eta_squared,
                "brown_forsythe_p_value": float(levene.pvalue),
                "shapiro_ba_p_value": float(shapiro_ba.pvalue),
                "shapiro_bei_p_value": float(shapiro_bei.pvalue),
                "inference_note": "descriptive; unit/run entries are nested within five patients",
            }
        )
    return pd.DataFrame(rows)


def patient_summary(table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        table.groupby(["subject", "direction", "metric"], sort=False)["value_ms"]
        .agg(
            n="size",
            median_ms="median",
            q25_ms=lambda values: float(np.percentile(values, 25)),
            q75_ms=lambda values: float(np.percentile(values, 75)),
        )
        .reset_index()
    )
    return summary


def patient_exact_signflip_tests(table: pd.DataFrame) -> pd.DataFrame:
    """Exact one-sided tests of equal-patient mean latency differences.

    The prespecified directional alternative is BA-family-preferring latency
    earlier than BEI-family-preferring latency, i.e. BA minus BEI < 0.
    """
    rows: list[dict[str, object]] = []
    for metric in ("onset", "peak"):
        patient_medians = (
            table.loc[table["metric"].eq(metric)]
            .groupby(["subject", "direction"])["value_ms"]
            .median()
            .unstack()
            .reindex(PATIENTS)
        )
        differences = (
            patient_medians["active_gt_passive"]
            - patient_medians["passive_gt_active"]
        ).to_numpy(float)
        observed = float(np.mean(differences))
        null = np.asarray(
            [
                np.mean(differences * np.asarray(signs, dtype=float))
                for signs in product((-1.0, 1.0), repeat=len(differences))
            ],
            dtype=float,
        )
        rows.append(
            {
                "metric": metric,
                "test": "exact patient-level sign-flip",
                "alternative": "BA-family-preferring latency < BEI-family-preferring latency",
                "n_patients": int(len(differences)),
                "n_patients_ba_earlier": int(np.sum(differences < 0)),
                "patient_equal_weight_mean_difference_ms": observed,
                "patient_median_difference_ms": float(np.median(differences)),
                "one_sided_p_value": float(np.mean(null <= observed)),
                "two_sided_p_value": float(np.mean(np.abs(null) >= abs(observed))),
            }
        )
    return pd.DataFrame(rows)


def draw_panel(table: pd.DataFrame, pooled_tests: pd.DataFrame) -> plt.Figure:
    configure_style()
    fig, ax = plt.subplots(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        facecolor="white",
    )
    order = (
        ("active_gt_passive", "onset", "A-pref. onset", ACTIVE),
        ("active_gt_passive", "peak", "A-pref. peak", ACTIVE),
        ("passive_gt_active", "onset", "P-pref. onset", PASSIVE),
        ("passive_gt_active", "peak", "P-pref. peak", PASSIVE),
    )
    y_positions = np.asarray([0.0, 1.0, 2.45, 3.45])
    rng = np.random.default_rng(20260806)
    for y, (direction, metric, _label, color) in zip(y_positions, order):
        values = table.loc[
            table["direction"].eq(direction) & table["metric"].eq(metric),
            "value_ms",
        ].dropna().to_numpy(float)
        jitter = rng.uniform(-0.14, 0.14, len(values))
        ax.scatter(
            values,
            np.full(len(values), y) + jitter,
            s=6.5,
            color=color,
            alpha=0.27,
            linewidth=0,
            rasterized=True,
        )
        mean = float(np.mean(values))
        sem = float(stats.sem(values))
        ci_half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
        ax.plot(
            [mean - ci_half, mean + ci_half],
            [y, y],
            color=color,
            lw=4.7,
            alpha=0.30,
            solid_capstyle="butt",
        )
        ax.plot([mean, mean], [y - 0.22, y + 0.22], color=INK, lw=1.0)
        ax.text(810, y, f"{int(np.floor(mean + 0.5))} ms", va="center", ha="left", color=color)

    ax.axhline(1.72, color=LIGHT_GRAY, lw=0.7)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.axvline(800, color=GRAY, lw=0.6, ls=":")
    ax.set_xlim(-15, 1040)
    ax.set_yticks(y_positions, [item[2] for item in order])
    ax.invert_yaxis()
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlabel("Time from marker onset (ms)")
    ax.set_title("Mean latency", loc="left", fontweight="bold", pad=3)
    onset_p = float(
        pooled_tests.loc[
            pooled_tests["metric"].eq("onset"), "one_sided_p_value"
        ].iloc[0]
    )
    ax.text(
        1.0,
        1.03,
        f"t P = {onset_p:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=PT,
    )
    ax.grid(axis="x", color=PALE_GRAY, lw=0.55)
    ax.text(
        -0.27,
        1.08,
        "i",
        transform=ax.transAxes,
        fontsize=PT,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.29, top=0.78)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    table = load_timing_data()
    tests = timing_tests(table)
    medians = (
        table.groupby(["direction", "metric"], sort=False)["value_ms"]
        .agg(n="size", median_ms="median", q25_ms=lambda x: np.percentile(x, 25), q75_ms=lambda x: np.percentile(x, 75))
        .reset_index()
    )

    table.to_csv(SOURCE / "fig3i_function_word_timing_by_unit.csv", index=False)
    tests.to_csv(
        SOURCE / "fig3i_function_word_timing_pooled_ttest_anova.csv", index=False
    )
    medians.to_csv(SOURCE / "fig3i_function_word_timing_summary.csv", index=False)

    fig = draw_panel(table, tests)
    fig.savefig(f"{STEM}.pdf", facecolor="white")
    fig.savefig(f"{STEM}.svg", facecolor="white")
    fig.savefig(f"{STEM}.png", dpi=600, facecolor="white")
    fig.savefig(f"{STEM}.tiff", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    summary = {
        "figure_contract": {
            "core_conclusion": "After pooling all unit/run entries, active- and passive-preferring onset means are nearly identical, and neither Welch's t-test nor one-way ANOVA detects a mean-latency difference.",
            "archetype": "single quantitative timing panel",
            "backend": "Python/matplotlib",
            "final_size_mm": [WIDTH_MM, HEIGHT_MM],
            "font_size_pt": PT,
        },
        "input_screen": {
            "alignment": "corrected function-word onset",
            "active_marker_family": ["BA", "GEI", "JIANG"],
            "passive_marker_family": ["BEI", "RANG", "JIAO"],
            "n_registered_unit_run_entries": 1782,
            "n_patients": 5,
            "n_runs": 8,
            "sdf_sigma_ms": 25,
            "pointwise_alpha": 0.025,
            "minimum_consecutive_bins": 6,
            "response_window_ms": [0, 800],
        },
        "direction_counts": {
            "BA-family-high": 80,
            "BEI-family-high": 92,
        },
        "timing_summary": medians.to_dict(orient="records"),
        "pooled_ttest_anova": tests.to_dict(orient="records"),
        "reviewer_risks": [
            "BA/BEI family is fully collinear with active/passive construction in the present stimuli.",
            "Unit/run entries are nested within five patients; group tests are descriptive.",
            "The pointwise-duration screen is not a study-wide family-wise error correction.",
            "Timing is conditional on post-selection as a screen-positive unit.",
        ],
    }
    (SOURCE / "fig3i_function_word_timing_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    onset = tests.loc[tests["metric"].eq("onset")].iloc[0]
    peak = tests.loc[tests["metric"].eq("peak")].iloc[0]
    legend = (
        "**Fig. 3i | Function-word-aligned construction-preference timing.** "
        "First qualifying-segment onset and direction-consistent effect-peak latencies "
        "for active-preferring (BA/GEI/JIANG; n = 80 pooled unit/run entries) and "
        "passive-preferring (BEI/RANG/JIAO; n = 92 pooled unit/run entries) units, aligned to the corrected "
        "function-word onset. Points denote unit/run entries, coloured bars show 95% confidence intervals of the mean, and black ticks and annotations show pooled means. "
        "Directional P values are from one-sided pooled Welch t-tests testing whether active-preferring latencies are earlier "
        f"(onset t({onset['welch_df']:.1f}) = {onset['welch_t']:.2f}, P = {onset['one_sided_p_value']:.3f}; "
        f"peak t({peak['welch_df']:.1f}) = {peak['welch_t']:.2f}, P = {peak['one_sided_p_value']:.3f}). "
        "Two-sided one-way ANOVAs likewise found no group effect "
        f"(onset F(1, {int(onset['anova_df2'])}) = {onset['anova_f']:.3f}, P = {onset['anova_p_value']:.3f}; "
        f"peak F(1, {int(peak['anova_df2'])}) = {peak['anova_f']:.3f}, P = {peak['anova_p_value']:.3f}). "
        "Unit/run entries are nested within five patients and eight runs. Units passed the registered screen of P < 0.025 for at least six "
        "consecutive 1-ms bins after Gaussian smoothing (sigma = 25 ms). BA/BEI family is "
        "collinear with active/passive construction in these stimuli; the pooled tests are therefore descriptive rather than patient-level population inference. Source data are "
        "provided with the figure.\n"
    )
    (SOURCE / "fig3i_function_word_timing_legend.md").write_text(legend, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
