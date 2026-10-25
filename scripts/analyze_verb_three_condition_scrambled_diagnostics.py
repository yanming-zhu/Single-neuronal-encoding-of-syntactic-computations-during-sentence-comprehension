#!/usr/bin/env python3
"""Diagnostics for why scrambled-preferring verb units appear."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
    MATCHED_THREE_CONDITION_VERB_IDS,
    OUT_DIR,
    PRIMARY_SIGNIFICANCE_COL,
    VERB_LABELS,
)


def finite_median(values: pd.Series | np.ndarray) -> float:
    vals = pd.Series(values).dropna().astype(float)
    return float(vals.median()) if len(vals) else float("nan")


def finite_quantile(values: pd.Series | np.ndarray, q: float) -> float:
    vals = pd.Series(values).dropna().astype(float)
    return float(vals.quantile(q)) if len(vals) else float("nan")


def build_unit_diagnostics(sig: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for row in sig.itertuples(index=False):
        means = {
            "active": float(row.qualifying_active_mean_hz),
            "passive": float(row.qualifying_passive_mean_hz),
            "scrambled": float(row.qualifying_scrambled_mean_hz),
        }
        syntax_mean = 0.5 * (means["active"] + means["passive"])
        preferred = str(row.timepoint_preferred_condition)
        sorted_means = sorted(means.items(), key=lambda item: item[1], reverse=True)
        rows.append(
            {
                "patient": str(row.patient),
                "run": str(row.run),
                "unit_id": int(row.unit_id),
                "preferred_condition": preferred,
                "active_qualifying_hz": means["active"],
                "passive_qualifying_hz": means["passive"],
                "scrambled_qualifying_hz": means["scrambled"],
                "syntax_mean_qualifying_hz": syntax_mean,
                "preferred_minus_second_hz": float(sorted_means[0][1] - sorted_means[1][1]),
                "scrambled_minus_syntax_mean_hz": means["scrambled"] - syntax_mean,
                "scrambled_above_baseline": bool(means["scrambled"] > 0),
                "syntax_mean_below_baseline": bool(syntax_mean < 0),
                "active_below_baseline": bool(means["active"] < 0),
                "passive_below_baseline": bool(means["passive"] < 0),
                "relative_only_scrambled_high": bool(preferred == "scrambled" and means["scrambled"] <= 0),
                "min_pointwise_p": float(row.min_pointwise_p),
                "first_segment_onset_ms": float(row.first_segment_onset_ms),
                "longest_segment_ms": float(row.longest_segment_ms),
            }
        )
    unit_diag = pd.DataFrame(rows)
    scr = unit_diag[unit_diag["preferred_condition"] == "scrambled"]
    summary = {
        "n_significant_units": int(len(unit_diag)),
        "n_scrambled_high": int(len(scr)),
        "scrambled_high_scrambled_above_baseline": int(scr["scrambled_above_baseline"].sum()),
        "scrambled_high_relative_only": int(scr["relative_only_scrambled_high"].sum()),
        "scrambled_high_syntax_mean_below_baseline": int(scr["syntax_mean_below_baseline"].sum()),
        "scrambled_high_both_active_passive_below_baseline": int(
            (scr["active_below_baseline"] & scr["passive_below_baseline"]).sum()
        ),
        "scrambled_high_median_scrambled_hz": finite_median(scr["scrambled_qualifying_hz"]),
        "scrambled_high_median_active_hz": finite_median(scr["active_qualifying_hz"]),
        "scrambled_high_median_passive_hz": finite_median(scr["passive_qualifying_hz"]),
        "scrambled_high_median_scrambled_minus_syntax_hz": finite_median(scr["scrambled_minus_syntax_mean_hz"]),
    }
    return unit_diag, summary


def build_patient_run_tables(unit_diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    patient_rows = []
    for patient, group in unit_diag.groupby("patient"):
        counts = group["preferred_condition"].value_counts()
        n = len(group)
        scr = int(counts.get("scrambled", 0))
        patient_rows.append(
            {
                "patient": patient,
                "n_significant_units": int(n),
                "active_high": int(counts.get("active", 0)),
                "passive_high": int(counts.get("passive", 0)),
                "scrambled_high": scr,
                "scrambled_high_fraction": float(scr / n) if n else np.nan,
            }
        )
    by_patient = pd.DataFrame(patient_rows).sort_values("patient")

    run_rows = []
    for (patient, run), group in unit_diag.groupby(["patient", "run"]):
        counts = group["preferred_condition"].value_counts()
        n = len(group)
        scr = int(counts.get("scrambled", 0))
        run_rows.append(
            {
                "patient": patient,
                "run": run,
                "n_significant_units": int(n),
                "active_high": int(counts.get("active", 0)),
                "passive_high": int(counts.get("passive", 0)),
                "scrambled_high": scr,
                "scrambled_high_fraction": float(scr / n) if n else np.nan,
            }
        )
    by_run = pd.DataFrame(run_rows).sort_values(["patient", "run"])

    contingency = by_patient[["scrambled_high", "active_high", "passive_high"]].to_numpy(int)
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        patient_test = {
            "test": "chi-square contingency",
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p_value),
            "min_expected_count": float(np.min(expected)),
        }
    except ValueError:
        patient_test = {"test": "chi-square contingency", "p_value": float("nan"), "min_expected_count": float("nan")}
    return by_patient, by_run, patient_test


def build_verb_item_table(metrics: pd.DataFrame, unit_diag: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    matched = metrics[metrics["verb_id"].isin(MATCHED_THREE_CONDITION_VERB_IDS)].copy()
    unit_keys = unit_diag[["patient", "run", "unit_id", "preferred_condition"]].copy()
    pivot = (
        matched.groupby(["patient", "run", "unit_id", "verb_id", "verb", "sentence_condition"])[
            "response_baseline_corrected_hz"
        ]
        .mean()
        .unstack("sentence_condition")
        .reset_index()
    )
    pivot = pivot.merge(unit_keys, on=["patient", "run", "unit_id"], how="inner")
    pivot = pivot.dropna(subset=list(CONDITIONS))
    pivot["syntax_mean_hz"] = 0.5 * (pivot["active"] + pivot["passive"])
    pivot["scrambled_minus_syntax_mean_hz"] = pivot["scrambled"] - pivot["syntax_mean_hz"]
    scrambled_units = pivot[pivot["preferred_condition"] == "scrambled"].copy()
    verb_rows = []
    for (verb_id, verb), group in scrambled_units.groupby(["verb_id", "verb"]):
        contrast = group["scrambled_minus_syntax_mean_hz"]
        verb_rows.append(
            {
                "verb_id": int(verb_id),
                "verb": str(verb),
                "n_scrambled_high_units": int(group[["patient", "run", "unit_id"]].drop_duplicates().shape[0]),
                "median_scrambled_minus_syntax_hz": finite_median(contrast),
                "q25_scrambled_minus_syntax_hz": finite_quantile(contrast, 0.25),
                "q75_scrambled_minus_syntax_hz": finite_quantile(contrast, 0.75),
                "n_units_positive_scrambled_minus_syntax": int((contrast > 0).sum()),
                "median_scrambled_hz": finite_median(group["scrambled"]),
                "median_active_hz": finite_median(group["active"]),
                "median_passive_hz": finite_median(group["passive"]),
            }
        )
    by_verb = pd.DataFrame(verb_rows).sort_values("verb_id")
    item_summary = {
        "n_matched_verbs": int(len(by_verb)),
        "n_verbs_with_positive_median_scrambled_minus_syntax": int(
            (by_verb["median_scrambled_minus_syntax_hz"] > 0).sum()
        ),
        "max_median_verb_contrast_hz": float(by_verb["median_scrambled_minus_syntax_hz"].max()),
        "min_median_verb_contrast_hz": float(by_verb["median_scrambled_minus_syntax_hz"].min()),
    }
    return by_verb, item_summary


def build_timing_summary(timing: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for condition in CONDITIONS:
        group = timing[timing["preferred_condition"] == condition]
        rows.append(
            {
                "preferred_condition": condition,
                "n_units": int(len(group)),
                "median_first_onset_ms": finite_median(group["first_significant_onset_ms"]),
                "median_preferred_peak_ms": finite_median(group["preferred_peak_ms"]),
                "median_effect_peak_ms": finite_median(group["effect_peak_ms"]),
            }
        )
    timing_summary = pd.DataFrame(rows)
    tests = []
    for metric in ["first_significant_onset_ms", "preferred_peak_ms", "effect_peak_ms"]:
        groups = [timing.loc[timing["preferred_condition"] == c, metric].dropna().to_numpy(float) for c in CONDITIONS]
        if all(len(g) > 0 for g in groups):
            tests.append({"metric": metric, "comparison": "three_groups", "p_value": float(stats.kruskal(*groups).pvalue)})
        for first, second in [("active", "scrambled"), ("passive", "scrambled"), ("active", "passive")]:
            a = timing.loc[timing["preferred_condition"] == first, metric].dropna().to_numpy(float)
            b = timing.loc[timing["preferred_condition"] == second, metric].dropna().to_numpy(float)
            p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) and len(b) else np.nan
            tests.append(
                {
                    "metric": metric,
                    "comparison": f"{first}_vs_{second}",
                    "p_value": p_value,
                    "median_difference_ms": finite_median(a) - finite_median(b),
                }
            )
    return timing_summary, {"tests": tests}


def plot_diagnostics(
    unit_diag: pd.DataFrame,
    by_patient: pd.DataFrame,
    by_verb: pd.DataFrame,
    timing_summary: pd.DataFrame,
) -> dict[str, str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_verb_three_condition_scrambled_diagnostics.png"
    pdf_path = fig_dir / "fig_verb_three_condition_scrambled_diagnostics.pdf"
    font_prop = setup_matplotlib()

    fig = plt.figure(figsize=(15.6, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)
    ax_abs = fig.add_subplot(gs[0, 0])
    ax_patient = fig.add_subplot(gs[0, 1])
    ax_verb = fig.add_subplot(gs[1, 0])
    ax_timing = fig.add_subplot(gs[1, 1])

    scr = unit_diag[unit_diag["preferred_condition"] == "scrambled"]
    values = [
        scr["active_qualifying_hz"].to_numpy(float),
        scr["passive_qualifying_hz"].to_numpy(float),
        scr["scrambled_qualifying_hz"].to_numpy(float),
    ]
    labels = ["active", "passive", "scrambled"]
    bp = ax_abs.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(CONDITION_COLORS[label])
        patch.set_alpha(0.18)
        patch.set_edgecolor(CONDITION_COLORS[label])
    rng = np.random.default_rng(0)
    for i, (label, arr) in enumerate(zip(labels, values), start=1):
        x = i + rng.uniform(-0.08, 0.08, size=len(arr))
        ax_abs.scatter(x, arr, s=22, color=CONDITION_COLORS[label], alpha=0.78, edgecolors="white", linewidths=0.35)
    ax_abs.axhline(0, color="0.35", lw=0.9, linestyle="--")
    ax_abs.set_ylabel("baseline-corrected Hz in significant bins")
    ax_abs.set_title("Scrambled-high units: absolute response", loc="left", fontsize=12, weight="bold")
    ax_abs.grid(axis="y", color="0.9", lw=0.7)

    y = np.arange(len(by_patient))
    left = np.zeros(len(by_patient))
    for condition, column in [("active", "active_high"), ("passive", "passive_high"), ("scrambled", "scrambled_high")]:
        vals = by_patient[column].to_numpy(float)
        ax_patient.barh(y, vals, left=left, color=CONDITION_COLORS[condition], label=condition)
        left += vals
    ax_patient.set_yticks(y)
    ax_patient.set_yticklabels(by_patient["patient"])
    ax_patient.set_xlabel("time-screen units")
    ax_patient.set_title("Preference counts by patient", loc="left", fontsize=12, weight="bold")
    ax_patient.legend(frameon=False, ncol=3, loc="lower right")
    ax_patient.grid(axis="x", color="0.9", lw=0.7)

    x = np.arange(len(by_verb))
    heights = by_verb["median_scrambled_minus_syntax_hz"].to_numpy(float)
    ax_verb.bar(x, heights, color=CONDITION_COLORS["scrambled"], alpha=0.85)
    ax_verb.axhline(0, color="0.35", lw=0.9, linestyle="--")
    ax_verb.set_xticks(x)
    ax_verb.set_xticklabels([f"{int(v)}\n{label}" for v, label in zip(by_verb["verb_id"], by_verb["verb"])], fontproperties=font_prop)
    ax_verb.set_ylabel("median scrambled - syntax mean Hz")
    ax_verb.set_title("Scrambled-high units: item-level contrast", loc="left", fontsize=12, weight="bold")
    ax_verb.grid(axis="y", color="0.9", lw=0.7)

    x = np.arange(len(timing_summary))
    width = 0.24
    ax_timing.bar(x - width, timing_summary["median_first_onset_ms"], width, color="#7B3294", label="onset")
    ax_timing.bar(x, timing_summary["median_preferred_peak_ms"], width, color="#008B8B", label="condition peak")
    ax_timing.bar(x + width, timing_summary["median_effect_peak_ms"], width, color="#4B4B4B", label="effect peak")
    ax_timing.set_xticks(x)
    ax_timing.set_xticklabels([f"{c}-high" for c in timing_summary["preferred_condition"]])
    ax_timing.set_ylabel("ms after verb onset")
    ax_timing.set_title("Timing by preference group", loc="left", fontsize=12, weight="bold")
    ax_timing.legend(frameon=False, ncol=3, loc="upper left")
    ax_timing.grid(axis="y", color="0.9", lw=0.7)

    fig.suptitle("Diagnostics for scrambled-high verb responses", y=0.995, fontsize=15, weight="bold")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> None:
    results = pd.read_csv(OUT_DIR / "verb_three_condition_unit_results.csv")
    metrics = pd.read_csv(OUT_DIR / "verb_three_condition_unit_trial_response_metrics.csv")
    timing = pd.read_csv(OUT_DIR / "verb_three_condition_population_timing_by_unit.csv")
    sig = results[results[PRIMARY_SIGNIFICANCE_COL].astype(bool)].copy()
    sig = sig[sig["timepoint_preferred_condition"].isin(CONDITIONS)].copy()

    unit_diag, unit_summary = build_unit_diagnostics(sig)
    by_patient, by_run, patient_test = build_patient_run_tables(unit_diag)
    by_verb, item_summary = build_verb_item_table(metrics, unit_diag)
    timing_summary, timing_tests = build_timing_summary(timing)
    paths = plot_diagnostics(unit_diag, by_patient, by_verb, timing_summary)

    unit_diag.to_csv(OUT_DIR / "verb_three_condition_scrambled_absolute_vs_relative.csv", index=False)
    by_patient.to_csv(OUT_DIR / "verb_three_condition_scrambled_by_patient.csv", index=False)
    by_run.to_csv(OUT_DIR / "verb_three_condition_scrambled_by_run.csv", index=False)
    by_verb.to_csv(OUT_DIR / "verb_three_condition_scrambled_by_verb.csv", index=False)
    timing_summary.to_csv(OUT_DIR / "verb_three_condition_scrambled_timing_summary.csv", index=False)

    payload = {
        "unit_summary": unit_summary,
        "patient_test": patient_test,
        "item_summary": item_summary,
        "timing_summary": timing_summary.to_dict(orient="records"),
        "timing_tests": timing_tests["tests"],
        "outputs": paths,
        "interpretation": {
            "primary": "Scrambled-high should be described as scrambled-condition sensitivity, not stronger semantic encoding.",
            "checks": [
                "absolute-vs-relative response in qualifying bins",
                "patient/run concentration",
                "matched-verb item contrast",
                "onset/peak timing by preference group",
            ],
        },
    }
    json_path = OUT_DIR / "verb_three_condition_scrambled_diagnostics_summary.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
