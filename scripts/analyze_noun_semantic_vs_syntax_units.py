#!/usr/bin/env python3
"""Classify noun-role units as semantic, syntactic, or mixed encoders."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import scripts.analyze_noun_role_structure_encoding as analysis


OUT_DIR = analysis.OUT_DIR
FIG_DIR = OUT_DIR / "figures"
FACTOR_TESTS = OUT_DIR / "noun_role_multifactor_encoding_factor_unit_tests.csv"
RESPONSE_METRICS = OUT_DIR / "noun_role_unit_trial_response_metrics.csv"
OVERALL_DECODE = OUT_DIR / "noun_role_multifactor_decoding_overall_summary.csv"

METRIC_LABEL = {
    "bc": "baseline-corrected",
    "raw": "raw response",
}
CLASS_ORDER = ["role_only", "noun_only", "noun_and_role"]
CLASS_LABEL = {
    "role_only": "syntax only",
    "noun_only": "semantic only",
    "noun_and_role": "semantic + syntax",
    "neither": "neither",
}
CLASS_COLOR = {
    "role_only": analysis.ACTIVE,
    "noun_only": analysis.PREDICATE,
    "noun_and_role": "#7A5195",
    "neither": "#D0D0D0",
}
FACTOR_COLOR = {
    "noun": analysis.PREDICATE,
    "role": analysis.ACTIVE,
    "voice": "#F58518",
}


def binom_enrichment_p(k: int, n: int, p0: float = 0.05) -> float:
    if n <= 0:
        return np.nan
    return float(stats.binom.sf(k - 1, n, p0)) if k > 0 else 1.0


def add_unit_ids(classified: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    unit_maps: dict[tuple[str, str], dict[int, int]] = {}
    for (patient, run), group in metrics.groupby(["patient", "run"], sort=False):
        unit_ids = sorted(group["unit_id"].dropna().astype(int).unique())
        unit_maps[(patient, run)] = {idx: unit_id for idx, unit_id in enumerate(unit_ids)}

    out = classified.copy()
    out["unit_id"] = [
        unit_maps.get((row.patient, row.run), {}).get(int(row.unit_index), np.nan)
        for row in out.itertuples(index=False)
    ]
    cols = out.columns.tolist()
    insert_at = cols.index("unit_index") + 1
    cols.insert(insert_at, cols.pop(cols.index("unit_id")))
    return out[cols]


def classify_units(factor_tests: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    values = ["p_value", "q_value_run_factor_fdr", "partial_eta2", "f_value", "full_model_r2"]
    wide = factor_tests.pivot_table(
        index=["patient", "run", "metric", "unit_index"],
        columns="factor",
        values=values,
        aggfunc="first",
    )
    wide.columns = [f"{stat}_{factor}" for stat, factor in wide.columns]
    wide = wide.reset_index()

    for prefix, suffix in (("p_value", "uncorrected"), ("q_value_run_factor_fdr", "run_factor_fdr")):
        noun = wide[f"{prefix}_noun"] < 0.05
        role = wide[f"{prefix}_role"] < 0.05
        voice = wide[f"{prefix}_voice"] < 0.05
        wide[f"semantic_noun_sig_{suffix}"] = noun
        wide[f"syntax_role_sig_{suffix}"] = role
        wide[f"voice_sig_{suffix}"] = voice
        wide[f"class_semantic_syntax_{suffix}"] = np.select(
            [noun & role, noun & ~role, ~noun & role],
            ["noun_and_role", "noun_only", "role_only"],
            default="neither",
        )

    return add_unit_ids(wide, metrics)


def summarize(classified: pd.DataFrame, threshold_suffix: str) -> pd.DataFrame:
    rows = []
    for (patient, metric), group in classified.groupby(["patient", "metric"], sort=False):
        semantic = group[f"semantic_noun_sig_{threshold_suffix}"]
        syntax = group[f"syntax_role_sig_{threshold_suffix}"]
        voice = group[f"voice_sig_{threshold_suffix}"]
        class_col = group[f"class_semantic_syntax_{threshold_suffix}"]
        rows.append(
            {
                "scope": "patient",
                "patient": patient,
                "metric": metric,
                "threshold": threshold_suffix,
                "n_units": len(group),
                "semantic_noun_units": int(semantic.sum()),
                "syntax_role_units": int(syntax.sum()),
                "voice_units": int(voice.sum()),
                "noun_only_units": int((class_col == "noun_only").sum()),
                "role_only_units": int((class_col == "role_only").sum()),
                "noun_and_role_units": int((class_col == "noun_and_role").sum()),
                "neither_units": int((class_col == "neither").sum()),
                "semantic_pct": float(semantic.mean() * 100),
                "syntax_pct": float(syntax.mean() * 100),
                "voice_pct": float(voice.mean() * 100),
                "semantic_enrichment_p_vs_5pct": binom_enrichment_p(int(semantic.sum()), len(group)),
                "syntax_enrichment_p_vs_5pct": binom_enrichment_p(int(syntax.sum()), len(group)),
                "voice_enrichment_p_vs_5pct": binom_enrichment_p(int(voice.sum()), len(group)),
            }
        )

    for metric, group in classified.groupby("metric", sort=False):
        semantic = group[f"semantic_noun_sig_{threshold_suffix}"]
        syntax = group[f"syntax_role_sig_{threshold_suffix}"]
        voice = group[f"voice_sig_{threshold_suffix}"]
        class_col = group[f"class_semantic_syntax_{threshold_suffix}"]
        rows.append(
            {
                "scope": "overall",
                "patient": "all",
                "metric": metric,
                "threshold": threshold_suffix,
                "n_units": len(group),
                "semantic_noun_units": int(semantic.sum()),
                "syntax_role_units": int(syntax.sum()),
                "voice_units": int(voice.sum()),
                "noun_only_units": int((class_col == "noun_only").sum()),
                "role_only_units": int((class_col == "role_only").sum()),
                "noun_and_role_units": int((class_col == "noun_and_role").sum()),
                "neither_units": int((class_col == "neither").sum()),
                "semantic_pct": float(semantic.mean() * 100),
                "syntax_pct": float(syntax.mean() * 100),
                "voice_pct": float(voice.mean() * 100),
                "semantic_enrichment_p_vs_5pct": binom_enrichment_p(int(semantic.sum()), len(group)),
                "syntax_enrichment_p_vs_5pct": binom_enrichment_p(int(syntax.sum()), len(group)),
                "voice_enrichment_p_vs_5pct": binom_enrichment_p(int(voice.sum()), len(group)),
            }
        )

    return pd.DataFrame(rows)


def plot_class_counts(ax, classified: pd.DataFrame, metric: str) -> None:
    sub = classified[classified["metric"] == metric]
    patients = sorted(sub["patient"].unique())
    bottoms = np.zeros(len(patients))
    x = np.arange(len(patients))
    for class_name in CLASS_ORDER:
        vals = np.array(
            [
                int(
                    (
                        sub[(sub["patient"] == patient)]["class_semantic_syntax_run_factor_fdr"]
                        == class_name
                    ).sum()
                )
                for patient in patients
            ],
            dtype=float,
        )
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=CLASS_LABEL[class_name],
            color=CLASS_COLOR[class_name],
            width=0.68,
        )
        for xi, bottom, val in zip(x, bottoms, vals):
            if val > 0:
                ax.text(xi, bottom + val / 2, str(int(val)), ha="center", va="center", color="white", fontsize=8)
        bottoms += vals
    totals = [len(sub[sub["patient"] == patient]) for patient in patients]
    for xi, total in zip(x, totals):
        ax.text(xi, max(bottoms.max(), 1) + 0.5, f"n={total}", ha="center", va="bottom", fontsize=7, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels(patients)
    ax.set_ylabel("FDR<0.05 units")
    ax.set_title(f"{METRIC_LABEL[metric]}: strict unit classes", loc="left", weight="bold")
    ax.grid(axis="y", color="0.9")
    ax.set_ylim(0, max(bottoms.max() + 4, 5))


def plot_uncorrected_rates(ax, classified: pd.DataFrame) -> None:
    metrics = ["bc", "raw"]
    factors = ["noun", "role", "voice"]
    width = 0.22
    x = np.arange(len(metrics))
    for i, factor in enumerate(factors):
        vals = []
        labels = []
        for metric in metrics:
            sub = classified[classified["metric"] == metric]
            sig = sub[f"{'semantic_noun' if factor == 'noun' else 'syntax_role' if factor == 'role' else 'voice'}_sig_uncorrected"]
            vals.append(sig.mean() * 100)
            labels.append(f"{int(sig.sum())}/{len(sig)}")
        positions = x + (i - 1) * width
        ax.bar(positions, vals, width=width, color=FACTOR_COLOR[factor], label=factor)
        for xi, val, label in zip(positions, vals, labels):
            ax.text(xi, val + 0.7, label, ha="center", va="bottom", fontsize=7)
    ax.axhline(5, color="0.25", ls="--", lw=1, label="5% null")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in metrics])
    ax.set_ylabel("uncorrected p<0.05 units (%)")
    ax.set_title("Exploratory p<0.05 rates vs 5% null", loc="left", weight="bold")
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    ax.set_ylim(0, 18)


def plot_decoding(ax, overall: pd.DataFrame) -> None:
    target_order = ["noun", "role", "voice"]
    metrics = ["bc", "raw"]
    width = 0.34
    x = np.arange(len(target_order))
    colors = {"bc": "#4C78A8", "raw": "#9E9E9E"}
    for i, metric in enumerate(metrics):
        sub = overall[overall["metric"] == metric].set_index("target")
        vals = np.array([sub.loc[target, "accuracy"] * 100 for target in target_order])
        positions = x + (i - 0.5) * width
        ax.bar(positions, vals, width=width, color=colors[metric], label=METRIC_LABEL[metric])
        for xi, val in zip(positions, vals):
            ax.text(xi, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.axhline(5, color="0.25", ls=":", lw=1)
    ax.axhline(50, color="0.25", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["noun identity", "role", "voice"])
    ax.set_ylabel("population decoding accuracy (%)")
    ax.set_title("Population decoding check", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.9")
    ax.set_ylim(0, 80)


def plot_summary(classified: pd.DataFrame, overall: pd.DataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(exist_ok=True)
    png_path = FIG_DIR / "fig_noun_semantic_vs_syntax_unit_encoding.png"
    pdf_path = FIG_DIR / "fig_noun_semantic_vs_syntax_unit_encoding.pdf"

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4))
    ax_bc, ax_raw, ax_rates, ax_decode = axes.ravel()
    plot_class_counts(ax_bc, classified, "bc")
    plot_class_counts(ax_raw, classified, "raw")
    ax_bc.legend(frameon=False, loc="upper left")
    plot_uncorrected_rates(ax_rates, classified)
    plot_decoding(ax_decode, overall)
    fig.suptitle(
        "Semantic (noun identity) vs syntactic (role/position) encoding at unit and population levels",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> None:
    analysis.setup_matplotlib()
    factor_tests = pd.read_csv(FACTOR_TESTS)
    metrics = pd.read_csv(RESPONSE_METRICS, usecols=["patient", "run", "unit_id"])
    overall = pd.read_csv(OVERALL_DECODE)

    classified = classify_units(factor_tests, metrics)
    summaries = pd.concat(
        [
            summarize(classified, "run_factor_fdr"),
            summarize(classified, "uncorrected"),
        ],
        ignore_index=True,
    )

    classified_path = OUT_DIR / "noun_role_semantic_vs_syntax_unit_classification.csv"
    summary_path = OUT_DIR / "noun_role_semantic_vs_syntax_unit_summary.csv"
    classified.to_csv(classified_path, index=False)
    summaries.to_csv(summary_path, index=False)
    fig_paths = plot_summary(classified, overall)

    overall_summary = summaries[(summaries["scope"] == "overall")].copy()
    print("overall unit summary")
    print(
        overall_summary[
            [
                "metric",
                "threshold",
                "n_units",
                "semantic_noun_units",
                "syntax_role_units",
                "voice_units",
                "noun_only_units",
                "role_only_units",
                "noun_and_role_units",
                "semantic_pct",
                "syntax_pct",
                "semantic_enrichment_p_vs_5pct",
                "syntax_enrichment_p_vs_5pct",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )
    print("\noutputs")
    for path in [classified_path, summary_path, Path(fig_paths["png"]), Path(fig_paths["pdf"])]:
        print(path)


if __name__ == "__main__":
    main()
