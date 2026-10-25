#!/usr/bin/env python3
"""Plot statistical-test sensitivity for the good-only Rank-4 result."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_active_passive_geometry_transform_20260806 as base


ROOT = base.ROOT
DATA_DIR = (
    base.DATA_ROOT / "combined_active_passive_nested_regularized_alignment_20260807"
)
REPORT_DIR = ROOT / "reports/active_passive_nested_regularized_alignment_20260807"
FIGURE_DIR = REPORT_DIR / "figures"
PATIENT_PATH = DATA_DIR / "patient_summary.csv"
METHOD = "original_procrustes"


def one_sample_t(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))


def exact_signflip_null(values: np.ndarray, statistic: str) -> tuple[float, np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    if statistic == "mean":
        function = lambda sample: float(np.mean(sample))
    elif statistic == "studentized":
        function = one_sample_t
    else:
        raise ValueError(statistic)
    observed = function(values)
    null = np.asarray(
        [
            function(values * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ],
        dtype=float,
    )
    p_value = float(np.mean(null >= observed - 1e-12))
    return observed, null, p_value


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def compute_statistics(patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    original = (
        patient[patient["method"] == METHOD]
        .sort_values("patient")
        .reset_index(drop=True)
    )
    values = original["retrieval_margin"].to_numpy(dtype=float)
    observed_mean, mean_null, signflip_p = exact_signflip_null(values, "mean")
    observed_t, studentized_null, studentized_p = exact_signflip_null(
        values, "studentized"
    )
    t_result = stats.ttest_1samp(values, 0.0)
    t_one_sided_p = float(t_result.pvalue / 2 if t_result.statistic >= 0 else 1 - t_result.pvalue / 2)
    wilcoxon = stats.wilcoxon(values, alternative="greater", method="exact")
    sign = stats.binomtest(int(np.sum(values > 0)), len(values), 0.5, alternative="greater")

    method_t_p = []
    method_order = []
    for method, frame in patient.groupby("method", sort=True):
        test = stats.ttest_1samp(frame["retrieval_margin"].to_numpy(dtype=float), 0.0)
        method_t_p.append(
            float(test.pvalue / 2 if test.statistic >= 0 else 1 - test.pvalue / 2)
        )
        method_order.append(method)
    method_t_holm = dict(zip(method_order, holm_adjust(np.asarray(method_t_p))))

    test_rows = [
        {
            "test": "Exact sign-flip (mean)",
            "statistic": observed_mean,
            "p_value": signflip_p,
            "family": "distribution-free",
        },
        {
            "test": "Studentized sign-flip",
            "statistic": observed_t,
            "p_value": studentized_p,
            "family": "distribution-free",
        },
        {
            "test": "Exact Wilcoxon",
            "statistic": float(wilcoxon.statistic),
            "p_value": float(wilcoxon.pvalue),
            "family": "distribution-free",
        },
        {
            "test": "Binomial sign test",
            "statistic": float(np.sum(values > 0)),
            "p_value": float(sign.pvalue),
            "family": "distribution-free",
        },
        {
            "test": "Parametric one-sample t",
            "statistic": float(t_result.statistic),
            "p_value": t_one_sided_p,
            "family": "parametric-normal",
        },
        {
            "test": "Parametric t + Holm (3 methods)",
            "statistic": float(t_result.statistic),
            "p_value": float(method_t_holm[METHOD]),
            "family": "parametric-normal",
        },
    ]
    tests = pd.DataFrame(test_rows)
    null_rows = []
    for index in range(len(mean_null)):
        null_rows.append(
            {
                "sign_index": index,
                "mean_statistic": mean_null[index],
                "studentized_t_statistic": studentized_null[index],
                "mean_exceeds_observed": bool(mean_null[index] >= observed_mean - 1e-12),
                "studentized_exceeds_observed": bool(
                    studentized_null[index] >= observed_t - 1e-12
                ),
            }
        )
    shapiro = stats.shapiro(values)
    summary = {
        "n_patients": int(len(values)),
        "patient_mean_margin": float(values.mean()),
        "patient_sd_margin": float(values.std(ddof=1)),
        "positive_patients": int(np.sum(values > 0)),
        "observed_t": float(t_result.statistic),
        "t_degrees_of_freedom": int(len(values) - 1),
        "shapiro_w": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "note": "Shapiro-Wilk has very low diagnostic power at n=5",
    }
    return tests, pd.DataFrame(null_rows), summary


def plot_figure(
    patient: pd.DataFrame,
    tests: pd.DataFrame,
    null: pd.DataFrame,
    summary: dict[str, float],
) -> None:
    base.configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    original = (
        patient[patient["method"] == METHOD]
        .sort_values("patient")
        .reset_index(drop=True)
    )
    values = original["retrieval_margin"].to_numpy(dtype=float)
    patients = original["patient"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.3), constrained_layout=True)

    ax = axes[0, 0]
    x = np.linspace(-0.10, 0.10, len(values))
    ax.scatter(
        x,
        values,
        s=48,
        color=base.OKABE_ITO["purple"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    for x_value, y_value, patient_name in zip(x, values, patients):
        ax.annotate(
            patient_name,
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
        )
    ax.scatter(0, values.mean(), marker="_", s=260, color="black", linewidth=2.2)
    ax.axhline(0, color=base.OKABE_ITO["gray"], lw=0.9, ls="--")
    ax.set_xlim(-0.19, 0.19)
    ax.set_xticks([0], ["Original rank-4\nProcrustes"])
    ax.set_ylabel("Held-pair retrieval margin")
    ax.set_title("A  Patient-level effect")
    ax.text(
        0.02,
        0.98,
        f"Mean = {values.mean():.4f}\nPositive = {np.sum(values > 0)}/5",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )

    ax = axes[0, 1]
    order = [
        "Exact sign-flip (mean)",
        "Studentized sign-flip",
        "Exact Wilcoxon",
        "Binomial sign test",
        "Parametric one-sample t",
        "Parametric t + Holm (3 methods)",
    ]
    plot = tests.set_index("test").loc[order].reset_index()
    y = np.arange(len(plot))[::-1]
    colors = [
        base.OKABE_ITO["blue"] if family == "distribution-free" else base.OKABE_ITO["vermillion"]
        for family in plot["family"]
    ]
    ax.hlines(y, 0, plot["p_value"], color=colors, lw=2)
    ax.scatter(plot["p_value"], y, color=colors, s=38, edgecolor="white", linewidth=0.5)
    ax.axvline(0.05, color=base.OKABE_ITO["gray"], lw=1.0, ls="--")
    for y_value, p_value in zip(y, plot["p_value"]):
        ax.text(p_value + 0.006, y_value, f"{p_value:.4f}", va="center", fontsize=6.7)
    ax.set_yticks(y, order)
    ax.set_xlim(0, max(0.22, float(plot["p_value"].max()) + 0.04))
    ax.set_xlabel("One-sided P value")
    ax.set_title("B  Test-choice sensitivity")
    ax.text(
        0.98,
        0.04,
        "blue: distribution-free\nred: normal-model dependent",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
    )

    ax = axes[1, 0]
    grouped = (
        null.groupby("mean_statistic", as_index=False)
        .agg(count=("sign_index", "size"), exceeds=("mean_exceeds_observed", "max"))
        .sort_values("mean_statistic")
    )
    for row in grouped.itertuples(index=False):
        color = base.OKABE_ITO["vermillion"] if row.exceeds else base.OKABE_ITO["light_gray"]
        ax.vlines(row.mean_statistic, 0, row.count, color=color, lw=2.1)
        ax.scatter(row.mean_statistic, row.count, color=color, s=20, zorder=3)
    observed_mean = float(summary["patient_mean_margin"])
    ax.axvline(observed_mean, color=base.OKABE_ITO["purple"], lw=1.8)
    ax.axvline(0, color=base.OKABE_ITO["gray"], lw=0.8, ls="--")
    ax.set_xlabel("Mean margin under 32 sign patterns")
    ax.set_ylabel("Frequency")
    ax.set_title("C  Exact discrete null")
    ax.text(
        0.03,
        0.96,
        "2/32 null values ≥ observed\nExact P = 0.0625",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )

    ax = axes[1, 1]
    degrees = int(summary["t_degrees_of_freedom"])
    observed_t = float(summary["observed_t"])
    grid = np.linspace(-5.5, 5.5, 600)
    density = stats.t.pdf(grid, df=degrees)
    ax.plot(grid, density, color=base.OKABE_ITO["black"], lw=1.7)
    tail = grid >= observed_t
    ax.fill_between(
        grid[tail],
        0,
        density[tail],
        color=base.OKABE_ITO["vermillion"],
        alpha=0.55,
    )
    ax.axvline(observed_t, color=base.OKABE_ITO["vermillion"], lw=1.8)
    ax.axvline(0, color=base.OKABE_ITO["gray"], lw=0.8, ls="--")
    t_p = float(
        tests.loc[tests["test"] == "Parametric one-sample t", "p_value"].iloc[0]
    )
    ax.set_xlabel(f"t statistic under t({degrees}) reference distribution")
    ax.set_ylabel("Density")
    ax.set_title("D  Parametric continuous null")
    ax.text(
        0.03,
        0.96,
        f"Observed t = {observed_t:.3f}\nOne-sided P = {t_p:.4f}\nRequires a normal patient-effect model",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )

    fig.suptitle(
        "Good-only statistical sensitivity of rank-4 Procrustes alignment\n"
        "Target noun, 0–800 ms; five patients are the independent units",
        fontsize=10,
    )
    stem = FIGURE_DIR / "Fig_7_good_only_statistical_test_sensitivity"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(tests: pd.DataFrame, summary: dict[str, float]) -> None:
    lines = [
        "# good_only统计检验敏感性",
        "",
        "分析对象固定为target-noun onset后0–800 ms、Rank-4 Original Procrustes retrieval margin；5位患者为独立单位。",
        "",
        "| 检验 | 统计量 | 单侧P | 假设类别 |",
        "|---|---:|---:|---|",
    ]
    for row in tests.itertuples(index=False):
        lines.append(
            f"| {row.test} | {row.statistic:.5f} | {row.p_value:.5f} | {row.family} |"
        )
    lines.extend(
        [
            "",
            f"患者均值margin={summary['patient_mean_margin']:.5f}，SD={summary['patient_sd_margin']:.5f}，正向患者={summary['positive_patients']}/5。",
            "",
            "- 分布无关检验均未达到P<0.05；精确符号翻转和Wilcoxon均为P=0.0625。",
            "- 参数单样本t检验为P=0.0177，三方法Holm校正后P=0.0498；该结果依赖患者效应服从正态总体的模型假设。",
            "- n=5时Shapiro–Wilk无法有效确认正态性，因此t检验只能作为假设敏感性结果，不能替代精确患者检验。",
            "",
        ]
    )
    (REPORT_DIR / "GOOD_ONLY_STATISTICAL_SENSITIVITY_REPORT_zh.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    caption = (
        "Fig. 7 | Statistical-test sensitivity of the good-only rank-4 Procrustes result. "
        "a, Held-pair retrieval margins for the five patients. b, One-sided P values from "
        "distribution-free and parametric tests applied to the same patient effects. c, The "
        "32-value exact sign-flip null distribution; two sign patterns were at least as large "
        "as the observed mean. d, The continuous t(4) reference distribution, whose smaller "
        "tail probability relies on a normal population model for patient effects. The "
        "parametric result is an assumption-sensitive secondary analysis."
    )
    (REPORT_DIR / "Fig7_caption_en.txt").write_text(caption, encoding="utf-8")


def main() -> None:
    patient = pd.read_csv(PATIENT_PATH)
    tests, null, summary = compute_statistics(patient)
    tests.to_csv(DATA_DIR / "good_only_statistical_test_sensitivity.csv", index=False)
    null.to_csv(DATA_DIR / "good_only_original_procrustes_exact_null.csv", index=False)
    (DATA_DIR / "good_only_statistical_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_figure(patient, tests, null, summary)
    write_report(tests, summary)
    expected = {
        "Exact sign-flip (mean)": 0.0625,
        "Exact Wilcoxon": 0.0625,
        "Parametric one-sample t": 0.017673,
    }
    observed = tests.set_index("test")["p_value"]
    validation = {
        name: bool(np.isclose(observed[name], value, atol=1e-5))
        for name, value in expected.items()
    }
    validation["passed"] = bool(all(validation.values()))
    (REPORT_DIR / "Fig7_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not validation["passed"]:
        raise RuntimeError(f"Validation failed: {validation}")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
