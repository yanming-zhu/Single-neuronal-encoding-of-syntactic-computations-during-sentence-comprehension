#!/usr/bin/env python3
"""Decode noun identity, role, and voice from noun-aligned unit responses."""

from __future__ import annotations

import os
import re
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import scripts.analyze_noun_role_structure_encoding as analysis


RNG = np.random.default_rng(20260723)
METRICS = {
    "bc": "response_baseline_corrected_hz",
    "raw": "response_hz",
}
METRIC_LABEL = {
    "bc": "baseline-corrected",
    "raw": "raw response",
}
CHANCE_NOUN = 1 / 20


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * pv.size / np.arange(1, pv.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(pv)
    out[order] = adjusted
    q[valid] = out
    return q


def parse_pair_id(sound_file: str) -> str:
    match = re.search(r"(?:act|pass)(\d+)", str(sound_file))
    return match.group(1) if match else str(sound_file)


def event_matrix(metrics: pd.DataFrame, metric_col: str, patient: str, run: str) -> tuple[pd.DataFrame, np.ndarray]:
    event_cols = [
        "patient",
        "run",
        "trial_index",
        "sound_file",
        "sentence_condition",
        "role",
        "noun",
        "counterpart_noun",
        "event_bin_1000Hz",
    ]
    run_metrics = metrics[(metrics["patient"] == patient) & (metrics["run"] == run)].copy()
    meta = run_metrics[event_cols].drop_duplicates().sort_values(["trial_index", "role"]).reset_index(drop=True)
    mat = (
        run_metrics.pivot_table(
            index=event_cols,
            columns="unit_id",
            values=metric_col,
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["trial_index", "role"])
        .reset_index(drop=True)
    )
    feature_cols = [col for col in mat.columns if isinstance(col, (int, np.integer))]
    x = mat[feature_cols].to_numpy(float)
    meta = mat[event_cols].copy()
    meta["pair_id"] = meta["sound_file"].map(parse_pair_id)
    return meta, x


def train_standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    return (x_train - mean) / sd, (x_test - mean) / sd


def nearest_template_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    xtr, xte = train_standardize(x_train, x_test)
    d2 = ((xte[:, None, :] - xtr[None, :, :]) ** 2).sum(axis=2)
    return y_train[np.argmin(d2, axis=1)]


def decode_noun_cross_role(meta: pd.DataFrame, x: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    pred_rows = []
    for train_role, test_role in (("subject", "predicate"), ("predicate", "subject")):
        train_mask = meta["role"].to_numpy() == train_role
        test_mask = meta["role"].to_numpy() == test_role
        y_train = meta.loc[train_mask, "noun"].to_numpy(str)
        y_test = meta.loc[test_mask, "noun"].to_numpy(str)
        y_pred = nearest_template_predictions(x[train_mask], y_train, x[test_mask])
        test_meta = meta.loc[test_mask].reset_index(drop=True)
        for row, truth, pred in zip(test_meta.itertuples(index=False), y_test, y_pred):
            pred_rows.append(
                {
                    "trial_index": int(row.trial_index),
                    "role": str(row.role),
                    "sentence_condition": str(row.sentence_condition),
                    "noun": str(truth),
                    "predicted_noun": str(pred),
                    "correct": bool(truth == pred),
                    "train_role": train_role,
                    "test_role": test_role,
                    "pair_id": str(row.pair_id),
                }
            )
    pred_df = pd.DataFrame(pred_rows)
    n = len(pred_df)
    k = int(pred_df["correct"].sum())
    summary = {
        "n_tests": n,
        "n_correct": k,
        "accuracy": k / n if n else np.nan,
        "chance": CHANCE_NOUN,
        "binom_p": stats.binom.sf(k - 1, n, CHANCE_NOUN) if k > 0 else 1.0,
    }
    return pred_df, summary


def binary_decoder_cv(meta: pd.DataFrame, x: np.ndarray, target: str, group_col: str) -> dict[str, float]:
    y = meta[target].to_numpy(str)
    groups = meta[group_col].to_numpy(str)
    pred = np.empty_like(y)
    ok = np.zeros(len(y), dtype=bool)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        if np.unique(y[train]).size < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="liblinear",
                max_iter=2000,
                random_state=20260723,
            ),
        )
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
        ok[test] = True
    y_ok = y[ok]
    pred_ok = pred[ok]
    n = len(y_ok)
    k = int((y_ok == pred_ok).sum())
    return {
        "n_tests": n,
        "n_correct": k,
        "accuracy": k / n if n else np.nan,
        "chance": 0.5,
        "binom_p": stats.binomtest(k, n, 0.5).pvalue if n else np.nan,
    }


def design_matrix(meta: pd.DataFrame, factors: list[str]) -> np.ndarray:
    intercept = np.ones((len(meta), 1), dtype=float)
    if not factors:
        return intercept
    enc = OneHotEncoder(drop="first", sparse_output=False, dtype=float)
    coded = enc.fit_transform(meta[factors].astype(str))
    return np.column_stack([intercept, coded])


def fit_rss(x_design: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    resid = y - x_design @ coef
    return float(np.sum(resid**2)), int(np.linalg.matrix_rank(x_design))


def factor_model_tests(meta: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    factors = ["noun", "role", "sentence_condition"]
    full_x = design_matrix(meta, factors)
    rows = []
    for unit_i in range(x.shape[1]):
        y = x[:, unit_i].astype(float)
        if not np.isfinite(y).all() or np.nanstd(y) < 1e-8:
            continue
        rss_full, rank_full = fit_rss(full_x, y)
        df_error = len(y) - rank_full
        if df_error <= 0 or rss_full <= 0:
            continue
        tss = float(np.sum((y - y.mean()) ** 2))
        full_r2 = 1.0 - rss_full / tss if tss > 0 else np.nan
        for factor in factors:
            reduced_factors = [item for item in factors if item != factor]
            reduced_x = design_matrix(meta, reduced_factors)
            rss_reduced, rank_reduced = fit_rss(reduced_x, y)
            df_factor = rank_full - rank_reduced
            ss_factor = max(0.0, rss_reduced - rss_full)
            if df_factor <= 0:
                f_value = np.nan
                p_value = np.nan
            else:
                f_value = (ss_factor / df_factor) / (rss_full / df_error)
                p_value = float(stats.f.sf(f_value, df_factor, df_error))
            partial_eta2 = ss_factor / (ss_factor + rss_full) if (ss_factor + rss_full) > 0 else np.nan
            rows.append(
                {
                    "unit_index": unit_i,
                    "factor": "voice" if factor == "sentence_condition" else factor,
                    "df_factor": df_factor,
                    "df_error": df_error,
                    "f_value": f_value,
                    "p_value": p_value,
                    "partial_eta2": partial_eta2,
                    "full_model_r2": full_r2,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_run_factor_fdr"] = np.nan
        for factor, idx in out.groupby("factor").groups.items():
            out.loc[idx, "q_value_run_factor_fdr"] = bh_fdr(out.loc[idx, "p_value"].to_numpy(float))
        out["significant_uncorrected_p05"] = out["p_value"] < 0.05
        out["significant_run_factor_fdr05"] = out["q_value_run_factor_fdr"] < 0.05
    return out


def plot_noun_summary(noun_summary: pd.DataFrame, out_dir: Path, font_prop) -> dict[str, str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_noun_identity_cross_role_decoding_by_noun.png"
    pdf_path = fig_dir / "fig_noun_identity_cross_role_decoding_by_noun.pdf"

    plot_df = noun_summary[noun_summary["metric"] == "bc"].copy()
    plot_df = plot_df.sort_values("accuracy", ascending=True)
    y = np.arange(len(plot_df))
    colors = np.where(plot_df["q_value_fdr"] < 0.05, analysis.ACTIVE, "#9E9E9E")
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    ax.barh(y, plot_df["accuracy"] * 100, color=colors, alpha=0.9)
    ax.axvline(CHANCE_NOUN * 100, color="0.25", ls="--", lw=1, label="chance = 5%")
    for yi, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            row.accuracy * 100 + 0.8,
            yi,
            f"{int(row.n_correct)}/{int(row.n_tests)}",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["noun"].tolist(), fontproperties=font_prop)
    ax.set_xlabel("cross-role noun decoding accuracy (%)")
    ax.set_title("Noun identity decoding by noun", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="0.9")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def plot_multifactor_summary(
    run_summary: pd.DataFrame,
    factor_summary: pd.DataFrame,
    out_dir: Path,
) -> dict[str, str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    png_path = fig_dir / "fig_noun_role_multifactor_decoding_summary.png"
    pdf_path = fig_dir / "fig_noun_role_multifactor_decoding_summary.pdf"

    patient_decode = (
        run_summary.groupby(["patient", "metric"], as_index=False)
        .agg(
            noun_correct=("noun_correct", "sum"),
            noun_tests=("noun_tests", "sum"),
            role_correct=("role_correct", "sum"),
            role_tests=("role_tests", "sum"),
            voice_correct=("voice_correct", "sum"),
            voice_tests=("voice_tests", "sum"),
        )
    )
    for target in ("noun", "role", "voice"):
        patient_decode[f"{target}_accuracy"] = patient_decode[f"{target}_correct"] / patient_decode[f"{target}_tests"]
    factor_patient = (
        factor_summary[factor_summary["metric"] == "bc"]
        .groupby(["patient", "factor"], as_index=False)
        .agg(
            n_units=("n_units", "sum"),
            fdr_units=("n_run_factor_fdr05", "sum"),
            median_partial_eta2=("median_partial_eta2", "median"),
        )
    )

    patients = sorted(patient_decode["patient"].unique())
    x = np.arange(len(patients))
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8))
    ax_noun, ax_binary, ax_fdr, ax_eta = axes.ravel()

    def patient_values(metric: str, column: str) -> np.ndarray:
        sub = patient_decode[patient_decode["metric"] == metric].set_index("patient")
        return np.array([sub.loc[p, column] if p in sub.index else np.nan for p in patients], dtype=float)

    width = 0.34
    ax_noun.bar(x - width / 2, patient_values("bc", "noun_accuracy") * 100, width=width, label="baseline-corrected")
    ax_noun.bar(x + width / 2, patient_values("raw", "noun_accuracy") * 100, width=width, label="raw response")
    ax_noun.axhline(CHANCE_NOUN * 100, color="0.25", ls="--", lw=1, label="chance = 5%")
    ax_noun.set_xticks(x)
    ax_noun.set_xticklabels(patients)
    ax_noun.set_ylabel("accuracy (%)")
    ax_noun.set_title("A. Cross-role noun identity decoding", loc="left", weight="bold")
    ax_noun.legend(frameon=False, fontsize=7)
    ax_noun.grid(axis="y", color="0.9")

    role_bc = patient_values("bc", "role_accuracy") * 100
    voice_bc = patient_values("bc", "voice_accuracy") * 100
    role_raw = patient_values("raw", "role_accuracy") * 100
    voice_raw = patient_values("raw", "voice_accuracy") * 100
    ax_binary.plot(x, role_bc, marker="o", color=analysis.ACTIVE, label="role bc")
    ax_binary.plot(x, role_raw, marker="o", color=analysis.ACTIVE, ls="--", label="role raw")
    ax_binary.plot(x, voice_bc, marker="s", color=analysis.PREDICATE, label="voice bc")
    ax_binary.plot(x, voice_raw, marker="s", color=analysis.PREDICATE, ls="--", label="voice raw")
    ax_binary.axhline(50, color="0.25", ls="--", lw=1, label="chance = 50%")
    ax_binary.set_xticks(x)
    ax_binary.set_xticklabels(patients)
    ax_binary.set_ylim(0, 105)
    ax_binary.set_ylabel("accuracy (%)")
    ax_binary.set_title("B. Role and voice decoding", loc="left", weight="bold")
    ax_binary.legend(frameon=False, fontsize=7, ncol=2)
    ax_binary.grid(axis="y", color="0.9")

    factors = ["noun", "role", "voice"]
    colors = {"noun": "#4C78A8", "role": "#59A14F", "voice": "#F58518"}
    width = 0.24
    for i, factor in enumerate(factors):
        sub = factor_patient[factor_patient["factor"] == factor].set_index("patient")
        vals = np.array([sub.loc[p, "fdr_units"] if p in sub.index else 0 for p in patients], dtype=float)
        ax_fdr.bar(x + (i - 1) * width, vals, width=width, color=colors[factor], label=factor)
        for xi, val in zip(x + (i - 1) * width, vals):
            if val:
                ax_fdr.text(xi, val + 0.25, f"{int(val)}", ha="center", va="bottom", fontsize=7)
    ax_fdr.set_xticks(x)
    ax_fdr.set_xticklabels(patients)
    ax_fdr.set_ylabel("FDR<0.05 units")
    ax_fdr.set_title("C. Encoding-model factor counts", loc="left", weight="bold")
    ax_fdr.legend(frameon=False)
    ax_fdr.grid(axis="y", color="0.9")

    for factor in factors:
        sub = factor_patient[factor_patient["factor"] == factor].set_index("patient")
        vals = np.array([sub.loc[p, "median_partial_eta2"] if p in sub.index else np.nan for p in patients], dtype=float)
        ax_eta.plot(x, vals, marker="o", color=colors[factor], label=factor)
    ax_eta.set_xticks(x)
    ax_eta.set_xticklabels(patients)
    ax_eta.set_ylabel("median partial eta squared")
    ax_eta.set_title("D. Encoding-model effect size", loc="left", weight="bold")
    ax_eta.legend(frameon=False)
    ax_eta.grid(axis="y", color="0.9")

    fig.suptitle("Multifactor noun-role decoding and encoding summary", fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> None:
    font_prop = analysis.setup_matplotlib()
    out_dir = analysis.OUT_DIR
    metrics = pd.read_csv(out_dir / "noun_role_unit_trial_response_metrics.csv")
    run_order = list(metrics[["patient", "run"]].drop_duplicates().itertuples(index=False, name=None))

    run_rows = []
    pred_tables = []
    factor_tables = []
    for metric_name, metric_col in METRICS.items():
        for patient, run in run_order:
            meta, x = event_matrix(metrics, metric_col, patient, run)
            noun_pred, noun_summary = decode_noun_cross_role(meta, x)
            noun_pred.insert(0, "metric", metric_name)
            noun_pred.insert(0, "run", run)
            noun_pred.insert(0, "patient", patient)
            pred_tables.append(noun_pred)

            role_summary = binary_decoder_cv(meta, x, "role", "noun")
            voice_summary = binary_decoder_cv(meta, x, "sentence_condition", "pair_id")
            run_rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "metric": metric_name,
                    "n_units": x.shape[1],
                    "noun_correct": noun_summary["n_correct"],
                    "noun_tests": noun_summary["n_tests"],
                    "noun_accuracy": noun_summary["accuracy"],
                    "noun_binom_p": noun_summary["binom_p"],
                    "role_correct": role_summary["n_correct"],
                    "role_tests": role_summary["n_tests"],
                    "role_accuracy": role_summary["accuracy"],
                    "role_binom_p": role_summary["binom_p"],
                    "voice_correct": voice_summary["n_correct"],
                    "voice_tests": voice_summary["n_tests"],
                    "voice_accuracy": voice_summary["accuracy"],
                    "voice_binom_p": voice_summary["binom_p"],
                }
            )

            factor_tests = factor_model_tests(meta, x)
            factor_tests.insert(0, "metric", metric_name)
            factor_tests.insert(0, "run", run)
            factor_tests.insert(0, "patient", patient)
            factor_tables.append(factor_tests)

    run_summary = pd.DataFrame(run_rows)
    predictions = pd.concat(pred_tables, ignore_index=True)
    factor_unit = pd.concat(factor_tables, ignore_index=True)

    noun_rows = []
    for (metric_name, noun), group in predictions.groupby(["metric", "noun"], sort=False):
        n = len(group)
        k = int(group["correct"].sum())
        noun_rows.append(
            {
                "metric": metric_name,
                "noun": noun,
                "n_tests": n,
                "n_correct": k,
                "accuracy": k / n if n else np.nan,
                "chance": CHANCE_NOUN,
                "binom_p": stats.binom.sf(k - 1, n, CHANCE_NOUN) if k > 0 else 1.0,
            }
        )
    noun_summary = pd.DataFrame(noun_rows)
    for metric_name, idx in noun_summary.groupby("metric").groups.items():
        noun_summary.loc[idx, "q_value_fdr"] = bh_fdr(noun_summary.loc[idx, "binom_p"].to_numpy(float))
    noun_summary["significant_fdr05"] = noun_summary["q_value_fdr"] < 0.05

    factor_summary = (
        factor_unit.groupby(["patient", "run", "metric", "factor"], as_index=False)
        .agg(
            n_units=("unit_index", "nunique"),
            n_uncorrected_p05=("significant_uncorrected_p05", "sum"),
            n_run_factor_fdr05=("significant_run_factor_fdr05", "sum"),
            median_partial_eta2=("partial_eta2", "median"),
            mean_partial_eta2=("partial_eta2", "mean"),
            median_full_model_r2=("full_model_r2", "median"),
        )
    )

    run_summary.to_csv(out_dir / "noun_role_multifactor_decoding_run_summary.csv", index=False)
    predictions.to_csv(out_dir / "noun_role_multifactor_noun_predictions.csv", index=False)
    noun_summary.to_csv(out_dir / "noun_role_multifactor_decoding_by_noun.csv", index=False)
    factor_unit.to_csv(out_dir / "noun_role_multifactor_encoding_factor_unit_tests.csv", index=False)
    factor_summary.to_csv(out_dir / "noun_role_multifactor_encoding_factor_summary.csv", index=False)

    noun_fig = plot_noun_summary(noun_summary, out_dir, font_prop)
    multi_fig = plot_multifactor_summary(run_summary, factor_summary, out_dir)

    overall = []
    for metric_name, group in run_summary.groupby("metric"):
        for target in ("noun", "role", "voice"):
            correct = int(group[f"{target}_correct"].sum())
            tests = int(group[f"{target}_tests"].sum())
            chance = CHANCE_NOUN if target == "noun" else 0.5
            p_value = (
                stats.binom.sf(correct - 1, tests, chance)
                if target == "noun"
                else stats.binomtest(correct, tests, chance).pvalue
            )
            overall.append(
                {
                    "metric": metric_name,
                    "target": target,
                    "n_correct": correct,
                    "n_tests": tests,
                    "accuracy": correct / tests if tests else np.nan,
                    "chance": chance,
                    "binom_p": p_value,
                }
            )
    overall_summary = pd.DataFrame(overall)
    overall_summary.to_csv(out_dir / "noun_role_multifactor_decoding_overall_summary.csv", index=False)

    print("overall")
    print(overall_summary.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print("\npatient summaries")
    print(run_summary.groupby(["patient", "metric"], as_index=False).agg(
        noun_correct=("noun_correct", "sum"),
        noun_tests=("noun_tests", "sum"),
        role_correct=("role_correct", "sum"),
        role_tests=("role_tests", "sum"),
        voice_correct=("voice_correct", "sum"),
        voice_tests=("voice_tests", "sum"),
    ).to_string(index=False))
    print("\noutputs")
    for path in [
        out_dir / "noun_role_multifactor_decoding_overall_summary.csv",
        out_dir / "noun_role_multifactor_decoding_run_summary.csv",
        out_dir / "noun_role_multifactor_decoding_by_noun.csv",
        out_dir / "noun_role_multifactor_encoding_factor_summary.csv",
        Path(noun_fig["png"]),
        Path(noun_fig["pdf"]),
        Path(multi_fig["png"]),
        Path(multi_fig["pdf"]),
    ]:
        print(path)


if __name__ == "__main__":
    main()
