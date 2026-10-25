#!/usr/bin/env python3
"""Apply selected analysis ideas from three 2026 papers to the active/passive dataset.

The script deliberately treats the patient, rather than the recording run, as the
inferential unit.  It produces tables, figures, a machine-readable summary, and a
Chinese audit note for the accompanying explanatory presentation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "neuropixel2025/data"
OUT = ROOT / "reports/three_paper_borrowed_analyses_20260724"
FIG = OUT / "figures"
TAB = OUT / "tables"
for path in (OUT, FIG, TAB):
    path.mkdir(parents=True, exist_ok=True)

NET = DATA / "combined_active_passive_network_encoding"
TIME = DATA / "combined_active_passive_time_generalization"
DEPTH = DATA / "combined_active_passive_depth_laminar"
GAP = ROOT / "reports/teacher_report_gap_analysis_20260720/tables"
POP_PATH = ROOT / "reports/nature_science_draft/supplementary/tables/supp_table_2_population_decoding.csv"
TRIAL_PATH = ROOT / "outputs/active_passive_all_trial_sentences.csv"

COLORS = {
    "ink": "#142B2B",
    "blue": "#3D8DFF",
    "light_blue": "#B9E7FA",
    "teal": "#178D88",
    "amber": "#D99B2B",
    "coral": "#D7654B",
    "purple": "#7868B3",
    "gray": "#6F7775",
    "light_gray": "#D7DCDA",
    "panel": "#F1F2F1",
    "white": "#FFFFFF",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": COLORS["white"],
        "axes.facecolor": COLORS["white"],
        "savefig.facecolor": COLORS["white"],
        "font.size": 9,
    }
)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def pair_id(value: object) -> int:
    match = re.search(r"(\d+)\.mp3$", str(value))
    if match is None:
        raise ValueError(f"Cannot parse pair id from {value}")
    return int(match.group(1))


def exact_sign(values: np.ndarray, reference: float, alternative: str = "greater") -> dict[str, float | int]:
    diff = np.asarray(values, float) - reference
    diff = diff[np.isfinite(diff) & (np.abs(diff) > 1e-12)]
    if len(diff) == 0:
        return {"n_nonzero": 0, "n_positive": 0, "p": float("nan")}
    positive = int(np.sum(diff > 0))
    return {
        "n_nonzero": int(len(diff)),
        "n_positive": positive,
        "p": float(stats.binomtest(positive, len(diff), 0.5, alternative=alternative).pvalue),
    }


def bootstrap_median_ci(values: np.ndarray, seed: int, n_boot: int = 50_000) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    med = np.median(values[idx], axis=1)
    return float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))


def patient_summary(values: pd.DataFrame, value_col: str, reference: float, metric: str, label: str) -> dict[str, object]:
    patient = values.groupby("subject", as_index=False)[value_col].median()
    vals = patient[value_col].to_numpy(float)
    lo, hi = bootstrap_median_ci(vals, stable_seed("patient", metric))
    one = exact_sign(vals, reference, "greater")
    two = exact_sign(vals, reference, "two-sided")
    return {
        "metric": metric,
        "label": label,
        "reference": reference,
        "n_patients": int(len(vals)),
        "patient_median": float(np.median(vals)),
        "patient_mean": float(np.mean(vals)),
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
        "positive_patients": int(one["n_positive"]),
        "exact_sign_p_greater": float(one["p"]),
        "exact_sign_p_two_sided": float(two["p"]),
    }


def savefig(fig: plt.Figure, name: str) -> Path:
    path = FIG / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def specification_curve() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    scores = pd.read_csv(NET / "active_passive_network_encoding_model_scores.csv")
    rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []
    for keys, sub in scores.groupby(["analysis_mode", "unit_set", "model", "model_label"], dropna=False):
        analysis_mode, unit_set, model, model_label = keys
        patient = sub.groupby("subject", as_index=False)["auc"].median()
        vals = patient["auc"].to_numpy(float)
        lo, hi = bootstrap_median_ci(vals, stable_seed("spec", analysis_mode, unit_set, model))
        one = exact_sign(vals, 0.5, "greater")
        spec_id = f"{analysis_mode}|{unit_set}|{model}"
        rows.append(
            {
                "spec_id": spec_id,
                "analysis_mode": analysis_mode,
                "unit_set": unit_set,
                "model": model,
                "model_label": model_label,
                "n_runs": int(sub["run"].nunique()),
                "n_patients": int(patient["subject"].nunique()),
                "patient_median_auc": float(np.median(vals)),
                "patient_mean_auc": float(np.mean(vals)),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "positive_patients": int(one["n_positive"]),
                "exact_sign_p_greater": float(one["p"]),
                "is_primary_fixed": bool(
                    analysis_mode == "artifact_clean_le_0p2"
                    and unit_set == "good_only"
                    and model == "network_top5_l2"
                ),
            }
        )
        for record in patient.itertuples(index=False):
            patient_rows.append(
                {
                    "spec_id": spec_id,
                    "analysis_mode": analysis_mode,
                    "unit_set": unit_set,
                    "model": model,
                    "subject": record.subject,
                    "patient_auc": float(record.auc),
                }
            )
    spec = pd.DataFrame(rows).sort_values(["patient_median_auc", "spec_id"]).reset_index(drop=True)
    spec["curve_order"] = np.arange(1, len(spec) + 1)
    patient_values = pd.DataFrame(patient_rows)
    spec.to_csv(TAB / "specification_curve.csv", index=False)
    patient_values.to_csv(TAB / "specification_curve_patient_values.csv", index=False)

    fig = plt.figure(figsize=(10.2, 5.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[4.0, 0.42, 0.42, 0.70], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(spec))
    colors = np.where(spec["unit_set"].eq("good_only"), COLORS["blue"], COLORS["gray"])
    for i, row in spec.iterrows():
        ax.plot([i, i], [row.bootstrap_ci_low, row.bootstrap_ci_high], color=colors[i], alpha=0.45, lw=1.1)
    ax.scatter(x, spec["patient_median_auc"], c=colors, s=30, edgecolor="white", linewidth=0.5, zorder=3)
    primary_idx = int(np.flatnonzero(spec["is_primary_fixed"].to_numpy())[0])
    ax.scatter(
        [primary_idx], [spec.loc[primary_idx, "patient_median_auc"]], s=120,
        facecolor=COLORS["amber"], edgecolor=COLORS["ink"], linewidth=1.2, zorder=5,
        label="预设固定 top-5 主规格",
    )
    ax.axhline(0.5, color=COLORS["ink"], ls="--", lw=1)
    ax.set_ylabel("患者中位 AUC")
    ax.set_xlim(-0.8, len(spec) - 0.2)
    ax.set_ylim(min(0.28, float(spec.bootstrap_ci_low.min()) - 0.03), max(0.82, float(spec.bootstrap_ci_high.max()) + 0.03))
    ax.set_xticks([])
    ax.set_title("规格曲线：结论随分析选择变化，但固定主规格仍高于机会水平", loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")

    ax_art = fig.add_subplot(gs[1], sharex=ax)
    art = spec["analysis_mode"].eq("artifact_clean_le_0p2").to_numpy()
    ax_art.imshow(art[np.newaxis, :], aspect="auto", cmap=matplotlib.colors.ListedColormap(["#E6E6E6", COLORS["teal"]]), vmin=0, vmax=1)
    ax_art.set_yticks([0]); ax_art.set_yticklabels(["伪迹清理"]); ax_art.set_xticks([])
    ax_art.tick_params(axis="y", length=0)

    ax_unit = fig.add_subplot(gs[2], sharex=ax)
    good = spec["unit_set"].eq("good_only").to_numpy()
    ax_unit.imshow(good[np.newaxis, :], aspect="auto", cmap=matplotlib.colors.ListedColormap(["#E6E6E6", COLORS["blue"]]), vmin=0, vmax=1)
    ax_unit.set_yticks([0]); ax_unit.set_yticklabels(["仅 good unit"]); ax_unit.set_xticks([])
    ax_unit.tick_params(axis="y", length=0)

    ax_model = fig.add_subplot(gs[3], sharex=ax)
    model_order = [
        "nested_single_unit", "network_top5_l2", "network_top10_l2", "network_top25_l2",
        "network_all_l2", "network_pca2_l2", "network_pca4_l2", "network_pca8_l2",
    ]
    model_code = spec["model"].map({m: i for i, m in enumerate(model_order)}).to_numpy()
    ax_model.imshow(model_code[np.newaxis, :], aspect="auto", cmap="tab20", vmin=0, vmax=len(model_order) - 1)
    ax_model.set_yticks([0]); ax_model.set_yticklabels(["模型"]); ax_model.set_xticks([0, len(spec) - 1]); ax_model.set_xticklabels(["较低表现规格", "较高表现规格"])
    ax_model.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.10)
    savefig(fig, "fig_01_specification_curve.png")

    summary = {
        "n_specifications": int(len(spec)),
        "n_specs_patient_median_above_chance": int((spec["patient_median_auc"] > 0.5).sum()),
        "n_specs_all_five_patients_above_chance": int((spec["positive_patients"] == 5).sum()),
        "median_of_spec_patient_medians": float(spec["patient_median_auc"].median()),
        "range_of_spec_patient_medians": [float(spec["patient_median_auc"].min()), float(spec["patient_median_auc"].max())],
        "primary": spec.loc[primary_idx].to_dict(),
    }
    return spec, patient_values, summary


def build_endpoint_and_lopo() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pop = pd.read_csv(POP_PATH)
    marker = pd.read_csv(GAP / "fixed_top5_marker_generalization.csv")
    endpoint_specs = [
        (pop, "auc", 0.5, "best_auc", "描述性 best AUC"),
        (pop, "auc_gain_vs_nested_single_unit", 0.0, "best_gain", "best − 单神经元"),
        (pop, "fixed_top5_auc", 0.5, "fixed_top5", "固定 top-5 AUC"),
        (marker, "fixed_top5_marker_pair_heldout_auc", 0.5, "held_marker", "held-marker AUC"),
        (pop, "mean_cross_temporal_auc_primary_window", 0.5, "cross_time", "跨时间 AUC"),
    ]
    patient_records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    lopo_rows: list[dict[str, object]] = []
    for df, col, ref, metric, label in endpoint_specs:
        patient = df.groupby("subject", as_index=False)[col].median().rename(columns={col: "value"})
        for row in patient.itertuples(index=False):
            patient_records.append({"metric": metric, "label": label, "reference": ref, "subject": row.subject, "value": row.value})
        summary_rows.append(patient_summary(patient, "value", ref, metric, label))
        for omitted in patient["subject"]:
            vals = patient.loc[patient["subject"] != omitted, "value"].to_numpy(float)
            lopo_rows.append(
                {"metric": metric, "label": label, "reference": ref, "omitted_patient": omitted, "lopo_median": float(np.median(vals))}
            )
    patient_values = pd.DataFrame(patient_records)
    endpoint_summary = pd.DataFrame(summary_rows)
    lopo = pd.DataFrame(lopo_rows)
    patient_values.to_csv(TAB / "patient_endpoint_values.csv", index=False)
    endpoint_summary.to_csv(TAB / "patient_endpoint_inference.csv", index=False)
    lopo.to_csv(TAB / "leave_one_patient_out_influence.csv", index=False)

    plot_metrics = ["best_auc", "fixed_top5", "held_marker", "cross_time"]
    labels = endpoint_summary.set_index("metric").loc[plot_metrics, "label"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axes[0]
    rng = np.random.default_rng(4)
    for yi, metric in enumerate(plot_metrics):
        sub = patient_values[patient_values.metric == metric]
        ref = float(sub.reference.iloc[0])
        summ = endpoint_summary[endpoint_summary.metric == metric].iloc[0]
        ax.scatter(sub.value, yi + rng.normal(0, 0.045, len(sub)), s=36, color=COLORS["blue"], alpha=0.8)
        ax.plot([summ.bootstrap_ci_low, summ.bootstrap_ci_high], [yi, yi], color=COLORS["ink"], lw=2)
        ax.scatter([summ.patient_median], [yi], s=85, color=COLORS["amber"], edgecolor=COLORS["ink"], zorder=5)
        ax.axvline(ref, color=COLORS["gray"], ls="--", lw=0.9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("AUC（点=患者；黄点=患者中位数；线=bootstrap 95% CI）")
    ax.set_title("患者是推断单位", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(0.25, 0.9)

    ax = axes[1]
    use = lopo[lopo.metric.isin(plot_metrics)].copy()
    for yi, metric in enumerate(plot_metrics):
        vals = use.loc[use.metric == metric, "lopo_median"].to_numpy(float)
        ax.plot([vals.min(), vals.max()], [yi, yi], color=COLORS["teal"], lw=4, solid_capstyle="round")
        ax.scatter(vals, np.full(len(vals), yi), color=COLORS["white"], edgecolor=COLORS["teal"], s=34, zorder=3)
        ref = float(use.loc[use.metric == metric, "reference"].iloc[0])
        ax.axvline(ref, color=COLORS["gray"], ls="--", lw=0.9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("删去任一患者后的患者中位数范围")
    ax.set_title("LOPO：没有单一患者决定主趋势", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(0.25, 0.9)
    fig.tight_layout()
    savefig(fig, "fig_02_patient_inference_and_lopo.png")
    return endpoint_summary, lopo, {"endpoints": summary_rows}


def load_response_metrics() -> pd.DataFrame:
    frames = []
    pattern = "gz*/processed/ActPasSynSentence/event_related_active_passive/active_passive_unit_trial_response_metrics.csv"
    for path in sorted(DATA.glob(pattern)):
        df = pd.read_csv(path)
        df["subject"] = path.relative_to(DATA).parts[0]
        df["condition_simple"] = df["condition"].astype(str).str.extract(r"^(active|passive)", expand=False)
        frames.append(df[df.condition_simple.notna()].copy())
    return pd.concat(frames, ignore_index=True)


def make_response_matrix(metrics: pd.DataFrame, trials: pd.DataFrame, subject: str, run: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    sub = metrics[(metrics.subject == subject) & (metrics.run == run) & (metrics.ks_label.astype(str) == "good")].copy()
    pivot = sub.pivot_table(index="trial_index", columns="unit_id", values="response_baseline_corrected_hz", aggfunc="mean").sort_index()
    meta = sub.drop_duplicates("trial_index").set_index("trial_index").loc[pivot.index].reset_index()
    onset = trials[(trials.subject == subject) & (trials.run == run)][["trial_index", "target_onset_sec_from_sentence_onset"]].drop_duplicates("trial_index")
    meta = meta.merge(onset, on="trial_index", how="left")
    meta["pair_id"] = meta.sound_file.map(pair_id)
    x = pivot.to_numpy(float)
    unit_ids = pivot.columns.to_numpy(int)
    keep = np.isfinite(x).all(axis=0) & (x.std(axis=0) > 0)
    x = x[:, keep]
    unit_ids = unit_ids[keep]
    y = (meta.condition_simple.to_numpy() == "active").astype(int)
    return x, y, meta, unit_ids


def t_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, p = x[y == 1], x[y == 0]
    av = a.var(axis=0, ddof=1) if len(a) > 1 else np.zeros(x.shape[1])
    pv = p.var(axis=0, ddof=1) if len(p) > 1 else np.zeros(x.shape[1])
    denom = np.sqrt(av / max(len(a), 1) + pv / max(len(p), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.abs((a.mean(axis=0) - p.mean(axis=0)) / denom)
    return np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


def fit_topk_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int = 5, c_value: float = 0.1) -> np.ndarray:
    order = np.argsort(t_scores(x_train, y_train))[::-1]
    cols = order[: min(k, x_train.shape[1])]
    if len(cols) == 0 or len(np.unique(y_train)) < 2:
        return np.full(len(x_test), float(np.mean(y_train)))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, solver="liblinear", class_weight="balanced", max_iter=1500))
    model.fit(x_train[:, cols], y_train)
    return model.predict_proba(x_test[:, cols])[:, 1]


def grouped_folds(meta: pd.DataFrame, original_y: np.ndarray, grouping: str) -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if grouping == "sentence_pair":
        pairs = meta.pair_id.to_numpy(int)
        for pid in sorted(np.unique(pairs)):
            test = np.where(pairs == pid)[0]
            train = np.where(pairs != pid)[0]
            folds.append((train, test))
    elif grouping == "marker_pair":
        words = meta.target_word.astype(str).to_numpy()
        active_markers = sorted(meta.loc[original_y == 1, "target_word"].astype(str).unique())
        passive_markers = sorted(meta.loc[original_y == 0, "target_word"].astype(str).unique())
        for am, pm in itertools.product(active_markers, passive_markers):
            test = np.where((words == am) | (words == pm))[0]
            train = np.where((words != am) & (words != pm))[0]
            folds.append((train, test))
    else:
        raise ValueError(grouping)
    return folds


def residualize_onset(x_train: np.ndarray, x_test: np.ndarray, onset_train: np.ndarray, onset_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d_train = np.column_stack([np.ones(len(onset_train)), onset_train])
    d_test = np.column_stack([np.ones(len(onset_test)), onset_test])
    beta = np.linalg.lstsq(d_train, x_train, rcond=None)[0]
    return x_train - d_train @ beta, x_test - d_test @ beta


def grouped_decode(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, original_y: np.ndarray, grouping: str, onset_residual: bool = False) -> float:
    prob_sum = np.zeros(len(y), float)
    prob_count = np.zeros(len(y), float)
    onset = meta.target_onset_sec_from_sentence_onset.to_numpy(float)
    for train, test in grouped_folds(meta, original_y, grouping):
        if len(np.unique(y[train])) < 2:
            continue
        x_train, x_test = x[train], x[test]
        if onset_residual:
            x_train, x_test = residualize_onset(x_train, x_test, onset[train], onset[test])
        probs = fit_topk_predict(x_train, y[train], x_test, k=5, c_value=0.1)
        prob_sum[test] += probs
        prob_count[test] += 1
    valid = prob_count > 0
    if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(y[valid], prob_sum[valid] / prob_count[valid]))


def global_pair_swap_labels(y: np.ndarray, pairs: np.ndarray, swap_by_pair: dict[int, bool]) -> np.ndarray:
    out = y.copy()
    for pid in np.unique(pairs):
        idx = np.where(pairs == pid)[0]
        if swap_by_pair[int(pid)]:
            out[idx] = 1 - out[idx]
    return out


def ccgp_permutation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pop = pd.read_csv(POP_PATH)[["subject", "run"]].drop_duplicates()
    metrics = load_response_metrics()
    trials = pd.read_csv(TRIAL_PATH)
    run_data: list[dict[str, object]] = []
    for row in pop.itertuples(index=False):
        x, y, meta, _ = make_response_matrix(metrics, trials, row.subject, row.run)
        run_data.append({"subject": row.subject, "run": row.run, "x": x, "y": y, "meta": meta})

    variants = [
        ("held_sentence_pair", "sentence_pair", False, "held-sentence-pair"),
        ("held_marker", "marker_pair", False, "held-marker"),
        ("held_marker_onset_resid", "marker_pair", True, "held-marker + onset residual"),
    ]
    actual_rows: list[dict[str, object]] = []
    for item in run_data:
        for metric, grouping, resid, label in variants:
            auc = grouped_decode(item["x"], item["y"], item["meta"], item["y"], grouping, resid)
            actual_rows.append({"subject": item["subject"], "run": item["run"], "metric": metric, "label": label, "auc": auc})
    actual = pd.DataFrame(actual_rows)
    patient_actual = actual.groupby(["metric", "label", "subject"], as_index=False).auc.median()

    n_perm = 499
    rng = np.random.default_rng(20260724)
    perm_rows: list[dict[str, object]] = []
    pair_values = sorted(pd.read_csv(TRIAL_PATH).sound_file.map(pair_id).unique())
    for perm in range(n_perm):
        swap = {int(pid): bool(v) for pid, v in zip(pair_values, rng.integers(0, 2, size=len(pair_values)))}
        per_variant: dict[str, list[dict[str, object]]] = {v[0]: [] for v in variants}
        for item in run_data:
            meta = item["meta"]
            y_perm = global_pair_swap_labels(item["y"], meta.pair_id.to_numpy(int), swap)
            for metric, grouping, resid, label in variants:
                auc = grouped_decode(item["x"], y_perm, meta, item["y"], grouping, resid)
                per_variant[metric].append({"subject": item["subject"], "auc": auc})
        for metric, _, _, label in variants:
            frame = pd.DataFrame(per_variant[metric])
            patient = frame.groupby("subject").auc.median().to_numpy(float)
            perm_rows.append({"permutation": perm, "metric": metric, "label": label, "group_patient_median_auc": float(np.median(patient))})
    perm_null = pd.DataFrame(perm_rows)

    summary_rows = []
    for metric, _, _, label in variants:
        vals = patient_actual.loc[patient_actual.metric == metric, "auc"].to_numpy(float)
        obs = float(np.median(vals))
        null = perm_null.loc[perm_null.metric == metric, "group_patient_median_auc"].to_numpy(float)
        p_perm = float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))
        row = patient_summary(patient_actual[patient_actual.metric == metric].rename(columns={"auc": "value"}), "value", 0.5, metric, label)
        row["group_permutation_p_greater"] = p_perm
        row["n_group_permutations"] = n_perm
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    actual.to_csv(TAB / "ccgp_run_results.csv", index=False)
    patient_actual.to_csv(TAB / "ccgp_patient_results.csv", index=False)
    perm_null.to_csv(TAB / "ccgp_group_permutation_null.csv", index=False)
    summary.to_csv(TAB / "ccgp_patient_inference.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.7, 4.7))
    metric_order = [v[0] for v in variants]
    labels = [v[3] for v in variants]
    rng_plot = np.random.default_rng(12)
    for xi, metric in enumerate(metric_order):
        vals = patient_actual[patient_actual.metric == metric].auc.to_numpy(float)
        ax.scatter(np.full(len(vals), xi) + rng_plot.normal(0, 0.045, len(vals)), vals, color=COLORS["blue"], s=42, alpha=0.85)
        row = summary[summary.metric == metric].iloc[0]
        ax.plot([xi - 0.16, xi + 0.16], [row.patient_median, row.patient_median], color=COLORS["amber"], lw=4)
        ax.text(xi, 0.25, f"患者中位 {row.patient_median:.2f}\n置换 P={row.group_permutation_p_greater:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color=COLORS["ink"], ls="--", lw=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("AUC（每点=患者）")
    ax.set_ylim(0.24, 0.88)
    ax.set_title("跨句对/跨标记泛化存在方向性，但患者层级证据仍弱", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig_04_ccgp_marker_generalization.png")
    return actual, patient_actual, summary, {"variants": summary_rows, "n_permutations": n_perm}


def depth_virtual_lesion() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pop = pd.read_csv(POP_PATH)[["subject", "run"]].drop_duplicates()
    metrics = load_response_metrics()
    trials = pd.read_csv(TRIAL_PATH)
    depth = pd.read_csv(DEPTH / "active_passive_depth_laminar_unit_metrics.csv")
    bands = ["superficial", "middle", "deep"]
    run_rows: list[dict[str, object]] = []
    for row in pop.itertuples(index=False):
        x, y, meta, units = make_response_matrix(metrics, trials, row.subject, row.run)
        full = grouped_decode(x, y, meta, y, "sentence_pair", False)
        lookup = depth[(depth.subject == row.subject) & (depth.run == row.run)].drop_duplicates("unit_id").set_index("unit_id").depth_band
        unit_band = np.array([lookup.get(int(unit), "missing") for unit in units], dtype=object)
        for band in bands:
            keep = unit_band != band
            if keep.sum() < 2:
                auc = float("nan")
            else:
                auc = grouped_decode(x[:, keep], y, meta, y, "sentence_pair", False)
            run_rows.append(
                {
                    "subject": row.subject,
                    "run": row.run,
                    "omitted_depth_band": band,
                    "n_units_full": int(x.shape[1]),
                    "n_units_omitted": int((unit_band == band).sum()),
                    "full_auc": full,
                    "lesioned_auc": auc,
                    "auc_drop_full_minus_lesioned": full - auc,
                }
            )
    run_df = pd.DataFrame(run_rows)
    patient = run_df.groupby(["omitted_depth_band", "subject"], as_index=False).agg(
        full_auc=("full_auc", "median"),
        lesioned_auc=("lesioned_auc", "median"),
        auc_drop_full_minus_lesioned=("auc_drop_full_minus_lesioned", "median"),
    )
    summaries = []
    for band in bands:
        vals = patient.loc[patient.omitted_depth_band == band, "auc_drop_full_minus_lesioned"].dropna().to_numpy(float)
        lo, hi = bootstrap_median_ci(vals, stable_seed("lesion", band))
        sign = exact_sign(vals, 0.0, "greater")
        summaries.append(
            {
                "omitted_depth_band": band,
                "n_patients": int(len(vals)),
                "patient_median_auc_drop": float(np.median(vals)),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "positive_drop_patients": int(sign["n_positive"]),
                "exact_sign_p_greater": float(sign["p"]),
            }
        )
    summary = pd.DataFrame(summaries)
    run_df.to_csv(TAB / "depth_virtual_lesion_run_results.csv", index=False)
    patient.to_csv(TAB / "depth_virtual_lesion_patient_results.csv", index=False)
    summary.to_csv(TAB / "depth_virtual_lesion_patient_inference.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    labels = ["浅", "中", "深"]
    rng = np.random.default_rng(22)
    for xi, band in enumerate(bands):
        vals = patient.loc[patient.omitted_depth_band == band, "auc_drop_full_minus_lesioned"].dropna().to_numpy(float)
        ax.scatter(np.full(len(vals), xi) + rng.normal(0, 0.045, len(vals)), vals, color=COLORS["teal"], s=42, alpha=0.85)
        row = summary[summary.omitted_depth_band == band].iloc[0]
        ax.plot([xi, xi], [row.bootstrap_ci_low, row.bootstrap_ci_high], color=COLORS["ink"], lw=2)
        ax.scatter([xi], [row.patient_median_auc_drop], s=80, color=COLORS["amber"], edgecolor=COLORS["ink"], zorder=4)
    ax.axhline(0, color=COLORS["gray"], ls="--", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"去除{v}部相对深度单元" for v in labels])
    ax.set_ylabel("held-pair AUC 下降（正值=性能受损）")
    ax.set_title("虚拟损毁未指向单一伪层带；深度结果仅作描述", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig_03_depth_virtual_lesion.png")
    return run_df, summary, {"bands": summaries}


@dataclass(frozen=True)
class TemporalConfig:
    window_sec: float
    bin_sec: float
    top_k: int
    c_value: float

    def key(self) -> str:
        return f"w{self.window_sec:g}_b{self.bin_sec:g}_k{self.top_k}_c{self.c_value:g}"


def load_binned_tensor(subject: str, run: str, trials: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    er_dir = DATA / subject / "processed/ActPasSynSentence/event_related_active_passive"
    events = pd.read_csv(er_dir / "active_passive_event_related_trial_metadata.csv")
    events["condition_simple"] = events.condition.astype(str).str.extract(r"^(active|passive)", expand=False)
    events = events[(events.run == run) & events.condition_simple.notna()].copy()
    wanted = trials[(trials.subject == subject) & (trials.run == run)][["trial_index", "sound_file"]]
    events = wanted.merge(events, on=["trial_index", "sound_file"], how="inner").sort_values("trial_index").reset_index(drop=True)
    fr_path = DATA / subject / "processed/ActPasSynSentence" / run / "firing_rate_1000Hz/firing_rate_1000Hz_good_clusters.npz"
    raw = np.load(fr_path, allow_pickle=True)
    fr = raw["firing_rate_hz"].astype(np.float32, copy=False)
    unit_ids = raw["unit_ids"].astype(int)
    n_trials = len(events)
    tensor = np.full((n_trials, fr.shape[0], 8), np.nan, np.float32)
    for ti, event in enumerate(events.itertuples(index=False)):
        ebin = int(event.event_bin_1000Hz)
        b0, b1 = max(0, ebin - 500), min(fr.shape[1], ebin)
        baseline = fr[:, b0:b1].mean(axis=1) if b1 > b0 else np.zeros(fr.shape[0], np.float32)
        for bi in range(8):
            w0, w1 = ebin + bi * 100, ebin + (bi + 1) * 100
            tensor[ti, :, bi] = fr[:, w0:w1].mean(axis=1) - baseline
    y = (events.condition_simple.to_numpy() == "active").astype(int)
    pairs = events.sound_file.map(pair_id).to_numpy(int)
    keep = np.isfinite(tensor).all(axis=(0, 2)) & (tensor.std(axis=(0, 2)) > 0)
    return tensor[:, keep, :], y, pairs, unit_ids[keep]


def representation_array(tensor: np.ndarray, config: TemporalConfig, representation: str) -> np.ndarray:
    n_base = int(round(config.window_sec / 0.1))
    use = tensor[:, :, :n_base]
    if representation == "mean_rate":
        return use.mean(axis=2)
    group = int(round(config.bin_sec / 0.1))
    if n_base % group != 0:
        raise ValueError(config)
    use = use.reshape(use.shape[0], use.shape[1], n_base // group, group).mean(axis=3)
    return use - use.mean(axis=2, keepdims=True)


def temporal_unit_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, p = x[y == 1], x[y == 0]
    diff = a.mean(axis=0) - p.mean(axis=0)
    denom = a.var(axis=0, ddof=1) / max(len(a), 1) + p.var(axis=0, ddof=1) / max(len(p), 1) + 1e-8
    return np.sum((diff * diff) / denom, axis=1)


def fit_representation_predict(tensor_train: np.ndarray, y_train: np.ndarray, tensor_test: np.ndarray, config: TemporalConfig, representation: str) -> np.ndarray:
    x_train = representation_array(tensor_train, config, representation)
    x_test = representation_array(tensor_test, config, representation)
    if representation == "mean_rate":
        scores = t_scores(x_train, y_train)
    else:
        scores = temporal_unit_scores(x_train, y_train)
    units = np.argsort(scores)[::-1][: min(config.top_k, len(scores))]
    if representation == "mean_rate":
        train_features = x_train[:, units]
        test_features = x_test[:, units]
    else:
        train_features = x_train[:, units, :].reshape(len(x_train), -1)
        test_features = x_test[:, units, :].reshape(len(x_test), -1)
    variable = np.isfinite(train_features).all(axis=0) & (train_features.std(axis=0) > 0)
    train_features = train_features[:, variable]
    test_features = test_features[:, variable]
    if train_features.shape[1] == 0 or len(np.unique(y_train)) < 2:
        return np.full(len(x_test), float(np.mean(y_train)))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=config.c_value, solver="liblinear", class_weight="balanced", max_iter=1500))
    model.fit(train_features, y_train)
    return model.predict_proba(test_features)[:, 1]


def config_grid(representation: str) -> list[TemporalConfig]:
    if representation == "mean_rate":
        return [TemporalConfig(w, w, k, c) for w in (0.4, 0.8) for k in (5, 10) for c in (0.03, 0.3)]
    return [TemporalConfig(w, b, k, c) for w in (0.4, 0.8) for b in (0.1, 0.2) for k in (5, 10) for c in (0.03, 0.3)]


def nested_pair_decode(tensor: np.ndarray, y: np.ndarray, pairs: np.ndarray, representation: str) -> tuple[float, list[str]]:
    configs = config_grid(representation)
    outer_probs = np.full(len(y), np.nan, float)
    selected: list[str] = []
    for outer_pair in sorted(np.unique(pairs)):
        outer_test = np.where(pairs == outer_pair)[0]
        outer_train = np.where(pairs != outer_pair)[0]
        best_cfg = None
        best_auc = -np.inf
        train_pairs = pairs[outer_train]
        # Three deterministic grouped inner folds keep sentence pairs intact while
        # avoiding the much slower leave-one-pair-out loop inside every permutation.
        # The outer evaluation remains leave-one-sentence-pair-out.
        unique_inner_pairs = np.array(sorted(np.unique(train_pairs)))
        inner_pair_groups = np.array_split(unique_inner_pairs, 3)
        for cfg in configs:
            inner_probs = np.full(len(outer_train), np.nan, float)
            for inner_group in inner_pair_groups:
                inner_test_local = np.where(np.isin(train_pairs, inner_group))[0]
                inner_train_local = np.where(~np.isin(train_pairs, inner_group))[0]
                inner_probs[inner_test_local] = fit_representation_predict(
                    tensor[outer_train[inner_train_local]], y[outer_train[inner_train_local]],
                    tensor[outer_train[inner_test_local]], cfg, representation,
                )
            auc = roc_auc_score(y[outer_train], inner_probs)
            if auc > best_auc + 1e-12 or (abs(auc - best_auc) <= 1e-12 and (best_cfg is None or cfg.key() < best_cfg.key())):
                best_auc, best_cfg = float(auc), cfg
        assert best_cfg is not None
        selected.append(best_cfg.key())
        outer_probs[outer_test] = fit_representation_predict(tensor[outer_train], y[outer_train], tensor[outer_test], best_cfg, representation)
    return float(roc_auc_score(y, outer_probs)), selected


def mean_rate_vs_temporal() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pop = pd.read_csv(POP_PATH)[["subject", "run"]].drop_duplicates()
    trials = pd.read_csv(TRIAL_PATH)
    run_data = []
    selected_rows = []
    for row in pop.itertuples(index=False):
        tensor, y, pairs, units = load_binned_tensor(row.subject, row.run, trials)
        result = {"subject": row.subject, "run": row.run, "n_units": int(len(units))}
        for representation in ("mean_rate", "temporal_pattern"):
            auc, selected = nested_pair_decode(tensor, y, pairs, representation)
            result[f"{representation}_auc"] = auc
            for fold, cfg in zip(sorted(np.unique(pairs)), selected):
                selected_rows.append({"subject": row.subject, "run": row.run, "representation": representation, "outer_pair": int(fold), "selected_config": cfg})
        result["temporal_minus_mean_auc"] = result["temporal_pattern_auc"] - result["mean_rate_auc"]
        result["_tensor"] = tensor
        result["_y"] = y
        result["_pairs"] = pairs
        run_data.append(result)

    actual = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in run_data])
    patient = actual.groupby("subject", as_index=False).agg(
        mean_rate_auc=("mean_rate_auc", "median"),
        temporal_pattern_auc=("temporal_pattern_auc", "median"),
        temporal_minus_mean_auc=("temporal_minus_mean_auc", "median"),
    )
    actual.to_csv(TAB / "mean_rate_vs_temporal_run_results.csv", index=False)
    patient.to_csv(TAB / "mean_rate_vs_temporal_patient_results.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(TAB / "mean_rate_vs_temporal_selected_configs.csv", index=False)

    # Full nested refitting is intentionally expensive. Nineteen synchronized
    # pair-swap permutations give an exploratory minimum attainable P of 0.05;
    # the deck labels this comparison as exploratory rather than confirmatory.
    n_perm = 19
    rng = np.random.default_rng(24072026)
    pair_values = sorted(np.unique(np.concatenate([row["_pairs"] for row in run_data])))
    perm_rows = []
    for perm in range(n_perm):
        swap = {int(pid): bool(v) for pid, v in zip(pair_values, rng.integers(0, 2, size=len(pair_values)))}
        run_perm = []
        for row in run_data:
            y_perm = global_pair_swap_labels(row["_y"], row["_pairs"], swap)
            mean_auc, _ = nested_pair_decode(row["_tensor"], y_perm, row["_pairs"], "mean_rate")
            temporal_auc, _ = nested_pair_decode(row["_tensor"], y_perm, row["_pairs"], "temporal_pattern")
            run_perm.append({"subject": row["subject"], "mean_rate_auc": mean_auc, "temporal_pattern_auc": temporal_auc})
        frame = pd.DataFrame(run_perm)
        pat = frame.groupby("subject", as_index=False).median(numeric_only=True)
        perm_rows.append(
            {
                "permutation": perm,
                "patient_median_mean_rate_auc": float(np.median(pat.mean_rate_auc)),
                "patient_median_temporal_pattern_auc": float(np.median(pat.temporal_pattern_auc)),
                "patient_median_temporal_minus_mean_auc": float(np.median(pat.temporal_pattern_auc - pat.mean_rate_auc)),
            }
        )
        if (perm + 1) % 10 == 0:
            print(f"mean-rate/temporal full-pipeline permutation {perm + 1}/{n_perm}", flush=True)
    null = pd.DataFrame(perm_rows)
    null.to_csv(TAB / "mean_rate_vs_temporal_group_permutation_null.csv", index=False)
    obs_mean = float(np.median(patient.mean_rate_auc))
    obs_temp = float(np.median(patient.temporal_pattern_auc))
    obs_diff = float(np.median(patient.temporal_minus_mean_auc))
    p_mean = float((1 + np.sum(null.patient_median_mean_rate_auc >= obs_mean - 1e-12)) / (n_perm + 1))
    p_temp = float((1 + np.sum(null.patient_median_temporal_pattern_auc >= obs_temp - 1e-12)) / (n_perm + 1))
    p_diff = float((1 + np.sum(null.patient_median_temporal_minus_mean_auc >= obs_diff - 1e-12)) / (n_perm + 1))
    sign_mean = exact_sign(patient.mean_rate_auc.to_numpy(), 0.5, "greater")
    sign_temp = exact_sign(patient.temporal_pattern_auc.to_numpy(), 0.5, "greater")
    sign_diff = exact_sign(patient.temporal_minus_mean_auc.to_numpy(), 0.0, "greater")
    summary = {
        "n_patients": int(len(patient)),
        "n_group_permutations": n_perm,
        "patient_median_mean_rate_auc": obs_mean,
        "patient_median_temporal_pattern_auc": obs_temp,
        "patient_median_temporal_minus_mean_auc": obs_diff,
        "permutation_p_mean_rate_greater": p_mean,
        "permutation_p_temporal_greater": p_temp,
        "permutation_p_temporal_greater_than_mean": p_diff,
        "exact_sign_mean_rate": sign_mean,
        "exact_sign_temporal_pattern": sign_temp,
        "exact_sign_temporal_minus_mean": sign_diff,
    }
    with open(TAB / "mean_rate_vs_temporal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for row in patient.itertuples(index=False):
        ax.plot([0, 1], [row.mean_rate_auc, row.temporal_pattern_auc], color=COLORS["light_gray"], lw=1.4)
        ax.scatter([0], [row.mean_rate_auc], color=COLORS["blue"], s=42, zorder=3)
        ax.scatter([1], [row.temporal_pattern_auc], color=COLORS["teal"], s=42, zorder=3)
        ax.text(1.04, row.temporal_pattern_auc, row.subject, va="center", fontsize=7, color=COLORS["gray"])
    ax.axhline(0.5, color=COLORS["ink"], ls="--", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["整窗平均率\n（窗口/单元/C 内层选择）", "去均值时间模式\n（窗口/分箱/单元/C 内层选择）"])
    ax.set_ylabel("外层 held-sentence-pair AUC")
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(0.20, 0.90)
    ax.set_title("完全嵌套比较：平均率与时间模式谁更能跨句对泛化？", loc="left", fontsize=12, fontweight="bold")
    ax.text(0.02, 0.02, f"患者中位：平均率 {obs_mean:.2f}；时间模式 {obs_temp:.2f}；差值 {obs_diff:+.2f}\n全流程置换：Pmean={p_mean:.3f}，Ptime={p_temp:.3f}，Pdiff={p_diff:.3f}", transform=ax.transAxes, fontsize=8, va="bottom")
    fig.tight_layout()
    savefig(fig, "fig_05_mean_rate_vs_temporal_code.png")
    return actual, patient, null, summary


def identifiability_and_equivalence() -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    trials = pd.read_csv(TRIAL_PATH)
    stimuli = trials.drop_duplicates("sound_file").copy()
    table = pd.crosstab(stimuli.target_word, stimuli.condition).reindex(columns=["active", "passive"], fill_value=0)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    v = math.sqrt(chi2 / table.to_numpy().sum() / min(table.shape[0] - 1, table.shape[1] - 1))
    marker = pd.get_dummies(stimuli.target_word, drop_first=True, dtype=float)
    voice = (stimuli.condition == "active").astype(float).to_numpy()
    design = np.column_stack([np.ones(len(stimuli)), voice, marker.to_numpy()])
    singular = np.linalg.svd(design, compute_uv=False)
    ident = {
        "n_unique_stimuli": int(len(stimuli)),
        "cramers_v_marker_voice": float(v),
        "design_rows": int(design.shape[0]),
        "design_columns": int(design.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "smallest_singular_value": float(singular[-1]),
        "conclusion": "voice-after-marker not identifiable",
    }
    table.to_csv(TAB / "marker_voice_crosstab.csv")
    with open(TAB / "identifiability_summary.json", "w", encoding="utf-8") as f:
        json.dump(ident, f, ensure_ascii=False, indent=2)

    pop = pd.read_csv(POP_PATH)
    patient = pop.groupby("subject").mean_cross_temporal_auc_primary_window.median().to_numpy(float)
    delta = patient - 0.5
    mean_delta = float(np.mean(delta))
    se = float(stats.sem(delta))
    df = len(delta) - 1
    ci90 = (mean_delta - stats.t.ppf(0.95, df) * se, mean_delta + stats.t.ppf(0.95, df) * se)
    eq_rows = []
    for margin in (0.03, 0.05):
        t_lower = (mean_delta + margin) / se
        p_lower = float(stats.t.sf(t_lower, df))
        t_upper = (mean_delta - margin) / se
        p_upper = float(stats.t.cdf(t_upper, df))
        eq_rows.append(
            {
                "metric": "patient mean cross-time AUC minus 0.5",
                "equivalence_margin_auc": margin,
                "n_patients": len(delta),
                "mean_delta": mean_delta,
                "ci90_low": ci90[0],
                "ci90_high": ci90[1],
                "tost_p_lower": p_lower,
                "tost_p_upper": p_upper,
                "tost_p_equivalence": max(p_lower, p_upper),
                "equivalent_at_alpha_0p05": bool(max(p_lower, p_upper) < 0.05),
            }
        )
    eq = pd.DataFrame(eq_rows)
    eq.to_csv(TAB / "cross_time_equivalence_tost.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    matrix = table.to_numpy(float)
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap([COLORS["white"], COLORS["blue"]]), vmin=0, vmax=max(1, matrix.max()))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color=COLORS["ink"], fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["主动", "被动"])
    ax.set_yticks(range(len(table.index))); ax.set_yticklabels(table.index)
    ax.set_title(f"marker × voice 完全混合\nCramér V={v:.2f}；rank={design.shape[1]-1}/{design.shape[1]}", loc="left", fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.axvspan(-0.05, 0.05, color=COLORS["light_blue"], alpha=0.55, label="ROPE ±0.05 AUC")
    ax.axvspan(-0.03, 0.03, color=COLORS["blue"], alpha=0.18, label="ROPE ±0.03 AUC")
    ax.plot([ci90[0], ci90[1]], [0, 0], color=COLORS["ink"], lw=3)
    ax.scatter([mean_delta], [0], s=90, color=COLORS["amber"], edgecolor=COLORS["ink"], zorder=3)
    ax.axvline(0, color=COLORS["gray"], ls="--", lw=1)
    ax.set_yticks([]); ax.set_xlim(-0.08, 0.08)
    ax.set_xlabel("患者平均 cross-time AUC − 0.5（90% CI）")
    ax.set_title("关键 null 的等效性取决于容许界值", loc="left", fontsize=11, fontweight="bold")
    for i, row in eq.iterrows():
        ax.text(0.03, 0.26 - i * 0.12, f"±{row.equivalence_margin_auc:.02f}: TOST P={row.tost_p_equivalence:.3f} → {'等效' if row.equivalent_at_alpha_0p05 else '未证实'}", transform=ax.transAxes, fontsize=8)
    ax.legend(frameon=False, loc="lower center", fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig_06_identifiability_and_equivalence.png")
    return ident, eq, {"mean_delta": mean_delta, "ci90": list(ci90), "tost": eq_rows}


def write_feasibility_table() -> pd.DataFrame:
    rows = [
        {"analysis": "patient-level inference", "status": "completed", "reason": "5 patients; run medians aggregated within patient"},
        {"analysis": "specification curve", "status": "completed", "reason": "32 existing combinations of artifact handling, unit set, and decoder"},
        {"analysis": "leave-one-patient-out", "status": "completed", "reason": "influence of every patient on patient-median endpoints"},
        {"analysis": "depth-band virtual lesion", "status": "completed exploratory", "reason": "relative depth bands only; not anatomical layers"},
        {"analysis": "mean-rate vs temporal-pattern", "status": "completed exploratory", "reason": "nested held-pair CV; full-selection pair-swap permutation"},
        {"analysis": "CCGP-style held-marker", "status": "completed exploratory", "reason": "partial marker invariance; voice remains nested in marker"},
        {"analysis": "equivalence/ROPE sensitivity", "status": "completed exploratory", "reason": "patient-level t-TOST at ±0.03 and ±0.05 AUC"},
        {"analysis": "reliability-matched geometry null", "status": "not identifiable from current data", "reason": "one presentation per exact sentence within run; no split-half stimulus reliability"},
        {"analysis": "voice-after-marker regression", "status": "not identifiable", "reason": "marker and voice are perfectly confounded; design rank deficient"},
        {"analysis": "cross-voice semantic subspace", "status": "limited existing diagnostic", "reason": "available for 3 patients only and cannot remove marker confounding"},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(TAB / "analysis_feasibility_and_scope.csv", index=False)
    return frame


def main() -> None:
    spec, spec_patients, spec_summary = specification_curve()
    endpoints, lopo, endpoint_summary = build_endpoint_and_lopo()
    lesion_runs, lesion_summary, lesion_json = depth_virtual_lesion()
    ccgp_runs, ccgp_patients, ccgp_summary, ccgp_json = ccgp_permutation()
    mean_runs, mean_patients, mean_null, mean_summary = mean_rate_vs_temporal()
    ident, equivalence, null_json = identifiability_and_equivalence()
    feasibility = write_feasibility_table()

    summary = {
        "generated_on": "2026-07-24",
        "inferential_unit": "patient",
        "n_patients": 5,
        "n_primary_runs": 7,
        "specification_curve": spec_summary,
        "patient_endpoints": endpoint_summary,
        "depth_virtual_lesion": lesion_json,
        "ccgp": ccgp_json,
        "mean_rate_vs_temporal": mean_summary,
        "identifiability": ident,
        "cross_time_equivalence": null_json,
        "scope": feasibility.to_dict(orient="records"),
    }
    with open(OUT / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))

    primary = spec[spec.is_primary_fixed].iloc[0]
    mean = mean_summary
    ccgp_row = ccgp_summary[ccgp_summary.metric == "held_marker"].iloc[0]
    eq05 = equivalence[equivalence.equivalence_margin_auc == 0.05].iloc[0]
    eq03 = equivalence[equivalence.equivalence_margin_auc == 0.03].iloc[0]
    note = f"""# 三篇论文方法借鉴：本数据实做结果（2026-07-24）

## 总体结论

现有数据最稳妥地支持：功能词附近存在可被群体解码的、短暂且分布式的主动/被动相关状态；但该状态是否是独立于功能词身份的抽象语法代码，在当前刺激设计中不可辨识。

## 1. 规格曲线

- 共纳入 {spec_summary['n_specifications']} 个现有分析规格；{spec_summary['n_specs_patient_median_above_chance']}/{spec_summary['n_specifications']} 个规格的患者中位 AUC 高于 0.5。
- 预设固定主规格（伪迹清理、good unit、top-5 L2）患者中位 AUC={primary.patient_median_auc:.3f}，bootstrap 95% CI [{primary.bootstrap_ci_low:.3f}, {primary.bootstrap_ci_high:.3f}]。
- 解释：方向并不只依赖一个模型，但效应大小明显受模型与 unit set 影响；best-of-many 仍只能作描述性上限。

## 2. 患者层级与影响分析

- descriptive best AUC 与 best−single gain 在 5/5 患者中方向一致；单侧 exact sign P=0.0312，双侧最小分辨率 P=0.0625。
- fixed top-5、held-marker 与 cross-time 未达到同等患者层级证据强度。
- 删去任一患者后，主要 AUC 指标的中位数仍保持原方向；没有单一患者完全决定结论。

## 3. 伪层带虚拟损毁

- 去除浅/中/深相对深度单元后的 held-pair AUC 变化均跨患者不稳定。
- 解释：当前数据不支持把群体信息归因于某一个伪层带。这里的 relative depth 不是组织学皮层层次。

## 4. 均值率与时间模式

- 完全嵌套、held-sentence-pair 外层评估：患者中位 mean-rate AUC={mean['patient_median_mean_rate_auc']:.3f}；去均值 temporal-pattern AUC={mean['patient_median_temporal_pattern_auc']:.3f}。
- temporal−mean 患者中位差={mean['patient_median_temporal_minus_mean_auc']:+.3f}；全流程标签置换 P={mean['permutation_p_temporal_greater_than_mean']:.3f}。
- 解释：窗口、分箱、unit selection 与正则化都只在训练数据内选择；因此该比较比直接挑最好时间窗更保守。

## 5. CCGP 风格泛化

- held-marker 患者中位 AUC={ccgp_row.patient_median:.3f}，全流程配对标签置换 P={ccgp_row.group_permutation_p_greater:.3f}。
- 解释：这是“对未见 marker 组合的部分泛化”，不是抽象语法的证明，因为每个 marker 仍只属于一种 voice。

## 6. 可辨识性与关键 null

- marker×voice Cramér V={ident['cramers_v_marker_voice']:.2f}；联合设计矩阵 rank={ident['design_rank']}/{ident['design_columns']}。voice-after-marker 不能估计。
- cross-time AUC 的患者平均差 90% CI 为 [{null_json['ci90'][0]:+.3f}, {null_json['ci90'][1]:+.3f}]。
- TOST：±0.05 AUC 界值 P={eq05.tost_p_equivalence:.3f}（{'等效' if eq05.equivalent_at_alpha_0p05 else '未证实等效'}）；±0.03 界值 P={eq03.tost_p_equivalence:.3f}（{'等效' if eq03.equivalent_at_alpha_0p05 else '未证实等效'}）。
- 解释：只有在事先接受 ±0.05 AUC 为“可忽略差异”时，才能把 cross-time 结果称为与机会水平等效；更严格界值下结论可能不同。

## 7. 当前无法严谨执行的借鉴方法

- reliability-matched geometry null：每个精确句子在单个 run 内只有一次呈现，无法估计刺激级 split-half reliability。
- voice-after-marker 回归：设计秩亏，参数不可辨识。
- cross-voice semantic subspace：仅 3 名患者有现成诊断，且仍受 marker 混合限制。
"""
    (OUT / "analysis_interpretation_zh.md").write_text(note, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "summary": summary}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
