#!/usr/bin/env python3
"""Unsupervised classification using only the timing of corrected A-P effects.

Two questions are separated deliberately:

1. Do complete direction-free, amplitude-free temporal envelopes form stable
   discrete clusters?  This is evaluated with PCA/K-means, bootstrap ARI, and
   standard separation metrics.
2. Can peak latency provide a simpler descriptive timing taxonomy?  One-
   dimensional Gaussian mixture models are compared by BIC and by leave-one-
   patient-out reproducibility.

Condition direction and effect magnitude never enter either clustering model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


SOURCE_DIR = ROOT / "reports/active_passive_go_style_interpretable_20260723"
PROFILE_PATH = SOURCE_DIR / "interpretable_cluster_unit_profiles_5ms.csv"
NODE_PATH = SOURCE_DIR / "interpretable_cluster_nodes.csv"
OUT_DIR = ROOT / "reports/active_passive_unsupervised_time_clustering_20260724"

INK = "#17322F"
GRAY = "#6B7B78"
GRID = "#DDE5E2"
WHITE = "#FFFFFF"
RED = "#D95F4C"
BLUE = "#2C7FB8"
GREEN = "#24936E"
GOLD = "#D6A84A"
RUN_ORDER = [
    "gz001t_g0_actpas_run1",
    "gz002_g0_actpas_run2",
    "gz003t2_g0_actpas_run1",
    "gz003t2_g0_actpas_run4",
    "gz004t1_g0_actpas_run0",
    "gz004t1_g3_actpas_run3",
    "gz005t0_g0_actpas_run0",
    "gz005t0_g3_actpas_run3",
]
RUN_SHORT = {
    "gz001t_g0_actpas_run1": "P1 · run1",
    "gz002_g0_actpas_run2": "P2 · run2",
    "gz003t2_g0_actpas_run1": "P3 · run1",
    "gz003t2_g0_actpas_run4": "P3 · run4",
    "gz004t1_g0_actpas_run0": "P4 · run0",
    "gz004t1_g3_actpas_run3": "P4 · run3",
    "gz005t0_g0_actpas_run0": "P5 · run0",
    "gz005t0_g3_actpas_run3": "P5 · run3",
}
TIME_BAND_NAMES = {1: "Early", 2: "Middle", 3: "Late"}
POOLED_K4_BAND_NAMES = {1: "Early", 2: "Middle", 3: "Late", 4: "Ultra-late edge"}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": WHITE,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor=WHITE, **kwargs)
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    long = pd.read_csv(PROFILE_PATH)
    nodes = pd.read_csv(NODE_PATH)
    long["node_key"] = (
        long["subject"].astype(str)
        + "|"
        + long["run"].astype(str)
        + "|"
        + long["unit_id"].astype(int).astype(str)
    )
    times = np.sort(long["time_ms"].unique().astype(int))
    order = nodes["node_key"].astype(str).tolist()
    masked = (
        long.pivot(index="node_key", columns="time_ms", values="qualifying_masked_contrast_hz")
        .reindex(index=order, columns=times)
        .to_numpy(float)
    )
    if masked.shape != (172, 160) or np.any(~np.isfinite(masked)):
        raise ValueError(f"unexpected masked profile matrix: {masked.shape}")
    if np.any(np.abs(masked).sum(axis=1) <= 0):
        raise ValueError("selected unit with no qualifying temporal energy")
    if not nodes["corrected_contiguous_gt5ms"].astype(bool).all():
        raise ValueError("input includes units that fail the corrected >5 ms criterion")
    return nodes.copy(), times, masked


def time_only_features(masked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    envelope = np.abs(masked)
    probability = envelope / envelope.sum(axis=1, keepdims=True)
    # Euclidean distance after square-root embedding is proportional to the
    # Hellinger distance between temporal energy distributions.
    hellinger = np.sqrt(probability)
    return envelope, probability, hellinger


def bootstrap_stability(
    scores: np.ndarray,
    reference: np.ndarray,
    k: int,
    n_boot: int = 100,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(240724 + k)
    values: list[float] = []
    n = len(scores)
    for boot in range(n_boot):
        sample = rng.integers(0, n, size=n)
        model = KMeans(n_clusters=k, n_init=40, random_state=240724 + 1000 * k + boot)
        model.fit(scores[sample])
        values.append(float(adjusted_rand_score(reference, model.predict(scores))))
    array = np.asarray(values)
    return float(array.mean()), float(np.percentile(array, 5)), float(np.percentile(array, 95))


def evaluate_complete_envelopes(hellinger: np.ndarray) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    pca = PCA(n_components=0.90, svd_solver="full")
    scores = pca.fit_transform(hellinger)
    rows: list[dict[str, float | int]] = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=150, random_state=240724)
        labels = model.fit_predict(scores)
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scores)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=15,
            reg_covar=1e-5,
            random_state=240724,
        ).fit(scores)
        stability, p05, p95 = bootstrap_stability(scores, labels, k)
        counts = np.bincount(labels, minlength=k)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(scores, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(scores, labels)),
                "davies_bouldin": float(davies_bouldin_score(scores, labels)),
                "bootstrap_ari_mean": stability,
                "bootstrap_ari_p05": p05,
                "bootstrap_ari_p95": p95,
                "kmeans_vs_ward_ari": float(adjusted_rand_score(labels, ward)),
                "gmm_diag_bic": float(gmm.bic(scores)),
                "gmm_diag_aic": float(gmm.aic(scores)),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "inertia": float(model.inertia_),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["delta_gmm_bic"] = metrics["gmm_diag_bic"] - metrics["gmm_diag_bic"].min()
    metrics["passes_stability_threshold"] = (
        (metrics["bootstrap_ari_mean"] >= 0.75) & (metrics["min_cluster_size"] >= 8)
    )
    return metrics, pca, scores


def temporal_moments(envelope: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    weights = envelope / envelope.sum(axis=1, keepdims=True)
    center = (weights * times[None, :]).sum(axis=1)
    variance = (weights * (times[None, :] - center[:, None]) ** 2).sum(axis=1)
    peak = times[np.argmax(envelope, axis=1)]
    onset = np.asarray([times[np.flatnonzero(row > 0)[0]] for row in envelope])
    offset = np.asarray([times[np.flatnonzero(row > 0)[-1]] for row in envelope])
    return pd.DataFrame(
        {
            "temporal_peak_ms": peak.astype(int),
            "temporal_center_ms": center,
            "temporal_sd_ms": np.sqrt(variance),
            "qualifying_onset_5ms_ms": onset.astype(int),
            "qualifying_offset_5ms_ms": offset.astype(int),
        }
    )


def ordered_prediction(
    model: GaussianMixture,
    component_order: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float).reshape(-1, 1)
    probabilities = model.predict_proba(x)[:, component_order]
    labels = np.argmax(probabilities, axis=1).astype(int) + 1
    return labels, probabilities


def fit_ordered_peak_gmm(
    peaks: np.ndarray,
    k: int,
    random_state: int,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x = np.asarray(peaks, dtype=float).reshape(-1, 1)
    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=100,
        reg_covar=1e-3,
        random_state=random_state,
    ).fit(x)
    order = np.argsort(model.means_.ravel())
    labels, probabilities = ordered_prediction(model, order, peaks)
    variances = model.covariances_.reshape(k, -1)[:, 0]
    parameters = pd.DataFrame(
        {
            "time_cluster": np.arange(1, k + 1, dtype=int),
            "component_mean_ms": model.means_.ravel()[order],
            "component_sd_ms": np.sqrt(variances[order]),
            "component_weight": model.weights_.ravel()[order],
        }
    )
    parameters["label"] = [f"T{value}" for value in parameters["time_cluster"]]
    return model, order, labels, probabilities, parameters


def leave_one_patient_out_peak(
    nodes: pd.DataFrame,
    peaks: np.ndarray,
    reference: np.ndarray,
    k: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    predicted_all = np.zeros(len(nodes), dtype=int)
    rows: list[dict[str, object]] = []
    subjects = sorted(nodes["subject"].astype(str).unique())
    for fold, subject in enumerate(subjects):
        test = nodes["subject"].astype(str).eq(subject).to_numpy()
        train = ~test
        model, order, _, _, parameters = fit_ordered_peak_gmm(
            peaks[train],
            k,
            random_state=240724 + 100 * k + fold,
        )
        predicted, heldout_probability = ordered_prediction(model, order, peaks[test])
        predicted_all[test] = predicted
        rows.append(
            {
                "k": k,
                "held_out_subject": subject,
                "n_test_units": int(test.sum()),
                "ari_vs_full_solution": float(adjusted_rand_score(reference[test], predicted)),
                "agreement_fraction": float(np.mean(reference[test] == predicted)),
                "mean_max_posterior": float(heldout_probability.max(axis=1).mean()),
                "training_component_means_ms": ";".join(
                    f"{value:.1f}" for value in parameters["component_mean_ms"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "overall_ari_vs_full_solution": float(adjusted_rand_score(reference, predicted_all)),
        "overall_agreement_fraction": float(np.mean(reference == predicted_all)),
        "median_fold_ari": float(frame["ari_vs_full_solution"].median()),
        "min_fold_ari": float(frame["ari_vs_full_solution"].min()),
        "median_fold_agreement": float(frame["agreement_fraction"].median()),
        "min_fold_agreement": float(frame["agreement_fraction"].min()),
    }
    return frame, summary


def evaluate_peak_gmms(
    nodes: pd.DataFrame,
    peaks: np.ndarray,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    x = peaks.reshape(-1, 1).astype(float)
    rows: list[dict[str, object]] = []
    fold_results: dict[int, pd.DataFrame] = {}
    for k in range(1, 7):
        model, _, labels, _, parameters = fit_ordered_peak_gmm(peaks, k, 240724)
        folds, lopo = leave_one_patient_out_peak(nodes, peaks, labels, k)
        fold_results[k] = folds
        counts = np.bincount(labels, minlength=k + 1)[1:]
        rows.append(
            {
                "k": k,
                "bic": float(model.bic(x)),
                "aic": float(model.aic(x)),
                "silhouette_peak_time": (
                    float(silhouette_score(x, labels)) if k > 1 and len(np.unique(labels)) > 1 else np.nan
                ),
                "component_means_ms": ";".join(
                    f"{value:.1f}" for value in parameters["component_mean_ms"]
                ),
                "component_sds_ms": ";".join(
                    f"{value:.1f}" for value in parameters["component_sd_ms"]
                ),
                "component_weights": ";".join(
                    f"{value:.4f}" for value in parameters["component_weight"]
                ),
                "min_component_size": int(counts.min()),
                "max_component_size": int(counts.max()),
                "lopo_overall_ari": lopo["overall_ari_vs_full_solution"],
                "lopo_overall_agreement": lopo["overall_agreement_fraction"],
                "lopo_median_fold_ari": lopo["median_fold_ari"],
                "lopo_min_fold_ari": lopo["min_fold_ari"],
                "lopo_median_fold_agreement": lopo["median_fold_agreement"],
                "lopo_min_fold_agreement": lopo["min_fold_agreement"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["delta_bic"] = frame["bic"] - frame["bic"].min()
    relative = np.exp(-0.5 * frame["delta_bic"].to_numpy(float))
    frame["bic_relative_weight"] = relative / relative.sum()
    return frame, fold_results


def choose_primary_peak_k(selection: pd.DataFrame) -> tuple[int, str]:
    # This is a transparent descriptive rule, not a preregistered inferential
    # test: exclude strongly BIC-disfavored models (delta BIC >= 10), then use
    # patient-level generalization rather than pooled fit to choose resolution.
    candidates = selection.loc[(selection["k"] >= 2) & (selection["delta_bic"] < 10)].copy()
    if candidates.empty:
        row = selection.loc[selection["k"] >= 2].sort_values("bic").iloc[0]
        return int(row["k"]), "无ΔBIC<10候选；使用BIC最优的非平凡模型"
    row = candidates.sort_values(
        ["lopo_overall_ari", "lopo_overall_agreement", "delta_bic"],
        ascending=[False, False, True],
    ).iloc[0]
    return (
        int(row["k"]),
        "在ΔBIC<10的可竞争模型中，选择leave-one-patient-out ARI最高的时间分辨率",
    )


def summarize_clusters(
    nodes: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    moments: pd.DataFrame,
    parameters: pd.DataFrame,
    band_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    if band_names is None:
        band_names = TIME_BAND_NAMES
    joined = pd.concat([nodes.reset_index(drop=True), moments.reset_index(drop=True)], axis=1)
    joined["time_cluster"] = labels
    joined["max_posterior"] = probabilities.max(axis=1)
    rows: list[dict[str, object]] = []
    for cluster in sorted(np.unique(labels)):
        frame = joined.loc[joined["time_cluster"].eq(cluster)]
        parameter = parameters.loc[parameters["time_cluster"].eq(cluster)].iloc[0]
        rows.append(
            {
                "time_cluster": int(cluster),
                "label": f"T{cluster}",
                "descriptive_name": band_names.get(int(cluster), f"Band {cluster}"),
                "n_units": int(len(frame)),
                "n_subjects": int(frame["subject"].nunique()),
                "n_sessions": int(frame["run"].nunique()),
                "component_mean_ms": float(parameter["component_mean_ms"]),
                "component_sd_ms": float(parameter["component_sd_ms"]),
                "component_weight": float(parameter["component_weight"]),
                "observed_peak_min_ms": float(frame["temporal_peak_ms"].min()),
                "observed_peak_max_ms": float(frame["temporal_peak_ms"].max()),
                "median_peak_ms": float(frame["temporal_peak_ms"].median()),
                "peak_iqr_low_ms": float(frame["temporal_peak_ms"].quantile(0.25)),
                "peak_iqr_high_ms": float(frame["temporal_peak_ms"].quantile(0.75)),
                "median_center_ms": float(frame["temporal_center_ms"].median()),
                "median_temporal_sd_ms": float(frame["temporal_sd_ms"].median()),
                "median_qualifying_onset_ms": float(frame["qualifying_onset_5ms_ms"].median()),
                "median_qualifying_offset_ms": float(frame["qualifying_offset_5ms_ms"].median()),
                "mean_max_posterior": float(frame["max_posterior"].mean()),
                "fraction_posterior_ge_0_80": float(frame["max_posterior"].ge(0.80).mean()),
                "active_direction_fraction": float(frame["corrected_direction"].eq("active_gt_passive").mean()),
                "passive_direction_fraction": float(frame["corrected_direction"].eq("passive_gt_active").mean()),
            }
        )
    return pd.DataFrame(rows)


def session_composition(nodes: pd.DataFrame, labels: np.ndarray, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        nodes.assign(time_cluster=labels)
        .groupby(["run", "time_cluster"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=RUN_ORDER, columns=range(1, k + 1), fill_value=0)
    )
    counts.columns = [f"T{value}_count" for value in counts.columns]
    counts.insert(0, "session", counts.index)
    counts.insert(1, "session_label", [RUN_SHORT[value] for value in counts.index])
    counts["n_significant_units"] = counts.filter(like="_count").sum(axis=1).astype(int)
    fractions = counts[["session", "session_label", "n_significant_units"]].copy()
    denominator = counts["n_significant_units"].replace(0, np.nan)
    for cluster in range(1, k + 1):
        fractions[f"T{cluster}_fraction"] = (
            counts[f"T{cluster}_count"].div(denominator).fillna(0.0)
        )
    return counts.reset_index(drop=True), fractions.reset_index(drop=True)


def cluster_palette(k: int) -> list[tuple[float, float, float, float] | str]:
    if k == 3:
        return ["#3B5BA7", "#1B9E77", "#E6A32F"]
    cmap = plt.get_cmap("viridis")
    return [cmap(value) for value in np.linspace(0.14, 0.80, k)]


def plot_model_selection(
    envelope_metrics: pd.DataFrame,
    peak_selection: pd.DataFrame,
    primary_k: int,
) -> None:
    bic_best_k = int(peak_selection.loc[peak_selection["bic"].idxmin(), "k"])
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), facecolor=WHITE)
    ax_sil, ax_boot, ax_bic, ax_lopo = axes.ravel()

    ax_sil.plot(envelope_metrics["k"], envelope_metrics["silhouette"], color=BLUE, marker="o", lw=2)
    best_sil = envelope_metrics.loc[envelope_metrics["silhouette"].idxmax()]
    ax_sil.scatter([best_sil["k"]], [best_sil["silhouette"]], s=70, color=BLUE, zorder=3)
    ax_sil.annotate(
        f"maximum = {best_sil['silhouette']:.3f}",
        xy=(best_sil["k"], best_sil["silhouette"]),
        xytext=(-76, 22),
        textcoords="offset points",
        color=GRAY,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": GRID},
    )
    ax_sil.set_title("A · Complete envelope: weak separation", loc="left", fontweight="bold", color=INK)
    ax_sil.set_ylabel("Silhouette")

    ax_boot.plot(envelope_metrics["k"], envelope_metrics["bootstrap_ari_mean"], color=GREEN, marker="o", lw=2)
    ax_boot.fill_between(
        envelope_metrics["k"],
        envelope_metrics["bootstrap_ari_p05"],
        envelope_metrics["bootstrap_ari_p95"],
        color=GREEN,
        alpha=0.13,
        lw=0,
    )
    ax_boot.axhline(0.75, color=RED, ls="--", lw=1.2, label="predefined stability threshold")
    ax_boot.set_ylim(0, 1.02)
    ax_boot.set_title("B · Complete envelope: no stable K", loc="left", fontweight="bold", color=INK)
    ax_boot.set_ylabel("Bootstrap ARI vs full solution")
    ax_boot.legend(frameon=False, loc="upper left")

    ax_bic.plot(peak_selection["k"], peak_selection["delta_bic"], color=GRAY, marker="o", lw=1.8)
    bic_row = peak_selection.loc[peak_selection["k"].eq(bic_best_k)].iloc[0]
    primary_row = peak_selection.loc[peak_selection["k"].eq(primary_k)].iloc[0]
    ax_bic.scatter([bic_best_k], [bic_row["delta_bic"]], s=95, color=RED, marker="*", zorder=4, label=f"BIC-only best: K={bic_best_k}")
    ax_bic.scatter([primary_k], [primary_row["delta_bic"]], s=75, color=GOLD, zorder=4, label=f"patient-robust: K={primary_k}")
    ax_bic.axhline(10, color=GRID, ls="--", lw=1)
    ax_bic.set_title("C · Peak-latency GMM: pooled fit", loc="left", fontweight="bold", color=INK)
    ax_bic.set_xlabel("GMM components, K")
    ax_bic.set_ylabel("ΔBIC (lower is better)")
    ax_bic.legend(frameon=False, loc="upper left")

    cv = peak_selection.loc[peak_selection["k"] >= 2]
    ax_lopo.plot(cv["k"], cv["lopo_overall_agreement"], color=BLUE, marker="o", lw=2, label="exact agreement")
    ax_lopo.plot(cv["k"], cv["lopo_overall_ari"], color=GREEN, marker="s", lw=2, label="adjusted Rand index")
    ax_lopo.scatter(
        [primary_k, primary_k],
        [primary_row["lopo_overall_agreement"], primary_row["lopo_overall_ari"]],
        s=80,
        facecolor=GOLD,
        edgecolor="none",
        zorder=4,
    )
    ax_lopo.set_ylim(0, 1.03)
    ax_lopo.set_title("D · Peak-latency GMM: patient generalization", loc="left", fontweight="bold", color=INK)
    ax_lopo.set_xlabel("GMM components, K")
    ax_lopo.set_ylabel("Leave-one-patient-out score")
    ax_lopo.legend(frameon=False, loc="lower left")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID, lw=0.65)
    ax_sil.set_xticks(envelope_metrics["k"])
    ax_boot.set_xticks(envelope_metrics["k"])
    ax_bic.set_xticks(peak_selection["k"])
    ax_lopo.set_xticks(cv["k"])

    fig.suptitle(
        "Full temporal profiles form a continuum; peak timing supports three patient-robust bands",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.947,
        "Direction and magnitude removed · K=4 minimizes pooled BIC, but K=3 generalizes markedly better across patients",
        ha="left",
        fontsize=8.5,
        color=GRAY,
    )
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.075, right=0.98, hspace=0.40, wspace=0.27)
    save_figure(fig, "Fig_time_only_model_selection")


def plot_time_clusters(
    nodes: pd.DataFrame,
    times: np.ndarray,
    probability: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    moments: pd.DataFrame,
    parameters: pd.DataFrame,
    summary: pd.DataFrame,
    session_fractions: pd.DataFrame,
    stem: str = "Fig_time_only_clusters_and_sessions",
    figure_title: str = "Peak timing alone yields three descriptive bands: early, middle, and late",
    figure_subtitle: str = (
        "Membership uses only the peak time of |A−P| in corrected qualifying bins; "
        "active/passive direction and effect magnitude are excluded"
    ),
) -> None:
    k = len(np.unique(labels))
    colors = cluster_palette(k)
    order_parts: list[np.ndarray] = []
    boundaries: list[int] = []
    total = 0
    for cluster in range(1, k + 1):
        idx = np.flatnonzero(labels == cluster)
        sort_frame = moments.iloc[idx][["temporal_peak_ms", "temporal_center_ms"]]
        idx = idx[np.lexsort((sort_frame["temporal_center_ms"], sort_frame["temporal_peak_ms"]))]
        order_parts.append(idx)
        total += len(idx)
        boundaries.append(total)
    order = np.concatenate(order_parts)

    fig = plt.figure(figsize=(14.2, 8.2), facecolor=WHITE)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.25, 1, 1],
        height_ratios=[1.05, 1],
        wspace=0.32,
        hspace=0.38,
    )
    ax_heat = fig.add_subplot(grid[:, 0])
    ax_hist = fig.add_subplot(grid[0, 1:])
    ax_mean = fig.add_subplot(grid[1, 1])
    ax_comp = fig.add_subplot(grid[1, 2])

    image = ax_heat.imshow(
        np.sqrt(probability[order]),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        extent=[0, 800, len(order), 0],
    )
    start = 0
    for cluster, boundary in enumerate(boundaries, start=1):
        row = summary.loc[summary["time_cluster"].eq(cluster)].iloc[0]
        center = (start + boundary) / 2
        ax_heat.text(
            -38,
            center,
            f"T{cluster}  {row['descriptive_name']}\nn={boundary-start} · median {row['median_peak_ms']:.0f} ms",
            ha="right",
            va="center",
            color=colors[cluster - 1],
            fontweight="bold",
            fontsize=8,
        )
        ax_heat.axhline(boundary, color=WHITE, lw=2.2)
        start = boundary
    ax_heat.set_title("A · Significant-effect timing by unit", loc="left", fontweight="bold", color=INK)
    ax_heat.set_xlabel("Time from corrected function-word onset (ms)")
    ax_heat.set_ylabel("172 units sorted by GMM band and peak latency")
    ax_heat.set_yticks([])
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.02)
    cbar.set_label("√ normalized |A−P| temporal energy")

    peaks = moments["temporal_peak_ms"].to_numpy(float)
    bins = np.arange(-12.5, 813, 25)
    bin_width = float(np.diff(bins)[0])
    ax_hist.hist(peaks, bins=bins, color="#E7ECEA", edgecolor="none", label="observed peak latency")
    dense = np.linspace(0, 795, 1600)
    mixture = np.zeros_like(dense)
    for row in parameters.itertuples(index=False):
        curve = row.component_weight * norm.pdf(dense, row.component_mean_ms, row.component_sd_ms) * len(peaks) * bin_width
        mixture += curve
        color = colors[row.time_cluster - 1]
        ax_hist.fill_between(dense, 0, curve, color=color, alpha=0.16, lw=0)
        ax_hist.plot(dense, curve, color=color, lw=2, label=f"T{row.time_cluster}: μ={row.component_mean_ms:.0f} ms")
        ax_hist.axvline(row.component_mean_ms, color=color, lw=1, ls=":")
    ax_hist.plot(dense, mixture, color=INK, lw=1.8, ls="--", label=f"{k}-component mixture")
    confident = float(np.mean(probabilities.max(axis=1) >= 0.80))
    ax_hist.text(
        0.985,
        0.93,
        f"{confident:.1%} of units: posterior ≥ 0.80",
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        color=GRAY,
        fontsize=8.5,
    )
    ax_hist.set_xlim(0, 800)
    ax_hist.set_title("B · Peak-latency distribution and selected GMM", loc="left", fontweight="bold", color=INK)
    ax_hist.set_xlabel("Peak latency (ms)")
    ax_hist.set_ylabel("Units per 25-ms bin")
    ax_hist.legend(frameon=False, ncol=2, loc="upper left")

    for cluster in range(1, k + 1):
        idx = labels == cluster
        mean = probability[idx].mean(axis=0)
        sem = probability[idx].std(axis=0, ddof=1) / np.sqrt(int(idx.sum()))
        color = colors[cluster - 1]
        ax_mean.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.16, lw=0)
        ax_mean.plot(times, mean, color=color, lw=2, label=f"T{cluster} (n={int(idx.sum())})")
    ax_mean.set_xlim(0, 795)
    ax_mean.set_title("C · Mean normalized temporal envelope", loc="left", fontweight="bold", color=INK)
    ax_mean.set_xlabel("Time (ms)")
    ax_mean.set_ylabel("Fraction of temporal effect energy")
    ax_mean.legend(frameon=False, fontsize=7.5)

    y = np.arange(len(session_fractions))
    left = np.zeros(len(session_fractions))
    for cluster in range(1, k + 1):
        values = session_fractions[f"T{cluster}_fraction"].to_numpy(float)
        ax_comp.barh(y, values, left=left, height=0.67, color=colors[cluster - 1], edgecolor="none", label=f"T{cluster}")
        left += values
    totals = session_fractions["n_significant_units"].astype(int).to_numpy()
    for index, value in enumerate(totals):
        ax_comp.text(1.02, index, f"n={value}", va="center", fontsize=7, color=INK)
    ax_comp.set_yticks(y, session_fractions["session_label"])
    ax_comp.invert_yaxis()
    ax_comp.set_xlim(0, 1.10)
    ax_comp.set_xlabel("Fraction of significant units")
    ax_comp.set_title("D · Timing-band composition by session", loc="left", fontweight="bold", color=INK)
    ax_comp.legend(frameon=False, ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    ax_comp.spines["left"].set_visible(False)
    ax_comp.tick_params(axis="y", length=0)

    for ax in (ax_hist, ax_mean, ax_comp):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID, lw=0.6, axis="y" if ax is not ax_comp else "x")

    fig.suptitle(
        figure_title,
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.950,
        figure_subtitle,
        ha="left",
        fontsize=8.5,
        color=GRAY,
    )
    fig.subplots_adjust(top=0.89, bottom=0.12, left=0.11, right=0.985)
    save_figure(fig, stem)


def write_report(
    envelope_metrics: pd.DataFrame,
    peak_selection: pd.DataFrame,
    primary_k: int,
    reason: str,
    summary: pd.DataFrame,
    lopo_folds: pd.DataFrame,
) -> None:
    bic_best_k = int(peak_selection.loc[peak_selection["bic"].idxmin(), "k"])
    primary = peak_selection.loc[peak_selection["k"].eq(primary_k)].iloc[0]
    bic_best = peak_selection.loc[peak_selection["k"].eq(bic_best_k)].iloc[0]
    best_boot = envelope_metrics.loc[envelope_metrics["bootstrap_ari_mean"].idxmax()]
    best_sil = envelope_metrics.loc[envelope_metrics["silhouette"].idxmax()]
    worst_fold = lopo_folds.loc[lopo_folds["agreement_fraction"].idxmin()]

    lines = [
        "# 只按时间进行的无监督分类",
        "",
        "## 一句话结论",
        "",
        f"完整时间轨迹更像连续谱，而不是清楚分离的离散cluster；如果需要一个可汇报、且能够跨患者复现的时间摘要，峰值时间可分为{primary_k}个描述性时间带：早期、中期和晚期。",
        "",
        "## 输入与预处理",
        "",
        "- 输入为172个通过corrected显著标准的unit；每个unit有0–795 ms、5 ms分辨率的时间轨迹。",
        "- corrected标准为连续显著时间>5 ms，并使用多条件校正后的显著性阈值；本脚本再次核对所有输入unit都满足corrected_contiguous_gt5ms。",
        "- 对每个时间点取|A−P|，因此主动大于被动或主动小于被动都不会影响分类。",
        "- 每个unit的绝对效应轨迹归一化为总和1，因此总体放电强弱和效应幅度不会影响分类。",
        "- 完整包络分析使用平方根概率，即Hellinger时间能量表示；峰值GMM只使用每个unit最大显著绝对效应出现的时间。",
        "",
        "## 为什么不把完整160点轨迹硬分成cluster",
        "",
        f"- K=2–8中，最高silhouette只有{best_sil['silhouette']:.3f}（K={int(best_sil['k'])}）。",
        f"- 最高bootstrap ARI只有{best_boot['bootstrap_ari_mean']:.3f}（K={int(best_boot['k'])}），低于预设稳定阈值0.75。",
        "- 因此，完整时间包络没有支持高度稳定的离散类别；此前临时K=7只能视为探索性切分，不能作为主要生物学结果。",
        "",
        f"## 峰值时间模型：为什么主结果用K={primary_k}，而不是BIC最低的K={bic_best_k}",
        "",
        f"- pooled BIC最低的是K={bic_best_k}；K={primary_k}的ΔBIC={primary['delta_bic']:.2f}。这说明单看总体拟合，{bic_best_k}个成分略占优势。",
        f"- 但K={bic_best_k}患者留一的总体ARI={bic_best['lopo_overall_ari']:.3f}、一致率={bic_best['lopo_overall_agreement']:.1%}，跨患者不稳。",
        f"- K={primary_k}患者留一的总体ARI={primary['lopo_overall_ari']:.3f}、一致率={primary['lopo_overall_agreement']:.1%}；最差患者为{worst_fold['held_out_subject']}，一致率仍为{worst_fold['agreement_fraction']:.1%}。",
        f"- 因此采用透明的描述性规则：{reason}。这不是预注册的推断检验，而是为了避免被单一患者驱动的模型选择。",
        "",
        "## 三个时间带",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.label}（{row.descriptive_name}）: n={row.n_units}；GMM均值={row.component_mean_ms:.0f} ms；"
            f"peak中位数={row.median_peak_ms:.0f} ms（IQR {row.peak_iqr_low_ms:.0f}–{row.peak_iqr_high_ms:.0f}）；"
            f"患者={row.n_subjects}，session={row.n_sessions}；posterior≥0.80为{row.fraction_posterior_ge_0_80:.1%}；"
            f"分类后描述的A>P比例={row.active_direction_fraction:.1%}。"
        )
    lines.extend(
        [
            "",
            "## 应该如何解释",
            "",
            "- 可以说：显著主动—被动差异在时间上形成早、中、晚三个可复现的描述性时间带。",
            "- 更稳妥地说：时间组织是连续的，但三个宽时间带能以较低复杂度概括主要峰值时序。",
            "- 不能说：数据发现了三种天然神经元类型、三个解剖回路或三个因果处理阶段。",
            "- A>P/A<P比例只是分类后的描述，不参与cluster生成，不能用于证明cluster由主动或被动条件驱动。",
            "- P4 run3当前没有通过corrected标准的unit，因此session组成图保留n=0空行，避免误以为数据遗漏。",
            "",
            "## 推荐汇报句",
            "",
            "> 去除主动/被动方向和效应幅度后，完整时间轨迹不支持稳定的离散类别；但单元显著效应的峰值时间可由早、中、晚三个描述性时间带概括。三分类在患者留一验证中的总体一致率为"
            f"{primary['lopo_overall_agreement']:.1%}，因此比pooled BIC偏好的四分类更适合作为跨患者汇报结果。",
            "",
        ]
    )
    (OUT_DIR / "README_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes, times, masked = load_data()
    envelope, probability, hellinger = time_only_features(masked)
    moments = temporal_moments(envelope, times)

    envelope_metrics, pca, scores = evaluate_complete_envelopes(hellinger)
    peaks = moments["temporal_peak_ms"].to_numpy(float)
    peak_selection, fold_results = evaluate_peak_gmms(nodes, peaks)
    primary_k, reason = choose_primary_peak_k(peak_selection)
    model, order, labels, posterior, parameters = fit_ordered_peak_gmm(peaks, primary_k, 240724)
    selected_lopo = fold_results[primary_k]
    summary = summarize_clusters(nodes, labels, posterior, moments, parameters)
    session_counts, session_fractions = session_composition(nodes, labels, primary_k)

    # Preserve the pooled-BIC K=4 solution as a clearly labeled supplement.
    # It is useful for showing what the extra component represents, but it is
    # not promoted to the cross-patient primary result.
    _, _, pooled_k4_labels, pooled_k4_posterior, pooled_k4_parameters = fit_ordered_peak_gmm(
        peaks, 4, 240724
    )
    pooled_k4_summary = summarize_clusters(
        nodes,
        pooled_k4_labels,
        pooled_k4_posterior,
        moments,
        pooled_k4_parameters,
        band_names=POOLED_K4_BAND_NAMES,
    )
    pooled_k4_session_counts, pooled_k4_session_fractions = session_composition(
        nodes, pooled_k4_labels, 4
    )

    assignments = pd.concat([nodes.reset_index(drop=True), moments], axis=1)
    assignments["time_cluster"] = labels
    assignments["time_cluster_label"] = [f"T{value}" for value in labels]
    assignments["time_cluster_name"] = [TIME_BAND_NAMES.get(int(value), f"Band {value}") for value in labels]
    assignments["gmm_max_posterior"] = posterior.max(axis=1)
    assignments["gmm_uncertain_posterior_lt_0_80"] = assignments["gmm_max_posterior"] < 0.80
    clipped = np.clip(posterior, 1e-12, 1.0)
    assignments["gmm_normalized_entropy"] = -(clipped * np.log(clipped)).sum(axis=1) / np.log(primary_k)
    for cluster in range(1, primary_k + 1):
        assignments[f"gmm_posterior_T{cluster}"] = posterior[:, cluster - 1]

    # Assertions make the analysis auditable and catch accidental changes to
    # the selected population or probability ordering.
    if not np.allclose(posterior.sum(axis=1), 1.0, atol=1e-8):
        raise AssertionError("GMM posterior probabilities do not sum to one")
    if sorted(assignments["time_cluster"].unique()) != list(range(1, primary_k + 1)):
        raise AssertionError("ordered time-cluster labels are incomplete")
    if int(session_counts.loc[session_counts["session"].eq("gz004t1_g3_actpas_run3"), "n_significant_units"].iloc[0]) != 0:
        raise AssertionError("expected P4 run3 to remain an explicit zero-unit session")

    assignments.to_csv(OUT_DIR / "time_cluster_assignments.csv", index=False)
    summary.to_csv(OUT_DIR / "time_cluster_summary.csv", index=False)
    parameters.to_csv(OUT_DIR / "peak_latency_gmm_components.csv", index=False)
    envelope_metrics.to_csv(OUT_DIR / "time_cluster_model_selection.csv", index=False)
    peak_selection.to_csv(OUT_DIR / "peak_latency_gmm_selection.csv", index=False)
    selected_lopo.to_csv(OUT_DIR / "leave_one_patient_out_validation.csv", index=False)
    pd.concat(fold_results.values(), ignore_index=True).to_csv(
        OUT_DIR / "leave_one_patient_out_all_k.csv", index=False
    )
    session_counts.to_csv(OUT_DIR / "time_cluster_session_counts.csv", index=False)
    session_fractions.to_csv(OUT_DIR / "time_cluster_session_fractions.csv", index=False)
    pd.crosstab(assignments["time_cluster_label"], assignments["corrected_direction"], margins=True).to_csv(
        OUT_DIR / "time_cluster_by_direction.csv"
    )
    pd.crosstab(assignments["time_cluster_label"], assignments["cluster"], margins=True).to_csv(
        OUT_DIR / "time_cluster_vs_predefined_six.csv"
    )

    pooled_k4_assignments = pd.concat([nodes.reset_index(drop=True), moments], axis=1)
    pooled_k4_assignments["pooled_k4_cluster"] = pooled_k4_labels
    pooled_k4_assignments["pooled_k4_label"] = [f"T{value}" for value in pooled_k4_labels]
    pooled_k4_assignments["pooled_k4_name"] = [
        POOLED_K4_BAND_NAMES[int(value)] for value in pooled_k4_labels
    ]
    pooled_k4_assignments["gmm_max_posterior"] = pooled_k4_posterior.max(axis=1)
    for cluster in range(1, 5):
        pooled_k4_assignments[f"gmm_posterior_T{cluster}"] = pooled_k4_posterior[:, cluster - 1]
    pooled_k4_assignments.to_csv(OUT_DIR / "pooled_bic_k4_assignments.csv", index=False)
    pooled_k4_summary.to_csv(OUT_DIR / "pooled_bic_k4_summary.csv", index=False)
    pooled_k4_parameters.to_csv(OUT_DIR / "pooled_bic_k4_components.csv", index=False)
    pooled_k4_session_counts.to_csv(OUT_DIR / "pooled_bic_k4_session_counts.csv", index=False)
    pd.crosstab(
        pd.Series(labels, name="primary_K3"),
        pd.Series(pooled_k4_labels, name="pooled_BIC_K4"),
        margins=True,
    ).to_csv(OUT_DIR / "pooled_bic_k4_vs_primary_k3.csv")

    plot_model_selection(envelope_metrics, peak_selection, primary_k)
    plot_time_clusters(
        nodes,
        times,
        probability,
        labels,
        posterior,
        moments,
        parameters,
        summary,
        session_fractions,
    )
    plot_time_clusters(
        nodes,
        times,
        probability,
        pooled_k4_labels,
        pooled_k4_posterior,
        moments,
        pooled_k4_parameters,
        pooled_k4_summary,
        pooled_k4_session_fractions,
        stem="Fig_pooled_bic_k4_time_clusters_and_sessions",
        figure_title="Pooled BIC solution resolves four peak-latency components",
        figure_subtitle=(
            "Supplementary pooled fit only · T4 is a small 765–795 ms boundary component · "
            "direction and magnitude remain excluded"
        ),
    )
    write_report(
        envelope_metrics,
        peak_selection,
        primary_k,
        reason,
        summary,
        selected_lopo,
    )

    bic_best_k = int(peak_selection.loc[peak_selection["bic"].idxmin(), "k"])
    primary_selection = peak_selection.loc[peak_selection["k"].eq(primary_k)].iloc[0]
    audit = {
        "n_units": int(len(nodes)),
        "n_subjects": int(nodes["subject"].nunique()),
        "n_observed_sessions": int(nodes["run"].nunique()),
        "n_displayed_sessions_including_zero": int(len(RUN_ORDER)),
        "n_time_bins": int(len(times)),
        "time_range_ms": [int(times.min()), int(times.max())],
        "corrected_unit_criterion_checked": "all corrected_contiguous_gt5ms == True",
        "complete_envelope_feature": "sqrt(abs(qualifying A-P) / temporal sum); Hellinger temporal energy",
        "primary_cluster_feature": "peak latency of abs(qualifying_masked_contrast_hz)",
        "direction_removed": True,
        "amplitude_removed": True,
        "pca_components_90pct": int(scores.shape[1]),
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
        "complete_envelope_any_stable_k": bool(envelope_metrics["passes_stability_threshold"].any()),
        "complete_envelope_max_bootstrap_ari": float(envelope_metrics["bootstrap_ari_mean"].max()),
        "complete_envelope_max_silhouette": float(envelope_metrics["silhouette"].max()),
        "pooled_bic_best_peak_k": bic_best_k,
        "primary_patient_robust_peak_k": int(primary_k),
        "selection_reason": reason,
        "primary_delta_bic": float(primary_selection["delta_bic"]),
        "primary_lopo_ari": float(primary_selection["lopo_overall_ari"]),
        "primary_lopo_agreement": float(primary_selection["lopo_overall_agreement"]),
        "cluster_counts": assignments["time_cluster"].value_counts().sort_index().astype(int).to_dict(),
        "component_parameters": parameters.to_dict(orient="records"),
        "fraction_posterior_ge_0_80": float(assignments["gmm_max_posterior"].ge(0.80).mean()),
        "interpretation": "three descriptive timing bands; full temporal organization remains closer to a continuum",
        "pooled_bic_k4": {
            "cluster_counts": pooled_k4_assignments["pooled_k4_cluster"]
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict(),
            "component_parameters": pooled_k4_parameters.to_dict(orient="records"),
            "t4_subjects": int(
                pooled_k4_assignments.loc[
                    pooled_k4_assignments["pooled_k4_cluster"].eq(4), "subject"
                ].nunique()
            ),
            "t4_sessions": int(
                pooled_k4_assignments.loc[
                    pooled_k4_assignments["pooled_k4_cluster"].eq(4), "run"
                ].nunique()
            ),
            "warning": "K=4 has poor patient-level generalization; T4 lies at the analysis-window boundary",
        },
    }
    (OUT_DIR / "analysis_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("\nPeak-latency model selection:")
    print(peak_selection.to_string(index=False))
    print("\nPrimary time-band summary:")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
