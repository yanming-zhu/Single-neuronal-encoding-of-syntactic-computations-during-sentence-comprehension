from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon
from PIL import Image
from scipy.spatial import ConvexHull


NET = ROOT / "neuropixel2025" / "data" / "combined_active_passive_network_encoding"
NETWORK_REPORT = ROOT / "reports" / "active_passive_network_encoding_figures"
ARCHETYPE_REPORT = ROOT / "reports" / "active_passive_go_style_interpretable_20260723"
ABLATION = (
    ROOT
    / "neuropixel2025"
    / "data"
    / "combined_active_passive_population_candidate_panels"
)
TRANSFER = (
    ROOT
    / "neuropixel2025"
    / "data"
    / "combined_active_passive_sentence_pair_transfer"
)
OUT = ROOT / "reports" / "nature_science_draft" / "assembled_figures_20260805"
SOURCE = OUT / "source_data" / "fig2_population_180mm_revised"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

WIDTH_MM = 180.0
HEIGHT_MM = 160.0
PT = 8.0

ACTIVE = "#357ABD"
PASSIVE = "#D95F4C"
PURPLE = "#7C6BB0"
ORANGE = "#D98C3A"
GRAY = "#6E7175"
LIGHT_GRAY = "#D8DDE3"
DARK = "#222222"

PATIENT_COLORS = {
    "gz001": "#0072B2",
    "gz002": "#D55E00",
    "gz003": "#009E73",
    "gz004": "#CC79A7",
    "gz005": "#E69F00",
}

CLUSTER_ORDER = ("A-E", "A-M", "A-L", "P-E", "P-M", "P-L")
CLUSTER_INFO = {
    "A-E": {"short": "A>P Early", "timing": "E", "color": "#9ECAE1"},
    "A-M": {"short": "A>P Middle", "timing": "M", "color": "#4292C6"},
    "A-L": {"short": "A>P Late", "timing": "L", "color": "#08519C"},
    "P-E": {"short": "A<P Early", "timing": "E", "color": "#FCAE91"},
    "P-M": {"short": "A<P Middle", "timing": "M", "color": "#FB6A4A"},
    "P-L": {"short": "A<P Late", "timing": "L", "color": "#CB181D"},
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": PT,
        "axes.labelsize": PT,
        "axes.titlesize": PT,
        "xtick.labelsize": PT,
        "ytick.labelsize": PT,
        "legend.fontsize": PT,
        "figure.titlesize": PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PT,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def load_endpoint_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(NET / "active_passive_network_encoding_model_scores.csv")
    primary = scores[
        (scores.analysis_mode == "artifact_clean_le_0p2")
        & (scores.unit_set == "good_only")
    ].copy()

    best = pd.read_csv(NET / "active_passive_best_network_models.csv")
    best_primary = best[
        (best.analysis_mode == "artifact_clean_le_0p2")
        & (best.unit_set == "good_only")
    ].copy()

    leave = pd.read_csv(NET / "active_passive_leave_pair_out_decoding.csv")
    leave = leave[leave.unit_set == "good_only"].rename(columns={"patient": "subject"})
    top5 = primary[primary.model == "network_top5_l2"][
        ["subject", "run", "auc"]
    ].rename(columns={"auc": "fixed_top5_auc"})

    run_endpoints = (
        best_primary[
            [
                "subject",
                "run",
                "nested_single_unit_auc",
                "auc",
                "auc_gain_vs_nested_single_unit",
            ]
        ]
        .rename(columns={"auc": "descriptive_best_population_auc"})
        .merge(top5, on=["subject", "run"], validate="one_to_one")
        .merge(
            leave[["subject", "run", "auc_leave_pair_out"]],
            on=["subject", "run"],
            validate="one_to_one",
        )
        .sort_values(["subject", "run"])
    )
    patient_endpoints = run_endpoints.groupby("subject", as_index=False).median(numeric_only=True)
    return run_endpoints, patient_endpoints


def load_sensitivity_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    best = pd.read_csv(NET / "active_passive_best_network_models.csv")
    specs = [
        ("all_trials", "all_clusters", "All clusters\nall trials", 0),
        ("artifact_clean_le_0p2", "all_clusters", "All clusters\nartifact-clean", 1),
        ("all_trials", "good_only", "Good only\nall trials", 2),
        ("artifact_clean_le_0p2", "good_only", "Good only\nartifact-clean", 3),
    ]
    frames: list[pd.DataFrame] = []
    for mode, unit_set, label, order in specs:
        frame = best[(best.analysis_mode == mode) & (best.unit_set == unit_set)][
            ["subject", "run", "nested_single_unit_auc", "auc"]
        ].copy()
        frame["analysis_mode"] = mode
        frame["unit_set"] = unit_set
        frame["spec_label"] = label
        frame["spec_order"] = order
        frames.append(frame)
    run_level = pd.concat(frames, ignore_index=True)
    patient_level = (
        run_level.groupby(
            ["subject", "analysis_mode", "unit_set", "spec_label", "spec_order"],
            as_index=False,
        )[["nested_single_unit_auc", "auc"]]
        .median()
        .rename(columns={"auc": "descriptive_best_population_auc"})
        .sort_values(["spec_order", "subject"])
    )
    return run_level, patient_level


def load_weight_data() -> pd.DataFrame:
    weights = pd.read_csv(NETWORK_REPORT / "active_passive_go_style_unit_nodes.csv")
    required = {"unit_id", "coef", "abs_coef", "depth_um", "ks_label"}
    if not required.issubset(weights.columns):
        raise ValueError(f"Weight table missing columns: {sorted(required - set(weights.columns))}")
    return weights.dropna(subset=["coef", "depth_um"]).sort_values("depth_um").copy()


def load_top_unit_ablation_data() -> dict[str, pd.DataFrame]:
    tables = {
        "ranked_run_values": pd.read_csv(ABLATION / "ablation_ranked_run_values.csv"),
        "random_run_draws": pd.read_csv(ABLATION / "ablation_random_run_draws.csv"),
        "patient_values": pd.read_csv(ABLATION / "ablation_patient_values.csv"),
        "random_patient_draws": pd.read_csv(
            ABLATION / "ablation_random_patient_draws.csv"
        ),
        "group_summary": pd.read_csv(ABLATION / "ablation_group_summary.csv"),
    }
    patient = tables["patient_values"]
    summary = tables["group_summary"].sort_values("remove_order")
    if patient.subject.nunique() != 5:
        raise ValueError(
            f"Expected five ablation patients, found {patient.subject.nunique()}"
        )
    if summary.remove_label.astype(int).tolist() != [0, 1, 2, 5, 10]:
        raise ValueError("Top-unit ablation removal levels changed")
    return tables


def load_sentence_pair_transfer_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    patient = pd.read_csv(TRANSFER / "sentence_pair_transfer_patient_values.csv")
    inference = pd.read_csv(TRANSFER / "sentence_pair_transfer_inference.csv")
    expected_models = {"nested_single_unit", "fixed_top5_population"}
    if set(patient.model.unique()) != expected_models:
        raise ValueError("Sentence-pair transfer models changed")
    if patient.subject.nunique() != 5:
        raise ValueError(
            f"Expected five transfer patients, found {patient.subject.nunique()}"
        )
    return patient, inference


def load_archetype_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(ARCHETYPE_REPORT / "interpretable_cluster_nodes.csv")
    edges = pd.read_csv(ARCHETYPE_REPORT / "interpretable_cluster_edges.csv")
    profiles = pd.read_csv(ARCHETYPE_REPORT / "interpretable_cluster_mean_profiles_1ms.csv")
    counts = nodes.cluster.value_counts().to_dict()
    expected = {"A-E": 36, "A-M": 21, "A-L": 23, "P-E": 30, "P-M": 37, "P-L": 25}
    if counts != expected:
        raise ValueError(f"Archetype counts changed: {counts}")
    return nodes, edges, profiles


def draw_population_gain(ax: plt.Axes, patient_endpoints: pd.DataFrame) -> None:
    panel_label(ax, "a")
    for row in patient_endpoints.itertuples(index=False):
        ax.plot(
            [0, 1],
            [row.nested_single_unit_auc, row.descriptive_best_population_auc],
            color=LIGHT_GRAY,
            lw=1.0,
            zorder=1,
        )
    ax.scatter(
        np.zeros(len(patient_endpoints)),
        patient_endpoints.nested_single_unit_auc,
        s=23,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=0.8,
        zorder=2,
    )
    ax.scatter(
        np.ones(len(patient_endpoints)),
        patient_endpoints.descriptive_best_population_auc,
        s=25,
        color=ACTIVE,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    ax.axhline(0.5, color=GRAY, ls="--", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["nested\nsingle", "best\npopulation*"])
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(0.30, 0.84)
    ax.set_ylabel("AUC")
    ax.set_title("Population readout gain", loc="left", fontweight="bold", pad=4)
    median_best = float(patient_endpoints.descriptive_best_population_auc.median())
    median_gain = float(patient_endpoints.auc_gain_vs_nested_single_unit.median())
    ax.text(
        0.02,
        0.97,
        f"median best = {median_best:.2f}\nmedian gain = {median_gain:.2f}\nn = 5 patients",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.text(0.5, -0.31, "* descriptive selection", transform=ax.transAxes, ha="center", va="top")


def draw_sentence_pair_transfer(
    ax: plt.Axes,
    patient_transfer: pd.DataFrame,
    transfer_inference: pd.DataFrame,
) -> None:
    panel_label(ax, "b")
    wide = (
        patient_transfer.pivot(index="subject", columns="model", values="auc")
        .sort_index()[["nested_single_unit", "fixed_top5_population"]]
    )
    matrix = wide.to_numpy(float)
    display = np.vstack([matrix, np.median(matrix, axis=0, keepdims=True)])
    ax.imshow(
        display,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=0.86),
        aspect="auto",
        interpolation="nearest",
    )
    for row_index in range(display.shape[0]):
        for column_index in range(display.shape[1]):
            value = display[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value - 0.5) > 0.17 else DARK,
                fontsize=PT,
                fontweight="bold" if row_index == display.shape[0] - 1 else "normal",
            )
    ax.axhline(4.5, color="white", lw=1.2)
    p_lookup = transfer_inference.set_index("comparison")[
        "group_permutation_p_greater"
    ]
    ax.set_xticks(
        [0, 1],
        [
            f"single unit\nP={p_lookup['nested_single_unit']:.3f}",
            f"top-5 population\nP={p_lookup['fixed_top5_population']:.3f}",
        ],
    )
    ax.set_yticks(np.arange(6), [*wide.index.tolist(), "median"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Transfer to unseen sentence pairs", loc="left", fontweight="bold", pad=4)


def draw_weight_depth(ax: plt.Axes, weights: pd.DataFrame) -> None:
    panel_label(ax, "c")
    scale = max(float(np.nanpercentile(weights.abs_coef, 98)), 1e-6)
    sizes = 10 + 85 * np.clip(weights.abs_coef.to_numpy(float) / scale, 0, 1)
    colors = np.where(weights.coef.to_numpy(float) >= 0, ACTIVE, PASSIVE)
    ax.scatter(
        weights.coef,
        weights.depth_um,
        s=sizes,
        c=colors,
        alpha=0.73,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axvline(0, color=GRAY, ls="--", lw=0.8)
    for row in weights.nlargest(4, "abs_coef").itertuples(index=False):
        ax.text(row.coef, row.depth_um, f" {int(row.unit_id)}", ha="left", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("standardized coefficient\n(active +; passive −)")
    ax.set_ylabel("probe depth (µm)")
    ax.set_title("Example-run decoder weights", loc="left", fontweight="bold", pad=4)


def draw_sensitivity(ax: plt.Axes, patient_sensitivity: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.14, y=1.09)
    specs = (
        patient_sensitivity[["spec_order", "spec_label"]]
        .drop_duplicates()
        .sort_values("spec_order")
    )
    for subject, frame in patient_sensitivity.groupby("subject"):
        frame = frame.sort_values("spec_order")
        color = PATIENT_COLORS[str(subject)]
        for row in frame.itertuples(index=False):
            x = float(row.spec_order)
            ax.plot(
                [x - 0.11, x + 0.11],
                [row.nested_single_unit_auc, row.descriptive_best_population_auc],
                color=color,
                lw=0.75,
                alpha=0.65,
            )
            ax.scatter(
                [x - 0.11, x + 0.11],
                [row.nested_single_unit_auc, row.descriptive_best_population_auc],
                s=13,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.8,
                zorder=3,
            )
    nested_medians = (
        patient_sensitivity.groupby("spec_order").nested_single_unit_auc.median().reindex(range(4))
    )
    best_medians = (
        patient_sensitivity.groupby("spec_order").descriptive_best_population_auc.median().reindex(range(4))
    )
    x = np.arange(4)
    ax.plot(x - 0.11, nested_medians, color=GRAY, marker="o", ms=5.0, lw=1.4, label="single unit")
    ax.plot(x + 0.11, best_medians, color=ACTIVE, marker="o", ms=5.0, lw=1.4, label="best pop.*")
    ax.axhline(0.5, color=GRAY, ls="--", lw=0.8)
    ax.set_xticks(specs.spec_order, specs.spec_label)
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(0.20, 0.86)
    ax.set_ylabel("AUC")
    ax.set_title("QC and artifact sensitivity", loc="left", fontweight="bold", pad=4)
    ax.legend(
        loc="lower right",
        ncol=2,
        handlelength=1.1,
        columnspacing=0.8,
        handletextpad=0.4,
    )


def draw_top_unit_ablation(
    ax: plt.Axes,
    patient_values: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    panel_label(ax, "e", x=-0.14, y=1.09)
    removal_labels = [1, 2, 5, 10]
    wide = (
        patient_values[patient_values.remove_label.isin(removal_labels)]
        .pivot(
            index="subject",
            columns="remove_label",
            values="random_minus_ranked_removal_auc",
        )
        .sort_index()[removal_labels]
    )
    medians = (
        summary.set_index("remove_label")
        .loc[removal_labels, "random_minus_ranked_patient_median_auc"]
        .to_numpy(float)
    )
    display = np.vstack([wide.to_numpy(float), medians[np.newaxis, :]])
    ax.imshow(
        display,
        cmap="PuOr",
        norm=TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.22),
        aspect="auto",
        interpolation="nearest",
    )
    for row_index in range(display.shape[0]):
        for column_index in range(display.shape[1]):
            value = display[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.13 else DARK,
                fontsize=PT,
                fontweight="bold" if row_index == display.shape[0] - 1 else "normal",
            )
    ax.axhline(4.5, color="white", lw=1.2)
    p_values = (
        summary.set_index("remove_label")
        .loc[removal_labels, "random_minus_ranked_exact_sign_flip_p_greater"]
        .to_numpy(float)
    )
    ax.set_xticks(
        np.arange(len(removal_labels)),
        [f"{label}\nP={p_value:.3f}" for label, p_value in zip(removal_labels, p_values)],
    )
    ax.set_yticks(np.arange(6), [*wide.index.tolist(), "median"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("top-ranked units removed\npositive = greater AUC loss than random removal")
    ax.set_title("Top-unit ablation footprint", loc="left", fontweight="bold", pad=4)


def expanded_hull(points: np.ndarray, pad: float = 0.17) -> np.ndarray | None:
    if len(points) < 3:
        return None
    try:
        vertices = ConvexHull(points).vertices
    except Exception:
        return None
    polygon = points[vertices]
    center = polygon.mean(axis=0, keepdims=True)
    return center + (polygon - center) * (1 + pad)


def archetype_group_relations(profiles: pd.DataFrame) -> pd.DataFrame:
    vectors = {
        str(cluster): frame.sort_values("time_ms")[
            "mean_qualifying_masked_contrast_hz"
        ].to_numpy(float)
        for cluster, frame in profiles.groupby("cluster", sort=False)
    }
    rows: list[dict[str, object]] = []
    for index, cluster_a in enumerate(CLUSTER_ORDER):
        for cluster_b in CLUSTER_ORDER[index + 1 :]:
            raw_a = vectors[cluster_a]
            raw_b = vectors[cluster_b]
            aligned_a = raw_a if cluster_a.startswith("A-") else -raw_a
            aligned_b = raw_b if cluster_b.startswith("A-") else -raw_b
            raw_denominator = float(np.linalg.norm(raw_a) * np.linalg.norm(raw_b))
            aligned_denominator = float(
                np.linalg.norm(aligned_a) * np.linalg.norm(aligned_b)
            )
            raw_cosine = (
                float(np.dot(raw_a, raw_b) / raw_denominator)
                if raw_denominator > 0
                else np.nan
            )
            aligned_cosine = (
                float(np.dot(aligned_a, aligned_b) / aligned_denominator)
                if aligned_denominator > 0
                else np.nan
            )
            same_direction = cluster_a[0] == cluster_b[0]
            same_latency = cluster_a[-1] == cluster_b[-1]
            latency_order = {"E": 0, "M": 1, "L": 2}
            adjacent_latency = bool(
                same_direction
                and abs(latency_order[cluster_a[-1]] - latency_order[cluster_b[-1]]) == 1
            )
            if (not same_direction) and same_latency:
                panel_f_relation_type = "vertical_matched_latency"
            elif adjacent_latency:
                panel_f_relation_type = "horizontal_adjacent_latency"
            else:
                panel_f_relation_type = "not_displayed"
            rows.append(
                {
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "raw_profile_cosine": raw_cosine,
                    "sign_aligned_profile_cosine": aligned_cosine,
                    "same_direction": same_direction,
                    "same_latency": same_latency,
                    "adjacent_latency": adjacent_latency,
                    "panel_f_relation_type": panel_f_relation_type,
                    "displayed_in_panel_f": panel_f_relation_type != "not_displayed",
                }
            )
    return pd.DataFrame(rows)


def draw_archetype_network(
    ax: plt.Axes,
    fig: plt.Figure,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    panel_label(ax, "f", x=-0.05, y=1.04)
    node_map = nodes.set_index("node_key")
    centers = nodes.groupby("cluster")[["layout_x", "layout_y"]].mean()
    relations = archetype_group_relations(profiles)

    # Two pale lanes expose the 2 x 3 hierarchy without treating layout as
    # anatomy.  Matched-latency links summarize group-mean temporal profiles.
    ax.axhspan(0.10, 1.48, color=ACTIVE, alpha=0.035, zorder=-4)
    ax.axhspan(-1.48, -0.10, color=PASSIVE, alpha=0.035, zorder=-4)
    ax.text(
        -2.50,
        0.85,
        "A > P",
        color=ACTIVE,
        rotation=90,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.text(
        -2.50,
        -0.85,
        "A < P",
        color=PASSIVE,
        rotation=90,
        ha="center",
        va="center",
        fontweight="bold",
    )
    vertical_relations = relations[
        relations.panel_f_relation_type == "vertical_matched_latency"
    ]
    for row in vertical_relations.itertuples(index=False):
        start = centers.loc[row.cluster_a]
        stop = centers.loc[row.cluster_b]
        similarity = max(0.0, float(row.sign_aligned_profile_cosine))
        x_position = float((start.layout_x + stop.layout_x) / 2)
        ax.plot(
            [x_position, x_position],
            [0.32, -0.32],
            color=PURPLE,
            lw=0.65 + 2.1 * similarity,
            alpha=0.30 + 0.55 * similarity,
            solid_capstyle="round",
            zorder=-1,
        )
        ax.scatter(
            [x_position],
            [0.0],
            s=13 + 28 * similarity,
            facecolor="white",
            edgecolor=PURPLE,
            linewidth=0.65,
            zorder=0,
        )
        ax.text(
            x_position + 0.08,
            0.0,
            f"{similarity:.2f}",
            ha="left",
            va="center",
            color=PURPLE,
            fontsize=PT,
        )
    horizontal_relations = relations[
        relations.panel_f_relation_type == "horizontal_adjacent_latency"
    ]
    for row in horizontal_relations.itertuples(index=False):
        start = centers.loc[row.cluster_a]
        stop = centers.loc[row.cluster_b]
        if float(start.layout_x) > float(stop.layout_x):
            start, stop = stop, start
            left_cluster, right_cluster = row.cluster_b, row.cluster_a
        else:
            left_cluster, right_cluster = row.cluster_a, row.cluster_b
        left_points = nodes[nodes.cluster == left_cluster]
        right_points = nodes[nodes.cluster == right_cluster]
        x_start = float(left_points.layout_x.max()) + 0.035
        x_stop = float(right_points.layout_x.min()) - 0.035
        x_mid = float((x_start + x_stop) / 2)
        y_position = float((start.layout_y + stop.layout_y) / 2)
        similarity = float(row.sign_aligned_profile_cosine)
        magnitude = min(abs(similarity), 1.0)
        link_color = "#4F766F" if similarity >= 0 else ORANGE
        ax.plot(
            [x_start, x_stop],
            [y_position, y_position],
            color=link_color,
            lw=0.65 + 2.1 * magnitude,
            alpha=0.45 + 0.45 * magnitude,
            ls="-" if similarity >= 0 else "--",
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            [x_mid],
            [y_position],
            s=13 + 28 * magnitude,
            facecolor="white",
            edgecolor=link_color,
            linewidth=0.65,
            zorder=3,
        )
        label_offset = 0.14 if y_position > 0 else -0.14
        ax.text(
            x_mid,
            y_position + label_offset,
            f"{similarity:.2f}",
            ha="center",
            va="center",
            color=link_color,
            fontsize=PT,
            zorder=4,
        )
    for cluster in CLUSTER_ORDER:
        frame = nodes[nodes.cluster == cluster]
        points = frame[["layout_x", "layout_y"]].to_numpy(float)
        polygon = expanded_hull(points)
        color = CLUSTER_INFO[cluster]["color"]
        if polygon is not None:
            ax.add_patch(Polygon(polygon, closed=True, facecolor=color, edgecolor="none", alpha=0.11, zorder=0))
        center = points.mean(axis=0)
        is_active = cluster.startswith("A-")
        label_offset = 0.34 if is_active else -0.40
        ax.text(
            center[0],
            center[1] + label_offset,
            f"{CLUSTER_INFO[cluster]['short'].split()[-1]}\nn={len(frame)}",
            ha="center",
            va="bottom" if is_active else "top",
            fontweight="bold",
        )
    for edge in edges.itertuples(index=False):
        start = node_map.loc[edge.node_a]
        stop = node_map.loc[edge.node_b]
        similarity = float(edge.cosine_similarity)
        ax.plot(
            [start.layout_x, stop.layout_x],
            [start.layout_y, stop.layout_y],
            color="#566763",
            lw=0.45 + 1.0 * similarity,
            alpha=0.30 + 0.40 * similarity,
            zorder=1,
        )
    duration = nodes.corrected_longest_segment_ms.to_numpy(float)
    sizes = 7 + 32 * np.sqrt(duration / max(float(duration.max()), 1.0))
    effect = nodes.direction_consistent_mean_contrast_hz.to_numpy(float)
    limit = max(0.5, float(np.nanpercentile(np.abs(effect), 98)))
    scatter = ax.scatter(
        nodes.layout_x,
        nodes.layout_y,
        s=sizes,
        c=effect,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        edgecolors="none",
        alpha=0.95,
        zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.026, pad=0.012)
    colorbar.ax.set_title("A − P\n(Hz)", pad=2)
    colorbar.ax.tick_params(labelsize=PT)
    ax.set_xlim(-2.72, 2.75)
    ax.set_ylim(-1.70, 1.73)
    ax.set_axis_off()
    ax.set_title("Relations among six response groups", loc="left", fontweight="bold", pad=4)
    ax.text(
        0.01,
        -0.025,
        "Internal edges: unit-profile similarity.\n"
        "Horizontal links: adjacent-latency group cosine.\n"
        "Vertical links: matched-latency cross-direction group cosine.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=GRAY,
        fontsize=PT,
    )


def draw_profile_axis(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    clusters: tuple[str, str, str],
    title: str,
    show_xlabel: bool,
) -> None:
    spans = ((0, 267), (267, 533), (533, 800))
    for index, (start, stop) in enumerate(spans):
        ax.axvspan(start, stop, color="#F2F4F3" if index != 1 else "#E9EEEC", alpha=0.78, zorder=0)
    for cluster in clusters:
        frame = profiles[profiles.cluster == cluster]
        x = frame.time_ms.to_numpy(float)
        mean = frame.mean_qualifying_masked_contrast_hz.to_numpy(float)
        sem = frame.sem_qualifying_masked_contrast_hz.to_numpy(float)
        color = CLUSTER_INFO[cluster]["color"]
        n_units = int(frame.n_units.iloc[0])
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.13, lw=0)
        ax.plot(x, mean, color=color, lw=1.35, label=f"{CLUSTER_INFO[cluster]['timing']} n={n_units}")
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xlim(0, 799)
    ax.set_ylabel("A − P (Hz)")
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.legend(loc="upper right", ncol=3, handlelength=1.0, columnspacing=0.7, handletextpad=0.3)
    if show_xlabel:
        ax.set_xlabel("time from corrected\nfunction-word onset (ms)")
    else:
        ax.tick_params(labelbottom=False)


def draw_profiles(ax_active: plt.Axes, ax_passive: plt.Axes, profiles: pd.DataFrame) -> None:
    panel_label(ax_active, "g", x=-0.27, y=1.08)
    draw_profile_axis(ax_active, profiles, ("A-E", "A-M", "A-L"), "Active-dominant", False)
    draw_profile_axis(ax_passive, profiles, ("P-E", "P-M", "P-L"), "Passive-dominant", True)


def draw_figure() -> tuple[plt.Figure, dict[str, object]]:
    run_endpoints, patient_endpoints = load_endpoint_data()
    sensitivity_runs, sensitivity_patients = load_sensitivity_data()
    weights = load_weight_data()
    ablation_tables = load_top_unit_ablation_data()
    transfer_patients, transfer_inference = load_sentence_pair_transfer_data()
    nodes, edges, profiles = load_archetype_data()
    group_relations = archetype_group_relations(profiles)
    matched_relations = group_relations[
        group_relations.panel_f_relation_type == "vertical_matched_latency"
    ].set_index("cluster_a")
    horizontal_relations = group_relations[
        group_relations.panel_f_relation_type == "horizontal_adjacent_latency"
    ].set_index(["cluster_a", "cluster_b"])
    if patient_endpoints.shape[0] != 5:
        raise ValueError(f"Expected five patients, found {patient_endpoints.shape[0]}")

    fig = plt.figure(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=[0.92, 0.72, 1.34],
        left=0.070,
        right=0.975,
        bottom=0.085,
        top=0.970,
        wspace=0.95,
        hspace=0.68,
    )
    ax_gain = fig.add_subplot(grid[0, 0:2])
    ax_checks = fig.add_subplot(grid[0, 2:4])
    ax_weights = fig.add_subplot(grid[0, 4:6])
    ax_sensitivity = fig.add_subplot(grid[1, 0:3])
    ax_generalization = fig.add_subplot(grid[1, 3:6])
    bottom_grid = grid[2, :].subgridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.42)
    ax_network = fig.add_subplot(bottom_grid[0, 0])
    profile_grid = bottom_grid[0, 1].subgridspec(2, 1, hspace=0.36)
    ax_active = fig.add_subplot(profile_grid[0, 0])
    ax_passive = fig.add_subplot(profile_grid[1, 0])

    draw_population_gain(ax_gain, patient_endpoints)
    draw_sentence_pair_transfer(ax_checks, transfer_patients, transfer_inference)
    draw_weight_depth(ax_weights, weights)
    draw_sensitivity(ax_sensitivity, sensitivity_patients)
    draw_top_unit_ablation(
        ax_generalization,
        ablation_tables["patient_values"],
        ablation_tables["group_summary"],
    )
    draw_archetype_network(ax_network, fig, nodes, edges, profiles)
    draw_profiles(ax_active, ax_passive, profiles)

    metadata: dict[str, object] = {
        "figure": "Fig. 2 revised",
        "core_conclusion": (
            "Construction-linked information is recoverable from a compact, train-selected population "
            "readout, transfers to unseen sentence pairs at the population but not single-unit level, "
            "depends disproportionately on a small training-ranked subset of units, remains visible "
            "across unit-quality and artifact-filtering specifications, and is expressed through "
            "temporally heterogeneous response archetypes."
        ),
        "width_mm": WIDTH_MM,
        "height_mm": HEIGHT_MM,
        "font_size_pt": PT,
        "n_patients": int(patient_endpoints.shape[0]),
        "n_primary_decoding_runs": int(run_endpoints.shape[0]),
        "n_example_weighted_units": int(len(weights)),
        "n_archetype_unit_run_entries": int(len(nodes)),
        "n_archetype_edges": int(len(edges)),
        "archetype_matched_latency_sign_aligned_cosine": {
            "early": float(
                matched_relations.loc["A-E", "sign_aligned_profile_cosine"]
            ),
            "middle": float(
                matched_relations.loc["A-M", "sign_aligned_profile_cosine"]
            ),
            "late": float(
                matched_relations.loc["A-L", "sign_aligned_profile_cosine"]
            ),
        },
        "archetype_adjacent_latency_sign_aligned_cosine": {
            "active_early_middle": float(
                horizontal_relations.loc[("A-E", "A-M"), "sign_aligned_profile_cosine"]
            ),
            "active_middle_late": float(
                horizontal_relations.loc[("A-M", "A-L"), "sign_aligned_profile_cosine"]
            ),
            "passive_early_middle": float(
                horizontal_relations.loc[("P-E", "P-M"), "sign_aligned_profile_cosine"]
            ),
            "passive_middle_late": float(
                horizontal_relations.loc[("P-M", "P-L"), "sign_aligned_profile_cosine"]
            ),
        },
        "patient_median_nested_single_auc": float(patient_endpoints.nested_single_unit_auc.median()),
        "patient_median_fixed_top5_auc": float(patient_endpoints.fixed_top5_auc.median()),
        "patient_median_descriptive_best_population_auc": float(
            patient_endpoints.descriptive_best_population_auc.median()
        ),
        "patient_median_descriptive_gain": float(
            patient_endpoints.auc_gain_vs_nested_single_unit.median()
        ),
        "patient_median_leave_pair_out_auc": float(patient_endpoints.auc_leave_pair_out.median()),
        "sentence_pair_transfer_nested_single_unit_auc": float(
            transfer_inference.set_index("comparison").loc[
                "nested_single_unit", "patient_median"
            ]
        ),
        "sentence_pair_transfer_nested_single_unit_permutation_p": float(
            transfer_inference.set_index("comparison").loc[
                "nested_single_unit", "group_permutation_p_greater"
            ]
        ),
        "sentence_pair_transfer_top5_population_auc": float(
            transfer_inference.set_index("comparison").loc[
                "fixed_top5_population", "patient_median"
            ]
        ),
        "sentence_pair_transfer_top5_population_permutation_p": float(
            transfer_inference.set_index("comparison").loc[
                "fixed_top5_population", "group_permutation_p_greater"
            ]
        ),
        "n_top_unit_ablation_runs": int(
            ablation_tables["ranked_run_values"][["subject", "run"]]
            .drop_duplicates()
            .shape[0]
        ),
        "top_unit_ablation_remove5_ranked_patient_median_auc": float(
            ablation_tables["group_summary"].set_index("remove_label").loc[
                5, "ranked_removal_patient_median_auc"
            ]
        ),
        "top_unit_ablation_remove5_random_patient_median_auc": float(
            ablation_tables["group_summary"].set_index("remove_label").loc[
                5, "random_removal_patient_median_auc"
            ]
        ),
        "top_unit_ablation_remove5_random_minus_ranked_patient_median_auc": float(
            ablation_tables["group_summary"].set_index("remove_label").loc[
                5, "random_minus_ranked_patient_median_auc"
            ]
        ),
        "top_unit_ablation_remove5_exact_sign_flip_p_greater": float(
            ablation_tables["group_summary"].set_index("remove_label").loc[
                5, "random_minus_ranked_exact_sign_flip_p_greater"
            ]
        ),
        "reviewer_risks": [
            "Active/passive construction and function-word identity/acoustics covary in the current design.",
            "Best-population model selection is descriptive rather than confirmatory.",
            "The weight-depth panel is a representative full-data fit from one run.",
            "Top-unit ablation is computational and does not establish causal necessity of neurons.",
            "Ablation ranking is recomputed within each training fold; the analysis is exploratory.",
            "Archetype nodes are selected unit/run entries; their similarity edges are descriptive and non-anatomical.",
            "Patients, not runs or units, are the independent biological units for cross-patient summaries.",
            "Held-sentence-pair transfer is supported for the fixed top-5 population, not for the nested single-unit readout.",
        ],
    }
    return fig, metadata


def export_source_data(metadata: dict[str, object]) -> None:
    run_endpoints, patient_endpoints = load_endpoint_data()
    sensitivity_runs, sensitivity_patients = load_sensitivity_data()
    weights = load_weight_data()
    ablation_tables = load_top_unit_ablation_data()
    transfer_patients, transfer_inference = load_sentence_pair_transfer_data()
    nodes, edges, profiles = load_archetype_data()
    group_relations = archetype_group_relations(profiles)
    for stale_name in (
        "panel_e_archetype_nodes.csv",
        "panel_e_archetype_edges.csv",
        "panel_f_archetype_mean_profiles.csv",
        "panel_e_held_marker_patient_values.csv",
        "panel_e_held_marker_patient_inference.csv",
        "panel_e_population_size_ranked_run_values.csv",
        "panel_e_population_size_random_run_draws.csv",
        "panel_e_population_size_patient_values.csv",
        "panel_e_population_size_random_patient_draws.csv",
        "panel_e_population_size_group_summary.csv",
    ):
        (SOURCE / stale_name).unlink(missing_ok=True)
    run_endpoints.to_csv(SOURCE / "panels_a_b_run_endpoints.csv", index=False)
    patient_endpoints.to_csv(SOURCE / "panels_a_b_patient_medians.csv", index=False)
    transfer_patients.to_csv(
        SOURCE / "panel_b_sentence_pair_transfer_patient_values.csv", index=False
    )
    transfer_inference.to_csv(
        SOURCE / "panel_b_sentence_pair_transfer_inference.csv", index=False
    )
    sensitivity_runs.to_csv(SOURCE / "panel_d_sensitivity_run_values.csv", index=False)
    sensitivity_patients.to_csv(SOURCE / "panel_d_sensitivity_patient_medians.csv", index=False)
    weights.to_csv(SOURCE / "panel_c_example_decoder_weights.csv", index=False)
    ablation_tables["ranked_run_values"].to_csv(
        SOURCE / "panel_e_top_unit_ablation_ranked_run_values.csv", index=False
    )
    ablation_tables["random_run_draws"].to_csv(
        SOURCE / "panel_e_top_unit_ablation_random_run_draws.csv", index=False
    )
    ablation_tables["patient_values"].to_csv(
        SOURCE / "panel_e_top_unit_ablation_patient_values.csv", index=False
    )
    ablation_tables["random_patient_draws"].to_csv(
        SOURCE / "panel_e_top_unit_ablation_random_patient_draws.csv", index=False
    )
    ablation_tables["group_summary"].to_csv(
        SOURCE / "panel_e_top_unit_ablation_group_summary.csv", index=False
    )
    nodes.to_csv(SOURCE / "panel_f_archetype_nodes.csv", index=False)
    edges.to_csv(SOURCE / "panel_f_archetype_edges.csv", index=False)
    group_relations.to_csv(SOURCE / "panel_f_group_relations.csv", index=False)
    profiles.to_csv(SOURCE / "panel_g_archetype_mean_profiles.csv", index=False)
    (SOURCE / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def export_figure(fig: plt.Figure) -> dict[str, str]:
    stem = OUT / "fig2_population_180mm_revised"
    outputs = {
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
        "tiff": str(stem.with_suffix(".tiff")),
    }
    fig.savefig(outputs["png"], dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(outputs["pdf"], facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(outputs["svg"], facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(
        outputs["tiff"],
        dpi=600,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(outputs["tiff"]) as source:
        if source.mode == "RGBA":
            rgb = Image.new("RGB", source.size, "white")
            rgb.paste(source, mask=source.getchannel("A"))
        else:
            rgb = source.convert("RGB")
        rgb.save(outputs["tiff"], dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)
    return outputs


def write_legend(metadata: dict[str, object]) -> Path:
    similarities = metadata["archetype_matched_latency_sign_aligned_cosine"]
    horizontal_similarities = metadata[
        "archetype_adjacent_latency_sign_aligned_cosine"
    ]
    sections = [
        "**Fig. 2 | Distributed construction-linked information in human cortical populations.**",
        (
            f"**a,** Nested single-unit and descriptively selected best-population AUC after "
            f"within-patient run aggregation (n = {metadata['n_patients']} patients); lines link "
            f"patients. Median best-population AUC={metadata['patient_median_descriptive_best_population_auc']:.2f} "
            f"and median gain={metadata['patient_median_descriptive_gain']:.2f}."
        ),
        (
            "**b,** Leave-one-sentence-pair-out transfer using identical folds. Cells show patient "
            f"AUCs and their median. Nested single-unit: AUC={metadata['sentence_pair_transfer_nested_single_unit_auc']:.3f}, "
            f"permutation P={metadata['sentence_pair_transfer_nested_single_unit_permutation_p']:.3f}; "
            f"training-fold-ranked top-5 population: AUC={metadata['sentence_pair_transfer_top5_population_auc']:.2f}, "
            f"P={metadata['sentence_pair_transfer_top5_population_permutation_p']:.3f}."
        ),
        (
            "**c,** Standardized coefficients from one representative full-data population fit "
            "against probe depth; area encodes magnitude and sign encodes readout direction."
        ),
        (
            "**d,** Patient AUCs and medians across unit-quality and artifact-filtering specifications; "
            "best-population selection is descriptive."
        ),
        (
            "**e,** Top-unit ablation heatmap. Each cell is AUC after random removal minus AUC after "
            "removal of the same number of training-fold-ranked units; positive values indicate "
            "greater disruption by ranked removal. Ranking and removal were repeated within each "
            f"training fold of 20 repeated stratified fivefold cross-validation ({metadata['n_top_unit_ablation_runs']} "
            f"runs; n = {metadata['n_patients']} patients). After removing five units, ranked- and "
            f"random-removal median AUCs were {metadata['top_unit_ablation_remove5_ranked_patient_median_auc']:.2f} "
            f"and {metadata['top_unit_ablation_remove5_random_patient_median_auc']:.2f} "
            f"(difference={metadata['top_unit_ablation_remove5_random_minus_ranked_patient_median_auc']:.2f}; "
            f"one-sided exact sign-flip P={metadata['top_unit_ablation_remove5_exact_sign_flip_p_greater']:.3f}). "
            "This exploratory computational ablation does not imply causal neuronal necessity."
        ),
        (
            f"**f,** Six response groups cross contrast direction with early, middle or late peak "
            f"latency ({metadata['n_archetype_unit_run_entries']} unit/run entries). Lanes group "
            "active- and passive-dominant types; within-lane edges encode unit-profile similarity. "
            "Horizontal links encode adjacent-latency group-profile cosine similarity "
            f"(active E–M={horizontal_similarities['active_early_middle']:.2f}, "
            f"M–L={horizontal_similarities['active_middle_late']:.2f}; passive "
            f"E–M={horizontal_similarities['passive_early_middle']:.2f}, "
            f"M–L={horizontal_similarities['passive_middle_late']:.2f}). Vertical links encode "
            "sign-aligned matched-latency cosine similarity "
            f"(early={similarities['early']:.2f}, middle={similarities['middle']:.2f}, "
            f"late={similarities['late']:.2f})."
        ),
        "**g,** Mean qualifying contrast profiles ± s.e.m. for the six groups.",
        (
            "Active/passive construction, function-word identity and associated acoustics covary; "
            "the panels therefore do not establish abstract voice coding. Source data are provided "
            "with the figure."
        ),
    ]
    legend = " ".join(sections) + "\n"
    path = OUT / "fig2_population_180mm_revised_legend.md"
    path.write_text(legend, encoding="utf-8")
    return path


def main() -> None:
    fig, metadata = draw_figure()
    export_source_data(metadata)
    outputs = export_figure(fig)
    legend_path = write_legend(metadata)
    manifest = {
        "outputs": outputs,
        "legend": str(legend_path),
        "source_data_directory": str(SOURCE),
        "metadata": metadata,
    }
    manifest_path = OUT / "fig2_population_180mm_revised_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
