#!/usr/bin/env python3
"""Make an interpretable GO-style network for the corrected active/passive screen.

The six clusters are deterministic response archetypes rather than opaque
community IDs: corrected direction (Active > Passive or Active < Passive)
crossed with the latency tercile of the largest condition difference inside
the unit's qualifying significant segments (Early, Middle, Late).

Edges are descriptive temporal-profile similarity.  Within each archetype,
every node is linked to its two nearest cosine-similarity neighbors, and a
maximum spanning tree is added to guarantee a connected, readable module.
This is a response-phenotype map, not anatomical or causal connectivity.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity

import analyze_active_passive_corrected_timepoint_screen_20260722 as corrected


SCREEN_DIR = ROOT / "outputs/active_passive_corrected_timepoint_screen_20260722"
SCREEN_PATH = SCREEN_DIR / "unit_crosswalk.csv"
SEGMENT_PATH = SCREEN_DIR / "qualifying_segments.csv"
OUT_DIR = ROOT / "reports/active_passive_go_style_interpretable_20260723"

ACTIVE = "#2C7FB8"
PASSIVE = "#D95F4C"
INK = "#17322F"
GRAY = "#6B7B78"
GRID = "#DDE5E2"
WHITE = "#FFFFFF"
K_NEIGHBORS = 2

RUN_SHORT = {
    "gz001t_g0_actpas_run1": "gz001 · run1",
    "gz002_g0_actpas_run2": "gz002 · run2",
    "gz003t2_g0_actpas_run1": "gz003 · run1",
    "gz003t2_g0_actpas_run4": "gz003 · run4",
    "gz004t1_g0_actpas_run0": "gz004 · run0",
    "gz004t1_g3_actpas_run3": "gz004 · run3",
    "gz005t0_g0_actpas_run0": "gz005 · run0",
    "gz005t0_g3_actpas_run3": "gz005 · run3",
}

CLUSTER_ORDER = ("A-E", "A-M", "A-L", "P-E", "P-M", "P-L")
CLUSTER_INFO = {
    "A-E": {"label": "Active > Passive · Early", "short": "A>P Early", "direction": "active_gt_passive", "timing": "Early", "color": "#9ECAE1"},
    "A-M": {"label": "Active > Passive · Middle", "short": "A>P Middle", "direction": "active_gt_passive", "timing": "Middle", "color": "#4292C6"},
    "A-L": {"label": "Active > Passive · Late", "short": "A>P Late", "direction": "active_gt_passive", "timing": "Late", "color": "#08519C"},
    "P-E": {"label": "Active < Passive · Early", "short": "A<P Early", "direction": "passive_gt_active", "timing": "Early", "color": "#FCAE91"},
    "P-M": {"label": "Active < Passive · Middle", "short": "A<P Middle", "direction": "passive_gt_active", "timing": "Middle", "color": "#FB6A4A"},
    "P-L": {"label": "Active < Passive · Late", "short": "A<P Late", "direction": "passive_gt_active", "timing": "Late", "color": "#CB181D"},
}


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


def load_registered_screen() -> tuple[pd.DataFrame, pd.DataFrame]:
    screen = pd.read_csv(SCREEN_PATH)
    screen["corrected_contiguous_gt5ms"] = screen["corrected_contiguous_gt5ms"].astype(bool)
    selected = screen.loc[screen["corrected_contiguous_gt5ms"]].copy()
    segments = pd.read_csv(SEGMENT_PATH)
    segments = segments.loc[segments["alignment"].eq("corrected")].copy()
    if len(screen) != 1782 or len(selected) != 172:
        raise ValueError(f"registered screen mismatch: all={len(screen)}, selected={len(selected)}")
    if selected["corrected_direction"].value_counts().to_dict() != {
        "passive_gt_active": 92,
        "active_gt_passive": 80,
    }:
        raise ValueError("registered direction counts changed")
    return selected, segments


def timing_class(peak_ms: int) -> tuple[str, str]:
    if peak_ms < 267:
        return "E", "Early"
    if peak_ms < 533:
        return "M", "Middle"
    return "L", "Late"


def extract_profiles(selected: pd.DataFrame, segments: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    node_rows: list[dict[str, object]] = []
    raw_profiles: list[np.ndarray] = []
    masked_profiles: list[np.ndarray] = []
    for run in corrected.RUN_ORDER:
        run_selected = selected.loc[selected["run"].eq(run)].copy()
        if run_selected.empty:
            continue
        subject = str(run_selected["subject"].iloc[0])
        trials = corrected.load_trials(subject, run)
        archive = np.load(
            corrected._task_dir(subject) / run / "firing_rate_1000Hz/firing_rate_1000Hz_all_clusters.npz",
            allow_pickle=True,
        )
        unit_ids = archive["unit_ids"].astype(int)
        firing_rate = archive["firing_rate_hz"].astype(np.float32, copy=False)
        epochs = corrected.extract_smoothed_epochs(
            firing_rate,
            trials["corrected_event_bin"].to_numpy(int),
        )
        condition = trials["condition"].astype(str).to_numpy()
        response = epochs[:, :, 1000:1800]
        difference = response[condition == "active"].mean(axis=0) - response[condition == "passive"].mean(axis=0)
        unit_to_column = {int(unit): index for index, unit in enumerate(unit_ids)}

        for row in run_selected.itertuples(index=False):
            unit_id = int(row.unit_id)
            profile = difference[unit_to_column[unit_id]].astype(float)
            mask = np.zeros(800, dtype=bool)
            unit_segments = segments.loc[
                segments["subject"].eq(subject)
                & segments["run"].eq(run)
                & segments["unit_id"].eq(unit_id)
            ]
            for segment in unit_segments.itertuples(index=False):
                mask[int(segment.start_ms) : int(segment.stop_ms_exclusive)] = True
            if not mask.any():
                raise ValueError(f"selected unit lacks qualifying segment: {subject} {run} {unit_id}")
            masked = np.where(mask, profile, 0.0)
            peak_ms = int(np.flatnonzero(mask)[np.argmax(np.abs(profile[mask]))])
            timing_code, timing_label = timing_class(peak_ms)
            direction_code = "A" if row.corrected_direction == "active_gt_passive" else "P"
            cluster = f"{direction_code}-{timing_code}"
            direction_consistent = mask & (
                (profile > 0) if row.corrected_direction == "active_gt_passive" else (profile < 0)
            )
            if not direction_consistent.any():
                raise ValueError(f"no direction-consistent qualifying bins: {subject} {run} {unit_id}")
            node_rows.append(
                {
                    **row._asdict(),
                    "cluster": cluster,
                    "cluster_label": CLUSTER_INFO[cluster]["label"],
                    "peak_latency_class": timing_label,
                    "qualifying_peak_ms": peak_ms,
                    "qualifying_peak_active_minus_passive_hz": float(profile[peak_ms]),
                    "direction_consistent_mean_contrast_hz": float(profile[direction_consistent].mean()),
                    "total_qualifying_ms": int(mask.sum()),
                }
            )
            raw_profiles.append(profile)
            masked_profiles.append(masked)
    nodes = pd.DataFrame(node_rows)
    raw = np.stack(raw_profiles)
    masked = np.stack(masked_profiles)
    nodes["cluster"] = pd.Categorical(nodes["cluster"], categories=CLUSTER_ORDER, ordered=True)
    nodes = nodes.sort_values(["cluster", "subject", "run", "unit_id"]).reset_index(drop=True)
    # Reorder the profile matrices to match the sorted node table.
    original_keys = [
        (str(row["subject"]), str(row["run"]), int(row["unit_id"]))
        for row in node_rows
    ]
    key_to_index = {key: index for index, key in enumerate(original_keys)}
    order = [key_to_index[(str(row.subject), str(row.run), int(row.unit_id))] for row in nodes.itertuples(index=False)]
    return nodes, raw[order], masked[order]


def build_similarity_edges(nodes: pd.DataFrame, masked_profiles: np.ndarray) -> pd.DataFrame:
    edge_map: dict[tuple[int, int], dict[str, object]] = {}
    node_key = [f"{row.subject}|{row.run}|{int(row.unit_id)}" for row in nodes.itertuples(index=False)]
    for cluster in CLUSTER_ORDER:
        indices = np.flatnonzero(nodes["cluster"].astype(str).eq(cluster).to_numpy())
        if len(indices) < 2:
            continue
        similarity = cosine_similarity(masked_profiles[indices])
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(similarity, -np.inf)

        complete = nx.Graph()
        complete.add_nodes_from(range(len(indices)))
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                complete.add_edge(i, j, weight=float(max(similarity[i, j], 0.0)))
        spanning = nx.maximum_spanning_tree(complete, weight="weight")

        selected_pairs: dict[tuple[int, int], set[str]] = {}
        for i in range(len(indices)):
            neighbors = np.argsort(similarity[i])[::-1][: min(K_NEIGHBORS, len(indices) - 1)]
            for j in neighbors:
                pair = tuple(sorted((int(i), int(j))))
                selected_pairs.setdefault(pair, set()).add("top2_neighbor")
        for i, j in spanning.edges():
            pair = tuple(sorted((int(i), int(j))))
            selected_pairs.setdefault(pair, set()).add("maximum_spanning_tree")

        for (local_i, local_j), reasons in selected_pairs.items():
            global_i = int(indices[local_i])
            global_j = int(indices[local_j])
            key_i = node_key[global_i]
            key_j = node_key[global_j]
            pair_key = tuple(sorted((global_i, global_j)))
            edge_map[pair_key] = {
                "cluster": cluster,
                "node_a": key_i,
                "node_b": key_j,
                "subject_a": str(nodes.loc[global_i, "subject"]),
                "run_a": str(nodes.loc[global_i, "run"]),
                "unit_id_a": int(nodes.loc[global_i, "unit_id"]),
                "subject_b": str(nodes.loc[global_j, "subject"]),
                "run_b": str(nodes.loc[global_j, "run"]),
                "unit_id_b": int(nodes.loc[global_j, "unit_id"]),
                "cosine_similarity": float(max(similarity[local_i, local_j], 0.0)),
                "selection_reason": "+".join(sorted(reasons)),
            }
    return pd.DataFrame(edge_map.values())


def cluster_layout(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[dict[str, np.ndarray], nx.Graph]:
    graph = nx.Graph()
    node_keys = [f"{row.subject}|{row.run}|{int(row.unit_id)}" for row in nodes.itertuples(index=False)]
    graph.add_nodes_from(node_keys)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.node_a, row.node_b, weight=float(row.cosine_similarity))

    centers = {
        "A-E": np.array([-1.65, 0.85]),
        "A-M": np.array([0.00, 0.85]),
        "A-L": np.array([1.65, 0.85]),
        "P-E": np.array([-1.65, -0.85]),
        "P-M": np.array([0.00, -0.85]),
        "P-L": np.array([1.65, -0.85]),
    }
    positions: dict[str, np.ndarray] = {}
    node_key_series = pd.Series(node_keys, index=nodes.index)
    for cluster_i, cluster in enumerate(CLUSTER_ORDER):
        members = node_key_series.loc[nodes["cluster"].astype(str).eq(cluster)].tolist()
        subgraph = graph.subgraph(members)
        local = nx.kamada_kawai_layout(
            subgraph,
            weight=None,
            scale=0.64,
        )
        for node, point in local.items():
            positions[node] = centers[cluster] + np.asarray(point, dtype=float)
    return positions, graph


def expanded_hull(points: np.ndarray, pad: float = 0.18) -> np.ndarray | None:
    if len(points) < 3:
        return None
    try:
        vertices = ConvexHull(points).vertices
    except Exception:
        return None
    polygon = points[vertices]
    center = polygon.mean(axis=0, keepdims=True)
    return center + (polygon - center) * (1 + pad)


def profile_summary(nodes: pd.DataFrame, raw: np.ndarray, masked: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in CLUSTER_ORDER:
        indices = np.flatnonzero(nodes["cluster"].astype(str).eq(cluster).to_numpy())
        for time_ms in range(800):
            raw_values = raw[indices, time_ms]
            masked_values = masked[indices, time_ms]
            rows.append(
                {
                    "cluster": cluster,
                    "cluster_label": CLUSTER_INFO[cluster]["label"],
                    "time_ms": time_ms,
                    "n_units": int(len(indices)),
                    "mean_active_minus_passive_hz": float(np.mean(raw_values)),
                    "sem_active_minus_passive_hz": float(np.std(raw_values, ddof=1) / np.sqrt(len(indices))),
                    "mean_qualifying_masked_contrast_hz": float(np.mean(masked_values)),
                    "sem_qualifying_masked_contrast_hz": float(np.std(masked_values, ddof=1) / np.sqrt(len(indices))),
                    "fraction_units_qualifying": float(np.mean(masked_values != 0)),
                }
            )
    return pd.DataFrame(rows)


def draw_profile_panel(ax: plt.Axes, summary: pd.DataFrame, clusters: tuple[str, str, str], title: str) -> None:
    for start, stop, label in ((0, 267, "Early"), (267, 533, "Middle"), (533, 800, "Late")):
        ax.axvspan(start, stop, color="#F2F4F3" if label != "Middle" else "#E9EEEC", alpha=0.75, zorder=0)
        ax.text((start + stop) / 2, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.3, color=GRAY)
    for cluster in clusters:
        frame = summary.loc[summary["cluster"].eq(cluster)]
        x = frame["time_ms"].to_numpy(float)
        mean = frame["mean_qualifying_masked_contrast_hz"].to_numpy(float)
        sem = frame["sem_qualifying_masked_contrast_hz"].to_numpy(float)
        color = CLUSTER_INFO[cluster]["color"]
        n_units = int(frame["n_units"].iloc[0])
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.13, lw=0)
        ax.plot(x, mean, color=color, lw=2.0, label=f"{CLUSTER_INFO[cluster]['timing']} (n={n_units})")
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xlim(0, 799)
    ax.set_ylabel("Mean qualifying\nA − P contrast (Hz)")
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.01), fontsize=7.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.55, zorder=0)


def plot_main_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    positions: dict[str, np.ndarray],
    profile_stats: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    fig = plt.figure(figsize=(12.2, 6.9), facecolor=WHITE)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.52, 1.0], height_ratios=[1, 1], wspace=0.30, hspace=0.33)
    ax = fig.add_subplot(grid[:, 0])
    ax.set_facecolor("#FCFCFA")

    node_keys = np.array([f"{row.subject}|{row.run}|{int(row.unit_id)}" for row in nodes.itertuples(index=False)])
    for cluster in CLUSTER_ORDER:
        members = node_keys[nodes["cluster"].astype(str).eq(cluster).to_numpy()]
        points = np.array([positions[node] for node in members])
        polygon = expanded_hull(points)
        color = CLUSTER_INFO[cluster]["color"]
        if polygon is not None:
            ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.10, edgecolor="none", zorder=0)
        center = points.mean(axis=0)
        ax.text(
            center[0],
            center[1] + 0.64,
            f"{CLUSTER_INFO[cluster]['short']}\nn={len(members)}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=INK,
            fontweight="bold",
        )

    for row in edges.itertuples(index=False):
        start = positions[row.node_a]
        stop = positions[row.node_b]
        similarity = float(row.cosine_similarity)
        ax.plot(
            [start[0], stop[0]],
            [start[1], stop[1]],
            color="#566763",
            lw=1.10 + 1.85 * similarity,
            alpha=0.48 + 0.45 * similarity,
            zorder=1,
        )

    duration = nodes["corrected_longest_segment_ms"].to_numpy(float)
    sizes = 18 + 82 * np.sqrt(duration / max(float(duration.max()), 1.0))
    effect = nodes["direction_consistent_mean_contrast_hz"].to_numpy(float)
    effect_limit = max(0.5, float(np.nanpercentile(np.abs(effect), 98)))
    norm = TwoSlopeNorm(vmin=-effect_limit, vcenter=0.0, vmax=effect_limit)
    scatter = ax.scatter(
        [positions[node][0] for node in node_keys],
        [positions[node][1] for node in node_keys],
        s=sizes,
        c=effect,
        cmap="RdBu",
        norm=norm,
        edgecolors="none",
        linewidths=0,
        alpha=0.95,
        zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.030, pad=0.012)
    colorbar.set_label("Direction-consistent qualifying A − P contrast (Hz)")
    colorbar.ax.tick_params(labelsize=7, length=2)
    ax.set_xlim(-2.55, 2.55)
    ax.set_ylim(-1.72, 1.73)
    ax.set_axis_off()
    ax.set_title("Six interpretable response archetypes", loc="left", fontsize=12, fontweight="bold", color=INK, pad=7)

    ax_active = fig.add_subplot(grid[0, 1])
    draw_profile_panel(ax_active, profile_stats, ("A-E", "A-M", "A-L"), "Active-dominant")
    ax_active.tick_params(labelbottom=False)
    ax_passive = fig.add_subplot(grid[1, 1])
    draw_profile_panel(ax_passive, profile_stats, ("P-E", "P-M", "P-L"), "Passive-dominant")
    ax_passive.set_xlabel("Time from corrected function-word onset (ms)")

    fig.suptitle(
        "GO-style network of corrected active/passive response archetypes",
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.045,
        0.947,
        "172 significant unit/session entries · clusters = direction × peak-latency tercile · edges = within-cluster temporal similarity",
        ha="left",
        fontsize=8.2,
        color=GRAY,
    )
    handles = [
        Line2D([0], [0], color="#6D7977", lw=1.8, label="Temporal-profile similarity edge"),
        Patch(facecolor="#9ECAE1", edgecolor="none", alpha=0.35, label="Active-dominant cluster hull"),
        Patch(facecolor="#FCAE91", edgecolor="none", alpha=0.35, label="Passive-dominant cluster hull"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.045, 0.015), ncol=3, frameon=False, fontsize=7.8)
    fig.text(
        0.985,
        0.022,
        "Response-phenotype similarity; not anatomical or causal connectivity",
        ha="right",
        fontsize=7.5,
        color=GRAY,
    )
    fig.subplots_adjust(top=0.89, bottom=0.10, left=0.045, right=0.98)
    return save_figure(fig, "Fig_GO_style_interpretable_response_archetypes")


def session_composition(nodes: pd.DataFrame) -> pd.DataFrame:
    observed = (
        nodes.groupby(["run", "cluster"], observed=True)
        .size()
        .rename("n_units")
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for run in corrected.RUN_ORDER:
        total = int(nodes["run"].eq(run).sum())
        for cluster in CLUSTER_ORDER:
            n_units = int(observed.get((run, cluster), observed.get((run, str(cluster)), 0)))
            rows.append(
                {
                    "subject": run[:5],
                    "run": run,
                    "session_label": RUN_SHORT[run],
                    "cluster": cluster,
                    "n_units": n_units,
                    "fraction": float(n_units / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_composition(composition: pd.DataFrame) -> tuple[Path, Path, Path]:
    order = [RUN_SHORT[run] for run in corrected.RUN_ORDER]
    fig, ax = plt.subplots(figsize=(9.4, 4.7), facecolor=WHITE)
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for cluster in CLUSTER_ORDER:
        frame = (
            composition.loc[composition["cluster"].astype(str).eq(cluster)]
            .set_index("session_label")
            .reindex(order)
        )
        values = frame["fraction"].fillna(0).to_numpy(float)
        ax.barh(y, values, left=left, color=CLUSTER_INFO[cluster]["color"], height=0.68, label=CLUSTER_INFO[cluster]["short"])
        left += values
    totals = composition.groupby("session_label")["n_units"].sum().reindex(order).fillna(0).astype(int)
    for index, total in enumerate(totals):
        ax.text(1.015, index, f"n={total}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Fraction of corrected significant units")
    ax.set_title("Every session is expressed as the same six response archetypes", loc="left", fontweight="bold", color=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    legend_order = ("A-E", "P-E", "A-M", "P-M", "A-L", "P-L")
    legend_handles = [
        Patch(facecolor=CLUSTER_INFO[cluster]["color"], edgecolor="none", label=CLUSTER_INFO[cluster]["short"])
        for cluster in legend_order
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.suptitle("Cross-session composition of interpretable clusters", x=0.08, y=0.98, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.83, bottom=0.25, left=0.19, right=0.95)
    return save_figure(fig, "Fig_GO_style_interpretable_cluster_composition")


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path, Path]:
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    svg = OUT_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(pdf, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(svg, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return png, pdf, svg


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected, segments = load_registered_screen()
    nodes, raw_profiles, masked_profiles = extract_profiles(selected, segments)
    edges = build_similarity_edges(nodes, masked_profiles)
    positions, graph = cluster_layout(nodes, edges)
    node_keys = [f"{row.subject}|{row.run}|{int(row.unit_id)}" for row in nodes.itertuples(index=False)]
    nodes["node_key"] = node_keys
    nodes["layout_x"] = [float(positions[key][0]) for key in node_keys]
    nodes["layout_y"] = [float(positions[key][1]) for key in node_keys]
    nodes["degree"] = [int(graph.degree(key)) for key in node_keys]

    profile_stats = profile_summary(nodes, raw_profiles, masked_profiles)
    composition = session_composition(nodes)
    cluster_summary = (
        nodes.groupby(["cluster", "cluster_label", "corrected_direction", "peak_latency_class"], observed=True)
        .agg(
            n_units=("unit_id", "size"),
            n_subjects=("subject", "nunique"),
            n_sessions=("run", "nunique"),
            median_peak_ms=("qualifying_peak_ms", "median"),
            median_longest_segment_ms=("corrected_longest_segment_ms", "median"),
            median_window_cohens_d=("corrected_window_cohens_d", "median"),
            median_direction_consistent_contrast_hz=("direction_consistent_mean_contrast_hz", "median"),
        )
        .reset_index()
    )
    edge_cluster_summary = edges.groupby("cluster").agg(
        n_edges=("node_a", "size"),
        median_cosine_similarity=("cosine_similarity", "median"),
        min_cosine_similarity=("cosine_similarity", "min"),
        max_cosine_similarity=("cosine_similarity", "max"),
    ).reset_index()
    cluster_summary = cluster_summary.merge(edge_cluster_summary, on="cluster", how="left")

    profile_rows: list[dict[str, object]] = []
    for node_i, row in enumerate(nodes.itertuples(index=False)):
        for time_ms in range(0, 800, 5):
            profile_rows.append(
                {
                    "subject": row.subject,
                    "run": row.run,
                    "unit_id": int(row.unit_id),
                    "cluster": str(row.cluster),
                    "time_ms": time_ms,
                    "active_minus_passive_hz": float(raw_profiles[node_i, time_ms]),
                    "qualifying_masked_contrast_hz": float(masked_profiles[node_i, time_ms]),
                }
            )

    nodes.to_csv(OUT_DIR / "interpretable_cluster_nodes.csv", index=False)
    edges.to_csv(OUT_DIR / "interpretable_cluster_edges.csv", index=False)
    cluster_summary.to_csv(OUT_DIR / "interpretable_cluster_summary.csv", index=False)
    composition.to_csv(OUT_DIR / "interpretable_cluster_session_composition.csv", index=False)
    profile_stats.to_csv(OUT_DIR / "interpretable_cluster_mean_profiles_1ms.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(OUT_DIR / "interpretable_cluster_unit_profiles_5ms.csv", index=False)

    main_paths = plot_main_network(nodes, edges, positions, profile_stats)
    composition_paths = plot_composition(composition)
    pd.DataFrame(
        [
            {"figure": "response_archetypes", "png": str(main_paths[0]), "pdf": str(main_paths[1]), "svg": str(main_paths[2])},
            {"figure": "session_composition", "png": str(composition_paths[0]), "pdf": str(composition_paths[1]), "svg": str(composition_paths[2])},
        ]
    ).to_csv(OUT_DIR / "figure_manifest.csv", index=False)

    method = {
        "node_screen": {
            "source": str(SCREEN_PATH),
            "n_all_unit_run_entries": 1782,
            "n_selected": 172,
            "corrected_event": "file-play onset + target_onset_sec_in_audio",
            "sdf_sigma_ms": 25,
            "pointwise_alpha": 0.025,
            "duration_rule": ">=6 consecutive 1-ms bins, no gap filling",
        },
        "cluster_definition": {
            "direction": "corrected qualifying-direction majority: Active > Passive or Active < Passive",
            "timing": "latency tercile of the largest absolute active-passive SDF difference inside qualifying segments",
            "early_ms": [0, 267],
            "middle_ms": [267, 533],
            "late_ms": [533, 800],
            "n_archetypes": 6,
            "counts": nodes["cluster"].astype(str).value_counts().reindex(CLUSTER_ORDER).to_dict(),
        },
        "edge_definition": {
            "profile": "active-minus-passive corrected-onset SDF, retained only inside qualifying significant segments",
            "similarity": "cosine similarity",
            "selection": f"union of each node's top {K_NEIGHBORS} within-cluster neighbors and the within-cluster maximum spanning tree",
            "purpose": "readable response-phenotype similarity map; descriptive, not inferential",
        },
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(method, handle, ensure_ascii=False, indent=2)

    readme = """# 可解释的GO-style active/passive response network

## Cluster定义

每个节点均先通过当前corrected筛选。Cluster不是黑箱community detection，而是两个可直接解释的维度：

1. 显著时间内的主要方向：Active > Passive或Active < Passive；
2. 显著时间内最大差异的潜伏期：Early 0–266 ms、Middle 267–532 ms、Late 533–799 ms。

因此固定得到六个响应原型：A>P Early/Middle/Late和A<P Early/Middle/Late。

## Edge定义

只在同一响应原型内部连边。每个unit连接到显著差异时间轨迹最相似的两个邻居；另加入最大生成树保证整个cluster连通。边宽和透明度随cosine similarity增加。该边是response-phenotype similarity，不是解剖或因果连接，也不是新的显著性检验。

## 图形编码

- 节点颜色：与主导方向一致的显著时间内平均A−P差异，蓝色为A>P，红色为A<P；
- 节点大小：最长连续显著时间；
- 节点没有edgecolor；
- 半透明背景：六个可解释cluster；
- 右侧曲线：每个cluster在显著时间内的平均A−P差异轨迹；
- 第二张图：8个session中六类cluster的组成。
"""
    (OUT_DIR / "README_zh.md").write_text(readme, encoding="utf-8")

    print(cluster_summary.to_string(index=False))
    print(f"\nNodes={len(nodes)}, edges={len(edges)}, min degree={nodes['degree'].min()}")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
