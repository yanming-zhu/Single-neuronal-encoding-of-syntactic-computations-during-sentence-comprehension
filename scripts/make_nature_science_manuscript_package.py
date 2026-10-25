from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "neuropixel2025" / "data"
OUT = ROOT / "reports" / "nature_science_draft"
FIG = OUT / "updated_figures"
SUPP = OUT / "supplementary"
SUPP_FIG = SUPP / "figures"
SUPP_TAB = SUPP / "tables"

for p in (FIG, SUPP_FIG, SUPP_TAB):
    p.mkdir(parents=True, exist_ok=True)

NET = DATA / "combined_active_passive_network_encoding"
TIME = DATA / "combined_active_passive_time_generalization"
VAR = DATA / "combined_active_passive_variance_partitioning"
STATE = DATA / "combined_active_passive_state_space_rsa"
BASE = DATA / "combined_active_passive_baseline_state"
DEPTH = DATA / "combined_active_passive_depth_laminar"
IED = DATA / "combined_active_passive_ied_artifact_control"
SEM = DATA / "combined_active_passive_semantic_grammatical_feature_model"
PART = DATA / "combined_neuron_network_information_partition"
NOUN = DATA / "combined_noun_role_structure_encoding"
VERB_AP = DATA / "combined_verb_active_passive_unit_activity"
VERB_TRI = DATA / "combined_verb_three_condition_unit_activity"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "blue": "#357ABD",
    "red": "#D95F4C",
    "green": "#4D9A57",
    "purple": "#7C6BB0",
    "orange": "#D98C3A",
    "gray": "#6E7175",
    "light_gray": "#D8DDE3",
    "dark": "#222222",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def letter(ax: plt.Axes, s: str) -> None:
    ax.text(
        -0.14,
        1.05,
        s,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def short_run(run: str) -> str:
    mapping = {
        "gz001t_g0_actpas_run1": "gz001 r1",
        "gz002_g0_actpas_run2": "gz002 r2",
        "gz003t2_g0_actpas_run1": "gz003 r1",
        "gz003t2_g0_actpas_run4": "gz003 r4",
        "gz004t1_g0_actpas_run0": "gz004 r0",
        "gz004t1_g3_actpas_run3": "gz004 r3",
        "gz005t0_g0_actpas_run0": "gz005 r0",
        "gz005t0_g3_actpas_run3": "gz005 r3",
    }
    return mapping.get(run, run)


def ptxt(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def medtxt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def plot_points(ax: plt.Axes, x: int | float, vals, color: str) -> None:
    vals = np.asarray(pd.Series(vals).dropna(), dtype=float)
    if vals.size == 0:
        return
    jitter = np.linspace(-0.08, 0.08, vals.size)
    ax.scatter(np.full(vals.size, x) + jitter, vals, s=18, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)


def add_image_panel(ax: plt.Axes, path: Path, title: str, label: str | None = None) -> None:
    if label:
        letter(ax, label)
    ax.imshow(mpimg.imread(path))
    ax.set_title(title, loc="left", fontsize=10, pad=4)
    ax.axis("off")


def annotate_panel(ax: plt.Axes, text: str, xy=(0.02, 0.02), color=COLORS["gray"]) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )


def get_primary_scores() -> pd.DataFrame:
    scores = pd.read_csv(NET / "active_passive_network_encoding_model_scores.csv")
    primary = scores[(scores.analysis_mode == "artifact_clean_le_0p2") & (scores.unit_set == "good_only")]
    return primary.copy()


def hypothesis_adjudication_dataframe(
    old_stats: dict,
    time: dict,
    base: dict,
) -> pd.DataFrame:
    rows = [
        {
            "test": "Cell-level role code",
            "stable_syntax_cell_prediction": "Many raw-response role-selective units should survive FDR.",
            "observed_result": (
                f"Raw noun-slot: {old_stats['fig3']['noun_raw_run_fdr']} run-FDR; "
                f"verb A/P: {old_stats['fig4']['verb_ap_run_fdr']} run-FDR."
            ),
            "interpretation": "Against a strong localist syntax-cell account.",
        },
        {
            "test": "Distributed readout",
            "stable_syntax_cell_prediction": "Population readout is not the primary signature.",
            "observed_result": (
                f"Best population AUC={old_stats['fig2']['best_clean_good_median_auc']:.2f}, "
                f"P={ptxt(old_stats['fig2']['best_clean_good_wilcoxon_p'])}; "
                f"gain over single units P={ptxt(old_stats['fig2']['best_clean_good_gain_p'])}."
            ),
            "interpretation": "Supports a distributed population-state code.",
        },
        {
            "test": "Temporal stability",
            "stable_syntax_cell_prediction": "A stable abstract code should generalize across time.",
            "observed_result": (
                f"Diagonal peak={time['median_diagonal_peak']['median_auc']:.2f} at "
                f"{time['median_diagonal_peak']['time_sec']:.2f}s; cross-time median="
                f"{time['primary_window_summary']['mean_cross_temporal_auc_0_to_0p8']['median']:.2f}, "
                f"P={ptxt(time['primary_window_summary']['mean_cross_temporal_auc_0_to_0p8']['wilcoxon_p_greater'])}."
            ),
            "interpretation": "Supports a transient rather than stationary state.",
        },
        {
            "test": "Baseline independence",
            "stable_syntax_cell_prediction": "Post-onset role coding should survive baseline-state controls.",
            "observed_result": (
                f"BC noun-slot: {old_stats['fig3']['noun_bc_run_fdr']} run-FDR; "
                f"raw noun-slot: {old_stats['fig3']['noun_raw_run_fdr']}; "
                f"residual AUC={base['residual_response_auc']['median']:.2f}."
            ),
            "interpretation": "Role information is entangled with local cortical state.",
        },
        {
            "test": "Lexical abstraction",
            "stable_syntax_cell_prediction": "Voice should remain separable from function-word identity.",
            "observed_result": "Voice is not identifiable after function-word identity.",
            "interpretation": "Abstract syntax is not proven; a crossed design is required.",
        },
    ]
    return pd.DataFrame(rows)


def draw_hypothesis_table(ax: plt.Axes, df: pd.DataFrame) -> None:
    letter(ax, "a")
    ax.axis("off")
    ax.set_title("Model comparison", loc="left", fontsize=10)

    card_y = 0.66
    card_h = 0.25
    cards = [
        (
            0.03,
            "Stable syntax-cell code",
            "local cells | stationary | abstract",
            COLORS["red"],
            "not supported",
        ),
        (
            0.52,
            "Transient population state",
            "distributed | time-local | mixed",
            COLORS["green"],
            "supported",
        ),
    ]
    for x, title, body, color, verdict in cards:
        ax.add_patch(
            plt.Rectangle((x, card_y), 0.43, card_h, transform=ax.transAxes, fc="white", ec=color, lw=1.2)
        )
        ax.text(x + 0.02, card_y + card_h - 0.075, title, transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=color)
        ax.text(x + 0.02, card_y + 0.08, body, transform=ax.transAxes, fontsize=7.0, va="bottom", color=COLORS["dark"])
        ax.text(
            x + 0.29,
            card_y + 0.08,
            verdict,
            transform=ax.transAxes,
            fontsize=7.0,
            va="bottom",
            color=color,
            fontweight="bold",
        )

    evidence = [
        ("single cells", "raw/verb FDR=0", COLORS["red"], "against localist"),
        ("population", "AUC=0.66\ngain P=0.039", COLORS["green"], "distributed"),
        ("time", "peak=0.67\ncross-time=0.51", COLORS["green"], "transient"),
        ("baseline", "BC effects\nresidual=0.40", COLORS["orange"], "state-entangled"),
        ("identity", "not separable\nafter word ID", COLORS["orange"], "abstract unproven"),
    ]
    ax.text(0.03, 0.54, "Evidence pattern", transform=ax.transAxes, fontsize=8.2, fontweight="bold")
    start_x, gap, box_w, box_h = 0.03, 0.012, 0.18, 0.30
    for i, (label, obs, color, verdict) in enumerate(evidence):
        x = start_x + i * (box_w + gap)
        ax.add_patch(plt.Rectangle((x, 0.16), box_w, box_h, transform=ax.transAxes, fc="#F9FAFB", ec="#D8DDE3", lw=0.6))
        ax.scatter([x + 0.022], [0.405], transform=ax.transAxes, s=26, color=color, clip_on=False)
        ax.text(x + 0.045, 0.43, label, transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="center")
        ax.text(x + 0.015, 0.315, obs, transform=ax.transAxes, fontsize=5.8, va="top")
        ax.text(x + 0.015, 0.205, textwrap.fill(verdict, width=15), transform=ax.transAxes, fontsize=5.7, va="top", color=color)
    ax.text(0.03, 0.04, "Detailed prediction-by-result table: Supplementary Table 7.", transform=ax.transAxes, fontsize=6.5, color=COLORS["gray"])


def fig1_assay_dataset_qc() -> Path:
    noun_units = pd.read_csv(NOUN / "noun_role_structure_encoding_units.csv")
    run_counts = (
        noun_units.groupby(["patient", "run", "ks_label"]).size().unstack(fill_value=0).reset_index()
    )
    for col in ("good", "mua"):
        if col not in run_counts:
            run_counts[col] = 0
    run_counts["total"] = run_counts["good"] + run_counts["mua"]
    run_counts = run_counts.sort_values(["patient", "run"])

    fig = plt.figure(figsize=(8.8, 5.7))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.35, 1.0, 1.25],
        height_ratios=[1, 1],
        wspace=0.55,
        hspace=0.72,
    )
    ax_design = fig.add_subplot(gs[0, 0])
    ax_epoch = fig.add_subplot(gs[1, 0])
    ax_site = fig.add_subplot(gs[:, 1])
    ax_units = fig.add_subplot(gs[0, 2])
    ax_quality = fig.add_subplot(gs[1, 2])

    ax_design.axis("off")
    letter(ax_design, "a")
    ax_design.set_title("Role-reversal sentence assay", loc="left", fontsize=9.5)
    rows = [
        ("Active", "N1  ba/gei/jiang  N2  V", "N1 = agent-like subject; N2 = patient-like noun", COLORS["blue"]),
        ("Passive", "N2  bei/rang/jiao  N1  V", "roles reversed", COLORS["red"]),
    ]
    y = 0.82
    for name, form, note, color in rows:
        ax_design.text(0.02, y, name, color=color, fontweight="bold")
        ax_design.text(0.02, y - 0.13, form, family="DejaVu Sans Mono", fontsize=8)
        ax_design.text(0.02, y - 0.25, note, color=COLORS["gray"], fontsize=7)
        y -= 0.40
    ax_design.text(
        0.02,
        0.00,
        "Question: stable syntax cells\nor a transient population state\nfor assigning sentence roles?",
        color=COLORS["dark"],
        fontsize=7.2,
    )

    letter(ax_epoch, "b")
    ax_epoch.set_title("Event-locked response windows", loc="left", fontsize=10)
    ax_epoch.axhline(0, color=COLORS["dark"], lw=1)
    ax_epoch.axvspan(-1, 0, color=COLORS["light_gray"], alpha=0.9, label="baseline")
    ax_epoch.axvspan(0, 0.8, color="#BFD3EA", alpha=0.9, label="response")
    ax_epoch.axvline(0, color=COLORS["dark"], lw=1)
    ax_epoch.set_xlim(-1.1, 1.5)
    ax_epoch.set_ylim(-0.12, 0.12)
    ax_epoch.set_yticks([])
    ax_epoch.set_xlabel("time from event onset (s)")
    ax_epoch.legend(frameon=False, fontsize=7, loc="upper right")

    letter(ax_site, "c")
    ax_site.set_title("Recording sites", loc="left", fontsize=9.5)
    theta = np.linspace(0, 2 * np.pi, 400)
    x = 0.5 + 0.40 * np.cos(theta) * (1 + 0.08 * np.sin(3 * theta))
    y = 0.52 + 0.30 * np.sin(theta) * (1 + 0.10 * np.cos(4 * theta))
    ax_site.fill(x, y, color="#E8EBED", ec="#AEB5BB", lw=1.0)
    for gx in np.linspace(0.22, 0.76, 5):
        ax_site.plot([gx, gx + 0.05], [0.32, 0.72], color="white", lw=0.8, alpha=0.8)
    pts = np.array([[0.53, 0.55], [0.58, 0.50], [0.63, 0.51], [0.56, 0.46], [0.66, 0.42]])
    label_offsets = [(-42, 17), (-42, -12), (13, 16), (13, -2), (13, 0)]
    for i, (px, py) in enumerate(pts):
        ax_site.scatter(px, py, s=90, color=plt.cm.tab10(i), edgecolor="black", zorder=5)
        ax_site.annotate(
            f"gz{i+1:03d}",
            xy=(px, py),
            xytext=label_offsets[i],
            textcoords="offset points",
            fontsize=6.5,
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.4, color=COLORS["gray"], shrinkA=3, shrinkB=3),
        )
    ax_site.set_xlim(0, 1)
    ax_site.set_ylim(0, 1)
    ax_site.axis("off")
    ax_site.text(0.08, 0.06, "5 patients, 8 runs\nschematic locations", fontsize=7, color=COLORS["gray"])

    letter(ax_units, "d")
    labels = run_counts["run"].map(short_run)
    yloc = np.arange(run_counts.shape[0])
    ax_units.barh(yloc, run_counts["good"], color=COLORS["blue"], label="good")
    ax_units.barh(yloc, run_counts["mua"], left=run_counts["good"], color=COLORS["orange"], label="MUA")
    ax_units.set_yticks(yloc)
    ax_units.set_yticklabels(labels, fontsize=7)
    ax_units.invert_yaxis()
    ax_units.set_xlabel("unit entries")
    ax_units.set_title("Units by run", loc="left", fontsize=9.5)
    ax_units.legend(frameon=False, fontsize=7)

    letter(ax_quality, "e")
    totals = [noun_units.shape[0], int((noun_units.ks_label == "good").sum()), int((noun_units.ks_label == "mua").sum())]
    ax_quality.bar(["all", "good", "MUA"], totals, color=[COLORS["gray"], COLORS["blue"], COLORS["orange"]])
    for i, v in enumerate(totals):
        ax_quality.text(i, v + 8, str(v), ha="center", fontsize=8)
    ax_quality.set_ylabel("unit entries")
    ax_quality.set_title("Analysis table composition", loc="left", fontsize=9.5)
    ax_quality.text(0.02, -0.26, "Primary population analyses use artifact-clean good units.", transform=ax_quality.transAxes, fontsize=6.8, color=COLORS["gray"])

    fig.suptitle("A transient human cortical population state for assigning sentence roles", fontsize=11.5, y=0.99)
    fig.tight_layout(rect=[0.01, 0.04, 1, 0.95])
    path = FIG / "fig1_assay_dataset_qc.png"
    savefig(fig, path)
    return path


def fig2_population_temporal() -> Path:
    primary = get_primary_scores()
    best = pd.read_csv(NET / "active_passive_best_network_models.csv")
    best = best[(best.analysis_mode == "artifact_clean_le_0p2") & (best.unit_set == "good_only")]
    leave = pd.read_csv(NET / "active_passive_leave_pair_out_decoding.csv")
    leave_good = leave[leave.unit_set == "good_only"]
    summary = read_json(TIME / "active_passive_time_generalization_summary.json")

    model_order = [
        "nested_single_unit",
        "network_top5_l2",
        "network_top10_l2",
        "network_top25_l2",
        "network_all_l2",
        "network_pca2_l2",
        "network_pca4_l2",
        "network_pca8_l2",
    ]
    labels = ["single", "top5", "top10", "top25", "all", "PCA2", "PCA4", "PCA8"]
    pivot = (
        primary.pivot_table(index="run", columns="model", values="auc", aggfunc="mean")
        .reindex(columns=model_order)
        .sort_index()
    )

    z = np.load(TIME / "active_passive_time_generalization_matrices.npz", allow_pickle=True)
    centers = z["centers_sec"]
    diag = z["diag_auc"]
    mat = np.nanmedian(z["auc_matrices"], axis=0)
    diag_med = np.nanmedian(diag, axis=0)

    fig = plt.figure(figsize=(7.5, 6.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.05], height_ratios=[1.05, 1.0])
    ax_hm = fig.add_subplot(gs[:, 0])
    ax_pair = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_diag = fig.add_subplot(gs[1, 1])
    ax_mat = fig.add_subplot(gs[1, 2])

    letter(ax_hm, "a")
    im = ax_hm.imshow(pivot.values, vmin=0.25, vmax=0.85, cmap="viridis", aspect="auto")
    ax_hm.set_xticks(np.arange(len(model_order)))
    ax_hm.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax_hm.set_yticks(np.arange(pivot.shape[0]))
    ax_hm.set_yticklabels([short_run(r) for r in pivot.index], fontsize=7)
    ax_hm.set_title("Function-word active/passive decoding", loc="left", fontsize=10)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax_hm.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color="white" if val > 0.64 or val < 0.42 else "black")
    cb = fig.colorbar(im, ax=ax_hm, shrink=0.55, pad=0.02)
    cb.set_label("AUC")

    letter(ax_pair, "b")
    b = best.set_index("run").reindex(pivot.index).dropna(subset=["auc"])
    for _, row in b.iterrows():
        ax_pair.plot([0, 1], [row.nested_single_unit_auc, row.auc], color=COLORS["light_gray"], lw=1)
    ax_pair.scatter(np.zeros(len(b)), b.nested_single_unit_auc, facecolor="white", edgecolor=COLORS["gray"], s=25)
    ax_pair.scatter(np.ones(len(b)), b.auc, color=COLORS["blue"], s=25)
    ax_pair.axhline(0.5, color=COLORS["gray"], ls="--", lw=0.8)
    ax_pair.set_xticks([0, 1])
    ax_pair.set_xticklabels(["single", "best\npopulation"])
    ax_pair.set_ylim(0.25, 0.88)
    ax_pair.set_ylabel("AUC")
    p_best = wilcoxon(b["auc"] - 0.5, alternative="greater").pvalue
    p_gain = wilcoxon(b["auc_gain_vs_nested_single_unit"], alternative="greater").pvalue
    ax_pair.set_title("Population-state gain", loc="left", fontsize=10)
    ax_pair.text(0.05, 0.78, f"median AUC={b.auc.median():.2f}\nP={ptxt(p_best)}\ngain P={ptxt(p_gain)}", fontsize=7)

    letter(ax_bar, "c")
    vals = [
        primary[primary.model == "network_top5_l2"]["auc"].median(),
        b["auc"].median(),
        leave_good["auc_leave_pair_out"].median(),
    ]
    names = ["fixed\ntop5", "best\nmodel", "leave\npair out"]
    cols = [COLORS["purple"], COLORS["blue"], COLORS["orange"]]
    ax_bar.bar(names, vals, color=cols, alpha=0.9)
    for i, v in enumerate(vals):
        ax_bar.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=8)
    ax_bar.axhline(0.5, color=COLORS["gray"], ls="--", lw=0.8)
    ax_bar.set_ylim(0.3, 0.75)
    ax_bar.set_ylabel("AUC")
    ax_bar.set_title("Generalization checks", loc="left", fontsize=10)

    letter(ax_diag, "d")
    for row in diag:
        ax_diag.plot(centers, row, color=COLORS["light_gray"], lw=0.8, alpha=0.75)
    ax_diag.plot(centers, diag_med, color=COLORS["blue"], lw=2.0)
    ax_diag.axhline(0.5, color=COLORS["gray"], ls="--", lw=0.8)
    ax_diag.axvspan(0, 0.8, color="#BFD3EA", alpha=0.25)
    ax_diag.set_xlim(-0.75, 1.75)
    ax_diag.set_ylim(0.25, 0.95)
    ax_diag.set_xlabel("time from function word (s)")
    ax_diag.set_ylabel("AUC")
    ax_diag.set_title("Transient top5 decoder", loc="left", fontsize=10)
    ax_diag.text(0.02, 0.93, f"peak median={summary['median_diagonal_peak']['median_auc']:.2f} at {summary['median_diagonal_peak']['time_sec']:.2f}s", transform=ax_diag.transAxes, fontsize=7)

    letter(ax_mat, "e")
    im2 = ax_mat.imshow(mat, extent=[centers[0], centers[-1], centers[-1], centers[0]], cmap="viridis", vmin=0.35, vmax=0.75, aspect="auto")
    ax_mat.plot([centers[0], centers[-1]], [centers[0], centers[-1]], color="white", lw=0.8, alpha=0.85)
    ax_mat.set_xlabel("test time (s)")
    ax_mat.set_ylabel("train time (s)")
    ax_mat.set_title("Weak cross-time stability", loc="left", fontsize=10)
    fig.colorbar(im2, ax=ax_mat, shrink=0.65, pad=0.02)
    annotate_panel(ax_mat, "Off-diagonal stability is weak", color=COLORS["dark"])

    fig.tight_layout()
    path = FIG / "fig2_population_temporal_dynamics.png"
    savefig(fig, path)
    return path


def fig3_role_state_geometry() -> Path:
    noun_robust = pd.read_csv(NOUN / "noun_role_metric_robustness_summary_by_run.csv")
    state_sum = read_json(STATE / "active_passive_state_space_rsa_summary.json")
    demix = pd.read_csv(STATE / "active_passive_demixed_marginal_variance.csv")
    state = pd.read_csv(STATE / "active_passive_state_space_run_summary.csv")
    base = pd.read_csv(BASE / "active_passive_baseline_state_run_summary.csv")
    example_subject = OUT / "example_subject_unit_cropped.png"
    example_predicate = OUT / "example_predicate_unit_cropped.png"

    fig = plt.figure(figsize=(7.5, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
    ax_ex1 = fig.add_subplot(gs[0, 0])
    ax_ex2 = fig.add_subplot(gs[0, 1])
    ax_counts = fig.add_subplot(gs[0, 2])
    ax_demix = fig.add_subplot(gs[1, 0])
    ax_cv = fig.add_subplot(gs[1, 1])
    ax_base = fig.add_subplot(gs[1, 2])

    add_image_panel(ax_ex1, example_subject, "Subject-slot example", "a")
    add_image_panel(ax_ex2, example_predicate, "Predicate-slot example", "b")

    letter(ax_counts, "c")
    order = ["raw_response_hz", "local_baseline_hz", "baseline_corrected_hz"]
    colors = [COLORS["gray"], COLORS["orange"], COLORS["blue"]]
    totals = noun_robust.groupby("metric")[["p05", "run_fdr05", "global_fdr05"]].sum().reindex(order)
    x = np.arange(len(order))
    ax_counts.bar(x - 0.22, totals["p05"], width=0.22, color=COLORS["light_gray"], label="p<0.05")
    ax_counts.bar(x, totals["run_fdr05"], width=0.22, color=colors, label="run FDR")
    ax_counts.bar(x + 0.22, totals["global_fdr05"], width=0.22, color=[COLORS["gray"], COLORS["orange"], COLORS["blue"]], alpha=0.45, label="global FDR")
    ax_counts.axhline(719 * 0.05, color=COLORS["gray"], ls="--", lw=0.8)
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(["raw\nresponse", "local\nbaseline", "baseline\ncorrected"], fontsize=7)
    ax_counts.set_ylabel("units")
    ax_counts.set_title("Cell metrics are state-sensitive", loc="left", fontsize=10)
    ax_counts.legend(frameon=False, fontsize=6)
    ax_counts.text(1.35, 95, "5% null", fontsize=6, color=COLORS["gray"])

    letter(ax_demix, "d")
    cats = ["syntax", "sentence_pair", "word_identity"]
    vals = [demix[c].median() for c in cats]
    ax_demix.bar(["syntax", "sentence\npair", "word\nidentity"], vals, color=[COLORS["red"], COLORS["blue"], COLORS["green"]])
    for i, c in enumerate(cats):
        plot_points(ax_demix, i, demix[c], COLORS["gray"])
        ax_demix.text(i, vals[i] + 0.025, f"{vals[i]:.3f}", ha="center", fontsize=7)
    ax_demix.set_ylim(0, max(vals) * 1.35)
    ax_demix.set_ylabel("marginal variance fraction")
    ax_demix.set_title("Mixed marginal variance", loc="left", fontsize=10)

    letter(ax_cv, "e")
    metrics = [
        ("cv_syntax_axis_median_diff", COLORS["blue"], "held-out\naxis diff"),
        ("cv_syntax_axis_frac_pair_positive", COLORS["green"], "fraction\npairs +"),
        ("syntax_silhouette", COLORS["orange"], "silhouette"),
    ]
    for i, (col, color, label) in enumerate(metrics):
        vals_i = state[col]
        ax_cv.bar(i, vals_i.median(), color=color, alpha=0.85)
        plot_points(ax_cv, i, vals_i, COLORS["dark"])
    ax_cv.axhline(0, color=COLORS["gray"], lw=0.8)
    ax_cv.axhline(0.5, color=COLORS["gray"], ls=":", lw=0.8)
    ax_cv.set_xticks(range(len(metrics)))
    ax_cv.set_xticklabels([m[2] for m in metrics], fontsize=7)
    ax_cv.set_title("Held-out role-state axis", loc="left", fontsize=10)
    ax_cv.text(0.02, 0.93, f"median held-out diff P={ptxt(state_sum['cv_syntax_axis_median_diff']['wilcoxon_p'])}\nfrac pairs+ P={ptxt(state_sum['cv_syntax_axis_frac_pair_positive']['wilcoxon_p'])}", transform=ax_cv.transAxes, fontsize=7)

    letter(ax_base, "f")
    auc_cols = ["baseline_auc", "raw_response_auc", "response_auc", "residual_response_auc"]
    auc_labels = ["baseline", "raw\nresp.", "baseline-\ncorr.", "residual"]
    auc_colors = [COLORS["gray"], COLORS["orange"], COLORS["blue"], COLORS["purple"]]
    for i, col in enumerate(auc_cols):
        ax_base.bar(i, base[col].median(), color=auc_colors[i], alpha=0.85)
        plot_points(ax_base, i, base[col], COLORS["dark"])
    ax_base.axhline(0.5, color=COLORS["gray"], ls="--", lw=0.8)
    ax_base.set_xticks(range(len(auc_cols)))
    ax_base.set_xticklabels(auc_labels, fontsize=6.5)
    ax_base.set_ylim(0.2, 0.85)
    ax_base.set_ylabel("AUC")
    ax_base.set_title("Baseline-state control", loc="left", fontsize=10)
    annotate_panel(ax_base, "Residualized response decoding is weak", xy=(0.04, 0.90), color=COLORS["dark"])

    fig.suptitle(
        "Matched nouns reveal a mixed lexical–structural cortical state",
        fontsize=11.5,
        y=0.995,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.91, wspace=0.58, hspace=0.76)
    path = FIG / "fig3_role_state_geometry.png"
    savefig(fig, path)
    return path


def fig4_specificity_controls() -> Path:
    stats = read_json(OUT / "figure_statistics_summary.json")
    time = read_json(TIME / "active_passive_time_generalization_summary.json")
    base = read_json(BASE / "active_passive_baseline_state_summary.json")
    var = pd.read_csv(VAR / "active_passive_variance_partitioning_run_summary.csv")
    ied = pd.read_csv(IED / "active_passive_ied_artifact_control_run_summary.csv")
    sem = pd.read_csv(SEM / "active_passive_semantic_grammatical_run_summary.csv")
    depth = pd.read_csv(DEPTH / "active_passive_depth_laminar_patient_summary.csv")
    tri = pd.read_csv(VERB_TRI / "verb_three_condition_summary_by_run.csv")
    vap = pd.read_csv(VERB_AP / "verb_active_passive_summary_by_run.csv")
    hyp = hypothesis_adjudication_dataframe(stats, time, base)

    fig = plt.figure(figsize=(8.8, 8.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 1.0], wspace=0.58, hspace=0.72)
    ax_hyp = fig.add_subplot(gs[0, :])
    ax_counts = fig.add_subplot(gs[1, 0])
    ax_var = fig.add_subplot(gs[1, 1])
    ax_sem = fig.add_subplot(gs[1, 2])
    ax_art = fig.add_subplot(gs[2, 0])
    ax_verb = fig.add_subplot(gs[2, 1])
    ax_depth = fig.add_subplot(gs[2, 2])

    draw_hypothesis_table(ax_hyp, hyp)

    letter(ax_counts, "b")
    names = ["noun\nslot", "verb\nA/P", "verb\n3-cond"]
    p05 = [stats["fig4"]["noun_p05"], stats["fig4"]["verb_ap_p05"], stats["fig4"]["verb_3cond_p05"]]
    fdr = [stats["fig4"]["noun_run_fdr"], stats["fig4"]["verb_ap_run_fdr"], stats["fig4"]["verb_3cond_run_fdr"]]
    x = np.arange(3)
    ax_counts.bar(x - 0.17, p05, width=0.32, color=COLORS["light_gray"], label="p<0.05")
    ax_counts.bar(x + 0.17, fdr, width=0.32, color=[COLORS["blue"], COLORS["orange"], COLORS["red"]], label="run FDR")
    ax_counts.axhline(stats["fig4"]["null_expected_per_719"], color=COLORS["gray"], ls="--", lw=0.8)
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(names)
    ax_counts.set_ylabel("unit count")
    ax_counts.set_title("Cell-level specificity", loc="left", fontsize=10)
    ax_counts.legend(frameon=False, fontsize=7)

    letter(ax_var, "c")
    cols = [
        "median_unique_acoustic_all",
        "median_unique_sentence_pair_all",
        "median_word_within_syntax",
        "median_syntax_after_word_identity",
    ]
    labs = ["acoustic", "sentence\npair", "word\nwithin voice", "voice\nafter word ID"]
    for i, col in enumerate(cols):
        vals = var[col]
        ax_var.bar(i, vals.median(), color=[COLORS["gray"], COLORS["green"], COLORS["orange"], COLORS["red"]][i], alpha=0.85)
        plot_points(ax_var, i, vals, COLORS["dark"])
    ax_var.axhline(0, color=COLORS["gray"], lw=0.8)
    ax_var.set_xticks(range(len(cols)))
    ax_var.set_xticklabels(labs, fontsize=6.5, rotation=15, ha="right")
    ax_var.set_ylabel("cross-validated unique R²")
    ax_var.set_title("Abstraction is not identifiable", loc="left", fontsize=10)

    letter(ax_sem, "d")
    sem_cols = [
        "median_cross_voice_semantic_pair_delta",
        "median_cross_voice_grammar_voice_delta",
        "median_leave_pair_grammar_voice_delta",
    ]
    sem_labs = ["semantic\ncross-voice", "voice\ncross-voice", "voice\nleave-pair"]
    sem_colors = [COLORS["green"], COLORS["blue"], COLORS["red"]]
    for i, col in enumerate(sem_cols):
        ax_sem.bar(i, sem[col].median(), color=sem_colors[i], alpha=0.85)
        plot_points(ax_sem, i, sem[col], COLORS["dark"])
    ax_sem.axhline(0, color=COLORS["gray"], lw=0.8)
    ax_sem.set_xticks(range(len(sem_cols)))
    ax_sem.set_xticklabels(sem_labs, fontsize=7)
    ax_sem.set_ylabel("delta R²")
    ax_sem.set_title("Semantic outgeneralizes voice", loc="left", fontsize=10)

    letter(ax_art, "e")
    art_main = ied[ied.threshold == "artifact_le_0p2"].copy()
    art_all = ied[ied.threshold == "all_trials"].copy()
    joined = art_main.merge(art_all[["run", "n_raw_p05"]], on="run", how="left", suffixes=("", "_all"))
    ax_art.scatter(joined["n_raw_p05_all"], joined["n_artifact_controlled_p05"], s=36, color=COLORS["purple"])
    lim = max(joined["n_raw_p05_all"].max(), joined["n_artifact_controlled_p05"].max()) + 1
    ax_art.plot([0, lim], [0, lim], color=COLORS["gray"], ls="--", lw=0.8)
    label_offsets = {
        "gz005 r0": (-35, 7),
        "gz003 r4": (5, -9),
        "gz005 r3": (5, 7),
        "gz004 r0": (5, -8),
        "gz004 r3": (5, 6),
    }
    for _, row in joined.iterrows():
        label = short_run(row.run)
        ax_art.annotate(
            label,
            xy=(row["n_raw_p05_all"], row["n_artifact_controlled_p05"]),
            xytext=label_offsets.get(label, (5, 5)),
            textcoords="offset points",
            fontsize=6,
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.35, color=COLORS["gray"], shrinkA=3, shrinkB=3),
        )
    ax_art.set_xlim(-0.5, lim)
    ax_art.set_ylim(-0.5, lim)
    ax_art.set_xlabel("raw p<0.05 units")
    ax_art.set_ylabel("artifact-controlled p<0.05 units")
    ax_art.set_title("IED/artifact sensitivity", loc="left", fontsize=10)

    letter(ax_verb, "f")
    labels = [short_run(r) for r in vap.sort_values("run").run]
    idx = np.arange(len(labels))
    vap_s = vap.sort_values("run")
    tri_s = tri.set_index("run").reindex(vap_s.run).reset_index()
    ax_verb.bar(idx - 0.18, vap_s.n_uncorrected_p05, width=0.34, color=COLORS["orange"], label="verb A/P")
    tri_uncorrected_col = (
        "n_uncorrected_p05"
        if "n_uncorrected_p05" in tri_s.columns
        else "n_aggregate_friedman_p05"
    )
    ax_verb.bar(
        idx + 0.18,
        tri_s[tri_uncorrected_col],
        width=0.34,
        color=COLORS["red"],
        label="verb 3-cond",
    )
    ax_verb.set_xticks(idx)
    ax_verb.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax_verb.set_ylabel("uncorrected units")
    ax_verb.set_title("Verb-window by run", loc="left", fontsize=10)
    ax_verb.legend(frameon=False, fontsize=7)

    letter(ax_depth, "g")
    ax_depth.scatter(depth["deep_minus_superficial_abs_effect_hz"], depth["deep_minus_superficial_signed_effect_hz"], s=45, color=COLORS["blue"])
    ax_depth.axhline(0, color=COLORS["gray"], lw=0.8)
    ax_depth.axvline(0, color=COLORS["gray"], lw=0.8)
    for _, row in depth.iterrows():
        ax_depth.text(row.deep_minus_superficial_abs_effect_hz + 0.03, row.deep_minus_superficial_signed_effect_hz, row.subject, fontsize=7)
    ax_depth.set_xlabel("deep-superficial |effect| (Hz)")
    ax_depth.set_ylabel("deep-superficial signed effect (Hz)")
    ax_depth.set_title("Depth: descriptive", loc="left", fontsize=10)

    fig.suptitle(
        "Control analyses disfavour a stable syntax-cell code",
        fontsize=11.5,
        y=0.99,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.91, wspace=0.58, hspace=0.76)
    path = FIG / "fig4_specificity_controls_scope.png"
    savefig(fig, path)
    return path


def supp_image_grid(items: list[tuple[Path, str]], out: Path, title: str, cols: int = 2) -> Path:
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < len(items):
            add_image_panel(ax, items[i][0], items[i][1], chr(ord("a") + i))
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, out)
    return out


def make_supp_figures() -> dict[str, Path]:
    figs = {}
    figs["supp_fig1"] = supp_image_grid(
        [
            (TIME / "fig_time_resolved_decoding_diagonal.png", "Time-resolved diagonal decoding"),
            (TIME / "fig_cross_temporal_decoding_median_matrix.png", "Median cross-temporal matrix"),
            (TIME / "fig_cross_temporal_decoding_by_run.png", "Cross-temporal matrices by run"),
        ],
        SUPP_FIG / "supp_fig1_time_generalization.png",
        "Supplementary Fig. 1 | Time-resolved and cross-temporal active/passive decoding",
        cols=2,
    )
    figs["supp_fig2"] = supp_image_grid(
        [
            (VAR / "fig_variance_partitioning_run_medians.png", "Run-level unique R2 medians"),
            (VAR / "fig_variance_partitioning_unit_distributions.png", "Unit-level unique R2 distributions"),
            (VAR / "fig_syntax_candidate_vs_word_identity.png", "Voice identifiability with function-word identity"),
        ],
        SUPP_FIG / "supp_fig2_variance_partitioning.png",
        "Supplementary Fig. 2 | Acoustic, word-identity and voice variance partitioning",
        cols=2,
    )
    figs["supp_fig3"] = supp_image_grid(
        [
            (STATE / "fig_state_space_pca_by_run.png", "PCA state spaces by run"),
            (STATE / "fig_demixed_marginal_variance.png", "Demixed-style variance"),
            (STATE / "fig_rsa_model_correlations.png", "RSA model correlations"),
            (STATE / "fig_rsa_multiple_betas.png", "Multiple-regression RSA betas"),
        ],
        SUPP_FIG / "supp_fig3_state_space_rsa.png",
        "Supplementary Fig. 3 | Population state-space and representational geometry",
        cols=2,
    )
    figs["supp_fig4"] = supp_image_grid(
        [
            (BASE / "fig_baseline_vs_evoked_decoding_auc.png", "Baseline, evoked and residual decoding"),
            (BASE / "fig_baseline_vs_evoked_axis_projection.png", "Baseline and evoked syntax-axis projection"),
            (BASE / "fig_baseline_evoked_axis_cosine.png", "Baseline/evoked axis cosine"),
            (BASE / "fig_unit_baseline_response_correlations.png", "Unit baseline-response correlations"),
        ],
        SUPP_FIG / "supp_fig4_baseline_state.png",
        "Supplementary Fig. 4 | Baseline-state controls",
        cols=2,
    )
    figs["supp_fig5"] = supp_image_grid(
        [
            (DEPTH / "fig_patient_depth_signed_effects.png", "Signed active-passive effects by depth"),
            (DEPTH / "fig_patient_depth_abs_effects.png", "Absolute effects by depth"),
            (DEPTH / "fig_patient_depth_band_signed_effects.png", "Depth-band summaries"),
            (IED / "fig_artifact_balance_by_run.png", "Artifact balance by run"),
            (IED / "fig_artifact_only_decoding_auc.png", "Artifact-only decoding"),
            (IED / "fig_raw_vs_artifact_controlled_effects.png", "Raw vs artifact-controlled effects"),
        ],
        SUPP_FIG / "supp_fig5_depth_artifact_controls.png",
        "Supplementary Fig. 5 | Depth organization and IED/artifact controls",
        cols=2,
    )
    figs["supp_fig6"] = supp_image_grid(
        [
            (SEM / "fig_semantic_grammar_feature_run_medians.png", "Semantic and grammar feature deltas"),
            (SEM / "fig_semantic_vs_grammar_unit_scatter.png", "Semantic vs grammar unit deltas"),
            (SEM / "fig_semantic_grammar_model_r2.png", "Feature-model R2"),
            (VERB_AP / "figures" / "fig_verb_active_passive_summary.png", "Verb active/passive unit tests"),
            (VERB_TRI / "figures" / "fig_verb_three_condition_summary.png", "Verb active/passive/scrambled tests"),
        ],
        SUPP_FIG / "supp_fig6_semantic_grammar_verb_controls.png",
        "Supplementary Fig. 6 | Semantic/grammar feature model and verb controls",
        cols=2,
    )
    fig7 = SUPP_FIG / "supp_fig7_neuron_network_information_partition.png"
    fig7.write_bytes((PART / "fig_neuron_network_information_partition.png").read_bytes())
    figs["supp_fig7"] = fig7
    return figs


def rounded_csv(df: pd.DataFrame, path: Path) -> Path:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(6)
    out.to_csv(path, index=False)
    return path


def make_tables() -> dict[str, Path]:
    noun_units = pd.read_csv(NOUN / "noun_role_structure_encoding_units.csv")
    run_counts = noun_units.groupby(["patient", "run", "ks_label"]).size().unstack(fill_value=0).reset_index()
    for col in ("good", "mua"):
        if col not in run_counts:
            run_counts[col] = 0
    run_counts["n_unit_entries"] = run_counts["good"] + run_counts["mua"]
    run_counts["run_label"] = run_counts["run"].map(short_run)
    run_counts = run_counts[["patient", "run", "run_label", "n_unit_entries", "good", "mua"]]

    primary = get_primary_scores()
    best = pd.read_csv(NET / "active_passive_best_network_models.csv")
    best = best[(best.analysis_mode == "artifact_clean_le_0p2") & (best.unit_set == "good_only")]
    leave = pd.read_csv(NET / "active_passive_leave_pair_out_decoding.csv")
    leave = leave[leave.unit_set == "good_only"].rename(columns={"patient": "subject"})
    time_run = pd.read_csv(TIME / "active_passive_time_generalization_run_summary.csv")
    pop = (
        best[["subject", "run", "n_trials", "n_units_variable", "model", "auc", "nested_single_unit_auc", "auc_gain_vs_nested_single_unit"]]
        .merge(
            primary[primary.model == "network_top5_l2"][["subject", "run", "auc"]].rename(columns={"auc": "fixed_top5_auc"}),
            on=["subject", "run"],
            how="left",
        )
        .merge(leave[["subject", "run", "auc_leave_pair_out"]], on=["subject", "run"], how="left")
        .merge(time_run[["subject", "run", "mean_diag_auc_primary_window", "max_diag_auc_primary_window", "peak_diag_time_sec", "mean_cross_temporal_auc_primary_window"]], on=["subject", "run"], how="left")
    )

    noun_robust = pd.read_csv(NOUN / "noun_role_metric_robustness_summary_by_run.csv")
    vap = pd.read_csv(VERB_AP / "verb_active_passive_summary_by_run.csv")
    tri = pd.read_csv(VERB_TRI / "verb_three_condition_summary_by_run.csv")
    unit_tests = noun_robust[noun_robust.metric == "baseline_corrected_hz"].copy()
    unit_tests = unit_tests.merge(
        vap[["patient", "run", "n_uncorrected_p05", "n_run_fdr05"]].rename(
            columns={"n_uncorrected_p05": "verb_ap_p05", "n_run_fdr05": "verb_ap_run_fdr05"}
        ),
        on=["patient", "run"],
        how="left",
    ).merge(
        tri[["patient", "run", "n_uncorrected_p05", "n_run_fdr05"]].rename(
            columns={"n_uncorrected_p05": "verb_3cond_p05", "n_run_fdr05": "verb_3cond_run_fdr05"}
        ),
        on=["patient", "run"],
        how="left",
    )

    state = pd.read_csv(STATE / "active_passive_state_space_run_summary.csv")
    demix = pd.read_csv(STATE / "active_passive_demixed_marginal_variance.csv")
    rsa = pd.read_csv(STATE / "active_passive_rsa_run_summary.csv")
    base = pd.read_csv(BASE / "active_passive_baseline_state_run_summary.csv")
    geometry = state.merge(demix[["subject", "run", "syntax", "sentence_pair", "word_identity"]], on=["subject", "run"], how="left")
    geometry = geometry.merge(rsa[["subject", "run", "rho_syntax", "rho_sentence_pair", "rho_word_identity", "model_r2"]], on=["subject", "run"], how="left")
    geometry = geometry.merge(
        base[["subject", "run", "baseline_auc", "raw_response_auc", "response_auc", "residual_response_auc", "baseline_response_axis_cosine", "median_unit_baseline_response_r"]],
        on=["subject", "run"],
        how="left",
    )

    var = pd.read_csv(VAR / "active_passive_variance_partitioning_run_summary.csv")
    depth = pd.read_csv(DEPTH / "active_passive_depth_laminar_patient_summary.csv")
    art = pd.read_csv(IED / "active_passive_ied_artifact_control_run_summary.csv")
    sem = pd.read_csv(SEM / "active_passive_semantic_grammatical_run_summary.csv")
    partition = pd.read_csv(PART / "neuron_network_information_partition_table.csv")
    old_stats = read_json(OUT / "figure_statistics_summary.json")
    time_summary = read_json(TIME / "active_passive_time_generalization_summary.json")
    baseline_summary = read_json(BASE / "active_passive_baseline_state_summary.json")
    hyp = hypothesis_adjudication_dataframe(old_stats, time_summary, baseline_summary)

    tables = {
        "supp_table_1_recording_runs": rounded_csv(run_counts, SUPP_TAB / "supp_table_1_recording_runs.csv"),
        "supp_table_2_population_decoding": rounded_csv(pop, SUPP_TAB / "supp_table_2_population_decoding.csv"),
        "supp_table_3_unit_selectivity_counts": rounded_csv(unit_tests, SUPP_TAB / "supp_table_3_unit_selectivity_counts.csv"),
        "supp_table_4_state_geometry_baseline": rounded_csv(geometry, SUPP_TAB / "supp_table_4_state_geometry_baseline.csv"),
        "supp_table_5_variance_partitioning": rounded_csv(var, SUPP_TAB / "supp_table_5_variance_partitioning.csv"),
        "supp_table_6a_depth_patient_summary": rounded_csv(depth, SUPP_TAB / "supp_table_6a_depth_patient_summary.csv"),
        "supp_table_6b_artifact_control": rounded_csv(art, SUPP_TAB / "supp_table_6b_artifact_control.csv"),
        "supp_table_6c_semantic_grammar": rounded_csv(sem, SUPP_TAB / "supp_table_6c_semantic_grammar.csv"),
        "supp_table_7_hypothesis_adjudication": rounded_csv(hyp, SUPP_TAB / "supp_table_7_hypothesis_adjudication.csv"),
        "supp_table_8_neuron_network_partition": rounded_csv(partition, SUPP_TAB / "supp_table_8_neuron_network_partition.csv"),
    }

    index_lines = [
        "# Supplementary tables",
        "",
        "| Table | File | Contents |",
        "| --- | --- | --- |",
    ]
    descriptions = {
        "supp_table_1_recording_runs": "Recording-run unit counts and quality labels.",
        "supp_table_2_population_decoding": "Run-level population decoding, fixed-model, leave-pair-out and time-generalization summaries.",
        "supp_table_3_unit_selectivity_counts": "Noun-slot and verb-window candidate counts by run.",
        "supp_table_4_state_geometry_baseline": "State-space, demixed-style variance, RSA and baseline-control metrics.",
        "supp_table_5_variance_partitioning": "Cross-validated encoding/variance-partitioning controls.",
        "supp_table_6a_depth_patient_summary": "Patient-level depth/laminar descriptive summary.",
        "supp_table_6b_artifact_control": "Run-level IED/artifact balance and artifact-controlled unit effects.",
        "supp_table_6c_semantic_grammar": "Semantic/grammatical feature-model run summaries.",
        "supp_table_7_hypothesis_adjudication": "Prediction-by-result adjudication of stable syntax-cell and transient population-state accounts.",
        "supp_table_8_neuron_network_partition": "Information-by-information synthesis of unit-level and population-level evidence.",
    }
    for key, path in tables.items():
        index_lines.append(f"| {key.replace('_', ' ')} | `{path.name}` | {descriptions[key]} |")
    (SUPP_TAB / "supplementary_tables_index.md").write_text("\n".join(index_lines) + "\n")
    return tables


def rel(path: Path) -> str:
    return str(path.resolve())


def make_manuscript(paths: dict[str, Path], supp: dict[str, Path], tables: dict[str, Path]) -> Path:
    old_stats = read_json(OUT / "figure_statistics_summary.json")
    time = read_json(TIME / "active_passive_time_generalization_summary.json")
    state = read_json(STATE / "active_passive_state_space_rsa_summary.json")
    base = read_json(BASE / "active_passive_baseline_state_summary.json")
    var = read_json(VAR / "active_passive_variance_partitioning_summary.json")
    sem = read_json(SEM / "active_passive_semantic_grammatical_summary.json")
    part = read_json(PART / "neuron_network_information_partition_summary.json")
    depth = read_json(DEPTH / "active_passive_depth_laminar_summary.json")
    ied = read_json(IED / "active_passive_ied_artifact_control_summary.json")

    text = f"""# A transient human cortical population state for assigning sentence roles

## Abstract

Understanding a sentence requires assigning words to event roles. At cellular resolution, this computation could be implemented as a stable abstract syntax-cell code, or as a transient population state that mixes grammatical cues with lexical, semantic and contextual information. We tested these alternatives using acute human Neuropixels recordings during Mandarin active/passive role-reversal sentences. Across five patients and eight runs, function-word-locked population activity carried active/passive information: the descriptive best artifact-clean good-unit population model reached median AUC={old_stats['fig2']['best_clean_good_median_auc']:.2f} across seven runs (one-sided Wilcoxon P={ptxt(old_stats['fig2']['best_clean_good_wilcoxon_p'])}) and improved over nested single-unit decoding (median gain={old_stats['fig2']['best_clean_good_gain_median']:.2f}, P={ptxt(old_stats['fig2']['best_clean_good_gain_p'])}). Time-resolved decoding peaked transiently at AUC={time['median_diagonal_peak']['median_auc']:.2f}, but cross-temporal generalization was weak (median={time['primary_window_summary']['mean_cross_temporal_auc_0_to_0p8']['median']:.2f}, P={ptxt(time['primary_window_summary']['mean_cross_temporal_auc_0_to_0p8']['wilcoxon_p_greater'])}). Noun-slot effects were enriched after baseline correction ({old_stats['fig3']['noun_bc_run_fdr']} run-FDR units), but raw responses and verb-window controls yielded zero run-FDR units. A neuron-versus-network partition synthesis supported a partial division: active/passive contrast was network-dominant, semantic-pair identity was visible in both unit and population readouts, and noun-slot information was unit-available with only indirect population-state support. Population geometry revealed a weak-to-moderate held-out active/passive role-state axis, whereas RSA, variance partitioning and semantic/grammar controls did not support a clean material-invariant syntax category. These results argue against a strong localist, stationary syntax-cell account and support a transient human cortical population state for assigning sentence roles.

## Introduction

Sentence comprehension requires a role-binding operation: a listener must determine who did what to whom as words arrive in time. Linguistic theories disagree about how abstract this operation is, but they share a computational demand: the brain must assign nouns to event roles and update that assignment when function words and word order change the sentence. The cellular-scale neural implementation of this operation in humans remains unresolved.

We framed the problem as a test between two neural coding hypotheses. A strong localist account predicts stable neurons or low-dimensional axes that encode syntactic role or active/passive voice independently of lexical material, timing and baseline state. A population-state account predicts that role assignment should be distributed, temporally local and mixed with word identity, sentence-pair structure, semantic content and cortical state. This distinction is not identical to a formal linguistic debate, but it is a falsifiable neural implementation question with direct consequences for theories of sentence comprehension.

Mandarin active/passive constructions provide a compact assay. In active *ba/gei/jiang* sentences and passive *bei/rang/jiao* sentences, closely related lexical materials can be arranged so that nouns and verbs participate in reversed role structures. We analyzed acute Neuropixels recordings from left temporal cortical sites in five patients listening to these sentences. We combined population decoding, time-resolved and cross-temporal decoding, paired noun-identity tests, state-space analyses, variance partitioning, baseline-state controls, artifact controls, depth summaries and verb-window negative controls. The result is a sharpened claim: the current data support a transient population state for sentence-role assignment and argue against a strong stable syntax-cell code, while not yet proving an abstract role code independent of function-word identity and acoustics.

## Results

### A role-reversal assay separates two neural coding hypotheses

We analyzed auditory Mandarin active/passive sentences recorded from five patients and eight usable runs (Fig. 1). The main unit-level table contained 719 unit entries, including 461 units labelled good and 258 multi-unit activity entries. Responses were measured in event-locked windows after function-word or noun onset and compared with local pre-event baselines. The assay asks whether cellular activity contains a stable syntax-like code or a transient population state for sentence-role assignment. It is deliberately not a complete abstraction test: active/passive structure changes together with function-word identity and acoustics, and noun role is partly linked to serial position.

![Figure 1]({rel(paths['fig1'])})

### Population activity carries sentence-role information better than single units

Function-word-locked population activity contained active/passive information (Fig. 2a,b). In artifact-clean good-unit runs, the descriptive best population model reached median AUC={old_stats['fig2']['best_clean_good_median_auc']:.2f} (n={old_stats['fig2']['best_clean_good_n_runs']}, P={ptxt(old_stats['fig2']['best_clean_good_wilcoxon_p'])}) and improved over nested single-unit decoding by a median AUC gain of {old_stats['fig2']['best_clean_good_gain_median']:.2f} (P={ptxt(old_stats['fig2']['best_clean_good_gain_p'])}). A fixed sparse top-5 decoder reached median AUC={old_stats['fig2']['top5_clean_good_median_auc']:.2f}, but only approached significance (P={ptxt(old_stats['fig2']['top5_clean_good_p'])}). Leave-sentence-pair-out decoding was weaker, indicating that cross-material generalization is not yet established.

Time-resolved decoding further sharpened this interpretation. A fixed top-5 decoder showed transient above-chance peaks, with the median diagonal AUC peaking at {time['median_diagonal_peak']['median_auc']:.2f} around {time['median_diagonal_peak']['time_sec']:.2f} s. However, the mean diagonal AUC across the 0-0.8 s primary window was modest (median={time['primary_window_summary']['mean_diag_auc_0_to_0p8']['median']:.2f}, P={ptxt(time['primary_window_summary']['mean_diag_auc_0_to_0p8']['wilcoxon_p_greater'])}), and mean cross-temporal generalization in the primary window was close to chance (median={time['primary_window_summary']['mean_cross_temporal_auc_0_to_0p8']['median']:.2f}, P={ptxt(time['primary_window_summary']['mean_cross_temporal_auc_0_to_0p8']['wilcoxon_p_greater'])}). Thus, sentence-role information is detectable at the population level, but the code behaves like a transient state rather than a stable category label.

![Figure 2]({rel(paths['fig2'])})

### Noun-slot and geometry analyses reveal a mixed cortical state

The same noun identity could appear in different structural slots, enabling paired subject-slot versus post-function-word noun-slot tests. Baseline-corrected noun-slot effects were enriched: {old_stats['fig3']['noun_bc_p05']}/719 units reached uncorrected P<0.05, {old_stats['fig3']['noun_bc_run_fdr']} survived run-level FDR and {old_stats['fig3']['noun_bc_global_fdr']} survived global FDR. Example units showed strong subject-preferring and predicate-preferring responses (Fig. 3a,b). However, raw response tests yielded only {old_stats['fig3']['noun_raw_p05']} uncorrected candidates and no run-FDR units, while local baseline itself yielded {old_stats['fig3']['noun_baseline_run_fdr']} run-FDR units. In GZ002, baseline differences and baseline-corrected role effects were strongly anticorrelated (r={old_stats['fig3']['gz002_baseline_bc_corr']:.2f}). The safest interpretation is therefore not a pure post-onset noun-role code, but a structure-sensitive local state spanning baseline and response windows.

Population geometry analyses supported the same cautious picture. A descriptive syntax axis separated active and passive responses within runs (median d-prime={state['syntax_axis_dprime']['median']:.2f}, P={ptxt(state['syntax_axis_dprime']['wilcoxon_p'])}), and leave-one-pair-out projection was positive by a median of {state['cv_syntax_axis_median_diff']['median']:.2f} (P={ptxt(state['cv_syntax_axis_median_diff']['wilcoxon_p'])}); the fraction of held-out sentence pairs with active greater than passive was also above 0.5 (median={state['cv_syntax_axis_frac_pair_positive']['median']:.1f}, P={ptxt(state['cv_syntax_axis_frac_pair_positive']['wilcoxon_p'])}). But silhouette and RSA effects were near zero, and demixed-style marginal variance was dominated by sentence-pair and word-identity structure (median fractions {state['marginal_fraction_sentence_pair']['median']:.3f} and {state['marginal_fraction_word_identity']['median']:.3f}) rather than syntax alone (median {state['marginal_fraction_syntax']['median']:.3f}).

Baseline-state controls also constrained the claim. Baseline-only decoding was not reliable (median AUC={base['baseline_auc']['median']:.2f}), baseline-corrected response decoding was stronger (median AUC={base['response_auc']['median']:.2f}, P={ptxt(base['response_auc']['wilcoxon_p'])}), and residualized response decoding was weak (median AUC={base['residual_response_auc']['median']:.2f}). This pattern suggests that the baseline correction helps reveal a reproducible state difference but also removes variance entangled with the effect.

![Figure 3]({rel(paths['fig3'])})

### Hypothesis adjudication disfavors a stable syntax-cell code

We summarized the logic as a prediction-by-result adjudication (Fig. 4a). A strong syntax-cell account predicts robust raw-response role-selective units, stable cross-temporal coding and separability from function-word identity. The observed pattern runs in the opposite direction: role information is strongest in population models and baseline-corrected state measures, temporal stability is weak, and current variance partitioning cannot identify syntax independently of function-word identity.

We then asked explicitly whether different information types are better described as unit-level or network-level codes (Supplementary Fig. 7 and Supplementary Table 8). The result was a partial division rather than a clean dissociation. The active/passive function-word-period contrast was network-dominant: nested single-unit decoding was not reliably above chance (median AUC=0.60, P=0.062), whereas the best population readout was above chance and improved over nested single units (median AUC=0.66, gain=0.11, P=0.039). By contrast, proposition or semantic-pair identity was unit-available (cross-voice semantic-pair delta R2={sem['run_median_signed_rank']['median_cross_voice_semantic_pair_delta']['median']:.3f}, P={ptxt(sem['run_median_signed_rank']['median_cross_voice_semantic_pair_delta']['wilcoxon_p'])}) and also dominated population marginal variance (sentence-pair fraction={state['marginal_fraction_sentence_pair']['median']:.3f}, P={ptxt(state['marginal_fraction_sentence_pair']['wilcoxon_p'])}). Noun-slot information was more restricted: baseline-corrected unit tests identified {old_stats['fig3']['noun_bc_run_fdr']} run-FDR units, but the available population evidence was indirect, coming from a related active/passive role-state axis rather than from direct network-level noun-slot decoding. In contrast, abstract voice independent of function-word identity remained not identifiable in the current design.

Verb-window analyses provided a negative control. Active/passive verb tests produced {old_stats['fig4']['verb_ap_p05']}/719 uncorrected candidates and zero run-FDR units; active/passive/scrambled three-condition tests produced {old_stats['fig4']['verb_3cond_p05']}/719 uncorrected candidates and zero run-FDR units, close to the {old_stats['fig4']['null_expected_per_719']:.1f} candidates expected under a 5% null (Fig. 4b,f). Cross-validated variance partitioning did not recover positive unique active/passive variance after accounting for function-word identity, as expected because function-word identity fully determines active/passive condition in the current stimulus set (Fig. 4c). A semantic/grammatical feature model showed stronger cross-voice generalization for semantic sentence-pair structure (median delta R2={sem['run_median_signed_rank']['median_cross_voice_semantic_pair_delta']['median']:.3f}, P={ptxt(sem['run_median_signed_rank']['median_cross_voice_semantic_pair_delta']['wilcoxon_p'])}) than for grammatical voice (Fig. 4d). IED/artifact summaries showed that artifact-controlled unit effects were broadly comparable to raw effects rather than producing a hidden syntax effect (Fig. 4e). Depth analyses across {depth['n_patients_analyzed']} patients and {depth['n_units_analyzed']} good-unit entries were descriptive, with no basis for a strong laminar claim (Fig. 4g).

![Figure 4]({rel(paths['fig4'])})

## Discussion

The central conclusion is that sentence-role assignment in this dataset is best described as a transient human cortical population state. The positive evidence is population-level active/passive decoding, a transient time-resolved peak, weak-to-moderate held-out role-axis projection and enriched baseline-corrected noun-slot effects. The explicit unit-versus-network synthesis makes this claim more precise: the data support a partial information partition, not an exclusive split. Semantic/proposition identity is available in unit-level readouts and also shapes population geometry, whereas the function-word-period active/passive contrast is strongest as a population readout. Noun-slot information should be treated more narrowly: it is unit-available in the current analyses, while the population evidence is indirect and does not yet demonstrate direct network-level noun-slot encoding. The negative evidence is equally important: raw-response single-unit role effects and verb-window role effects do not survive run-level FDR, cross-temporal stability is weak and syntax cannot be identified independently of function-word identity in the present stimulus set.

This pattern directly weakens a strong stable syntax-cell hypothesis: a sparse, stationary, material-invariant cellular code for active/passive voice or sentence role is not the dominant signature in these recordings. The data instead fit a state-space account in which function words, word order, lexical identity and semantic context move a cortical population through a temporary state that constrains how the next noun or verb is interpreted. In linguistic terms, the result is most compatible with cue-based, construction-sensitive and predictive-state views of comprehension, but it does not decide among those theories.

The critical limitation is identifiability. Active/passive condition is confounded with function-word identity and acoustics, and noun role is partly confounded with serial position and pre-event state. For that reason, the current manuscript should not claim proof of an abstract syntactic-role code. Its stronger and safer claim is falsificatory: the observed cellular data do not behave as predicted by a strong localist and stationary syntax-cell code.

Several weaknesses become useful hypotheses. First, baseline sensitivity may reflect preparatory or context-setting state changes rather than nuisance alone; future analyses should model baseline as an interpretable state variable. Second, weak cross-temporal generalization suggests a dynamic code tied to local processing demands rather than a stationary symbol. Third, stronger semantic-pair generalization suggests that lexical/semantic organization may provide the scaffold on which grammatical role assignment is computed.

The decisive next experiment is a crossed lexical/acoustic design in which the same function words, nouns and verbs are balanced across role assignments. Additional trials should support sentence-pair-held-out decoding as the primary confirmatory analysis. Acoustic envelope, phoneme/onset, duration, function-word identity, semantic-pair identity and syntactic-role predictors should be fit jointly with hierarchical patient/run inference. With those controls, the current result could be extended from falsifying a strong syntax-cell account to demonstrating an abstract population code for sentence roles.

## Methods

### Participants and recordings

The analyses used acute human Neuropixels recordings collected during clinical procedures from five patients. Recording sites were approximated to left temporal cortical regions. Probe-depth and anatomical labels should be treated as approximate in the current draft.

### Stimuli and task structure

Participants listened to Mandarin active/passive sentences. Active trials used active markers such as *ba/gei/jiang* and passive trials used passive markers such as *bei/rang/jiao*. The design contrasted role-reversal sentence forms while keeping lexical materials closely related. Scrambled materials were available for verb-window three-condition controls in a smaller subset.

### Spike sorting and unit inclusion

Raw recordings were processed through the project spike-sorting and firing-rate pipeline. Unit entries carried quality labels (`good` and `mua`). Population decoding emphasized artifact-clean good-unit analyses; unit-count summaries include all unit entries used in the noun-slot table unless otherwise specified.

### Event alignment and response metrics

Function-word, noun and verb events were aligned using event metadata and audio-onset tables. Primary response windows were 0-0.8 s after event onset. Baseline-corrected responses subtracted a local pre-event baseline, usually -1 to 0 s relative to the aligned event. Raw response and local baseline metrics were recomputed as robustness controls.

### Population decoding

Active/passive decoding used run-wise cross-validated classifiers trained on function-word-locked firing-rate features. Models included nested single-unit decoders, sparse top-k-unit L2 logistic classifiers, all-unit L2 classifiers and PCA-based classifiers. The descriptive best model selected the highest-performing population model within each run. Fixed-model analyses used the top-5-unit L2 classifier. Leave-sentence-pair-out decoding grouped active/passive trials by matched sentence pair to test lexical generalization.

### Time-resolved and cross-temporal decoding

Time-resolved decoding used sliding windows from -0.75 to 1.75 s around the function word with 0.2 s windows and 0.05 s step size. At each training time bin, a fixed top-5 L2 logistic decoder was trained and evaluated at every test time bin. Diagonal values correspond to ordinary time-resolved decoding; off-diagonal values test temporal generalization.

### Noun-slot and verb-window unit tests

For noun-slot analyses, each unit was tested with paired noun identities comparing sentence-initial subject-slot responses with post-function-word noun responses. Wilcoxon signed-rank tests were performed per unit and corrected by run, patient and global FDR. The same framework was recomputed for raw response, local baseline and baseline-corrected metrics. Verb active/passive analyses compared paired verb identities across active and passive conditions. Three-condition verb analyses compared active, passive and scrambled contexts.

### Population geometry and RSA

For state-space analyses, function-word responses were z-scored within run and summarized with PCA, syntax-axis projections, demixed-style marginal variance and representational similarity analysis. The syntax axis was evaluated descriptively within run and more conservatively with leave-one-pair-out projection tests. Demixed-style variance estimated the fraction of condition-mean variance associated with syntax, sentence-pair and word-identity factors. RSA compared neural dissimilarities with model matrices for syntax, sentence pair and word identity.

### Variance partitioning and feature models

Cross-validated ridge-regression models estimated unique variance associated with acoustic features, sentence-pair identity, function-word identity and active/passive voice. Because function-word identity fully determines active/passive condition in the current stimulus set, syntax-after-word-identity estimates are diagnostic for identifiability rather than clean abstract syntax. A semantic/grammatical feature model tested semantic-pair and grammatical-voice generalization in the subset of runs with sufficient clean trials and complete feature coverage.

### Neuron-versus-network information partition

To test whether different information types were better described as unit-level or population-level signals, we synthesized the pre-specified outputs above into an information-partition table. Unit-level support was defined by nested single-unit decoding, run/global FDR unit tests or unit-level cross-validated feature-model deltas, depending on the information type. Direct network-level support was defined by population decoding, population marginal-variance evidence for the same information type or a direct held-out population readout. A positive population gain over nested single-unit decoding was treated as network-dominant evidence. For noun-slot information, active/passive population-axis geometry was marked as indirect network support rather than direct network-level noun-slot encoding. Information types that could not be separated from function-word identity in the current stimulus set were labelled not identifiable rather than assigned to either level.

### Baseline-state, depth and artifact controls

Baseline-state analyses compared baseline-only, raw response, baseline-corrected response and residualized response decoders and syntax-axis projections. Depth analyses summarized active/passive effect magnitudes as a function of approximate probe depth and patient. IED/artifact controls quantified artifact fractions around sentence onset, tested artifact-only decoding and compared raw unit effects with artifact-controlled estimates across artifact thresholds.

### Statistics

Run-level non-parametric tests used one-sided Wilcoxon signed-rank tests against chance or zero where appropriate. Unit-level analyses used per-unit tests followed by false-discovery-rate correction within run, patient and globally. Because each run contained small trial counts, run-level inference should be interpreted cautiously and is prioritized over unit-count inflation.

## Main figure legends

**Figure 1 | A human cellular assay for testing sentence-role coding hypotheses.**  
**a,** Mandarin active/passive role-reversal design, framed as a test of stable syntax-cell versus transient population-state accounts. **b,** Event-locked baseline and response windows. **c,** Schematic approximate left temporal recording sites across five patients. **d,** Unit entries by run, separated by quality label. **e,** Analysis table composition.

**Figure 2 | Sentence-role information is distributed and temporally transient.**  
**a,** Cross-validated active/passive decoding across models and runs. **b,** Descriptive best population model compared with nested single-unit decoding. **c,** Fixed top-5, descriptive best-model and leave-sentence-pair-out AUC summaries. **d,** Time-resolved diagonal decoding using a fixed top-5 decoder. **e,** Median cross-temporal decoding matrix showing weak temporal stability.

**Figure 3 | Noun-slot effects and population geometry reveal a mixed role-state.**  
**a,b,** Example units with subject-slot and predicate-slot preferences. **c,** Noun-slot candidate counts across raw response, local baseline and baseline-corrected metrics. **d,** Demixed-style marginal variance fractions showing strong sentence-pair and word-identity structure. **e,** Held-out role-state axis metrics. **f,** Baseline-state decoding controls.

**Figure 4 | Hypothesis adjudication disfavors a stable syntax-cell code.**  
**a,** Prediction-by-result adjudication of the stable syntax-cell account against the transient population-state account. **b,** Candidate counts for noun-slot and verb-window tests. **c,** Cross-validated variance-partitioning controls. **d,** Semantic/grammatical feature-model deltas. **e,** Raw versus artifact-controlled unit-candidate counts. **f,** Verb-window candidate counts by run. **g,** Descriptive depth summary.

## Supplementary figure legends

**Supplementary Figure 1 | Time-resolved and cross-temporal decoding details.** Run-level and median fixed top-5 decoding around the function-word epoch.

**Supplementary Figure 2 | Variance-partitioning controls.** Acoustic, sentence-pair, function-word identity and active/passive voice encoding-model summaries.

**Supplementary Figure 3 | State-space and RSA analyses.** PCA projections, demixed-style variance, RSA correlations and multiple-regression RSA betas.

**Supplementary Figure 4 | Baseline-state analyses.** Baseline, evoked, raw and residualized response decoding and syntax-axis summaries.

**Supplementary Figure 5 | Depth and IED/artifact controls.** Depth summaries, artifact balance, artifact-only decoding and raw versus artifact-controlled effects.

**Supplementary Figure 6 | Semantic/grammatical feature model and verb controls.** Semantic-pair and grammatical-voice model deltas together with verb-window negative controls.

**Supplementary Figure 7 | Neuron-versus-network information partition.** Evidence synthesis assigning each information type to unit-available, network-dominant, direct mixed, indirect network-support, not identifiable or not reliably detected categories.

## Supplementary tables

The supplementary tables are stored as CSV files in `{rel(SUPP_TAB)}`:

1. `{tables['supp_table_1_recording_runs'].name}`: recording-run unit counts and quality labels.
2. `{tables['supp_table_2_population_decoding'].name}`: run-level population decoding, fixed-model, leave-pair-out and time-generalization summaries.
3. `{tables['supp_table_3_unit_selectivity_counts'].name}`: noun-slot and verb-window candidate counts.
4. `{tables['supp_table_4_state_geometry_baseline'].name}`: state-space, RSA and baseline-control metrics.
5. `{tables['supp_table_5_variance_partitioning'].name}`: variance-partitioning controls.
6. `{tables['supp_table_6a_depth_patient_summary'].name}`, `{tables['supp_table_6b_artifact_control'].name}`, `{tables['supp_table_6c_semantic_grammar'].name}`: depth, artifact and semantic/grammar control summaries.
7. `{tables['supp_table_7_hypothesis_adjudication'].name}`: prediction-by-result adjudication of stable syntax-cell and transient population-state accounts.
8. `{tables['supp_table_8_neuron_network_partition'].name}`: evidence synthesis testing which information types are unit-available, network-dominant, directly mixed, indirectly supported by population state or not identifiable.

## Logic audit

The reframed manuscript makes a falsificatory claim: these human cellular data do not support a strong stable syntax-cell account of sentence-role assignment. The positive claim is narrower: sentence-role information appears as a transient, distributed population state with a partial unit/network information partition. Noun-slot information is treated as unit-available with indirect population-state support, not as directly demonstrated network-level noun-slot encoding. The manuscript still does not prove abstract syntactic-role coding independent of function-word identity, acoustics or serial position. The main unresolved vulnerabilities are: active/passive markers are acoustically and lexically distinct; noun role and position are partly confounded; the strongest noun-slot effect depends on baseline correction; the best population decoder is descriptive; and trial counts remain small. The most important next experiment is a crossed lexical/acoustic design with enough trials for sentence-pair-held-out decoding as the primary confirmatory analysis.
"""
    out = OUT / "manuscript_full_draft.md"
    out.write_text(text)
    (OUT / "manuscript_draft.md").write_text(text)
    return out


def main() -> None:
    main_figs = {
        "fig1": fig1_assay_dataset_qc(),
        "fig2": fig2_population_temporal(),
        "fig3": fig3_role_state_geometry(),
        "fig4": fig4_specificity_controls(),
    }
    supp_figs = make_supp_figures()
    tables = make_tables()
    manuscript = make_manuscript(main_figs, supp_figs, tables)

    summary = {
        "main_figures": {k: str(v) for k, v in main_figs.items()},
        "supplementary_figures": {k: str(v) for k, v in supp_figs.items()},
        "supplementary_tables": {k: str(v) for k, v in tables.items()},
        "manuscript": str(manuscript),
    }
    (OUT / "manuscript_package_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
