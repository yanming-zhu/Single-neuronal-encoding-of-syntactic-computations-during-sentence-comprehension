from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "nature_science_draft"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "neuropixel2025" / "data"
NET = DATA / "combined_active_passive_network_encoding"
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


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def letter(ax, s):
    ax.text(
        -0.08,
        1.05,
        s,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def short_run(run):
    if "gz001" in run:
        return "gz001 r1"
    if "gz002" in run:
        return "gz002 r2"
    if "gz003t2_g0_actpas_run1" in run:
        return "gz003 r1"
    if "gz003t2_g0_actpas_run4" in run:
        return "gz003 r4"
    if "gz004t1_g0_actpas_run0" in run:
        return "gz004 r0"
    if "gz004t1_g3_actpas_run3" in run:
        return "gz004 r3"
    if "gz005t0_g0_actpas_run0" in run:
        return "gz005 r0"
    if "gz005t0_g3_actpas_run3" in run:
        return "gz005 r3"
    return run


def format_p(p):
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def fig1_design():
    noun_units = pd.read_csv(NOUN / "noun_role_structure_encoding_units.csv")
    scores = pd.read_csv(NET / "active_passive_network_encoding_model_scores.csv")
    run_units = (
        noun_units.groupby(["patient", "run"])["unit_id"]
        .count()
        .reset_index(name="usable units")
        .sort_values(["patient", "run"])
    )

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], width_ratios=[1.2, 1.0, 1.1])
    ax_task = fig.add_subplot(gs[0, 0])
    ax_timeline = fig.add_subplot(gs[1, 0])
    ax_brain = fig.add_subplot(gs[:, 1])
    ax_summary = fig.add_subplot(gs[:, 2])

    # Task logic.
    ax_task.axis("off")
    ax_task.set_xlim(0, 1)
    ax_task.set_ylim(0, 1)
    letter(ax_task, "a")
    ax_task.set_title("Mandarin role-reversal design", loc="left", fontsize=10, pad=6)
    ax_task.text(0.02, 0.85, "Active", color=COLORS["blue"], fontweight="bold", fontsize=8)
    ax_task.text(0.02, 0.73, "N1 ba/gei N2 V", fontsize=8, family="DejaVu Sans Mono")
    ax_task.text(0.02, 0.62, "N1 agent; N2 patient", color=COLORS["gray"], fontsize=7)
    ax_task.text(0.02, 0.43, "Passive", color=COLORS["red"], fontweight="bold", fontsize=8)
    ax_task.text(0.02, 0.31, "N2 bei/rang N1 V", fontsize=8, family="DejaVu Sans Mono")
    ax_task.text(0.02, 0.20, "same words; reversed roles", color=COLORS["gray"], fontsize=7)
    ax_task.text(0.02, 0.04, "Key contrast: lexical content held close,\nstructural roles reversed.", color=COLORS["gray"], fontsize=7)

    # Trial timeline.
    letter(ax_timeline, "b")
    ax_timeline.set_title("Analysis epochs", loc="left", fontsize=10, pad=6)
    ax_timeline.set_xlim(-1.1, 2.1)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.axhline(0.45, color=COLORS["dark"], lw=1)
    ax_timeline.axvspan(-1.0, 0.0, ymin=0.36, ymax=0.54, color=COLORS["light_gray"])
    ax_timeline.axvspan(0.0, 0.8, ymin=0.36, ymax=0.54, color="#BFD3EA")
    ax_timeline.axvline(0, color=COLORS["dark"], lw=1)
    ax_timeline.text(-0.62, 0.70, "baseline", ha="center", fontsize=7)
    ax_timeline.text(0.47, 0.83, "response", ha="center", fontsize=7, color=COLORS["blue"])
    ax_timeline.text(0, 0.20, "function word\nor noun onset", ha="center", va="top", fontsize=7)
    ax_timeline.text(0.50, -0.28, "time from event onset (s)", transform=ax_timeline.transAxes, ha="center", fontsize=8)
    ax_timeline.set_yticks([])

    # Lightweight schematic recording locations. The draft deliberately avoids
    # embedding the high-resolution surface render, which can make quick
    # manuscript iteration painfully slow.
    letter(ax_brain, "c")
    ax_brain.set_title("Left temporal recordings", loc="left", fontsize=10, pad=6)
    theta = np.linspace(0, 2 * np.pi, 400)
    x = 0.5 + 0.42 * np.cos(theta) * (1 + 0.08 * np.sin(3 * theta))
    y = 0.52 + 0.28 * np.sin(theta) * (1 + 0.10 * np.cos(4 * theta))
    ax_brain.fill(x, y, color="#E5E8EA", ec="#AEB5BB", lw=1.0)
    for gx in np.linspace(0.20, 0.78, 5):
        ax_brain.plot([gx, gx + 0.05], [0.32, 0.72], color="white", lw=0.8, alpha=0.7)
    for gy in np.linspace(0.38, 0.68, 5):
        ax_brain.plot([0.18, 0.82], [gy, gy + 0.03 * np.sin(gy * 15)], color="white", lw=0.8, alpha=0.7)
    pts = np.array([[0.54, 0.54], [0.58, 0.50], [0.62, 0.51], [0.56, 0.46], [0.66, 0.42]])
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (px, py) in enumerate(pts):
        ax_brain.scatter(px, py, s=95, color=cols[i], edgecolor="black", zorder=5)
    ax_brain.set_xlim(0, 1)
    ax_brain.set_ylim(0, 1)
    ax_brain.axis("off")
    ax_brain.text(0.33, 0.15, "5 patients", fontsize=7, color=COLORS["gray"])
    ax_brain.text(
        0.03,
        0.01,
        "Schematic of approximate left temporal entry sites",
        transform=ax_brain.transAxes,
        color=COLORS["gray"],
        fontsize=6.5,
    )

    # Summary bars.
    letter(ax_summary, "d")
    ax_summary.set_title("Dataset used in this draft", loc="left", fontsize=10, pad=6)
    labels = run_units["run"].map(short_run).tolist()
    y = np.arange(len(labels))
    ax_summary.barh(y, run_units["usable units"], color=COLORS["gray"], alpha=0.85)
    ax_summary.set_yticks(y)
    ax_summary.set_yticklabels(labels, fontsize=7)
    ax_summary.invert_yaxis()
    ax_summary.set_xlabel("usable units")
    for yi, val in zip(y, run_units["usable units"]):
        ax_summary.text(val + 3, yi, str(int(val)), va="center", fontsize=7)
    ax_summary.text(
        0.02,
        -0.13,
        f"5 patients, {run_units.shape[0]} runs, {noun_units.shape[0]} unit entries; "
        f"{scores[['subject','run']].drop_duplicates().shape[0]} decoding runs.",
        transform=ax_summary.transAxes,
        fontsize=7,
        color=COLORS["gray"],
    )

    fig.suptitle("A cellular assay for argument-structure assignment", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig1_task_and_recordings")


def fig2_population_decoding():
    scores = pd.read_csv(NET / "active_passive_network_encoding_model_scores.csv")
    best = pd.read_csv(NET / "active_passive_best_network_models.csv")
    fixed = pd.read_csv(NET / "active_passive_fixed_model_chance_summary.csv")

    primary = scores[(scores.analysis_mode == "artifact_clean_le_0p2") & (scores.unit_set == "good_only")]
    if primary.empty:
        primary = scores[(scores.analysis_mode == "all_trials") & (scores.unit_set == "good_only")]
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
    model_labels = {
        "nested_single_unit": "single\nunit",
        "network_top5_l2": "top 5",
        "network_top10_l2": "top 10",
        "network_top25_l2": "top 25",
        "network_all_l2": "all",
        "network_pca2_l2": "PCA2",
        "network_pca4_l2": "PCA4",
        "network_pca8_l2": "PCA8",
    }
    pivot = (
        primary.pivot_table(index="run", columns="model", values="auc", aggfunc="mean")
        .reindex(columns=model_order)
        .sort_index()
    )
    labels = [short_run(r) for r in pivot.index]

    bprimary = best[(best.analysis_mode == "artifact_clean_le_0p2") & (best.unit_set == "good_only")]
    bprimary = bprimary.set_index("run").reindex(pivot.index).dropna(subset=["auc"])
    best_p = wilcoxon(bprimary["auc"] - 0.5, alternative="greater").pvalue
    gain_p = wilcoxon(bprimary["auc_gain_vs_nested_single_unit"], alternative="greater").pvalue

    top5 = primary[primary.model == "network_top5_l2"].set_index("run").reindex(pivot.index)
    top5_p = wilcoxon(top5["auc"].dropna() - 0.5, alternative="greater").pvalue

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.1, 1.0])
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_pair = fig.add_subplot(gs[0, 1])
    ax_fixed = fig.add_subplot(gs[1, 1])

    letter(ax_heat, "a")
    im = ax_heat.imshow(pivot.values, vmin=0.25, vmax=0.85, cmap="viridis", aspect="auto")
    ax_heat.set_xticks(np.arange(len(model_order)))
    ax_heat.set_xticklabels([model_labels[m] for m in model_order], rotation=35, ha="right", fontsize=7)
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_yticklabels(labels)
    ax_heat.set_title("Cross-validated active/passive decoding", loc="left", fontsize=10, pad=6)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="white" if val < 0.45 or val > 0.65 else "black")
    cb = fig.colorbar(im, ax=ax_heat, shrink=0.55, pad=0.02)
    cb.set_label("AUC")
    ax_heat.axvline(0.5, color="white", lw=0.8, alpha=0.8)

    letter(ax_pair, "b")
    x0, x1 = 0, 1
    for _, row in bprimary.iterrows():
        ax_pair.plot([x0, x1], [row["nested_single_unit_auc"], row["auc"]], color=COLORS["light_gray"], lw=1)
        ax_pair.scatter([x0], [row["nested_single_unit_auc"]], color="white", edgecolor=COLORS["gray"], s=28, zorder=3)
        ax_pair.scatter([x1], [row["auc"]], color=COLORS["blue"], edgecolor="white", s=32, zorder=3)
    ax_pair.axhline(0.5, color=COLORS["gray"], lw=1, ls="--")
    ax_pair.set_xticks([x0, x1])
    ax_pair.set_xticklabels(["nested\nsingle unit", "best\npopulation"])
    ax_pair.set_ylabel("AUC")
    ax_pair.set_ylim(0.25, 0.9)
    ax_pair.set_title("Descriptive best population model", loc="left", fontsize=10, pad=6)
    ax_pair.text(
        0.02,
        0.96,
        f"median AUC={bprimary['auc'].median():.2f}\nP>{format_p(best_p)}\ngain P={format_p(gain_p)}",
        transform=ax_pair.transAxes,
        va="top",
        fontsize=7,
    )

    letter(ax_fixed, "c")
    use = fixed[(fixed.analysis_mode == "artifact_clean_le_0p2") & (fixed.unit_set == "good_only")].copy()
    use = use[use.model.isin(model_order)].set_index("model").reindex(model_order).reset_index()
    xs = np.arange(len(use))
    ax_fixed.bar(xs, use["median_auc"], color=[COLORS["gray"] if m == "nested_single_unit" else COLORS["blue"] for m in use["model"]], alpha=0.8)
    ax_fixed.axhline(0.5, color=COLORS["gray"], ls="--", lw=1)
    ax_fixed.set_xticks(xs)
    ax_fixed.set_xticklabels([model_labels[m] for m in use["model"]], rotation=35, ha="right", fontsize=7)
    ax_fixed.set_ylabel("median AUC")
    ax_fixed.set_ylim(0.4, 0.72)
    ax_fixed.set_title("Fixed-model sensitivity", loc="left", fontsize=10, pad=6)
    for x, auc, p in zip(xs, use["median_auc"], use["wilcoxon_gt_chance_p"]):
        ax_fixed.text(x, auc + 0.012, f"P={format_p(p)}", ha="center", fontsize=6, rotation=90)
    ax_fixed.text(
        0.02,
        0.05,
        f"top-5 fixed model: median={top5['auc'].median():.2f}, P={format_p(top5_p)}",
        transform=ax_fixed.transAxes,
        fontsize=7,
        color=COLORS["dark"],
    )

    fig.suptitle("Function-word activity contains distributed active/passive information", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "fig2_population_decoding")

    return {
        "best_clean_good_n_runs": int(len(bprimary)),
        "best_clean_good_median_auc": float(bprimary["auc"].median()),
        "best_clean_good_wilcoxon_p": float(best_p),
        "best_clean_good_gain_median": float(bprimary["auc_gain_vs_nested_single_unit"].median()),
        "best_clean_good_gain_p": float(gain_p),
        "top5_clean_good_median_auc": float(top5["auc"].median()),
        "top5_clean_good_p": float(top5_p),
    }


def fig3_noun_role():
    robustness = pd.read_csv(NOUN / "noun_role_metric_robustness_by_unit.csv")
    summary = pd.read_csv(NOUN / "noun_role_metric_robustness_summary_by_run.csv")
    units = pd.read_csv(NOUN / "noun_role_structure_encoding_units.csv")
    gz002 = robustness[(robustness.metric == "baseline_corrected_hz") & (robustness.run == "gz002_g0_actpas_run2")]
    gz002_base = robustness[(robustness.metric == "local_baseline_hz") & (robustness.run == "gz002_g0_actpas_run2")]
    gz002_raw = robustness[(robustness.metric == "raw_response_hz") & (robustness.run == "gz002_g0_actpas_run2")]
    merged = (
        gz002[["unit_id", "subject_minus_predicate", "p_value"]]
        .rename(columns={"subject_minus_predicate": "bc_diff", "p_value": "bc_p"})
        .merge(
            gz002_base[["unit_id", "subject_minus_predicate"]].rename(columns={"subject_minus_predicate": "baseline_diff"}),
            on="unit_id",
        )
        .merge(
            gz002_raw[["unit_id", "subject_minus_predicate"]].rename(columns={"subject_minus_predicate": "raw_diff"}),
            on="unit_id",
        )
    )

    metrics = ["raw_response_hz", "local_baseline_hz", "baseline_corrected_hz"]
    metric_labels = ["raw response", "local baseline", "baseline-corrected"]
    total = (
        robustness.groupby("metric")
        .agg(
            p05=("significant_uncorrected_p05", "sum"),
            run_fdr05=("significant_run_fdr05", "sum"),
            global_fdr05=("significant_global_fdr05", "sum"),
        )
        .reindex(metrics)
    )

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2)
    ax_design = fig.add_subplot(gs[0, 0])
    ax_counts = fig.add_subplot(gs[0, 1])
    ax_run = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])

    letter(ax_design, "a")
    ax_design.axis("off")
    ax_design.set_xlim(0, 1)
    ax_design.set_ylim(0, 1)
    ax_design.set_title("Same noun, different structural slot", loc="left", fontsize=10, pad=6)
    ax_design.text(0.03, 0.78, "subject slot", color=COLORS["blue"], fontweight="bold")
    ax_design.text(0.43, 0.78, "function\nword", color=COLORS["gray"], ha="center", fontsize=8)
    ax_design.text(0.77, 0.78, "post-FW\nnoun slot", color=COLORS["red"], fontweight="bold", ha="center", fontsize=8)
    ax_design.text(0.03, 0.60, "insect", color=COLORS["blue"], fontsize=12)
    ax_design.text(0.43, 0.60, "ba", color=COLORS["gray"], fontsize=12, ha="center")
    ax_design.text(0.77, 0.60, "leaf", color=COLORS["red"], fontsize=12, ha="center")
    ax_design.text(0.03, 0.38, "leaf", color=COLORS["blue"], fontsize=12)
    ax_design.text(0.43, 0.38, "bei", color=COLORS["gray"], fontsize=12, ha="center")
    ax_design.text(0.77, 0.38, "insect", color=COLORS["red"], fontsize=12, ha="center")
    ax_design.annotate(
        "",
        xy=(0.12, 0.54),
        xytext=(0.77, 0.44),
        arrowprops=dict(arrowstyle="<->", lw=1, color=COLORS["dark"]),
    )
    ax_design.text(
        0.03,
        0.12,
        "Per unit: paired Wilcoxon across 20 noun identities.",
        fontsize=8,
        color=COLORS["gray"],
    )

    letter(ax_counts, "b")
    x = np.arange(len(metrics))
    width = 0.24
    ax_counts.bar(x - width, total["p05"], width, label="p<0.05", color=COLORS["light_gray"], edgecolor=COLORS["gray"])
    ax_counts.bar(x, total["run_fdr05"], width, label="run FDR<0.05", color=COLORS["blue"])
    ax_counts.bar(x + width, total["global_fdr05"], width, label="global FDR<0.05", color=COLORS["red"])
    ax_counts.axhline(719 * 0.05, color=COLORS["gray"], ls="--", lw=1)
    ax_counts.text(2.25, 719 * 0.05 + 3, "5% null expectation", fontsize=7, color=COLORS["gray"])
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax_counts.set_ylabel("units")
    ax_counts.set_title("Role effect depends on response metric", loc="left", fontsize=10, pad=6)
    ax_counts.legend(frameon=False, fontsize=7)

    letter(ax_run, "c")
    s = summary[summary.metric == "baseline_corrected_hz"].copy()
    s["short"] = s["run"].map(short_run)
    order = s.sort_values(["patient", "run"])["run"].tolist()
    s = s.set_index("run").loc[order].reset_index()
    ypos = np.arange(len(s))
    ax_run.barh(ypos, s["subject_pref_p05"], color=COLORS["blue"], label="subject-pref.")
    ax_run.barh(ypos, -s["predicate_pref_p05"], color=COLORS["red"], label="post-FW noun-pref.")
    ax_run.axvline(0, color=COLORS["dark"], lw=0.8)
    ax_run.set_yticks(ypos)
    ax_run.set_yticklabels(s["short"], fontsize=7)
    ax_run.invert_yaxis()
    ax_run.set_xlabel("uncorrected p<0.05 units")
    ax_run.set_title("Direction of baseline-corrected candidates", loc="left", fontsize=10, pad=6)
    ax_run.legend(frameon=False, fontsize=7, loc="lower right")

    letter(ax_scatter, "d")
    sig = merged["bc_p"] < 0.05
    ax_scatter.scatter(merged.loc[~sig, "baseline_diff"], merged.loc[~sig, "bc_diff"], s=12, color="#BFC5CC", alpha=0.7)
    ax_scatter.scatter(merged.loc[sig, "baseline_diff"], merged.loc[sig, "bc_diff"], s=18, color=COLORS["blue"], alpha=0.85, label="BC p<0.05")
    ax_scatter.axhline(0, color=COLORS["gray"], lw=0.8)
    ax_scatter.axvline(0, color=COLORS["gray"], lw=0.8)
    ax_scatter.set_xlabel("subject - post-FW baseline (Hz)")
    ax_scatter.set_ylabel("baseline-corrected role effect (Hz)")
    ax_scatter.set_title("GZ002 baseline diagnostic", loc="left", fontsize=10, pad=6)
    r = np.corrcoef(merged["baseline_diff"], merged["bc_diff"])[0, 1]
    ax_scatter.text(0.04, 0.95, f"r={r:.2f}\nraw p<0.05: 10/204\nBC run-FDR: 21/204", transform=ax_scatter.transAxes, va="top", fontsize=7)

    fig.suptitle("Noun-slot effects are present but partly driven by local baseline dynamics", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "fig3_noun_role_robustness")

    return {
        "noun_units": int(units.shape[0]),
        "noun_bc_p05": int(total.loc["baseline_corrected_hz", "p05"]),
        "noun_bc_run_fdr": int(total.loc["baseline_corrected_hz", "run_fdr05"]),
        "noun_bc_global_fdr": int(total.loc["baseline_corrected_hz", "global_fdr05"]),
        "noun_raw_p05": int(total.loc["raw_response_hz", "p05"]),
        "noun_raw_run_fdr": int(total.loc["raw_response_hz", "run_fdr05"]),
        "noun_baseline_run_fdr": int(total.loc["local_baseline_hz", "run_fdr05"]),
        "gz002_baseline_bc_corr": float(r),
    }


def fig4_specificity():
    noun = pd.read_csv(NOUN / "noun_role_structure_encoding_units.csv")
    verb = pd.read_csv(VERB_AP / "verb_active_passive_unit_results.csv")
    tri = pd.read_csv(VERB_TRI / "verb_three_condition_unit_results.csv")
    datasets = [
        ("noun slot", noun["p_value"].values, noun["paired_cohens_dz"].values, noun["significant_run_fdr05"].values, COLORS["blue"]),
        ("verb active/passive", verb["p_value"].values, verb["paired_cohens_dz"].values, verb["significant_run_fdr05"].values, COLORS["orange"]),
        ("verb 3-condition", tri["p_value_friedman"].values, tri["max_minus_min_bc_hz"].values, tri["significant_run_fdr05"].values, COLORS["purple"]),
    ]

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2)
    ax_ecdf = fig.add_subplot(gs[0, 0])
    ax_counts = fig.add_subplot(gs[0, 1])
    ax_effect = fig.add_subplot(gs[1, 0])
    ax_model = fig.add_subplot(gs[1, 1])

    letter(ax_ecdf, "a")
    grid = np.linspace(0, 1, 200)
    for label, ps, _, _, color in datasets:
        ps = ps[np.isfinite(ps)]
        ecdf = np.array([(ps <= t).mean() for t in grid])
        ax_ecdf.plot(grid, ecdf, label=label, color=color, lw=1.8)
    ax_ecdf.plot([0, 1], [0, 1], color=COLORS["gray"], ls="--", lw=1, label="uniform null")
    ax_ecdf.set_xlabel("P-value threshold")
    ax_ecdf.set_ylabel("fraction of units")
    ax_ecdf.set_title("P-value enrichment", loc="left", fontsize=10, pad=6)
    ax_ecdf.legend(frameon=False, fontsize=7)

    letter(ax_counts, "b")
    labels = [d[0] for d in datasets]
    p05 = [int(np.sum(d[1] < 0.05)) for d in datasets]
    fdr = [int(np.sum(d[3])) for d in datasets]
    null = [0.05 * np.sum(np.isfinite(d[1])) for d in datasets]
    x = np.arange(len(labels))
    ax_counts.bar(x - 0.18, p05, 0.35, color=COLORS["light_gray"], edgecolor=COLORS["gray"], label="p<0.05")
    ax_counts.bar(x + 0.18, fdr, 0.35, color=[d[4] for d in datasets], label="run FDR<0.05")
    for xi, val in zip(x, null):
        ax_counts.hlines(val, xi - 0.38, xi + 0.38, color=COLORS["gray"], ls="--", lw=1)
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(labels, rotation=18, ha="right")
    ax_counts.set_ylabel("units")
    ax_counts.set_title("Observed signals versus null expectation", loc="left", fontsize=10, pad=6)
    ax_counts.legend(frameon=False, fontsize=7)

    letter(ax_effect, "c")
    for i, (label, _, eff, sig, color) in enumerate(datasets[:2]):
        eff = np.asarray(eff, dtype=float)
        eff = np.abs(eff[np.isfinite(eff)])
        parts = ax_effect.violinplot(eff, positions=[i], widths=0.65, showmeans=False, showextrema=False, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("none")
            pc.set_alpha(0.35)
        parts["cmedians"].set_color(color)
        ax_effect.scatter(np.full_like(eff, i, dtype=float) + np.random.default_rng(10 + i).normal(0, 0.035, eff.size), eff, s=5, color=color, alpha=0.25)
    ax_effect.set_xticks([0, 1])
    ax_effect.set_xticklabels(["noun slot\nCohen dz", "verb A/P\nCohen dz"])
    ax_effect.set_ylabel("absolute paired effect size")
    ax_effect.set_title("Effect-size distribution", loc="left", fontsize=10, pad=6)

    letter(ax_model, "d")
    ax_model.axis("off")
    ax_model.set_title("Working model for the main text", loc="left", fontsize=10, pad=6)
    ax_model.set_xlim(0, 1)
    ax_model.set_ylim(0, 1)
    boxes = [
        (0.04, 0.67, "function-word\nepoch", COLORS["blue"]),
        (0.36, 0.67, "distributed\npopulation state", COLORS["green"]),
        (0.70, 0.67, "argument-role\nassignment", COLORS["red"]),
        (0.38, 0.30, "verb-locked\ncontrast weak", COLORS["orange"]),
    ]
    for x0, y0, txt, color in boxes:
        rect = plt.Rectangle((x0, y0), 0.24, 0.17, facecolor=color, alpha=0.15, edgecolor=color, lw=1.2)
        ax_model.add_patch(rect)
        ax_model.text(x0 + 0.12, y0 + 0.085, txt, ha="center", va="center", fontsize=7, color=COLORS["dark"])
    ax_model.annotate("", xy=(0.36, 0.755), xytext=(0.28, 0.755), arrowprops=dict(arrowstyle="->", lw=1, color=COLORS["dark"]))
    ax_model.annotate("", xy=(0.70, 0.755), xytext=(0.60, 0.755), arrowprops=dict(arrowstyle="->", lw=1, color=COLORS["dark"]))
    ax_model.annotate("", xy=(0.50, 0.47), xytext=(0.50, 0.67), arrowprops=dict(arrowstyle="-|>", lw=1, color=COLORS["gray"], ls="--"))
    ax_model.text(
        0.05,
        0.07,
        "Interpretation: strongest evidence supports a distributed state\nchange around function words and noun slots, not sparse\nverb-specific active/passive neurons.",
        fontsize=6.5,
        color=COLORS["gray"],
    )

    fig.suptitle("The strongest signal is role/slot-related, not a verb-locked active/passive code", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "fig4_specificity_and_logic")

    return {
        "noun_p05": int(np.sum(noun["p_value"].values < 0.05)),
        "noun_run_fdr": int(noun["significant_run_fdr05"].sum()),
        "verb_ap_p05": int(np.sum(verb["p_value"].values < 0.05)),
        "verb_ap_run_fdr": int(verb["significant_run_fdr05"].sum()),
        "verb_3cond_p05": int(np.sum(tri["p_value_friedman"].values < 0.05)),
        "verb_3cond_run_fdr": int(tri["significant_run_fdr05"].sum()),
        "null_expected_per_719": float(719 * 0.05),
    }


def main():
    summary = {}
    fig1_design()
    summary["fig2"] = fig2_population_decoding()
    summary["fig3"] = fig3_noun_role()
    summary["fig4"] = fig4_specificity()
    with open(OUT / "figure_statistics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote figures and summary to {OUT}")


if __name__ == "__main__":
    main()
