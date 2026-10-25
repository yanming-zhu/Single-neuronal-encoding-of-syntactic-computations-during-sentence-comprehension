#!/usr/bin/env python3
"""Synthesize unit-level and population-level evidence by information type."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "neuropixel2025" / "data"
OUT_DIR = DATA / "combined_neuron_network_information_partition"
REPORT_PATH = ROOT / "reports" / "neuron_network_information_partition.md"

NET = DATA / "combined_active_passive_network_encoding"
SEM = DATA / "combined_active_passive_semantic_grammatical_feature_model"
STATE = DATA / "combined_active_passive_state_space_rsa"
VAR = DATA / "combined_active_passive_variance_partitioning"
TIME = DATA / "combined_active_passive_time_generalization"
TEMPORAL = DATA / "combined_active_passive_temporal_dynamics"
NATURE = ROOT / "reports" / "nature_science_draft"


LABEL_TEXT = {
    "network_dominant": "network-dominant",
    "network_available": "network-available",
    "unit_available": "unit-available",
    "unit_available_indirect_network": "unit-available; indirect network support",
    "mixed_unit_and_network": "unit + network",
    "mixed_network_enhanced": "unit + network-enhanced",
    "not_identifiable": "not identifiable",
    "not_reliably_detected": "not reliably detected",
}

LABEL_COLOR = {
    "network_dominant": "#2F6FBB",
    "network_available": "#6A8FC9",
    "unit_available": "#C94E36",
    "unit_available_indirect_network": "#C94E36",
    "mixed_unit_and_network": "#3E7B52",
    "mixed_network_enhanced": "#007C89",
    "not_identifiable": "#D89B1D",
    "not_reliably_detected": "#777777",
}

FIG_PALETTE = {
    "unit": "#D55E00",
    "network": "#0072B2",
    "indirect": "#E69F00",
    "not": "#C8CDD3",
    "dark": "#222222",
    "gray": "#6E7378",
    "light": "#E8EBEF",
    "panel": "#F7F8FA",
}


SHORT_INFORMATION_LABELS = {
    "Active/passive contrast at function-word period": "A/P contrast\nfunction word",
    "Single-unit temporal dynamics": "Single-unit\ntemporal dynamics",
    "Proposition or semantic-pair identity": "Proposition /\nsemantic pair",
    "Noun-slot or role-like local state": "Noun slot /\nrole-like state",
    "Abstract voice independent of function-word identity": "Abstract voice\nword-independent",
    "Stable time-generalized role code": "Stable\ntime-generalized code",
    "Verb-window active/passive context": "Verb-window\ncontext",
}


def apply_partition_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "savefig.facecolor": "white",
        }
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ptxt(p_value: float | None) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def assign_partition_label(
    *,
    unit_supported: bool,
    network_supported: bool,
    network_gain_supported: bool,
    identifiable: bool,
    network_support_direct: bool = True,
) -> str:
    """Classify the evidence pattern for one information type."""
    if not identifiable:
        return "not_identifiable"
    if unit_supported and network_supported and not network_support_direct:
        return "unit_available_indirect_network"
    if unit_supported and network_supported and network_gain_supported:
        return "mixed_network_enhanced"
    if unit_supported and network_supported:
        return "mixed_unit_and_network"
    if network_supported and network_gain_supported:
        return "network_dominant"
    if network_supported:
        return "network_available"
    if unit_supported:
        return "unit_available"
    return "not_reliably_detected"


def bool_from_p(summary: dict, alpha: float = 0.05) -> bool:
    p_value = summary.get("wilcoxon_p")
    return bool(p_value is not None and np.isfinite(p_value) and p_value < alpha)


def bool_from_p_greater(summary: dict, alpha: float = 0.05) -> bool:
    p_value = summary.get("wilcoxon_p_greater")
    return bool(p_value is not None and np.isfinite(p_value) and p_value < alpha)


def temporal_dynamics_unit_supported(summary: dict, alpha: float = 0.05) -> bool:
    """Require above-chance temporal readout and gain over the scalar mean-rate readout."""
    run_level = summary.get("run_level_summary", {})
    temporal_auc = run_level.get("best_temporal_profile_auc", {})
    temporal_gain = run_level.get("best_temporal_profile_gain_vs_own_mean_rate", {})
    return (
        bool_from_p_greater(temporal_auc, alpha=alpha)
        and float(temporal_auc.get("median", np.nan)) > 0.5
        and bool_from_p_greater(temporal_gain, alpha=alpha)
        and float(temporal_gain.get("median", np.nan)) > 0.0
    )


def make_partition_rows() -> pd.DataFrame:
    nature_stats = read_json(NATURE / "figure_statistics_summary.json")
    net = read_json(NET / "active_passive_network_encoding_summary.json")
    sem = read_json(SEM / "active_passive_semantic_grammatical_summary.json")
    state = read_json(STATE / "active_passive_state_space_rsa_summary.json")
    var = read_json(VAR / "active_passive_variance_partitioning_summary.json")
    time = read_json(TIME / "active_passive_time_generalization_summary.json")
    temporal = read_json(TEMPORAL / "active_passive_single_unit_temporal_dynamics_summary.json")

    clean_good = net["primary"]["artifact_clean_le_0p2/good_only"]
    sem_rank = sem["run_median_signed_rank"]

    rows: list[dict[str, object]] = []

    def add_row(
        information: str,
        operational_test: str,
        unit_supported: bool,
        network_supported: bool,
        network_gain_supported: bool,
        identifiable: bool,
        unit_evidence: str,
        network_evidence: str,
        conclusion: str,
        caveat: str,
        network_support_direct: bool = True,
    ) -> None:
        label = assign_partition_label(
            unit_supported=unit_supported,
            network_supported=network_supported,
            network_gain_supported=network_gain_supported,
            identifiable=identifiable,
            network_support_direct=network_support_direct,
        )
        rows.append(
            {
                "information": information,
                "operational_test": operational_test,
                "unit_supported": unit_supported,
                "network_supported": network_supported,
                "network_support_direct": network_support_direct,
                "network_gain_supported": network_gain_supported,
                "identifiable_in_current_design": identifiable,
                "partition_label": label,
                "partition_label_text": LABEL_TEXT[label],
                "unit_evidence": unit_evidence,
                "network_evidence": network_evidence,
                "conclusion": conclusion,
                "caveat": caveat,
            }
        )

    nested = clean_good["nested_single_unit_auc"]
    best = clean_good["best_population_auc"]
    gain = clean_good["best_population_gain_vs_nested_single_unit"]
    add_row(
        "Active/passive contrast at function-word period",
        "Artifact-clean good-unit decoding of active vs passive trials.",
        unit_supported=bool_from_p(nested),
        network_supported=bool_from_p(best),
        network_gain_supported=bool_from_p(gain),
        identifiable=True,
        unit_evidence=(
            f"Nested single-unit median AUC={nested['median']:.2f}, "
            f"P={ptxt(nested['wilcoxon_p'])}."
        ),
        network_evidence=(
            f"Best population median AUC={best['median']:.2f}, "
            f"P={ptxt(best['wilcoxon_p'])}; median gain={gain['median']:.2f}, "
            f"P={ptxt(gain['wilcoxon_p'])}."
        ),
        conclusion="The task contrast is strongest as a population readout rather than a reliable single-unit code.",
        caveat="This is an active/passive plus function-word/acoustic contrast, not proof of abstract syntax.",
    )

    temporal_summary = temporal["run_level_summary"]
    temporal_auc = temporal_summary["best_temporal_profile_auc"]
    temporal_gain = temporal_summary["best_temporal_profile_gain_vs_own_mean_rate"]
    temporal_best_gain = temporal_summary["best_unit_temporal_gain_auc"]
    time_peak_auc = time["primary_window_summary"]["max_diag_auc_0_to_0p8"]
    time_cross = time["primary_window_summary"]["mean_cross_temporal_auc_0_to_0p8"]
    add_row(
        "Single-unit temporal dynamics",
        "Single-unit temporal-profile decoding compared with scalar mean firing-rate decoding.",
        unit_supported=temporal_dynamics_unit_supported(temporal),
        network_supported=bool_from_p_greater(time_peak_auc),
        network_gain_supported=False,
        identifiable=True,
        unit_evidence=(
            f"Best temporal-profile unit median AUC={temporal_auc['median']:.2f}, "
            f"P={ptxt(temporal_auc['wilcoxon_p_greater'])}; temporal gain over the same unit's "
            f"mean-rate readout median={temporal_gain['median']:.2f}, "
            f"P={ptxt(temporal_gain['wilcoxon_p_greater'])}; best unit gain median="
            f"{temporal_best_gain['median']:.2f}, P={ptxt(temporal_best_gain['wilcoxon_p_greater'])}."
        ),
        network_evidence=(
            f"Population time-resolved decoding has transient peak AUC median={time_peak_auc['median']:.2f}, "
            f"P={ptxt(time_peak_auc['wilcoxon_p_greater'])}, while cross-time generalization remains "
            f"near chance (median={time_cross['median']:.2f}, P={ptxt(time_cross['wilcoxon_p_greater'])})."
        ),
        conclusion="A scalar firing-rate window is incomplete; discriminability depends on response timing.",
        caveat="This is a descriptive temporal readout of the current stimulus contrast, not proof of abstract syntax.",
    )

    semantic = sem_rank["median_cross_voice_semantic_pair_delta"]
    sentence_pair = state["marginal_fraction_sentence_pair"]
    add_row(
        "Proposition or semantic-pair identity",
        "Cross-voice semantic-pair feature model and population marginal variance.",
        unit_supported=bool_from_p(semantic),
        network_supported=bool_from_p(sentence_pair),
        network_gain_supported=False,
        identifiable=True,
        unit_evidence=(
            f"Unit-level cross-voice semantic-pair delta median={semantic['median']:.3f}, "
            f"P={ptxt(semantic['wilcoxon_p'])}."
        ),
        network_evidence=(
            f"Population condition-mean variance is dominated by sentence pair "
            f"(median fraction={sentence_pair['median']:.3f}, P={ptxt(sentence_pair['wilcoxon_p'])})."
        ),
        conclusion="Semantic/proposition identity is unit-available and also a major axis of population state.",
        caveat="Each proposition has unique lexical material, so this is semantic/lexical-proposition information, not pure role syntax.",
    )

    noun_bc = nature_stats["fig3"]["noun_bc_run_fdr"]
    noun_global = nature_stats["fig3"]["noun_bc_global_fdr"]
    noun_raw = nature_stats["fig3"]["noun_raw_run_fdr"]
    heldout_axis = state["cv_syntax_axis_median_diff"]
    frac_positive = state["cv_syntax_axis_frac_pair_positive"]
    add_row(
        "Noun-slot or role-like local state",
        "Paired noun-slot unit tests and leave-one-pair-out population role-axis projection.",
        unit_supported=noun_bc > 0,
        network_supported=bool_from_p(heldout_axis) or bool_from_p(frac_positive),
        network_gain_supported=False,
        identifiable=True,
        network_support_direct=False,
        unit_evidence=(
            f"Baseline-corrected noun-slot: {noun_bc} run-FDR units and {noun_global} global-FDR units; "
            f"raw response: {noun_raw} run-FDR units."
        ),
        network_evidence=(
            f"Indirect population-state support only: held-out active/passive role-axis median diff={heldout_axis['median']:.2f}, "
            f"P={ptxt(heldout_axis['wilcoxon_p'])}; fraction pair-positive median="
            f"{frac_positive['median']:.1f}, P={ptxt(frac_positive['wilcoxon_p'])}."
        ),
        conclusion="Noun-slot information is unit-available; direct network-level noun-slot encoding has not yet been demonstrated.",
        caveat="Population geometry supports a related active/passive role-state axis, but noun role is partly confounded with position and local baseline state.",
    )

    voice_leave = sem_rank["median_leave_pair_grammar_voice_delta"]
    syntax_after_word = var["median_syntax_after_word_identity"]
    rsa_syntax = state["rho_syntax"]
    add_row(
        "Abstract voice independent of function-word identity",
        "Leave-pair voice model, syntax after word identity, and RSA syntax geometry.",
        unit_supported=False,
        network_supported=False,
        network_gain_supported=False,
        identifiable=False,
        unit_evidence=(
            f"Leave-pair unit voice delta median={voice_leave['median']:.3f}, "
            f"P={ptxt(voice_leave['wilcoxon_p'])}."
        ),
        network_evidence=(
            f"Syntax after word identity median={syntax_after_word['median']:.3f}, "
            f"P={ptxt(syntax_after_word['wilcoxon_p_greater'])}; RSA syntax rho median="
            f"{rsa_syntax['median']:.3f}, P={ptxt(rsa_syntax['wilcoxon_p'])}."
        ),
        conclusion="The present design does not prove abstract syntax independent of lexical/function-word identity.",
        caveat="Function-word identity fully determines active/passive class in the current materials.",
    )

    cross_time = time["primary_window_summary"]["mean_cross_temporal_auc_0_to_0p8"]
    peak = time["median_diagonal_peak"]
    add_row(
        "Stable time-generalized role code",
        "Time-resolved diagonal decoding versus cross-temporal generalization.",
        unit_supported=False,
        network_supported=False,
        network_gain_supported=False,
        identifiable=True,
        unit_evidence="No stable single-unit time-generalization result was observed in this analysis set.",
        network_evidence=(
            f"Network decoding peaked transiently at AUC={peak['median_auc']:.2f} around "
            f"{peak['time_sec']:.2f}s, but cross-time median={cross_time['median']:.2f}, "
            f"P={ptxt(cross_time['wilcoxon_p_greater'])}."
        ),
        conclusion="The code is better described as transient than stationary.",
        caveat="This result constrains stability, not whether transient information is meaningful.",
    )

    verb_ap = nature_stats["fig4"]["verb_ap_run_fdr"]
    verb_three = nature_stats["fig4"]["verb_3cond_run_fdr"]
    add_row(
        "Verb-window active/passive context",
        "Verb active/passive and active/passive/scrambled unit-window controls.",
        unit_supported=False,
        network_supported=False,
        network_gain_supported=False,
        identifiable=True,
        unit_evidence=f"Verb A/P: {verb_ap} run-FDR units; verb three-condition: {verb_three} run-FDR units.",
        network_evidence="No positive population-level verb-window claim is used for the manuscript conclusion.",
        conclusion="Verb-window condition information is not reliably detected at the unit level in the current controls.",
        caveat="A negative result does not prove absence of all verb-related information.",
    )

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "information",
        "partition_label_text",
        "unit_evidence",
        "network_evidence",
        "conclusion",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in columns) + " |")
    return "\n".join(lines)


def make_figure(df: pd.DataFrame, out_dir: Path) -> Path:
    apply_partition_style()
    y = np.arange(df.shape[0])
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax.set_xlim(-1.15, 2.65)
    ax.set_ylim(-0.8, df.shape[0] - 0.2)
    ax.invert_yaxis()
    ax.axis("off")

    for row_y in y:
        ax.axhline(row_y, color=FIG_PALETTE["light"], lw=0.8, zorder=0)

    headers = [("Unit", 0), ("Population", 0.88), ("Conclusion", 1.9)]
    for text, x_pos in headers:
        ax.text(
            x_pos,
            -0.55,
            text,
            ha="center" if x_pos >= 0 else "right",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=FIG_PALETTE["dark"],
        )

    def evidence_marker(x_pos: float, y_pos: float, supported: bool, kind: str, indirect: bool = False, not_identifiable: bool = False) -> None:
        if not_identifiable:
            ax.scatter([x_pos], [y_pos], s=150, marker="s", facecolor="#F6E5B3", edgecolor=FIG_PALETTE["indirect"], lw=1.1, zorder=3)
            ax.text(x_pos, y_pos, "n.i.", ha="center", va="center", fontsize=6.4, fontweight="bold", color="#7A4B00", zorder=4)
            return
        if indirect:
            ax.scatter([x_pos], [y_pos], s=168, marker="D", facecolor=FIG_PALETTE["indirect"], edgecolor="white", lw=0.8, zorder=3)
            ax.text(x_pos, y_pos, "ind", ha="center", va="center", fontsize=6.3, fontweight="bold", color="white", zorder=4)
            return
        color = FIG_PALETTE[kind] if supported else "white"
        edge = FIG_PALETTE[kind] if supported else FIG_PALETTE["not"]
        ax.scatter([x_pos], [y_pos], s=168, marker="o", facecolor=color, edgecolor=edge, lw=1.2, zorder=3)
        ax.text(
            x_pos,
            y_pos,
            "yes" if supported else "no",
            ha="center",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color="white" if supported else FIG_PALETTE["gray"],
            zorder=4,
        )

    for i, row in df.iterrows():
        information = str(row["information"])
        row_label = SHORT_INFORMATION_LABELS.get(information, "\n".join(textwrap.wrap(information, 18)))
        ax.text(
            -0.22,
            i,
            row_label,
            ha="right",
            va="center",
            fontsize=7.6,
            color=FIG_PALETTE["dark"],
            linespacing=1.05,
        )

        not_identifiable = not bool(row["identifiable_in_current_design"])
        indirect = str(row["partition_label"]) == "unit_available_indirect_network"
        evidence_marker(0, i, bool(row["unit_supported"]), "unit", not_identifiable=not_identifiable)
        evidence_marker(
            0.88,
            i,
            bool(row["network_supported"]),
            "network",
            indirect=indirect,
            not_identifiable=not_identifiable,
        )

        label = str(row["partition_label"])
        label_text = LABEL_TEXT[label]
        if label == "unit_available_indirect_network":
            label_text = "unit + indirect\npopulation"
        else:
            label_text = "\n".join(textwrap.wrap(label_text, 18))
        ax.text(
            1.9,
            i,
            label_text,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="white",
            linespacing=1.05,
            bbox=dict(boxstyle="round,pad=0.34,rounding_size=0.12", fc=LABEL_COLOR[label], ec="none"),
            zorder=5,
        )

    ax.text(
        -1.10,
        -1.02,
        "Information partition across unit and population readouts",
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=FIG_PALETTE["dark"],
    )
    ax.text(
        -1.10,
        -0.78,
        "Filled markers indicate supported evidence; amber marks indirect or not-identifiable readouts.",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=FIG_PALETTE["gray"],
    )

    path = out_dir / "fig_neuron_network_information_partition.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / "fig_neuron_network_information_partition.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(df: pd.DataFrame, summary: dict[str, object]) -> None:
    lines = [
        "# Neuron versus network information partition",
        "",
        "Conclusion: the hypothesis is partly true, but not as a clean either/or split. "
        "Some information is visible at the unit level, some is strongest in population readout, "
        "single-unit temporal dynamics add information beyond scalar mean firing rate, "
        "and abstract syntax independent of function-word identity is not identifiable in the current materials.",
        "",
        "## Partition table",
        "",
        markdown_table(df),
        "",
        "## Summary JSON",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- Unit-available information: semantic/proposition identity and baseline-corrected noun-slot effects.",
        "- Temporal dynamics: some single units separate active/passive better from response trajectories than from mean firing rate alone.",
        "- Network-dominant information: the active/passive function-word-period contrast.",
        "- Indirect network support: population geometry contains a related active/passive role-state axis, but this is not a direct noun-slot network decoder.",
        "- Not proven: abstract voice or syntax independent of function-word identity, and a stationary time-generalized syntax code.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(_: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = make_partition_rows()
    csv_path = OUT_DIR / "neuron_network_information_partition_table.csv"
    df.to_csv(csv_path, index=False)
    fig_path = make_figure(df, OUT_DIR)
    summary = {
        "partition_table_csv": str(csv_path),
        "figure_png": str(fig_path),
        "report_md": str(REPORT_PATH),
        "n_information_types": int(df.shape[0]),
        "label_counts": df["partition_label"].value_counts().to_dict(),
        "main_result": {
            "partly_true": True,
            "unit_available": df.loc[df["partition_label"].isin(["unit_available", "unit_available_indirect_network", "mixed_unit_and_network", "mixed_network_enhanced"]), "information"].tolist(),
            "network_available": df.loc[df["partition_label"].isin(["network_dominant", "network_available", "mixed_unit_and_network", "mixed_network_enhanced"]), "information"].tolist(),
            "indirect_network_support": df.loc[df["partition_label"] == "unit_available_indirect_network", "information"].tolist(),
            "not_identifiable_or_not_reliably_detected": df.loc[df["partition_label"].isin(["not_identifiable", "not_reliably_detected"]), "information"].tolist(),
        },
    }
    summary_path = OUT_DIR / "neuron_network_information_partition_summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(df, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
