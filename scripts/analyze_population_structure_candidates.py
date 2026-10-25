#!/usr/bin/env python3
"""Explore three non-decoder population-structure candidates for Fig. 2e.

The analyses ask complementary questions about construction-linked population
responses: (1) whether the active-minus-passive axis is coherent across held
sentence pairs, (2) how many population modes are occupied by paired contrast
vectors, and (3) whether residual cross-unit covariance is aligned with a
coding axis estimated from different sentence pairs. Runs are first aggregated
within patient; patients are the biological replicates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from analyze_active_passive_network_encoding import (
    apply_artifact_filter,
    find_er_dirs,
    load_artifact_table,
    load_metrics,
    make_population_matrix,
    stable_seed,
    subject_from_er_dir,
    training_t_scores,
)
from analyze_population_size_scaling import exact_sign_flip_p


OUT = (
    ROOT
    / "neuropixel2025"
    / "data"
    / "combined_active_passive_population_structure_candidates"
)
REPORT = ROOT / "reports" / "active_passive_population_structure_candidates.md"

WIDTH_MM = 90.0
HEIGHT_MM = 58.0
PT = 8.0

BLUE = "#357ABD"
ORANGE = "#D98C3A"
PURPLE = "#7C6BB0"
GRAY = "#777D82"
LIGHT_GRAY = "#D6DBE0"
DARK = "#222222"

PATIENT_COLORS = {
    "gz001": "#0072B2",
    "gz002": "#D55E00",
    "gz003": "#009E73",
    "gz004": "#CC79A7",
    "gz005": "#E69F00",
}


def apply_style() -> None:
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


def pair_id_from_sound_file(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"Cannot infer sentence-pair id from {value}")
    return int(match.group(1))


def standardize_all_trials(x: np.ndarray) -> np.ndarray:
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return (x - center) / scale


def paired_difference_matrix(
    x: np.ndarray,
    y: np.ndarray,
    trial_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    pair_ids = trial_meta["sound_file"].map(pair_id_from_sound_file).to_numpy(int)
    complete_pairs: list[int] = []
    differences: list[np.ndarray] = []
    z = standardize_all_trials(x)
    for pair_id in sorted(np.unique(pair_ids)):
        active = np.flatnonzero((pair_ids == pair_id) & (y == 1))
        passive = np.flatnonzero((pair_ids == pair_id) & (y == 0))
        if len(active) != 1 or len(passive) != 1:
            continue
        complete_pairs.append(int(pair_id))
        differences.append(z[active[0]] - z[passive[0]])
    if not differences:
        return np.empty((0, x.shape[1])), np.empty(0, dtype=int)
    return np.vstack(differences), np.asarray(complete_pairs, dtype=int)


def balanced_half_partitions(n_items: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_items < 4:
        return []
    half = n_items // 2
    all_items = np.arange(n_items, dtype=int)
    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    for combo in itertools.combinations(range(1, n_items), half - 1):
        left = np.asarray((0, *combo), dtype=int)
        right = np.setdiff1d(all_items, left, assume_unique=True)
        if abs(len(left) - len(right)) <= 1:
            partitions.append((left, right))
    return partitions


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(np.dot(a, b) / denom)


def split_half_axis_coherence(
    differences: np.ndarray,
    partitions: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    values: list[float] = []
    for left, right in partitions:
        value = cosine(differences[left].mean(axis=0), differences[right].mean(axis=0))
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def axis_coherence_null(
    differences: np.ndarray,
    partitions: list[tuple[np.ndarray, np.ndarray]],
    n_draws: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    values = np.empty(n_draws, dtype=float)
    for draw in range(n_draws):
        signs = rng.choice((-1.0, 1.0), size=(differences.shape[0], 1))
        values[draw] = split_half_axis_coherence(differences * signs, partitions)
    return values


def participation_ratio(differences: np.ndarray) -> float:
    singular = np.linalg.svd(differences, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    denom = float(np.sum(np.square(energy)))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(np.square(np.sum(energy)) / denom)


def dimensionality_null(
    differences: np.ndarray,
    n_draws: int,
    random_state: int,
) -> np.ndarray:
    """Unit-by-pair sign null preserving every contrast magnitude."""
    rng = np.random.default_rng(random_state)
    values = np.empty(n_draws, dtype=float)
    for draw in range(n_draws):
        signs = rng.choice((-1.0, 1.0), size=differences.shape)
        values[draw] = participation_ratio(differences * signs)
    return values


def condition_residuals(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    residuals = np.empty_like(z, dtype=float)
    for condition in (0, 1):
        rows = np.flatnonzero(y == condition)
        residuals[rows] = z[rows] - z[rows].mean(axis=0, keepdims=True)
    return residuals


def covariance_alignment_index(residuals: np.ndarray, axis: np.ndarray) -> float:
    norm = float(np.linalg.norm(axis))
    if norm <= 0 or not np.isfinite(norm):
        return np.nan
    unit_axis = axis / norm
    projected = residuals @ unit_axis
    total_variance = float(np.var(projected, ddof=1))
    unit_variances = np.var(residuals, axis=0, ddof=1)
    independent_variance = float(np.sum(np.square(unit_axis) * unit_variances))
    denom = total_variance + independent_variance
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float((total_variance - independent_variance) / denom)


def shuffle_residuals_within_condition(
    residuals: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = residuals.copy()
    for column in range(residuals.shape[1]):
        for condition in (0, 1):
            rows = np.flatnonzero(y == condition)
            shuffled[rows, column] = residuals[rng.permutation(rows), column]
    return shuffled


def split_covariance_alignment(
    x: np.ndarray,
    y: np.ndarray,
    trial_meta: pd.DataFrame,
    partitions: list[tuple[np.ndarray, np.ndarray]],
    complete_pair_ids: np.ndarray,
    top_k: int,
    n_null_draws: int,
    random_state: int,
) -> tuple[float, np.ndarray]:
    pair_ids = trial_meta["sound_file"].map(pair_id_from_sound_file).to_numpy(int)
    rng = np.random.default_rng(random_state)
    observed: list[float] = []
    null_by_draw: list[list[float]] = [[] for _ in range(n_null_draws)]

    for left, right in partitions:
        for train_pairs, test_pairs in ((left, right), (right, left)):
            train_ids = complete_pair_ids[train_pairs]
            test_ids = complete_pair_ids[test_pairs]
            train_rows = np.flatnonzero(np.isin(pair_ids, train_ids))
            test_rows = np.flatnonzero(np.isin(pair_ids, test_ids))
            x_train = x[train_rows]
            y_train = y[train_rows]
            x_test = x[test_rows]
            y_test = y[test_rows]
            if min(np.bincount(y_train, minlength=2)) < 2:
                continue
            if min(np.bincount(y_test, minlength=2)) < 2:
                continue

            center = np.mean(x_train, axis=0)
            scale = np.std(x_train, axis=0, ddof=0)
            scale[~np.isfinite(scale) | (scale == 0)] = 1.0
            z_train = (x_train - center) / scale
            z_test = (x_test - center) / scale
            columns = np.argsort(training_t_scores(x_train, y_train))[::-1][
                : min(top_k, x.shape[1])
            ]
            axis = (
                z_train[y_train == 1][:, columns].mean(axis=0)
                - z_train[y_train == 0][:, columns].mean(axis=0)
            )
            residuals = condition_residuals(z_test[:, columns], y_test)
            value = covariance_alignment_index(residuals, axis)
            if not np.isfinite(value):
                continue
            observed.append(value)
            for draw in range(n_null_draws):
                shuffled = shuffle_residuals_within_condition(residuals, y_test, rng)
                null_value = covariance_alignment_index(shuffled, axis)
                if np.isfinite(null_value):
                    null_by_draw[draw].append(null_value)

    null = np.asarray(
        [np.mean(values) if values else np.nan for values in null_by_draw], dtype=float
    )
    return (float(np.mean(observed)) if observed else np.nan), null


def aggregate_candidate(
    observed_runs: pd.DataFrame,
    null_runs: pd.DataFrame,
    observed_column: str,
    null_column: str,
    difference_direction: str,
    alternative: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_patients = (
        observed_runs.groupby("subject", as_index=False)[observed_column]
        .median()
        .rename(columns={observed_column: "observed"})
    )
    null_patient_draws = (
        null_runs.groupby(["subject", "draw"], as_index=False)[null_column]
        .median()
        .rename(columns={null_column: "null"})
    )
    null_patients = (
        null_patient_draws.groupby("subject", as_index=False)["null"]
        .agg(
            null_median="median",
            null_q025=lambda values: float(np.quantile(values, 0.025)),
            null_q975=lambda values: float(np.quantile(values, 0.975)),
        )
    )
    patients = observed_patients.merge(null_patients, on="subject", validate="one_to_one")
    if difference_direction == "observed_minus_null":
        patients["difference"] = patients.observed - patients.null_median
    elif difference_direction == "null_minus_observed":
        patients["difference"] = patients.null_median - patients.observed
    else:
        raise ValueError(f"Unknown difference direction: {difference_direction}")
    differences = patients.difference.to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "n_patients": int(patients.subject.nunique()),
                "observed_patient_median": float(np.median(patients.observed)),
                "observed_patient_q25": float(np.quantile(patients.observed, 0.25)),
                "observed_patient_q75": float(np.quantile(patients.observed, 0.75)),
                "null_patient_median": float(np.median(patients.null_median)),
                "null_patient_q25": float(np.quantile(patients.null_median, 0.25)),
                "null_patient_q75": float(np.quantile(patients.null_median, 0.75)),
                "difference_definition": difference_direction,
                "patient_median_difference": float(np.median(differences)),
                f"exact_sign_flip_p_{alternative.replace('-', '_')}": exact_sign_flip_p(
                    differences, alternative=alternative
                ),
            }
        ]
    )
    return patients, null_patient_draws, summary


def draw_paired_null(
    ax: plt.Axes,
    patients: pd.DataFrame,
    summary: pd.DataFrame,
    title: str,
    ylabel: str,
    colors: tuple[str, str],
    null_label: str,
    reference: float | None = None,
) -> None:
    patients = patients.sort_values("subject")
    for row in patients.itertuples(index=False):
        color = PATIENT_COLORS[str(row.subject)]
        ax.plot([0, 1], [row.observed, row.null_median], color=LIGHT_GRAY, lw=0.9, zorder=1)
        ax.scatter(
            [0, 1],
            [row.observed, row.null_median],
            s=20,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
    medians = [
        float(summary.iloc[0].observed_patient_median),
        float(summary.iloc[0].null_patient_median),
    ]
    for index, (median, color) in enumerate(zip(medians, colors)):
        ax.scatter(
            [index],
            [median],
            marker="D",
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        ax.plot([index - 0.15, index + 0.15], [median, median], color=DARK, lw=1.2)
    if reference is not None:
        ax.axhline(reference, color=GRAY, ls="--", lw=0.8)
    ax.set_xticks([0, 1], ["observed", null_label])
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=4)


def save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
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
    with Image.open(outputs["tiff"]) as tiff_image:
        if tiff_image.mode == "RGBA":
            rgb = Image.new("RGB", tiff_image.size, "white")
            rgb.paste(tiff_image, mask=tiff_image.getchannel("A"))
        else:
            rgb = tiff_image.convert("RGB")
        rgb.save(
            outputs["tiff"],
            dpi=(600, 600),
            compression="tiff_lzw",
        )
    plt.close(fig)
    return outputs


def draw_figures(
    axis_patients: pd.DataFrame,
    axis_summary: pd.DataFrame,
    dim_patients: pd.DataFrame,
    dim_summary: pd.DataFrame,
    covariance_patients: pd.DataFrame,
    covariance_summary: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    apply_style()
    outputs: dict[str, dict[str, str]] = {}

    candidates = [
        (
            axis_patients,
            axis_summary,
            "Cross-pair axis coherence",
            "split-half axis cosine",
            (BLUE, GRAY),
            "pair sign-\nflip null",
            0.0,
            "candidate_e_cross_pair_axis_coherence_90mm",
        ),
        (
            dim_patients,
            dim_summary,
            "Paired-contrast dimensionality",
            "effective population modes",
            (PURPLE, GRAY),
            "unit-by-pair\nsign null",
            None,
            "candidate_e_paired_contrast_dimensionality_90mm",
        ),
        (
            covariance_patients,
            covariance_summary,
            "Signal-noise covariance alignment",
            "covariance alignment index",
            (ORANGE, GRAY),
            "unit-shuffled\nresiduals",
            0.0,
            "candidate_e_signal_noise_alignment_90mm",
        ),
    ]
    for patients, summary, title, ylabel, colors, null_label, reference, stem in candidates:
        fig, ax = plt.subplots(
            figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white"
        )
        fig.subplots_adjust(left=0.21, right=0.98, bottom=0.25, top=0.87)
        draw_paired_null(
            ax, patients, summary, title, ylabel, colors, null_label, reference
        )
        if "dimensionality" in stem:
            ax.set_ylim(1.0, 10.2)
        outputs[stem] = save_figure(fig, OUT / stem)

    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 72 / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.25, top=0.87, wspace=0.58)
    for index, (ax, candidate) in enumerate(zip(axes, candidates)):
        patients, summary, title, ylabel, colors, null_label, reference, _ = candidate
        short_title = (
            "Cross-pair axis coherence"
            if index == 0
            else "Contrast dimensionality"
            if index == 1
            else "Signal-noise alignment"
        )
        draw_paired_null(
            ax, patients, summary, short_title, ylabel, colors, null_label, reference
        )
        if index == 1:
            ax.set_ylim(1.0, 10.2)
        ax.text(
            -0.24,
            1.08,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontsize=PT,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
    outputs["comparison"] = save_figure(
        fig, OUT / "population_structure_candidate_panels_180mm"
    )
    return outputs


def write_report(
    metadata: dict[str, object],
    axis_summary: pd.DataFrame,
    dim_summary: pd.DataFrame,
    covariance_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Non-decoder population-structure candidates for Fig. 2e",
        "",
        "## Figure contract",
        "",
        "- Core conclusion tested: construction-linked population information may be organized as a cross-sentence-pair axis with measurable dimensionality and covariance structure, rather than being described only by decoder accuracy.",
        "- Archetype: quantitative candidate grid; each 90-mm panel can occupy half of a 180-mm row.",
        "- Artifact fraction <=0.2; Kilosort `good` units; baseline-corrected 0-0.8 s response.",
        "- Runs are aggregated by median within patient; n=5 patients are the biological replicates.",
        "- All three analyses are exploratory and are reported together, including null and negative results.",
        "",
        "## Candidate 1: cross-pair coding-axis coherence",
        "",
        "The ten matched sentence pairs are split into balanced halves. Active-minus-passive population axes are estimated separately and compared by cosine similarity, averaged over all unique balanced partitions. The paired sign-flip null randomly swaps active/passive direction within sentence pairs while preserving each pair's population-vector magnitude.",
        "",
        axis_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Candidate 2: paired-contrast effective dimensionality",
        "",
        "The ten sentence-pair active-minus-passive population vectors form a pair-by-unit matrix. Its singular-value participation ratio estimates the effective number of occupied population modes (range 1-10 here). The unit-by-pair sign null preserves every contrast magnitude but breaks coherent condition direction across units and pairs. This is a structural diagnostic, not an intrinsic dimensionality estimate for unrestricted neural activity.",
        "",
        dim_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Candidate 3: signal-noise covariance alignment",
        "",
        "For each balanced pair split, a top-5 condition axis is selected and estimated from the training pairs only. Residual variability is evaluated on disjoint test pairs after condition-mean removal. The bounded alignment index compares variance along the axis with the variance expected from independent units. Independent within-condition trial shuffling destroys cross-unit covariance while preserving unit variances. Positive values mean shared variability increases noise along the readout axis; negative values mean it suppresses that noise.",
        "",
        covariance_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation boundary",
        "",
        "The current stimulus set collinearizes construction, function-word identity and associated acoustics. These panels therefore concern construction-linked population geometry and covariance, not abstract voice coding. Each sentence occurs once per run, so cross-pair coherence is cross-stimulus generalization rather than repeated-stimulus reliability.",
        "",
        "## Broader population-analysis landscape",
        "",
        "| Family | Analysis | Distinct question | Current-data feasibility | Non-redundant figure form | Priority |",
        "|---|---|---|---|---|---|",
        "| Ensemble dependence | Nested top-unit ablation | Does readout depend disproportionately on a small subset? | Completed | Ranked-vs-random removal curve | Main-text candidate |",
        "| Geometry | Cross-pair axis coherence | Does one signed contrast direction recur across sentence pairs? | Completed here | Observed-vs-paired-sign-null slope plot | Supplement |",
        "| Geometry | Paired-contrast participation ratio | Are pairwise contrast vectors low- or high-dimensional? | Completed here; rank capped at 10 | Effective modes vs structural null | Supplement |",
        "| Covariance | Signal-noise alignment | Does shared trial variability amplify or suppress noise along the readout axis? | Completed here; scalar covariance only | Alignment index vs unit-shuffled residuals | Supplement |",
        "| Ensemble stability | Fold-wise weight/selection stability | Are the same units repeatedly recruited by the decoder? | High | Jaccard/cosine stability vs label-null | Strong supplement candidate |",
        "| Generalization | Sentence-pair CCGP matrix | Which sentence pairs share a transferable population boundary? | Moderate; only one presentation per exact sentence | 10 x 10 train-test matrix plus patient summary | Supplement; overlaps leave-pair-out |",
        "| Dynamics | Latent trajectories, speed and curvature | Do conditions traverse different paths, not merely different endpoints? | Moderate-high from time-binned spikes | PCA/FA trajectories plus speed or turning-angle trace | Potential main panel if temporal story is central |",
        "| Dynamics | Time-varying coding-axis angle | Does the population code rotate or remain stable over 0-0.8 s? | High | angle/time or compact temporal-generalization summary | Supplement; partly overlaps time-generalization |",
        "| Shared variability | Residual covariance eigenspectrum/factor analysis | How many shared noise modes remain after condition and pair means are removed? | Moderate; residual rank <=18 per run | Scree/participation-ratio paired plot | Supplement |",
        "| Temporal organization | Unit-latency sequence stability | Is there a reproducible recruitment order across runs/patients? | Moderate; selection must be nested | sorted latency raster plus split-half rank correlation | Supplement |",
        "| Topology | Population-coupling communities | Are there stable co-fluctuating ensembles beyond response archetypes? | Low-moderate with 20 trials | consensus-community matrix | Exploratory supplement only |",
        "| Information structure | Synergy/redundancy or PID | Do unit combinations carry information unavailable to single units? | Low; trial counts are inadequate for reliable high-order estimates | redundancy-synergy spectrum | Not recommended for main text |",
        "| Mixed selectivity | Condition x sentence-pair interactions | Are units conjunctively selective for construction and content? | Not cleanly identifiable: one trial per condition-pair cell | coefficient map | Not recommended with current design |",
        "| Cross-patient alignment | Hyperalignment/shared manifold | Is a common geometry conserved across patients? | High risk: different sites and unmatched units | aligned-subspace similarity matrix | Not recommended without anatomical/functional anchors |",
        "| Pseudo-population | Pool units across patients | How would a much larger synthetic population decode? | Technically possible but destroys simultaneous covariance | scaling curve | Illustration only, never biological inference |",
        "",
        "## Literature anchors for the analysis families",
        "",
        "- Steinmetz et al., Nature 2019 (doi:10.1038/s41586-019-1787-x): distributed brain-wide population coding and trial-level engagement signals.",
        "- Stringer et al., Nature 2019 (doi:10.1038/s41586-019-1346-5): population dimensionality and eigenspectrum geometry.",
        "- Jamali et al., Nature 2024 (doi:10.1038/s41586-024-07643-2): context-dependent human single-cell ensembles and hierarchical population organization.",
        "- Tostado-Marcos et al., J Neurosci 2026 (doi:10.1523/JNEUROSCI.0580-25.2026): low-dimensional latent trajectories and generalization to novel neuronal subpopulations.",
        "- Lozano et al., Nature Neuroscience 2026 (doi:10.1038/s41593-026-02314-z): low-dimensional state-transition trajectories and opposing subpopulations.",
        "- Johnston et al., J Neurosci 2026 (doi:10.1523/JNEUROSCI.0185-26.2026): time-resolved Neuropixels decoding and representational-similarity clustering.",
        "- Hickman et al., J Neurophysiol 2026 (doi:10.1152/jn.00382.2025): low-dimensional shared spike-count variability and its information-enhancing or information-limiting effects.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    observed_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for er_dir in find_er_dirs():
        subject = subject_from_er_dir(er_dir)
        metrics = load_metrics(er_dir)
        artifact = load_artifact_table(er_dir)
        filtered = apply_artifact_filter(metrics, artifact, args.artifact_threshold)
        for run in sorted(metrics.run.unique()):
            x, y, trial_meta, n_units_input, n_units_variable = make_population_matrix(
                filtered, run, "good_only"
            )
            n_active = int(np.sum(y == 1))
            n_passive = int(np.sum(y == 0))
            common = {
                "subject": subject,
                "run": run,
                "n_trials": int(len(y)),
                "n_active": n_active,
                "n_passive": n_passive,
                "n_units_input": int(n_units_input),
                "n_units_variable": int(n_units_variable),
            }
            if (
                len(y) < args.min_trials_total
                or min(n_active, n_passive) < args.min_trials_per_class
                or n_units_variable < args.min_units
            ):
                skipped_rows.append({**common, "reason": "insufficient trials or units"})
                continue

            differences, complete_pair_ids = paired_difference_matrix(x, y, trial_meta)
            partitions = balanced_half_partitions(len(complete_pair_ids))
            if len(complete_pair_ids) < args.min_complete_pairs or not partitions:
                skipped_rows.append(
                    {
                        **common,
                        "n_complete_pairs": int(len(complete_pair_ids)),
                        "reason": "insufficient complete sentence pairs",
                    }
                )
                continue

            axis_seed = args.random_state + stable_seed(subject, run, "axis") % 1000000
            dim_seed = args.random_state + stable_seed(subject, run, "dim") % 1000000
            covariance_seed = (
                args.random_state + stable_seed(subject, run, "covariance") % 1000000
            )
            axis_observed = split_half_axis_coherence(differences, partitions)
            axis_null = axis_coherence_null(
                differences, partitions, args.null_draws, axis_seed
            )
            dim_observed = participation_ratio(differences)
            dim_null = dimensionality_null(
                differences, args.null_draws, dim_seed
            )
            covariance_observed, covariance_null = split_covariance_alignment(
                x,
                y,
                trial_meta,
                partitions,
                complete_pair_ids,
                top_k=args.top_k,
                n_null_draws=args.null_draws,
                random_state=covariance_seed,
            )
            observed_rows.append(
                {
                    **common,
                    "n_complete_pairs": int(len(complete_pair_ids)),
                    "n_balanced_partitions": int(len(partitions)),
                    "axis_coherence": axis_observed,
                    "paired_contrast_effective_modes": dim_observed,
                    "signal_noise_alignment": covariance_observed,
                }
            )
            for draw in range(args.null_draws):
                null_rows.append(
                    {
                        **common,
                        "n_complete_pairs": int(len(complete_pair_ids)),
                        "draw": int(draw),
                        "axis_coherence_null": float(axis_null[draw]),
                        "paired_contrast_effective_modes_null": float(dim_null[draw]),
                        "signal_noise_alignment_null": float(covariance_null[draw]),
                    }
                )

    observed_runs = pd.DataFrame(observed_rows).sort_values(["subject", "run"])
    null_runs = pd.DataFrame(null_rows).sort_values(["subject", "run", "draw"])
    skipped = pd.DataFrame(skipped_rows)
    if observed_runs.empty:
        raise RuntimeError("No eligible runs for population-structure analyses")
    if observed_runs.subject.nunique() != 5:
        raise RuntimeError(f"Expected five patients, found {observed_runs.subject.nunique()}")

    axis_patients, axis_patient_draws, axis_summary = aggregate_candidate(
        observed_runs,
        null_runs,
        observed_column="axis_coherence",
        null_column="axis_coherence_null",
        difference_direction="observed_minus_null",
        alternative="greater",
    )
    dim_patients, dim_patient_draws, dim_summary = aggregate_candidate(
        observed_runs,
        null_runs,
        observed_column="paired_contrast_effective_modes",
        null_column="paired_contrast_effective_modes_null",
        difference_direction="null_minus_observed",
        alternative="greater",
    )
    covariance_patients, covariance_patient_draws, covariance_summary = aggregate_candidate(
        observed_runs,
        null_runs,
        observed_column="signal_noise_alignment",
        null_column="signal_noise_alignment_null",
        difference_direction="observed_minus_null",
        alternative="two-sided",
    )

    tables = {
        "population_structure_run_values": observed_runs,
        "population_structure_null_run_draws": null_runs,
        "axis_coherence_patient_values": axis_patients,
        "axis_coherence_patient_null_draws": axis_patient_draws,
        "axis_coherence_group_summary": axis_summary,
        "paired_contrast_dimensionality_patient_values": dim_patients,
        "paired_contrast_dimensionality_patient_null_draws": dim_patient_draws,
        "paired_contrast_dimensionality_group_summary": dim_summary,
        "signal_noise_alignment_patient_values": covariance_patients,
        "signal_noise_alignment_patient_null_draws": covariance_patient_draws,
        "signal_noise_alignment_group_summary": covariance_summary,
        "skipped_runs": skipped,
    }
    source_data: dict[str, str] = {}
    for name, table in tables.items():
        path = OUT / f"{name}.csv"
        table.to_csv(path, index=False)
        source_data[name] = str(path)

    figures = draw_figures(
        axis_patients,
        axis_summary,
        dim_patients,
        dim_summary,
        covariance_patients,
        covariance_summary,
    )
    metadata: dict[str, object] = {
        "analysis": "non-decoder population-structure candidates for Fig. 2e",
        "status": "exploratory",
        "analysis_mode": "artifact_clean_le_0p2",
        "unit_set": "good_only",
        "metric": "response_baseline_corrected_hz",
        "artifact_threshold": args.artifact_threshold,
        "top_k_for_covariance_axis": args.top_k,
        "null_draws": args.null_draws,
        "random_state": args.random_state,
        "n_patients": int(observed_runs.subject.nunique()),
        "n_runs": int(len(observed_runs)),
        "n_complete_sentence_pairs_per_run": sorted(
            observed_runs.n_complete_pairs.astype(int).unique().tolist()
        ),
        "axis_coherence": axis_summary.iloc[0].to_dict(),
        "paired_contrast_dimensionality": dim_summary.iloc[0].to_dict(),
        "signal_noise_alignment": covariance_summary.iloc[0].to_dict(),
        "source_data": source_data,
        "figure_outputs": figures,
        "reviewer_risks": [
            "All three analyses are exploratory and must be reported together rather than selected by significance.",
            "Patients, not runs, sentence pairs, partitions or null draws, are the biological replicates.",
            "Each exact sentence occurs once per run, so axis coherence is cross-sentence-pair generalization rather than repeated-stimulus reliability.",
            "Construction, function-word identity and associated acoustics covary in the current stimulus set.",
            "The paired-contrast participation ratio is bounded by ten sentence pairs and uses a structural unit-by-pair sign null.",
            "The covariance estimate is regularized by a scalar quadratic form; it is not a full-rank noise-covariance estimate.",
        ],
    }
    metadata_path = OUT / "population_structure_candidate_panels_metadata.json"
    metadata["metadata_json"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metadata, axis_summary, dim_summary, covariance_summary)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--min-trials-total", type=int, default=16)
    parser.add_argument("--min-trials-per-class", type=int, default=7)
    parser.add_argument("--min-units", type=int, default=10)
    parser.add_argument("--min-complete-pairs", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    result = run_analysis(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
