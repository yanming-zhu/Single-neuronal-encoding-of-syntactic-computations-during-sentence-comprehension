#!/usr/bin/env python3
"""Compare single-unit and population transfer to unseen sentence pairs.

Both models use the same leave-one-sentence-pair-out outer folds.  Unit
selection is recomputed using training trials only: the single-unit model uses
the top training-fold unit and the population model uses the top five units.
Runs are aggregated within patient before group summaries or inference.
"""

from __future__ import annotations

import argparse
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
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score

from analyze_three_paper_borrowed_methods_20260724 import (
    POP_PATH,
    TRIAL_PATH,
    fit_topk_predict,
    global_pair_swap_labels,
    grouped_folds,
    load_response_metrics,
    make_response_matrix,
    pair_id,
)


OUT = (
    ROOT
    / "neuropixel2025"
    / "data"
    / "combined_active_passive_sentence_pair_transfer"
)
REPORT = ROOT / "reports" / "unit_population_sentence_pair_transfer.md"
OUT.mkdir(parents=True, exist_ok=True)

PT = 8.0
DARK = "#222222"
GRAY = "#777D82"
LIGHT_GRAY = "#D6DBE0"
UNIT_COLOR = "#777D82"
POP_COLOR = "#1E6A58"
PATIENT_COLORS = {
    "gz001": "#0072B2",
    "gz002": "#D55E00",
    "gz003": "#009E73",
    "gz004": "#CC79A7",
    "gz005": "#E69F00",
}
MODEL_SPECS = (
    ("nested_single_unit", "nested single unit", 1, 1.0),
    ("fixed_top5_population", "fixed top-5 population", 5, 0.1),
)


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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def grouped_decode(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    original_y: np.ndarray,
    k: int,
    c_value: float,
) -> float:
    probability_sum = np.zeros(len(y), dtype=float)
    probability_count = np.zeros(len(y), dtype=float)
    for train, test in grouped_folds(meta, original_y, "sentence_pair"):
        if len(np.unique(y[train])) < 2:
            continue
        probabilities = fit_topk_predict(
            x[train], y[train], x[test], k=k, c_value=c_value
        )
        probability_sum[test] += probabilities
        probability_count[test] += 1
    valid = probability_count > 0
    if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
        return float("nan")
    return float(
        roc_auc_score(y[valid], probability_sum[valid] / probability_count[valid])
    )


def exact_sign_p(values: np.ndarray, reference: float) -> tuple[int, float]:
    differences = np.asarray(values, dtype=float) - reference
    differences = differences[np.isfinite(differences) & (np.abs(differences) > 1e-12)]
    if len(differences) == 0:
        return 0, float("nan")
    n_positive = int(np.sum(differences > 0))
    p_value = float(
        stats.binomtest(n_positive, len(differences), 0.5, alternative="greater").pvalue
    )
    return n_positive, p_value


def load_run_data() -> list[dict[str, object]]:
    eligible = pd.read_csv(POP_PATH)[["subject", "run"]].drop_duplicates()
    metrics = load_response_metrics()
    trials = pd.read_csv(TRIAL_PATH)
    run_data: list[dict[str, object]] = []
    for row in eligible.itertuples(index=False):
        x, y, meta, unit_ids = make_response_matrix(
            metrics, trials, str(row.subject), str(row.run)
        )
        run_data.append(
            {
                "subject": str(row.subject),
                "run": str(row.run),
                "x": x,
                "y": y,
                "meta": meta,
                "n_units": int(len(unit_ids)),
            }
        )
    return run_data


def analyze(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_data = load_run_data()
    actual_rows: list[dict[str, object]] = []
    for item in run_data:
        for model, label, k, c_value in MODEL_SPECS:
            auc = grouped_decode(
                item["x"], item["y"], item["meta"], item["y"], k, c_value
            )
            actual_rows.append(
                {
                    "subject": item["subject"],
                    "run": item["run"],
                    "model": model,
                    "model_label": label,
                    "n_units": item["n_units"],
                    "auc": auc,
                }
            )
    run_values = pd.DataFrame(actual_rows).sort_values(["subject", "run", "model"])
    patient_values = (
        run_values.groupby(["subject", "model", "model_label"], as_index=False).auc.median()
        .sort_values(["model", "subject"])
    )

    pair_values = sorted(pd.read_csv(TRIAL_PATH).sound_file.map(pair_id).unique())
    rng = np.random.default_rng(args.random_state)
    null_rows: list[dict[str, object]] = []
    for permutation in range(args.n_permutations):
        swap = {
            int(pair): bool(value)
            for pair, value in zip(
                pair_values, rng.integers(0, 2, size=len(pair_values))
            )
        }
        permutation_rows: list[dict[str, object]] = []
        for item in run_data:
            permuted_y = global_pair_swap_labels(
                item["y"], item["meta"].pair_id.to_numpy(int), swap
            )
            for model, label, k, c_value in MODEL_SPECS:
                auc = grouped_decode(
                    item["x"],
                    permuted_y,
                    item["meta"],
                    item["y"],
                    k,
                    c_value,
                )
                permutation_rows.append(
                    {
                        "subject": item["subject"],
                        "model": model,
                        "model_label": label,
                        "auc": auc,
                    }
                )
        permutation_frame = pd.DataFrame(permutation_rows)
        patient_null = permutation_frame.groupby(
            ["subject", "model", "model_label"], as_index=False
        ).auc.median()
        medians = patient_null.groupby(["model", "model_label"]).auc.median()
        unit_null = float(medians.loc[("nested_single_unit", "nested single unit")])
        population_null = float(
            medians.loc[("fixed_top5_population", "fixed top-5 population")]
        )
        null_rows.extend(
            [
                {
                    "permutation": permutation,
                    "model": "nested_single_unit",
                    "model_label": "nested single unit",
                    "group_patient_median_auc": unit_null,
                },
                {
                    "permutation": permutation,
                    "model": "fixed_top5_population",
                    "model_label": "fixed top-5 population",
                    "group_patient_median_auc": population_null,
                },
            ]
        )
        if args.progress_every > 0 and (permutation + 1) % args.progress_every == 0:
            print(f"completed {permutation + 1}/{args.n_permutations} permutations", flush=True)
    permutation_null = pd.DataFrame(null_rows)

    summary_rows: list[dict[str, object]] = []
    for model, label, _, _ in MODEL_SPECS:
        values = patient_values.loc[patient_values.model == model, "auc"].to_numpy(float)
        observed = float(np.median(values))
        null = permutation_null.loc[
            permutation_null.model == model, "group_patient_median_auc"
        ].to_numpy(float)
        positive_patients, sign_p = exact_sign_p(values, 0.5)
        permutation_p = float(
            (1 + np.sum(null >= observed - 1e-12)) / (1 + len(null))
        )
        summary_rows.append(
            {
                "comparison": model,
                "label": label,
                "reference": 0.5,
                "n_patients": int(len(values)),
                "patient_median": observed,
                "patient_min": float(np.min(values)),
                "patient_max": float(np.max(values)),
                "positive_patients": positive_patients,
                "exact_sign_p_greater": sign_p,
                "group_permutation_p_greater": permutation_p,
                "n_group_permutations": int(len(null)),
            }
        )

    wide = patient_values.pivot(index="subject", columns="model", values="auc")
    patient_gain = (
        wide["fixed_top5_population"] - wide["nested_single_unit"]
    ).to_numpy(float)
    observed_gain = float(np.median(patient_gain))
    null_wide = permutation_null.pivot(
        index="permutation", columns="model", values="group_patient_median_auc"
    )
    null_gain = (
        null_wide["fixed_top5_population"] - null_wide["nested_single_unit"]
    ).to_numpy(float)
    gain_positive, gain_sign_p = exact_sign_p(patient_gain, 0.0)
    summary_rows.append(
        {
            "comparison": "population_minus_single_unit",
            "label": "population minus nested single unit",
            "reference": 0.0,
            "n_patients": int(len(patient_gain)),
            "patient_median": observed_gain,
            "patient_min": float(np.min(patient_gain)),
            "patient_max": float(np.max(patient_gain)),
            "positive_patients": gain_positive,
            "exact_sign_p_greater": gain_sign_p,
            "group_permutation_p_greater": float(
                (1 + np.sum(null_gain >= observed_gain - 1e-12))
                / (1 + len(null_gain))
            ),
            "n_group_permutations": int(len(null_gain)),
        }
    )
    inference = pd.DataFrame(summary_rows)
    return run_values, patient_values, permutation_null, inference


def density_polygon(values: np.ndarray, center: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    x = np.linspace(0.25, 0.86, 240)
    if np.std(values) <= 1e-12:
        density = np.zeros_like(x)
    else:
        density = stats.gaussian_kde(values, bw_method=0.22)(x)
    if density.max() > 0:
        density = 0.22 * density / density.max()
    return x, center + density


def draw_figure(
    patient_values: pd.DataFrame,
    permutation_null: pd.DataFrame,
    inference: pd.DataFrame,
) -> dict[str, str]:
    apply_style()
    wide = patient_values.pivot(index="subject", columns="model", values="auc").sort_index()
    matrix = wide[["nested_single_unit", "fixed_top5_population"]].to_numpy(float)
    group_median = np.median(matrix, axis=0, keepdims=True)
    display = np.vstack([matrix, group_median])

    fig = plt.figure(figsize=(90 / 25.4, 58 / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.88, 1.18],
        left=0.17,
        right=0.98,
        bottom=0.22,
        top=0.84,
        wspace=0.75,
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_null = fig.add_subplot(grid[0, 1])
    image = ax_heat.imshow(
        display,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=0.86),
        aspect="auto",
        interpolation="nearest",
    )
    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            value = display[row, column]
            ax_heat.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value - 0.5) > 0.17 else DARK,
                fontsize=7.2,
                fontweight="bold" if row == display.shape[0] - 1 else "normal",
            )
    ax_heat.axhline(4.5, color="white", lw=1.2)
    ax_heat.set_xticks([0, 1], ["single\nunit", "top-5\npopulation"])
    ax_heat.set_yticks(np.arange(6), [*wide.index.tolist(), "median"])
    ax_heat.set_title("Patient AUC", loc="left", fontweight="bold", pad=4)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    order = [
        ("nested_single_unit", "single unit", UNIT_COLOR, 1.0),
        ("fixed_top5_population", "top-5 population", POP_COLOR, 0.0),
    ]
    for model, label, color, center in order:
        null = permutation_null.loc[
            permutation_null.model == model, "group_patient_median_auc"
        ].to_numpy(float)
        x, upper = density_polygon(null, center)
        ax_null.fill_between(x, center, upper, color=color, alpha=0.25, lw=0)
        ax_null.plot(x, upper, color=color, lw=1.0)
        row = inference[inference.comparison == model].iloc[0]
        observed = float(row.patient_median)
        ax_null.scatter(
            observed,
            center,
            marker="D",
            s=31,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
        ax_null.text(
            0.855,
            center + 0.08,
            f"P={row.group_permutation_p_greater:.3f}",
            ha="right",
            va="bottom",
            color=color,
            fontsize=7.2,
        )
    ax_null.axvline(0.5, color=GRAY, ls="--", lw=0.8)
    ax_null.set_xlim(0.25, 0.86)
    ax_null.set_ylim(-0.16, 1.32)
    ax_null.set_yticks([1.0, 0.0], ["unit", "population"])
    ax_null.set_xlabel("patient-median AUC")
    ax_null.set_title("Paired-label null", loc="left", fontweight="bold", pad=4)
    ax_null.spines["left"].set_visible(False)
    ax_null.tick_params(axis="y", length=0)

    fig.suptitle(
        "Transfer to unseen sentence pairs",
        x=0.17,
        y=0.97,
        ha="left",
        fontsize=PT,
        fontweight="bold",
    )
    stem = OUT / "unit_population_sentence_pair_transfer_90mm"
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
        rgb = Image.new("RGB", source.size, "white")
        if source.mode == "RGBA":
            rgb.paste(source, mask=source.getchannel("A"))
        else:
            rgb.paste(source.convert("RGB"))
        rgb.save(outputs["tiff"], dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)
    return outputs


def write_report(metadata: dict[str, object]) -> None:
    summary = pd.DataFrame(metadata["inference"]).set_index("comparison")
    unit = summary.loc["nested_single_unit"]
    population = summary.loc["fixed_top5_population"]
    gain = summary.loc["population_minus_single_unit"]
    lines = [
        "# Unit and population transfer to unseen sentence pairs",
        "",
        "Both readouts use identical leave-one-sentence-pair-out folds. Unit selection is repeated inside each training fold. Runs are aggregated within patient; n=5 patients are the biological replicates.",
        "",
        "## Results",
        "",
        f"- Nested single unit: patient-median AUC={unit.patient_median:.3f}; {int(unit.positive_patients)}/5 patients above chance; synchronized pair-label permutation P={unit.group_permutation_p_greater:.3f}.",
        f"- Fixed top-5 population: patient-median AUC={population.patient_median:.3f}; {int(population.positive_patients)}/5 patients above chance; synchronized pair-label permutation P={population.group_permutation_p_greater:.3f}.",
        f"- Population minus unit: patient-median difference={gain.patient_median:+.3f}; {int(gain.positive_patients)}/5 positive; synchronized permutation P={gain.group_permutation_p_greater:.3f}.",
        "",
        "## Interpretation",
        "",
        "A sentence-pair-generalizing signal is supported at the compact population level. The same analysis does not establish reliable group-level transfer for a nested single-unit readout. Active/passive construction, function-word identity and associated acoustics covary, so the result is construction-linked rather than an abstract voice code.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    run_values, patient_values, permutation_null, inference = analyze(args)
    run_path = OUT / "sentence_pair_transfer_run_values.csv"
    patient_path = OUT / "sentence_pair_transfer_patient_values.csv"
    null_path = OUT / "sentence_pair_transfer_group_permutation_null.csv"
    inference_path = OUT / "sentence_pair_transfer_inference.csv"
    run_values.to_csv(run_path, index=False)
    patient_values.to_csv(patient_path, index=False)
    permutation_null.to_csv(null_path, index=False)
    inference.to_csv(inference_path, index=False)
    outputs = draw_figure(patient_values, permutation_null, inference)
    metadata: dict[str, object] = {
        "analysis": "unit and population leave-one-sentence-pair-out transfer",
        "status": "exploratory",
        "n_patients": int(patient_values.subject.nunique()),
        "n_runs": int(run_values[["subject", "run"]].drop_duplicates().shape[0]),
        "n_sentence_pairs": 10,
        "n_group_permutations": args.n_permutations,
        "single_unit_rule": "top absolute training-fold t-score; L2 logistic C=1.0",
        "population_rule": "top five absolute training-fold t-scores; L2 logistic C=0.1",
        "inference": inference.to_dict(orient="records"),
        "source_data": {
            "run_values": str(run_path),
            "patient_values": str(patient_path),
            "permutation_null": str(null_path),
            "inference": str(inference_path),
        },
        "figure_outputs": outputs,
        "interpretation_boundary": (
            "Construction, function-word identity and associated acoustics covary; "
            "patients, not runs or units, are biological replicates."
        ),
    }
    metadata_path = OUT / "sentence_pair_transfer_metadata.json"
    metadata["metadata_json"] = str(metadata_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=499)
    parser.add_argument("--random-state", type=int, default=20260806)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
