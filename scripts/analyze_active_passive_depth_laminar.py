#!/usr/bin/env python3
"""Individual-patient depth/pseudo-laminar organization of active/passive effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025/data"
OUT_DIR = DATA_ROOT / "combined_active_passive_depth_laminar"
REPORT_PATH = ROOT / "reports/active_passive_depth_laminar.md"
METRIC = "response_baseline_corrected_hz"
BANDS = ["superficial", "middle", "deep"]


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def simple_condition(value: object) -> str | None:
    text = str(value)
    if text.startswith("active"):
        return "active"
    if text.startswith("passive"):
        return "passive"
    return None


def subject_from_er_dir(er_dir: Path) -> str:
    return er_dir.relative_to(DATA_ROOT).parts[0]


def find_event_dirs() -> list[Path]:
    return sorted(DATA_ROOT.glob("gz*/processed/ActPasSynSentence/event_related_active_passive"))


def pair_id_from_sound_file(sound_file: str) -> int:
    match = re.search(r"(?:act|pass)(\d+)\.mp3$", str(sound_file))
    if not match:
        raise ValueError(f"Cannot infer pair id from {sound_file}")
    return int(match.group(1))


def load_metrics(er_dir: Path) -> pd.DataFrame:
    subject = subject_from_er_dir(er_dir)
    metrics = pd.read_csv(er_dir / "active_passive_unit_trial_response_metrics.csv")
    metrics["subject"] = subject
    metrics["condition_simple"] = metrics["condition"].map(simple_condition)
    metrics = metrics[metrics["condition_simple"].isin(["active", "passive"])].copy()
    metrics["pair_id"] = metrics["sound_file"].map(pair_id_from_sound_file)
    return metrics


def load_depth_metadata(er_dir: Path) -> pd.DataFrame:
    subject = subject_from_er_dir(er_dir)
    path = er_dir / "depth_activation" / f"{subject}_unit_depth_activation_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    depth = pd.read_csv(path)
    cols = [
        "run",
        "unit_id",
        "x_um_median",
        "depth_um_median",
        "depth_um_p10",
        "depth_um_p90",
        "amplitude_median",
    ]
    cols = [col for col in cols if col in depth.columns]
    depth = depth[cols].drop_duplicates(["run", "unit_id"]).copy()
    depth["subject"] = subject
    return depth


def load_artifact_table(er_dir: Path) -> pd.DataFrame | None:
    path = er_dir / "trial_artifact_overlap_minus1_to_plus2s.csv"
    if not path.exists():
        return None
    artifact = pd.read_csv(path)
    artifact["condition_simple"] = artifact["condition"].map(simple_condition)
    artifact = artifact[artifact["condition_simple"].isin(["active", "passive"])].copy()
    if artifact.empty:
        return None
    return artifact[["run", "trial_index", "artifact_fraction_minus1_to_plus2s"]].drop_duplicates(
        ["run", "trial_index"]
    )


def apply_filters(
    metrics: pd.DataFrame,
    artifact: pd.DataFrame | None,
    unit_set: str,
    artifact_threshold: float | None,
) -> pd.DataFrame:
    out = metrics.copy()
    if unit_set == "good_only":
        out = out[out["ks_label"].astype(str) == "good"].copy()
    elif unit_set != "all_clusters":
        raise ValueError(f"Unsupported unit_set: {unit_set}")
    if artifact_threshold is not None and artifact is not None:
        out = out.merge(artifact, on=["run", "trial_index"], how="left")
        keep = out["artifact_fraction_minus1_to_plus2s"].isna() | (
            out["artifact_fraction_minus1_to_plus2s"] <= artifact_threshold
        )
        out = out.loc[keep].copy()
    else:
        out["artifact_fraction_minus1_to_plus2s"] = np.nan
    return out


def cohens_d(active: np.ndarray, passive: np.ndarray) -> float:
    if len(active) < 2 or len(passive) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(active) - 1) * np.var(active, ddof=1) + (len(passive) - 1) * np.var(passive, ddof=1))
        / max(len(active) + len(passive) - 2, 1)
    )
    return float((np.mean(active) - np.mean(passive)) / pooled) if pooled > 0 else np.nan


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    p_finite = p[finite]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    n = len(ranked)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q_finite = np.empty_like(adjusted)
    q_finite[order] = adjusted
    q[finite] = q_finite
    return q


def compute_unit_effects(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (run, unit_id), unit_df in run_df.groupby(["run", "unit_id"]):
        active = unit_df.loc[unit_df["condition_simple"] == "active", METRIC].to_numpy(float)
        passive = unit_df.loc[unit_df["condition_simple"] == "passive", METRIC].to_numpy(float)
        if len(active) == 0 or len(passive) == 0:
            continue
        if len(active) >= 2 and len(passive) >= 2:
            try:
                p_value = float(stats.mannwhitneyu(active, passive, alternative="two-sided").pvalue)
            except ValueError:
                p_value = np.nan
        else:
            p_value = np.nan
        active_mean = float(np.mean(active))
        passive_mean = float(np.mean(passive))
        diff = active_mean - passive_mean
        rows.append(
            {
                "subject": unit_df["subject"].iloc[0],
                "run": run,
                "unit_id": int(unit_id),
                "ks_label": unit_df["ks_label"].astype(str).mode().iloc[0],
                "n_active_trials": int(len(active)),
                "n_passive_trials": int(len(passive)),
                "active_mean_bc_hz": active_mean,
                "passive_mean_bc_hz": passive_mean,
                "active_minus_passive_bc_hz": diff,
                "abs_active_passive_diff_hz": abs(diff),
                "cohens_d": cohens_d(active, passive),
                "p_value": p_value,
                "direction": "active_gt_passive" if diff > 0 else ("passive_gt_active" if diff < 0 else "no_difference"),
                "minus_log10_p": float(-np.log10(max(p_value, np.finfo(float).tiny))) if np.isfinite(p_value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_fdr_columns(units: pd.DataFrame) -> pd.DataFrame:
    out = units.copy()
    out["q_value_run_fdr"] = np.nan
    out["q_value_patient_fdr"] = np.nan
    for _, idx in out.groupby(["subject", "run"]).groups.items():
        out.loc[idx, "q_value_run_fdr"] = benjamini_hochberg(out.loc[idx, "p_value"].to_numpy(float))
    for _, idx in out.groupby("subject").groups.items():
        out.loc[idx, "q_value_patient_fdr"] = benjamini_hochberg(out.loc[idx, "p_value"].to_numpy(float))
    out["significant_uncorrected_p05"] = out["p_value"] < 0.05
    out["significant_run_fdr05"] = out["q_value_run_fdr"] < 0.05
    out["significant_patient_fdr05"] = out["q_value_patient_fdr"] < 0.05
    return out


def relative_depth_by_group(df: pd.DataFrame, group_col: str | list[str], depth_col: str) -> pd.Series:
    rel = pd.Series(np.nan, index=df.index, dtype=float)
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        values = df.loc[idx, depth_col].to_numpy(float)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            continue
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        if hi <= lo:
            rel.loc[idx] = 0.5
        else:
            rel.loc[idx] = (values - lo) / (hi - lo)
    return rel


def assign_depth_bands(df: pd.DataFrame, group_col: str | list[str], depth_col: str) -> pd.Series:
    bands = pd.Series(pd.NA, index=df.index, dtype="object")
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        sub = df.loc[idx, depth_col]
        valid = sub[np.isfinite(sub.to_numpy(float))]
        if valid.empty:
            continue
        ranks = valid.rank(method="first")
        labels = pd.qcut(ranks, q=min(3, len(valid)), labels=False, duplicates="drop")
        for label_value, row_idx in zip(labels.to_numpy(), valid.index):
            if pd.isna(label_value):
                continue
            if labels.max() == 0:
                band = "middle"
            elif labels.max() == 1:
                band = ["superficial", "deep"][int(label_value)]
            else:
                band = BANDS[int(label_value)]
            bands.loc[row_idx] = band
    return bands


def run_preserving_permutation_spearman(
    depth: np.ndarray,
    effect: np.ndarray,
    run: np.ndarray,
    n_perm: int = 5000,
    random_state: int = 0,
) -> dict[str, float]:
    depth = np.asarray(depth, dtype=float)
    effect = np.asarray(effect, dtype=float)
    run = np.asarray(run)
    valid = np.isfinite(depth) & np.isfinite(effect)
    depth = depth[valid]
    effect = effect[valid]
    run = run[valid]
    if len(depth) < 3 or np.nanstd(depth) == 0 or np.nanstd(effect) == 0:
        return {"n": int(len(depth)), "rho": np.nan, "p_two_sided": np.nan}
    rho = float(stats.spearmanr(depth, effect).correlation)
    rng = np.random.default_rng(random_state)
    perm = np.empty(n_perm, dtype=float)
    unique_runs = pd.unique(run)
    for i in range(n_perm):
        shuffled = depth.copy()
        for run_name in unique_runs:
            mask = run == run_name
            shuffled[mask] = rng.permutation(shuffled[mask])
        perm[i] = stats.spearmanr(shuffled, effect).correlation
    p = (np.sum(np.abs(perm) >= abs(rho)) + 1.0) / (n_perm + 1.0)
    return {"n": int(len(depth)), "rho": rho, "p_two_sided": float(p)}


def linregress_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r": np.nan, "p": np.nan}
    result = stats.linregress(x, y)
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r": float(result.rvalue),
        "p": float(result.pvalue),
    }


def summarize_band(group: pd.DataFrame, keys: dict[str, object]) -> dict[str, object]:
    return {
        **keys,
        "n_units": int(group.shape[0]),
        "median_relative_depth": float(group["relative_depth"].median()),
        "median_depth_um": float(group["depth_um_median"].median()),
        "median_signed_effect_hz": float(group["active_minus_passive_bc_hz"].median()),
        "median_abs_effect_hz": float(group["abs_active_passive_diff_hz"].median()),
        "fraction_active_gt_passive": float((group["active_minus_passive_bc_hz"] > 0).mean()),
        "n_uncorrected_p05": int(group["significant_uncorrected_p05"].sum()),
        "n_patient_fdr05": int(group["significant_patient_fdr05"].sum()),
    }


def patient_summary(units: pd.DataFrame, n_perm: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []

    for subject, sub in units.groupby("subject"):
        rel_signed = run_preserving_permutation_spearman(
            sub["relative_depth"].to_numpy(float),
            sub["active_minus_passive_bc_hz"].to_numpy(float),
            sub["run"].to_numpy(),
            n_perm=n_perm,
            random_state=random_state + stable_seed(subject, "signed"),
        )
        rel_abs = run_preserving_permutation_spearman(
            sub["relative_depth"].to_numpy(float),
            sub["abs_active_passive_diff_hz"].to_numpy(float),
            sub["run"].to_numpy(),
            n_perm=n_perm,
            random_state=random_state + stable_seed(subject, "abs"),
        )
        rel_sig = run_preserving_permutation_spearman(
            sub["relative_depth"].to_numpy(float),
            sub["minus_log10_p"].to_numpy(float),
            sub["run"].to_numpy(),
            n_perm=n_perm,
            random_state=random_state + stable_seed(subject, "sig"),
        )
        signed_lm = linregress_summary(sub["relative_depth"].to_numpy(float), sub["active_minus_passive_bc_hz"].to_numpy(float))
        abs_lm = linregress_summary(sub["relative_depth"].to_numpy(float), sub["abs_active_passive_diff_hz"].to_numpy(float))
        superficial = sub[sub["depth_band"] == "superficial"]
        deep = sub[sub["depth_band"] == "deep"]
        patient_rows.append(
            {
                "subject": subject,
                "n_runs": int(sub["run"].nunique()),
                "n_units": int(sub.shape[0]),
                "depth_min_um": float(sub["depth_um_median"].min()),
                "depth_max_um": float(sub["depth_um_median"].max()),
                "median_signed_effect_hz": float(sub["active_minus_passive_bc_hz"].median()),
                "median_abs_effect_hz": float(sub["abs_active_passive_diff_hz"].median()),
                "fraction_active_gt_passive": float((sub["active_minus_passive_bc_hz"] > 0).mean()),
                "n_uncorrected_p05": int(sub["significant_uncorrected_p05"].sum()),
                "n_patient_fdr05": int(sub["significant_patient_fdr05"].sum()),
                "rho_relative_depth_signed_effect": rel_signed["rho"],
                "p_relative_depth_signed_effect_perm": rel_signed["p_two_sided"],
                "rho_relative_depth_abs_effect": rel_abs["rho"],
                "p_relative_depth_abs_effect_perm": rel_abs["p_two_sided"],
                "rho_relative_depth_minus_log10_p": rel_sig["rho"],
                "p_relative_depth_minus_log10_p_perm": rel_sig["p_two_sided"],
                "slope_signed_effect_hz_per_relative_depth": signed_lm["slope"],
                "slope_signed_effect_lm_p": signed_lm["p"],
                "slope_abs_effect_hz_per_relative_depth": abs_lm["slope"],
                "slope_abs_effect_lm_p": abs_lm["p"],
                "deep_minus_superficial_signed_effect_hz": float(deep["active_minus_passive_bc_hz"].median() - superficial["active_minus_passive_bc_hz"].median())
                if len(deep) and len(superficial)
                else np.nan,
                "deep_minus_superficial_abs_effect_hz": float(deep["abs_active_passive_diff_hz"].median() - superficial["abs_active_passive_diff_hz"].median())
                if len(deep) and len(superficial)
                else np.nan,
            }
        )

        for run, run_sub in sub.groupby("run"):
            run_signed = run_preserving_permutation_spearman(
                run_sub["relative_depth"].to_numpy(float),
                run_sub["active_minus_passive_bc_hz"].to_numpy(float),
                run_sub["run"].to_numpy(),
                n_perm=n_perm,
                random_state=random_state + stable_seed(subject, run, "signed"),
            )
            run_abs = run_preserving_permutation_spearman(
                run_sub["relative_depth"].to_numpy(float),
                run_sub["abs_active_passive_diff_hz"].to_numpy(float),
                run_sub["run"].to_numpy(),
                n_perm=n_perm,
                random_state=random_state + stable_seed(subject, run, "abs"),
            )
            run_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "n_units": int(run_sub.shape[0]),
                    "depth_min_um": float(run_sub["depth_um_median"].min()),
                    "depth_max_um": float(run_sub["depth_um_median"].max()),
                    "median_signed_effect_hz": float(run_sub["active_minus_passive_bc_hz"].median()),
                    "median_abs_effect_hz": float(run_sub["abs_active_passive_diff_hz"].median()),
                    "rho_relative_depth_signed_effect": run_signed["rho"],
                    "p_relative_depth_signed_effect_perm": run_signed["p_two_sided"],
                    "rho_relative_depth_abs_effect": run_abs["rho"],
                    "p_relative_depth_abs_effect_perm": run_abs["p_two_sided"],
                    "n_uncorrected_p05": int(run_sub["significant_uncorrected_p05"].sum()),
                    "n_run_fdr05": int(run_sub["significant_run_fdr05"].sum()),
                }
            )

        for band, band_sub in sub.groupby("depth_band", dropna=True):
            band_rows.append(summarize_band(band_sub, {"subject": subject, "depth_band": band}))

    return pd.DataFrame(patient_rows), pd.DataFrame(run_rows), pd.DataFrame(band_rows)


def make_patient_scatter_figure(units: pd.DataFrame, out_dir: Path, y_col: str, ylabel: str, title: str, filename: str) -> str | None:
    if units.empty:
        return None
    subjects = list(units["subject"].drop_duplicates())
    n_cols = min(3, len(subjects))
    n_rows = int(np.ceil(len(subjects) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.3 * n_rows), squeeze=False)
    colors = {"active_gt_passive": "#2f6fbb", "passive_gt_active": "#c43c39", "no_difference": "0.45"}
    for ax, subject in zip(axes.ravel(), subjects):
        sub = units[units["subject"] == subject]
        for direction, ddf in sub.groupby("direction"):
            sig = ddf["significant_uncorrected_p05"].to_numpy(bool)
            ax.scatter(
                ddf.loc[~sig, "relative_depth"],
                ddf.loc[~sig, y_col],
                s=18,
                alpha=0.45,
                color=colors.get(direction, "0.45"),
                edgecolor="none",
            )
            ax.scatter(
                ddf.loc[sig, "relative_depth"],
                ddf.loc[sig, y_col],
                s=42,
                alpha=0.9,
                color=colors.get(direction, "0.45"),
                edgecolor="black",
                linewidth=0.5,
            )
        fit = linregress_summary(sub["relative_depth"].to_numpy(float), sub[y_col].to_numpy(float))
        if np.isfinite(fit["slope"]):
            xs = np.array([0.0, 1.0])
            ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color="black", lw=1.3)
        ax.axhline(0, color="0.75", ls="--", lw=0.9)
        ax.set_title(f"{subject} (n={len(sub)})", fontsize=10)
        ax.set_xlabel("Relative depth (superficial to deep)")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(subjects) :]:
        ax.axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def make_band_figure(band_summary: pd.DataFrame, out_dir: Path) -> str | None:
    if band_summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x_lookup = {band: i for i, band in enumerate(BANDS)}
    for subject, sub in band_summary.groupby("subject"):
        sub = sub.copy()
        sub["x"] = sub["depth_band"].map(x_lookup)
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["median_signed_effect_hz"], marker="o", lw=1.5, label=subject)
    ax.axhline(0, color="0.55", ls="--", lw=1.0)
    ax.set_xticks(range(len(BANDS)), ["Superficial", "Middle", "Deep"])
    ax.set_ylabel("Median active-passive effect (Hz)")
    ax.set_title("Patient-level pseudo-laminar signed effects")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_patient_depth_band_signed_effects.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_individual_patient_figures(units: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    patient_dir = out_dir / "patient_figures"
    patient_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    colors = {"active_gt_passive": "#2f6fbb", "passive_gt_active": "#c43c39", "no_difference": "0.45"}
    for subject, sub in units.groupby("subject"):
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), sharex=True)
        for ax, y_col, ylabel in [
            (axes[0], "active_minus_passive_bc_hz", "Active-passive effect (Hz)"),
            (axes[1], "abs_active_passive_diff_hz", "Absolute effect (Hz)"),
        ]:
            for direction, ddf in sub.groupby("direction"):
                sig = ddf["significant_uncorrected_p05"].to_numpy(bool)
                ax.scatter(
                    ddf.loc[~sig, "relative_depth"],
                    ddf.loc[~sig, y_col],
                    s=18,
                    alpha=0.45,
                    color=colors.get(direction, "0.45"),
                    edgecolor="none",
                )
                ax.scatter(
                    ddf.loc[sig, "relative_depth"],
                    ddf.loc[sig, y_col],
                    s=46,
                    alpha=0.9,
                    color=colors.get(direction, "0.45"),
                    edgecolor="black",
                    linewidth=0.5,
                )
            fit = linregress_summary(sub["relative_depth"].to_numpy(float), sub[y_col].to_numpy(float))
            if np.isfinite(fit["slope"]):
                xs = np.array([0.0, 1.0])
                ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color="black", lw=1.4)
            ax.axhline(0, color="0.75", ls="--", lw=0.9)
            ax.set_xlabel("Relative depth")
            ax.set_ylabel(ylabel)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{subject} depth organization (n={len(sub)} units)", y=1.02)
        fig.tight_layout()
        path = patient_dir / f"fig_depth_laminar_{subject}.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        paths[f"patient_{subject}_png"] = str(path)
    return paths


def make_figures(units: pd.DataFrame, band_summary: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    signed = make_patient_scatter_figure(
        units,
        out_dir,
        "active_minus_passive_bc_hz",
        "Active-passive effect (Hz)",
        "Individual-patient depth gradients: signed active-passive effects",
        "fig_patient_depth_signed_effects.png",
    )
    if signed:
        paths["patient_depth_signed_effects_png"] = signed
    absolute = make_patient_scatter_figure(
        units,
        out_dir,
        "abs_active_passive_diff_hz",
        "Absolute active-passive effect (Hz)",
        "Individual-patient depth gradients: absolute active-passive effects",
        "fig_patient_depth_abs_effects.png",
    )
    if absolute:
        paths["patient_depth_abs_effects_png"] = absolute
    band = make_band_figure(band_summary, out_dir)
    if band:
        paths["patient_depth_band_signed_effects_png"] = band
    paths.update(make_individual_patient_figures(units, out_dir))
    return paths


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report(
    patient_df: pd.DataFrame,
    run_df: pd.DataFrame,
    band_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    aggregate: dict[str, object],
    figure_paths: dict[str, str],
) -> None:
    lines = [
        "# Active/passive depth and pseudo-laminar organization",
        "",
        "This analysis is intentionally organized at the individual-patient level. Units are not pooled across patients to infer a single group laminar profile.",
        "Depth bands are pseudo-laminar tertiles based on relative depth within each patient's analyzed units after normalizing depth within run; they are not anatomical cortical layers unless validated with CSD or histology.",
        "",
        "## Interpretation Notes",
        "",
    ]
    if not patient_df.empty:
        for _, row in patient_df.iterrows():
            lines.append(
                f"- {row['subject']}: n={int(row['n_units'])} units, relative-depth signed-effect rho={row['rho_relative_depth_signed_effect']:.3g} "
                f"(perm p={row['p_relative_depth_signed_effect_perm']:.3g}), absolute-effect rho={row['rho_relative_depth_abs_effect']:.3g} "
                f"(perm p={row['p_relative_depth_abs_effect_perm']:.3g}), deep-superficial signed median difference={row['deep_minus_superficial_signed_effect_hz']:.3g} Hz."
            )
    lines.extend(
        [
            "",
            "A patient with a depth/laminar organization signal should show a consistent monotonic depth correlation or a clear superficial/middle/deep band difference within that same patient.",
            "In this dataset, the signed active-passive effect does not show a consistent patient-level depth gradient. The main suggestive patient-level pattern is gz005, where the absolute active-passive effect is larger in relatively superficial units than deeper units, but this should be treated as an individual-patient observation rather than a group laminar law.",
            "",
            "## Patient Summary",
            "",
            markdown_table(patient_df),
            "",
            "## Run Summary",
            "",
            markdown_table(run_df),
            "",
            "## Pseudo-Laminar Band Summary",
            "",
            markdown_table(band_df),
            "",
            "## Skipped Runs",
            "",
            markdown_table(skipped_df),
            "",
            "## Aggregate Metadata",
            "",
            "```json",
            json.dumps(aggregate, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Figures",
            "",
        ]
    )
    for key, path in figure_paths.items():
        lines.append(f"- {key}: `{path}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    analysis_mode = "all_trials" if args.artifact_threshold is None else (
        f"artifact_clean_le_{str(args.artifact_threshold).replace('.', 'p')}"
    )

    unit_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []
    missing_depth_rows: list[dict[str, object]] = []

    for er_dir in find_event_dirs():
        subject = subject_from_er_dir(er_dir)
        metrics = load_metrics(er_dir)
        depth_meta = load_depth_metadata(er_dir)
        artifact = load_artifact_table(er_dir)
        filtered = apply_filters(metrics, artifact, args.unit_set, args.artifact_threshold)
        for run in sorted(metrics["run"].unique()):
            run_df = filtered[filtered["run"] == run].copy()
            n_trials = run_df["trial_index"].nunique()
            n_active = run_df.loc[run_df["condition_simple"] == "active", "trial_index"].nunique()
            n_passive = run_df.loc[run_df["condition_simple"] == "passive", "trial_index"].nunique()
            common = {
                "analysis_mode": analysis_mode,
                "unit_set": args.unit_set,
                "subject": subject,
                "run": run,
                "n_trials": int(n_trials),
                "n_active": int(n_active),
                "n_passive": int(n_passive),
            }
            if n_trials < args.min_trials_total or min(n_active, n_passive) < args.min_trials_per_class:
                skipped_rows.append({**common, "reason": "insufficient artifact-clean active/passive trials"})
                continue
            effects = compute_unit_effects(run_df)
            if effects.empty:
                skipped_rows.append({**common, "reason": "no units after filtering"})
                continue
            run_depth = depth_meta[depth_meta["run"] == run].copy()
            if run_depth.empty:
                skipped_rows.append({**common, "reason": "missing depth metadata"})
                continue
            merged = effects.merge(run_depth, on=["subject", "run", "unit_id"], how="left")
            missing = merged["depth_um_median"].isna()
            if missing.any():
                missing_depth_rows.append(
                    {
                        **common,
                        "n_units_missing_depth": int(missing.sum()),
                        "n_units_total": int(merged.shape[0]),
                    }
                )
                merged = merged.loc[~missing].copy()
            if merged.empty:
                skipped_rows.append({**common, "reason": "all units missing depth metadata"})
                continue
            unit_frames.append(merged)
            print(f"analyzed {subject} {run}: trials={n_trials} units={merged.shape[0]}", flush=True)

    units = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    if not units.empty:
        units = add_fdr_columns(units)
        units["relative_depth"] = relative_depth_by_group(units, ["subject", "run"], "depth_um_median")
        units["depth_band"] = assign_depth_bands(units, "subject", "relative_depth")
    patient_df, run_df, band_df = patient_summary(units, args.n_perm, args.random_state) if not units.empty else (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    skipped_df = pd.DataFrame(skipped_rows)
    missing_depth_df = pd.DataFrame(missing_depth_rows)

    unit_path = OUT_DIR / "active_passive_depth_laminar_unit_metrics.csv"
    patient_path = OUT_DIR / "active_passive_depth_laminar_patient_summary.csv"
    run_path = OUT_DIR / "active_passive_depth_laminar_run_summary.csv"
    band_path = OUT_DIR / "active_passive_depth_laminar_band_summary.csv"
    skipped_path = OUT_DIR / "active_passive_depth_laminar_skipped_runs.csv"
    missing_path = OUT_DIR / "active_passive_depth_laminar_missing_depth.csv"
    summary_path = OUT_DIR / "active_passive_depth_laminar_summary.json"

    units.to_csv(unit_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    run_df.to_csv(run_path, index=False)
    band_df.to_csv(band_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    missing_depth_df.to_csv(missing_path, index=False)

    figure_paths = make_figures(units, band_df, OUT_DIR) if not units.empty else {}
    aggregate: dict[str, object] = {
        "analysis_mode": analysis_mode,
        "unit_set": args.unit_set,
        "metric": METRIC,
        "n_perm": args.n_perm,
        "n_patients_analyzed": int(patient_df.shape[0]),
        "n_runs_analyzed": int(run_df.shape[0]),
        "n_units_analyzed": int(units.shape[0]),
        "n_runs_skipped": int(skipped_df.shape[0]),
        "n_missing_depth_rows": int(missing_depth_df.shape[0]),
        "outputs": {
            "unit_metrics_csv": str(unit_path),
            "patient_summary_csv": str(patient_path),
            "run_summary_csv": str(run_path),
            "band_summary_csv": str(band_path),
            "skipped_runs_csv": str(skipped_path),
            "missing_depth_csv": str(missing_path),
            "summary_json": str(summary_path),
            "report_md": str(REPORT_PATH),
            **figure_paths,
        },
    }
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(patient_df, run_df, band_df, skipped_df, aggregate, figure_paths)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-set", choices=["good_only", "all_clusters"], default="good_only")
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--min-trials-total", type=int, default=10)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
