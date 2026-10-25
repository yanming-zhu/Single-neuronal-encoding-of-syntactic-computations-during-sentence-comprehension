#!/usr/bin/env python3
"""Compare unit-level and population/network encoding of active/passive trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025/data"
OUT_DIR = DATA_ROOT / "combined_active_passive_network_encoding"
REPORT_PATH = ROOT / "reports/active_passive_network_encoding_5patients.md"
METRIC = "response_baseline_corrected_hz"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    kind: str
    k: int | None = None
    c: float = 1.0


MODEL_SPECS = [
    ModelSpec("nested_single_unit", "Nested single unit", "single", c=1.0),
    ModelSpec("network_all_l2", "All units L2", "all", c=0.03),
    ModelSpec("network_top5_l2", "Top 5 units L2", "topk", k=5, c=0.1),
    ModelSpec("network_top10_l2", "Top 10 units L2", "topk", k=10, c=0.1),
    ModelSpec("network_top25_l2", "Top 25 units L2", "topk", k=25, c=0.1),
    ModelSpec("network_pca2_l2", "PCA 2 L2", "pca", k=2, c=1.0),
    ModelSpec("network_pca4_l2", "PCA 4 L2", "pca", k=4, c=1.0),
    ModelSpec("network_pca8_l2", "PCA 8 L2", "pca", k=8, c=1.0),
]
PERMUTED_MODELS = {
    "nested_single_unit",
    "network_all_l2",
    "network_top10_l2",
    "network_pca4_l2",
    "network_pca8_l2",
}


def simple_condition(value: object) -> str | None:
    text = str(value)
    if text.startswith("active"):
        return "active"
    if text.startswith("passive"):
        return "passive"
    return None


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def subject_from_er_dir(er_dir: Path) -> str:
    return er_dir.relative_to(DATA_ROOT).parts[0]


def find_er_dirs() -> list[Path]:
    return sorted(DATA_ROOT.glob("gz*/processed/ActPasSynSentence/event_related_active_passive"))


def load_metrics(er_dir: Path) -> pd.DataFrame:
    subject = subject_from_er_dir(er_dir)
    path = er_dir / "active_passive_unit_trial_response_metrics.csv"
    df = pd.read_csv(path)
    df["subject"] = subject
    df["condition_simple"] = df["condition"].map(simple_condition)
    df = df[df["condition_simple"].isin(["active", "passive"])].copy()
    return df


def load_artifact_table(er_dir: Path) -> pd.DataFrame | None:
    path = er_dir / "trial_artifact_overlap_minus1_to_plus2s.csv"
    if not path.exists():
        return None
    artifact = pd.read_csv(path)
    artifact["condition_simple"] = artifact["condition"].map(simple_condition)
    artifact = artifact[artifact["condition_simple"].isin(["active", "passive"])].copy()
    if artifact.empty:
        return None
    cols = ["run", "trial_index", "artifact_fraction_minus1_to_plus2s"]
    return artifact[cols].drop_duplicates(["run", "trial_index"])


def apply_artifact_filter(
    metrics: pd.DataFrame,
    artifact: pd.DataFrame | None,
    threshold: float | None,
) -> pd.DataFrame:
    if threshold is None or artifact is None:
        out = metrics.copy()
        out["artifact_fraction_minus1_to_plus2s"] = np.nan
        return out
    out = metrics.merge(artifact, on=["run", "trial_index"], how="left")
    keep = out["artifact_fraction_minus1_to_plus2s"].isna() | (
        out["artifact_fraction_minus1_to_plus2s"] <= threshold
    )
    return out.loc[keep].copy()


def make_population_matrix(
    metrics: pd.DataFrame,
    run: str,
    unit_set: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int, int]:
    run_df = metrics[metrics["run"] == run].copy()
    if unit_set == "good_only":
        run_df = run_df[run_df["ks_label"].astype(str) == "good"].copy()

    input_units = run_df[["run", "unit_id"]].drop_duplicates().shape[0]
    if run_df.empty:
        return np.empty((0, 0)), np.empty(0, dtype=int), pd.DataFrame(), input_units, 0

    pivot = run_df.pivot_table(
        index="trial_index",
        columns="unit_id",
        values=METRIC,
        aggfunc="mean",
    ).sort_index()
    trial_meta = (
        run_df.drop_duplicates(["trial_index"])
        .set_index("trial_index")
        .loc[pivot.index]
        .reset_index()
    )
    y = (trial_meta["condition_simple"].to_numpy() == "active").astype(int)
    x = pivot.to_numpy(dtype=float)
    finite_cols = np.isfinite(x).all(axis=0)
    variable_cols = np.nanstd(x, axis=0) > 0
    keep_cols = finite_cols & variable_cols
    return x[:, keep_cols], y, trial_meta, input_units, int(keep_cols.sum())


def training_t_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    active = x[y == 1]
    passive = x[y == 0]
    active_var = active.var(axis=0, ddof=1) if active.shape[0] > 1 else np.zeros(x.shape[1])
    passive_var = passive.var(axis=0, ddof=1) if passive.shape[0] > 1 else np.zeros(x.shape[1])
    denom = np.sqrt(active_var / max(active.shape[0], 1) + passive_var / max(passive.shape[0], 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs((active.mean(axis=0) - passive.mean(axis=0)) / denom)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def select_top_units(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    scores = training_t_scores(x, y)
    order = np.argsort(scores)[::-1]
    return order[: min(k, x.shape[1])]


def fit_predict_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    spec: ModelSpec,
) -> np.ndarray:
    if spec.kind == "single":
        cols = select_top_units(x_train, y_train, 1)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=spec.c,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
            ),
        )
        model.fit(x_train[:, cols], y_train)
        return model.predict_proba(x_test[:, cols])[:, 1]

    if spec.kind == "topk":
        if spec.k is None:
            raise ValueError(f"{spec.name} requires k")
        cols = select_top_units(x_train, y_train, spec.k)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=spec.c,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
            ),
        )
        model.fit(x_train[:, cols], y_train)
        return model.predict_proba(x_test[:, cols])[:, 1]

    if spec.kind == "pca":
        if spec.k is None:
            raise ValueError(f"{spec.name} requires k")
        n_components = min(spec.k, max(x_train.shape[0] - 2, 1), x_train.shape[1])
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=0),
            LogisticRegression(
                C=spec.c,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
            ),
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]

    if spec.kind == "all":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=spec.c,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
            ),
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]

    raise ValueError(f"Unknown model kind: {spec.kind}")


def cv_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    spec: ModelSpec,
    n_repeats: int,
    random_state: int,
) -> np.ndarray:
    min_class = int(np.bincount(y, minlength=2).min())
    if min_class < 2:
        raise ValueError("Need at least 2 trials per class")
    n_splits = min(5, min_class)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    prob_sum = np.zeros(len(y), dtype=float)
    prob_count = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(x, y):
        probs = fit_predict_model(x[train_idx], y[train_idx], x[test_idx], spec)
        prob_sum[test_idx] += probs
        prob_count[test_idx] += 1
    return prob_sum / prob_count


def score_probabilities(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = (probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    spec: ModelSpec,
    observed_auc: float,
    n_permutations: int,
    perm_repeats: int,
    random_state: int,
) -> tuple[float, float, float]:
    if n_permutations <= 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(random_state)
    null_aucs = []
    for perm_i in range(n_permutations):
        y_perm = rng.permutation(y)
        probs = cv_probabilities(
            x,
            y_perm,
            spec,
            n_repeats=perm_repeats,
            random_state=random_state + 1000 + perm_i,
        )
        null_aucs.append(score_probabilities(y_perm, probs)["auc"])
    null = np.asarray(null_aucs, dtype=float)
    p_value = (1.0 + np.sum(null >= observed_auc)) / (n_permutations + 1.0)
    return float(p_value), float(np.mean(null)), float(np.std(null, ddof=1))


def cv_condition_mean_r2(
    x: np.ndarray,
    y: np.ndarray,
    n_repeats: int,
    random_state: int,
) -> dict[str, float]:
    min_class = int(np.bincount(y, minlength=2).min())
    n_splits = min(5, min_class)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    yhat_null = np.zeros_like(x, dtype=float)
    yhat_condition = np.zeros_like(x, dtype=float)
    counts = np.zeros(x.shape[0], dtype=float)
    for train_idx, test_idx in cv.split(x, y):
        train_mean = x[train_idx].mean(axis=0)
        active_mean = x[train_idx][y[train_idx] == 1].mean(axis=0)
        passive_mean = x[train_idx][y[train_idx] == 0].mean(axis=0)
        yhat_null[test_idx] += train_mean
        for row_i in test_idx:
            yhat_condition[row_i] += active_mean if y[row_i] == 1 else passive_mean
        counts[test_idx] += 1
    yhat_null /= counts[:, None]
    yhat_condition /= counts[:, None]
    sse_null = float(np.sum((x - yhat_null) ** 2))
    sse_condition = float(np.sum((x - yhat_condition) ** 2))
    total = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return {
        "condition_mean_delta_r2": float((sse_null - sse_condition) / total) if total > 0 else np.nan,
        "condition_mean_r2_vs_unit_mean": float(1.0 - sse_condition / sse_null) if sse_null > 0 else np.nan,
    }


def describe_best_network(results: pd.DataFrame) -> pd.DataFrame:
    network = results[results["model"] != "nested_single_unit"].copy()
    idx = network.groupby(["analysis_mode", "unit_set", "subject", "run"])["auc"].idxmax()
    best = network.loc[idx].copy()
    unit = results[results["model"] == "nested_single_unit"][
        ["analysis_mode", "unit_set", "subject", "run", "auc", "balanced_accuracy"]
    ].rename(columns={"auc": "nested_single_unit_auc", "balanced_accuracy": "nested_single_unit_balanced_accuracy"})
    best = best.merge(unit, on=["analysis_mode", "unit_set", "subject", "run"], how="left")
    best["auc_gain_vs_nested_single_unit"] = best["auc"] - best["nested_single_unit_auc"]
    return best


def signed_rank_summary(values: np.ndarray, reference: float = 0.0) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out = {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else np.nan,
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "wilcoxon_p": np.nan,
    }
    if len(values) >= 3 and np.any(np.abs(values - reference) > 1e-12):
        out["wilcoxon_p"] = float(stats.wilcoxon(values - reference, alternative="greater").pvalue)
    return out


def compute_model_gain_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if results.empty:
        return pd.DataFrame(rows)
    for analysis_mode in sorted(results["analysis_mode"].unique()):
        for unit_set in sorted(results["unit_set"].unique()):
            sub = results[(results["analysis_mode"] == analysis_mode) & (results["unit_set"] == unit_set)]
            wide = sub.pivot_table(
                index=["subject", "run"],
                columns="model",
                values="auc",
                aggfunc="first",
            )
            if "nested_single_unit" not in wide:
                continue
            nested = wide["nested_single_unit"]
            for model in [col for col in wide.columns if col != "nested_single_unit"]:
                gain = (wide[model] - nested).dropna()
                auc = wide[model].dropna()
                gain_p = np.nan
                auc_p = np.nan
                if len(gain) >= 3 and np.any(np.abs(gain) > 1e-12):
                    gain_p = float(stats.wilcoxon(gain, alternative="greater").pvalue)
                if len(auc) >= 3 and np.any(np.abs(auc - 0.5) > 1e-12):
                    auc_p = float(stats.wilcoxon(auc - 0.5, alternative="greater").pvalue)
                rows.append(
                    {
                        "analysis_mode": analysis_mode,
                        "unit_set": unit_set,
                        "model": model,
                        "n_runs": int(len(gain)),
                        "median_auc": float(auc.median()) if len(auc) else np.nan,
                        "median_gain_vs_nested": float(gain.median()) if len(gain) else np.nan,
                        "mean_gain_vs_nested": float(gain.mean()) if len(gain) else np.nan,
                        "gain_wilcoxon_p": gain_p,
                        "auc_gt_chance_wilcoxon_p": auc_p,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["analysis_mode", "unit_set", "median_gain_vs_nested"],
        ascending=[True, True, False],
    )


def make_plots(results: pd.DataFrame, best: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    plot_paths: dict[str, str] = {}
    primary = results[(results["analysis_mode"] == "all_trials") & (results["unit_set"] == "all_clusters")].copy()
    if not primary.empty:
        order = [spec.name for spec in MODEL_SPECS]
        labels = {spec.name: spec.label for spec in MODEL_SPECS}
        primary["model"] = pd.Categorical(primary["model"], categories=order, ordered=True)
        fig, ax = plt.subplots(figsize=(11, 5.8))
        for run, sub in primary.sort_values("model").groupby("run", sort=False):
            ax.plot(
                np.arange(len(order)),
                [sub.loc[sub["model"] == model, "auc"].iloc[0] for model in order],
                marker="o",
                lw=1.2,
                alpha=0.72,
                label=run,
            )
        medians = primary.groupby("model", observed=False)["auc"].median().reindex(order)
        ax.plot(np.arange(len(order)), medians.to_numpy(), color="black", marker="s", lw=2.4, label="median")
        ax.axhline(0.5, color="0.45", lw=1, ls="--")
        ax.set_xticks(np.arange(len(order)), [labels[m] for m in order], rotation=35, ha="right")
        ax.set_ylabel("Cross-validated AUC")
        ax.set_title("Active/passive encoding: unit vs population models")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout()
        path = out_dir / "active_passive_network_model_auc_by_run.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plot_paths["auc_by_run"] = str(path)

    primary_best = best[(best["analysis_mode"] == "all_trials") & (best["unit_set"] == "all_clusters")].copy()
    if not primary_best.empty:
        labels = [f"{row.subject}\n{row.run.split('_')[-1]}" for row in primary_best.itertuples()]
        x = np.arange(len(primary_best))
        fig, ax = plt.subplots(figsize=(10.8, 4.8))
        width = 0.36
        ax.bar(x - width / 2, primary_best["nested_single_unit_auc"], width, label="Nested single unit", color="#777777")
        ax.bar(x + width / 2, primary_best["auc"], width, label="Best population model", color="#2f6fbb")
        ax.axhline(0.5, color="0.45", lw=1, ls="--")
        ax.set_xticks(x, labels, rotation=0)
        ax.set_ylabel("Cross-validated AUC")
        ax.set_title("Best population model vs fair single-unit baseline")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out_dir / "active_passive_best_population_vs_single_unit_auc.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plot_paths["best_population_vs_single_unit"] = str(path)

    return plot_paths


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 3) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append("" if not np.isfinite(value) else f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report(
    results: pd.DataFrame,
    best: pd.DataFrame,
    model_gain: pd.DataFrame,
    aggregate: dict[str, object],
    plots: dict[str, str],
) -> None:
    primary = results[(results["analysis_mode"] == "all_trials") & (results["unit_set"] == "all_clusters")].copy()
    primary_best = best[(best["analysis_mode"] == "all_trials") & (best["unit_set"] == "all_clusters")].copy()
    primary_wide = primary.pivot_table(
        index=["subject", "run"],
        columns="model",
        values="auc",
        aggfunc="first",
    ).reset_index()
    keep_cols = [
        "subject",
        "run",
        "nested_single_unit",
        "network_all_l2",
        "network_top10_l2",
        "network_pca4_l2",
        "network_pca8_l2",
    ]
    primary_wide = primary_wide[[col for col in keep_cols if col in primary_wide.columns]]
    best_table = primary_best[
        [
            "subject",
            "run",
            "model",
            "auc",
            "nested_single_unit_auc",
            "auc_gain_vs_nested_single_unit",
            "permutation_p",
        ]
    ].rename(
        columns={
            "model": "best_network_model",
            "auc": "best_network_auc",
        }
    )
    gain_table = model_gain[
        (model_gain["analysis_mode"] == "all_trials") & (model_gain["unit_set"] == "all_clusters")
    ].copy()

    lines = [
        "# Active/passive network encoding across 5 patients",
        "",
        "Metric: per-trial firing-rate response from 0-0.8 s after function-word onset minus baseline -1-0 s.",
        "Primary evidence is cross-validated AUC for predicting active vs passive from unit population responses.",
        "",
        "## Primary all-cluster all-trial AUC",
        "",
        markdown_table(primary_wide, list(primary_wide.columns)),
        "",
        "## Best population model per run",
        "",
        markdown_table(best_table, list(best_table.columns)),
        "",
        "## Fixed-model gain summary",
        "",
        markdown_table(gain_table, list(gain_table.columns)),
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Figures",
        "",
    ]
    for name, path in plots.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "Notes:",
            "- `nested_single_unit` selects the best unit inside each training fold before scoring held-out trials.",
            "- `best_network_model` is descriptive because it is selected after comparing several population models.",
            "- Permutation p values are one-sided against the model-specific label-shuffled null.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    analysis_modes = [("all_trials", None)]
    if args.artifact_threshold is not None:
        threshold_label = str(args.artifact_threshold).replace(".", "p")
        analysis_modes.append((f"artifact_clean_le_{threshold_label}", args.artifact_threshold))

    unit_sets = ["all_clusters", "good_only"] if args.unit_sets == "both" else [args.unit_sets]
    rows: list[dict[str, object]] = []
    encoding_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for er_dir in find_er_dirs():
        subject = subject_from_er_dir(er_dir)
        metrics = load_metrics(er_dir)
        artifact = load_artifact_table(er_dir)
        for analysis_mode, artifact_threshold in analysis_modes:
            filtered = apply_artifact_filter(metrics, artifact, artifact_threshold)
            for unit_set in unit_sets:
                for run in sorted(metrics["run"].unique()):
                    x, y, trial_meta, input_units, variable_units = make_population_matrix(filtered, run, unit_set)
                    n_active = int(np.sum(y == 1))
                    n_passive = int(np.sum(y == 0))
                    common_meta = {
                        "analysis_mode": analysis_mode,
                        "artifact_threshold": artifact_threshold,
                        "unit_set": unit_set,
                        "subject": subject,
                        "run": run,
                        "n_trials": int(len(y)),
                        "n_active": n_active,
                        "n_passive": n_passive,
                        "n_units_input": input_units,
                        "n_units_variable": variable_units,
                    }
                    if len(y) < args.min_trials_total or min(n_active, n_passive) < args.min_trials_per_class or variable_units < 1:
                        skipped_rows.append({**common_meta, "reason": "insufficient trials or units"})
                        continue

                    condition_r2 = cv_condition_mean_r2(
                        x,
                        y,
                        n_repeats=args.cv_repeats,
                        random_state=args.random_state,
                    )
                    encoding_rows.append({**common_meta, **condition_r2})

                    for spec in MODEL_SPECS:
                        probs = cv_probabilities(
                            x,
                            y,
                            spec,
                            n_repeats=args.cv_repeats,
                            random_state=args.random_state,
                        )
                        score = score_probabilities(y, probs)
                        perm_p = np.nan
                        null_mean = np.nan
                        null_sd = np.nan
                        if spec.name in PERMUTED_MODELS and args.n_permutations > 0:
                            perm_p, null_mean, null_sd = permutation_p_value(
                                x,
                                y,
                                spec,
                                observed_auc=score["auc"],
                                n_permutations=args.n_permutations,
                                perm_repeats=args.permutation_repeats,
                                random_state=args.random_state
                                + stable_seed(subject, run, unit_set, analysis_mode, spec.name) % 100000,
                            )
                        rows.append(
                            {
                                **common_meta,
                                "model": spec.name,
                                "model_label": spec.label,
                                **score,
                                "permutation_p": perm_p,
                                "permutation_null_auc_mean": null_mean,
                                "permutation_null_auc_sd": null_sd,
                                "cv_repeats": args.cv_repeats,
                                "n_permutations": args.n_permutations if spec.name in PERMUTED_MODELS else 0,
                                "permutation_repeats": args.permutation_repeats if spec.name in PERMUTED_MODELS else 0,
                            }
                        )

    results = pd.DataFrame(rows)
    encoding = pd.DataFrame(encoding_rows)
    skipped = pd.DataFrame(skipped_rows)
    best = describe_best_network(results) if not results.empty else pd.DataFrame()
    model_gain = compute_model_gain_summary(results)

    results_path = OUT_DIR / "active_passive_network_encoding_model_scores.csv"
    encoding_path = OUT_DIR / "active_passive_condition_mean_encoding_r2.csv"
    best_path = OUT_DIR / "active_passive_best_network_models.csv"
    skipped_path = OUT_DIR / "active_passive_network_encoding_skipped_runs.csv"
    model_gain_path = OUT_DIR / "active_passive_network_model_gain_summary.csv"
    results.to_csv(results_path, index=False)
    encoding.to_csv(encoding_path, index=False)
    best.to_csv(best_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    model_gain.to_csv(model_gain_path, index=False)

    aggregate: dict[str, object] = {
        "results_csv": str(results_path),
        "condition_mean_encoding_r2_csv": str(encoding_path),
        "best_network_models_csv": str(best_path),
        "skipped_runs_csv": str(skipped_path),
        "model_gain_summary_csv": str(model_gain_path),
        "primary": {},
    }
    for analysis_mode in sorted(results["analysis_mode"].unique()) if not results.empty else []:
        for unit_set in sorted(results["unit_set"].unique()):
            sub = results[(results["analysis_mode"] == analysis_mode) & (results["unit_set"] == unit_set)]
            if sub.empty:
                continue
            best_sub = best[(best["analysis_mode"] == analysis_mode) & (best["unit_set"] == unit_set)]
            nested = sub[sub["model"] == "nested_single_unit"]["auc"].to_numpy(float)
            key = f"{analysis_mode}/{unit_set}"
            aggregate["primary"][key] = {
                "n_runs": int(sub[["subject", "run"]].drop_duplicates().shape[0]),
                "nested_single_unit_auc": signed_rank_summary(nested, reference=0.5),
                "best_population_auc": signed_rank_summary(best_sub["auc"].to_numpy(float), reference=0.5),
                "best_population_gain_vs_nested_single_unit": signed_rank_summary(
                    best_sub["auc_gain_vs_nested_single_unit"].to_numpy(float),
                    reference=0.0,
                ),
                "model_median_auc": {
                    model: float(vals.median())
                    for model, vals in sub.groupby("model")["auc"]
                },
            }

    plots = make_plots(results, best, OUT_DIR)
    aggregate["figures"] = plots
    summary_path = OUT_DIR / "active_passive_network_encoding_summary.json"
    aggregate["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(results, best, model_gain, aggregate, plots)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--artifact-threshold", type=float, default=0.2)
    parser.add_argument("--unit-sets", choices=["all_clusters", "good_only", "both"], default="both")
    parser.add_argument("--min-trials-total", type=int, default=10)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
