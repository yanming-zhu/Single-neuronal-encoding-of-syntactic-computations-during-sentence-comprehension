#!/usr/bin/env python3
"""Create broader usable cluster sets from Kilosort metrics.

This does not overwrite Kilosort's original KSLabel == good set. It creates
additional filtered firing-rate matrices with explicit QC thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "neuropixel2025/data/gz003/processed/ActPasSynSentence"
RUNS = ("gz003t2_g0_actpas_run1", "gz003t2_g0_actpas_run4")

SETS = {
    # Close to common single-unit acceptance, but Kilosort already captured
    # this for run1. Useful as a reproducible quality table column.
    "strict_qc": {
        "max_contam_pct": 20.0,
        "min_spikes": 100,
        "min_amplitude": 8.0,
    },
    # Practical working set for this short task: includes Kilosort good and
    # moderate-contamination MUA with enough spikes to estimate ERFR.
    "usable_lenient": {
        "max_contam_pct": 50.0,
        "min_spikes": 500,
        "min_amplitude": 8.0,
    },
    # Very liberal exploratory set. Do not call these clean single units.
    "usable_very_lenient": {
        "max_contam_pct": 100.0,
        "min_spikes": 500,
        "min_amplitude": 8.0,
    },
}


def load_metrics(run: str) -> pd.DataFrame:
    ks_dir = TASK_DIR / run / "kilosort4"
    labels = pd.read_csv(ks_dir / "cluster_KSLabel.tsv", sep="\t")
    amps = pd.read_csv(ks_dir / "cluster_Amplitude.tsv", sep="\t")
    contam = pd.read_csv(ks_dir / "cluster_ContamPct.tsv", sep="\t")
    spike_clusters = np.load(ks_dir / "spike_clusters.npy").astype(int)
    spike_counts = np.bincount(spike_clusters)

    df = labels.merge(amps, on="cluster_id").merge(contam, on="cluster_id")
    df["n_spikes"] = [int(spike_counts[c]) if c < len(spike_counts) else 0 for c in df["cluster_id"]]
    df["ks_good"] = df["KSLabel"] == "good"
    for name, cfg in SETS.items():
        metric_pass = (
            (df["ContamPct"] <= cfg["max_contam_pct"])
            & (df["n_spikes"] >= cfg["min_spikes"])
            & (df["Amplitude"] >= cfg["min_amplitude"])
        )
        df[f"{name}_metric_pass"] = metric_pass
        df[name] = df["ks_good"] | metric_pass
    return df


def save_filtered_npz(run: str, metrics: pd.DataFrame, set_name: str) -> dict[str, object]:
    fr_dir = TASK_DIR / run / "firing_rate_1000Hz"
    all_npz = np.load(fr_dir / "firing_rate_1000Hz_all_clusters.npz", allow_pickle=True)
    unit_ids = all_npz["unit_ids"]
    selected_ids = metrics.loc[metrics[set_name], "cluster_id"].to_numpy(dtype=np.int64)
    selected_mask = np.isin(unit_ids, selected_ids)
    selected_metrics = metrics.set_index("cluster_id").loc[unit_ids[selected_mask]]

    out_path = fr_dir / f"firing_rate_1000Hz_{set_name}_clusters.npz"
    np.savez_compressed(
        out_path,
        firing_rate_hz=all_npz["firing_rate_hz"][selected_mask],
        unit_ids=unit_ids[selected_mask].astype(np.int64),
        time_sec=all_npz["time_sec"],
        ks_labels=all_npz["ks_labels"][selected_mask],
        amplitudes=selected_metrics["Amplitude"].to_numpy(dtype=float),
        contam_pct=selected_metrics["ContamPct"].to_numpy(dtype=float),
        n_spikes=selected_metrics["n_spikes"].to_numpy(dtype=np.int64),
        selection_name=set_name,
        source_all_clusters_npz=str(fr_dir / "firing_rate_1000Hz_all_clusters.npz"),
        output_sampling_rate_hz=all_npz["output_sampling_rate_hz"],
        source_sampling_rate_hz=all_npz["source_sampling_rate_hz"],
    )
    return {
        "npz": str(out_path),
        "n_clusters": int(selected_mask.sum()),
        "n_original_ks_good": int((selected_metrics["KSLabel"] == "good").sum()),
        "n_added_from_mua": int((selected_metrics["KSLabel"] != "good").sum()),
    }


def main() -> None:
    summary: dict[str, object] = {"sets": SETS, "runs": {}}
    for run in RUNS:
        metrics = load_metrics(run)
        out_dir = TASK_DIR / run / "firing_rate_1000Hz"
        metrics_path = out_dir / "cluster_quality_metrics_with_lenient_sets.csv"
        metrics.to_csv(metrics_path, index=False)

        run_summary: dict[str, object] = {
            "quality_metrics_csv": str(metrics_path),
            "n_clusters_total": int(len(metrics)),
            "ks_label_counts": {str(k): int(v) for k, v in metrics["KSLabel"].value_counts().items()},
            "sets": {},
        }
        for set_name in SETS:
            run_summary["sets"][set_name] = save_filtered_npz(run, metrics, set_name)
        summary["runs"][run] = run_summary

    summary_path = TASK_DIR / "lenient_good_cluster_sets_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
