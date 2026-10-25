#!/usr/bin/env python3
"""Convert Kilosort spike times to 1000 Hz firing-rate matrices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("neuropixel2025/data/gz003/processed/ActPasSynSentence")
RUNS = ["gz003t2_g0_actpas_run1", "gz003t2_g0_actpas_run4"]
OUT_RATE_HZ = 1000


def read_meta(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def load_cluster_labels(results_dir: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    label_file = results_dir / "cluster_KSLabel.tsv"
    if not label_file.exists():
        return labels
    with label_file.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cluster_id = row.get("cluster_id") or row.get("cluster_id ")
            label = row.get("KSLabel") or row.get("group") or ""
            if cluster_id is not None:
                labels[int(cluster_id)] = label
    return labels


def write_trial_bins(run_dir: Path, out_dir: Path, duration_bins: int) -> None:
    events_path = run_dir / "events.csv"
    if not events_path.exists():
        return
    with events_path.open() as f:
        rows = list(csv.DictReader(f))
    out_path = out_dir / "trial_bins_1000Hz.csv"
    fieldnames = [
        "trial_index",
        "sound_file",
        "condition",
        "marker_bin",
        "marker_time_sec",
        "window_start_bin",
        "window_end_bin_exclusive",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            marker_time = float(row["time_from_crop_start_sec"])
            marker_bin = int(round(marker_time * OUT_RATE_HZ))
            writer.writerow(
                {
                    "trial_index": row["trial_index"],
                    "sound_file": row["sound_file"],
                    "condition": row["condition"],
                    "marker_bin": marker_bin,
                    "marker_time_sec": f"{marker_time:.9f}",
                    "window_start_bin": max(0, marker_bin),
                    "window_end_bin_exclusive": min(duration_bins, marker_bin + 3500),
                }
            )


def convert_one(run_name: str) -> dict[str, object]:
    run_dir = ROOT / run_name
    ks_dir = run_dir / "kilosort4"
    spike_times_path = ks_dir / "spike_times.npy"
    spike_clusters_path = ks_dir / "spike_clusters.npy"
    if not spike_times_path.exists() or not spike_clusters_path.exists():
        raise FileNotFoundError(f"Missing Kilosort spikes in {ks_dir}")

    ap_meta = run_dir / f"{run_name}_imec0" / f"{run_name}_t0.imec0.ap.meta"
    meta = read_meta(ap_meta)
    fs = float(meta["imSampRate"])
    duration_sec = float(meta["fileTimeSecs"])
    n_bins = int(np.ceil(duration_sec * OUT_RATE_HZ))

    spike_samples = np.load(spike_times_path).astype(np.int64).ravel()
    spike_clusters = np.load(spike_clusters_path).astype(np.int64).ravel()
    if spike_samples.shape[0] != spike_clusters.shape[0]:
        raise ValueError(f"spike_times and spike_clusters length mismatch in {ks_dir}")

    unit_ids = np.unique(spike_clusters)
    bin_index = np.floor(spike_samples / fs * OUT_RATE_HZ).astype(np.int64)
    valid = (bin_index >= 0) & (bin_index < n_bins)
    bin_index = bin_index[valid]
    spike_clusters = spike_clusters[valid]

    unit_index = {unit_id: i for i, unit_id in enumerate(unit_ids)}
    firing_rate = np.zeros((unit_ids.size, n_bins), dtype=np.float32)
    for cluster_id, bin_id in zip(spike_clusters, bin_index):
        firing_rate[unit_index[int(cluster_id)], int(bin_id)] += OUT_RATE_HZ

    labels = load_cluster_labels(ks_dir)
    ks_labels = np.array([labels.get(int(unit_id), "") for unit_id in unit_ids], dtype=object)
    time_sec = np.arange(n_bins, dtype=np.float32) / OUT_RATE_HZ

    out_dir = run_dir / "firing_rate_1000Hz"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        out_dir / "firing_rate_1000Hz_all_clusters.npz",
        firing_rate_hz=firing_rate,
        unit_ids=unit_ids.astype(np.int64),
        time_sec=time_sec,
        ks_labels=ks_labels,
        source_spike_times=str(spike_times_path),
        source_spike_clusters=str(spike_clusters_path),
        source_sampling_rate_hz=fs,
        output_sampling_rate_hz=OUT_RATE_HZ,
    )

    good_mask = np.array([label == "good" for label in ks_labels], dtype=bool)
    np.savez_compressed(
        out_dir / "firing_rate_1000Hz_good_clusters.npz",
        firing_rate_hz=firing_rate[good_mask],
        unit_ids=unit_ids[good_mask].astype(np.int64),
        time_sec=time_sec,
        ks_labels=ks_labels[good_mask],
        source_sampling_rate_hz=fs,
        output_sampling_rate_hz=OUT_RATE_HZ,
    )
    write_trial_bins(run_dir, out_dir, n_bins)

    summary = {
        "run_name": run_name,
        "kilosort_dir": str(ks_dir),
        "output_dir": str(out_dir),
        "n_spikes": int(spike_samples.size),
        "n_clusters": int(unit_ids.size),
        "n_good_clusters": int(good_mask.sum()),
        "duration_sec": duration_sec,
        "n_time_bins": int(n_bins),
        "output_sampling_rate_hz": OUT_RATE_HZ,
        "matrix_shape_units_by_time": [int(unit_ids.size), int(n_bins)],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS + ["all"], default="all")
    args = parser.parse_args()
    runs = RUNS if args.run == "all" else [args.run]
    summaries = [convert_one(r) for r in runs]
    (ROOT / "firing_rate_1000Hz_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
