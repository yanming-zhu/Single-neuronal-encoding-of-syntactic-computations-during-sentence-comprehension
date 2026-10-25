#!/usr/bin/env python3
"""Detect gz004 AP artifacts and save a 1000 Hz artifact mask for each run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "neuropixel2025/data/gz004/processed/ActPasSynSentence"
RUNS = ("gz004t1_g0_actpas_run0", "gz004t1_g3_actpas_run3")
OUT_RATE_HZ = 1000

MAX_ABS_COUNT_THRESHOLD = 200
P999_ABS_COUNT_THRESHOLD = 120
EXPAND_MS = 50
CHUNK_SEC = 10


def read_meta(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(errors="replace").splitlines()
        if "=" in line
    )


def mask_to_windows(mask: np.ndarray) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            windows.append((start, i))
            start = None
    if start is not None:
        windows.append((start, len(mask)))
    return windows


def expand_mask(mask: np.ndarray, expand_bins: int) -> np.ndarray:
    if expand_bins <= 0 or not mask.any():
        return mask
    expanded = mask.copy()
    hit_idx = np.flatnonzero(mask)
    for idx in hit_idx:
        start = max(0, idx - expand_bins)
        stop = min(mask.size, idx + expand_bins + 1)
        expanded[start:stop] = True
    return expanded


def detect_one(run: str) -> dict[str, object]:
    run_dir = TASK_DIR / run
    imec_dir = run_dir / f"{run}_imec0"
    ap_bin = imec_dir / f"{run}_t0.imec0.ap.bin"
    ap_meta = imec_dir / f"{run}_t0.imec0.ap.meta"
    meta = read_meta(ap_meta)
    fs = float(meta["imSampRate"])
    n_saved = int(meta["nSavedChans"])
    duration_sec = float(meta["fileTimeSecs"])
    n_ap = int(meta["snsApLfSy"].split(",")[0])
    n_bins = int(np.ceil(duration_sec * OUT_RATE_HZ))

    raw = np.memmap(ap_bin, dtype=np.int16, mode="r")
    n_samples = raw.size // n_saved
    data = raw.reshape(n_samples, n_saved)

    max_abs = np.zeros(n_bins, dtype=np.float32)
    p999_abs = np.zeros(n_bins, dtype=np.float32)

    chunk_samples = int(round(CHUNK_SEC * fs))
    for chunk_start in range(0, n_samples, chunk_samples):
        chunk_stop = min(n_samples, chunk_start + chunk_samples)
        chunk = data[chunk_start:chunk_stop, :n_ap].astype(np.float32)
        start_bin = int(np.floor(chunk_start / fs * OUT_RATE_HZ))
        stop_bin = int(np.ceil(chunk_stop / fs * OUT_RATE_HZ))
        for bin_id in range(start_bin, min(stop_bin, n_bins)):
            sample_start = max(chunk_start, int(np.floor(bin_id / OUT_RATE_HZ * fs)))
            sample_stop = min(chunk_stop, int(np.floor((bin_id + 1) / OUT_RATE_HZ * fs)))
            if sample_stop <= sample_start:
                continue
            local = chunk[sample_start - chunk_start : sample_stop - chunk_start]
            abs_local = np.abs(local)
            max_abs[bin_id] = max(max_abs[bin_id], float(abs_local.max()))
            p999_abs[bin_id] = max(p999_abs[bin_id], float(np.percentile(abs_local, 99.9)))

    raw_mask = (max_abs >= MAX_ABS_COUNT_THRESHOLD) | (p999_abs >= P999_ABS_COUNT_THRESHOLD)
    artifact_mask = expand_mask(raw_mask, EXPAND_MS)
    windows = mask_to_windows(artifact_mask)

    out_dir = run_dir / "artifact_rejection"
    out_dir.mkdir(exist_ok=True)
    npz_path = out_dir / "ap_artifact_mask_1000Hz.npz"
    csv_path = out_dir / "ap_artifact_windows_1000Hz.csv"
    np.savez_compressed(
        npz_path,
        artifact_mask=artifact_mask,
        raw_artifact_mask=raw_mask,
        max_abs_count=max_abs,
        p999_abs_count=p999_abs,
        time_sec=np.arange(n_bins, dtype=np.float32) / OUT_RATE_HZ,
        output_sampling_rate_hz=OUT_RATE_HZ,
        max_abs_count_threshold=MAX_ABS_COUNT_THRESHOLD,
        p999_abs_count_threshold=P999_ABS_COUNT_THRESHOLD,
        expand_ms=EXPAND_MS,
        source_ap_bin=str(ap_bin),
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_index",
                "start_bin",
                "stop_bin_exclusive",
                "start_sec",
                "stop_sec",
                "duration_sec",
            ],
        )
        writer.writeheader()
        for i, (start, stop) in enumerate(windows, start=1):
            writer.writerow(
                {
                    "window_index": i,
                    "start_bin": start,
                    "stop_bin_exclusive": stop,
                    "start_sec": f"{start / OUT_RATE_HZ:.3f}",
                    "stop_sec": f"{stop / OUT_RATE_HZ:.3f}",
                    "duration_sec": f"{(stop - start) / OUT_RATE_HZ:.3f}",
                }
            )

    return {
        "run": run,
        "duration_sec": duration_sec,
        "n_bins": n_bins,
        "n_raw_artifact_bins": int(raw_mask.sum()),
        "n_expanded_artifact_bins": int(artifact_mask.sum()),
        "artifact_fraction": float(artifact_mask.mean()),
        "n_artifact_windows": len(windows),
        "npz": str(npz_path),
        "windows_csv": str(csv_path),
        "max_abs_count_threshold": MAX_ABS_COUNT_THRESHOLD,
        "p999_abs_count_threshold": P999_ABS_COUNT_THRESHOLD,
        "expand_ms": EXPAND_MS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS + ("all",), default="all")
    args = parser.parse_args()
    runs = RUNS if args.run == "all" else (args.run,)
    summary = [detect_one(run) for run in runs]
    out = TASK_DIR / "artifact_rejection_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
