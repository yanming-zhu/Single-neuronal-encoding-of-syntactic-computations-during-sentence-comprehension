#!/usr/bin/env python3
"""Extract gz003 ActPasSynSentence runs from the continuous SpikeGLX recording."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np


ROOT = Path("neuropixel2025/data/gz003")
SRC = ROOT / "neuropixel" / "gz003t2_g0"
OUT_ROOT = ROOT / "processed" / "ActPasSynSentence"
BEH_ROOT = ROOT / "beh" / "Act"

RUN_NAME_PREFIX = "gz003t2_g0_actpas"
ORIGINAL_EXPERIMENT_RUN_INDICES = [1, 4]
INTERVENING_TASKS_BY_ORIGINAL_RUN = {
    4: ["SemanticSentence", "Tone"],
}
MARKER_CHANNEL = 1
PRE_FIRST_MARKER_SEC = 1.0
POST_LAST_MARKER_SEC = 6.5
MARKER_THRESHOLD = None
EXPECTED_TRIALS = 85


def read_meta(path: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    lines: list[tuple[str, str]] = []
    values: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            lines.append((key, value))
            values[key] = value
    return lines, values


def write_meta(
    src_meta: Path,
    dst_meta: Path,
    dst_bin: Path,
    sample_rate: float,
    n_channels: int,
    start_sample: int,
    n_samples: int,
    notes: str,
) -> None:
    lines, values = read_meta(src_meta)
    first_sample = int(values.get("firstSample", "0")) + start_sample
    replacements = {
        "fileName": str(dst_bin),
        "fileSHA1": "",
        "fileSizeBytes": str(dst_bin.stat().st_size),
        "fileTimeSecs": f"{n_samples / sample_rate:.12f}",
        "firstSample": str(first_sample),
        "userNotes": notes,
    }
    seen = set()
    out_lines = []
    for key, value in lines:
        if key in replacements:
            out_lines.append(f"{key}={replacements[key]}")
            seen.add(key)
        else:
            out_lines.append(f"{key}={value}")
    for key, value in replacements.items():
        if key not in seen:
            out_lines.append(f"{key}={value}")
    expected_size = n_samples * n_channels * 2
    if dst_bin.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch for {dst_bin}: expected {expected_size}, got {dst_bin.stat().st_size}")
    dst_meta.write_text("\n".join(out_lines) + "\n")


def stream_copy_samples(src_bin: Path, dst_bin: Path, n_channels: int, start_sample: int, n_samples: int) -> None:
    bytes_per_sample = n_channels * 2
    start_byte = start_sample * bytes_per_sample
    bytes_to_copy = n_samples * bytes_per_sample
    chunk_size = 256 * 1024 * 1024
    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    with src_bin.open("rb") as src, dst_bin.open("wb") as dst:
        src.seek(start_byte)
        remaining = bytes_to_copy
        while remaining:
            block = src.read(min(chunk_size, remaining))
            if not block:
                raise RuntimeError(f"Unexpected EOF while copying {src_bin}")
            dst.write(block)
            remaining -= len(block)


def parse_eprime_txt(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-16", errors="ignore")
    trials = []
    for frame in text.split("*** LogFrame Start ***")[1:]:
        fields: dict[str, str] = {}
        for line in frame.splitlines():
            line = line.strip("\t\r ")
            if ": " in line:
                key, value = line.split(": ", 1)
                fields[key] = value.strip()
        if fields.get("Procedure") == "TrialProc":
            trials.append(fields)
    return trials


def stimulus_condition(sound_file: str) -> str:
    if "stim_actpas/act" in sound_file:
        return "active_ba_jiang_gei"
    if "stim_actpas/pass" in sound_file:
        return "passive_bei_rang_jiao"
    if "stim_actpas/non" in sound_file:
        return "actpas_scrambled"
    if "stim_syn/synv" in sound_file:
        return "synonym_verb"
    if "stim_syn/synn" in sound_file:
        return "synonym_noun"
    if "stim_syn/non" in sound_file:
        return "syn_scrambled"
    return "other"


def detect_marker_runs(nidq_bin: Path, nidq_meta: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    _, meta = read_meta(nidq_meta)
    n_channels = int(meta["nSavedChans"])
    sample_rate = float(meta["niSampRate"])
    n_samples = nidq_bin.stat().st_size // (2 * n_channels)
    data = np.memmap(nidq_bin, dtype=np.int16, mode="r", shape=(n_samples, n_channels))
    channel = data[:, MARKER_CHANNEL]
    ch_min = int(channel.min())
    ch_max = int(channel.max())
    threshold = MARKER_THRESHOLD if MARKER_THRESHOLD is not None else (ch_min + ch_max) / 2

    rising: list[int] = []
    prev = False
    chunk = 1_000_000
    for start in range(0, n_samples, chunk):
        arr = np.asarray(channel[start : min(start + chunk, n_samples)]) > threshold
        if arr.size == 0:
            continue
        if (not prev) and bool(arr[0]):
            rising.append(start)
        idx = np.flatnonzero((~arr[:-1]) & arr[1:]) + 1
        rising.extend((idx + start).tolist())
        prev = bool(arr[-1])

    times = np.array(rising, dtype=np.float64) / sample_rate
    runs: list[dict[str, object]] = []
    if len(times) > 1:
        start_idx = 0
        for i, interval in enumerate(np.diff(times)):
            if not (2.5 <= interval <= 4.8):
                if i - start_idx + 1 >= 20:
                    runs.append(make_run(rising, times, start_idx, i))
                start_idx = i + 1
        if len(times) - 1 - start_idx + 1 >= 20:
            runs.append(make_run(rising, times, start_idx, len(times) - 1))

    diagnostics = {
        "marker_channel": MARKER_CHANNEL,
        "threshold": threshold,
        "channel_min": ch_min,
        "channel_max": ch_max,
        "rising_edges_detected": len(rising),
        "candidate_runs": runs,
        "sample_rate": sample_rate,
        "n_samples": n_samples,
    }
    actpas_runs = [r for r in runs if int(r["n_markers"]) == EXPECTED_TRIALS]
    if len(actpas_runs) != 2:
        raise RuntimeError(f"Expected 2 ActPas runs with {EXPECTED_TRIALS} markers, found {len(actpas_runs)}")
    return actpas_runs, diagnostics


def make_run(rising: list[int], times: np.ndarray, start_idx: int, end_idx: int) -> dict[str, object]:
    segment_times = times[start_idx : end_idx + 1]
    return {
        "edge_index_start": start_idx,
        "edge_index_end": end_idx,
        "n_markers": int(end_idx - start_idx + 1),
        "first_marker_sample_nidq": int(rising[start_idx]),
        "last_marker_sample_nidq": int(rising[end_idx]),
        "first_marker_time_sec": float(segment_times[0]),
        "last_marker_time_sec": float(segment_times[-1]),
        "mean_marker_interval_sec": float(np.mean(np.diff(segment_times))),
        "duration_first_to_last_sec": float(segment_times[-1] - segment_times[0]),
        "marker_samples_nidq": [int(x) for x in rising[start_idx : end_idx + 1]],
        "marker_times_sec": [float(x) for x in segment_times],
    }


def stream_info(kind: str) -> dict[str, object]:
    if kind == "nidq":
        meta = SRC / "gz003t2_g0_t0.nidq.meta"
        bin_path = SRC / "gz003t2_g0_t0.nidq.bin"
        _, values = read_meta(meta)
        return {
            "kind": kind,
            "meta": meta,
            "bin": bin_path,
            "sample_rate": float(values["niSampRate"]),
            "n_channels": int(values["nSavedChans"]),
            "suffix": "nidq",
            "subdir": None,
        }
    imec = SRC / "gz003t2_g0_imec0"
    if kind == "ap":
        meta = imec / "gz003t2_g0_t0.imec0.ap.meta"
        bin_path = imec / "gz003t2_g0_t0.imec0.ap.bin"
    elif kind == "lf":
        meta = imec / "gz003t2_g0_t0.imec0.lf.meta"
        bin_path = imec / "gz003t2_g0_t0.imec0.lf.bin"
    else:
        raise ValueError(kind)
    _, values = read_meta(meta)
    return {
        "kind": kind,
        "meta": meta,
        "bin": bin_path,
        "sample_rate": float(values["imSampRate"]),
        "n_channels": int(values["nSavedChans"]),
        "suffix": f"imec0.{kind}",
        "subdir": "{run_name}_imec0",
    }


def write_events(run_dir: Path, run_name: str, detected: dict[str, object], trials: list[dict[str, str]], streams: list[dict[str, object]], crop_start_sec: float) -> None:
    marker_times = detected["marker_times_sec"]
    marker_samples = detected["marker_samples_nidq"]
    event_path = run_dir / "events.csv"
    with event_path.open("w", newline="") as f:
        fieldnames = [
            "trial_index",
            "sound_file",
            "condition",
            "marker",
            "original_nidq_sample",
            "time_from_original_start_sec",
            "time_from_crop_start_sec",
            "nidq_sample_in_crop",
            "ap_sample_in_crop",
            "lf_sample_in_crop",
            "sentence_onset_time_eprime_ms",
            "fixation_onset_time_eprime_ms",
            "sentence_resp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        stream_by_kind = {s["kind"]: s for s in streams}
        for i, trial in enumerate(trials):
            original_t = float(marker_times[i])
            relative_t = original_t - crop_start_sec
            writer.writerow(
                {
                    "trial_index": i + 1,
                    "sound_file": trial.get("SoundFile", ""),
                    "condition": stimulus_condition(trial.get("SoundFile", "")),
                    "marker": trial.get("marker", ""),
                    "original_nidq_sample": marker_samples[i],
                    "time_from_original_start_sec": f"{original_t:.9f}",
                    "time_from_crop_start_sec": f"{relative_t:.9f}",
                    "nidq_sample_in_crop": int(round(relative_t * stream_by_kind["nidq"]["sample_rate"])),
                    "ap_sample_in_crop": int(round(relative_t * stream_by_kind["ap"]["sample_rate"])),
                    "lf_sample_in_crop": int(round(relative_t * stream_by_kind["lf"]["sample_rate"])),
                    "sentence_onset_time_eprime_ms": trial.get("Sentence.OnsetTime", ""),
                    "fixation_onset_time_eprime_ms": trial.get("Fixation.OnsetTime", ""),
                    "sentence_resp": trial.get("Sentence.RESP", ""),
                }
            )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    nidq_info = stream_info("nidq")
    detected_runs, diagnostics = detect_marker_runs(nidq_info["bin"], nidq_info["meta"])
    streams = [stream_info("ap"), stream_info("lf"), nidq_info]
    behavior_files = sorted(BEH_ROOT.glob("sentence-003-*.txt"))
    if len(behavior_files) != len(detected_runs):
        raise RuntimeError(f"Expected {len(detected_runs)} behavior txt files, found {len(behavior_files)}")

    summary: dict[str, object] = {
        "source_recording": str(SRC),
        "output_root": str(OUT_ROOT),
        "pre_first_marker_sec": PRE_FIRST_MARKER_SEC,
        "post_last_marker_sec": POST_LAST_MARKER_SEC,
        "marker_diagnostics": diagnostics,
        "runs": [],
    }

    for extracted_idx, (detected, behavior_txt) in enumerate(zip(detected_runs, behavior_files), start=1):
        original_run_idx = ORIGINAL_EXPERIMENT_RUN_INDICES[extracted_idx - 1]
        run_name = f"{RUN_NAME_PREFIX}_run{original_run_idx}"
        run_dir = OUT_ROOT / run_name
        trials = parse_eprime_txt(behavior_txt)
        if len(trials) != EXPECTED_TRIALS:
            raise RuntimeError(f"{behavior_txt} has {len(trials)} trials, expected {EXPECTED_TRIALS}")

        first_marker = float(detected["first_marker_time_sec"])
        last_marker = float(detected["last_marker_time_sec"])
        crop_start_sec = max(0.0, first_marker - PRE_FIRST_MARKER_SEC)
        crop_end_sec = last_marker + POST_LAST_MARKER_SEC

        run_manifest: dict[str, object] = {
            "run_name": run_name,
            "original_experiment_run_index": original_run_idx,
            "extracted_actpas_index": extracted_idx,
            "intervening_tasks_since_previous_actpas": INTERVENING_TASKS_BY_ORIGINAL_RUN.get(original_run_idx, []),
            "behavior_txt": str(behavior_txt),
            "first_marker_time_sec": first_marker,
            "last_marker_time_sec": last_marker,
            "crop_start_sec_from_original": crop_start_sec,
            "crop_end_sec_from_original": crop_end_sec,
            "crop_duration_sec": crop_end_sec - crop_start_sec,
            "n_trials": len(trials),
            "streams": {},
        }

        for info in streams:
            sample_rate = float(info["sample_rate"])
            n_channels = int(info["n_channels"])
            src_bin = Path(info["bin"])
            src_meta = Path(info["meta"])
            total_samples = src_bin.stat().st_size // (2 * n_channels)
            start_sample = max(0, int(math.floor(crop_start_sec * sample_rate)))
            end_sample = min(total_samples, int(math.ceil(crop_end_sec * sample_rate)))
            n_samples = end_sample - start_sample
            if n_samples <= 0:
                raise RuntimeError(f"Empty crop for {info['kind']}")

            if info["kind"] == "nidq":
                dst_bin = run_dir / f"{run_name}_t0.nidq.bin"
                dst_meta = run_dir / f"{run_name}_t0.nidq.meta"
            else:
                imec_dir = run_dir / f"{run_name}_imec0"
                dst_bin = imec_dir / f"{run_name}_t0.{info['suffix']}.bin"
                dst_meta = imec_dir / f"{run_name}_t0.{info['suffix']}.meta"

            print(f"Copying {run_name} {info['kind']}: {n_samples} samples -> {dst_bin}", flush=True)
            stream_copy_samples(src_bin, dst_bin, n_channels, start_sample, n_samples)
            notes = (
                f"Extracted ActPasSynSentence {run_name}; "
                f"source={SRC}; crop_start_sec={crop_start_sec:.9f}; crop_end_sec={crop_end_sec:.9f}"
            )
            write_meta(src_meta, dst_meta, dst_bin, sample_rate, n_channels, start_sample, n_samples, notes)
            run_manifest["streams"][info["kind"]] = {
                "source_bin": str(src_bin),
                "output_bin": str(dst_bin),
                "output_meta": str(dst_meta),
                "sample_rate": sample_rate,
                "n_channels": n_channels,
                "start_sample_in_source": start_sample,
                "end_sample_in_source": end_sample,
                "n_samples": n_samples,
                "duration_sec": n_samples / sample_rate,
                "file_size_bytes": dst_bin.stat().st_size,
            }

        write_events(run_dir, run_name, detected, trials, streams, crop_start_sec)
        beh_dir = run_dir / "behavior"
        beh_dir.mkdir(exist_ok=True)
        shutil.copy2(behavior_txt, beh_dir / behavior_txt.name)
        edat = behavior_txt.with_suffix(".edat3")
        if edat.exists():
            shutil.copy2(edat, beh_dir / edat.name)

        (run_dir / "manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n")
        summary["runs"].append(run_manifest)

    (OUT_ROOT / "summary_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Done. Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
