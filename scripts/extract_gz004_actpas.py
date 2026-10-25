#!/usr/bin/env python3
"""Extract gz004 ActPasSynSentence runs from the multi-run SpikeGLX zip."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path("neuropixel2025/data/gz004")
ZIP_PATH = ROOT / "neuropixel" / "gz004t1_g0.zip"
OUT_ROOT = ROOT / "processed" / "ActPasSynSentence"
BEH_ROOT = ROOT / "beh" / "Act"

RUN_NAME_PREFIX = "gz004t1"
SOURCE_PREFIXES = ["gz004t1_g0", "gz004t1_g3"]
ORIGINAL_EXPERIMENT_RUN_INDICES = [0, 3]
BEHAVIOR_FILE_NAMES = ["sentence-004-1.txt", "sentence-004-2.txt"]
MARKER_CHANNEL = 1
PRE_FIRST_MARKER_SEC = 1.0
POST_LAST_MARKER_SEC = 6.5
MARKER_THRESHOLD = None
EXPECTED_TRIALS = 85
COPY_CHUNK_BYTES = 256 * 1024 * 1024


def read_meta_text(text: str) -> tuple[list[tuple[str, str]], dict[str, str]]:
    lines: list[tuple[str, str]] = []
    values: dict[str, str] = {}
    for raw in text.splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            lines.append((key, value))
            values[key] = value
    return lines, values


def write_meta(
    src_meta_text: str,
    dst_meta: Path,
    dst_bin: Path,
    sample_rate: float,
    n_channels: int,
    start_sample: int,
    n_samples: int,
    notes: str,
) -> None:
    lines, values = read_meta_text(src_meta_text)
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


def copy_member_range(
    zf: zipfile.ZipFile,
    member: str,
    start_byte: int,
    n_bytes: int,
    dst_bin: Path,
) -> None:
    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src:
        skipped = 0
        while skipped < start_byte:
            block = src.read(min(COPY_CHUNK_BYTES, start_byte - skipped))
            if not block:
                raise RuntimeError(f"Unexpected EOF while skipping {member}")
            skipped += len(block)

        written = 0
        print(f"Copying {member}: {n_bytes} bytes -> {dst_bin}", flush=True)
        with dst_bin.open("wb") as dst:
            while written < n_bytes:
                block = src.read(min(COPY_CHUNK_BYTES, n_bytes - written))
                if not block:
                    raise RuntimeError(f"Unexpected EOF while copying {member}")
                dst.write(block)
                written += len(block)
    if written != n_bytes:
        raise RuntimeError(f"Short copy for {dst_bin}: expected {n_bytes}, got {written}")


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
    return "unknown"


def load_nidq_channel_from_zip(
    zf: zipfile.ZipFile,
    nidq_member: str,
    n_samples: int,
    n_channels: int,
    channel_index: int,
) -> np.ndarray:
    channel = np.empty(n_samples, dtype=np.int16)
    bytes_per_sample = n_channels * 2
    chunk_samples = max(1, COPY_CHUNK_BYTES // bytes_per_sample)
    sample_offset = 0
    with zf.open(nidq_member) as src:
        while sample_offset < n_samples:
            want_samples = min(chunk_samples, n_samples - sample_offset)
            raw = src.read(want_samples * bytes_per_sample)
            if len(raw) != want_samples * bytes_per_sample:
                raise RuntimeError(f"Unexpected EOF while reading {nidq_member}")
            arr = np.frombuffer(raw, dtype=np.int16).reshape(want_samples, n_channels)
            channel[sample_offset : sample_offset + want_samples] = arr[:, channel_index]
            sample_offset += want_samples
    return channel


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


def detect_marker_run(zf: zipfile.ZipFile, prefix: str, nidq_info: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    n_channels = int(nidq_info["n_channels"])
    sample_rate = float(nidq_info["sample_rate"])
    n_samples = int(nidq_info["file_size_bytes"]) // (2 * n_channels)
    channel = load_nidq_channel_from_zip(zf, str(nidq_info["bin_member"]), n_samples, n_channels, MARKER_CHANNEL)
    ch_min = int(channel.min())
    ch_max = int(channel.max())
    threshold = MARKER_THRESHOLD if MARKER_THRESHOLD is not None else (ch_min + ch_max) / 2

    high = channel > threshold
    rising = np.flatnonzero((~high[:-1]) & high[1:]) + 1
    if high.size and high[0]:
        rising = np.concatenate(([0], rising))
    rising_list = rising.astype(int).tolist()
    times = rising.astype(np.float64) / sample_rate

    runs: list[dict[str, object]] = []
    if len(times) > 1:
        start_idx = 0
        for i, interval in enumerate(np.diff(times)):
            if not (2.5 <= interval <= 4.8):
                if i - start_idx + 1 >= 20:
                    runs.append(make_run(rising_list, times, start_idx, i))
                start_idx = i + 1
        if len(times) - 1 - start_idx + 1 >= 20:
            runs.append(make_run(rising_list, times, start_idx, len(times) - 1))

    diagnostics = {
        "source_prefix": prefix,
        "marker_channel": MARKER_CHANNEL,
        "threshold": threshold,
        "channel_min": ch_min,
        "channel_max": ch_max,
        "rising_edges_detected": len(rising_list),
        "candidate_runs": runs,
        "sample_rate": sample_rate,
        "n_samples": n_samples,
    }
    actpas_runs = [r for r in runs if int(r["n_markers"]) == EXPECTED_TRIALS]
    if len(actpas_runs) != 1:
        raise RuntimeError(f"{prefix}: expected 1 ActPas run with {EXPECTED_TRIALS} markers, found {len(actpas_runs)}")
    return actpas_runs[0], diagnostics


def stream_info(zf: zipfile.ZipFile, prefix: str, kind: str) -> dict[str, object]:
    if kind == "nidq":
        meta_member = f"{prefix}/{prefix}_t0.nidq.meta"
        bin_member = f"{prefix}/{prefix}_t0.nidq.bin"
        suffix = "nidq"
        subdir = None
    elif kind == "ap":
        meta_member = f"{prefix}/{prefix}_imec0/{prefix}_t0.imec0.ap.meta"
        bin_member = f"{prefix}/{prefix}_imec0/{prefix}_t0.imec0.ap.bin"
        suffix = "imec0.ap"
        subdir = "{run_name}_imec0"
    elif kind == "lf":
        meta_member = f"{prefix}/{prefix}_imec0/{prefix}_t0.imec0.lf.meta"
        bin_member = f"{prefix}/{prefix}_imec0/{prefix}_t0.imec0.lf.bin"
        suffix = "imec0.lf"
        subdir = "{run_name}_imec0"
    else:
        raise ValueError(kind)

    meta_text = zf.read(meta_member).decode("utf-8", errors="replace")
    _, values = read_meta_text(meta_text)
    sample_rate_key = "niSampRate" if kind == "nidq" else "imSampRate"
    return {
        "kind": kind,
        "meta_member": meta_member,
        "bin_member": bin_member,
        "meta_text": meta_text,
        "sample_rate": float(values[sample_rate_key]),
        "n_channels": int(values["nSavedChans"]),
        "suffix": suffix,
        "subdir": subdir,
        "file_size_bytes": int(values["fileSizeBytes"]),
    }


def write_events(
    run_dir: Path,
    detected: dict[str, object],
    trials: list[dict[str, str]],
    streams: list[dict[str, object]],
    crop_start_sec: float,
) -> None:
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
        for i, original_t in enumerate(marker_times):
            trial = trials[i] if i < len(trials) else {}
            relative_t = float(original_t) - crop_start_sec
            sound_file = trial.get("SoundFile", "")
            writer.writerow(
                {
                    "trial_index": i + 1,
                    "sound_file": sound_file,
                    "condition": stimulus_condition(sound_file),
                    "marker": trial.get("marker", ""),
                    "original_nidq_sample": marker_samples[i],
                    "time_from_original_start_sec": f"{float(original_t):.9f}",
                    "time_from_crop_start_sec": f"{relative_t:.9f}",
                    "nidq_sample_in_crop": int(round(relative_t * float(stream_by_kind["nidq"]["sample_rate"]))),
                    "ap_sample_in_crop": int(round(relative_t * float(stream_by_kind["ap"]["sample_rate"]))),
                    "lf_sample_in_crop": int(round(relative_t * float(stream_by_kind["lf"]["sample_rate"]))),
                    "sentence_onset_time_eprime_ms": trial.get("Sentence.OnsetTime", ""),
                    "fixation_onset_time_eprime_ms": trial.get("Fixation.OnsetTime", ""),
                    "sentence_resp": trial.get("Sentence.RESP", ""),
                }
            )


def load_behavior(path: Path) -> tuple[list[dict[str, str]], str | None]:
    if not path.exists():
        return [{} for _ in range(EXPECTED_TRIALS)], f"missing behavior file: {path}"
    trials = parse_eprime_txt(path)
    if len(trials) != EXPECTED_TRIALS:
        raise RuntimeError(f"{path} has {len(trials)} trials, expected {EXPECTED_TRIALS}")
    return trials, None


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        summary: dict[str, object] = {
            "source_zip": str(ZIP_PATH),
            "output_root": str(OUT_ROOT),
            "pre_first_marker_sec": PRE_FIRST_MARKER_SEC,
            "post_last_marker_sec": POST_LAST_MARKER_SEC,
            "expected_trials": EXPECTED_TRIALS,
            "marker_diagnostics": {},
            "runs": [],
        }

        for source_prefix, original_run_idx, behavior_name in zip(
            SOURCE_PREFIXES, ORIGINAL_EXPERIMENT_RUN_INDICES, BEHAVIOR_FILE_NAMES
        ):
            streams = [stream_info(zf, source_prefix, "ap"), stream_info(zf, source_prefix, "lf"), stream_info(zf, source_prefix, "nidq")]
            detected, diagnostics = detect_marker_run(zf, source_prefix, streams[-1])
            summary["marker_diagnostics"][source_prefix] = diagnostics

            run_name = f"{RUN_NAME_PREFIX}_g{original_run_idx}_actpas_run{original_run_idx}"
            run_dir = OUT_ROOT / run_name
            behavior_txt = BEH_ROOT / behavior_name
            trials, behavior_warning = load_behavior(behavior_txt)

            first_marker = float(detected["first_marker_time_sec"])
            last_marker = float(detected["last_marker_time_sec"])
            crop_start_sec = max(0.0, first_marker - PRE_FIRST_MARKER_SEC)
            crop_end_sec = last_marker + POST_LAST_MARKER_SEC

            run_manifest: dict[str, object] = {
                "run_name": run_name,
                "source_prefix": source_prefix,
                "original_experiment_run_index": original_run_idx,
                "behavior_txt": str(behavior_txt),
                "behavior_warning": behavior_warning,
                "first_marker_time_sec": first_marker,
                "last_marker_time_sec": last_marker,
                "crop_start_sec_from_original": crop_start_sec,
                "crop_end_sec_from_original": crop_end_sec,
                "crop_duration_sec": crop_end_sec - crop_start_sec,
                "n_trials": EXPECTED_TRIALS,
                "streams": {},
            }

            run_dir.mkdir(parents=True, exist_ok=True)
            write_events(run_dir, detected, trials, streams, crop_start_sec)
            if behavior_txt.exists():
                beh_dir = run_dir / "behavior"
                beh_dir.mkdir(exist_ok=True)
                shutil.copy2(behavior_txt, beh_dir / behavior_txt.name)
                edat = behavior_txt.with_suffix(".edat3")
                if edat.exists():
                    shutil.copy2(edat, beh_dir / edat.name)

            for info in streams:
                sample_rate = float(info["sample_rate"])
                n_channels = int(info["n_channels"])
                total_samples = int(info["file_size_bytes"]) // (2 * n_channels)
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

                copy_member_range(
                    zf,
                    str(info["bin_member"]),
                    start_sample * n_channels * 2,
                    n_samples * n_channels * 2,
                    dst_bin,
                )
                notes = (
                    f"Extracted ActPasSynSentence {run_name}; "
                    f"source_zip={ZIP_PATH}; source_prefix={source_prefix}; "
                    f"crop_start_sec={crop_start_sec:.9f}; crop_end_sec={crop_end_sec:.9f}"
                )
                write_meta(
                    str(info["meta_text"]),
                    dst_meta,
                    dst_bin,
                    sample_rate,
                    n_channels,
                    start_sample,
                    n_samples,
                    notes,
                )
                run_manifest["streams"][str(info["kind"])] = {
                    "source_zip_member": str(info["bin_member"]),
                    "output_bin": str(dst_bin),
                    "output_meta": str(dst_meta),
                    "sample_rate": sample_rate,
                    "n_channels": n_channels,
                    "start_sample_in_source": start_sample,
                    "end_sample_in_source": end_sample,
                    "n_samples": n_samples,
                    "duration_sec": n_samples / sample_rate,
                    "file_size_bytes": n_samples * n_channels * 2,
                }

            (run_dir / "manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n")
            summary["runs"].append(run_manifest)

        (OUT_ROOT / "summary_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Done. Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
