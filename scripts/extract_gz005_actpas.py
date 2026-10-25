#!/usr/bin/env python3
"""Extract gz005 ActPasSynSentence runs from separate SpikeGLX zip files."""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import extract_gz004_actpas as base


ROOT = Path("neuropixel2025/data/gz005")
OUT_ROOT = ROOT / "processed" / "ActPasSynSentence"
BEH_ROOT = ROOT / "beh" / "Act"
RUN_NAME_PREFIX = "gz005t0"

SOURCES = [
    {
        "source_prefix": "gz005t0_g0",
        "zip_path": ROOT / "neuropixel" / "gz005t0_g0.zip",
        "original_run_idx": 0,
        "behavior_name": "sentence-005-1.txt",
    },
    {
        "source_prefix": "gz005t0_g3",
        "zip_path": ROOT / "neuropixel" / "gz005t0_g3.zip",
        "original_run_idx": 3,
        "behavior_name": "sentence-005-2.txt",
    },
]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "source_zips": [str(source["zip_path"]) for source in SOURCES],
        "output_root": str(OUT_ROOT),
        "pre_first_marker_sec": base.PRE_FIRST_MARKER_SEC,
        "post_last_marker_sec": base.POST_LAST_MARKER_SEC,
        "expected_trials": base.EXPECTED_TRIALS,
        "marker_diagnostics": {},
        "runs": [],
    }

    for source in SOURCES:
        source_prefix = str(source["source_prefix"])
        zip_path = Path(source["zip_path"])
        original_run_idx = int(source["original_run_idx"])
        behavior_txt = BEH_ROOT / str(source["behavior_name"])
        run_name = f"{RUN_NAME_PREFIX}_g{original_run_idx}_actpas_run{original_run_idx}"
        run_dir = OUT_ROOT / run_name

        with zipfile.ZipFile(zip_path) as zf:
            streams = [
                base.stream_info(zf, source_prefix, "ap"),
                base.stream_info(zf, source_prefix, "lf"),
                base.stream_info(zf, source_prefix, "nidq"),
            ]
            detected, diagnostics = base.detect_marker_run(zf, source_prefix, streams[-1])
            summary["marker_diagnostics"][source_prefix] = diagnostics
            trials, behavior_warning = base.load_behavior(behavior_txt)

            first_marker = float(detected["first_marker_time_sec"])
            last_marker = float(detected["last_marker_time_sec"])
            crop_start_sec = max(0.0, first_marker - base.PRE_FIRST_MARKER_SEC)
            crop_end_sec = last_marker + base.POST_LAST_MARKER_SEC

            run_manifest: dict[str, object] = {
                "run_name": run_name,
                "source_zip": str(zip_path),
                "source_prefix": source_prefix,
                "original_experiment_run_index": original_run_idx,
                "behavior_txt": str(behavior_txt),
                "behavior_warning": behavior_warning,
                "first_marker_time_sec": first_marker,
                "last_marker_time_sec": last_marker,
                "crop_start_sec_from_original": crop_start_sec,
                "crop_end_sec_from_original": crop_end_sec,
                "crop_duration_sec": crop_end_sec - crop_start_sec,
                "n_trials": base.EXPECTED_TRIALS,
                "streams": {},
            }

            run_dir.mkdir(parents=True, exist_ok=True)
            base.write_events(run_dir, detected, trials, streams, crop_start_sec)
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

                base.copy_member_range(
                    zf,
                    str(info["bin_member"]),
                    start_sample * n_channels * 2,
                    n_samples * n_channels * 2,
                    dst_bin,
                )
                notes = (
                    f"Extracted ActPasSynSentence {run_name}; "
                    f"source_zip={zip_path}; source_prefix={source_prefix}; "
                    f"crop_start_sec={crop_start_sec:.9f}; crop_end_sec={crop_end_sec:.9f}"
                )
                base.write_meta(
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
