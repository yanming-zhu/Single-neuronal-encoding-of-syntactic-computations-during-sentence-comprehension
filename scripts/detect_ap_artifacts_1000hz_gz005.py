#!/usr/bin/env python3
"""Detect gz005 AP artifacts and save a 1000 Hz artifact mask for each run."""

from pathlib import Path

import detect_ap_artifacts_1000hz_gz004 as artifact_detection


artifact_detection.TASK_DIR = Path(__file__).resolve().parents[1] / "neuropixel2025/data/gz005/processed/ActPasSynSentence"
artifact_detection.RUNS = ("gz005t0_g0_actpas_run0", "gz005t0_g3_actpas_run3")


if __name__ == "__main__":
    artifact_detection.main()
