#!/usr/bin/env python3
"""Detect gz001 AP artifacts and save a 1000 Hz artifact mask."""

from __future__ import annotations

from pathlib import Path

import detect_ap_artifacts_1000hz_gz002 as artifact_detection


artifact_detection.TASK_DIR = Path(__file__).resolve().parents[1] / "neuropixel2025/data/gz001/processed/ActPasSynSentence"
artifact_detection.RUNS = ("gz001t_g0_actpas_run1",)


if __name__ == "__main__":
    artifact_detection.main()
