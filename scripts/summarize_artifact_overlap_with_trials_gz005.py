#!/usr/bin/env python3
"""Summarize gz005 artifact overlap for trial windows."""

from pathlib import Path

import summarize_artifact_overlap_with_trials as artifact_overlap


artifact_overlap.TASK_DIR = Path(__file__).resolve().parents[1] / "neuropixel2025/data/gz005/processed/ActPasSynSentence"
artifact_overlap.RUNS = ("gz005t0_g0_actpas_run0", "gz005t0_g3_actpas_run3")


if __name__ == "__main__":
    artifact_overlap.main()
