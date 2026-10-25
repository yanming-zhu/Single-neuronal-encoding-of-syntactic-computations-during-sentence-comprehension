#!/usr/bin/env python3
"""Run Kilosort4 for gz001 ActPasSynSentence runs."""

from __future__ import annotations

from pathlib import Path

import run_kilosort4_gz002_actpas as kilosort_runner


kilosort_runner.ROOT = Path("neuropixel2025/data/gz001/processed/ActPasSynSentence")
kilosort_runner.RUNS = ["gz001t_g0_actpas_run1"]


if __name__ == "__main__":
    kilosort_runner.main()
