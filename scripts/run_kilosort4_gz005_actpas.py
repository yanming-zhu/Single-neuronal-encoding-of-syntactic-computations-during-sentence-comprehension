#!/usr/bin/env python3
"""Run Kilosort4 for gz005 ActPasSynSentence runs."""

from pathlib import Path

import run_kilosort4_gz004_actpas as kilosort_runner


kilosort_runner.ROOT = Path("neuropixel2025/data/gz005/processed/ActPasSynSentence")
kilosort_runner.RUNS = ["gz005t0_g0_actpas_run0", "gz005t0_g3_actpas_run3"]


if __name__ == "__main__":
    kilosort_runner.main()
