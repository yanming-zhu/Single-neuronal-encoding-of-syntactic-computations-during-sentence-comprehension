#!/usr/bin/env python3
"""Create broader usable cluster sets from gz001 Kilosort metrics."""

from __future__ import annotations

from pathlib import Path

import make_lenient_good_cluster_sets_gz002 as cluster_sets


cluster_sets.TASK_DIR = Path(__file__).resolve().parents[1] / "neuropixel2025/data/gz001/processed/ActPasSynSentence"
cluster_sets.RUNS = ("gz001t_g0_actpas_run1",)


if __name__ == "__main__":
    cluster_sets.main()
