#!/usr/bin/env python3
"""Create broader usable cluster sets from gz005 Kilosort metrics."""

from pathlib import Path

import make_lenient_good_cluster_sets_gz004 as cluster_sets


cluster_sets.TASK_DIR = Path(__file__).resolve().parents[1] / "neuropixel2025/data/gz005/processed/ActPasSynSentence"
cluster_sets.RUNS = ("gz005t0_g0_actpas_run0", "gz005t0_g3_actpas_run3")


if __name__ == "__main__":
    cluster_sets.main()
