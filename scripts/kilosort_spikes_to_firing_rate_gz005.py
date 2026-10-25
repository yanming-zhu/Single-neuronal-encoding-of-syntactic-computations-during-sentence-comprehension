#!/usr/bin/env python3
"""Convert gz005 Kilosort spike times to 1000 Hz firing-rate matrices."""

from pathlib import Path

import kilosort_spikes_to_firing_rate_gz004 as converter


converter.ROOT = Path("neuropixel2025/data/gz005/processed/ActPasSynSentence")
converter.RUNS = ["gz005t0_g0_actpas_run0", "gz005t0_g3_actpas_run3"]


if __name__ == "__main__":
    converter.main()
