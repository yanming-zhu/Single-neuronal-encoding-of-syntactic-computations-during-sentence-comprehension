#!/usr/bin/env python3
"""Convert gz001 Kilosort spike times to 1000 Hz firing-rate matrices."""

from __future__ import annotations

from pathlib import Path

import kilosort_spikes_to_firing_rate_gz002 as converter


converter.ROOT = Path("neuropixel2025/data/gz001/processed/ActPasSynSentence")
converter.RUNS = ["gz001t_g0_actpas_run1"]


if __name__ == "__main__":
    converter.main()
