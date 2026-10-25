#!/usr/bin/env python3
"""Extract gz001 ActPasSynSentence runs directly from the gz001 SpikeGLX zip."""

from __future__ import annotations

from pathlib import Path

import extract_gz002_actpas as extract


extract.ROOT = Path("neuropixel2025/data/gz001")
extract.ZIP_PATH = extract.ROOT / "neuropixel" / "gz001t_g0.zip"
extract.ZIP_PREFIX = "gz001t_g0"
extract.OUT_ROOT = extract.ROOT / "processed" / "ActPasSynSentence"
extract.BEH_ROOT = extract.ROOT / "beh" / "Act"

extract.RUN_NAME_PREFIX = "gz001t_g0_actpas"
extract.ORIGINAL_EXPERIMENT_RUN_INDICES = [1]
extract.BEHAVIOR_FILE_NAMES = ["sentence-001-1.txt"]


if __name__ == "__main__":
    extract.main()
