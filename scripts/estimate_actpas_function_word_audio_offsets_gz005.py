#!/usr/bin/env python3
"""Estimate function-word onsets for gz005 ActPas active/passive audio."""

from pathlib import Path

import estimate_actpas_function_word_audio_offsets as offsets


ROOT = Path(__file__).resolve().parents[1]
offsets.AUDIO_DIR = ROOT / "neuropixel2025/files/实验设计/程序设计/neuropixel_m255_202511/ActPasSynSentence/scripts"
offsets.OUT_DIR = ROOT / "neuropixel2025/data/gz005/processed/ActPasSynSentence/event_related_active_passive"


if __name__ == "__main__":
    offsets.main()
