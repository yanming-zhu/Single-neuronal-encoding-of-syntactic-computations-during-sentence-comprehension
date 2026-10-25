#!/usr/bin/env python3
"""Estimate function-word onsets for gz001 ActPas active/passive audio."""

from __future__ import annotations

from pathlib import Path

import estimate_actpas_function_word_audio_offsets as offsets


offsets.OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "neuropixel2025/data/gz001/processed/ActPasSynSentence/event_related_active_passive"
)


if __name__ == "__main__":
    offsets.main()
