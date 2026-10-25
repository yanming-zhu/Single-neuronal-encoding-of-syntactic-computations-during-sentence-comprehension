from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_verb_three_condition_unit_activity.py"


def load_analysis_module():
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location("verb_three_condition", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scrambled_anchor_character_is_first_canonical_verb_character():
    analysis = load_analysis_module()

    assert set(analysis.SCRAMBLED_VERB_TARGETS) == set(analysis.SCRAMBLED_SENTENCES_BY_SOUND)
    for sound_file, target in analysis.SCRAMBLED_VERB_TARGETS.items():
        sentence = analysis.SCRAMBLED_SENTENCES_BY_SOUND[sound_file]
        verb_id = int(sound_file[-7:-4])
        canonical_verb = analysis.VERB_ORDER[verb_id - 1]
        assert sentence[target["char_index"]] == canonical_verb[0]
