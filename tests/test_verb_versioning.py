"""Small, data-free regression tests for corrected/historical verb input routing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

import analyze_verb_three_condition_unit_activity as analysis
import make_fig3_verb_noun_slot_180mm as fig3
from summarize_verb_three_condition_population_differences import compute_timing_tests
from verb_three_condition_versioning import (
    CORRECTED_VERSION,
    LEGACY_VERSION,
    VERSION_DIRECTORIES,
    resolve_input,
)


class VerbVersioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        for name in ("TAN_VERB_THREE_CONDITION_VERSION", "TAN_VERB_THREE_CONDITION_OUT_DIR"):
            os.environ.pop(name, None)

    def write_tables(self, version, counts):
        directory = self.root / VERSION_DIRECTORIES[version]
        directory.mkdir()
        rows = []
        for condition, count in counts.items():
            for index in range(count):
                unit_id = len(rows)
                rows.append({
                    "patient": "test_patient",
                    "run": "test_run",
                    "unit_id": unit_id,
                    "ks_label": "good",
                    "preferred_condition": condition,
                    "first_significant_onset_ms": 10 + 30 * unit_id,
                    "preferred_peak_ms": 70 + 40 * unit_id,
                    "effect_peak_ms": 80 + 45 * unit_id,
                })
        timing = pd.DataFrame(rows)
        timing.to_csv(directory / "verb_three_condition_population_timing_by_unit.csv", index=False)
        results = timing[["patient", "run", "unit_id"]].copy()
        # The analyzed population can be larger than the selected timing sample.
        results.loc[len(results)] = ["test_patient", "test_run", 999]
        results.to_csv(directory / "verb_three_condition_unit_results.csv", index=False)
        (directory / "verb_three_condition_summary.json").write_text(json.dumps({"unit_set": "synthetic_fixture"}))
        # This intentionally stale table must never determine Fig. 3 tests.
        (directory / "verb_three_condition_population_timing_tests.csv").write_text("not_a_valid_timing_test\n-99\n")
        return directory, timing

    def test_corrected_anchor_characters(self):
        self.assertEqual(set(analysis.SCRAMBLED_VERB_TARGETS), set(analysis.SCRAMBLED_SENTENCES_BY_SOUND))
        for sound, target in analysis.SCRAMBLED_VERB_TARGETS.items():
            canonical = analysis.VERB_ORDER[int(sound[-7:-4]) - 1]
            self.assertEqual(analysis.SCRAMBLED_SENTENCES_BY_SOUND[sound][target["char_index"]], canonical[0])

    def test_default_and_explicit_historical_routing(self):
        version, path = resolve_input(data_root=self.root)
        self.assertEqual(version, CORRECTED_VERSION)
        self.assertEqual(path, self.root / VERSION_DIRECTORIES[CORRECTED_VERSION])
        version, path = resolve_input(LEGACY_VERSION, data_root=self.root)
        self.assertEqual(path, self.root / VERSION_DIRECTORIES[LEGACY_VERSION])

    def test_environment_and_explicit_argument_precedence(self):
        os.environ["TAN_VERB_THREE_CONDITION_VERSION"] = LEGACY_VERSION
        os.environ["TAN_VERB_THREE_CONDITION_OUT_DIR"] = str(self.root / "env_tables")
        self.assertEqual(resolve_input(), (LEGACY_VERSION, self.root / "env_tables"))
        self.assertEqual(resolve_input(CORRECTED_VERSION, self.root / "cli_tables"),
                         (CORRECTED_VERSION, self.root / "cli_tables"))

    def test_historical_directory_requires_explicit_version(self):
        historical = self.root / VERSION_DIRECTORIES[LEGACY_VERSION]
        with self.assertRaisesRegex(ValueError, "explicitly"):
            resolve_input(input_dir=historical)
        self.assertEqual(resolve_input(LEGACY_VERSION, historical)[1], historical)

    def test_both_versions_use_table_counts_and_recalculate_comparisons(self):
        for version, counts in [
            (CORRECTED_VERSION, {"active": 3, "passive": 2, "scrambled": 2}),
            (LEGACY_VERSION, {"active": 2, "passive": 1, "scrambled": 1}),
        ]:
            with self.subTest(version=version):
                directory, timing = self.write_tables(version, counts)
                long, tests = fig3.load_three_condition_timing(directory, version=version)
                observed = long.drop_duplicates(["patient", "run", "unit_id"])["preferred_condition"].value_counts().to_dict()
                self.assertEqual(observed, counts)
                self.assertEqual(len(long), 2 * sum(counts.values()))
                self.assertEqual(long.attrs["n_units_analyzed"], sum(counts.values()) + 1)
                self.assertEqual(long.attrs["n_units_with_timing"], sum(counts.values()))
                self.assertEqual(long.attrs["input_version"], version)
                self.assertEqual(long.attrs["unit_set"], "synthetic_fixture")
                expected = stats.mannwhitneyu(
                    timing.loc[timing.preferred_condition.eq("active"), "first_significant_onset_ms"],
                    timing.loc[timing.preferred_condition.eq("passive"), "first_significant_onset_ms"],
                    alternative="two-sided",
                ).pvalue
                actual = tests.loc[tests.metric.eq("first_significant_onset_ms") & tests.comparison.eq("active_vs_passive"), "p_value"].iloc[0]
                self.assertAlmostEqual(actual, expected)

    def test_duplicate_timing_entries_are_rejected(self):
        directory, timing = self.write_tables(CORRECTED_VERSION, {"active": 2, "passive": 1, "scrambled": 1})
        pd.concat([timing, timing.iloc[:1]]).to_csv(directory / "verb_three_condition_population_timing_by_unit.csv", index=False)
        with self.assertRaisesRegex(ValueError, "unique"):
            fig3.load_three_condition_timing(directory, version=CORRECTED_VERSION)

    def test_constant_timing_does_not_crash_kruskal(self):
        timing = pd.DataFrame({"preferred_condition": ["active", "passive", "scrambled"]})
        for column in ("first_significant_onset_ms", "preferred_peak_ms", "effect_peak_ms"):
            timing[column] = 10
        tests = pd.DataFrame(compute_timing_tests(timing))
        self.assertTrue(tests.loc[tests.test.eq("kruskal"), "p_value"].isna().all())
        self.assertTrue((tests.loc[tests.test.eq("mannwhitneyu"), "p_value"] == 1).all())

    def test_relocated_copy_and_analyzer_environment_overrides(self):
        relocated = self.root / "relocated_package"
        shutil.copytree(PACKAGE_ROOT / "scripts", relocated / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
        code = (
            "import json; import analyze_verb_three_condition_unit_activity as a; "
            "print(json.dumps([str(a.ROOT),str(a.OUT_DIR),str(a.REPORT_PATH)]))"
        )
        env = dict(os.environ, PYTHONPATH=str(relocated / "scripts"), PYTHONDONTWRITEBYTECODE="1")
        env.pop("TAN_VERB_THREE_CONDITION_REPORT_PATH", None)
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, cwd=self.root)
        root, out, report = map(Path, json.loads(proc.stdout))
        self.assertEqual(root, relocated)
        self.assertEqual(out, relocated / "neuropixel2025/data" / VERSION_DIRECTORIES[CORRECTED_VERSION])
        self.assertEqual(report, relocated / "reports/verb_three_condition_unit_activity_corrected_20260812.md")
        env.update(TAN_VERB_THREE_CONDITION_OUT_DIR=str(self.root / "custom_tables"),
                   TAN_VERB_THREE_CONDITION_REPORT_PATH=str(self.root / "custom_report.md"))
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, cwd=self.root)
        _, out, report = map(Path, json.loads(proc.stdout))
        self.assertEqual(out, self.root / "custom_tables")
        self.assertEqual(report, self.root / "custom_report.md")


if __name__ == "__main__":
    unittest.main()
