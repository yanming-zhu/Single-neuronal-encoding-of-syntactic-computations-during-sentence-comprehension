"""Packaging checks use temporary synthetic packages and never touch research data."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


VALIDATOR = Path(__file__).resolve().parents[1] / "scripts" / "validate_code_package.py"
SPEC = importlib.util.spec_from_file_location("package_validator", VALIDATOR)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PackageValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def manifest(self, names):
        with (self.root / "SCRIPT_MANIFEST.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["archive_file", "sha256"], delimiter="\t")
            writer.writeheader()
            for name in names:
                writer.writerow({"archive_file": name, "sha256": hashlib.sha256((self.root / name).read_bytes()).hexdigest()})

    def validate(self):
        return MODULE.validate_package(self.root)

    def test_complete_package_and_canonical_symlink_deduplication(self):
        source = self.write("scripts/example.py", "import json\nVALUE = 1\n")
        (self.root / "alias.py").symlink_to(source)
        self.manifest(["scripts/example.py"])
        result = self.validate()
        self.assertTrue(result["ok"], result)
        syntax = next(item for item in result["passed"] if item["check"] == "python_syntax")
        self.assertEqual(syntax["canonical_files"], 1)

    def test_incomplete_manifest_fails(self):
        self.write("scripts/main.py", "pass\n")
        self.write("scripts/omitted.py", "pass\n")
        self.manifest(["scripts/main.py"])
        result = self.validate()
        self.assertFalse(result["ok"])
        self.assertTrue(any(item.get("file") == "scripts/omitted.py" and item["check"] == "manifest" for item in result["errors"]))

    def test_missing_dotted_local_import_fails(self):
        self.write("scripts/main.py", "from scripts.condition_contrast_figures.data import load_data\n")
        self.manifest(["scripts/main.py"])
        result = self.validate()
        self.assertFalse(result["ok"])
        self.assertTrue(any(item.get("module") == "scripts.condition_contrast_figures.data" for item in result["errors"]))

    def test_namespace_package_import_closure_without_execution(self):
        self.write("scripts/main.py", "from scripts import helper\nfrom scripts.condition_contrast_figures.data import load_data\n")
        self.write("scripts/helper.py", "raise RuntimeError('must not execute')\n")
        self.write("scripts/condition_contrast_figures/data.py", "def load_data():\n    return []\n")
        self.manifest(["scripts/main.py", "scripts/helper.py", "scripts/condition_contrast_figures/data.py"])
        self.assertTrue(self.validate()["ok"])

    def test_missing_project_prefixed_import_is_local_error(self):
        self.write("scripts/main.py", "import analyze_missing_project_module\n")
        self.manifest(["scripts/main.py"])
        result = self.validate()
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0]["check"], "local_import")

    def test_unavailable_external_dependency_is_a_skip(self):
        self.write("scripts/main.py", "import nonexistent_external_dependency_for_archive_test\n")
        self.manifest(["scripts/main.py"])
        result = self.validate()
        self.assertTrue(result["ok"], result)
        self.assertTrue(any(item["check"] == "external_dependency" for item in result["skips"]))

    def test_hash_mismatch_and_broken_symlink_fail(self):
        source = self.write("scripts/main.py", "pass\n")
        self.manifest(["scripts/main.py"])
        source.write_text("VALUE = 2\n")
        (self.root / "broken.py").symlink_to(self.root / "missing.py")
        checks = {item["check"] for item in self.validate()["errors"]}
        self.assertEqual(checks, {"manifest", "symlink"})

    def test_unavailable_node_is_a_skip(self):
        self.write("scripts/build.mjs", "const value = 1;\n")
        self.manifest(["scripts/build.mjs"])
        with patch.object(MODULE.shutil, "which", return_value=None):
            result = self.validate()
        self.assertTrue(result["ok"], result)
        self.assertTrue(any(item["check"] == "javascript_syntax" for item in result["skips"]))

    def test_invalid_python_context_fails_without_execution(self):
        self.write("scripts/main.py", "return 1\n")
        self.manifest(["scripts/main.py"])
        result = self.validate()
        self.assertFalse(result["ok"])
        self.assertTrue(any(item["check"] == "python_syntax" for item in result["errors"]))


if __name__ == "__main__":
    unittest.main()
