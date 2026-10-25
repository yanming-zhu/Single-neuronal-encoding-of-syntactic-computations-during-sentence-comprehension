"""Explicit input selection for corrected and historical verb timing results.

The analyzer always runs the corrected anchor definition. Downstream scripts
can inspect historical tables only by selecting ``legacy_20260723`` explicitly.
Directory overrides never change the analysis algorithm or certify provenance.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "neuropixel2025" / "data"
CORRECTED_VERSION = "corrected_20260812"
LEGACY_VERSION = "legacy_20260723"
VERSION_DIRECTORIES = {
    CORRECTED_VERSION: "combined_verb_three_condition_unit_activity_corrected_20260812",
    LEGACY_VERSION: "combined_verb_three_condition_unit_activity",
}
VERSION_LABELS = {
    CORRECTED_VERSION: "corrected scrambled-verb anchors (2026-08-12)",
    LEGACY_VERSION: "historical scrambled-verb anchors (2026-07-23)",
}


def resolve_input(
    version: str | None = None,
    input_dir: str | Path | None = None,
    *,
    data_root: Path = DATA_ROOT,
    validate: bool = True,
) -> tuple[str, Path]:
    """Resolve CLI > environment > corrected default, with no legacy fallback."""
    version = version or os.environ.get("TAN_VERB_THREE_CONDITION_VERSION", CORRECTED_VERSION)
    if version not in VERSION_DIRECTORIES:
        raise ValueError(f"Unknown three-condition version: {version!r}")
    override = input_dir or os.environ.get("TAN_VERB_THREE_CONDITION_OUT_DIR")
    directory = Path(override).expanduser() if override else data_root / VERSION_DIRECTORIES[version]
    # A known directory must not silently receive the other version's label.
    for known_version, name in VERSION_DIRECTORIES.items():
        if validate and directory.name == name and version != known_version:
            raise ValueError(
                f"{directory} belongs to {known_version}; select --verb-version {known_version} explicitly"
            )
    return version, directory


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verb-version",
        choices=tuple(VERSION_DIRECTORIES),
        default=None,
        help="Input provenance; defaults to TAN_VERB_THREE_CONDITION_VERSION or corrected_20260812.",
    )
    parser.add_argument(
        "--verb-input-dir",
        type=Path,
        default=None,
        help="Input/output table directory; overrides TAN_VERB_THREE_CONDITION_OUT_DIR.",
    )


def parse_downstream_args(description: str, *, unit_set: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    add_input_arguments(parser)
    if unit_set:
        parser.add_argument(
            "--unit-set",
            choices=("usable_lenient", "all_clusters", "good_clusters", "strict_qc"),
            default="usable_lenient",
        )
    args = parser.parse_args()
    args.verb_version, args.verb_input_dir = resolve_input(args.verb_version, args.verb_input_dir)
    return args


def input_metadata(version: str, directory: Path) -> dict[str, str]:
    return {
        "input_version": version,
        "input_version_label": VERSION_LABELS[version],
        "input_directory": str(directory.resolve()),
        "provenance_note": "Version is selected explicitly; custom input directories must contain matching tables.",
    }
