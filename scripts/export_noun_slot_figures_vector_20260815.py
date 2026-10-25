#!/usr/bin/env python3
"""Re-export the corrected noun-slot summary and heatmaps from saved derivatives.

This exporter does not rerun the statistical screen. It reloads the frozen
tables and population matrices, then calls the original plotting functions so
that visual content remains tied to the validated analysis outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_noun_slot_corrected_timepoint_screen_20260723 as source


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs/noun_slot_corrected_timepoint_screen_20260723"
EXPORT_DIR = (
    ROOT
    / "reports/noun_slot_unit_analysis_20260723"
    / "vector_exports_20260815"
)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    source.FIG_DIR = EXPORT_DIR

    sessions = pd.read_csv(DATA_DIR / "session_counts.csv")
    subjects = pd.read_csv(DATA_DIR / "subject_counts.csv")
    units = pd.read_csv(DATA_DIR / "corrected_noun_slot_unit_results.csv")
    population_meta = pd.read_csv(DATA_DIR / "population_heatmap_unit_order.csv")
    with np.load(
        DATA_DIR / "corrected_noun_slot_population_heatmaps_data.npz",
        allow_pickle=True,
    ) as archive:
        population_subject = archive["subject_hz"].astype(np.float32, copy=False)
        population_predicate = archive["predicate_hz"].astype(np.float32, copy=False)

    if len(units) != 1782 or len(population_meta) != 1782:
        raise ValueError(
            "Frozen derivatives no longer match the validated 1,782-unit analysis."
        )

    source.plot_screen_summary(sessions, subjects, len(units))
    source.plot_noun_class_control(units)

    payloads: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray, int]] = {}
    for direction in ("subject_gt_predicate", "predicate_gt_subject"):
        raw_count = int(
            (
                units["has_qualifying_segment"]
                & units["qualifying_direction"].eq(direction)
            ).sum()
        )
        metadata, subject_hz, predicate_hz, _ = source.filter_direction_payload(
            units,
            population_meta,
            population_subject,
            population_predicate,
            direction,
        )
        payloads[direction] = (metadata, subject_hz, predicate_hz, raw_count)

    rate_values = np.concatenate(
        [
            matrix.ravel()
            for _, subject_hz, predicate_hz, _ in payloads.values()
            for matrix in (subject_hz, predicate_hz)
        ]
    )
    difference_values = np.concatenate(
        [
            (subject_hz - predicate_hz).ravel()
            for _, subject_hz, predicate_hz, _ in payloads.values()
        ]
    )
    rate_vmax = max(1.0, float(np.nanpercentile(rate_values, 98)))
    difference_vmax = max(
        1.0,
        float(np.nanpercentile(np.abs(difference_values), 98)),
    )

    for direction, (metadata, subject_hz, predicate_hz, raw_count) in payloads.items():
        source.plot_direction_heatmap(
            metadata,
            subject_hz,
            predicate_hz,
            direction,
            rate_vmax,
            difference_vmax,
            raw_count,
        )

    manifest = {
        "analysis_source": "saved corrected noun-slot derivatives",
        "n_units": int(len(units)),
        "screen_positive": int(units["has_qualifying_segment"].sum()),
        "exports": sorted(path.name for path in EXPORT_DIR.iterdir()),
        "svg_policy": (
            "Editable SVG text and vector axes/annotations; dense heatmap image "
            "layers remain losslessly embedded raster content."
        ),
    }
    (EXPORT_DIR / "vector_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
