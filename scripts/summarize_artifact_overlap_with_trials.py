#!/usr/bin/env python3
"""Summarize artifact overlap for trial windows."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "neuropixel2025/data/gz003/processed/ActPasSynSentence"
RUNS = ("gz003t2_g0_actpas_run1", "gz003t2_g0_actpas_run4")
PRE_MS = 1000
POST_MS = 2000


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    all_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for run in RUNS:
        mask = np.load(TASK_DIR / run / "artifact_rejection/ap_artifact_mask_1000Hz.npz")["artifact_mask"]
        trial_bins = load_rows(TASK_DIR / run / "firing_rate_1000Hz/trial_bins_1000Hz.csv")
        run_rows: list[dict[str, object]] = []
        for row in trial_bins:
            marker = int(row["marker_bin"])
            start = max(0, marker - PRE_MS)
            stop = min(mask.size, marker + POST_MS + 1)
            window_mask = mask[start:stop]
            out = {
                "run": run,
                "trial_index": int(row["trial_index"]),
                "sound_file": row["sound_file"],
                "condition": row["condition"],
                "marker_bin": marker,
                "window_start_bin": start,
                "window_stop_bin_exclusive": stop,
                "artifact_fraction_minus1_to_plus2s": float(window_mask.mean()) if window_mask.size else np.nan,
                "artifact_ms_minus1_to_plus2s": int(window_mask.sum()),
                "artifact_free_minus1_to_plus2s": bool(not window_mask.any()),
            }
            run_rows.append(out)
            all_rows.append(out)
        fractions = np.array([r["artifact_fraction_minus1_to_plus2s"] for r in run_rows], dtype=float)
        summary[run] = {
            "n_trials": len(run_rows),
            "median_artifact_fraction": float(np.nanmedian(fractions)),
            "max_artifact_fraction": float(np.nanmax(fractions)),
            "n_trials_artifact_free": int(np.sum(fractions == 0)),
            "n_trials_artifact_fraction_le_0_2": int(np.sum(fractions <= 0.2)),
            "n_trials_artifact_fraction_gt_0_5": int(np.sum(fractions > 0.5)),
            "actpas_n_artifact_fraction_le_0_2": int(
                sum(
                    r["condition"] in {"active_ba_jiang_gei", "passive_bei_rang_jiao"}
                    and r["artifact_fraction_minus1_to_plus2s"] <= 0.2
                    for r in run_rows
                )
            ),
        }

    out_dir = TASK_DIR / "event_related_active_passive"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "trial_artifact_overlap_minus1_to_plus2s.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = out_dir / "trial_artifact_overlap_summary.json"
    summary["outputs"] = {"trial_artifact_overlap_csv": str(csv_path)}
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
