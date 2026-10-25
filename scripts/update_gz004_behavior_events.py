#!/usr/bin/env python3
"""Attach gz004 E-Prime behavior logs to previously extracted ActPas runs."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path("neuropixel2025/data/gz004")
TASK_DIR = ROOT / "processed" / "ActPasSynSentence"
BEH_DIR = ROOT / "beh" / "Act"

RUN_BEHAVIOR = {
    "gz004t1_g0_actpas_run0": "sentence-1-1.txt",
    "gz004t1_g3_actpas_run3": "sentence-0044-1.txt",
}
EXPECTED_TRIALS = 85


def parse_eprime_txt(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-16", errors="ignore")
    trials = []
    for frame in text.split("*** LogFrame Start ***")[1:]:
        fields: dict[str, str] = {}
        for line in frame.splitlines():
            line = line.strip("\t\r ")
            if ": " in line:
                key, value = line.split(": ", 1)
                fields[key] = value.strip()
        if fields.get("Procedure") == "TrialProc":
            trials.append(fields)
    return trials


def stimulus_condition(sound_file: str) -> str:
    if "stim_actpas/act" in sound_file:
        return "active_ba_jiang_gei"
    if "stim_actpas/pass" in sound_file:
        return "passive_bei_rang_jiao"
    if "stim_actpas/non" in sound_file:
        return "actpas_scrambled"
    if "stim_syn/synv" in sound_file:
        return "synonym_verb"
    if "stim_syn/synn" in sound_file:
        return "synonym_noun"
    if "stim_syn/non" in sound_file:
        return "syn_scrambled"
    return "other"


def update_events(run: str, behavior_name: str) -> dict[str, object]:
    run_dir = TASK_DIR / run
    events_path = run_dir / "events.csv"
    behavior_path = BEH_DIR / behavior_name
    trials = parse_eprime_txt(behavior_path)
    if len(trials) != EXPECTED_TRIALS:
        raise RuntimeError(f"{behavior_path} has {len(trials)} trials, expected {EXPECTED_TRIALS}")

    with events_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if len(rows) != EXPECTED_TRIALS:
        raise RuntimeError(f"{events_path} has {len(rows)} rows, expected {EXPECTED_TRIALS}")

    for row, trial in zip(rows, trials):
        sound_file = trial.get("SoundFile", "")
        row["sound_file"] = sound_file
        row["condition"] = stimulus_condition(sound_file)
        row["marker"] = trial.get("marker", "")
        row["sentence_onset_time_eprime_ms"] = trial.get("Sentence.OnsetTime", "")
        row["fixation_onset_time_eprime_ms"] = trial.get("Fixation.OnsetTime", "")
        row["sentence_resp"] = trial.get("Sentence.RESP", "")

    with events_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    beh_out = run_dir / "behavior"
    beh_out.mkdir(exist_ok=True)
    shutil.copy2(behavior_path, beh_out / behavior_path.name)
    edat = behavior_path.with_suffix(".edat3")
    if edat.exists():
        shutil.copy2(edat, beh_out / edat.name)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["behavior_txt"] = str(behavior_path)
    manifest["behavior_warning"] = None
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["condition"]] = counts.get(row["condition"], 0) + 1
    return {"run": run, "behavior_txt": str(behavior_path), "condition_counts": counts}


def main() -> None:
    summary = [update_events(run, behavior) for run, behavior in RUN_BEHAVIOR.items()]

    summary_path = TASK_DIR / "behavior_event_update_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
