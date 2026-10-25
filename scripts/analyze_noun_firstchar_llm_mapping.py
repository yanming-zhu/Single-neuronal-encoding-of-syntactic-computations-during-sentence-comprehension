#!/usr/bin/env python3
"""Map noun-first-character neural responses to BERT and Kimi states.

The analysis is deliberately item- and patient-aware:

* grammatical observations are the first character of both nouns in all
  20 active/passive sentences;
* scrambled observations are limited to the five recorded ActPas scrambled
  sentences (ten noun identities);
* neural features use usable-lenient units and four fixed 200-ms windows;
* model layers are summarized into fixed early/middle/late thirds;
* cross-position retrieval is evaluated within thematic-role candidate sets;
* inference is performed on five patient-level endpoints, not pooled units.

Kimi extraction requires the local MLX model and Metal access.  The remaining
stages run with the regular scientific Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

from scripts.analyze_noun_role_structure_encoding import (  # noqa: E402
    DATA_ROOT,
    PATIENTS,
    STIMULI,
    UNIT_SET_FILE,
    audio_envelope,
    audio_root,
    decode_mp3,
    onset_from_peak,
    run_npz_path,
    setup_matplotlib,
)


OUTPUT = ROOT / "reports/noun_firstchar_llm_mapping_20260725"
REP_DIR = OUTPUT / "representations"
FIG_DIR = OUTPUT / "figures"

REGISTRY_PATH = OUTPUT / "stimulus_registry.csv"
SCRAMBLED_ONSETS_PATH = OUTPUT / "scrambled_noun_firstchar_audio_onsets.csv"
EVENTS_PATH = OUTPUT / "noun_firstchar_event_metadata.csv"
AVAILABILITY_PATH = OUTPUT / "data_availability.csv"

BERT_MODEL_DIR = ROOT / "models/llm/google-bert_bert-base-chinese"
KIMI_MODEL_DIR = ROOT / "models/llm_exploratory/mlx-community_Kimi-VL-A3B-Thinking-4bit"
BERT_ID = "google-bert/bert-base-chinese"
KIMI_ID = "mlx-community/Kimi-VL-A3B-Thinking-4bit"

WINDOWS_MS = ((0, 200), (200, 400), (400, 600), (600, 800))
WINDOW_LABELS = tuple(f"{left}–{right}" for left, right in WINDOWS_MS)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
PROJECTION_DIM = 64
PROJECTION_SEED = 20260725

ACTIVE_AGENT_NOUNS = tuple(stim.subject_noun for stim in STIMULI if stim.condition == "active")
ACTIVE_PATIENT_NOUNS = tuple(stim.predicate_noun for stim in STIMULI if stim.condition == "active")
NOUN_TO_ROLE = {noun: "agent" for noun in ACTIVE_AGENT_NOUNS} | {
    noun: "patient" for noun in ACTIVE_PATIENT_NOUNS
}

RUNS = (
    ("gz001", "gz001t_g0_actpas_run1"),
    ("gz002", "gz002_g0_actpas_run2"),
    ("gz003", "gz003t2_g0_actpas_run1"),
    ("gz003", "gz003t2_g0_actpas_run4"),
    ("gz004", "gz004t1_g0_actpas_run0"),
    ("gz004", "gz004t1_g3_actpas_run3"),
    ("gz005", "gz005t0_g0_actpas_run0"),
    ("gz005", "gz005t0_g3_actpas_run3"),
)

SCRAMBLED_SENTENCES = {
    "stim_actpas/non001.mp3": "叶虫光了把啃树子",
    "stim_actpas/non003.mp3": "了走骨叼头将小狗",
    "stim_actpas/non005.mp3": "了踢飞球给足员球",
    "stim_actpas/non007.mp3": "爸爸竹爆着将了点",
    "stim_actpas/non009.mp3": "开切奶奶瓜西将了",
}

# Manual syllable peaks were selected from the existing waveform QC.  The
# target is always the first character of the original noun identity, even
# when the two noun characters are separated or reversed in the scramble.
SCRAMBLED_TARGETS = (
    ("stim_actpas/non001.mp3", 1, "虫子", 1, 0.610, 0.450),
    ("stim_actpas/non001.mp3", 1, "树叶", 6, 1.690, 1.510),
    ("stim_actpas/non003.mp3", 3, "骨头", 2, 0.690, 0.565),
    ("stim_actpas/non003.mp3", 3, "小狗", 6, 1.690, 1.545),
    ("stim_actpas/non005.mp3", 5, "足球", 5, 1.525, 1.300),
    ("stim_actpas/non005.mp3", 5, "球员", 7, 1.900, 1.740),
    ("stim_actpas/non007.mp3", 7, "爸爸", 0, 0.280, 0.190),
    ("stim_actpas/non007.mp3", 7, "爆竹", 3, 0.905, 0.790),
    ("stim_actpas/non009.mp3", 9, "奶奶", 2, 0.705, 0.600),
    ("stim_actpas/non009.mp3", 9, "西瓜", 5, 1.440, 1.295),
)


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dirs() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def sentence_id(sound_file: str) -> str:
    return Path(sound_file).stem


def build_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scrambled_nouns = {target[2] for target in SCRAMBLED_TARGETS}
    for stim in STIMULI:
        sid = sentence_id(stim.sound_file)
        pair_id = int(sid[-3:])
        for surface, noun in (("initial", stim.subject_noun), ("post_marker", stim.predicate_noun)):
            char_index = stim.sentence.index(noun)
            rows.append(
                {
                    "observation_id": f"{sid}:{noun}:{surface}",
                    "sentence_id": sid,
                    "pair_id": pair_id,
                    "sound_file": stim.sound_file,
                    "condition": stim.condition,
                    "voice": stim.condition,
                    "sentence": stim.sentence,
                    "noun": noun,
                    "first_char": noun[0],
                    "thematic_role": NOUN_TO_ROLE[noun],
                    "surface_position": surface,
                    "char_index": char_index,
                    "prefix": stim.sentence[: char_index + 1],
                    "scrambled_counterpart_recorded": noun in scrambled_nouns,
                }
            )
    for sound_file, pair_id, noun, char_index, _peak, _left in SCRAMBLED_TARGETS:
        sentence = SCRAMBLED_SENTENCES[sound_file]
        if sentence[char_index] != noun[0]:
            raise RuntimeError(f"Scrambled character mapping differs for {sound_file}/{noun}")
        sid = sentence_id(sound_file)
        rows.append(
            {
                "observation_id": f"{sid}:{noun}:scrambled",
                "sentence_id": sid,
                "pair_id": pair_id,
                "sound_file": sound_file,
                "condition": "scrambled",
                "voice": "scrambled",
                "sentence": sentence,
                "noun": noun,
                "first_char": noun[0],
                "thematic_role": NOUN_TO_ROLE[noun],
                "surface_position": "scrambled",
                "char_index": char_index,
                "prefix": sentence[: char_index + 1],
                "scrambled_counterpart_recorded": True,
            }
        )
    registry = pd.DataFrame(rows)
    if registry.shape[0] != 50 or registry["observation_id"].duplicated().any():
        raise RuntimeError("Expected exactly 50 unique first-character observations")
    registry.to_csv(REGISTRY_PATH, index=False)
    return registry


def scrambled_speech_starts() -> dict[str, float]:
    path = DATA_ROOT / "combined_verb_three_condition_unit_activity/verb_three_condition_audio_onsets.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["condition"].eq("scrambled")]
    return dict(zip(frame["sound_file"].astype(str), frame["speech_start_sec_in_audio"].astype(float)))


def build_scrambled_onsets() -> pd.DataFrame:
    import matplotlib.pyplot as plt

    setup_matplotlib()
    starts = scrambled_speech_starts()
    audio_dir = audio_root()
    rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(5, 1, figsize=(7.2, 8.0), sharex=True)
    by_sound: dict[str, list[tuple[int, str, int, float, float]]] = {}
    for sound_file, pair_id, noun, char_index, peak_sec, left_limit in SCRAMBLED_TARGETS:
        by_sound.setdefault(sound_file, []).append((pair_id, noun, char_index, peak_sec, left_limit))

    role_colors = {"agent": "#0072B2", "patient": "#D55E00"}
    for axis, (sound_file, targets) in zip(axes, by_sound.items()):
        audio = decode_mp3(audio_dir / sound_file)
        times, envelope = audio_envelope(audio)
        speech_start = float(starts[sound_file])
        axis.plot(times, envelope, color="#3F4854", linewidth=1.0)
        axis.axvline(speech_start, color="#888888", linestyle="--", linewidth=0.8)
        for pair_id, noun, char_index, peak_sec, left_limit in targets:
            if char_index == 0:
                onset_audio = speech_start
                source = "speech_start"
            else:
                onset_audio = onset_from_peak(times, envelope, peak_sec, left_limit)
                source = "manual_syllable_peak_plus_envelope_onset"
            onset_relative = max(0.0, onset_audio - speech_start)
            role = NOUN_TO_ROLE[noun]
            rows.append(
                {
                    "sound_file": sound_file,
                    "sentence_id": sentence_id(sound_file),
                    "pair_id": pair_id,
                    "sentence": SCRAMBLED_SENTENCES[sound_file],
                    "noun": noun,
                    "first_char": noun[0],
                    "thematic_role": role,
                    "char_index": char_index,
                    "speech_start_sec_in_audio": speech_start,
                    "target_peak_sec_in_audio": peak_sec,
                    "target_onset_sec_in_audio": onset_audio,
                    "target_onset_sec_from_sentence_onset": onset_relative,
                    "onset_source": source,
                }
            )
            axis.axvline(onset_audio, color=role_colors[role], linewidth=1.5)
            axis.text(
                onset_audio,
                float(envelope.max()) * 1.02,
                f"{noun[0]}·{role}",
                color=role_colors[role],
                fontsize=7,
                ha="center",
                va="bottom",
            )
        axis.set_title(f"{sentence_id(sound_file)}  {SCRAMBLED_SENTENCES[sound_file]}", loc="left", fontsize=8)
        axis.set_ylabel("RMS", fontsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Audio time (s)")
    figure.suptitle("Scrambled noun-first-character onset QC", fontsize=10, fontweight="bold", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(FIG_DIR / "Fig_S1_scrambled_firstchar_onset_QC.png", dpi=300)
    figure.savefig(FIG_DIR / "Fig_S1_scrambled_firstchar_onset_QC.pdf")
    plt.close(figure)
    frame = pd.DataFrame(rows).sort_values(["pair_id", "thematic_role"]).reset_index(drop=True)
    frame.to_csv(SCRAMBLED_ONSETS_PATH, index=False)
    return frame


def trial_bins_path(patient: str, run: str) -> Path:
    return DATA_ROOT / patient / "processed/ActPasSynSentence" / run / "firing_rate_1000Hz/trial_bins_1000Hz.csv"


def build_events(registry: pd.DataFrame, scrambled_onsets: pd.DataFrame) -> pd.DataFrame:
    grammar_events = pd.read_csv(
        DATA_ROOT / "combined_noun_role_structure_encoding/noun_role_event_metadata.csv"
    )
    grammar_events["surface_position"] = grammar_events["role"].map(
        {"subject": "initial", "predicate": "post_marker"}
    )
    grammar_events["sentence_id"] = grammar_events["sound_file"].map(sentence_id)
    grammar_events["observation_id"] = (
        grammar_events["sentence_id"].astype(str)
        + ":"
        + grammar_events["noun"].astype(str)
        + ":"
        + grammar_events["surface_position"].astype(str)
    )
    registry_lookup = registry.set_index("observation_id")
    grammar_events["thematic_role"] = grammar_events["noun"].map(NOUN_TO_ROLE)
    grammar_events["first_char"] = grammar_events["noun"].str[0]
    grammar_events["condition"] = grammar_events["sentence_condition"]
    grammar_events["pair_id"] = grammar_events["sentence_id"].str[-3:].astype(int)
    grammar_events["char_index"] = grammar_events["observation_id"].map(registry_lookup["char_index"])

    scrambled_lookup = scrambled_onsets.set_index(["sound_file", "noun"])
    scrambled_rows: list[dict[str, object]] = []
    for patient, run in RUNS:
        bins = pd.read_csv(trial_bins_path(patient, run))
        bins = bins.loc[bins["sound_file"].isin(SCRAMBLED_SENTENCES)]
        if bins.shape[0] != 5:
            raise RuntimeError(f"Expected five scrambled trials in {patient}/{run}")
        for trial in bins.itertuples(index=False):
            targets = [value for value in SCRAMBLED_TARGETS if value[0] == str(trial.sound_file)]
            for sound_file, pair_id, noun, char_index, _peak, _left in targets:
                onset = scrambled_lookup.loc[(sound_file, noun)]
                relative = float(onset.target_onset_sec_from_sentence_onset)
                event_bin = int(trial.marker_bin) + int(round(relative * 1000))
                sid = sentence_id(sound_file)
                scrambled_rows.append(
                    {
                        "patient": patient,
                        "run": run,
                        "trial_index": int(trial.trial_index),
                        "sound_file": sound_file,
                        "sentence_condition": "scrambled",
                        "sentence": SCRAMBLED_SENTENCES[sound_file],
                        "role": "scrambled",
                        "noun": noun,
                        "noun_onset_sec_from_sentence_onset": relative,
                        "event_bin_1000Hz": event_bin,
                        "event_time_sec": event_bin / 1000.0,
                        "onset_source": str(onset.onset_source),
                        "surface_position": "scrambled",
                        "sentence_id": sid,
                        "observation_id": f"{sid}:{noun}:scrambled",
                        "thematic_role": NOUN_TO_ROLE[noun],
                        "first_char": noun[0],
                        "condition": "scrambled",
                        "pair_id": pair_id,
                        "char_index": char_index,
                    }
                )
    keep = [
        "patient",
        "run",
        "trial_index",
        "sound_file",
        "sentence_condition",
        "sentence",
        "role",
        "noun",
        "first_char",
        "thematic_role",
        "surface_position",
        "condition",
        "sentence_id",
        "pair_id",
        "char_index",
        "observation_id",
        "noun_onset_sec_from_sentence_onset",
        "event_bin_1000Hz",
        "event_time_sec",
        "onset_source",
    ]
    events = pd.concat(
        [grammar_events.reindex(columns=keep), pd.DataFrame(scrambled_rows).reindex(columns=keep)],
        ignore_index=True,
    )
    events = events.sort_values(["patient", "run", "condition", "pair_id", "thematic_role"]).reset_index(drop=True)
    expected_per_run = events.groupby(["patient", "run"]).size()
    if not (expected_per_run == 50).all():
        raise RuntimeError(f"Every run must contain 50 events: {expected_per_run.to_dict()}")
    if events["observation_id"].map(registry_lookup.index.to_series()).isna().any():
        raise RuntimeError("Event registry join failed")
    events.to_csv(EVENTS_PATH, index=False)
    return events


def build_availability(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient, run in RUNS:
        cell = events.loc[events["patient"].eq(patient) & events["run"].eq(run)]
        archive_path = run_npz_path(patient, run, "usable_lenient")
        with np.load(archive_path, allow_pickle=True) as archive:
            labels = archive["ks_labels"].astype(str)
            unit_count = int(archive["unit_ids"].size)
            time_bins = int(archive["firing_rate_hz"].shape[1])
        rows.append(
            {
                "patient": patient,
                "run": run,
                "usable_lenient_units": unit_count,
                "good_units": int(np.sum(labels == "good")),
                "mua_units": int(np.sum(labels == "mua")),
                "recording_bins_1000hz": time_bins,
                "grammar_events": int(np.sum(cell["condition"].isin(["active", "passive"]))),
                "agent_events": int(np.sum(cell["thematic_role"].eq("agent"))),
                "patient_events": int(np.sum(cell["thematic_role"].eq("patient"))),
                "scrambled_events": int(np.sum(cell["condition"].eq("scrambled"))),
                "unique_grammar_nouns": int(cell.loc[cell["condition"].ne("scrambled"), "noun"].nunique()),
                "unique_scrambled_nouns": int(cell.loc[cell["condition"].eq("scrambled"), "noun"].nunique()),
                "all_epochs_in_bounds": bool(
                    ((cell["event_bin_1000Hz"] - 200 >= 0) & (cell["event_bin_1000Hz"] + 800 <= time_bins)).all()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(AVAILABILITY_PATH, index=False)
    return frame


def prepare() -> None:
    ensure_dirs()
    registry = build_registry()
    onsets = build_scrambled_onsets()
    events = build_events(registry, onsets)
    availability = build_availability(events)
    print(stable_json({"stage": "prepare", "registry_rows": len(registry), "event_rows": len(events), "runs": len(availability)}))


def tokenizer_rows(tokenizer, registry: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in registry.itertuples(index=False):
        prefix = str(source.prefix)
        start = int(source.char_index)
        stop = start + 1
        encoded = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
        hits = [
            index
            for index, (left, right) in enumerate(offsets)
            if left >= start and right <= stop and right > left
        ]
        exact = (
            bool(hits)
            and offsets[hits[0]][0] == start
            and offsets[hits[-1]][1] == stop
            and hits[-1] == len(encoded["input_ids"]) - 1
        )
        if not exact:
            raise RuntimeError(
                f"Tokenizer does not isolate first character: {source.observation_id} offsets={offsets}"
            )
        rows.append(
            {
                **source._asdict(),
                "input_ids": [int(value) for value in encoded["input_ids"]],
                "offsets": offsets,
                "token_start": int(hits[0]),
                "token_stop": int(hits[-1] + 1),
                "last_token_index": int(hits[-1]),
            }
        )
    return rows


def save_representation_cache(
    output: Path,
    *,
    model_id: str,
    registry: pd.DataFrame,
    hidden: np.ndarray,
    layer_ids: np.ndarray,
    token_rows: list[dict[str, object]],
) -> None:
    if hidden.shape[0] != len(registry) or hidden.shape[1] != len(layer_ids):
        raise RuntimeError("Representation axes differ from registry")
    np.savez_compressed(
        output,
        observation_ids=np.asarray(registry["observation_id"].astype(str).tolist()),
        sentence_ids=np.asarray(registry["sentence_id"].astype(str).tolist()),
        pair_ids=registry["pair_id"].astype(int).to_numpy(),
        conditions=np.asarray(registry["condition"].astype(str).tolist()),
        nouns=np.asarray(registry["noun"].astype(str).tolist()),
        first_chars=np.asarray(registry["first_char"].astype(str).tolist()),
        thematic_roles=np.asarray(registry["thematic_role"].astype(str).tolist()),
        surface_positions=np.asarray(registry["surface_position"].astype(str).tolist()),
        layer_ids=np.asarray(layer_ids, dtype=np.int64),
        hidden_first_char=np.asarray(hidden, dtype=np.float32),
    )
    qc = pd.DataFrame(
        [
            {
                "observation_id": row["observation_id"],
                "prefix": row["prefix"],
                "first_char": row["first_char"],
                "token_start": row["token_start"],
                "token_stop": row["token_stop"],
                "input_ids": stable_json(row["input_ids"]),
                "offsets": stable_json(row["offsets"]),
            }
            for row in token_rows
        ]
    )
    qc_path = output.with_suffix(".tokenization_qc.csv")
    qc.to_csv(qc_path, index=False)
    metadata = {
        "schema_version": "noun-firstchar-representation-v1",
        "model_id": model_id,
        "representation": "exact-prefix hidden state at original noun first character",
        "observation_count": int(len(registry)),
        "layer_count": int(len(layer_ids)),
        "hidden_size": int(hidden.shape[2]),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "tokenization_qc_sha256": sha256_file(qc_path),
        "formal_release_eligible": False,
        "analysis_role": "exploratory_first_character_mapping",
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_bert() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    ensure_dirs()
    registry = pd.read_csv(REGISTRY_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR, local_files_only=True)
    rows = tokenizer_rows(tokenizer, registry)
    model = AutoModel.from_pretrained(BERT_MODEL_DIR, local_files_only=True)
    model.eval()
    hidden_rows: list[np.ndarray] = []
    with torch.no_grad():
        for row in rows:
            encoded = tokenizer(
                row["prefix"],
                add_special_tokens=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            target = [
                index
                for index, (left, right) in enumerate(offsets)
                if int(left) == int(row["char_index"]) and int(right) == int(row["char_index"]) + 1
            ]
            if len(target) != 1:
                raise RuntimeError(f"BERT special-token target differs for {row['observation_id']}")
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden_rows.append(
                np.stack(
                    [state[0, target[0]].detach().cpu().numpy() for state in outputs.hidden_states],
                    axis=0,
                )
            )
    hidden = np.stack(hidden_rows, axis=0)
    save_representation_cache(
        REP_DIR / "bert_firstchar.npz",
        model_id=BERT_ID,
        registry=registry,
        hidden=hidden,
        layer_ids=np.arange(hidden.shape[1]),
        token_rows=rows,
    )
    print(stable_json({"stage": "extract_bert", "shape": hidden.shape}))


def extract_kimi() -> None:
    import mlx.core as mx
    from mlx_vlm.models.base import create_attention_mask
    from mlx_vlm.utils import load_model
    from transformers import AutoTokenizer

    ensure_dirs()
    registry = pd.read_csv(REGISTRY_PATH)
    tokenizer = AutoTokenizer.from_pretrained(
        KIMI_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    rows = tokenizer_rows(tokenizer, registry)
    model = load_model(KIMI_MODEL_DIR, lazy=True)
    language = model.language_model
    core = language.model
    hidden_rows: list[np.ndarray] = []
    for index, row in enumerate(rows):
        token_ids = np.asarray(row["input_ids"], dtype=np.int32)
        target = int(row["last_token_index"])
        inputs = mx.array(token_ids[None, :])
        state = core.embed_tokens(inputs)
        mask = create_attention_mask(state, None)
        levels: list[np.ndarray] = []
        value = state[0, target].astype(mx.float32)
        mx.eval(value)
        levels.append(np.asarray(value, dtype=np.float32))
        layers = list(core.layers)
        for layer_index, layer in enumerate(layers):
            state = layer(state, mask, None)
            if layer_index < len(layers) - 1:
                value = state[0, target].astype(mx.float32)
                mx.eval(value)
                levels.append(np.asarray(value, dtype=np.float32))
        state = core.norm(state)
        value = state[0, target].astype(mx.float32)
        mx.eval(value)
        levels.append(np.asarray(value, dtype=np.float32))
        hidden_rows.append(np.stack(levels, axis=0))
        print(stable_json({"stage": "extract_kimi", "row": index + 1, "total": len(rows)}), flush=True)
    hidden = np.stack(hidden_rows, axis=0)
    save_representation_cache(
        REP_DIR / "kimi_firstchar.npz",
        model_id=KIMI_ID,
        registry=registry,
        hidden=hidden,
        layer_ids=np.arange(hidden.shape[1]),
        token_rows=rows,
    )
    print(stable_json({"stage": "extract_kimi_done", "shape": hidden.shape}))


def load_representation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if arrays["hidden_first_char"].shape[0] != 50:
        raise RuntimeError(f"Representation row count differs: {path}")
    return arrays


def layer_bands(layer_count: int) -> dict[str, np.ndarray]:
    edges = np.linspace(0, layer_count, 4).round().astype(int)
    return {
        "early": np.arange(edges[0], edges[1]),
        "middle": np.arange(edges[1], edges[2]),
        "late": np.arange(edges[2], edges[3]),
    }


def normalized_band_states(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    hidden = arrays["hidden_first_char"].astype(np.float64)
    norms = np.linalg.norm(hidden, axis=2, keepdims=True)
    hidden = hidden / np.where(norms > 0, norms, 1.0)
    return {
        band: hidden[:, indices, :].mean(axis=1)
        for band, indices in layer_bands(hidden.shape[1]).items()
    }


def neural_windows(patient: str, run: str, run_events: pd.DataFrame, registry: pd.DataFrame) -> np.ndarray:
    with np.load(run_npz_path(patient, run, "usable_lenient"), allow_pickle=True) as archive:
        firing = np.asarray(archive["firing_rate_hz"], dtype=np.float32)
    event_lookup = run_events.set_index("observation_id")
    values = np.empty((len(registry), len(WINDOWS_MS), firing.shape[0]), dtype=np.float32)
    for observation_index, observation_id in enumerate(registry["observation_id"].astype(str)):
        row = event_lookup.loc[observation_id]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError(f"Duplicate run event: {patient}/{run}/{observation_id}")
        event_bin = int(row.event_bin_1000Hz)
        baseline = firing[:, event_bin - 200 : event_bin].mean(axis=1)
        for window_index, (left, right) in enumerate(WINDOWS_MS):
            values[observation_index, window_index] = (
                firing[:, event_bin + left : event_bin + right].mean(axis=1) - baseline
            )
    return values


def random_projection(dimension: int) -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED + dimension)
    matrix = rng.choice(np.array([-1.0, 1.0]), size=(dimension, PROJECTION_DIM))
    return matrix / math.sqrt(PROJECTION_DIM)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    # Dual ridge is stable when units/features outnumber observations.
    gram = x @ x.T
    coefficients = x.T @ np.linalg.solve(gram + alpha * np.eye(len(x)), y)
    return coefficients


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def choose_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    errors = {alpha: [] for alpha in ALPHAS}
    for group in np.unique(groups):
        validation = groups == group
        training = ~validation
        x_train, x_valid = standardize_train_test(x[training], x[validation])
        y_train, y_valid = standardize_train_test(y[training], y[validation])
        for alpha in ALPHAS:
            weights = fit_ridge(x_train, y_train, alpha)
            prediction = x_valid @ weights
            errors[alpha].append(float(np.mean((prediction - y_valid) ** 2)))
    return min(ALPHAS, key=lambda value: (np.mean(errors[value]), value))


def cosine_scores(prediction: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    prediction = prediction / np.where(
        np.linalg.norm(prediction, axis=1, keepdims=True) > 0,
        np.linalg.norm(prediction, axis=1, keepdims=True),
        1.0,
    )
    candidates = candidates / np.where(
        np.linalg.norm(candidates, axis=1, keepdims=True) > 0,
        np.linalg.norm(candidates, axis=1, keepdims=True),
        1.0,
    )
    return prediction @ candidates.T


def retrieval_rows(
    *,
    patient: str,
    run: str,
    model_id: str,
    band: str,
    window_label: str,
    direction: str,
    contrast: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    registry: pd.DataFrame,
) -> list[dict[str, object]]:
    projection = random_projection(y.shape[1])
    projected = y @ projection
    alpha = choose_alpha(
        x[train_indices],
        projected[train_indices],
        registry.iloc[train_indices]["pair_id"].to_numpy(int),
    )
    x_train, x_test = standardize_train_test(x[train_indices], x[test_indices])
    y_train, y_test = standardize_train_test(projected[train_indices], projected[test_indices])
    weights = fit_ridge(x_train, y_train, alpha)
    scores = cosine_scores(x_test @ weights, y_test)
    test_registry = registry.iloc[test_indices].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for role in ("agent", "patient"):
        role_rows = np.flatnonzero(test_registry["thematic_role"].to_numpy(str) == role)
        role_nouns = test_registry.iloc[role_rows]["noun"].astype(str).tolist()
        for local_position, test_row in enumerate(role_rows):
            noun = str(test_registry.iloc[test_row]["noun"])
            within_scores = scores[test_row, role_rows]
            order = np.argsort(-within_scores, kind="stable")
            true_local = role_nouns.index(noun)
            rank = int(np.flatnonzero(order == true_local)[0]) + 1
            predicted_noun = role_nouns[int(order[0])]
            rows.append(
                {
                    "patient": patient,
                    "run": run,
                    "model_id": model_id,
                    "band": band,
                    "window_ms": window_label,
                    "contrast": contrast,
                    "direction": direction,
                    "thematic_role": role,
                    "noun": noun,
                    "candidate_count": len(role_rows),
                    "chance_accuracy": 1.0 / len(role_rows),
                    "alpha": alpha,
                    "predicted_noun": predicted_noun,
                    "correct": int(predicted_noun == noun),
                    "rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                    "true_score": float(within_scores[true_local]),
                }
            )
    return rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def correlation_rdm(patterns: np.ndarray) -> np.ndarray:
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.where(norms > 1e-12, norms, 1.0)
    return 1.0 - normalized @ normalized.T


def partial_rsa(
    neural: np.ndarray,
    model: np.ndarray,
    meta: pd.DataFrame,
    controlled: bool,
) -> float:
    neural_rdm = correlation_rdm(neural)
    model_rdm = correlation_rdm(model)
    triangle = np.triu_indices(len(meta), 1)
    a = average_ranks(neural_rdm[triangle])
    b = average_ranks(model_rdm[triangle])
    if controlled:
        left, right = triangle
        controls = [
            (meta["noun"].to_numpy(str)[left] == meta["noun"].to_numpy(str)[right]).astype(float),
            (meta["surface_position"].to_numpy(str)[left] == meta["surface_position"].to_numpy(str)[right]).astype(float),
            (meta["voice"].to_numpy(str)[left] == meta["voice"].to_numpy(str)[right]).astype(float),
            (meta["pair_id"].to_numpy(int)[left] == meta["pair_id"].to_numpy(int)[right]).astype(float),
            (meta["thematic_role"].to_numpy(str)[left] == meta["thematic_role"].to_numpy(str)[right]).astype(float),
            np.abs(meta["char_index"].to_numpy(float)[left] - meta["char_index"].to_numpy(float)[right]),
        ]
        design = np.column_stack([np.ones(len(a)), *controls])
        a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
        b = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / denominator) if denominator > 0 else float("nan")


def exact_max_t(frame: pd.DataFrame, *, value: str, family_keys: list[str]) -> pd.DataFrame:
    # One-sided exact sign-flip max-T over the five patient endpoints.
    pivot = frame.pivot_table(index="patient", columns=family_keys, values=value, aggfunc="mean")
    patients = list(pivot.index)
    if patients != list(PATIENTS):
        raise RuntimeError(f"Patient endpoint set differs: {patients}")
    values = pivot.to_numpy(float)
    observed = values.mean(axis=0)
    null = []
    for mask in range(1 << len(patients)):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(len(patients))])
        null.append(np.max((values * signs[:, None]).mean(axis=0)))
    null = np.asarray(null)
    p_values = np.asarray([np.mean(null >= estimate - 1e-15) for estimate in observed])
    result = pivot.columns.to_frame(index=False)
    result["patient_mean"] = observed
    result["maxT_p_one_sided"] = p_values
    result["significant_maxT_0p05"] = p_values < 0.05
    return result


def bootstrap_ci(values: np.ndarray, seed: int = 20260725) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def run_analysis() -> None:
    from scipy.stats import wilcoxon

    ensure_dirs()
    registry = pd.read_csv(REGISTRY_PATH)
    events = pd.read_csv(EVENTS_PATH)
    model_paths = {
        BERT_ID: REP_DIR / "bert_firstchar.npz",
        KIMI_ID: REP_DIR / "kimi_firstchar.npz",
    }
    representations = {model_id: load_representation(path) for model_id, path in model_paths.items()}
    for arrays in representations.values():
        if not np.array_equal(arrays["observation_ids"].astype(str), registry["observation_id"].astype(str)):
            raise RuntimeError("Representation registry order differs")
    band_states = {model_id: normalized_band_states(arrays) for model_id, arrays in representations.items()}

    active = np.flatnonzero(registry["condition"].to_numpy(str) == "active")
    passive = np.flatnonzero(registry["condition"].to_numpy(str) == "passive")
    grammar = np.flatnonzero(registry["condition"].isin(["active", "passive"]).to_numpy())
    scrambled = np.flatnonzero(registry["condition"].to_numpy(str) == "scrambled")
    rsa_sets = {
        "agent_grammar": np.flatnonzero(
            registry["condition"].isin(["active", "passive"]).to_numpy()
            & registry["thematic_role"].eq("agent").to_numpy()
        ),
        "patient_grammar": np.flatnonzero(
            registry["condition"].isin(["active", "passive"]).to_numpy()
            & registry["thematic_role"].eq("patient").to_numpy()
        ),
        "scrambled": scrambled,
    }

    retrieval: list[dict[str, object]] = []
    rsa: list[dict[str, object]] = []
    for patient, run in RUNS:
        run_events = events.loc[events["patient"].eq(patient) & events["run"].eq(run)]
        neural = neural_windows(patient, run, run_events, registry)
        for model_id, model_bands in band_states.items():
            for band, model_state in model_bands.items():
                for window_index, window_label in enumerate(WINDOW_LABELS):
                    x = neural[:, window_index, :].astype(np.float64)
                    retrieval.extend(
                        retrieval_rows(
                            patient=patient,
                            run=run,
                            model_id=model_id,
                            band=band,
                            window_label=window_label,
                            direction="active_to_passive",
                            contrast="cross_position",
                            train_indices=active,
                            test_indices=passive,
                            x=x,
                            y=model_state,
                            registry=registry,
                        )
                    )
                    retrieval.extend(
                        retrieval_rows(
                            patient=patient,
                            run=run,
                            model_id=model_id,
                            band=band,
                            window_label=window_label,
                            direction="passive_to_active",
                            contrast="cross_position",
                            train_indices=passive,
                            test_indices=active,
                            x=x,
                            y=model_state,
                            registry=registry,
                        )
                    )
                    retrieval.extend(
                        retrieval_rows(
                            patient=patient,
                            run=run,
                            model_id=model_id,
                            band=band,
                            window_label=window_label,
                            direction="grammar_to_scrambled",
                            contrast="scrambled_transfer",
                            train_indices=grammar,
                            test_indices=scrambled,
                            x=x,
                            y=model_state,
                            registry=registry,
                        )
                    )
                    for set_name, indices in rsa_sets.items():
                        subset_meta = registry.iloc[indices].reset_index(drop=True)
                        for metric in ("raw", "controlled"):
                            rho = partial_rsa(
                                x[indices],
                                model_state[indices],
                                subset_meta,
                                controlled=metric == "controlled",
                            )
                            rsa.append(
                                {
                                    "patient": patient,
                                    "run": run,
                                    "model_id": model_id,
                                    "band": band,
                                    "window_ms": window_label,
                                    "set": set_name,
                                    "metric": metric,
                                    "rho": rho,
                                    "observation_count": len(indices),
                                }
                            )
        print(stable_json({"stage": "analyze_run", "patient": patient, "run": run}), flush=True)

    retrieval_frame = pd.DataFrame(retrieval)
    retrieval_frame.to_csv(OUTPUT / "retrieval_item_predictions.csv", index=False)
    run_retrieval = (
        retrieval_frame.groupby(
            ["patient", "run", "model_id", "band", "window_ms", "contrast", "direction", "thematic_role"],
            sort=False,
        )
        .agg(
            accuracy=("correct", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            chance_accuracy=("chance_accuracy", "first"),
            noun_count=("noun", "size"),
            alpha=("alpha", "first"),
        )
        .reset_index()
    )
    run_retrieval.to_csv(OUTPUT / "retrieval_run_scores.csv", index=False)
    patient_retrieval = (
        run_retrieval.groupby(
            ["patient", "model_id", "band", "window_ms", "contrast", "thematic_role"],
            sort=False,
        )
        .agg(
            accuracy=("accuracy", "mean"),
            mean_reciprocal_rank=("mean_reciprocal_rank", "mean"),
            chance_accuracy=("chance_accuracy", "first"),
        )
        .reset_index()
    )
    patient_retrieval["accuracy_minus_chance"] = patient_retrieval["accuracy"] - patient_retrieval["chance_accuracy"]
    patient_retrieval.to_csv(OUTPUT / "retrieval_patient_scores.csv", index=False)

    retrieval_stats_parts = []
    for (model_id, contrast, role), cell in patient_retrieval.groupby(
        ["model_id", "contrast", "thematic_role"], sort=False
    ):
        stats = exact_max_t(cell, value="accuracy_minus_chance", family_keys=["band", "window_ms"])
        stats.insert(0, "thematic_role", role)
        stats.insert(0, "contrast", contrast)
        stats.insert(0, "model_id", model_id)
        retrieval_stats_parts.append(stats)
    retrieval_stats = pd.concat(retrieval_stats_parts, ignore_index=True)
    retrieval_stats.to_csv(OUTPUT / "retrieval_patient_maxT_inference.csv", index=False)

    comparison_keys = [
        "patient",
        "band",
        "window_ms",
        "contrast",
        "thematic_role",
    ]
    model_comparison = patient_retrieval.pivot_table(
        index=comparison_keys,
        columns="model_id",
        values="accuracy",
        aggfunc="mean",
    ).reset_index()
    model_comparison["kimi_minus_bert_accuracy"] = (
        model_comparison[KIMI_ID] - model_comparison[BERT_ID]
    )
    model_comparison.to_csv(OUTPUT / "kimi_vs_bert_patient_accuracy.csv", index=False)
    comparison_stats_parts = []
    for (contrast, role), cell in model_comparison.groupby(
        ["contrast", "thematic_role"], sort=False
    ):
        stats = exact_max_t(
            cell,
            value="kimi_minus_bert_accuracy",
            family_keys=["band", "window_ms"],
        )
        stats.insert(0, "thematic_role", role)
        stats.insert(0, "contrast", contrast)
        comparison_stats_parts.append(stats)
    pd.concat(comparison_stats_parts, ignore_index=True).to_csv(
        OUTPUT / "kimi_vs_bert_patient_maxT_inference.csv",
        index=False,
    )

    rsa_frame = pd.DataFrame(rsa)
    rsa_frame.to_csv(OUTPUT / "rsa_run_scores.csv", index=False)
    patient_rsa = (
        rsa_frame.groupby(["patient", "model_id", "band", "window_ms", "set", "metric"], sort=False)["rho"]
        .mean()
        .reset_index()
    )
    patient_rsa.to_csv(OUTPUT / "rsa_patient_scores.csv", index=False)
    rsa_stats_parts = []
    for (model_id, set_name, metric), cell in patient_rsa.groupby(
        ["model_id", "set", "metric"], sort=False
    ):
        stats = exact_max_t(cell, value="rho", family_keys=["band", "window_ms"])
        stats.insert(0, "metric", metric)
        stats.insert(0, "set", set_name)
        stats.insert(0, "model_id", model_id)
        rsa_stats_parts.append(stats)
    rsa_stats = pd.concat(rsa_stats_parts, ignore_index=True)
    rsa_stats.to_csv(OUTPUT / "rsa_patient_maxT_inference.csv", index=False)

    summary_rows = []
    for keys, cell in patient_retrieval.groupby(
        ["model_id", "band", "window_ms", "contrast", "thematic_role"], sort=False
    ):
        values = cell["accuracy"].to_numpy(float)
        low, high = bootstrap_ci(values)
        chance = float(cell["chance_accuracy"].iloc[0])
        try:
            test = wilcoxon(values - chance, alternative="greater", method="exact")
            raw_p = float(test.pvalue)
        except ValueError:
            raw_p = 1.0
        summary_rows.append(
            dict(
                zip(
                    ["model_id", "band", "window_ms", "contrast", "thematic_role"],
                    keys,
                )
            )
            | {
                "patient_mean_accuracy": float(values.mean()),
                "patient_median_accuracy": float(np.median(values)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "chance_accuracy": chance,
                "patients_above_chance": int(np.sum(values > chance)),
                "wilcoxon_exact_p_greater_uncorrected": raw_p,
            }
        )
    retrieval_summary = pd.DataFrame(summary_rows).merge(
        retrieval_stats,
        on=["model_id", "band", "window_ms", "contrast", "thematic_role"],
        how="left",
        validate="one_to_one",
    )
    retrieval_summary.to_csv(OUTPUT / "retrieval_group_summary.csv", index=False)

    make_figures(registry, patient_retrieval, retrieval_summary, patient_rsa, rsa_stats)
    write_report(registry, events, patient_retrieval, retrieval_summary, patient_rsa, rsa_stats)
    print(stable_json({"stage": "analysis_done", "retrieval_rows": len(retrieval_frame), "rsa_rows": len(rsa_frame)}))


def figure_style() -> None:
    import matplotlib as mpl

    font_prop = setup_matplotlib()
    mpl.rcParams.update(
        {
            "font.family": font_prop.get_name(),
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def heatmap_matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    return (
        frame.pivot(index="band", columns="window_ms", values=value)
        .reindex(index=["early", "middle", "late"], columns=list(WINDOW_LABELS))
        .to_numpy(float)
    )


def make_figures(
    registry: pd.DataFrame,
    patient_retrieval: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    patient_rsa: pd.DataFrame,
    rsa_stats: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    figure_style()
    role_colors = {"agent": "#0072B2", "patient": "#D55E00"}
    model_labels = {KIMI_ID: "Kimi-VL-A3B (4-bit)", BERT_ID: "BERT-base Chinese"}

    # Figure 1: design and data audit.
    availability = pd.read_csv(AVAILABILITY_PATH)
    fig = plt.figure(figsize=(7.2, 4.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.46, wspace=0.36)
    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    ax.text(0.00, 0.94, "A", fontsize=11, fontweight="bold", va="top")
    ax.text(0.06, 0.94, "Same noun, different online contexts", fontsize=10, fontweight="bold", va="top")
    examples = [
        (0.08, 0.54, "虫子把树叶啃光了", "agent", "active / initial"),
        (0.39, 0.54, "树叶被虫子啃光了", "agent", "passive / post-marker"),
        (0.70, 0.54, "叶虫光了把啃树子", "agent", "scrambled"),
    ]
    for x, y, sentence, role, label in examples:
        ax.text(x, y + 0.20, label, ha="center", fontsize=8, color="#4B5563")
        ax.text(x, y, sentence, ha="center", fontsize=12)
        target = "虫"
        position = sentence.index(target)
        ax.text(x, y - 0.21, f"target: {target}  (character {position + 1})", ha="center", fontsize=7, color=role_colors[role])
    ax.text(
        0.5,
        0.04,
        "Important: noun identity never crosses thematic role; agent/patient is confounded with lexical semantics and animacy.",
        ha="center",
        fontsize=7.2,
        color="#9A3412",
    )

    ax_units = fig.add_subplot(grid[1, 0])
    x = np.arange(len(availability))
    ax_units.bar(x, availability["usable_lenient_units"], color="#56B4E9", edgecolor="white", width=0.74)
    ax_units.set_xticks(x, [f"{row.patient}\n{index + 1}" for index, row in availability.iterrows()])
    ax_units.set_ylabel("Usable-lenient units")
    ax_units.set_title("B  Eight runs nested in five patients", loc="left", fontweight="bold")
    ax_units.grid(axis="y", linewidth=0.4, alpha=0.3)

    ax_items = fig.add_subplot(grid[1, 1])
    labels = ["Agent\ngrammar", "Patient\ngrammar", "Scrambled\nsubset"]
    counts = [20, 20, 10]
    colors = [role_colors["agent"], role_colors["patient"], "#7A7A7A"]
    ax_items.bar(labels, counts, color=colors, width=0.62)
    for index, count in enumerate(counts):
        ax_items.text(index, count + 0.7, str(count), ha="center", fontweight="bold")
    ax_items.set_ylim(0, 23)
    ax_items.set_ylabel("First-character observations / run")
    ax_items.set_title("C  Scrambled covers 10/20 nouns", loc="left", fontweight="bold")
    fig.savefig(FIG_DIR / "Fig_1_design_and_data_audit.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig_1_design_and_data_audit.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Kimi controlled-RSA triptych.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    set_order = ["agent_grammar", "patient_grammar", "scrambled"]
    set_titles = ["Agent nouns (20)", "Patient nouns (20)", "Scrambled nouns (10)"]
    kimi_rsa = patient_rsa.loc[
        patient_rsa["model_id"].eq(KIMI_ID) & patient_rsa["metric"].eq("controlled")
    ]
    limits = []
    matrices = []
    for set_name in set_order:
        group = kimi_rsa.loc[kimi_rsa["set"].eq(set_name)].groupby(["band", "window_ms"])["rho"].mean().reset_index()
        matrix = heatmap_matrix(group, "rho")
        matrices.append(matrix)
        limits.extend(matrix[np.isfinite(matrix)].tolist())
    limit = max(0.08, float(np.max(np.abs(limits))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = None
    for axis, set_name, title, matrix in zip(axes, set_order, set_titles, matrices):
        image = axis.imshow(matrix, cmap="PuOr_r", norm=norm, aspect="auto")
        axis.set_xticks(range(4), WINDOW_LABELS, rotation=35, ha="right")
        axis.set_yticks(range(3), ["Early", "Middle", "Late"])
        axis.set_title(title, loc="left", fontweight="bold")
        stats = rsa_stats.loc[
            rsa_stats["model_id"].eq(KIMI_ID)
            & rsa_stats["set"].eq(set_name)
            & rsa_stats["metric"].eq("controlled")
        ]
        sig = stats.pivot(index="band", columns="window_ms", values="significant_maxT_0p05").reindex(
            index=["early", "middle", "late"], columns=list(WINDOW_LABELS)
        )
        for row in range(3):
            for column in range(4):
                text = f"{matrix[row, column]:+.02f}"
                if bool(sig.iloc[row, column]):
                    text += "*"
                axis.text(column, row, text, ha="center", va="center", fontsize=7, color="black")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Partial Spearman RSA (patient-equal mean)")
    fig.suptitle(
        "Kimi–neural first-character geometry after lexical/position controls",
        fontsize=10,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "Fig_2_kimi_agent_patient_scrambled_RSA_triptych.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig_2_kimi_agent_patient_scrambled_RSA_triptych.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: fixed-band retrieval comparison with patient points.
    fixed_band = "middle"
    panels = [
        ("cross_position", "agent", "Agent: active ↔ passive", 0.10),
        ("cross_position", "patient", "Patient: active ↔ passive", 0.10),
        ("scrambled_transfer", "agent", "Agent: grammar → scrambled", 0.20),
        ("scrambled_transfer", "patient", "Patient: grammar → scrambled", 0.20),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey=False, constrained_layout=True)
    for axis, (contrast, role, title, chance) in zip(axes.flat, panels):
        cell = patient_retrieval.loc[
            patient_retrieval["band"].eq(fixed_band)
            & patient_retrieval["contrast"].eq(contrast)
            & patient_retrieval["thematic_role"].eq(role)
        ]
        for model_index, model_id in enumerate((BERT_ID, KIMI_ID)):
            model_cell = cell.loc[cell["model_id"].eq(model_id)]
            pivot = model_cell.pivot(index="patient", columns="window_ms", values="accuracy").reindex(
                index=list(PATIENTS), columns=list(WINDOW_LABELS)
            )
            x = np.arange(4) + (model_index - 0.5) * 0.08
            mean = pivot.mean(axis=0).to_numpy(float)
            sem = pivot.std(axis=0, ddof=1).to_numpy(float) / math.sqrt(len(PATIENTS))
            color = "#009E73" if model_id == KIMI_ID else "#6B7280"
            marker = "o" if model_id == KIMI_ID else "s"
            axis.errorbar(
                x,
                mean,
                yerr=sem,
                color=color,
                marker=marker,
                linewidth=1.4,
                capsize=2,
                label=model_labels[model_id],
            )
            for row in pivot.to_numpy(float):
                axis.plot(x, row, color=color, alpha=0.14, linewidth=0.65)
        axis.axhline(chance, color="#B91C1C", linestyle="--", linewidth=0.9, label="Chance")
        axis.set_xticks(range(4), WINDOW_LABELS)
        axis.set_ylim(0, max(0.55, axis.get_ylim()[1]))
        axis.set_ylabel("Within-role top-1 accuracy")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", linewidth=0.4, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Fixed middle-third layers; five patient endpoints", fontsize=9.5, fontweight="bold", y=1.07)
    fig.savefig(FIG_DIR / "Fig_3_cross_context_retrieval_BERT_vs_Kimi.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig_3_cross_context_retrieval_BERT_vs_Kimi.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: all fixed Kimi layer bands without best-layer selection.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    for axis, (contrast, role, title, chance) in zip(axes.flat, panels):
        cell = retrieval_summary.loc[
            retrieval_summary["model_id"].eq(KIMI_ID)
            & retrieval_summary["contrast"].eq(contrast)
            & retrieval_summary["thematic_role"].eq(role)
        ]
        colors = {"early": "#56B4E9", "middle": "#009E73", "late": "#CC79A7"}
        markers = {"early": "o", "middle": "s", "late": "^"}
        for band in ("early", "middle", "late"):
            band_cell = cell.loc[cell["band"].eq(band)].set_index("window_ms").reindex(WINDOW_LABELS)
            x = np.arange(4)
            axis.plot(
                x,
                band_cell["patient_mean_accuracy"],
                color=colors[band],
                marker=markers[band],
                linewidth=1.4,
                label=band.capitalize(),
            )
            axis.fill_between(
                x,
                band_cell["bootstrap_ci_low"],
                band_cell["bootstrap_ci_high"],
                color=colors[band],
                alpha=0.11,
                linewidth=0,
            )
            significant = band_cell["significant_maxT_0p05"].fillna(False).to_numpy(bool)
            for xi, yi, flag in zip(x, band_cell["patient_mean_accuracy"], significant):
                if flag:
                    axis.text(xi, yi + 0.035, "*", ha="center", fontweight="bold")
        axis.axhline(chance, color="#B91C1C", linestyle="--", linewidth=0.9)
        axis.set_xticks(range(4), WINDOW_LABELS)
        axis.set_ylim(0, max(0.55, axis.get_ylim()[1]))
        axis.set_ylabel("Within-role top-1 accuracy")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", linewidth=0.4, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Kimi mapping across fixed layer thirds (no best-layer claim)", fontsize=9.5, fontweight="bold", y=1.07)
    fig.savefig(FIG_DIR / "Fig_4_kimi_fixed_layer_thirds_retrieval.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig_4_kimi_fixed_layer_thirds_retrieval.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(
    registry: pd.DataFrame,
    events: pd.DataFrame,
    patient_retrieval: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    patient_rsa: pd.DataFrame,
    rsa_stats: pd.DataFrame,
) -> None:
    def best_rows(model_id: str, contrast: str, role: str) -> pd.DataFrame:
        cell = retrieval_summary.loc[
            retrieval_summary["model_id"].eq(model_id)
            & retrieval_summary["contrast"].eq(contrast)
            & retrieval_summary["thematic_role"].eq(role)
        ].copy()
        return cell.sort_values(["maxT_p_one_sided", "patient_mean_accuracy"], ascending=[True, False]).head(1)

    lines = [
        "# 名词首字神经活动 → LLM 表征映射",
        "",
        "## 数据边界",
        "",
        f"- 分析包含 {len(PATIENTS)} 位患者、{len(RUNS)} 个 run，使用 `usable_lenient` units。",
        "- 每个 run 有 40 个语法句名词首字事件（20 Agent + 20 Patient）。",
        "- Scrambled 只有 5 条录音，覆盖 10/20 个名词（5 Agent + 5 Patient）。",
        "- 同一名词的语义角色不变：有生名词始终是 Agent，无生名词始终是 Patient。因此 Agent/Patient 差异与词义和生命性共线。",
        "",
        "## 主要检验",
        "",
        "1. `active ↔ passive`：在一种语态/表面位置上拟合神经→LLM ridge，在另一种语态/位置上检索同一名词。",
        "2. `grammar → scrambled`：在 40 个语法句事件上拟合，测试 10 个 Scrambled 名词首字。",
        "3. RSA：比较神经与 LLM 的项目间距离几何，并控制名词身份、表面位置、语态、句对和字在句中序位。",
        "4. 固定三个层段（early/middle/late）和四个 200-ms 时窗，不把全层最大值当作主结果。",
        "",
        "## 结果摘要",
        "",
        "- 主结论：Kimi 和 BERT 都没有任何固定层段×时窗通过患者级 exact maxT 校正。下列为探索性局部最大值，不是独立的最佳层证据。",
        "",
    ]
    for model_id, label in ((KIMI_ID, "Kimi"), (BERT_ID, "BERT")):
        for contrast, contrast_label in (("cross_position", "active↔passive"), ("scrambled_transfer", "grammar→scrambled")):
            for role in ("agent", "patient"):
                row = best_rows(model_id, contrast, role).iloc[0]
                lines.append(
                    f"- {label} / {contrast_label} / {role}: 描述性最高患者等权 top-1={row.patient_mean_accuracy:.3f} "
                    f"(机会水平={row.chance_accuracy:.3f}; fixed {row.band}, {row.window_ms} ms; "
                    f"maxT p={row.maxT_p_one_sided:.4f})。"
                )
    significant_retrieval = retrieval_summary.loc[retrieval_summary["significant_maxT_0p05"].fillna(False)]
    significant_rsa = rsa_stats.loc[rsa_stats["significant_maxT_0p05"].fillna(False)]
    lines.extend(
        [
            "",
            f"- 通过患者级 exact maxT 的 retrieval 格子数：{len(significant_retrieval)}。",
            f"- 通过患者级 exact maxT 的 RSA 格子数：{len(significant_rsa)}。",
            "",
            "## 如何解释",
            "",
            "- 若 active↔passive 在角色内高于机会水平，说明同一名词的神经模式可以预测 LLM 表征，并能跨表面位置/语态泛化。",
            "- 若 grammar→scrambled 降至机会水平，而语法句内泛化保留，才更符合‘结构上下文相关映射’。",
            "- 若 Agent、Patient、Scrambled 都相似，更可能映射的是字形/词身份/声学或一般语义，不是抽象句法角色。",
            "- 显著映射也不等于 LLM 和大脑使用同一算法；它只说明两者的表征几何具有可预测关系。",
            "",
            "## 文件",
            "",
            "- `figures/Fig_1_design_and_data_audit.pdf`",
            "- `figures/Fig_2_kimi_agent_patient_scrambled_RSA_triptych.pdf`",
            "- `figures/Fig_3_cross_context_retrieval_BERT_vs_Kimi.pdf`",
            "- `figures/Fig_4_kimi_fixed_layer_thirds_retrieval.pdf`",
            "- `retrieval_group_summary.csv`",
            "- `rsa_patient_maxT_inference.csv`",
        ]
    )
    (OUTPUT / "README_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "extract-bert", "extract-kimi", "analyze", "all-without-kimi"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"prepare", "all-without-kimi"}:
        prepare()
    if args.stage in {"extract-bert", "all-without-kimi"}:
        extract_bert()
    if args.stage == "extract-kimi":
        extract_kimi()
    if args.stage == "analyze":
        run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
