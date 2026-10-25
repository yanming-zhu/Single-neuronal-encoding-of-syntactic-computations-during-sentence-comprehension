#!/usr/bin/env python3
"""Extract Kimi hidden states for controlled noun-position perturbations.

The perturbations are deliberately diagnostic rather than linguistic claims:

* original: the registered full-noun prefix;
* noun_substituted: cyclic within-role noun replacement, position preserved;
* target_masked: the target noun is replaced by the same neutral noun;
* marker_neutralized: the immediately preceding marker is set to one form;
* target_only: all context before the target noun is removed.

All readouts use the last token of the full target noun and the same local
Kimi-VL 4-bit checkpoint used by the existing Fig. 4 analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from mlx_vlm.utils import load_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_kimi_vl_exploratory import extract_row


MODEL_DIR = (
    ROOT
    / "models/llm_exploratory"
    / "mlx-community_Kimi-VL-A3B-Thinking-4bit"
)
REGISTRY = ROOT / "reports/noun_firstchar_llm_mapping_20260725/stimulus_registry.csv"
OUTPUT = (
    ROOT
    / "reports/fig4_extended_data_20260807/source_data/kimi_perturbations"
)
MODEL_ID = "mlx-community/Kimi-VL-A3B-Thinking-4bit"
CONDITIONS = (
    "original",
    "noun_substituted",
    "target_masked",
    "marker_neutralized",
    "target_only",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_registry() -> list[dict[str, object]]:
    with REGISTRY.open(encoding="utf-8", newline="") as stream:
        rows = [
            dict(row)
            for row in csv.DictReader(stream)
            if row["condition"] in {"active", "passive"}
        ]
    if len(rows) != 40:
        raise RuntimeError(f"Expected 40 registered noun events, found {len(rows)}")
    for row in rows:
        row["pair_id"] = int(row["pair_id"])
    rows.sort(key=lambda row: (int(row["pair_id"]), str(row["condition"]), str(row["surface_position"])))
    return rows


def cyclic_noun_map(rows: list[dict[str, object]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for thematic_role in ("agent", "patient"):
        nouns = list(
            dict.fromkeys(
                str(row["noun"])
                for row in rows
                if row["thematic_role"] == thematic_role
            )
        )
        if len(nouns) != 10:
            raise RuntimeError(f"Expected 10 {thematic_role} nouns, found {len(nouns)}")
        mapping.update({noun: nouns[(index + 1) % len(nouns)] for index, noun in enumerate(nouns)})
    return mapping


def condition_prefix(
    source: dict[str, object],
    condition: str,
    noun_map: dict[str, str],
) -> tuple[str, str]:
    sentence = str(source["sentence"])
    noun = str(source["noun"])
    start = sentence.find(noun)
    if start < 0:
        raise RuntimeError(f"Target noun {noun!r} not found in {sentence!r}")
    if condition == "original":
        target = noun
        prefix = sentence[: start + len(noun)]
    elif condition == "noun_substituted":
        target = noun_map[noun]
        prefix = sentence[:start] + target
    elif condition == "target_masked":
        target = "东西"
        prefix = sentence[:start] + target
    elif condition == "marker_neutralized":
        target = noun
        left = sentence[:start]
        if source["surface_position"] == "post_marker":
            if not left:
                raise RuntimeError("Post-marker noun has no preceding context")
            left = left[:-1] + "把"
        prefix = left + target
    elif condition == "target_only":
        target = noun
        prefix = noun
    else:
        raise ValueError(condition)
    return prefix, target


def tokenize_readout(tokenizer, prefix: str, target: str) -> dict[str, object]:
    start = len(prefix) - len(target)
    stop = len(prefix)
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
        and all(offsets[left][1] == offsets[right][0] for left, right in zip(hits, hits[1:]))
    )
    if not exact:
        raise RuntimeError(
            f"Target tokenization is not exact: prefix={prefix!r}, target={target!r}, offsets={offsets}"
        )
    return {
        "input_ids": [int(value) for value in encoded["input_ids"]],
        "offsets": offsets,
        "token_start": int(hits[0]),
        "token_stop": int(hits[-1] + 1),
    }


def build_rows(tokenizer) -> list[dict[str, object]]:
    registry = read_registry()
    noun_map = cyclic_noun_map(registry)
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for source in registry:
            prefix, target = condition_prefix(source, condition, noun_map)
            tokenized = tokenize_readout(tokenizer, prefix, target)
            rows.append(
                {
                    "perturbation": condition,
                    "observation_id": str(source["observation_id"]),
                    "sentence_id": str(source["sentence_id"]),
                    "pair_id": int(source["pair_id"]),
                    "voice": str(source["voice"]),
                    "thematic_role": str(source["thematic_role"]),
                    "surface_position": str(source["surface_position"]),
                    "source_noun": str(source["noun"]),
                    "target_text": target,
                    "prefix": prefix,
                    **tokenized,
                }
            )
    expected = len(CONDITIONS) * 40
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} perturbation rows, found {len(rows)}")
    return rows


def write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "perturbation",
        "observation_id",
        "sentence_id",
        "pair_id",
        "voice",
        "thematic_role",
        "surface_position",
        "source_noun",
        "target_text",
        "prefix",
        "token_start",
        "token_stop",
        "input_ids",
        "offsets",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["input_ids"] = json.dumps(row["input_ids"], ensure_ascii=False)
            serializable["offsets"] = json.dumps(row["offsets"], ensure_ascii=False)
            writer.writerow(serializable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(model_dir / "config.json")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    rows = build_rows(tokenizer)
    write_metadata(output / "perturbation_readout_registry.csv", rows)

    started = time.time_ns()
    model = load_model(model_dir, lazy=True)
    language = model.language_model
    if language.config.num_hidden_layers != 27 or language.config.hidden_size != 2048:
        raise RuntimeError("Local Kimi architecture differs from the existing Fig. 4 cache")

    hidden_last: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        last, _mean, _surprisal = extract_row(language, row)
        if last.shape != (28, 2048) or not np.isfinite(last).all():
            raise RuntimeError(f"Invalid hidden state at perturbation row {index}")
        hidden_last.append(last)
        print(
            json.dumps(
                {
                    "readout": index,
                    "count": len(rows),
                    "perturbation": row["perturbation"],
                    "observation_id": row["observation_id"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    arrays_path = output / "representations.npz"
    np.savez_compressed(
        arrays_path,
        perturbations=np.asarray([row["perturbation"] for row in rows], dtype="U24"),
        observation_ids=np.asarray([row["observation_id"] for row in rows], dtype="U64"),
        pair_ids=np.asarray([row["pair_id"] for row in rows], dtype=np.int64),
        voices=np.asarray([row["voice"] for row in rows], dtype="U8"),
        thematic_roles=np.asarray([row["thematic_role"] for row in rows], dtype="U12"),
        surface_positions=np.asarray([row["surface_position"] for row in rows], dtype="U16"),
        source_nouns=np.asarray([row["source_noun"] for row in rows], dtype="U16"),
        target_text=np.asarray([row["target_text"] for row in rows], dtype="U16"),
        layer_ids=np.arange(28, dtype=np.int64),
        hidden_last=np.stack(hidden_last).astype(np.float32, copy=False),
    )
    metadata = {
        "model_id": MODEL_ID,
        "model_directory": str(model_dir),
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "conditions": list(CONDITIONS),
        "readout_count": len(rows),
        "layer_state_count": 28,
        "hidden_size": 2048,
        "readout_policy": "full-target last token at exact incremental prefix",
        "started_unix_ns": started,
        "finished_unix_ns": time.time_ns(),
        "representations_sha256": file_sha256(arrays_path),
        "interpretation": "Exploratory input perturbation; target-only removes serial and local prefix context together.",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mx.eval(mx.array([0]))
    print(json.dumps({"output": str(output), "readouts": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
