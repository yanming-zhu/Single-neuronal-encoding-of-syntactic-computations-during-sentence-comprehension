#!/usr/bin/env python3
"""Extract exact-prefix Kimi states for the noun-first-character registry.

This helper intentionally avoids pandas/scipy so it can run in the minimal
MLX environment used by the local Kimi-VL 4-bit model.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.utils import load_model
import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "reports/noun_firstchar_llm_mapping_20260725"
REGISTRY_PATH = OUTPUT_ROOT / "stimulus_registry.csv"
OUTPUT = OUTPUT_ROOT / "representations/kimi_firstchar.npz"
MODEL_DIR = ROOT / "models/llm_exploratory/mlx-community_Kimi-VL-A3B-Thinking-4bit"
MODEL_ID = "mlx-community/Kimi-VL-A3B-Thinking-4bit"


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry() -> list[dict[str, str]]:
    with REGISTRY_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 50 or len({row["observation_id"] for row in rows}) != 50:
        raise RuntimeError("Expected 50 unique first-character registry rows")
    return rows


def tokenize_rows(tokenizer, registry: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in registry:
        prefix = source["prefix"]
        start = int(source["char_index"])
        stop = start + 1
        encoded = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
        target_token = len(offsets) - 1
        boundary_valid = (
            bool(offsets)
            and offsets[target_token][0] <= start < offsets[target_token][1]
            and offsets[target_token][1] == stop
        )
        if not boundary_valid:
            raise RuntimeError(
                f"Kimi tokenizer does not end at the requested character boundary "
                f"for {source['observation_id']}: {offsets}"
            )
        exact = offsets[target_token] == (start, stop)
        rows.append(
            {
                **source,
                "input_ids": [int(value) for value in encoded["input_ids"]],
                "offsets": offsets,
                "token_start": target_token,
                "token_stop": target_token + 1,
                "last_token_index": target_token,
                "token_boundary_status": "exact_char" if exact else "merged_with_left_context",
                "token_span_text": prefix[offsets[target_token][0] : offsets[target_token][1]],
            }
        )
    return rows


def as_numpy(value: mx.array) -> np.ndarray:
    value = value.astype(mx.float32)
    mx.eval(value)
    return np.asarray(value, dtype=np.float32)


def extract_row(language, row: dict[str, object]) -> np.ndarray:
    token_ids = np.asarray(row["input_ids"], dtype=np.int32)
    target = int(row["last_token_index"])
    inputs = mx.array(token_ids[None, :])
    core = language.model
    hidden = core.embed_tokens(inputs)
    mask = create_attention_mask(hidden, None)
    states = [as_numpy(hidden[0, target])]
    layers = list(core.layers)
    for index, layer in enumerate(layers):
        hidden = layer(hidden, mask, None)
        if index < len(layers) - 1:
            states.append(as_numpy(hidden[0, target]))
    hidden = core.norm(hidden)
    states.append(as_numpy(hidden[0, target]))
    return np.stack(states, axis=0)


def write_qc(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "observation_id",
        "prefix",
        "first_char",
        "token_start",
        "token_stop",
        "input_ids",
        "offsets",
        "token_boundary_status",
        "token_span_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "observation_id": row["observation_id"],
                    "prefix": row["prefix"],
                    "first_char": row["first_char"],
                    "token_start": row["token_start"],
                    "token_stop": row["token_stop"],
                    "input_ids": stable_json(row["input_ids"]),
                    "offsets": stable_json(row["offsets"]),
                    "token_boundary_status": row["token_boundary_status"],
                    "token_span_text": row["token_span_text"],
                }
            )


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    rows = tokenize_rows(tokenizer, registry)
    model = load_model(MODEL_DIR, lazy=True)
    language = model.language_model
    hidden_rows: list[np.ndarray] = []
    for index, row in enumerate(rows):
        hidden_rows.append(extract_row(language, row))
        print(
            stable_json({"stage": "extract_kimi", "row": index + 1, "total": len(rows)}),
            flush=True,
        )
    hidden = np.stack(hidden_rows, axis=0)
    np.savez_compressed(
        OUTPUT,
        observation_ids=np.asarray([row["observation_id"] for row in registry]),
        sentence_ids=np.asarray([row["sentence_id"] for row in registry]),
        pair_ids=np.asarray([int(row["pair_id"]) for row in registry], dtype=np.int64),
        conditions=np.asarray([row["condition"] for row in registry]),
        nouns=np.asarray([row["noun"] for row in registry]),
        first_chars=np.asarray([row["first_char"] for row in registry]),
        thematic_roles=np.asarray([row["thematic_role"] for row in registry]),
        surface_positions=np.asarray([row["surface_position"] for row in registry]),
        layer_ids=np.arange(hidden.shape[1], dtype=np.int64),
        hidden_first_char=hidden.astype(np.float32),
    )
    qc_path = OUTPUT.with_suffix(".tokenization_qc.csv")
    write_qc(rows, qc_path)
    metadata = {
        "schema_version": "noun-firstchar-representation-v1",
        "model_id": MODEL_ID,
        "representation": "causal prefix-boundary hidden state immediately after original noun first character",
        "observation_count": len(registry),
        "layer_count": int(hidden.shape[1]),
        "hidden_size": int(hidden.shape[2]),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "tokenization_qc_sha256": sha256_file(qc_path),
        "formal_release_eligible": False,
        "analysis_role": "exploratory_first_character_mapping",
    }
    OUTPUT.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(stable_json({"stage": "extract_kimi_done", "shape": list(hidden.shape)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
