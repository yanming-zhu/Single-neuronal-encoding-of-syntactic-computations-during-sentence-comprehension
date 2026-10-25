#!/usr/bin/env python3
"""Extract exact-prefix Kimi-VL language states for the frozen 20 sentences.

This is intentionally separate from the frozen BERT/Qwen production pipeline.
It uses the MLX 4-bit Kimi-VL-A3B-Thinking conversion and writes an explicitly
exploratory cache.  Target spans are accepted only when the tokenizer covers
the registered character interval exactly in a prefix ending at that target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.utils import load_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    ROOT
    / "models/llm_exploratory"
    / "mlx-community_Kimi-VL-A3B-Thinking-4bit"
)
DEFAULT_QC = (
    ROOT
    / "reports/multirun_grouped_go_20260713/llm"
    / "model_representation_caches_v1/qwen3_0p6b_base/cache/tokenization_qc.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/multirun_grouped_go_20260713/llm"
    / "exploratory_kimi_vl_a3b_4bit_20260723/representation_cache"
)
MODEL_ID = "mlx-community/Kimi-VL-A3B-Thinking-4bit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def model_revision(model_dir: Path) -> str:
    metadata = model_dir / ".cache/huggingface/download/config.json.metadata"
    if metadata.is_file():
        first = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
        if len(first) == 40:
            return first
    return "local-exploratory"


def source_rows(qc_csv: Path, tokenizer) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    with qc_csv.open(encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            row = dict(source)
            if row["tokenization_status"] != "present":
                rejected.append(
                    {
                        "sentence_id": row["sentence_id"],
                        "constituent": row["constituent"],
                        "target_text": row["target_text"],
                        "reason": "registered_constituent_absent",
                    }
                )
                continue
            start = int(float(row["target_char_start"]))
            stop = int(float(row["target_char_stop"]))
            prefix = row["sentence"][:stop]
            encoded = tokenizer(
                prefix,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
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
                and all(
                    offsets[left][1] == offsets[right][0]
                    for left, right in zip(hits, hits[1:])
                )
            )
            if not exact:
                rejected.append(
                    {
                        "sentence_id": row["sentence_id"],
                        "constituent": row["constituent"],
                        "target_text": row["target_text"],
                        "reason": "kimi_token_boundary_crosses_registered_span",
                        "prefix": prefix,
                        "offsets": offsets,
                    }
                )
                continue
            accepted.append(
                {
                    "sentence_id": row["sentence_id"],
                    "sentence_pair_id": int(row["sentence_pair_id"]),
                    "voice": row["voice"],
                    "sentence": row["sentence"],
                    "constituent": row["constituent"],
                    "target_text": row["target_text"],
                    "target_char_start": start,
                    "target_char_stop": stop,
                    "prefix": prefix,
                    "input_ids": [int(value) for value in encoded["input_ids"]],
                    "offsets": offsets,
                    "token_start": hits[0],
                    "token_stop": hits[-1] + 1,
                    "last_token_index": hits[-1],
                }
            )
    return accepted, rejected


def as_numpy(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value.astype(mx.float32), dtype=np.float32)


def readout(value: mx.array, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        as_numpy(value[0, stop - 1]),
        as_numpy(mx.mean(value[0, start:stop], axis=0)),
    )


def extract_row(language, row: dict[str, object]) -> tuple[np.ndarray, np.ndarray, float]:
    token_ids = np.asarray(row["input_ids"], dtype=np.int32)
    start = int(row["token_start"])
    stop = int(row["token_stop"])
    inputs = mx.array(token_ids[None, :])
    core = language.model
    hidden = core.embed_tokens(inputs)
    mask = create_attention_mask(hidden, None)
    last_states: list[np.ndarray] = []
    mean_states: list[np.ndarray] = []

    last, mean = readout(hidden, start, stop)
    last_states.append(last)
    mean_states.append(mean)

    layers = list(core.layers)
    for index, layer in enumerate(layers):
        hidden = layer(hidden, mask, None)
        # Match Hugging Face hidden-state semantics: embedding, intermediate
        # block outputs, and the final normalized model output.
        if index < len(layers) - 1:
            last, mean = readout(hidden, start, stop)
            last_states.append(last)
            mean_states.append(mean)
    hidden = core.norm(hidden)
    last, mean = readout(hidden, start, stop)
    last_states.append(last)
    mean_states.append(mean)

    logits = language.lm_head(hidden).astype(mx.float32)
    predictable = [position for position in range(start, stop) if position > 0]
    if predictable:
        values = []
        for position in predictable:
            row_logits = logits[0, position - 1]
            log_probability = (
                row_logits[int(token_ids[position])] - mx.logsumexp(row_logits)
            )
            values.append(float(np.asarray(log_probability)))
        surprisal = -float(np.mean(values))
    else:
        surprisal = float("nan")
    return np.stack(last_states), np.stack(mean_states), surprisal


def write_qc(path: Path, rows: list[dict[str, object]], status: str) -> None:
    serializable = []
    for row in rows:
        item = dict(row)
        for key in ("input_ids", "offsets"):
            if key in item:
                item[key] = stable_json(item[key])
        item["kimi_status"] = status
        serializable.append(item)
    keys = sorted({key for row in serializable for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(serializable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-qc", type=Path, default=DEFAULT_QC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Kimi config is missing: {model_dir}")
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if len(weight_files) != 2:
        raise FileNotFoundError("Kimi 4-bit weight download is incomplete")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    accepted, rejected = source_rows(args.source_qc.resolve(), tokenizer)
    rows = [
        row
        for row in accepted
        if row["constituent"] in {"subject_noun", "second_noun", "verb"}
    ]
    required = {
        (row["sentence_id"], row["constituent"])
        for row in rows
        if row["constituent"] in {"subject_noun", "second_noun", "verb"}
    }
    if len(required) != 60:
        raise RuntimeError("Kimi exact-prefix QC lost a confirmatory noun/verb readout")

    started = time.time_ns()
    model = load_model(model_dir, lazy=True)
    language = model.language_model
    config = language.config
    if config.num_hidden_layers != 27 or config.hidden_size != 2048:
        raise RuntimeError("Kimi architecture differs from the exploratory contract")

    hidden_last: list[np.ndarray] = []
    hidden_mean: list[np.ndarray] = []
    surprisal: list[float] = []
    for index, row in enumerate(rows, start=1):
        last, mean, value = extract_row(language, row)
        if last.shape != (28, 2048) or mean.shape != (28, 2048):
            raise RuntimeError("Kimi hidden-state shape differs")
        if not np.isfinite(last).all() or not np.isfinite(mean).all():
            raise RuntimeError("Kimi hidden-state readout is non-finite")
        hidden_last.append(last)
        hidden_mean.append(mean)
        surprisal.append(value)
        print(
            stable_json(
                {
                    "readout": index,
                    "count": len(rows),
                    "sentence_id": row["sentence_id"],
                    "constituent": row["constituent"],
                }
            ),
            flush=True,
        )

    arrays_path = output / "representations.npz"
    np.savez_compressed(
        arrays_path,
        sentence_ids=np.asarray([row["sentence_id"] for row in rows], dtype="U32"),
        sentence_pair_ids=np.asarray([row["sentence_pair_id"] for row in rows], dtype=np.int64),
        voices=np.asarray([row["voice"] for row in rows], dtype="U8"),
        constituent_ids=np.asarray([row["constituent"] for row in rows], dtype="U32"),
        target_text=np.asarray([row["target_text"] for row in rows], dtype="U16"),
        token_start=np.asarray([row["token_start"] for row in rows], dtype=np.int64),
        token_stop=np.asarray([row["token_stop"] for row in rows], dtype=np.int64),
        layer_ids=np.arange(28, dtype=np.int64),
        hidden_last=np.stack(hidden_last).astype(np.float32, copy=False),
        hidden_mean=np.stack(hidden_mean).astype(np.float32, copy=False),
        token_surprisal=np.asarray(surprisal, dtype=np.float32),
    )
    write_qc(output / "tokenization_qc.csv", accepted, "accepted_exact_prefix")
    write_qc(
        output / "representation_readout_qc.csv",
        rows,
        "accepted_core_noun_verb_readout",
    )
    write_qc(output / "tokenization_rejections.csv", rejected, "rejected")

    config_hash = sha256_file(model_dir / "config.json")
    tokenizer_hash = sha256_file(model_dir / "tokenizer.json")
    weight_records = [
        {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in weight_files
    ]
    metadata = {
        "schema_version": "condition-contrast-kimi-exploratory-cache-v1",
        "analysis_role": "exploratory_model_replacement",
        "formal_release_eligible": False,
        "model_id": MODEL_ID,
        "model_revision": model_revision(model_dir),
        "model_directory": str(model_dir),
        "quantization": {"bits": 4, "group_size": 64},
        "source_model": "moonshotai/Kimi-VL-A3B-Thinking",
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "config_sha256": config_hash,
        "tokenizer_sha256": tokenizer_hash,
        "weight_files": weight_records,
        "source_qc_sha256": sha256_file(args.source_qc.resolve()),
        "representations_sha256": sha256_file(arrays_path),
        "extraction_mode": "character_anchored_incremental_prefix",
        "primary_pooling": "last_token",
        "hidden_state_policy": "embedding_intermediate_blocks_final_normalized",
        "readout_count": len(rows),
        "tokenization_accepted_count": len(accepted),
        "noncore_accepted_count": len(accepted) - len(rows),
        "rejected_readout_count": len(rejected),
        "confirmatory_readout_count": len(required),
        "layer_state_count": 28,
        "decoder_block_count": 27,
        "hidden_size": 2048,
        "surprisal_available": True,
        "started_unix_ns": started,
        "finished_unix_ns": time.time_ns(),
        "condition_comparison_only": True,
        "baseline_operation_used": False,
        "warning": (
            "Quantized Thinking/instruction checkpoint and exact-prefix token policy; "
            "results are exploratory and do not replace the frozen BERT/Qwen family."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(stable_json({"status": "complete", "output": str(output), "readouts": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
