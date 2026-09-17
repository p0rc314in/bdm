#!/usr/bin/env python3
"""Prepare an immutable multitask exact-recall stream for native SDM."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


FORMAT = "elastic_sdm_adaptive_recall_suite_v1"

OUTPUT_CLASSES = 192
POINTER_KEYS = 192
POINTER_HOPS = (1, 2, 4, 8)
SPAN_KEYS = 64
SPAN_VALUES = 64
MAXIMUM_SPAN = 16
QUERIES = 16

POINTER_MAP_OFFSET = 0
POINTER_MAP_TOKENS = POINTER_KEYS * POINTER_KEYS
POINTER_QUERY_OFFSET = POINTER_MAP_OFFSET + POINTER_MAP_TOKENS
POINTER_QUERY_TOKENS = len(POINTER_HOPS) * POINTER_KEYS
SPAN_MAP_OFFSET = POINTER_QUERY_OFFSET + POINTER_QUERY_TOKENS
SPAN_MAP_TOKENS = SPAN_KEYS * MAXIMUM_SPAN * SPAN_VALUES
SPAN_QUERY_OFFSET = SPAN_MAP_OFFSET + SPAN_MAP_TOKENS
SPAN_QUERY_TOKENS = SPAN_KEYS * MAXIMUM_SPAN
OVERWRITE_MAP_OFFSET = SPAN_QUERY_OFFSET + SPAN_QUERY_TOKENS
OVERWRITE_MAP_TOKENS = SPAN_MAP_TOKENS
OVERWRITE_QUERY_OFFSET = OVERWRITE_MAP_OFFSET + OVERWRITE_MAP_TOKENS
OVERWRITE_QUERY_TOKENS = SPAN_QUERY_TOKENS
VOCAB_SIZE = OVERWRITE_QUERY_OFFSET + OVERWRITE_QUERY_TOKENS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(
    root: Path,
    path: Path,
    dtype: str,
    shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": dtype,
        "shape": list(shape),
    }


def build_conditions() -> tuple[dict[str, Any], ...]:
    conditions: list[dict[str, Any]] = []
    for associations in (32, 64, 128, 192):
        for hops in POINTER_HOPS:
            conditions.append(
                {
                    "id": f"pointer_h{hops}_n{associations}",
                    "family": "pointer_chase",
                    "parameters": {
                        "associations": associations,
                        "hops": hops,
                    },
                    "memory_tokens": associations,
                    "sequence_length": associations + QUERIES,
                    "queries": QUERIES,
                    "groups": QUERIES,
                    "group_size": 1,
                    "output_support": POINTER_KEYS,
                }
            )
    for associations in (16, 32):
        for span_length in (1, 4, 8, 16):
            query_keys = QUERIES // span_length
            conditions.append(
                {
                    "id": f"span_s{span_length}_n{associations}",
                    "family": "span_recall",
                    "parameters": {
                        "associations": associations,
                        "span_length": span_length,
                        "query_keys": query_keys,
                    },
                    "memory_tokens": associations * span_length,
                    "sequence_length": associations * span_length + QUERIES,
                    "queries": QUERIES,
                    "groups": query_keys,
                    "group_size": span_length,
                    "output_support": SPAN_VALUES,
                }
            )
    for associations in (16, 32):
        for versions in (1, 2, 4):
            span_length = 4
            query_keys = QUERIES // span_length
            conditions.append(
                {
                    "id": f"overwrite_v{versions}_n{associations}",
                    "family": "overwrite_recall",
                    "parameters": {
                        "associations": associations,
                        "versions": versions,
                        "span_length": span_length,
                        "query_keys": query_keys,
                    },
                    "memory_tokens": associations * span_length * versions,
                    "sequence_length": (
                        associations * span_length * versions + QUERIES
                    ),
                    "queries": QUERIES,
                    "groups": query_keys,
                    "group_size": span_length,
                    "output_support": SPAN_VALUES,
                }
            )
    for index, condition in enumerate(conditions):
        condition["index"] = index
    return tuple(conditions)


CONDITIONS = build_conditions()
CONDITION_BY_ID = {str(row["id"]): row for row in CONDITIONS}


def balanced_schedule(
    generator: np.random.Generator,
    condition_count: int,
    steps: int,
) -> np.ndarray:
    schedule = np.resize(
        np.arange(condition_count, dtype=np.uint16),
        steps,
    )
    generator.shuffle(schedule)
    return schedule


def _unique_keys(
    generator: np.random.Generator,
    batch_size: int,
    vocabulary: int,
    count: int,
) -> np.ndarray:
    scores = generator.random((batch_size, vocabulary))
    return np.argsort(scores, axis=1)[:, :count].astype(np.uint16)


def _pointer_batch(
    generator: np.random.Generator,
    batch_size: int,
    condition: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    associations = int(condition["parameters"]["associations"])
    hops = int(condition["parameters"]["hops"])
    hop_index = POINTER_HOPS.index(hops)
    keys = _unique_keys(
        generator,
        batch_size,
        POINTER_KEYS,
        associations,
    )
    rows = np.arange(batch_size)[:, None]
    cycle_order = np.argsort(
        generator.random((batch_size, associations)),
        axis=1,
    )
    cycle_keys = keys[rows, cycle_order]
    cycle_targets = np.roll(cycle_keys, -1, axis=1)
    table = np.zeros((batch_size, POINTER_KEYS), dtype=np.uint16)
    table[rows, cycle_keys] = cycle_targets
    mapping_order = np.argsort(
        generator.random((batch_size, associations)),
        axis=1,
    )
    sources = keys[rows, mapping_order]
    targets = table[rows, sources]
    query_order = np.argsort(
        generator.random((batch_size, associations)),
        axis=1,
    )[:, :QUERIES]
    query_keys = keys[rows, query_order]

    labels = query_keys.copy()
    for _ in range(hops):
        labels = table[rows, labels]

    mappings = (
        POINTER_MAP_OFFSET + sources.astype(np.uint32) * POINTER_KEYS + targets
    )
    queries = (
        POINTER_QUERY_OFFSET
        + hop_index * POINTER_KEYS
        + query_keys.astype(np.uint32)
    )
    tokens = np.concatenate((mappings, queries), axis=1)
    return tokens.astype(np.uint32, copy=False), labels.astype(np.uint8)


def _span_batch(
    generator: np.random.Generator,
    batch_size: int,
    condition: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    parameters = condition["parameters"]
    associations = int(parameters["associations"])
    span_length = int(parameters["span_length"])
    query_keys_count = int(parameters["query_keys"])
    versions = int(parameters.get("versions", 1))
    keys = _unique_keys(
        generator,
        batch_size,
        SPAN_KEYS,
        associations,
    )
    values = generator.integers(
        0,
        SPAN_VALUES,
        size=(batch_size, versions, associations, span_length),
        dtype=np.uint16,
    )
    rows = np.arange(batch_size)[:, None]
    version_orders = np.argsort(
        generator.random((batch_size, versions, associations)),
        axis=2,
    )
    version_keys = np.take_along_axis(
        np.broadcast_to(keys[:, None, :], version_orders.shape),
        version_orders,
        axis=2,
    )
    version_values = np.take_along_axis(
        values,
        version_orders[:, :, :, None],
        axis=2,
    )
    slots = np.arange(span_length, dtype=np.uint32)
    mapping_offset = OVERWRITE_MAP_OFFSET if overwrite else SPAN_MAP_OFFSET
    mappings = (
        mapping_offset
        + (
            version_keys.astype(np.uint32)[:, :, :, None] * MAXIMUM_SPAN
            + slots
        )
        * SPAN_VALUES
        + version_values.astype(np.uint32)
    ).reshape(batch_size, -1)

    query_order = np.argsort(
        generator.random((batch_size, associations)),
        axis=1,
    )[:, :query_keys_count]
    query_keys = keys[rows, query_order]
    query_offset = OVERWRITE_QUERY_OFFSET if overwrite else SPAN_QUERY_OFFSET
    queries = (
        query_offset
        + query_keys.astype(np.uint32)[:, :, None] * MAXIMUM_SPAN
        + slots
    ).reshape(batch_size, -1)
    labels = values[
        np.arange(batch_size)[:, None],
        versions - 1,
        query_order,
    ].reshape(batch_size, -1)
    tokens = np.concatenate((mappings, queries), axis=1)
    return tokens.astype(np.uint32, copy=False), labels.astype(np.uint8)


def generate_batch(
    generator: np.random.Generator,
    batch_size: int,
    condition: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    family = str(condition["family"])
    if family == "pointer_chase":
        return _pointer_batch(generator, batch_size, condition)
    if family == "span_recall":
        return _span_batch(
            generator,
            batch_size,
            condition,
            overwrite=False,
        )
    if family == "overwrite_recall":
        return _span_batch(
            generator,
            batch_size,
            condition,
            overwrite=True,
        )
    raise ValueError(f"unknown condition family: {family}")


def _prepare_in_staging(
    output: Path,
    *,
    steps: int,
    batch_size: int,
    eval_examples: int,
    stream_seed: int,
    progress_every: int,
) -> Path:
    generator = np.random.default_rng(stream_seed)
    schedule = balanced_schedule(generator, len(CONDITIONS), steps)
    lengths = np.asarray(
        [int(CONDITIONS[int(index)]["sequence_length"]) for index in schedule],
        dtype=np.uint64,
    )
    train_offsets = np.empty(steps + 1, dtype=np.uint64)
    train_offsets[0] = 0
    np.cumsum(lengths * batch_size, out=train_offsets[1:])

    train_tokens_path = output / "train_tokens.uint32.bin"
    train_labels_path = output / "train_labels.uint8.bin"
    train_conditions_path = output / "train_condition_ids.uint16.bin"
    train_offsets_path = output / "train_token_offsets.uint64.bin"
    train_tokens = np.memmap(
        train_tokens_path,
        mode="w+",
        dtype=np.uint32,
        shape=(int(train_offsets[-1]),),
    )
    train_labels = np.memmap(
        train_labels_path,
        mode="w+",
        dtype=np.uint8,
        shape=(steps, batch_size, QUERIES),
    )
    for step, condition_index in enumerate(schedule):
        condition = CONDITIONS[int(condition_index)]
        tokens, labels = generate_batch(generator, batch_size, condition)
        start = int(train_offsets[step])
        stop = int(train_offsets[step + 1])
        train_tokens[start:stop] = tokens.reshape(-1)
        train_labels[step] = labels
        if progress_every > 0 and (step + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "training_stream",
                        "completed_steps": step + 1,
                        "total_steps": steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    train_tokens.flush()
    train_labels.flush()
    del train_tokens, train_labels
    schedule.tofile(train_conditions_path)
    train_offsets.tofile(train_offsets_path)

    records: dict[str, dict[str, Any]] = {
        "train_tokens": record(
            output,
            train_tokens_path,
            "uint32",
            (int(train_offsets[-1]),),
        ),
        "train_labels": record(
            output,
            train_labels_path,
            "uint8",
            (steps, batch_size, QUERIES),
        ),
        "train_condition_ids": record(
            output,
            train_conditions_path,
            "uint16",
            (steps,),
        ),
        "train_token_offsets": record(
            output,
            train_offsets_path,
            "uint64",
            (steps + 1,),
        ),
    }

    for split, seed_offset in (
        ("validation", 10_000_000),
        ("test", 20_000_000),
    ):
        split_generator = np.random.default_rng(stream_seed + seed_offset)
        split_offsets = np.empty(len(CONDITIONS) + 1, dtype=np.uint64)
        split_offsets[0] = 0
        split_lengths = np.asarray(
            [
                int(condition["sequence_length"]) * eval_examples
                for condition in CONDITIONS
            ],
            dtype=np.uint64,
        )
        np.cumsum(split_lengths, out=split_offsets[1:])
        token_path = output / f"{split}_tokens.uint32.bin"
        label_path = output / f"{split}_labels.uint8.bin"
        offset_path = output / f"{split}_token_offsets.uint64.bin"
        split_tokens = np.memmap(
            token_path,
            mode="w+",
            dtype=np.uint32,
            shape=(int(split_offsets[-1]),),
        )
        split_labels = np.memmap(
            label_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(CONDITIONS), eval_examples, QUERIES),
        )
        for index, condition in enumerate(CONDITIONS):
            tokens, labels = generate_batch(
                split_generator,
                eval_examples,
                condition,
            )
            start = int(split_offsets[index])
            stop = int(split_offsets[index + 1])
            split_tokens[start:stop] = tokens.reshape(-1)
            split_labels[index] = labels
        split_tokens.flush()
        split_labels.flush()
        del split_tokens, split_labels
        split_offsets.tofile(offset_path)
        records[f"{split}_tokens"] = record(
            output,
            token_path,
            "uint32",
            (int(split_offsets[-1]),),
        )
        records[f"{split}_labels"] = record(
            output,
            label_path,
            "uint8",
            (len(CONDITIONS), eval_examples, QUERIES),
        )
        records[f"{split}_token_offsets"] = record(
            output,
            offset_path,
            "uint64",
            (len(CONDITIONS) + 1,),
        )

    manifest = {
        "format": FORMAT,
        "task": "adaptive_depth_exact_recall_suite",
        "steps": steps,
        "batch_size": batch_size,
        "eval_examples": eval_examples,
        "stream_seed": stream_seed,
        "maximum_queries": QUERIES,
        "maximum_sequence_length": max(
            int(row["sequence_length"]) for row in CONDITIONS
        ),
        "output_classes": OUTPUT_CLASSES,
        "vocab_size": VOCAB_SIZE,
        "conditions": list(CONDITIONS),
        "families": {
            "pointer_chase": {
                "definition": (
                    "follow a random permutation-valued key map for a "
                    "query-specified dependent hop count"
                ),
                "key_vocabulary": POINTER_KEYS,
                "hop_depths": list(POINTER_HOPS),
                "association_counts": [32, 64, 128, 192],
            },
            "span_recall": {
                "definition": (
                    "retrieve every symbol of random multi-token values using "
                    "distinct key-and-slot queries"
                ),
                "key_vocabulary": SPAN_KEYS,
                "value_classes": SPAN_VALUES,
                "association_counts": [16, 32],
                "span_lengths": [1, 4, 8, 16],
            },
            "overwrite_recall": {
                "definition": (
                    "retrieve the latest random value after repeated updates "
                    "to the same key-and-slot identities"
                ),
                "key_vocabulary": SPAN_KEYS,
                "value_classes": SPAN_VALUES,
                "association_counts": [16, 32],
                "versions": [1, 2, 4],
                "span_length": 4,
            },
        },
        "token_codec": {
            "pointer_mapping": {
                "offset": POINTER_MAP_OFFSET,
                "count": POINTER_MAP_TOKENS,
                "formula": "offset + source_key * pointer_keys + target_key",
            },
            "pointer_query": {
                "offset": POINTER_QUERY_OFFSET,
                "count": POINTER_QUERY_TOKENS,
                "formula": "offset + hop_index * pointer_keys + start_key",
            },
            "span_mapping": {
                "offset": SPAN_MAP_OFFSET,
                "count": SPAN_MAP_TOKENS,
                "formula": (
                    "offset + (key * maximum_span + slot) * "
                    "span_values + value"
                ),
            },
            "span_query": {
                "offset": SPAN_QUERY_OFFSET,
                "count": SPAN_QUERY_TOKENS,
                "formula": "offset + key * maximum_span + slot",
            },
            "overwrite_mapping": {
                "offset": OVERWRITE_MAP_OFFSET,
                "count": OVERWRITE_MAP_TOKENS,
                "formula": (
                    "offset + (key * maximum_span + slot) * "
                    "span_values + value"
                ),
            },
            "overwrite_query": {
                "offset": OVERWRITE_QUERY_OFFSET,
                "count": OVERWRITE_QUERY_TOKENS,
                "formula": "offset + key * maximum_span + slot",
            },
        },
        "records": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def prepare_dataset(
    output: Path,
    *,
    steps: int,
    batch_size: int,
    eval_examples: int,
    stream_seed: int,
    progress_every: int = 1_000,
) -> Path:
    output = output.resolve()
    if output.exists():
        if any(output.iterdir()):
            raise ValueError(f"output directory is not empty: {output}")
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            dir=output.parent,
        )
    )
    try:
        manifest = _prepare_in_staging(
            staging,
            steps=steps,
            batch_size=batch_size,
            eval_examples=eval_examples,
            stream_seed=stream_seed,
            progress_every=progress_every,
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / manifest.name
