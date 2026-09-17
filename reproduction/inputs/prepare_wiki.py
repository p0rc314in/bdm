#!/usr/bin/env python3
"""Download and deterministically prepare canonical WikiText-103 locally."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reproduction.inputs.wikitext103_coverage import (  # noqa: E402
    CONTEXT_LENGTH,
    DATASET_CONFIGURATION,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    EFFECTIVE_BATCH_RECORDS,
    EVALUATION_STRIDE,
    EXPECTED_TOKEN_COUNTS,
    FORMAT,
    OPTIMIZER_STEPS_PER_PASS,
    PROTOCOL_ID,
    STREAM_SEED,
    TOKENIZER_ID,
    TRAIN_RECORDS_PER_PASS,
    TRAIN_TARGETS_PER_PASS,
    VOCAB_SIZE,
    array_file_record,
    build_evaluation_index,
    build_pass_permutations,
    build_training_index,
    sha256_file,
    validate_evaluation_index,
    validate_training_index,
)


SOURCE_FILES = {
    "train": (
        f"{DATASET_CONFIGURATION}/train-00000-of-00002.parquet",
        f"{DATASET_CONFIGURATION}/train-00001-of-00002.parquet",
    ),
    "validation": (
        f"{DATASET_CONFIGURATION}/validation-00000-of-00001.parquet",
    ),
    "test": (
        f"{DATASET_CONFIGURATION}/test-00000-of-00001.parquet",
    ),
}
REMOTE_PAYLOAD_PREFIXES = ("tokens/", "indexes/", "tokenizer/")
EXPECTED_MANIFEST_SHA256 = (
    "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6"
)
EXPECTED_PAYLOAD_SHA256 = (
    "f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e"
)
EXPECTED_INVENTORY_SHA256 = (
    "efa0a0b857d184dccdecf6adafda5d646ec45bae91f01a15cd1165f9fe9ad6b3"
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(array)
    with path.open("wb") as handle:
        contiguous.tofile(handle)
    return array_file_record(path, contiguous, path.parents[1])


def download_sources(cache_root: Path) -> dict[str, list[Path]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "install huggingface-hub in the local preparation environment"
        ) from error
    resolved: dict[str, list[Path]] = {}
    for split, names in SOURCE_FILES.items():
        resolved[split] = []
        for name in names:
            path = hf_hub_download(
                DATASET_REPOSITORY,
                filename=name,
                repo_type="dataset",
                revision=DATASET_REVISION,
                cache_dir=cache_root,
            )
            resolved[split].append(Path(path))
    return resolved


def serialize_split(
    source_paths: list[Path],
    *,
    split: str,
    destination: Path,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("install pyarrow in the local preparation environment") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with destination.open("wb") as output:
        for source_path in source_paths:
            parquet_file = parquet.ParquetFile(source_path)
            if parquet_file.schema_arrow.names != ["text"]:
                raise ValueError(f"unexpected {split} parquet schema: {source_path}")
            for batch in parquet_file.iter_batches(columns=["text"], batch_size=16_384):
                for value in batch.column(0).to_pylist():
                    if not isinstance(value, str):
                        raise ValueError(f"{split} contains a non-string row")
                    encoded = value.encode("utf-8")
                    output.write(encoded)
                    if not encoded.endswith(b"\n"):
                        output.write(b"\n")
                    rows += 1
    return {
        "path": destination.name,
        "rows": rows,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "rule": (
            "dataset row order; UTF-8; append one newline only when a row "
            "does not already end in one"
        ),
    }


def tokenize_split(
    serialized: Path,
    *,
    split: str,
    destination: Path,
    encoding: Any,
) -> tuple[dict[str, Any], int]:
    text = serialized.read_text(encoding="utf-8")
    tokens_native = encoding.encode_to_numpy(text)
    del text
    if len(tokens_native) != EXPECTED_TOKEN_COUNTS[split]:
        raise ValueError(
            f"{split} produced {len(tokens_native)} tokens; "
            f"expected {EXPECTED_TOKEN_COUNTS[split]}"
        )
    if int(tokens_native.max()) >= VOCAB_SIZE or int(tokens_native.min()) < 0:
        raise ValueError(f"{split} contains a token outside the GPT-2 vocabulary")
    tokens = np.asarray(tokens_native, dtype="<u2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        tokens.tofile(handle)

    decoded_digest = hashlib.sha256()
    decoded_bytes = 0
    for start in range(0, len(tokens), 1_000_000):
        block = encoding.decode_bytes(tokens[start : start + 1_000_000].tolist())
        decoded_digest.update(block)
        decoded_bytes += len(block)
    if decoded_digest.hexdigest() != sha256_file(serialized):
        raise ValueError(f"{split} token stream does not round-trip to its serialization")
    if decoded_bytes != serialized.stat().st_size:
        raise ValueError(f"{split} decoded byte count changed")
    return (
        {
            "path": destination.relative_to(destination.parents[1]).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "dtype": tokens.dtype.str,
            "shape": [len(tokens)],
        },
        decoded_bytes,
    )


def payload_digest(root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def inventory_digest(records: dict[str, dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_once(
    source_paths: dict[str, list[Path]],
    destination: Path,
    *,
    passes: int,
    stream_seed: int,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    if passes <= 0:
        raise ValueError("passes must be positive")
    if stream_seed != STREAM_SEED:
        raise ValueError(f"canonical stream seed must be {STREAM_SEED}")
    destination.mkdir(parents=True)

    try:
        import pyarrow
        import tiktoken
    except ImportError as error:
        raise RuntimeError(
            "install pyarrow and tiktoken in the local preparation environment"
        ) from error
    encoding = tiktoken.get_encoding("gpt2")
    if encoding.n_vocab != VOCAB_SIZE:
        raise ValueError("GPT-2 vocabulary size changed")

    source_records: dict[str, list[dict[str, Any]]] = {}
    serialization: dict[str, Any] = {}
    token_streams: dict[str, Any] = {}
    decoded_bytes: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        source_records[split] = []
        copied_sources = []
        for source_path, repository_name in zip(
            source_paths[split],
            SOURCE_FILES[split],
            strict=True,
        ):
            copied = destination / "source_shards" / repository_name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, copied)
            copied_sources.append(copied)
            source_records[split].append(
                {
                    "repository_path": repository_name,
                    "path": copied.relative_to(destination).as_posix(),
                    "bytes": copied.stat().st_size,
                    "sha256": sha256_file(copied),
                }
            )
        serialized = destination / "serialized" / f"{split}.txt"
        serialization[split] = serialize_split(
            copied_sources,
            split=split,
            destination=serialized,
        )
        serialization[split]["path"] = serialized.relative_to(destination).as_posix()
        token_path = destination / "tokens" / f"{split}.uint16.bin"
        token_streams[split], decoded_bytes[split] = tokenize_split(
            serialized,
            split=split,
            destination=token_path,
            encoding=encoding,
        )

    byte_lengths = np.asarray(
        [len(encoding.decode_single_token_bytes(index)) for index in range(VOCAB_SIZE)],
        dtype="<u2",
    )
    byte_length_path = destination / "tokenizer" / "gpt2_token_byte_lengths.uint16.bin"
    byte_length_record = write_array(byte_length_path, byte_lengths)

    train_starts, train_targets = build_training_index(EXPECTED_TOKEN_COUNTS["train"])
    permutations = build_pass_permutations(
        len(train_starts),
        passes,
        stream_seed,
    )
    coverage = validate_training_index(
        train_starts,
        train_targets,
        permutations,
        token_count=EXPECTED_TOKEN_COUNTS["train"],
    )
    if coverage["records_per_pass"] != TRAIN_RECORDS_PER_PASS:
        raise ValueError("canonical training record count changed")
    if coverage["targets_per_pass"] != TRAIN_TARGETS_PER_PASS:
        raise ValueError("canonical training target count changed")

    indexes_root = destination / "indexes"
    starts_path = indexes_root / "train_starts.uint64.bin"
    targets_path = indexes_root / "train_target_counts.uint16.bin"
    starts_record = write_array(starts_path, train_starts)
    targets_record = write_array(targets_path, train_targets)
    permutation_records = []
    for pass_index, permutation in enumerate(permutations):
        path = indexes_root / f"train_pass_{pass_index:03d}_record_ids.uint32.bin"
        record = write_array(path, permutation)
        record["pass_index"] = pass_index
        record["records"] = len(permutation)
        record["scored_targets"] = int(train_targets[permutation].sum(dtype=np.uint64))
        permutation_records.append(record)

    evaluation_records: dict[str, dict[str, Any]] = {}
    evaluation_splits: dict[str, Any] = {}
    for split in ("validation", "test"):
        starts, lengths, offsets, counts = build_evaluation_index(
            EXPECTED_TOKEN_COUNTS[split]
        )
        audit = validate_evaluation_index(
            starts,
            lengths,
            offsets,
            counts,
            token_count=EXPECTED_TOKEN_COUNTS[split],
        )
        records = {}
        for name, array, suffix in (
            ("starts", starts, "uint64"),
            ("token_lengths", lengths, "uint16"),
            ("score_offsets", offsets, "uint16"),
            ("target_counts", counts, "uint16"),
        ):
            path = indexes_root / f"{split}_{name}.{suffix}.bin"
            records[name] = write_array(path, array)
        stream_path = destination / token_streams[split]["path"]
        stream = np.memmap(
            stream_path,
            mode="r",
            dtype=np.dtype(token_streams[split]["dtype"]),
            shape=tuple(token_streams[split]["shape"]),
        )
        scored_bytes = int(byte_lengths[np.asarray(stream[1:], dtype=np.uint16)].sum(dtype=np.uint64))
        if scored_bytes != decoded_bytes[split] - int(byte_lengths[int(stream[0])]):
            raise ValueError(f"{split} scored byte accounting changed")
        evaluation_records[split] = records
        evaluation_splits[split] = {
            **audit,
            "scored_bytes": scored_bytes,
            "first_token_context_only": True,
        }

    remote_paths = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
        and path.relative_to(destination).as_posix().startswith(REMOTE_PAYLOAD_PREFIXES)
    )
    remote_sha = payload_digest(destination, remote_paths)
    warmup_steps = round(passes * OPTIMIZER_STEPS_PER_PASS * 0.025)
    checkpoint_steps = sorted(
        {
            (OPTIMIZER_STEPS_PER_PASS + 1) // 2,
            *(
                pass_index * OPTIMIZER_STEPS_PER_PASS
                for pass_index in range(1, passes + 1)
            ),
        }
    )
    manifest = {
        "schema_version": 1,
        "format": FORMAT,
        "protocol_id": PROTOCOL_ID,
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "configuration": DATASET_CONFIGURATION,
            "revision": DATASET_REVISION,
            "official_split_roles": ["train", "validation", "test"],
            "source_files": source_records,
        },
        "serialization": {
            "encoding": "UTF-8",
            "splits": serialization,
        },
        "tokenizer": {
            "id": TOKENIZER_ID,
            "encoding": "gpt2",
            "vocabulary_size": VOCAB_SIZE,
            "package": "tiktoken",
            "package_version": metadata.version("tiktoken"),
            "byte_lengths": byte_length_record,
        },
        "token_streams": token_streams,
        "training": {
            "objective": "causal_next_token_prediction",
            "context_length": CONTEXT_LENGTH,
            "record_tokens": CONTEXT_LENGTH + 1,
            "records_per_pass": TRAIN_RECORDS_PER_PASS,
            "scored_targets_per_pass": TRAIN_TARGETS_PER_PASS,
            "passes": passes,
            "total_target_presentations": TRAIN_TARGETS_PER_PASS * passes,
            "effective_batch_records": EFFECTIVE_BATCH_RECORDS,
            "optimizer_steps_per_pass": OPTIMIZER_STEPS_PER_PASS,
            "total_optimizer_steps": OPTIMIZER_STEPS_PER_PASS * passes,
            "warmup_steps_at_2_5_percent": warmup_steps,
            "stream_seed": stream_seed,
            "permutation_generator": "numpy.PCG64; sequential permutation per pass",
            "replacement_sampling": False,
            "reset_state_at_record_boundary": True,
            "final_partial_record_retained": True,
            "padding_excluded_from_loss": True,
            "index": {
                "starts": starts_record,
                "target_counts": targets_record,
            },
            "permutations": permutation_records,
            "checkpoint_steps": checkpoint_steps,
            "coverage_audit": coverage,
        },
        "evaluation": {
            "context_length": CONTEXT_LENGTH,
            "stride": EVALUATION_STRIDE,
            "score_each_transition_once": True,
            "reset_state_at_window_boundary": True,
            "indexes": evaluation_records,
            "splits": evaluation_splits,
        },
        "remote_payload": {
            "files": remote_paths,
            "sha256": remote_sha,
        },
        "build_environment": {
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "tiktoken": metadata.version("tiktoken"),
        },
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/data/wikitext103_gpt2_causal_t2048_coverage_v1",
    )
    parser.add_argument(
        "--download-cache",
        type=Path,
        default=ROOT / "runs/data/.cache/huggingface-wikitext103",
    )
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--stream-seed", type=int, default=STREAM_SEED)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    cache = (
        args.download_cache
        if args.download_cache.is_absolute()
        else ROOT / args.download_cache
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refusing to replace existing canonical payload: {output}"
        )
    cache.mkdir(parents=True, exist_ok=True)
    sources = download_sources(cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    first = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-a-", dir=output.parent))
    second = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-b-", dir=output.parent))
    first.rmdir()
    second.rmdir()
    promoted = False
    try:
        first_manifest = build_once(
            sources,
            first,
            passes=args.passes,
            stream_seed=args.stream_seed,
        )
        second_manifest = build_once(
            sources,
            second,
            passes=args.passes,
            stream_seed=args.stream_seed,
        )
        first_inventory = inventory(first)
        second_inventory = inventory(second)
        if first_inventory != second_inventory:
            changed = sorted(set(first_inventory) | set(second_inventory))
            changed = [
                name
                for name in changed
                if first_inventory.get(name) != second_inventory.get(name)
            ]
            raise RuntimeError(f"deterministic builds differ: {changed}")
        if first_manifest != second_manifest:
            raise RuntimeError("deterministic manifests differ")
        manifest_sha = sha256_file(first / "manifest.json")
        if manifest_sha != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError(
                f"canonical manifest SHA-256 is {manifest_sha}; "
                f"expected {EXPECTED_MANIFEST_SHA256}"
            )
        if first_manifest["remote_payload"]["sha256"] != EXPECTED_PAYLOAD_SHA256:
            raise RuntimeError("canonical training payload SHA-256 changed")
        tree_sha = inventory_digest(first_inventory)
        if tree_sha != EXPECTED_INVENTORY_SHA256:
            raise RuntimeError("canonical double-build inventory SHA-256 changed")
        os.replace(first, output)
        promoted = True
        attestation = {
            "schema_version": 1,
            "status": "byte_identical_double_build",
            "builds": 2,
            "inventory_sha256": tree_sha,
            "manifest_sha256": manifest_sha,
            "remote_payload_sha256": first_manifest["remote_payload"]["sha256"],
            "files": len(first_inventory),
            "protocol_id": PROTOCOL_ID,
            "passes": args.passes,
            "stream_seed": args.stream_seed,
        }
        atomic_json(output / "DOUBLE_BUILD.json", attestation)
        print(json.dumps(attestation, indent=2, sort_keys=True))
    finally:
        if not promoted and first.exists():
            shutil.rmtree(first)
        if second.exists():
            shutil.rmtree(second)


if __name__ == "__main__":
    main()
