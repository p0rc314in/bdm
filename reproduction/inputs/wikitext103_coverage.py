"""Canonical WikiText-103 coverage data identities and immutable loaders."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


PROTOCOL_ID = "wikitext103-gpt2-causal-t2048-coverage-v1"
FORMAT = "wikitext103_gpt2_causal_t2048_coverage_v1"
DATASET_REPOSITORY = "Salesforce/wikitext"
DATASET_CONFIGURATION = "wikitext-103-raw-v1"
DATASET_REVISION = "5fddba447aa4e75996922ea0d6b18b42f0a81cc4"
TOKENIZER_ID = "tiktoken:gpt2"
VOCAB_SIZE = 50_257
CONTEXT_LENGTH = 2_048
EVALUATION_STRIDE = 512
EFFECTIVE_BATCH_RECORDS = 8
STREAM_SEED = 20_260_818
EXPECTED_TOKEN_COUNTS = {
    "train": 117_980_450,
    "validation": 247_417,
    "test": 283_427,
}
TRAIN_RECORDS_PER_PASS = 57_608
TRAIN_TARGETS_PER_PASS = 117_980_449
OPTIMIZER_STEPS_PER_PASS = 7_201


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_file_record(path: Path, array: np.ndarray, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def build_training_index(
    token_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if token_count < 2:
        raise ValueError("a causal stream requires at least two tokens")
    starts = np.arange(0, token_count - 1, CONTEXT_LENGTH, dtype="<u8")
    targets = np.minimum(
        CONTEXT_LENGTH,
        token_count - 1 - starts.astype(np.int64),
    ).astype("<u2")
    return starts, targets


def build_pass_permutations(
    records: int,
    passes: int,
    stream_seed: int,
) -> list[np.ndarray]:
    if records <= 0 or passes <= 0:
        raise ValueError("records and passes must be positive")
    generator = np.random.Generator(np.random.PCG64(stream_seed))
    return [generator.permutation(records).astype("<u4") for _ in range(passes)]


def validate_training_index(
    starts: np.ndarray,
    target_counts: np.ndarray,
    permutations: list[np.ndarray],
    *,
    token_count: int,
) -> dict[str, Any]:
    expected_starts, expected_targets = build_training_index(token_count)
    if not np.array_equal(starts, expected_starts):
        raise ValueError("training record starts do not tile the corpus")
    if not np.array_equal(target_counts, expected_targets):
        raise ValueError("training target counts do not cover the corpus tail")
    records = len(starts)
    wanted_ids = np.arange(records, dtype="<u4")
    coverage = []
    for pass_index, permutation in enumerate(permutations):
        if permutation.shape != (records,):
            raise ValueError(f"pass {pass_index} has an invalid permutation shape")
        if not np.array_equal(np.sort(permutation), wanted_ids):
            raise ValueError(f"pass {pass_index} is missing or duplicating records")
        coverage.append(
            {
                "pass_index": pass_index,
                "records": records,
                "distinct_record_ids": records,
                "scored_targets": int(target_counts[permutation].sum(dtype=np.uint64)),
            }
        )
    targets_per_pass = int(target_counts.sum(dtype=np.uint64))
    return {
        "records_per_pass": records,
        "targets_per_pass": targets_per_pass,
        "passes": coverage,
        "total_target_presentations": targets_per_pass * len(permutations),
    }


def build_evaluation_index(
    token_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build stride windows that score each transition exactly once."""

    if token_count < 2:
        raise ValueError("an evaluation stream requires at least two tokens")
    starts: list[int] = []
    token_lengths: list[int] = []
    score_offsets: list[int] = []
    target_counts: list[int] = []
    target_start = 1
    while target_start < token_count:
        target_end = min(token_count, target_start + EVALUATION_STRIDE)
        window_start = max(0, target_end - (CONTEXT_LENGTH + 1))
        starts.append(window_start)
        token_lengths.append(target_end - window_start)
        score_offsets.append(target_start - (window_start + 1))
        target_counts.append(target_end - target_start)
        target_start = target_end
    return (
        np.asarray(starts, dtype="<u8"),
        np.asarray(token_lengths, dtype="<u2"),
        np.asarray(score_offsets, dtype="<u2"),
        np.asarray(target_counts, dtype="<u2"),
    )


def validate_evaluation_index(
    starts: np.ndarray,
    token_lengths: np.ndarray,
    score_offsets: np.ndarray,
    target_counts: np.ndarray,
    *,
    token_count: int,
) -> dict[str, int]:
    expected = build_evaluation_index(token_count)
    observed = (starts, token_lengths, score_offsets, target_counts)
    if any(not np.array_equal(left, right) for left, right in zip(observed, expected)):
        raise ValueError("evaluation windows do not match the canonical stride index")
    if int(target_counts.sum(dtype=np.uint64)) != token_count - 1:
        raise ValueError("evaluation index does not score every transition once")
    if int(token_lengths.max()) - 1 > CONTEXT_LENGTH:
        raise ValueError("evaluation window exceeds the context limit")
    return {
        "windows": len(starts),
        "scored_targets": int(target_counts.sum(dtype=np.uint64)),
        "maximum_input_tokens": int(token_lengths.max()) - 1,
    }


def _record_path(root: Path, record: dict[str, Any]) -> Path:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe manifest path: {record['path']!r}")
    return root / relative


def _verified_memmap(root: Path, record: dict[str, Any]) -> np.memmap:
    path = _record_path(root, record)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"size mismatch for {path}")
    digest = sha256_file(path)
    if digest != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")
    dtype = np.dtype(record["dtype"])
    shape = tuple(int(value) for value in record["shape"])
    if int(np.prod(shape, dtype=np.int64)) * dtype.itemsize != path.stat().st_size:
        raise ValueError(f"shape does not match file size for {path}")
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def validate_manifest_identity(manifest: dict[str, Any], *, passes: int | None = None) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("format") != FORMAT:
        raise ValueError("unexpected canonical WikiText manifest schema")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected WikiText protocol ID")
    dataset = manifest.get("dataset", {})
    expected_dataset = {
        "repository": DATASET_REPOSITORY,
        "configuration": DATASET_CONFIGURATION,
        "revision": DATASET_REVISION,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise ValueError(f"dataset {key} changed")
    tokenizer = manifest.get("tokenizer", {})
    if tokenizer.get("id") != TOKENIZER_ID or tokenizer.get("vocabulary_size") != VOCAB_SIZE:
        raise ValueError("tokenizer identity changed")
    training = manifest.get("training", {})
    expected_training = {
        "context_length": CONTEXT_LENGTH,
        "effective_batch_records": EFFECTIVE_BATCH_RECORDS,
        "records_per_pass": TRAIN_RECORDS_PER_PASS,
        "scored_targets_per_pass": TRAIN_TARGETS_PER_PASS,
        "optimizer_steps_per_pass": OPTIMIZER_STEPS_PER_PASS,
        "stream_seed": STREAM_SEED,
        "replacement_sampling": False,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training identity changed at {key}")
    if passes is not None and training.get("passes") != passes:
        raise ValueError("manifest pass budget disagrees with the resolved run")
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("context_length") != CONTEXT_LENGTH:
        raise ValueError("evaluation context changed")
    if evaluation.get("stride") != EVALUATION_STRIDE:
        raise ValueError("evaluation stride changed")
    for split, expected in EXPECTED_TOKEN_COUNTS.items():
        if int(manifest["token_streams"][split]["shape"][0]) != expected:
            raise ValueError(f"{split} token identity changed")


@dataclass(frozen=True)
class TrainBatch:
    tokens: np.ndarray
    target_mask: np.ndarray
    record_ids: np.ndarray
    pass_index: int
    target_count: int


@dataclass(frozen=True)
class EvaluationBatch:
    tokens: np.ndarray
    target_mask: np.ndarray
    scored_bytes: int
    target_count: int
    window_index: int


class CanonicalWikiTextData:
    """Hash-verified canonical streams, indexes, and matched permutations."""

    def __init__(self, manifest_path: Path, *, passes: int | None = None) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text())
        validate_manifest_identity(self.manifest, passes=passes)
        self.passes = int(self.manifest["training"]["passes"])
        self.streams = {
            split: _verified_memmap(self.root, record)
            for split, record in self.manifest["token_streams"].items()
        }
        training_index = self.manifest["training"]["index"]
        self.train_starts = _verified_memmap(self.root, training_index["starts"])
        self.train_target_counts = _verified_memmap(
            self.root,
            training_index["target_counts"],
        )
        self.permutations = [
            _verified_memmap(self.root, record)
            for record in self.manifest["training"]["permutations"]
        ]
        self.token_byte_lengths = _verified_memmap(
            self.root,
            self.manifest["tokenizer"]["byte_lengths"],
        )
        self.evaluation_indexes: dict[str, dict[str, np.memmap]] = {}
        for split, records in self.manifest["evaluation"]["indexes"].items():
            self.evaluation_indexes[split] = {
                key: _verified_memmap(self.root, record)
                for key, record in records.items()
            }
        coverage = validate_training_index(
            self.train_starts,
            self.train_target_counts,
            self.permutations,
            token_count=len(self.streams["train"]),
        )
        if coverage["records_per_pass"] != TRAIN_RECORDS_PER_PASS:
            raise ValueError("canonical record count changed")
        if coverage["targets_per_pass"] != TRAIN_TARGETS_PER_PASS:
            raise ValueError("canonical target count changed")
        if coverage["total_target_presentations"] != int(
            self.manifest["training"]["total_target_presentations"]
        ):
            raise ValueError("target-presentation total changed")
        self.coverage = coverage
        for split, index in self.evaluation_indexes.items():
            audit = validate_evaluation_index(
                index["starts"],
                index["token_lengths"],
                index["score_offsets"],
                index["target_counts"],
                token_count=len(self.streams[split]),
            )
            declared = self.manifest["evaluation"]["splits"][split]
            if audit["scored_targets"] != int(declared["scored_targets"]):
                raise ValueError(f"{split} evaluation target count changed")
        self._step_targets = self._build_step_targets()
        self._cumulative_targets = np.concatenate(
            (
                np.zeros(1, dtype=np.uint64),
                np.cumsum(self._step_targets, dtype=np.uint64),
            )
        )

    def _build_step_targets(self) -> np.ndarray:
        rows = []
        for permutation in self.permutations:
            batches = np.asarray(permutation).reshape(
                OPTIMIZER_STEPS_PER_PASS,
                EFFECTIVE_BATCH_RECORDS,
            )
            rows.append(
                np.asarray(self.train_target_counts)[batches]
                .sum(axis=1, dtype=np.uint64)
                .astype("<u8")
            )
        return np.concatenate(rows)

    @property
    def training_steps(self) -> int:
        return self.passes * OPTIMIZER_STEPS_PER_PASS

    def target_presentations_through_step(self, step: int) -> int:
        if step < 0 or step > self.training_steps:
            raise ValueError("step is outside the prepared schedule")
        return int(self._cumulative_targets[step])

    def train_batch(self, step_index: int) -> TrainBatch:
        if step_index < 0 or step_index >= self.training_steps:
            raise ValueError("training step is outside the prepared schedule")
        pass_index, step_in_pass = divmod(step_index, OPTIMIZER_STEPS_PER_PASS)
        begin = step_in_pass * EFFECTIVE_BATCH_RECORDS
        record_ids = np.asarray(
            self.permutations[pass_index][begin : begin + EFFECTIVE_BATCH_RECORDS],
            dtype=np.uint32,
        )
        tokens = np.zeros(
            (EFFECTIVE_BATCH_RECORDS, CONTEXT_LENGTH + 1),
            dtype=np.uint16,
        )
        mask = np.zeros(
            (EFFECTIVE_BATCH_RECORDS, CONTEXT_LENGTH),
            dtype=np.bool_,
        )
        target_total = 0
        stream = self.streams["train"]
        for row, record_id in enumerate(record_ids):
            start = int(self.train_starts[int(record_id)])
            count = int(self.train_target_counts[int(record_id)])
            tokens[row, : count + 1] = stream[start : start + count + 1]
            mask[row, :count] = True
            target_total += count
        if target_total != int(self._step_targets[step_index]):
            raise ValueError("training batch target count disagrees with its index")
        return TrainBatch(
            tokens=tokens,
            target_mask=mask,
            record_ids=record_ids,
            pass_index=pass_index,
            target_count=target_total,
        )

    def evaluation_batches(self, split: str) -> Iterator[EvaluationBatch]:
        if split not in self.evaluation_indexes:
            raise ValueError(f"split {split!r} has no evaluation index")
        stream = self.streams[split]
        index = self.evaluation_indexes[split]
        for window in range(len(index["starts"])):
            start = int(index["starts"][window])
            token_length = int(index["token_lengths"][window])
            offset = int(index["score_offsets"][window])
            count = int(index["target_counts"][window])
            values = np.asarray(stream[start : start + token_length], dtype=np.uint16)
            mask = np.zeros(token_length - 1, dtype=np.bool_)
            mask[offset : offset + count] = True
            targets = values[1:][mask]
            byte_count = int(
                np.asarray(self.token_byte_lengths)[targets].sum(dtype=np.uint64)
            )
            yield EvaluationBatch(
                tokens=values[None, :],
                target_mask=mask[None, :],
                scored_bytes=byte_count,
                target_count=count,
                window_index=window,
            )

    def coverage_report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stream_seed": STREAM_SEED,
            "replacement_sampling": False,
            "records_per_pass": TRAIN_RECORDS_PER_PASS,
            "scored_targets_per_pass": TRAIN_TARGETS_PER_PASS,
            "passes": self.coverage["passes"],
            "total_target_presentations": self.coverage[
                "total_target_presentations"
            ],
            "optimizer_steps_per_pass": OPTIMIZER_STEPS_PER_PASS,
            "total_optimizer_steps": self.training_steps,
            "manifest_sha256": sha256_file(self.manifest_path),
            "payload_sha256": self.manifest["remote_payload"]["sha256"],
        }


__all__ = [
    "CanonicalWikiTextData",
    "CONTEXT_LENGTH",
    "DATASET_CONFIGURATION",
    "DATASET_REPOSITORY",
    "DATASET_REVISION",
    "EFFECTIVE_BATCH_RECORDS",
    "EVALUATION_STRIDE",
    "EXPECTED_TOKEN_COUNTS",
    "FORMAT",
    "OPTIMIZER_STEPS_PER_PASS",
    "PROTOCOL_ID",
    "STREAM_SEED",
    "TOKENIZER_ID",
    "TRAIN_RECORDS_PER_PASS",
    "TRAIN_TARGETS_PER_PASS",
    "VOCAB_SIZE",
    "array_file_record",
    "build_evaluation_index",
    "build_pass_permutations",
    "build_training_index",
    "canonical_json_sha256",
    "sha256_file",
    "validate_evaluation_index",
    "validate_manifest_identity",
    "validate_training_index",
]
