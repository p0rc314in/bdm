"""Hash-verified deterministic BabyLM causal-language stream."""

from __future__ import annotations

from collections.abc import Iterator
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file
from .spec import DERIVED_FORMAT, SPEC


def _dtype(name: str) -> np.dtype[Any]:
    aliases = {
        "uint8": np.dtype(np.uint8),
        "uint16": np.dtype("<u2"),
        "uint32": np.dtype("<u4"),
        "uint64": np.dtype("<u8"),
    }
    return aliases.get(name, np.dtype(name))


def _records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256", "dtype", "shape"}.issubset(value):
            yield value
        for child in value.values():
            yield from _records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records(child)


class BabyLMData:
    def __init__(self, manifest_path: Path, immutable_root: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        immutable_root = immutable_root.resolve()
        if immutable_root != self.root and immutable_root not in self.root.parents:
            raise ValueError("manifest must live below the immutable input root")
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("format") != DERIVED_FORMAT:
            raise ValueError("unexpected BabyLM derived manifest format")

        self.hashes: dict[str, str] = {}
        for record in _records(self.manifest):
            relative = str(record["path"])
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise ValueError(f"record escapes payload root: {relative}")
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"record size changed: {relative}")
            digest = sha256_file(path)
            if digest != record["sha256"]:
                raise ValueError(f"record SHA-256 changed: {relative}")
            self.hashes[relative] = digest

        corpus = self.manifest["corpus"]
        training = corpus["training"]
        tokenizer = self.manifest["tokenizer"]
        self.sequence_length = int(training["sequence_length"])
        self.vocab_size = int(tokenizer["vocab_size"])
        self.eot_token = int(tokenizer["eot_token"])
        if self.sequence_length != SPEC.context_length:
            raise ValueError("sequence length changed")
        if self.vocab_size != SPEC.vocab_size:
            raise ValueError("vocabulary changed")
        if int(training["epochs"]) != SPEC.passes:
            raise ValueError("training pass count changed")

        self.train_tokens = self._binary(corpus["train_tokens"])
        self.train_starts = self._array(training["starts"])
        expected = int(training["complete_records_per_epoch"]) * SPEC.passes
        if len(self.train_starts) != expected:
            raise ValueError("ten-pass training order has the wrong extent")

    def _path(self, record: dict[str, Any]) -> Path:
        return self.root / str(record["path"])

    def _binary(self, record: dict[str, Any]) -> np.memmap:
        return np.memmap(
            self._path(record),
            mode="r",
            dtype=_dtype(str(record["dtype"])),
            shape=tuple(int(value) for value in record["shape"]),
        )

    def _array(self, record: dict[str, Any]) -> np.ndarray:
        values = np.load(self._path(record), mmap_mode="r", allow_pickle=False)
        if values.shape != tuple(int(value) for value in record["shape"]):
            raise ValueError(f"record shape changed: {record['path']}")
        if values.dtype != _dtype(str(record["dtype"])):
            raise ValueError(f"record dtype changed: {record['path']}")
        return values

    def training_steps(self, batch_size: int = SPEC.batch_size) -> int:
        return math.ceil(len(self.train_starts) / batch_size)

    def training_tokens(self) -> int:
        return len(self.train_starts) * self.sequence_length

    def train_batch(self, step_index: int, batch_size: int) -> np.ndarray:
        first = step_index * batch_size
        starts = self.train_starts[first : first + batch_size]
        values = np.empty((len(starts), self.sequence_length + 1), dtype=np.uint16)
        for row, raw_start in enumerate(starts):
            start = int(raw_start)
            values[row] = self.train_tokens[start : start + self.sequence_length + 1]
        return values


class TokenSplit:
    """One immutable variable-length classification split."""

    def __init__(
        self,
        tokens: np.memmap,
        offsets: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.tokens = tokens
        self.offsets = offsets
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def sequence(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return np.asarray(self.tokens[start:stop])


class BabyLMEvaluationData:
    """The pinned official tasks, packed without changing their scoring rules.

    The canonical BabyLM input copies only these checksum-addressed evaluation
    records into its immutable payload.
    """

    def __init__(self, manifest_path: Path, immutable_root: Path) -> None:
        canonical = BabyLMData(manifest_path, immutable_root)
        self.root = canonical.root / "evaluation" / "packed"
        source_manifest_path = canonical.root / "legacy_packed_manifest.json"
        self.manifest = json.loads(source_manifest_path.read_text())
        self.records: dict[str, dict[str, Any]] = {}
        for record in _records(self.manifest):
            relative = str(record["path"])
            if not relative.startswith(("zero_shot/", "finetune/")):
                continue
            prior = self.records.get(relative)
            if prior is not None and prior != record:
                raise ValueError(f"conflicting evaluation record: {relative}")
            self.records[relative] = record
        for relative, record in sorted(self.records.items()):
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise ValueError(f"evaluation record escapes payload: {relative}")
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"evaluation record size changed: {relative}")
            if sha256_file(path) != str(record["sha256"]):
                raise ValueError(f"evaluation record hash changed: {relative}")

        zero = self.manifest["zero_shot"]
        self.zero_tokens = self._binary(zero["records"]["tokens"])
        self.zero_arrays = {
            key: self._array(zero["records"][key])
            for key in (
                "candidate_offsets",
                "score_starts",
                "example_offsets",
                "labels",
                "task_ids",
                "subdomain_ids",
                "length_normalized",
            )
        }
        self.zero_tasks = tuple(str(value) for value in zero["tasks"])
        self.zero_subdomains = tuple(str(value) for value in zero["subdomains"])
        self.eot_token = int(self.manifest["tokenizer"]["eot_token"])
        self._finetune_cache: dict[tuple[str, str], TokenSplit] = {}

    def _path(self, record: dict[str, Any]) -> Path:
        return self.root / str(record["path"])

    def _binary(self, record: dict[str, Any]) -> np.memmap:
        return np.memmap(
            self._path(record),
            mode="r",
            dtype=_dtype(str(record["dtype"])),
            shape=tuple(int(value) for value in record["shape"]),
        )

    def _array(self, record: dict[str, Any]) -> np.ndarray:
        values = np.load(self._path(record), mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(int(value) for value in record["shape"])
        expected_dtype = _dtype(str(record["dtype"]))
        if values.shape != expected_shape or values.dtype != expected_dtype:
            raise ValueError(f"evaluation array changed: {record['path']}")
        return values

    def zero_candidate(self, index: int) -> np.ndarray:
        offsets = self.zero_arrays["candidate_offsets"]
        return np.asarray(
            self.zero_tokens[int(offsets[index]) : int(offsets[index + 1])]
        )

    def finetune_split(self, task: str, split: str) -> TokenSplit:
        key = (task, split)
        if key not in self._finetune_cache:
            record = self.manifest["finetune"]["tasks"][task]["records"][split]
            self._finetune_cache[key] = TokenSplit(
                self._binary(record["tokens"]),
                self._array(record["offsets"]),
                self._array(record["labels"]),
            )
        return self._finetune_cache[key]

    def finetune_order(self, task: str) -> np.ndarray:
        record = self.manifest["finetune"]["tasks"][task]["records"][
            "train_order"
        ]
        return self._array(record)
