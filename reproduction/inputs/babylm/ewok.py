from __future__ import annotations

import json

from pathlib import Path

from typing import Any

import numpy as np

from .io import sha256_file

from .spec import EVALUATOR_REVISION, EWOK_DATASET, EWOK_PACKED_FORMAT, EWOK_REVISION

def dtype(name: str) -> np.dtype[Any]:
    aliases = {
        "uint8": np.dtype(np.uint8),
        "uint16": np.dtype("<u2"),
        "uint32": np.dtype("<u4"),
        "uint64": np.dtype("<u8"),
    }
    return aliases.get(name, np.dtype(name))

class FullEWoKData:
    def __init__(self, manifest_path: Path) -> None:
        self.root = manifest_path.resolve().parent
        self.manifest = json.loads(manifest_path.read_text())
        expected_path = Path(__file__).resolve().parents[1] / "identities/babylm-ewok-records.json"
        declared = json.loads(expected_path.read_text())
        if {key: self.manifest.get(key) for key in declared} != declared:
            raise ValueError("full EWoK differs from the recorded input identities")
        expected = {
            "format": EWOK_PACKED_FORMAT,
            "dataset": EWOK_DATASET,
            "dataset_revision": EWOK_REVISION,
            "official_evaluator_revision": EVALUATOR_REVISION,
        }
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise ValueError(f"full EWoK identity changed at {key}")
        self.records = self.manifest["records"]
        for record in self.records.values():
            path = (self.root / str(record["path"])).resolve()
            if self.root not in path.parents:
                raise ValueError("full EWoK record escapes payload root")
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"full EWoK record size changed: {path.name}")
            if sha256_file(path) != str(record["sha256"]):
                raise ValueError(f"full EWoK record hash changed: {path.name}")
        self.zero_tokens = self.binary(self.records["tokens"])
        self.zero_arrays = {
            name: self.array(self.records[name])
            for name in (
                "candidate_offsets",
                "score_starts",
                "example_offsets",
                "labels",
                "task_ids",
                "subdomain_ids",
                "length_normalized",
            )
        }
        self.zero_tasks = ("ewok",)
        self.zero_subdomains = tuple(self.manifest["subdomains"])
        self.eot_token = int(self.manifest["tokenizer"]["eot"])

    def path(self, record: dict[str, Any]) -> Path:
        return self.root / str(record["path"])

    def binary(self, record: dict[str, Any]) -> np.memmap:
        return np.memmap(
            self.path(record),
            mode="r",
            dtype=dtype(str(record["dtype"])),
            shape=tuple(int(value) for value in record["shape"]),
        )

    def array(self, record: dict[str, Any]) -> np.ndarray:
        values = np.load(self.path(record), mmap_mode="r", allow_pickle=False)
        if values.shape != tuple(record["shape"]) or values.dtype != dtype(record["dtype"]):
            raise ValueError(f"full EWoK array changed: {record['path']}")
        return values

    def zero_candidate(self, index: int) -> np.ndarray:
        offsets = self.zero_arrays["candidate_offsets"]
        return np.asarray(
            self.zero_tokens[int(offsets[index]) : int(offsets[index + 1])]
        )
