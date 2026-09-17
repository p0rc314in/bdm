from __future__ import annotations

import argparse

from collections import defaultdict

import json

import os

from pathlib import Path

import time

from typing import Any

import numpy as np

from .io import atomic_json, sha256_file

from .spec import (
    CANONICAL_SDM_COMMIT,
    CHECKPOINT_EXPOSURES_MILLIONS,
    EVALUATION_SDM_COMMIT,
    EVALUATION_REVISION,
    EVALUATOR_REVISION,
    SPEC,
    validate_arm,
)

def load_record(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = root / str(record["path"])
    if sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"packed evaluation record changed: {path}")
    shape = tuple(int(value) for value in record["shape"])
    dtype = np.dtype(str(record["dtype"]))
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        values = np.memmap(path, mode="r", dtype=dtype, shape=shape)
    if values.shape != shape or values.dtype != dtype:
        raise ValueError(f"packed evaluation record contract changed: {path}")
    return values

class PackedEvaluation:
    def __init__(self, root: Path, expected_manifest_sha256: str) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        digest = sha256_file(manifest_path)
        if digest != expected_manifest_sha256:
            raise ValueError(
                f"checkpoint evaluation manifest changed: {digest} != "
                f"{expected_manifest_sha256}"
            )
        self.manifest_sha256 = digest
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != (
            "babylm2026_strict_gpt2_checkpoint_evaluation_v1"
        ):
            raise ValueError("unsupported checkpoint evaluation input")
        self.sections: dict[str, dict[str, np.ndarray]] = {}
        for name in ("fast_zero_shot", "reading", "aoa"):
            self.sections[name] = {
                key: load_record(root, record)
                for key, record in self.manifest[name]["records"].items()
            }
        self.eot_token = int(self.manifest["tokenizer"]["eot_token"])

    def sequence(self, section: str, index: int) -> np.ndarray:
        arrays = self.sections[section]
        offsets = arrays["offsets"]
        return arrays["tokens"][int(offsets[index]) : int(offsets[index + 1])]
