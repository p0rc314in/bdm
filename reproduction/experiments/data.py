"""Readers for the existing immutable W&B benchmark artifacts. No data generation."""

import hashlib
import json
from pathlib import Path

import numpy as np


DATASETS = {
    "wiki": ("wiki",
             "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6"),
    "recall": ("recall",
               "b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800"),
}
PROTOCOLS = {"wiki": "wikitext103-gpt2-causal-t2048-coverage-v1", "recall": "adaptive-recall-seed102337-v1"}
STEPS = {"wiki": 21603, "recall": 30000}
VALIDATION_STEPS = (3601, 7201, 14402, 21603)


def sha256(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class Dataset:
    def __init__(self, root, benchmark):
        self.root, self.benchmark = Path(root), benchmark
        if sha256(self.root / "manifest.json") != DATASETS[benchmark][1]:
            raise ValueError("Dataset manifest differs from the adopted benchmark")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if benchmark == "wiki":
            self.streams = {k: self.array(v) for k, v in m["token_streams"].items()}
            self.starts = self.array(m["training"]["index"]["starts"])
            self.counts = self.array(m["training"]["index"]["target_counts"])
            self.permutations = [self.array(v) for v in m["training"]["permutations"]]
            self.byte_lengths = self.array(m["tokenizer"]["byte_lengths"])
            self.indexes = {split: {k: self.array(v) for k, v in rows.items()}
                            for split, rows in m["evaluation"]["indexes"].items()}
            assert m["training"]["total_optimizer_steps"] == STEPS[benchmark]
            assert m["training"]["checkpoint_steps"] == list(VALIDATION_STEPS)
        else:
            self.arrays = {k: self.array(v) for k, v in m["records"].items()}
            self.conditions = m["conditions"]
            assert (m["steps"], m["batch_size"], m["eval_examples"], len(self.conditions)) == (30000, 32, 2048, 30)

    def array(self, record):
        path = (self.root / record["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Dataset path escapes its directory")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Dataset checksum failed: {record['path']}")
        return np.memmap(path, mode="r", dtype=record["dtype"], shape=tuple(record["shape"]))

    def train_batch(self, step):
        if not 0 <= step < STEPS[self.benchmark]:
            raise ValueError("Step outside benchmark schedule")
        if self.benchmark == "wiki":
            epoch, batch = divmod(step, 7201)
            ids = self.permutations[epoch][batch * 8:(batch + 1) * 8]
            tokens = np.zeros((8, 2049), dtype=np.int64)
            labels = np.full((8, 2048), -100, dtype=np.int64)
            for i, record in enumerate(ids):
                start, count = int(self.starts[record]), int(self.counts[record])
                tokens[i, :count + 1] = self.streams["train"][start:start + count + 1]
                labels[i, :count] = tokens[i, 1:count + 1]
            return tokens[:, :-1].copy(), labels
        a = self.arrays
        condition = self.conditions[int(a["train_condition_ids"][step])]
        start, end = a["train_token_offsets"][step:step + 2]
        return (np.array(a["train_tokens"][start:end], dtype=np.int64).reshape(32, condition["sequence_length"]),
                np.array(a["train_labels"][step], dtype=np.int64))

    def evaluation_batches(self, split, limit_per_condition=None):
        if self.benchmark == "wiki":
            index = self.indexes[split]
            for start, length, offset, count in zip(index["starts"], index["token_lengths"],
                                                   index["score_offsets"], index["target_counts"], strict=True):
                start, length, offset, count = map(int, (start, length, offset, count))
                tokens = np.array(self.streams[split][start:start + length], dtype=np.int64)[None]
                labels = np.full((1, length - 1), -100, dtype=np.int64)
                labels[:, offset:offset + count] = tokens[:, offset + 1:offset + count + 1]
                yield tokens[:, :-1].copy(), labels, "all"
        else:
            for condition in self.conditions:
                i, length = condition["index"], condition["sequence_length"]
                start, end = self.arrays[f"{split}_token_offsets"][i:i + 2]
                tokens = self.arrays[f"{split}_tokens"][start:end].reshape(2048, length)
                labels = self.arrays[f"{split}_labels"][i]
                for begin in range(0, limit_per_condition or 2048, 8):
                    yield (np.array(tokens[begin:begin + 8], dtype=np.int64),
                           np.array(labels[begin:begin + 8], dtype=np.int64), condition["id"])
