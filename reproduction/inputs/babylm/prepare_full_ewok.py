#!/usr/bin/env python3
"""Build the pinned full-EWoK terminal-evaluation inputs locally."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Any

from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import tiktoken

from reproduction.inputs.babylm.io import atomic_json, sha256_file
from reproduction.inputs.babylm.spec import (
    EVALUATOR_REVISION,
    EWOK_DATASET,
    EWOK_PACKED_FORMAT,
    EWOK_REVISION,
)


EXPECTED_DOMAINS = 11
EXPECTED_EXAMPLES = 7_618


def write_npy(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)


def record(root: Path, path: Path, dtype: str, shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": dtype,
        "shape": list(shape),
    }


def candidate_tokens(
    encoder: tiktoken.Encoding, sentence: str, completion: str
) -> tuple[list[int], int]:
    raw = sentence.encode("utf-8")
    completion_start = len(raw) - len(completion.encode("utf-8"))
    encoded = encoder.encode_ordinary(sentence)
    pieces = [encoder.decode_single_token_bytes(token) for token in encoded]
    if b"".join(pieces) != raw:
        raise ValueError("GPT-2 token byte reconstruction failed")
    cursor = 0
    score_start = len(encoded)
    for index, piece in enumerate(pieces):
        cursor += len(piece)
        if cursor > completion_start:
            score_start = index
            break
    score_start = max(score_start, 1)
    if score_start >= len(encoded):
        raise ValueError("EWoK candidate has no scoreable completion tokens")
    return encoded, score_start


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def deterministic_tar(source: Path, destination: Path) -> None:
    with tarfile.open(destination, "w", format=tarfile.GNU_FORMAT) as archive:
        for path in sorted(value for value in source.rglob("*") if value.is_file()):
            data = path.read_bytes()
            info = tarfile.TarInfo(f"full-ewok/{path.relative_to(source).as_posix()}")
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise ValueError("HF_TOKEN is required for gated EWoK access")
    if args.output.exists() or args.archive.exists():
        raise FileExistsError("output and archive paths must be new")

    args.nltk_data.mkdir(parents=True, exist_ok=True)
    if not nltk.download(
        "punkt_tab", download_dir=str(args.nltk_data), quiet=True, raise_on_error=True
    ):
        raise RuntimeError("could not prepare pinned NLTK tokenizer data")
    nltk.data.path.insert(0, str(args.nltk_data))

    vocab = {line.strip() for line in args.vocab.read_text().splitlines()}
    dataset = load_dataset(
        EWOK_DATASET,
        revision=EWOK_REVISION,
        split="test",
        token=token,
    )
    filtered: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in dataset:
        row = dict(item)
        if any(
            word not in vocab
            for key in ("Context1", "Context2", "Target1", "Target2")
            for word in word_tokenize(str(row[key]).lower())
        ):
            continue
        domain = str(row["Domain"])
        filtered[domain].append(row)
        swapped = dict(row)
        swapped["Context1"], swapped["Context2"] = row["Context2"], row["Context1"]
        swapped["Target1"], swapped["Target2"] = row["Target2"], row["Target1"]
        filtered[domain].append(swapped)

    domains = sorted(filtered)
    examples = sum(len(filtered[domain]) for domain in domains)
    if len(domains) != EXPECTED_DOMAINS or examples != EXPECTED_EXAMPLES:
        raise ValueError(
            f"official EWoK filter changed: domains={len(domains)}, examples={examples}"
        )

    args.output.mkdir(parents=True)
    encoder = tiktoken.get_encoding("gpt2")
    token_path = args.output / "tokens.bin"
    candidate_offsets = [0]
    score_starts: list[int] = []
    example_offsets = [0]
    domain_ids: list[int] = []
    token_count = 0
    with token_path.open("wb") as handle:
        for domain_index, domain in enumerate(domains):
            for row in filtered[domain]:
                target = str(row["Target1"])
                completion = " " + target
                for context in (str(row["Context1"]), str(row["Context2"])):
                    sentence = f"{context} {target}"
                    tokens, score_start = candidate_tokens(
                        encoder, sentence, completion
                    )
                    np.asarray(tokens, dtype="<u2").tofile(handle)
                    token_count += len(tokens)
                    candidate_offsets.append(token_count)
                    score_starts.append(score_start)
                example_offsets.append(len(score_starts))
                domain_ids.append(domain_index)

    arrays = {
        "candidate_offsets": np.asarray(candidate_offsets, dtype="<u8"),
        "score_starts": np.asarray(score_starts, dtype="<u4"),
        "example_offsets": np.asarray(example_offsets, dtype="<u4"),
        "labels": np.zeros(examples, dtype=np.uint8),
        "task_ids": np.zeros(examples, dtype=np.uint8),
        "subdomain_ids": np.asarray(domain_ids, dtype="<u2"),
        "length_normalized": np.zeros(examples, dtype=np.uint8),
    }
    records = {
        "tokens": record(args.output, token_path, "uint16", (token_count,))
    }
    for name, values in arrays.items():
        path = args.output / f"{name}.npy"
        write_npy(path, values)
        records[name] = record(args.output, path, str(values.dtype), values.shape)

    source_files = []
    for cache in dataset.cache_files:
        path = Path(str(cache["filename"]))
        source_files.append(
            {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "format": EWOK_PACKED_FORMAT,
        "dataset": EWOK_DATASET,
        "dataset_revision": EWOK_REVISION,
        "dataset_fingerprint": dataset._fingerprint,
        "official_evaluator_revision": EVALUATOR_REVISION,
        "official_filter": "evaluation_pipeline/ewok/dl_and_filter.py",
        "vocab_sha256": sha256_file(args.vocab),
        "nltk_version": nltk.__version__,
        "nltk_data_tree_sha256": tree_hash(args.nltk_data),
        "source_files": source_files,
        "tokenizer": {"implementation": "tiktoken", "encoding": "gpt2", "eot": 50_256},
        "tasks": ["ewok"],
        "subdomains": domains,
        "examples": examples,
        "candidates": len(score_starts),
        "gpt2_tokens": token_count,
        "records": records,
    }
    atomic_json(args.output / "manifest.json", manifest)
    deterministic_tar(args.output, args.archive)
    print(
        json.dumps(
            {
                "marker": "COMPLETE",
                "examples": examples,
                "candidates": len(score_starts),
                "payload_tree_sha256": tree_hash(args.output),
                "archive_sha256": sha256_file(args.archive),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
