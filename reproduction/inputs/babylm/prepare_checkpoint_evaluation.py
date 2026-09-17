#!/usr/bin/env python3
"""Pack the pinned BabyLM fast-checkpoint and human-likeness inputs.

The source task semantics follow ``babylm-org/babylm-eval`` revision
``6f825c291e2c4c78ad33b1935fd64d45f52642dc``.  Packing is a lossless local
tokenization step for the campaign's fixed GPT-2 tokenizer; scoring remains
the upstream task scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
from typing import Any, Iterable

import numpy as np
import tiktoken

from reproduction.inputs.babylm.io import atomic_json, sha256_file
from reproduction.inputs.babylm.spec import EVALUATION_REVISION, EVALUATOR_REVISION


def source(path: Path, identifier: str) -> dict[str, Any]:
    return {
        "identifier": identifier,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def record(root: Path, path: Path, dtype: str, shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": dtype,
        "shape": list(shape),
    }


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def encode_candidate(
    encoder: tiktoken.Encoding, sentence: str, completion_start_byte: int
) -> tuple[list[int], int]:
    tokens = encoder.encode_ordinary(sentence)
    pieces = [encoder.decode_single_token_bytes(token) for token in tokens]
    if b"".join(pieces) != sentence.encode("utf-8"):
        raise ValueError("GPT-2 byte reconstruction changed")
    cursor = 0
    score_start = len(tokens)
    for index, piece in enumerate(pieces):
        cursor += len(piece)
        if cursor > completion_start_byte:
            score_start = index
            break
    score_start = max(score_start, 1)
    if score_start >= len(tokens):
        raise ValueError(f"candidate has no scoreable continuation: {sentence!r}")
    return tokens, score_start


def fast_rows(
    official: Path, global_parallel: Path, global_nonparallel: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = official / "fast_eval"
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for directory, task in (
        ("blimp_fast", "blimp"),
        ("supplement_fast", "blimp_supplement"),
    ):
        for path in sorted((root / directory).glob("*.jsonl")):
            sources.append(source(path, f"{directory}/{path.name}"))
            for raw in jsonl(path):
                rows.append(
                    {
                        "task": task,
                        "subdomain": str(raw.get("UID", path.stem)),
                        "candidates": (
                            (str(raw["sentence_good"]), 0),
                            (str(raw["sentence_bad"]), 0),
                        ),
                        "label": 0,
                        "length_normalized": False,
                    }
                )

    entity = root / "entity_tracking_fast" / "regular.jsonl"
    sources.append(source(entity, "entity_tracking_fast/regular.jsonl"))
    for raw in jsonl(entity):
        options = [str(value) for value in raw["options"]]
        if any("nothing" in value for value in options):
            continue
        prefix = str(raw["input_prefix"])
        rows.append(
            {
                "task": "entity_tracking",
                "subdomain": f"regular_{int(raw['numops'])}_ops",
                "candidates": tuple(
                    (prefix + value, len(prefix.encode("utf-8")))
                    for value in options
                ),
                "label": 0,
                "length_normalized": False,
            }
        )

    ewok = root / "ewok_fast.zip"
    sources.append(source(ewok, "fast_eval/ewok_fast.zip"))
    with zipfile.ZipFile(ewok) as archive:
        for name in sorted(value for value in archive.namelist() if value.endswith(".jsonl")):
            with (
                archive.open(name, pwd=b"BabyLM2025") as binary,
                io.TextIOWrapper(binary, encoding="utf-8") as handle,
            ):
                for line in handle:
                    raw = json.loads(line)
                    target = str(raw["Target1"])
                    completion = " " + target
                    first = f"{raw['Context1']} {target}"
                    second = f"{raw['Context2']} {target}"
                    rows.append(
                        {
                            "task": "ewok_fast",
                            "subdomain": str(raw["Domain"]),
                            "candidates": (
                                (first, len(first.encode("utf-8")) - len(completion.encode("utf-8"))),
                                (second, len(second.encode("utf-8")) - len(completion.encode("utf-8"))),
                            ),
                            "label": 0,
                            "length_normalized": False,
                        }
                    )

    def add_global(path: Path, task: str, choices: int) -> None:
        sources.append(source(path, f"{task}/{path.name}"))
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle, delimiter="\t"):
                prompt = str(raw["prompt"])
                start = len(prompt.encode("utf-8"))
                rows.append(
                    {
                        "task": task,
                        "subdomain": task,
                        "candidates": tuple(
                            (f"{prompt} {raw[f'solution{index}']}", start)
                            for index in range(choices)
                        ),
                        "label": int(raw["label"]),
                        "length_normalized": True,
                    }
                )

    add_global(global_parallel, "global_piqa_parallel", 4)
    add_global(global_nonparallel, "global_piqa_nonparallel", 2)
    return rows, sources


def pack_scored_sequences(
    output: Path,
    name: str,
    sequences: list[tuple[list[int], int]],
    metadata: dict[str, np.ndarray],
) -> dict[str, Any]:
    root = output / name
    root.mkdir(parents=True)
    tokens_path = root / "tokens.bin"
    offsets = [0]
    score_starts = []
    with tokens_path.open("wb") as handle:
        for tokens, score_start in sequences:
            np.asarray(tokens, dtype="<u2").tofile(handle)
            offsets.append(offsets[-1] + len(tokens))
            score_starts.append(score_start)
    arrays = {
        "offsets": np.asarray(offsets, dtype="<u8"),
        "score_starts": np.asarray(score_starts, dtype="<u4"),
        **metadata,
    }
    records = {
        "tokens": record(output, tokens_path, "uint16", (offsets[-1],))
    }
    for key, values in arrays.items():
        path = root / f"{key}.npy"
        with path.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        records[key] = record(output, path, str(values.dtype), values.shape)
    return {"sequences": len(sequences), "tokens": offsets[-1], "records": records}


def pack_fast(
    output: Path,
    encoder: tiktoken.Encoding,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sequences: list[tuple[list[int], int]] = []
    example_offsets = [0]
    labels = []
    tasks: list[str] = []
    task_ids = []
    subdomains: list[str] = []
    subdomain_ids = []
    normalized = []
    for row in rows:
        for sentence, start in row["candidates"]:
            sequences.append(encode_candidate(encoder, sentence, int(start)))
        example_offsets.append(len(sequences))
        labels.append(int(row["label"]))
        if row["task"] not in tasks:
            tasks.append(row["task"])
        task_ids.append(tasks.index(row["task"]))
        if row["subdomain"] not in subdomains:
            subdomains.append(row["subdomain"])
        subdomain_ids.append(subdomains.index(row["subdomain"]))
        normalized.append(bool(row["length_normalized"]))
    packed = pack_scored_sequences(
        output,
        "fast_zero_shot",
        sequences,
        {
            "example_offsets": np.asarray(example_offsets, dtype="<u4"),
            "labels": np.asarray(labels, dtype=np.uint8),
            "task_ids": np.asarray(task_ids, dtype=np.uint8),
            "subdomain_ids": np.asarray(subdomain_ids, dtype="<u2"),
            "length_normalized": np.asarray(normalized, dtype=np.uint8),
        },
    )
    return {**packed, "examples": len(labels), "tasks": tasks, "subdomains": subdomains}


def pack_reading(output: Path, encoder: tiktoken.Encoding, csv_path: Path) -> dict[str, Any]:
    sequences: list[tuple[list[int], int]] = []
    current = []
    previous = []

    def append(context: str, word: str) -> int:
        context_tokens = encoder.encode_ordinary(context)
        target_tokens = encoder.encode_ordinary(word)
        if not context_tokens or not target_tokens:
            raise ValueError("reading item has empty context or target")
        sequences.append((context_tokens + target_tokens, len(context_tokens)))
        return len(sequences) - 1

    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            current.append(append(str(row["item"] or "None"), str(row["word"])))
            if row.get("prev_item") and row.get("prev_word"):
                previous.append(append(str(row["prev_item"]), str(row["prev_word"])))
            else:
                previous.append(-1)
    return pack_scored_sequences(
        output,
        "reading",
        sequences,
        {
            "current_indices": np.asarray(current, dtype="<i4"),
            "previous_indices": np.asarray(previous, dtype="<i4"),
        },
    )


def pack_aoa(output: Path, encoder: tiktoken.Encoding, path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    words = [word for word, contexts in raw.items() if len(word) > 1 and len(contexts) >= 0]
    sequences: list[tuple[list[int], int]] = []
    word_ids = []
    context_ids = []
    for word_id, word in enumerate(words):
        for context_id, row in enumerate(raw[word]):
            prefix = str(row["context"]).strip() + " "
            sentence = prefix + word.strip()
            sequences.append(encode_candidate(encoder, sentence, len(prefix.encode("utf-8"))))
            word_ids.append(word_id)
            context_ids.append(context_id)
    packed = pack_scored_sequences(
        output,
        "aoa",
        sequences,
        {
            "word_ids": np.asarray(word_ids, dtype="<u2"),
            "context_ids": np.asarray(context_ids, dtype="<u2"),
        },
    )
    return {**packed, "words": words}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--global-parallel", type=Path, required=True)
    parser.add_argument("--global-nonparallel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    encoder = tiktoken.get_encoding("gpt2")
    rows, sources = fast_rows(
        args.official_root, args.global_parallel, args.global_nonparallel
    )
    manifest = {
        "schema_version": 1,
        "format": "babylm2026_strict_gpt2_checkpoint_evaluation_v1",
        "tokenizer": {"encoding": "gpt2", "vocab_size": 50_257, "eot_token": 50_256},
        "evaluation_data_revision": EVALUATION_REVISION,
        "evaluation_implementation_revision": EVALUATOR_REVISION,
        "sources": sources,
        "fast_zero_shot": pack_fast(args.output, encoder, rows),
        "reading": pack_reading(
            args.output,
            encoder,
            args.official_root / "full_eval" / "reading" / "reading_data.csv",
        ),
        "aoa": pack_aoa(
            args.output,
            encoder,
            args.official_root / "full_eval" / "aoa" / "cdi_childes.json",
        ),
    }
    atomic_json(args.output / "manifest.json", manifest)
    closure = []
    for path in sorted(value for value in args.output.rglob("*") if value.is_file()):
        closure.append(
            {
                "path": path.relative_to(args.output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    catalog = hashlib.sha256(
        json.dumps(closure, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(
        args.output / "IMMUTABLE.json",
        {"schema_version": 1, "status": "closed", "catalog_sha256": catalog, "records": closure},
    )
    print(json.dumps({"manifest": sha256_file(args.output / "manifest.json"), "catalog": catalog}, sort_keys=True))


if __name__ == "__main__":
    main()
