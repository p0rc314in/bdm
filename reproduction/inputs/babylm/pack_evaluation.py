#!/usr/bin/env python3
"""Prepare immutable GPT-2-token BabyLM 2026 Strict training and evaluation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

FORMAT = "causal_overwrite_bus_babylm2026_strict_gpt2_v1"
GPT2_VOCAB_SIZE = 50_257
GPT2_EOT = 50_256

BABYLM_DATASET = "BabyLM-community/BabyLM-2026-Strict"
BABYLM_REVISION = "9e57baaaa91ac3c638746be14d1d5fa6c789f4cf"
BABYLM_EVAL_DATASET = "BabyLM-community/BabyLM-2026-Strict-Evals"
BABYLM_EVAL_REVISION = "8d52da9424a9ff30b9e8266c4f751aba9c504233"
BABYLM_EVALUATOR = "babylm-org/babylm-eval"
BABYLM_EVALUATOR_REVISION = "6f825c291e2c4c78ad33b1935fd64d45f52642dc"
GLOBAL_PIQA_PARALLEL = "mrlbenchmarks/global-piqa-parallel"
GLOBAL_PIQA_PARALLEL_REVISION = "b0b18516a8bc2cb1106bce3dd4db32848ca715ea"
GLOBAL_PIQA_NONPARALLEL = "mrlbenchmarks/global-piqa-nonparallel"
GLOBAL_PIQA_NONPARALLEL_REVISION = (
    "6777742fa3634c0583cda3b7f8a482ea7b1b0937"
)

CORPUS_FILES = (
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
)

FINETUNE_TASKS: dict[str, dict[str, Any]] = {
    "boolq": {
        "a": ("question",),
        "b": ("passage",),
        "labels": 2,
        "epochs": 10,
        "batch_size": 16,
        "selection_metric": "accuracy",
    },
    "mnli": {
        "a": ("premise",),
        "b": ("hypothesis",),
        "labels": 3,
        "epochs": 10,
        "batch_size": 32,
        "selection_metric": "accuracy",
    },
    "mrpc": {
        "a": ("sentence1",),
        "b": ("sentence2",),
        "labels": 2,
        "epochs": 10,
        "batch_size": 32,
        "selection_metric": "f1",
    },
    "multirc": {
        "a": ("question", "answer"),
        "b": ("paragraph",),
        "a_template": "Question: {} Answer: {}",
        "labels": 2,
        "epochs": 10,
        "batch_size": 16,
        "selection_metric": "accuracy",
    },
    "qqp": {
        "a": ("question1",),
        "b": ("question2",),
        "labels": 2,
        "epochs": 10,
        "batch_size": 32,
        "selection_metric": "f1",
    },
    "rte": {
        "a": ("sentence1",),
        "b": ("sentence2",),
        "labels": 2,
        "epochs": 10,
        "batch_size": 32,
        "selection_metric": "accuracy",
    },
    "wsc": {
        "a": ("span2_text", "span1_text"),
        "b": ("text",),
        "a_template": 'Does "{}" refer to "{}" in this passage?',
        "labels": 2,
        "epochs": 30,
        "batch_size": 32,
        "selection_metric": "accuracy",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def output_record(
    root: Path,
    path: Path,
    *,
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


def source_record(path: Path, identifier: str) -> dict[str, Any]:
    return {
        "identifier": identifier,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _candidate_score_start(
    encoder: tiktoken.Encoding,
    sentence: str,
    completion_start_byte: int,
) -> tuple[list[int], int]:
    encoded = encoder.encode_ordinary(sentence)
    raw = sentence.encode("utf-8")
    token_bytes = [encoder.decode_single_token_bytes(token) for token in encoded]
    if b"".join(token_bytes) != raw:
        raise ValueError("GPT-2 token byte reconstruction failed")
    if not 0 <= completion_start_byte <= len(raw):
        raise ValueError("completion start is outside the sentence")
    cursor = 0
    start = len(encoded)
    for index, piece in enumerate(token_bytes):
        cursor += len(piece)
        if cursor > completion_start_byte:
            start = index
            break
    if len(encoded) < 2:
        raise ValueError(f"candidate has fewer than two GPT-2 tokens: {sentence!r}")
    start = max(start, 1)
    if start >= len(encoded):
        raise ValueError(f"candidate completion has no scoreable tokens: {sentence!r}")
    return encoded, start


def _jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def zero_shot_rows(
    eval_dir: Path,
    global_parallel_path: Path,
    global_nonparallel_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    full = eval_dir / "evaluation_data" / "full_eval"

    blimp_dir = full / "blimp_filtered"
    for path in sorted(blimp_dir.glob("*.jsonl")):
        sources.append(source_record(path, f"{BABYLM_EVAL_DATASET}/{path.name}"))
        for raw in _jsonl_rows(path):
            rows.append(
                {
                    "task": "blimp",
                    "subdomain": str(raw["UID"]),
                    "candidates": (
                        (str(raw["sentence_good"]), 0),
                        (str(raw["sentence_bad"]), 0),
                    ),
                    "label": 0,
                    "length_normalized": False,
                }
            )

    supplement_dir = full / "supplement_filtered"
    for path in sorted(supplement_dir.glob("*.jsonl")):
        sources.append(source_record(path, f"{BABYLM_EVAL_DATASET}/{path.name}"))
        for raw in _jsonl_rows(path):
            rows.append(
                {
                    "task": "blimp_supplement",
                    "subdomain": path.stem,
                    "candidates": (
                        (str(raw["sentence_good"]), 0),
                        (str(raw["sentence_bad"]), 0),
                    ),
                    "label": 0,
                    "length_normalized": False,
                }
            )

    comps_dir = full / "comps"
    comps_names = {
        "comps_base": "base",
        "comps_wugs": "wugs",
        "comps_wugs_dist-before": "wugs_dist_before",
        "comps_wugs_dist-in-between": "wugs_dist_in_between",
    }
    for path in sorted(comps_dir.glob("*.jsonl")):
        sources.append(source_record(path, f"{BABYLM_EVAL_DATASET}/{path.name}"))
        subdomain = comps_names[path.stem]
        for raw in _jsonl_rows(path):
            completion = str(raw["property_phrase"])
            good = f"{raw['prefix_acceptable']} {completion}"
            bad = f"{raw['prefix_unacceptable']} {completion}"
            rows.append(
                {
                    "task": "comps",
                    "subdomain": subdomain,
                    "candidates": (
                        (
                            good,
                            len(good.encode("utf-8"))
                            - len(completion.encode("utf-8")),
                        ),
                        (
                            bad,
                            len(bad.encode("utf-8"))
                            - len(completion.encode("utf-8")),
                        ),
                    ),
                    "label": 0,
                    "length_normalized": False,
                }
            )

    entity_dir = full / "entity_tracking"
    for path in sorted(entity_dir.glob("*.jsonl")):
        sources.append(source_record(path, f"{BABYLM_EVAL_DATASET}/{path.name}"))
        for raw in _jsonl_rows(path):
            options = [str(value) for value in raw["options"]]
            if any("nothing" in option for option in options):
                continue
            prefix = str(raw["input_prefix"])
            start = len(prefix.encode("utf-8"))
            rows.append(
                {
                    "task": "entity_tracking",
                    "subdomain": f"{path.stem}_{int(raw['numops'])}_ops",
                    "candidates": tuple((prefix + option, start) for option in options),
                    "label": 0,
                    "length_normalized": False,
                }
            )

    ewok_zip = (
        eval_dir
        / "evaluation_data"
        / "fast_eval"
        / "ewok_fast.zip"
    )
    sources.append(
        source_record(ewok_zip, f"{BABYLM_EVAL_DATASET}/ewok_fast.zip")
    )
    with zipfile.ZipFile(ewok_zip) as archive:
        names = sorted(
            name for name in archive.namelist() if name.endswith(".jsonl")
        )
        for name in names:
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
                                (
                                    first,
                                    len(first.encode("utf-8"))
                                    - len(completion.encode("utf-8")),
                                ),
                                (
                                    second,
                                    len(second.encode("utf-8"))
                                    - len(completion.encode("utf-8")),
                                ),
                            ),
                            "label": 0,
                            "length_normalized": False,
                        }
                    )

    def append_global(
        path: Path,
        task: str,
        solutions: int,
        identifier: str,
    ) -> None:
        sources.append(source_record(path, identifier))
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for raw in reader:
                prompt = str(raw["prompt"])
                start = len(prompt.encode("utf-8"))
                candidates = tuple(
                    (f"{prompt} {raw[f'solution{index}']}", start)
                    for index in range(solutions)
                )
                rows.append(
                    {
                        "task": task,
                        "subdomain": task,
                        "candidates": candidates,
                        "label": int(raw["label"]),
                        "length_normalized": True,
                    }
                )

    append_global(
        global_parallel_path,
        "global_piqa_parallel",
        4,
        f"{GLOBAL_PIQA_PARALLEL}/data/parallel_eng_latn.tsv",
    )
    append_global(
        global_nonparallel_path,
        "global_piqa_nonparallel",
        2,
        f"{GLOBAL_PIQA_NONPARALLEL}/data/nonparallel_eng_latn.tsv",
    )
    return rows, sources


def prepare_zero_shot(
    eval_dir: Path,
    global_parallel_path: Path,
    global_nonparallel_path: Path,
    output_dir: Path,
    encoder: tiktoken.Encoding,
    *,
    maximum_sequence_length: int,
) -> dict[str, Any]:
    rows, sources = zero_shot_rows(
        eval_dir,
        global_parallel_path,
        global_nonparallel_path,
    )
    zero_dir = output_dir / "zero_shot"
    zero_dir.mkdir(parents=True, exist_ok=True)
    token_path = zero_dir / "tokens.bin"
    candidate_offsets = [0]
    score_starts: list[int] = []
    example_offsets = [0]
    labels: list[int] = []
    task_ids: list[int] = []
    subdomain_ids: list[int] = []
    normalized: list[int] = []
    task_catalog: list[str] = []
    task_lookup: dict[str, int] = {}
    subdomain_catalog: list[str] = []
    subdomain_lookup: dict[str, int] = {}

    def catalog_id(
        value: str,
        lookup: dict[str, int],
        catalog: list[str],
    ) -> int:
        if value not in lookup:
            lookup[value] = len(catalog)
            catalog.append(value)
        return lookup[value]

    token_count = 0
    with token_path.open("wb") as token_handle:
        for row in rows:
            task_id = catalog_id(str(row["task"]), task_lookup, task_catalog)
            subdomain_id = catalog_id(
                str(row["subdomain"]),
                subdomain_lookup,
                subdomain_catalog,
            )
            candidates = row["candidates"]
            for sentence, completion_start_byte in candidates:
                tokens, score_start = _candidate_score_start(
                    encoder,
                    sentence,
                    int(completion_start_byte),
                )
                if len(tokens) > maximum_sequence_length:
                    raise ValueError(
                        "zero-shot candidate exceeds maximum sequence length: "
                        f"{len(tokens)} > {maximum_sequence_length}"
                    )
                np.asarray(tokens, dtype="<u2").tofile(token_handle)
                token_count += len(tokens)
                candidate_offsets.append(token_count)
                score_starts.append(score_start)
            example_offsets.append(len(score_starts))
            labels.append(int(row["label"]))
            task_ids.append(task_id)
            subdomain_ids.append(subdomain_id)
            normalized.append(bool(row["length_normalized"]))

    arrays: dict[str, np.ndarray] = {
        "candidate_offsets": np.asarray(candidate_offsets, dtype="<u8"),
        "score_starts": np.asarray(score_starts, dtype="<u4"),
        "example_offsets": np.asarray(example_offsets, dtype="<u4"),
        "labels": np.asarray(labels, dtype=np.uint8),
        "task_ids": np.asarray(task_ids, dtype=np.uint8),
        "subdomain_ids": np.asarray(subdomain_ids, dtype="<u2"),
        "length_normalized": np.asarray(normalized, dtype=np.uint8),
    }
    records: dict[str, Any] = {
        "tokens": output_record(
            output_dir,
            token_path,
            dtype="uint16",
            shape=(token_count,),
        )
    }
    for name, values in arrays.items():
        path = zero_dir / f"{name}.npy"
        write_npy(path, values)
        records[name] = output_record(
            output_dir,
            path,
            dtype=str(values.dtype),
            shape=values.shape,
        )
    counts_by_task = {
        task: int(sum(task_ids_array == index))
        for index, task in enumerate(task_catalog)
        for task_ids_array in [arrays["task_ids"]]
    }
    return {
        "official_scoring": {
            "whole_sentence_sum": ["blimp", "blimp_supplement"],
            "continuation_sum": [
                "comps",
                "entity_tracking",
                "ewok_fast",
            ],
            "length_normalized_continuation_sum": [
                "global_piqa_parallel",
                "global_piqa_nonparallel",
            ],
            "tie_break": "lowest candidate index",
            "note": (
                "The official harness uses a random tie break. Exact floating "
                "ties are recorded and deterministically choose the lowest "
                "index so matched arms are reproducible."
            ),
        },
        "tasks": task_catalog,
        "subdomains": subdomain_catalog,
        "examples": len(rows),
        "candidates": len(score_starts),
        "gpt2_tokens": token_count,
        "examples_by_task": counts_by_task,
        "records": records,
        "source_files": sources,
    }


def _format_finetune_text(
    raw: dict[str, Any],
    task_config: dict[str, Any],
) -> tuple[str, str | None]:
    a_values = [str(raw[key]) for key in task_config["a"]]
    a_template = task_config.get("a_template")
    a_text = (
        str(a_template).format(*a_values)
        if a_template is not None
        else " ".join(a_values)
    )
    b_keys = task_config.get("b")
    if b_keys is None:
        return a_text, None
    b_values = [str(raw[key]) for key in b_keys]
    b_template = task_config.get("b_template")
    b_text = (
        str(b_template).format(*b_values)
        if b_template is not None
        else " ".join(b_values)
    )
    return a_text, b_text


def _truncate_pair(
    first: list[int],
    second: list[int] | None,
    maximum_length: int,
) -> list[int]:
    if second is None:
        return first[:maximum_length]
    first_stop = len(first)
    second_stop = len(second)
    while first_stop + second_stop > maximum_length:
        if first_stop > second_stop:
            first_stop -= 1
        else:
            second_stop -= 1
    return first[:first_stop] + second[:second_stop]


def prepare_finetune(
    eval_dir: Path,
    output_dir: Path,
    encoder: tiktoken.Encoding,
    *,
    maximum_sequence_length: int,
    seed: int,
) -> dict[str, Any]:
    source_dir = (
        eval_dir
        / "evaluation_data"
        / "full_eval"
        / "glue_filtered"
    )
    root = output_dir / "finetune"
    root.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, Any] = {}
    source_files: list[dict[str, Any]] = []
    for task_index, (task, task_config) in enumerate(FINETUNE_TASKS.items()):
        task_root = root / task
        task_root.mkdir(parents=True, exist_ok=True)
        split_records: dict[str, Any] = {}
        split_examples: dict[str, int] = {}
        for split in ("train", "valid"):
            source_path = source_dir / f"{task}.{split}.jsonl"
            source_files.append(
                source_record(
                    source_path,
                    f"{BABYLM_EVAL_DATASET}/{source_path.name}",
                )
            )
            tokens_path = task_root / f"{split}_tokens.bin"
            offsets = [0]
            labels: list[int] = []
            with tokens_path.open("wb") as token_handle:
                token_count = 0
                for raw in _jsonl_rows(source_path):
                    first_text, second_text = _format_finetune_text(
                        raw,
                        task_config,
                    )
                    first = encoder.encode_ordinary(first_text)
                    second = (
                        None
                        if second_text is None
                        else encoder.encode_ordinary(second_text)
                    )
                    tokens = _truncate_pair(
                        first,
                        second,
                        maximum_sequence_length,
                    )
                    if not tokens:
                        tokens = [GPT2_EOT]
                    label = int(raw["label"])
                    if not 0 <= label < int(task_config["labels"]):
                        raise ValueError(f"invalid {task} label: {label}")
                    np.asarray(tokens, dtype="<u2").tofile(token_handle)
                    token_count += len(tokens)
                    offsets.append(token_count)
                    labels.append(label)
            offsets_array = np.asarray(offsets, dtype="<u8")
            labels_array = np.asarray(labels, dtype=np.uint8)
            offsets_path = task_root / f"{split}_offsets.npy"
            labels_path = task_root / f"{split}_labels.npy"
            write_npy(offsets_path, offsets_array)
            write_npy(labels_path, labels_array)
            split_examples[split] = len(labels)
            split_records[split] = {
                "tokens": output_record(
                    output_dir,
                    tokens_path,
                    dtype="uint16",
                    shape=(token_count,),
                ),
                "offsets": output_record(
                    output_dir,
                    offsets_path,
                    dtype="uint64",
                    shape=offsets_array.shape,
                ),
                "labels": output_record(
                    output_dir,
                    labels_path,
                    dtype="uint8",
                    shape=labels_array.shape,
                ),
            }

        train_examples = split_examples["train"]
        orders: list[np.ndarray] = []
        for epoch in range(int(task_config["epochs"])):
            generator = np.random.Generator(
                np.random.PCG64(
                    seed
                    + task_index * 1_000_003
                    + epoch * 10_000_019
                )
            )
            orders.append(
                generator.permutation(train_examples).astype(
                    "<u4",
                    copy=False,
                )
            )
        order = np.concatenate(orders)
        order_path = task_root / "train_order.npy"
        write_npy(order_path, order)
        split_records["train_order"] = output_record(
            output_dir,
            order_path,
            dtype="uint32",
            shape=order.shape,
        )
        tasks[task] = {
            "num_labels": int(task_config["labels"]),
            "epochs": int(task_config["epochs"]),
            "batch_size": int(task_config["batch_size"]),
            "selection_metric": str(task_config["selection_metric"]),
            "sequence_length": maximum_sequence_length,
            "examples": split_examples,
            "records": split_records,
        }
    return {
        "official_hyperparameters": {
            "learning_rate": 3e-5,
            "weight_decay": 0.01,
            "warmup_proportion": 0.06,
            "minimum_learning_rate_factor": 0.1,
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "classifier_dropout": 0.1,
            "classifier_layer_norm_epsilon": 1e-5,
            "seed": seed,
            "best_validation_checkpoint": True,
        },
        "padding_equivalence": (
            "Sequences are right padded and the last non-pad hidden state is "
            "selected. Causality makes this representation independent of "
            "future padding while avoiding left-pad state contamination."
        ),
        "tasks": tasks,
        "source_files": source_files,
    }
