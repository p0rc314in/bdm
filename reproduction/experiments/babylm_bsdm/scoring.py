from __future__ import annotations
from collections import defaultdict
from typing import Any
import math
import numpy as np
import torch
from torch import nn
from .spec import SPEC

def hidden_states(model, input_ids):
    return model.norm(model._hidden(input_ids))

class ClassifierHead(nn.Module):
    def __init__(self, labels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(SPEC.width, 1e-5, elementwise_affine=False),
            nn.Linear(SPEC.width, SPEC.width),
            nn.GELU(),
            nn.LayerNorm(SPEC.width, 1e-5, elementwise_affine=False),
            nn.Dropout(0.1),
            nn.Linear(SPEC.width, labels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

class SequenceClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, labels: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = ClassifierHead(labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.backbone.auxiliary_coefficient:
            self.backbone.elastic_valid = torch.arange(input_ids.shape[1],device=input_ids.device)[None,:] < lengths[:,None]
        hidden = hidden_states(self.backbone, input_ids)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return self.classifier(hidden[rows, lengths - 1])

def classification_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    accuracy = float((predictions == labels).mean())
    if int(labels.max()) > 1:
        return {"accuracy": accuracy}
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    f1_denominator = 2 * tp + fp + fn
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": accuracy,
        "f1": 0.0 if not f1_denominator else 2 * tp / f1_denominator,
        "mcc": 0.0 if not mcc_denominator else (tp * tn - fp * fn) / mcc_denominator,
    }

def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.1, 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

def aggregate_fast(
    data: PackedEvaluation, losses: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    arrays = data.sections["fast_zero_shot"]
    example_offsets = arrays["example_offsets"]
    labels = arrays["labels"]
    task_ids = arrays["task_ids"]
    subdomain_ids = arrays["subdomain_ids"]
    normalized = arrays["length_normalized"]
    sequence_offsets = arrays["offsets"]
    score_starts = arrays["score_starts"]
    predictions = np.empty(len(labels), dtype=np.uint8)
    correct = np.empty(len(labels), dtype=np.uint8)
    ties = 0
    for example in range(len(labels)):
        first = int(example_offsets[example])
        stop = int(example_offsets[example + 1])
        scores = -losses[first:stop].copy()
        if int(normalized[example]):
            counts = np.asarray(
                [
                    int(sequence_offsets[index + 1])
                    - int(sequence_offsets[index])
                    - int(score_starts[index])
                    for index in range(first, stop)
                ],
                dtype=np.float64,
            )
            scores /= counts
        winners = np.flatnonzero(scores == scores.max())
        ties += int(len(winners) > 1)
        predictions[example] = int(winners[0])
        correct[example] = predictions[example] == int(labels[example])

    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for example, value in enumerate(correct):
        row = counts[(int(task_ids[example]), int(subdomain_ids[example]))]
        row[0] += int(value)
        row[1] += 1
    tasks: dict[str, Any] = {}
    task_names = data.manifest["fast_zero_shot"]["tasks"]
    subdomain_names = data.manifest["fast_zero_shot"]["subdomains"]
    for task_index, task in enumerate(task_names):
        subdomains = sorted(
            {
                int(subdomain_ids[index])
                for index in np.flatnonzero(task_ids == task_index)
            }
        )
        subdomain_accuracy = {
            subdomain_names[subdomain]: (
                counts[(task_index, subdomain)][0]
                / counts[(task_index, subdomain)][1]
            )
            for subdomain in subdomains
        }
        if task == "entity_tracking":
            split_accuracy = {}
            for split in ("regular", "ambiref", "move_contents"):
                values = [
                    accuracy
                    for name, accuracy in subdomain_accuracy.items()
                    if name.startswith(split)
                ]
                if values:
                    split_accuracy[split] = sum(values) / len(values)
            average = sum(split_accuracy.values()) / len(split_accuracy)
        else:
            split_accuracy = {}
            average = sum(subdomain_accuracy.values()) / len(subdomain_accuracy)
        indices = np.flatnonzero(task_ids == task_index)
        tasks[str(task)] = {
            "accuracy": average,
            "micro_accuracy": float(correct[indices].mean()),
            "examples": len(indices),
            "subdomain_accuracy": subdomain_accuracy,
            "split_accuracy": split_accuracy,
        }
    return (
        {
            "tasks": tasks,
            "macro_average": sum(row["accuracy"] for row in tasks.values())
            / len(tasks),
            "examples": len(labels),
            "candidates": len(losses),
            "exact_score_ties": ties,
        },
        predictions,
    )
