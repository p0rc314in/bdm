from __future__ import annotations
from typing import Any
import torch
from torch import nn
POINTER_KEYS=192
POINTER_HOPS=(1,2,4,8)
MAXIMUM_SPAN=16
SPAN_VALUES=64
SPAN_KEYS=64
OUTPUT_CLASSES=192
QUERIES=16
POINTER_MAP_OFFSET=0
POINTER_QUERY_OFFSET=36864
SPAN_MAP_OFFSET=37632
SPAN_QUERY_OFFSET=103168
OVERWRITE_MAP_OFFSET=104192
OVERWRITE_QUERY_OFFSET=169728


class AdaptiveRecallEmbedding(nn.Module):
    """Exact semantic-role adapter from the accepted adaptive-recall task."""

    _POINTER_MAPPING = 0
    _POINTER_QUERY = 1
    _SPAN_MAPPING = 2
    _SPAN_QUERY = 3
    _OVERWRITE_MAPPING = 4
    _OVERWRITE_QUERY = 5

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0 or width % 2:
            raise ValueError("recall embedding width must be positive and even")
        self.width = width
        self.half_width = width // 2
        self.identity = nn.Embedding(OUTPUT_CLASSES, self.half_width)
        self.slot = nn.Embedding(MAXIMUM_SPAN, self.half_width)
        self.role = nn.Embedding(6, width)
        self.hop = nn.Embedding(len(POINTER_HOPS), width)

    @staticmethod
    def _local(token_ids: torch.Tensor, offset: int, count: int) -> torch.Tensor:
        return (token_ids - offset).clamp(min=0, max=count - 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be [B,T]")
        first = torch.zeros(
            *token_ids.shape,
            self.half_width,
            device=token_ids.device,
            dtype=self.identity.weight.dtype,
        )
        second = torch.zeros_like(first)
        role_ids = torch.zeros_like(token_ids)
        hop_features = torch.zeros(
            *token_ids.shape,
            self.width,
            device=token_ids.device,
            dtype=self.identity.weight.dtype,
        )

        pointer_mapping = (token_ids >= POINTER_MAP_OFFSET) & (
            token_ids < POINTER_QUERY_OFFSET
        )
        local = self._local(token_ids, POINTER_MAP_OFFSET, POINTER_KEYS * POINTER_KEYS)
        first = torch.where(
            pointer_mapping.unsqueeze(-1),
            self.identity(local // POINTER_KEYS),
            first,
        )
        second = torch.where(
            pointer_mapping.unsqueeze(-1),
            self.identity(local % POINTER_KEYS),
            second,
        )

        pointer_query = (token_ids >= POINTER_QUERY_OFFSET) & (
            token_ids < SPAN_MAP_OFFSET
        )
        local = self._local(
            token_ids,
            POINTER_QUERY_OFFSET,
            len(POINTER_HOPS) * POINTER_KEYS,
        )
        first = torch.where(
            pointer_query.unsqueeze(-1),
            self.identity(local % POINTER_KEYS),
            first,
        )
        hop_features = torch.where(
            pointer_query.unsqueeze(-1),
            self.hop(local // POINTER_KEYS),
            hop_features,
        )
        role_ids = torch.where(
            pointer_query,
            torch.full_like(role_ids, self._POINTER_QUERY),
            role_ids,
        )

        span_mapping = (token_ids >= SPAN_MAP_OFFSET) & (token_ids < SPAN_QUERY_OFFSET)
        local = self._local(
            token_ids,
            SPAN_MAP_OFFSET,
            SPAN_KEYS * MAXIMUM_SPAN * SPAN_VALUES,
        )
        value = local % SPAN_VALUES
        key_slot = local // SPAN_VALUES
        first = torch.where(
            span_mapping.unsqueeze(-1),
            self.identity(key_slot // MAXIMUM_SPAN)
            + self.slot(key_slot % MAXIMUM_SPAN),
            first,
        )
        second = torch.where(span_mapping.unsqueeze(-1), self.identity(value), second)
        role_ids = torch.where(
            span_mapping,
            torch.full_like(role_ids, self._SPAN_MAPPING),
            role_ids,
        )

        span_query = (token_ids >= SPAN_QUERY_OFFSET) & (
            token_ids < OVERWRITE_MAP_OFFSET
        )
        local = self._local(token_ids, SPAN_QUERY_OFFSET, SPAN_KEYS * MAXIMUM_SPAN)
        first = torch.where(
            span_query.unsqueeze(-1),
            self.identity(local // MAXIMUM_SPAN) + self.slot(local % MAXIMUM_SPAN),
            first,
        )
        role_ids = torch.where(
            span_query,
            torch.full_like(role_ids, self._SPAN_QUERY),
            role_ids,
        )

        overwrite_mapping = (token_ids >= OVERWRITE_MAP_OFFSET) & (
            token_ids < OVERWRITE_QUERY_OFFSET
        )
        local = self._local(
            token_ids,
            OVERWRITE_MAP_OFFSET,
            SPAN_KEYS * MAXIMUM_SPAN * SPAN_VALUES,
        )
        value = local % SPAN_VALUES
        key_slot = local // SPAN_VALUES
        first = torch.where(
            overwrite_mapping.unsqueeze(-1),
            self.identity(key_slot // MAXIMUM_SPAN)
            + self.slot(key_slot % MAXIMUM_SPAN),
            first,
        )
        second = torch.where(
            overwrite_mapping.unsqueeze(-1), self.identity(value), second
        )
        role_ids = torch.where(
            overwrite_mapping,
            torch.full_like(role_ids, self._OVERWRITE_MAPPING),
            role_ids,
        )

        overwrite_query = (token_ids >= OVERWRITE_QUERY_OFFSET) & (
            token_ids < OVERWRITE_QUERY_OFFSET + SPAN_KEYS * MAXIMUM_SPAN
        )
        local = self._local(
            token_ids,
            OVERWRITE_QUERY_OFFSET,
            SPAN_KEYS * MAXIMUM_SPAN,
        )
        first = torch.where(
            overwrite_query.unsqueeze(-1),
            self.identity(local // MAXIMUM_SPAN) + self.slot(local % MAXIMUM_SPAN),
            first,
        )
        role_ids = torch.where(
            overwrite_query,
            torch.full_like(role_ids, self._OVERWRITE_QUERY),
            role_ids,
        )
        return torch.cat((first, second), dim=-1) + self.role(role_ids) + hop_features


def initialize_task_embedding(module: AdaptiveRecallEmbedding, seed: int) -> None:
    """Retain the accepted task adapter's independent semantic RNG stream."""

    base = seed * 100_000 + 101
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base)
        for child in module.modules():
            if child is module:
                continue
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()


def validate_codec(manifest: dict[str, Any]) -> None:
    expected = {
        "pointer_mapping": (POINTER_MAP_OFFSET, POINTER_KEYS * POINTER_KEYS),
        "pointer_query": (POINTER_QUERY_OFFSET, len(POINTER_HOPS) * POINTER_KEYS),
        "span_mapping": (
            SPAN_MAP_OFFSET,
            SPAN_KEYS * MAXIMUM_SPAN * SPAN_VALUES,
        ),
        "span_query": (SPAN_QUERY_OFFSET, SPAN_KEYS * MAXIMUM_SPAN),
        "overwrite_mapping": (
            OVERWRITE_MAP_OFFSET,
            SPAN_KEYS * MAXIMUM_SPAN * SPAN_VALUES,
        ),
        "overwrite_query": (
            OVERWRITE_QUERY_OFFSET,
            SPAN_KEYS * MAXIMUM_SPAN,
        ),
    }
    observed = {
        name: (int(row["offset"]), int(row["count"]))
        for name, row in manifest["token_codec"].items()
    }
    if observed != expected:
        raise ValueError(f"adaptive-recall codec changed: {observed}")
    if int(manifest["output_classes"]) != OUTPUT_CLASSES:
        raise ValueError("adaptive-recall output class count changed")
