# Copyright (c) 2026 p0rc314in

"""Routing projections for Sparse Delta Memory."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SharedResidualRouter(nn.Module):
    """One routing base plus independent learned read and write residuals."""

    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        if input_width <= 0 or output_width <= 0:
            raise ValueError("router widths must be positive")
        self.base = nn.Linear(input_width, output_width)
        self.read_residual = nn.Linear(input_width, output_width)
        self.write_residual = nn.Linear(input_width, output_width)

    def read(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(
            hidden,
            self.base.weight + self.read_residual.weight,
            self.base.bias + self.read_residual.bias,
        )

    def write(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(
            hidden,
            self.base.weight + self.write_residual.weight,
            self.base.bias + self.write_residual.bias,
        )

    @torch.no_grad()
    def initialize_from_independent(
        self,
        *,
        std: float,
        read_generator: torch.Generator | None = None,
        write_generator: torch.Generator | None = None,
    ) -> None:
        """Initialize the base from two matched native projections."""

        if std <= 0:
            raise ValueError("router initialization scale must be positive")
        read = torch.empty_like(self.base.weight)
        write = torch.empty_like(self.base.weight)
        for weight, generator in (
            (read, read_generator),
            (write, write_generator),
        ):
            nn.init.trunc_normal_(
                weight,
                mean=0.0,
                std=std,
                a=-3.0 * std,
                b=3.0 * std,
                generator=generator,
            )
        self.base.weight.copy_(0.5 * (read + write))
        self.base.bias.zero_()
        self.read_residual.weight.zero_()
        self.read_residual.bias.zero_()
        self.write_residual.weight.zero_()
        self.write_residual.bias.zero_()
