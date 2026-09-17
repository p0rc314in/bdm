"""Checkpoint-compatible controller and initialization methods."""

from __future__ import annotations
import hashlib
import math
import torch
from torch import nn
import torch.nn.functional as F
Tensor = torch.Tensor

class ParameterMethods:
    @property
    def total_rows(self) -> int:
        return self.bank_count * self.num_heads * self.bank_size

    @property
    def touched_rows(self) -> int:
        return self.num_writes * self.num_heads * self.bank_size

    @property
    def logical_reads(self) -> int:
        return self.num_reads * self.num_heads * self.bank_size

    @property
    def logical_writes(self) -> int:
        return self.num_writes * self.num_heads * self.bank_size

    @property
    def initial_state_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in (
                self.initial_state,
                self.initial_row_factor,
                self.initial_column_factor,
            )
            if parameter is not None
        )

    def materialize_initial_state(self) -> Tensor:
        """Return the logical ``[banks, bank_size, value_width]`` state.

        Product-key mode shares two additive factor tables across logical rows.
        The dense state remains sequence workspace, but it is no longer a full
        persistent parameter with a full optimizer state.
        """

        if self.num_heads != 1:
            return (self.initial_row_factor[:, None, :] +
                    self.initial_column_factor[None, :, :]).reshape(
                        self.bank_count, self.num_heads, self.bank_size, self.head_value_width)
        if self.initial_state is not None:
            return self.initial_state
        if (
            self.initial_row_factor is not None
            and self.initial_column_factor is not None
        ):
            return (
                self.initial_row_factor[:, None, :]
                + self.initial_column_factor[None, :, :]
            ).reshape(self.bank_count, self.bank_size, self.value_width)
        return self.dt_bias.new_zeros(
            self.bank_count,
            self.bank_size,
            self.value_width,
            dtype=torch.float32,
        )

    def _role_generator(
        self,
        role: str,
        parameter: Tensor,
        initialization_seed: int | None,
    ) -> torch.Generator | None:
        if initialization_seed is None:
            return None
        role_key = f"bank_sparse_gdn2.{self.layer_id}.{role}".encode()
        role_hash = int.from_bytes(
            hashlib.blake2b(role_key, digest_size=8).digest(),
            byteorder="little",
        )
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed((initialization_seed ^ role_hash) % (1 << 63))
        return generator

    def init_weights(
        self,
        init_std: float | None = None,
        factor: float = 1.0,
        width_scaling: float | None = None,
        *,
        initialization_seed: int | None = None,
    ) -> None:
        """Initialize GDN2 controllers and the pinned shared-residual router.

        The dense GDN2 projections retain their scaled-Xavier initialization.
        Shared routing uses the residualized-routing initializer:
        initialize two matched native projections, store their mean as the
        shared base, and zero both residuals.
        """

        del width_scaling
        if initialization_seed is None:
            initialization_seed = self.initialization_seed
        if initialization_seed is not None and initialization_seed < 0:
            raise ValueError("initialization_seed must be non-negative")
        gain = 2**-2.5
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.startswith("shared_router."):
                    continue
                nn.init.xavier_uniform_(
                    module.weight,
                    gain=gain,
                    generator=self._role_generator(
                        f"{name}.weight", module.weight, initialization_seed
                    ),
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.shared_router is not None:
            route_std = (init_std or self.hidden_size**-0.5) / factor
            self.shared_router.initialize_from_independent(
                std=route_std,
                read_generator=self._role_generator(
                    "read_router.weight",
                    self.shared_router.base.weight,
                    initialization_seed,
                ),
                write_generator=self._role_generator(
                    "write_router.weight",
                    self.shared_router.base.weight,
                    initialization_seed,
                ),
            )
        with torch.no_grad():
            self.A_log.uniform_(
                1.0,
                16.0,
                generator=self._role_generator(
                    "decay_rate", self.A_log, initialization_seed
                ),
            ).log_()
            log_dt = torch.empty_like(self.dt_bias).uniform_(
                math.log(0.001),
                math.log(0.1),
                generator=self._role_generator(
                    "time_step", self.dt_bias, initialization_seed
                ),
            )
            dt = log_dt.exp().clamp_min(1e-4)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            for role, parameter in (
                ("initial_state.full", self.initial_state),
                ("initial_state.row", self.initial_row_factor),
                ("initial_state.column", self.initial_column_factor),
            ):
                if parameter is not None:
                    std = (init_std or self.hidden_size**-0.5) / factor
                    nn.init.trunc_normal_(
                        parameter,
                        mean=0.0,
                        std=std,
                        a=-3 * std,
                        b=3 * std,
                        generator=self._role_generator(
                            role, parameter, initialization_seed
                        ),
                    )
            self.output_norm_weight.fill_(1.0)
            if hasattr(self, 'output_norm_bias'):
                self.output_norm_bias.zero_()

    def reset_parameters(self) -> None:
        self.init_weights(initialization_seed=self.initialization_seed)

    def _route_projections(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if self.shared_router is not None:
            return self.shared_router.read(hidden), self.shared_router.write(hidden)
        if self.read_route_proj is None or self.write_route_proj is None:
            raise RuntimeError("independent bank routing projections are absent")
        return self.read_route_proj(hidden), self.write_route_proj(hidden)

    def _controllers(
        self, hidden: Tensor, *, batch_qk: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.num_heads != 1:
            from .heads import controllers
            return controllers(self, hidden)
        q_projected = F.silu(self.q_proj(hidden))
        k_projected = F.silu(self.k_proj(hidden))
        # PyTorch normalization is an fp32 autocast op. FLA derives the WY
        # triangular-factor dtype from k, so leaving normalized q/k in fp32
        # while v/w remain bf16 makes its recompute dot ill-typed. Normalize
        # accurately, then return to the projection dtype used by the kernel.
        if batch_qk:
            q, k = self._normalize_qk(torch.cat((q_projected, k_projected), dim=0)).chunk(2, dim=0)
        else:
            q = self._normalize_qk(q_projected)
            k = self._normalize_qk(k_projected)
        v = F.silu(self.v_proj(hidden))
        g = -self.A_log.float().exp() * F.softplus(
            self.f_proj(hidden).float() + self.dt_bias.float()
        )
        b = torch.sigmoid(self.b_proj(hidden))
        w = torch.sigmoid(self.w_proj(hidden))
        return q, k, v, g, b, w

    def _normalize_qk(self, value: Tensor) -> Tensor:
        if self.config.qk_normalization == 'nvidia':
            if value.is_cuda:
                from fla.modules.l2norm import l2norm
                return l2norm(value, eps=1e-6)
            return (value.float() * torch.rsqrt(value.float().square().sum(-1, keepdim=True) + 1e-6)).to(value.dtype)
        return F.normalize(value.float(), dim=-1).to(value.dtype)

    def _project_bank_output(self, hidden: Tensor, raw: Tensor) -> Tensor:
        if self.config.output_normalization == 'layer':
            normalized = F.layer_norm(raw, (self.value_width,), self.output_norm_weight,
                                      self.output_norm_bias, self.norm_eps)
            return self.o_proj(normalized * torch.sigmoid(self.output_gate(hidden)).to(raw.dtype))
        variance = raw.float().square().mean(dim=-1, keepdim=True)
        normalized = raw * torch.rsqrt(variance + self.norm_eps).to(raw.dtype)
        normalized = normalized * self.output_norm_weight.to(raw.dtype)
        return self.o_proj(
            normalized * torch.sigmoid(self.output_gate(hidden)).to(raw.dtype)
        )
