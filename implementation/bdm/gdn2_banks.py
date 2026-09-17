"""Reference bank equations and causal packing."""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .config import near_square_factors
Tensor = torch.Tensor


class _StoredState(torch.autograd.Function):
    """Rounded forward storage with the kernels' FP32 adjoint accumulation."""
    @staticmethod
    def forward(ctx, value, dtype):
        return value.to(dtype).float()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def reference_packed_gdn2(*, q, k, v, g, b, w, initial_state, cu_seqlens, scale,
                         state_dtype=torch.float32, state_interval=16):
    """Small dense-bank oracle including packed-tensor storage precision.

    Pack first so route-weighted q/b/w round exactly at the same boundary as
    the production operator. Readouts likewise round before token scatter.
    ``serial_banked_gdn2`` remains the independent unrounded equation oracle.
    """
    output = torch.zeros_like(v, dtype=torch.float32)
    for sequence in range(cu_seqlens.numel() - 1):
        start, end = int(cu_seqlens[sequence]), int(cu_seqlens[sequence + 1])
        state = _StoredState.apply(initial_state[sequence, 0], state_dtype)
        for position in range(start, end):
            if state_dtype == torch.bfloat16 and (position - start) % state_interval == 0:
                state = _StoredState.apply(state, state_dtype)
            state = state * g[0, position, 0].float().exp().unsqueeze(-1)
            key = k[0, position, 0].float()
            erase = ((b[0, position, 0].float() * key).unsqueeze(-1) * state).sum(-2)
            update = w[0, position, 0].float() * v[0, position, 0].float() - erase
            state = state + key.unsqueeze(-1) * update.unsqueeze(-2)
            output[0, position, 0] = scale * (q[0, position, 0].float().unsqueeze(-1) * state).sum(-2)
    return output.to(v.dtype)

@dataclass(frozen=True)
class PackedBankEvents:
    """Causally ordered variable-length sequences, one per batch-bank pair."""

    q: Tensor
    k: Tensor
    v: Tensor
    g: Tensor
    b: Tensor
    w: Tensor
    cu_seqlens: Tensor
    token_indices: Tensor
    sequence_indices: Tensor
    read_events: Tensor

    @property
    def event_count(self) -> int:
        return self.token_indices.numel()


@dataclass(frozen=True)
class BankGDN2Diagnostics:
    write_indices: Tensor
    read_indices: Tensor
    write_weights: Tensor
    read_weights: Tensor
    events_per_bank: Tensor


def product_key_route(
    projected: Tensor,
    *,
    bank_count: int,
    selected_banks: int,
    batch_factors: bool = False,
) -> tuple[Tensor, Tensor]:
    """Exact top-k product-key routing without an SDM kernel dependency."""

    if projected.ndim != 3:
        raise ValueError("projected routes must be [batch, time, key_width]")
    first_width, second_width = near_square_factors(bank_count)
    if projected.shape[-1] != first_width + second_width:
        raise ValueError("projected route width does not match the bank table")
    if not 1 <= selected_banks <= bank_count:
        raise ValueError("selected bank count is outside the bank table")

    first, second = torch.split(projected, [first_width, second_width], dim=-1)
    subselected = min(selected_banks, first_width, second_width)
    if batch_factors:
        # Independent factors share a launch. Rectangular product tables pad
        # the shorter factor with excluded scores, retaining native top-k.
        factors = (projected.reshape(*projected.shape[:-1], 2, first_width)
                   if first_width == second_width else
                   torch.stack((F.pad(first, (0, second_width - first_width), value=-float('inf')),
                                second), dim=-2))
        values, indices = torch.topk(factors, k=subselected, dim=-1)
        first_values, second_values = values.unbind(-2)
        first_indices, second_indices = indices.unbind(-2)
    else:
        first_values, first_indices = torch.topk(first, k=subselected, dim=-1)
        second_values, second_indices = torch.topk(second, k=subselected, dim=-1)
    pair_values = (first_values.unsqueeze(-1) + second_values.unsqueeze(-2)).flatten(-2)
    count = min(selected_banks, pair_values.shape[-1])
    values, pair_indices = torch.topk(pair_values, k=count, dim=-1)
    first_choice = torch.div(pair_indices, subselected, rounding_mode="floor")
    second_choice = pair_indices.remainder(subselected)
    row = torch.gather(first_indices, -1, first_choice)
    column = torch.gather(second_indices, -1, second_choice)
    indices = row * second_width + column
    return torch.softmax(values.float(), dim=-1).to(projected.dtype), indices


def pack_bank_events(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    write_indices: Tensor,
    write_weights: Tensor,
    read_indices: Tensor,
    read_weights: Tensor,
    *,
    bank_count: int,
    kernel_key_width: int | None = None,
    routed_decay: bool = False,
) -> PackedBankEvents:
    """Turn selected bank accesses into dense causal GDN2 subsequences.

    Each selected write becomes an update event and each selected read becomes
    a non-mutating query event. Sorting by ``(batch, bank, time, phase)`` makes
    overlapping accesses execute write-before-read without requiring a sparse
    memory recurrence. The event count is fixed at ``BT * (W + R)``.
    """

    if not (q.shape == k.shape == g.shape == b.shape):
        raise ValueError("q, k, g, and b must share [batch,time,bank_size]")
    if not (v.shape == w.shape):
        raise ValueError("v and w must share [batch,time,value_width]")
    if q.shape[:2] != v.shape[:2]:
        raise ValueError("controller tensors disagree on batch or time")
    if write_indices.shape != write_weights.shape:
        raise ValueError("write route indices and weights must align")
    if read_indices.shape != read_weights.shape:
        raise ValueError("read route indices and weights must align")
    if write_indices.shape[:2] != q.shape[:2] or read_indices.shape[:2] != q.shape[:2]:
        raise ValueError("routes disagree with controller batch or time")
    if write_indices.dtype != torch.long or read_indices.dtype != torch.long:
        raise ValueError("bank indices must be int64")

    batch, time, key_width = q.shape
    kernel_key_width = key_width if kernel_key_width is None else kernel_key_width
    if kernel_key_width < key_width:
        raise ValueError(
            "kernel key width cannot be smaller than the logical bank size"
        )
    if kernel_key_width > 256:
        raise ValueError("GDN2 supports kernel key widths up to 256")
    value_width = v.shape[-1]
    tokens = torch.arange(batch * time, device=q.device, dtype=torch.long).reshape(
        batch, time
    )
    times = torch.arange(time, device=q.device, dtype=torch.long).view(1, time)
    batch_offsets = (
        torch.arange(batch, device=q.device, dtype=torch.long).view(batch, 1)
        * bank_count
    )

    write_tokens = tokens.unsqueeze(-1).expand_as(write_indices).reshape(-1)
    read_tokens = tokens.unsqueeze(-1).expand_as(read_indices).reshape(-1)
    write_times = times.unsqueeze(-1).expand_as(write_indices).reshape(-1)
    read_times = times.unsqueeze(-1).expand_as(read_indices).reshape(-1)
    write_sequences = (write_indices + batch_offsets.unsqueeze(-1)).reshape(-1)
    read_sequences = (read_indices + batch_offsets.unsqueeze(-1)).reshape(-1)

    # Phase 0 is mutation and phase 1 is observation. This also makes separate
    # read/write routers exact when they happen to select the same bank.
    write_order = (write_sequences * time + write_times) * 2
    read_order = (read_sequences * time + read_times) * 2 + 1
    order = torch.argsort(torch.cat((write_order, read_order)))

    flat_q = q.reshape(batch * time, key_width)
    flat_k = k.reshape(batch * time, key_width)
    flat_v = v.reshape(batch * time, value_width)
    flat_g = g.reshape(batch * time, key_width)
    flat_b = b.reshape(batch * time, key_width)
    flat_w = w.reshape(batch * time, value_width)
    flat_write_weights = write_weights.reshape(-1)
    flat_read_weights = read_weights.reshape(-1)

    zero_write_q = flat_q.new_zeros(write_tokens.numel(), key_width)
    zero_read_g = flat_g.new_zeros(read_tokens.numel(), key_width)
    zero_read_b = flat_b.new_zeros(read_tokens.numel(), key_width)
    zero_read_w = flat_w.new_zeros(read_tokens.numel(), value_width)

    event_q = torch.cat(
        (
            zero_write_q,
            flat_q[read_tokens] * flat_read_weights.unsqueeze(-1),
        )
    )[order]
    event_k = torch.cat((flat_k[write_tokens], flat_k[read_tokens]))[order]
    event_v = torch.cat((flat_v[write_tokens], flat_v[read_tokens]))[order]
    write_g = flat_g[write_tokens]
    if routed_decay:
        write_g = write_g.float() * flat_write_weights.float().unsqueeze(-1)
    event_g = torch.cat((write_g, zero_read_g))[order]
    event_b = torch.cat(
        (
            flat_b[write_tokens] * flat_write_weights.unsqueeze(-1),
            zero_read_b,
        )
    )[order]
    event_w = torch.cat(
        (
            flat_w[write_tokens] * flat_write_weights.unsqueeze(-1),
            zero_read_w,
        )
    )[order]

    # FLA's small-K training path currently produces an fp32/bf16 Triton dot
    # mismatch.  Physical zero padding selects the regular tensor-core shape
    # without adding logical rows: padded q/k/b are zero, padded g is zero, and
    # the corresponding initial-state rows are also zero at the call site.
    key_padding = kernel_key_width - key_width
    if key_padding:
        event_q = F.pad(event_q, (0, key_padding))
        event_k = F.pad(event_k, (0, key_padding))
        event_g = F.pad(event_g, (0, key_padding))
        event_b = F.pad(event_b, (0, key_padding))

    event_sequences = torch.cat((write_sequences, read_sequences))[order]
    event_tokens = torch.cat((write_tokens, read_tokens))[order]
    read_events = torch.cat(
        (
            torch.zeros_like(write_tokens, dtype=torch.bool),
            torch.ones_like(read_tokens, dtype=torch.bool),
        )
    )[order]
    counts = torch.bincount(event_sequences, minlength=batch * bank_count)
    cu_seqlens = F.pad(counts.cumsum(0), (1, 0)).to(torch.long)

    def packed(tensor: Tensor) -> Tensor:
        return tensor.unsqueeze(0).unsqueeze(2).contiguous()

    return PackedBankEvents(
        q=packed(event_q),
        k=packed(event_k),
        v=packed(event_v),
        g=packed(event_g),
        b=packed(event_b),
        w=packed(event_w),
        cu_seqlens=cu_seqlens,
        token_indices=event_tokens,
        sequence_indices=event_sequences,
        read_events=read_events,
    )


def scatter_bank_outputs(
    event_outputs: Tensor,
    token_indices: Tensor,
    *,
    batch: int,
    time: int,
) -> Tensor:
    """Restore packed read outputs to token order by summing routed banks."""

    if (
        event_outputs.ndim != 4
        or event_outputs.shape[0] != 1
        or event_outputs.shape[2] != 1
    ):
        raise ValueError("event outputs must be [1, events, 1, value_width]")
    flat = event_outputs[0, :, 0]
    output = flat.new_zeros(batch * time, flat.shape[-1])
    output.index_add_(0, token_indices, flat)
    return output.reshape(batch, time, flat.shape[-1])


def serial_banked_gdn2(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    initial_state: Tensor,
    write_indices: Tensor,
    write_weights: Tensor,
    read_indices: Tensor,
    read_weights: Tensor,
    *,
    scale: float | None = None,
    routed_decay: bool = False,
) -> tuple[Tensor, Tensor]:
    """Literal bank-level equation oracle with separate read/write routes."""

    batch, time, key_width = q.shape
    if initial_state.ndim != 4 or initial_state.shape[0] != batch:
        raise ValueError("initial state must be [batch,banks,bank_size,value_width]")
    if initial_state.shape[2] != key_width or initial_state.shape[3] != v.shape[-1]:
        raise ValueError("initial state dimensions disagree with the controllers")
    state = initial_state.float().clone()
    outputs: list[Tensor] = []
    scale = key_width**-0.5 if scale is None else scale
    batch_ids = torch.arange(batch, device=q.device)

    for position in range(time):
        for rank in range(write_indices.shape[-1]):
            bank = write_indices[:, position, rank]
            route = write_weights[:, position, rank].float()
            selected = state[batch_ids, bank]
            log_decay = g[:, position].float()
            if routed_decay:
                log_decay = log_decay * route.unsqueeze(-1)
            decayed = selected * log_decay.exp().unsqueeze(-1)
            erase = (
                (b[:, position].float() * route.unsqueeze(-1) * k[:, position].float())
                .unsqueeze(-1)
                .mul(decayed)
                .sum(-2)
            )
            write_value = (
                w[:, position].float() * route.unsqueeze(-1) * v[:, position].float()
                - erase
            )
            updated = decayed + k[:, position].float().unsqueeze(
                -1
            ) * write_value.unsqueeze(-2)
            next_state = state.clone()
            next_state[batch_ids, bank] = updated
            state = next_state

        token_output = v.new_zeros(batch, v.shape[-1], dtype=torch.float32)
        for rank in range(read_indices.shape[-1]):
            bank = read_indices[:, position, rank]
            route = read_weights[:, position, rank].float()
            selected = state[batch_ids, bank]
            reading = q[:, position].float().unsqueeze(-1).mul(selected).sum(-2)
            token_output = token_output + scale * route.unsqueeze(-1) * reading
        outputs.append(token_output)
    return torch.stack(outputs, dim=1).to(v.dtype), state
