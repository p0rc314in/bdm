"""Independent GDN2 heads inside every routed bank.

bank_size is the key width of one head. value_width is the concatenated value
width. All heads share the same outer bank selections, not recurrent state.
H=1 uses the original controller and dispatch without reshaping or extra ops.
"""
import torch
from torch.nn import functional as F


def controllers(m, hidden):
    shape = (*hidden.shape[:-1], m.num_heads)
    q0 = F.silu(m.q_proj(hidden)).reshape(*shape, m.bank_size)
    k0 = F.silu(m.k_proj(hidden)).reshape(*shape, m.bank_size)
    q = m._normalize_qk(q0)
    k = m._normalize_qk(k0)
    v = F.silu(m.v_proj(hidden)).reshape(*shape, m.head_value_width)
    g = -m.A_log.float().exp()[:, None] * F.softplus(
        (m.f_proj(hidden).float() + m.dt_bias.float()).reshape(*shape, m.bank_size))
    b = m.b_proj(hidden).sigmoid().reshape(*shape, m.bank_size)
    w = m.w_proj(hidden).sigmoid().reshape(*shape, m.head_value_width)
    return q, k, v, g, b, w


def forward_heads(m, hidden, *, training_function=None):
    """Reuse the existing bank kernel once per head; no new recurrence kernel."""
    from .gdn2_banks import pack_bank_events, reference_packed_gdn2, scatter_bank_outputs
    tensors = m._controllers(hidden)
    rr, wr = m._route_projections(hidden)
    ww, wi = m._route(wr, m.num_writes)
    rw, ri = m._route(rr, m.num_reads)
    initial = m.materialize_initial_state()
    reads = []
    for head in range(m.num_heads):
        q, k, v, g, b, w = (x[..., head, :].contiguous() for x in tensors)
        state = initial[:, head].contiguous()
        if training_function is not None:
            raw = training_function(q, k, v, g, b, w, wi, ww, ri, rw, state,
                m.bank_count, m.bank_size ** -.5, None, None, compact_replay=m.compact_replay)
        elif m.backend == 'role':
            from .role_bank_kernel import role_bank_training
            if m.config.routed_decay:
                from .routed_update import bank_training as role_bank_training
            raw = role_bank_training(q, k, v, g, b, w, wi, ww, ri, rw, state,
                m.bank_count, m.bank_size ** -.5, compact_replay=m.compact_replay,
                state_dtype=getattr(torch, m.config.training_state_dtype))
        else:
            events = pack_bank_events(q, k, v, g, b, w, wi, ww, ri, rw,
                bank_count=m.bank_count, kernel_key_width=m.bank_size,
                routed_decay=m.config.routed_decay)
            expanded = state.unsqueeze(0).expand(hidden.shape[0], -1, -1, -1).reshape(
                hidden.shape[0] * m.bank_count, 1, m.bank_size, m.head_value_width).contiguous().float()
            kwargs = dict(q=events.q, k=events.k, v=events.v, g=events.g, b=events.b, w=events.w,
                scale=m.bank_size ** -.5, initial_state=expanded, cu_seqlens=events.cu_seqlens)
            if m.backend == 'reference':
                values = reference_packed_gdn2(**kwargs,
                    state_dtype=getattr(torch, m.config.training_state_dtype),
                    state_interval=min(16, max(4, 128 // (1 << (m.bank_size - 1).bit_length()))))
            else:
                if m.config.training_state_dtype != 'float32':
                    raise NotImplementedError('The FLA backend does not implement BF16 training-state storage')
                from fla.ops.gdn2 import chunk_gdn2
                values, _ = chunk_gdn2(**kwargs, output_final_state=False, use_qk_l2norm_in_kernel=False)
            raw = scatter_bank_outputs(values, events.token_indices, batch=hidden.shape[0], time=hidden.shape[1])
        reads.append(raw)
    return m._project_bank_output(hidden, torch.cat(reads, dim=-1))
