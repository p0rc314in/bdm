"""Experimental dense bank training with explicit read/write event roles.

This operator's contract is bank dispatch, not arbitrary packed GDN2 inputs.
Write events have no output. Read events do not mutate state. Scalar controls
retain scalar storage and derivatives, while every selected bank row is used.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _load_gates(G, B, W, e, rows, values, BK: tl.constexpr,
                D: tl.constexpr, SCALAR: tl.constexpr, SOURCE_W_KW: tl.constexpr = 0, ORDER = None, WW = None):
    if SCALAR:
        g = tl.full(rows.shape, 0, tl.float32) + tl.load(G + e).to(tl.float32)
        b = tl.full(rows.shape, 0, tl.float32) + tl.load(B + e).to(tl.float32)
        w = tl.full(values.shape, 0, tl.float32) + tl.load(W + e).to(tl.float32)
    else:
        g = tl.load(G + e * BK + rows, rows < BK, 0).to(tl.float32)
        b = tl.load(B + e * BK + rows, rows < BK, 0).to(tl.float32)
        if SOURCE_W_KW:
            original = tl.load(ORDER + e)
            w = tl.load(W + (original // SOURCE_W_KW) * D + values, values < D, 0).to(tl.float32)
            w = (w * tl.load(WW + original).to(tl.float32)).to(W.dtype.element_ty).to(tl.float32)
        else:
            w = tl.load(W + e * D + values, values < D, 0).to(tl.float32)
    return g, b, w


@tr.jit
def _write(K, V, G, B, W, h, e, rows, values,
           BK: tl.constexpr, D: tl.constexpr, SCALAR: tl.constexpr,
           SOURCE_V_KW: tl.constexpr = 0, ORDER = None, SOURCE_W_KW: tl.constexpr = 0, WW = None):
    k = tl.load(K + e * BK + rows, rows < BK, 0).to(tl.float32)
    vi = tl.load(ORDER + e) // SOURCE_V_KW if SOURCE_V_KW else e
    v = tl.load(V + vi * D + values, values < D, 0).to(tl.float32)
    g, b, w = _load_gates(G, B, W, e, rows, values, BK, D, SCALAR, SOURCE_W_KW, ORDER, WW)
    hd = h * tl.exp(g)[:, None]
    a = b * k
    erase = tl.sum(a[:, None] * hd, 0)
    return hd + k[:, None] * (w * v - erase)[None, :]


@tr.jit
def _role_forward(Q, K, V, G, B, W, INIT, CU, COFF, SNAP, OUT, ORDER,
                  WRITE_EVENTS: tl.constexpr, BK: tl.constexpr, D: tl.constexpr,
                  RK: tl.constexpr, VD: tl.constexpr, C: tl.constexpr,
                  SCALE: tl.constexpr, INIT_SEQS: tl.constexpr, SCALAR: tl.constexpr,
                  ROUND_STATE: tl.constexpr = False):
    sequence, tile = tl.program_id(0), tl.program_id(1)
    rows, values = tl.arange(0, RK), tile * VD + tl.arange(0, VD)
    mask = (rows[:, None] < BK) & (values[None, :] < D)
    start, end = tl.load(CU + sequence), tl.load(CU + sequence + 1)
    first = tl.load(COFF + sequence)
    h = tl.load(INIT + (sequence % INIT_SEQS) * BK * D + rows[:, None] * D + values[None, :], mask, 0).to(tl.float32)
    for e in range(start, end):
        if (e - start) % C == 0 and e > start:
            if ROUND_STATE:
                # Continue from exactly the persisted BF16 state. Backward
                # restores this same boundary rather than a rounded surrogate
                # of an unrounded forward trajectory.
                h = h.to(tl.bfloat16).to(tl.float32)
            chunk = first + (e - start) // C - 1
            tl.store(SNAP + chunk * BK * D + rows[:, None] * D + values[None, :], h, mask)
        if tl.load(ORDER + e) < WRITE_EVENTS:
            h = _write(K, V, G, B, W, h, e, rows, values, BK, D, SCALAR)
            result = tl.full((VD,), 0, tl.float32)
        else:
            q = tl.load(Q + e * BK + rows, rows < BK, 0).to(tl.float32)
            result = tl.sum(q[:, None] * h, 0) * SCALE
        tl.store(OUT + e * D + values, result, values < D)


@tr.jit
def _role_backward(Q, K, V, G, B, W, INIT, CU, COFF, SNAP, DO, ORDER,
                   DQ, DK, DV, DG, DB, DW, DH,
                   WRITE_EVENTS: tl.constexpr, E: tl.constexpr,
                   BK: tl.constexpr, D: tl.constexpr, RK: tl.constexpr,
                   VD: tl.constexpr, C: tl.constexpr, SCALE: tl.constexpr,
                   INIT_SEQS: tl.constexpr, SCALAR: tl.constexpr):
    sequence, tile = tl.program_id(0), tl.program_id(1)
    rows, values = tl.arange(0, RK), tile * VD + tl.arange(0, VD)
    mask = (rows[:, None] < BK) & (values[None, :] < D)
    start, end = tl.load(CU + sequence), tl.load(CU + sequence + 1)
    first = tl.load(COFF + sequence)
    count = (end - start + C - 1) // C
    dh = tl.zeros((RK, VD), tl.float32)
    for reverse_chunk in range(count):
        chunk = count - 1 - reverse_chunk
        base = start + chunk * C
        length = tl.minimum(C, end - base)
        if chunk == 0:
            h = tl.load(INIT + (sequence % INIT_SEQS) * BK * D + rows[:, None] * D + values[None, :], mask, 0).to(tl.float32)
        else:
            h = tl.load(SNAP + (first + chunk - 1) * BK * D + rows[:, None] * D + values[None, :], mask, 0).to(tl.float32)
        history = ()
        # Keep each checkpoint-local state as its own register tensor. The
        # bounded static loop selects a tuple member at compile time, so no
        # history-axis reduction or gather is needed in the reverse pass.
        for j in tl.static_range(C):
            history += (h,)
            if j < length:
                if tl.load(ORDER + base + j) < WRITE_EVENTS:
                    h = _write(K, V, G, B, W, h, base + j, rows, values, BK, D, SCALAR)
        for history_index in tl.static_range(C - 1, -1, -1):
            if history_index < length:
                e = base + history_index
                h0 = history[history_index]
                dq = tl.zeros((RK,), tl.float32)
                dk = tl.zeros((RK,), tl.float32)
                dg = tl.zeros((RK,), tl.float32)
                db = tl.zeros((RK,), tl.float32)
                dv = tl.zeros((VD,), tl.float32)
                dw = tl.zeros((VD,), tl.float32)
                if tl.load(ORDER + e) < WRITE_EVENTS:
                    k = tl.load(K + e * BK + rows, rows < BK, 0).to(tl.float32)
                    v = tl.load(V + e * D + values, values < D, 0).to(tl.float32)
                    g, b, w = _load_gates(G, B, W, e, rows, values, BK, D, SCALAR)
                    decay = tl.exp(g)
                    hd = h0 * decay[:, None]
                    a = b * k
                    erase = tl.sum(a[:, None] * hd, 0)
                    u = w * v - erase
                    du = tl.sum(dh * k[:, None], 0)
                    contraction = tl.sum(hd * du[None, :], 1)
                    dprime = dh - a[:, None] * du[None, :]
                    dk = tl.sum(dh * u[None, :], 1) - b * contraction
                    db = -k * contraction
                    dg = tl.sum(dprime * hd, 1)
                    dv, dw = du * w, du * v
                    dh = dprime * decay[:, None]
                else:
                    q = tl.load(Q + e * BK + rows, rows < BK, 0).to(tl.float32)
                    do = tl.load(DO + e * D + values, values < D, 0).to(tl.float32)
                    dq = tl.sum(h0 * do[None, :], 1) * SCALE
                    dh = dh + q[:, None] * do[None, :] * SCALE
                tl.store(DQ + (tile * E + e) * BK + rows, dq, rows < BK)
                tl.store(DK + (tile * E + e) * BK + rows, dk, rows < BK)
                tl.store(DV + e * D + values, dv, values < D)
                if SCALAR:
                    tl.store(DG + tile * E + e, tl.sum(tl.where(rows < BK, dg, 0), 0))
                    tl.store(DB + tile * E + e, tl.sum(tl.where(rows < BK, db, 0), 0))
                    tl.store(DW + tile * E + e, tl.sum(tl.where(values < D, dw, 0), 0))
                else:
                    tl.store(DG + (tile * E + e) * BK + rows, dg, rows < BK)
                    tl.store(DB + (tile * E + e) * BK + rows, db, rows < BK)
                    tl.store(DW + e * D + values, dw, values < D)
    tl.store(DH + sequence * BK * D + rows[:, None] * D + values[None, :], dh, mask)


def _replay_settings(bank_size, value_width, dtype, capability, compact):
    chunk = min(16, max(4, 128 // tr.next_power_of_2(bank_size)))
    if dtype != torch.bfloat16 or compact is False:
        return chunk, torch.float32
    if compact is None:
        # The full-model speed/memory tradeoff wins clearly for A100 B4/B8
        # and Hopper B4. Retain the existing path for the other geometries;
        # callers can explicitly choose either policy for their own workload.
        compact = value_width >= 64 and (
            (capability == (8, 0) and bank_size in (4, 8))
            or (capability == (9, 0) and bank_size == 4)
        )
    return (min(8, chunk), torch.bfloat16) if compact else (chunk, torch.float32)


def _hopper_training_supported(q, value_width, compact):
    # The row-register forward represents eight bank rows. Value channels are
    # independently tiled and masked; they impose no exact-width requirement.
    # Explicit BF16 replay remains a separate numerical execution policy.
    return (q.is_cuda and q.shape[-1] == 8 and value_width > 0
            and compact is not True and q.dtype in (torch.bfloat16, torch.float32)
            and torch.cuda.get_device_capability(q.device) == (9, 0))


def forward(packed, initial, cu, order, write_events, scale, compact_replay=None, state_dtype=None):
    q, k, v, g, b, w = packed
    bk, d = q.shape[-1], v.shape[-1]
    scalar = g.shape[-1] == b.shape[-1] == w.shape[-1] == 1
    c, snapshot_dtype = _replay_settings(
        bk, d, q.dtype, torch.cuda.get_device_capability(q.device), compact_replay)
    if state_dtype is not None:
        snapshot_dtype = state_dtype
    counts = ((cu[1:] - cu[:-1] - 1) // c).clamp_min(0)
    offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0))
    # The legacy compact_replay option compresses saved backward checkpoints.
    # Explicit state_dtype instead rounds the forward trajectory at persisted
    # boundaries too, so replay starts from the exact same stored values.
    # A narrow checkpoint store changes Triton's reduction layout for wider
    # banks, even before the first checkpoint. Keep the FP32 forward kernel
    # specialization there, then compress its saved output in a separate pass.
    forward_snapshot_dtype = torch.float32 if bk > 8 else snapshot_dtype
    snap = torch.empty((int(offsets[-1]), bk, d), device=q.device, dtype=forward_snapshot_dtype)
    out = torch.empty_like(v)
    vd = min(64, tr.next_power_of_2(d))
    _role_forward[(cu.numel() - 1, tr.cdiv(d, vd))](
        *packed, initial, cu, offsets, snap, out, order, write_events, bk, d,
        tr.next_power_of_2(bk), vd, c, scale, initial.shape[0], scalar,
        ROUND_STATE=state_dtype == torch.bfloat16,
        num_warps=4, enable_fp_fusion=False)
    return out, (snap.to(snapshot_dtype), offsets, c, vd)


def backward(packed, initial, cu, order, write_events, scale, saved, do):
    q, k, v, g, b, w = packed
    snap, offsets, c, vd = saved
    e, bk, d = q.shape[1], q.shape[-1], v.shape[-1]
    if bk > 8:
        # Retain compact checkpoints between layers, but give this layer's
        # backward the original FP32 pointer/layout specialization. Its
        # temporary expansion lives only for this backward invocation.
        snap = snap.float()
    scalar = g.shape[-1] == b.shape[-1] == w.shape[-1] == 1
    tiles = tr.cdiv(d, vd)
    partials = [torch.empty((tiles, e, t.shape[-1]), device=q.device, dtype=torch.float32) for t in (q, k, g, b)]
    dq, dk, dg, db = partials
    dv = torch.empty_like(v)
    dw = torch.empty((tiles, e, 1), device=q.device, dtype=torch.float32) if scalar else torch.empty_like(w)
    dh = torch.empty((cu.numel() - 1, 1, bk, d), device=q.device, dtype=torch.float32)
    _role_backward[(cu.numel() - 1, tiles)](
        *packed, initial, cu, offsets, snap, do, order, dq, dk, dv, dg, db, dw, dh,
        write_events, e, bk, d, tr.next_power_of_2(bk), vd, c, scale, initial.shape[0], scalar,
        num_warps=4, enable_fp_fusion=False)
    dq, dk, dg, db = (p.sum(0).unsqueeze(0).unsqueeze(2).to(t.dtype) for p, t in zip(partials, (q, k, g, b)))
    if scalar:
        dw = dw.sum(0).unsqueeze(0).unsqueeze(2).to(w.dtype)
    return dq, dk, dv, dg, db, dw, dh


class RoleBankTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, b, w, wi, ww, ri, rw, state, bank_count, scale, compact_replay, state_dtype):
        from . import role_bank_dispatch as dispatch
        meta = dispatch.metadata(wi, ri, bank_count)
        packed = dispatch.pack(q, k, v, g, b, w, ww, rw, meta, state.shape[1])
        output, (snap, offsets, c, vd) = forward(
            packed, state.to(state_dtype), meta.cu_seqlens, meta.order, wi.numel(), scale, compact_replay, state_dtype=state_dtype)
        ctx.save_for_backward(q, k, v, g, b, w, ww, rw, state, meta.order, meta.inverse, meta.tokens, meta.cu_seqlens, snap, offsets)
        ctx.options = (scale, c, vd, wi.numel(), state_dtype)
        return dispatch.readout(output, meta.inverse, meta.tokens, batch=q.shape[0], time=q.shape[1],
                                write_banks=wi.shape[-1], read_banks=ri.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        from . import role_bank_dispatch as dispatch
        q, k, v, g, b, w, ww, rw, state, order, inverse, tokens, cu, snap, offsets = ctx.saved_tensors
        scale, c, vd, write_events, state_dtype = ctx.options
        meta = dispatch.Dispatch(order, inverse, tokens, cu)
        packed = dispatch.pack(q, k, v, g, b, w, ww, rw, meta, state.shape[1])
        do = grad_output.reshape(-1, grad_output.shape[-1])[tokens].unsqueeze(0).unsqueeze(2).contiguous()
        grads = backward(packed, state.to(state_dtype), cu, order, write_events, scale, (snap, offsets, c, vd), do)
        dq, dk, dv, dg, db, dw, dww, drw = dispatch.vjp((q, k, v, g, b, w), ww, rw, meta, grads[:6])
        ds = grads[6].to(state.dtype).reshape(q.shape[0], *state.shape).sum(0)
        return dq, dk, dv, dg, db, dw, None, dww, None, drw, ds, None, None, None, None


def role_bank_training(*args, compact_replay=None, state_dtype=torch.float32):
    return RoleBankTraining.apply(*args, compact_replay, state_dtype)
