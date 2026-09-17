"""Hopper bank-8 training with interval-64 FP32 checkpoints.

Decay exponentials are evaluated once per dispatch. Wide-value backward uses
hierarchical replay to limit repeated state updates. Nonreentrant block
checkpoint recomputation produces the same transient snapshots used by backward;
ordinary training can retain snapshots for backward. When whole blocks are
checkpointed, PyTorch's saved-tensor hooks discard the original snapshots and
retain those produced by recomputation instead.
"""

import torch
import triton as tr
import triton.language as tl


from .role_bank_kernel import _load_gates

@tr.jit
def _write(K, V, G, B, W, h, e, rows, values,
           BK: tl.constexpr, D: tl.constexpr, SCALAR: tl.constexpr,
           SOURCE_V_KW: tl.constexpr = 0, ORDER = None, SOURCE_W_KW: tl.constexpr = 0, WW = None):
    k = tl.load(K + e * BK + rows, rows < BK, 0).to(tl.float32)
    vi = tl.load(ORDER + e) // SOURCE_V_KW if SOURCE_V_KW else e
    v = tl.load(V + vi * D + values, values < D, 0).to(tl.float32)
    g, b, w = _load_gates(G, B, W, e, rows, values, BK, D, SCALAR, SOURCE_W_KW, ORDER, WW)
    hd = h * g[:, None]
    a = b * k
    erase = tl.sum(a[:, None] * hd, 0)
    return hd + k[:, None] * (w * v - erase)[None, :]




@tr.jit
def _dh_boundaries(Q, K, G, B, CU, COFF, DO, ORDER, TOKENS, DHBOUND,
                   WRITE_EVENTS: tl.constexpr, BK: tl.constexpr, D: tl.constexpr,
                   RK: tl.constexpr, VD: tl.constexpr, C: tl.constexpr,
                   SCALE: tl.constexpr, SCALAR: tl.constexpr, TOKEN_DO: tl.constexpr,
                   FLAT_GRID: tl.constexpr = False):
    if FLAT_GRID:
        tile = tl.program_id(0) % tl.cdiv(D, VD)
        sequence = (tl.program_id(0) // tl.cdiv(D, VD)).to(tl.int64)
    else:
        tile, sequence = tl.program_id(0), tl.program_id(1)
    rows, values = tl.arange(0, RK), tile * VD + tl.arange(0, VD)
    mask = (rows[:, None] < BK) & (values[None, :] < D)
    start, end = tl.load(CU + sequence), tl.load(CU + sequence + 1)
    first = tl.load(COFF + sequence)
    dh = tl.zeros((RK, VD), tl.float32)
    if start == end:
        tl.store(DHBOUND + first * BK * D + rows[:, None] * D + values[None, :], dh, mask)
    for e in range(end - 1, start - 1, -1):
        if e == end - 1 or (e - start + 1) % C == 0:
            chunk = (e - start) // C
            tl.store(DHBOUND + (first + chunk) * BK * D + rows[:, None] * D + values[None, :], dh, mask)
        if tl.load(ORDER + e) < WRITE_EVENTS:
            k = tl.load(K + e * BK + rows, rows < BK, 0).to(tl.float32)
            if SCALAR:
                g = tl.full((RK,), 0, tl.float32) + tl.load(G + e).to(tl.float32)
                b = tl.full((RK,), 0, tl.float32) + tl.load(B + e).to(tl.float32)
            else:
                g = tl.load(G + e * BK + rows, rows < BK, 0).to(tl.float32)
                b = tl.load(B + e * BK + rows, rows < BK, 0).to(tl.float32)
            a = b * k
            du = tl.sum(dh * k[:, None], 0)
            dprime = dh - a[:, None] * du[None, :]
            dh = dprime * g[:, None]
        else:
            q = tl.load(Q + e * BK + rows, rows < BK, 0).to(tl.float32)
            do_index = tl.load(TOKENS + e) if TOKEN_DO else e
            do = tl.load(DO + do_index * D + values, values < D, 0).to(tl.float32)
            dh = dh + q[:, None] * do[None, :] * SCALE

@tr.jit
def _event_vjp(e, h0, dh, rows, values, tile, Q, K, V, G, B, W, DO, ORDER, TOKENS, DQ, DK, DV, DG, DB, DW, WW, WRITE_EVENTS: tl.constexpr, E: tl.constexpr, BK: tl.constexpr, D: tl.constexpr, RK: tl.constexpr, VD: tl.constexpr, SCALE: tl.constexpr, SCALAR: tl.constexpr, TOKEN_DO: tl.constexpr, SOURCE_V_KW: tl.constexpr, SOURCE_W_KW: tl.constexpr, WRITE_GRADS_ONLY: tl.constexpr, IS_WRITE: tl.constexpr):
    dq = tl.zeros((RK,), tl.float32)
    dk = tl.zeros((RK,), tl.float32)
    dg = tl.zeros((RK,), tl.float32)
    db = tl.zeros((RK,), tl.float32)
    dv = tl.zeros((VD,), tl.float32)
    dw = tl.zeros((VD,), tl.float32)
    if IS_WRITE:
        k = tl.load(K + e * BK + rows, rows < BK, 0).to(tl.float32)
        vi = tl.load(ORDER + e) // SOURCE_V_KW if SOURCE_V_KW else e
        v = tl.load(V + vi * D + values, values < D, 0).to(tl.float32)
        g, b, w = _load_gates(G, B, W, e, rows, values, BK, D, SCALAR, SOURCE_W_KW, ORDER, WW)
        decay = g
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
        do_index = tl.load(TOKENS + e) if TOKEN_DO else e
        do = tl.load(DO + do_index * D + values, values < D, 0).to(tl.float32)
        dq = tl.sum(h0 * do[None, :], 1) * SCALE
        dh = dh + q[:, None] * do[None, :] * SCALE
    tl.store(DQ + (tile * E + e) * BK + rows, dq, rows < BK)
    tl.store(DK + (tile * E + e) * BK + rows, dk, rows < BK)
    value_index = tl.load(ORDER + e) if WRITE_GRADS_ONLY else e
    write_live = IS_WRITE if WRITE_GRADS_ONLY else True
    tl.store(DV + value_index * D + values, dv, (values < D) & write_live)
    if SCALAR:
        tl.store(DG + tile * E + e, tl.sum(tl.where(rows < BK, dg, 0), 0))
        tl.store(DB + tile * E + e, tl.sum(tl.where(rows < BK, db, 0), 0))
        tl.store(DW + tile * E + e, tl.sum(tl.where(values < D, dw, 0), 0))
    else:
        tl.store(DG + (tile * E + e) * BK + rows, dg, rows < BK)
        tl.store(DB + (tile * E + e) * BK + rows, db, rows < BK)
        tl.store(DW + value_index * D + values, dw, (values < D) & write_live)
    return dh


@tr.jit
def _role_backward(Q, K, V, G, B, W, INIT, CU, COFF, CHSEQ, CHOFF, DHBOUND, SNAP, DO, ORDER, TOKENS,
                   DQ, DK, DV, DG, DB, DW, DH,
                   WRITE_EVENTS: tl.constexpr, E: tl.constexpr,
                   BK: tl.constexpr, D: tl.constexpr, RK: tl.constexpr,
                   VD: tl.constexpr, C: tl.constexpr, BC: tl.constexpr, SCALE: tl.constexpr,
                   INIT_SEQS: tl.constexpr, SCALAR: tl.constexpr, NSEQ: tl.constexpr, TOKEN_DO: tl.constexpr, SOURCE_V_KW: tl.constexpr, SOURCE_W_KW: tl.constexpr, WW, WRITE_GRADS_ONLY: tl.constexpr, HISTORY: tl.constexpr):
    tile, cid = tl.program_id(0) % tl.cdiv(D, VD), tl.program_id(0) // tl.cdiv(D, VD)
    # A single temporary FP32 adjoint buffer crosses 2**31 elements at T32K.
    # Widen before multiplying by the bank-state stride.
    cid = cid.to(tl.int64)
    if cid < tl.load(CHOFF + NSEQ):
        sequence = tl.load(CHSEQ + cid)
        rows, values = tl.arange(0, RK), tile * VD + tl.arange(0, VD)
        mask = (rows[:, None] < BK) & (values[None, :] < D)
        start, end = tl.load(CU + sequence), tl.load(CU + sequence + 1)
        first = tl.load(COFF + sequence)
        dh = tl.load(DHBOUND + cid * BK * D + rows[:, None] * D + values[None, :], mask, 0)
        chunk = cid - tl.load(CHOFF + sequence)
        for reverse_group in range(tl.cdiv(BC, HISTORY)):
            base = start + chunk * BC + (tl.cdiv(BC, HISTORY) - 1 - reverse_group) * HISTORY
            length = tl.minimum(HISTORY, end - base)
            if base < end:
                state_chunk = (base - start) // C
                if state_chunk == 0:
                    h = tl.load(INIT + (sequence % INIT_SEQS) * BK * D + rows[:, None] * D + values[None, :], mask, 0).to(tl.float32)
                else:
                    h = tl.load(SNAP + (first + state_chunk - 1) * BK * D + rows[:, None] * D + values[None, :], mask, 0).to(tl.float32)
                for e in range(start + state_chunk * C, base):
                    if tl.load(ORDER + e) < WRITE_EVENTS:
                        h = _write(K, V, G, B, W, h, e, rows, values, BK, D, SCALAR, SOURCE_V_KW, ORDER, SOURCE_W_KW, WW)
                history = ()
                # Keep each checkpoint-local state as its own register tensor. The
                # bounded static loop selects a tuple member at compile time, so no
                # history-axis reduction or gather is needed in the reverse pass.
                for j in tl.static_range(HISTORY):
                    history += (h,)
                    if j < length:
                        if tl.load(ORDER + base + j) < WRITE_EVENTS:
                            h = _write(K, V, G, B, W, h, base + j, rows, values, BK, D, SCALAR, SOURCE_V_KW, ORDER, SOURCE_W_KW, WW)
                for history_index in tl.static_range(HISTORY - 1, -1, -1):
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
                            vi = tl.load(ORDER + e) // SOURCE_V_KW if SOURCE_V_KW else e
                            v = tl.load(V + vi * D + values, values < D, 0).to(tl.float32)
                            g, b, w = _load_gates(G, B, W, e, rows, values, BK, D, SCALAR, SOURCE_W_KW, ORDER, WW)
                            decay = g
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
                            do_index = tl.load(TOKENS + e) if TOKEN_DO else e
                            do = tl.load(DO + do_index * D + values, values < D, 0).to(tl.float32)
                            dq = tl.sum(h0 * do[None, :], 1) * SCALE
                            dh = dh + q[:, None] * do[None, :] * SCALE
                        tl.store(DQ + (tile * E + e) * BK + rows, dq, rows < BK)
                        tl.store(DK + (tile * E + e) * BK + rows, dk, rows < BK)
                        value_index = tl.load(ORDER + e) if WRITE_GRADS_ONLY else e
                        write_live = value_index < WRITE_EVENTS if WRITE_GRADS_ONLY else True
                        tl.store(DV + value_index * D + values, dv, (values < D) & write_live)
                        if SCALAR:
                            tl.store(DG + tile * E + e, tl.sum(tl.where(rows < BK, dg, 0), 0))
                            tl.store(DB + tile * E + e, tl.sum(tl.where(rows < BK, db, 0), 0))
                            tl.store(DW + tile * E + e, tl.sum(tl.where(values < D, dw, 0), 0))
                        else:
                            tl.store(DG + (tile * E + e) * BK + rows, dg, rows < BK)
                            tl.store(DB + (tile * E + e) * BK + rows, db, rows < BK)
                            tl.store(DW + value_index * D + values, dw, (values < D) & write_live)
        if chunk == 0:
            tl.store(DH + sequence * BK * D + rows[:, None] * D + values[None, :], dh, mask)


@tr.jit
def _window_sequences(OFFSETS, SEQUENCES, X: tl.constexpr):
    sequence = tl.program_id(0)
    first, end = tl.load(OFFSETS + sequence), tl.load(OFFSETS + sequence + 1)
    for base in range(first, end, X):
        i = base + tl.arange(0, X)
        tl.store(SEQUENCES + i, sequence, i < end)


@tr.jit
def _exp_gates(G, OUT, N: tl.constexpr, X: tl.constexpr):
    i=tl.program_id(0)*X+tl.arange(0,X)
    v=tl.load(G+i,i<N,0).to(tl.float32)
    tl.store(OUT+i,tl.exp(v),i<N)

def exp_gates(g):
    out=torch.empty_like(g,dtype=torch.float32)
    _exp_gates[(tr.cdiv(g.numel(),1024),)](g,out,g.numel(),1024,enable_fp_fusion=False)
    return out


@torch.compiler.disable
def _inside_backward():
    # Only inspected after the existing forward graph break. Nonreentrant
    # checkpoint replay runs inside the enclosing autograd graph task.
    return torch._C._current_graph_task_id() >= 0

def forward(packed, initial, cu, order, write_events, scale, compact_replay=None, source_v_kw=0, source_w_kw=0, write_weights=None, retain_snapshots=False, state_dtype=torch.float32):
    q, k, v, g, b, w = packed
    g = exp_gates(g)
    packed = (q,k,v,g,b,w)
    bk, d = q.shape[-1], v.shape[-1]
    vd = min(1024, tr.next_power_of_2(d))
    # Preserve the existing forward graph partition: merging across it changes
    # BF16 fusion and the subsequent discrete router decisions. Backward is
    # compiled separately, without device-to-host allocation dependencies.
    torch._dynamo.graph_break()
    reuse_replay = retain_snapshots or _inside_backward()
    c = 64
    if reuse_replay:
        counts = ((cu[1:] - cu[:-1] - 1) // c).clamp_min(0)
        offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0))
        snap = torch.empty((q.shape[1] // c, bk, d), device=q.device, dtype=state_dtype)
    else:
        # Match checkpoint metadata while retaining only one scalar in normal forward.
        snap = torch.empty((), device=q.device, dtype=state_dtype).expand(q.shape[1] // c, bk, d)
        offsets = torch.empty((), device=cu.device, dtype=cu.dtype).expand(cu.numel())
    out = torch.empty((1, q.shape[1], 1, d), device=v.device, dtype=v.dtype)
    from .forward_rows import forward_rows_precomputed as forward_rows
    # CUDA grid.y is limited to 65535. Flatten only larger bank/batch grids;
    # smaller validated shapes retain their original launch and arithmetic.
    flat_grid = cu.numel() - 1 > 65535
    grid = (tr.cdiv(d, vd) * (cu.numel() - 1),) if flat_grid else (tr.cdiv(d, vd), cu.numel() - 1)
    forward_rows[grid](
        *packed, initial, cu, offsets, snap, out, order, write_events, d, vd, c,
        scale, initial.shape[0], STORE_OUTPUT=True, STORE_CHECKPOINTS=reuse_replay, SOURCE_V_KW=source_v_kw, SOURCE_W_KW=source_w_kw, WW=write_weights,
        ROUND_STATE=state_dtype == torch.bfloat16,
        FLAT_GRID=flat_grid, num_warps=4, enable_fp_fusion=False)
    return out, (snap, offsets, c, vd)


@torch.compile
def backward(packed, initial, cu, order, write_events, scale, saved, do, tokens=None, source_v_kw=0, source_w_kw=0, write_weights=None, compact_grads=False):
    q, k, v, g, b, w = packed
    g = exp_gates(g)
    packed = (q,k,v,g,b,w)
    snap, offsets, c, vd = saved
    e, bk, d = q.shape[1], q.shape[-1], v.shape[-1]
    forward_vd = vd
    vd = min(512, tr.next_power_of_2(d))
    bc = 64
    c = bc
    history_group = 3 if vd == 512 else 4
    flat_grid = cu.numel() - 1 > 65535
    if snap.stride(0) == 0:
        counts = ((cu[1:] - cu[:-1] - 1) // c).clamp_min(0)
        offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0))
        snap = torch.empty((e // bc, bk, d), device=q.device, dtype=initial.dtype)
        from .forward_rows import forward_rows_precomputed as forward_rows
        flat_grid = cu.numel() - 1 > 65535
        grid = (tr.cdiv(d, forward_vd) * (cu.numel() - 1),) if flat_grid else (tr.cdiv(d, forward_vd), cu.numel() - 1)
        forward_rows[grid](
            *packed, initial, cu, offsets, snap, v, order, write_events, d, forward_vd, c,
            scale, initial.shape[0], STORE_OUTPUT=False, STORE_CHECKPOINTS=True, SOURCE_V_KW=source_v_kw, SOURCE_W_KW=source_w_kw, WW=write_weights,
            ROUND_STATE=initial.dtype == torch.bfloat16,
            FLAT_GRID=flat_grid, num_warps=4, enable_fp_fusion=False)
    scalar = g.shape[-1] == b.shape[-1] == w.shape[-1] == 1
    tiles = tr.cdiv(d, vd)
    partials = [torch.empty((tiles, e, t.shape[-1]), device=q.device, dtype=torch.float32) for t in (q, k, g, b)]
    dq, dk, dg, db = partials
    grad_events = write_events if compact_grads else e
    dv = torch.empty((1, grad_events, 1, d), device=v.device, dtype=v.dtype)
    dw = torch.empty((tiles, e, 1), device=q.device, dtype=torch.float32) if scalar else torch.empty((1, grad_events, 1, d), device=w.device, dtype=w.dtype)
    dh = torch.empty((cu.numel() - 1, 1, bk, d), device=q.device, dtype=torch.float32)
    chunk_counts = ((cu[1:] - cu[:-1] - 1) // bc).clamp_min(0) + 1
    chunk_offsets = torch.nn.functional.pad(chunk_counts.cumsum(0), (1, 0))
    # Sum ceil(length / bc), counting empty sequences, is at most E/bc + S.
    capacity = e // bc + cu.numel() - 1
    chunk_sequences = torch.empty((capacity,), device=cu.device, dtype=cu.dtype)
    _window_sequences[(cu.numel() - 1,)](chunk_offsets, chunk_sequences, 128)
    dhbound = torch.empty((chunk_sequences.numel(), bk, d), device=q.device, dtype=torch.float32)
    grid = (tiles * (cu.numel() - 1),) if flat_grid else (tiles, cu.numel() - 1)
    _dh_boundaries[grid](
        q, k, g, b, cu, chunk_offsets, do, order, tokens if tokens is not None else order, dhbound,
        write_events, bk, d, tr.next_power_of_2(bk), vd, bc, scale, scalar, tokens is not None,
        FLAT_GRID=flat_grid, num_warps=2, enable_fp_fusion=False)
    from .training_hopper_replay import backward_tree
    backward_kernel = backward_tree
    backward_kernel[(tiles * chunk_sequences.numel(),)](
        *packed, initial, cu, offsets, chunk_sequences, chunk_offsets, dhbound, snap, do, order, tokens if tokens is not None else order, dq, dk, dv, dg, db, dw, dh,
        write_events, e, bk, d, tr.next_power_of_2(bk), vd, c, bc, scale, initial.shape[0], scalar, cu.numel() - 1, tokens is not None, source_v_kw, source_w_kw, write_weights, compact_grads, history_group,
        num_warps=4 if vd == 512 else 2, maxnreg=256 if vd == 512 else 128, enable_fp_fusion=False)
    dq, dk, dg, db = (p.sum(0).unsqueeze(0).unsqueeze(2).to(t.dtype) for p, t in zip(partials, (q, k, g, b)))
    if scalar:
        dw = dw.sum(0).unsqueeze(0).unsqueeze(2).to(w.dtype)
    return dq, dk, dv, dg, db, dw, dh
