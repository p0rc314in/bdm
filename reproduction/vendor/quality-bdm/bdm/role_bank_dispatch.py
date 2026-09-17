"""Causal bank dispatch with one retained permutation and an explicit VJP.

Bank width and read/write selection counts are independent kernel parameters.
Exact bank widths only; scalar controllers use physically scalar packed gates.
Fields unused by an event's role are unspecified. Consumers must honor roles.
"""

from dataclasses import dataclass

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _keys(WI, RI, KEYS, E: tl.constexpr, NT: tl.constexpr, T: tl.constexpr,
          N: tl.constexpr, KW: tl.constexpr, KR: tl.constexpr, X: tl.constexpr):
    e = tl.program_id(0).to(tl.int64) * X + tl.arange(0, X)
    read = e >= NT * KW
    route = tl.where(read, e - NT * KW, e)
    token = tl.where(read, route // KR, route // KW)
    bank = tl.load(WI + route, (e < E) & ~read, 0) + tl.load(RI + route, (e < E) & read, 0)
    sequence = (token // T) * N + bank
    tl.store(KEYS + e, (sequence * T + token % T) * 2 + read, e < E)


@tr.jit
def _bounds(KEYS, CU, E: tl.constexpr, SEQS: tl.constexpr, T: tl.constexpr, IT: tl.constexpr, X: tl.constexpr):
    s = tl.program_id(0) * X + tl.arange(0, X)
    left = tl.full((X,), 0, tl.int64)
    right = tl.full((X,), E, tl.int64)
    for _ in range(IT):
        mid = (left + right) // 2
        key = tl.load(KEYS + mid, mid < E, 0x7fffffffffffffff)
        smaller = key < s.to(tl.int64) * (T * 2)
        left = tl.where(smaller, mid + 1, left)
        right = tl.where(smaller, right, mid)
    tl.store(CU + s, left, s <= SEQS)


@tr.jit
def _pack(Q, K, V, G, B, W, WW, RW, ORDER, INVERSE, TOKENS,
          EQ, EK, EV, EG, EB, EW, E: tl.constexpr, NT: tl.constexpr,
          BK: tl.constexpr, PK: tl.constexpr, D: tl.constexpr,
          GK: tl.constexpr, BC: tl.constexpr, WC: tl.constexpr,
          KW: tl.constexpr, KR: tl.constexpr, C: tl.constexpr, ROLE_LIVE: tl.constexpr, TOKEN_VALUES: tl.constexpr = False, TOKEN_WRITES: tl.constexpr = False):
    e = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    original = tl.load(ORDER + e)
    read = original >= NT * KW
    route = tl.where(read, original - NT * KW, original)
    token = tl.where(read, route // KR, route // KW)
    weight = tl.load(WW + route, ~read, 0).to(tl.float32) + tl.load(RW + route, read, 0).to(tl.float32)
    q = tl.load(Q + token * BK + c, (c < BK) & read, 0).to(tl.float32)
    k = tl.load(K + token * BK + c, (c < BK) & (~read | (not ROLE_LIVE)), 0)
    g = tl.load(G + token * GK + c, (c < GK) & ~read, 0)
    b = tl.load(B + token * BC + c, (c < BC) & ~read, 0).to(tl.float32)
    if not TOKEN_VALUES:
        v = tl.load(V + token * D + c, (c < D) & (~read | (not ROLE_LIVE)), 0)
    if not TOKEN_WRITES:
        w = tl.load(W + token * WC + c, (c < WC) & ~read, 0).to(tl.float32)
    # The role kernel consumes only Q for a read and K/V/G/B/W for a write.
    # Keep the existing layouts, but avoid moving dead fields through HBM.
    tl.store(EQ + e * PK + c, q * weight, (c < PK) & (read | (not ROLE_LIVE)))
    tl.store(EK + e * PK + c, k, (c < PK) & (~read | (not ROLE_LIVE)))
    tl.store(EG + e * GK + c, g, (c < GK) & (~read | (not ROLE_LIVE)))
    tl.store(EB + e * BC + c, b * weight, (c < BC) & (~read | (not ROLE_LIVE)))
    if not TOKEN_VALUES:
        tl.store(EV + e * D + c, v, (c < D) & (~read | (not ROLE_LIVE)))
    if not TOKEN_WRITES:
        tl.store(EW + e * WC + c, w * weight, (c < WC) & (~read | (not ROLE_LIVE)))
    tl.store(INVERSE + original, e)
    tl.store(TOKENS + e, token)


@tr.jit
def _vjp(Q, B, W, WW, RW, INV, DQ, DK, DV, DG, DB, DW,
         OQ, OK, OV, OG, OB, OW, OWW, ORW,
         NT: tl.constexpr, BK: tl.constexpr, PK: tl.constexpr, D: tl.constexpr,
         GK: tl.constexpr, BC: tl.constexpr, WC: tl.constexpr,
         KW: tl.constexpr, KR: tl.constexpr, C: tl.constexpr, ROLE_LIVE: tl.constexpr, WRITE_GRADS_ONLY: tl.constexpr = False):
    token = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    q0 = tl.load(Q + token * BK + c, c < BK, 0)
    b0 = tl.load(B + token * BC + c, c < BC, 0)
    w0 = tl.load(W + token * WC + c, c < WC, 0)
    qsum = tl.full((C,), 0, tl.float32)
    kwsum = tl.full((C,), 0, tl.float32)
    krsum = tl.full((C,), 0, tl.float32)
    vwsum = tl.full((C,), 0, tl.float32)
    vrsum = tl.full((C,), 0, tl.float32)
    gsum = tl.full((C,), 0, tl.float32)
    bsum = tl.full((C,), 0, tl.float32)
    wsum = tl.full((C,), 0, tl.float32)
    for j in range(KW):
        route = token * KW + j
        e = tl.load(INV + route).to(tl.int64)
        weight = tl.load(WW + route).to(tl.float32)
        dk = tl.load(DK + e * PK + c, c < BK, 0).to(q0.dtype).to(tl.float32)
        value_index = route if WRITE_GRADS_ONLY else e
        dv = tl.load(DV + value_index * D + c, c < D, 0).to(w0.dtype).to(tl.float32)
        dg = tl.load(DG + e * GK + c, c < GK, 0).to(tl.float32)
        db = tl.load(DB + e * BC + c, c < BC, 0).to(b0.dtype).to(tl.float32)
        dw = tl.load(DW + value_index * WC + c, c < WC, 0).to(w0.dtype).to(tl.float32)
        kwsum += dk
        # PyTorch's wide gather backward rounds each indexed addition to the
        # source dtype. Preserve that order for the value channels.
        vwsum = (vwsum + dv).to(w0.dtype).to(tl.float32)
        gsum += dg
        bsum += (db * weight).to(b0.dtype).to(tl.float32)
        wsum = (wsum + (dw * weight).to(w0.dtype).to(tl.float32)).to(w0.dtype).to(tl.float32)
        wb = tl.sum((db * b0.to(tl.float32)).to(b0.dtype).to(tl.float32), 0).to(b0.dtype).to(tl.float32)
        wv = tl.sum((dw * w0.to(tl.float32)).to(w0.dtype).to(tl.float32), 0).to(w0.dtype).to(tl.float32)
        tl.store(OWW + route, wb + wv)
    for j in range(KR):
        route = token * KR + j
        e = tl.load(INV + NT * KW + route).to(tl.int64)
        weight = tl.load(RW + route).to(tl.float32)
        dq = tl.load(DQ + e * PK + c, c < BK, 0).to(q0.dtype).to(tl.float32)
        if not ROLE_LIVE:
            krsum += tl.load(DK + e * PK + c, c < BK, 0).to(q0.dtype).to(tl.float32)
            vrsum = (vrsum + tl.load(DV + e * D + c, c < D, 0).to(w0.dtype).to(tl.float32)).to(w0.dtype).to(tl.float32)
        qsum += (dq * weight).to(q0.dtype).to(tl.float32)
        wr = tl.sum((dq * q0.to(tl.float32)).to(q0.dtype).to(tl.float32), 0)
        tl.store(ORW + route, wr)
    tl.store(OQ + token * BK + c, qsum, c < BK)
    # The two original gather branches each round before their gradients add.
    tl.store(OK + token * BK + c, kwsum.to(q0.dtype).to(tl.float32) + krsum.to(q0.dtype).to(tl.float32), c < BK)
    tl.store(OV + token * D + c, vwsum.to(w0.dtype).to(tl.float32) + vrsum.to(w0.dtype).to(tl.float32), c < D)
    tl.store(OG + token * GK + c, gsum, c < GK)
    tl.store(OB + token * BC + c, bsum, c < BC)
    tl.store(OW + token * WC + c, wsum, c < WC)


@dataclass
class Dispatch:
    order: torch.Tensor
    inverse: torch.Tensor
    tokens: torch.Tensor
    cu_seqlens: torch.Tensor


def metadata(wi, ri, bank_count):
    batch, time, kw = wi.shape
    kr = ri.shape[-1]
    nt, e = batch * time, batch * time * (kw + kr)
    keys = torch.empty(e, device=wi.device, dtype=torch.int64)
    _keys[(tr.cdiv(e, 256),)](wi, ri, keys, e, nt, time, bank_count, kw, kr, 256)
    sorted_keys, order = torch.sort(keys)
    cu = torch.empty(batch * bank_count + 1, device=wi.device, dtype=torch.int64)
    _bounds[(tr.cdiv(cu.numel(), 128),)](sorted_keys, cu, e, batch * bank_count, time, (e + 1).bit_length(), 128)
    return Dispatch(order, torch.empty(e, device=wi.device, dtype=torch.int64),
                    torch.empty(e, device=wi.device, dtype=torch.int64), cu)


def pack(q, k, v, g, b, w, ww, rw, dispatch, physical_width, token_values=False, token_writes=False):
    bk, d, nt = q.shape[-1], v.shape[-1], q.shape[0] * q.shape[1]
    e = dispatch.order.numel()
    def output(source, width):
        return torch.empty((1, e, 1, width), device=q.device, dtype=source.dtype)
    # Vector controllers require exact bank widths. Scalar controllers are an
    # explicit direct-kernel mode and retain one physical gate channel.
    scalar = g.shape[-1] == b.shape[-1] == w.shape[-1] == 1
    if not scalar:
        assert g.shape[-1] == b.shape[-1] == bk and w.shape[-1] == d
        assert physical_width == bk, "fused dispatch requires exact bank width"
    gk, bc, wc = g.shape[-1], b.shape[-1], w.shape[-1]
    eq, ek, eg, eb = (output(t, width) for t, width in
                         ((q, physical_width), (k, physical_width),
                          (g, gk), (b, bc)))
    ew = w if token_writes else output(w, wc)
    ev = v if token_values else output(v, d)
    # Role-unused value channels do not depend on the model's value width.
    capability = torch.cuda.get_device_capability(q.device)
    role_live = capability == (12, 0) or (capability == (9, 0) and bk == 8)
    # When wide values and writes stay in token storage, this kernel only moves
    # the bank-width controller fields. Do not launch value-width lane vectors
    # for fields it never loads or stores.
    packed_width = max(physical_width, gk, bc, 1 if token_values else d,
                       1 if token_writes else wc)
    warps = 1 if token_values and token_writes else 4
    _pack[(e,)](q, k, v, g, b, w, ww, rw, dispatch.order, dispatch.inverse, dispatch.tokens,
                eq, ek, ev, eg, eb, ew, e, nt, bk, physical_width, d, gk, bc, wc,
                ww.shape[-1], rw.shape[-1], tr.next_power_of_2(packed_width), role_live, token_values, token_writes, num_warps=warps)
    return eq, ek, ev, eg, eb, ew


def vjp(sources, ww, rw, dispatch, gradients, compact_values=False):
    q, k, v, g, b, w = sources
    outputs = tuple(torch.empty_like(t) for t in (*sources, ww, rw))
    dq, dk, dv, dg, db, dw = gradients
    capability = torch.cuda.get_device_capability(q.device)
    role_live = compact_values or (capability == (9, 0) and q.shape[-1] == 8)
    _vjp[(q.shape[0] * q.shape[1],)](
        q, b, w, ww, rw, dispatch.inverse, dq, dk, dv, dg, db, dw, *outputs,
        q.shape[0] * q.shape[1], q.shape[-1], dq.shape[-1], v.shape[-1], g.shape[-1], b.shape[-1], w.shape[-1], ww.shape[-1], rw.shape[-1],
        tr.next_power_of_2(max(q.shape[-1], v.shape[-1])), role_live, compact_values, num_warps=4, enable_fp_fusion=False)
    return outputs


@tr.jit
def _readout(EVENTS, INVERSE, OUTPUT, NT: tl.constexpr, D: tl.constexpr,
             KW: tl.constexpr, KR: tl.constexpr, RK: tl.constexpr, C: tl.constexpr):
    token, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    routes, values = tl.arange(0, RK), tile * C + tl.arange(0, C)
    positions = tl.load(INVERSE + NT * KW + token * KR + routes, routes < KR, 0x7fffffffffffffff)
    previous = tl.full((), -1, tl.int64)
    total = tl.zeros((C,), tl.float32)
    # Original index_add visits packed events in bank order. Select the same
    # read order; write events contribute exactly zero and need no load.
    for _ in range(KR):
        event = tl.min(tl.where(positions > previous, positions, 0x7fffffffffffffff), 0)
        value = tl.load(EVENTS + event.to(tl.int64) * D + values, values < D, 0)
        total = (total + value.to(tl.float32)).to(value.dtype).to(tl.float32)
        previous = event
    tl.store(OUTPUT + token * D + values, total, values < D)


def readout(event_outputs, inverse, tokens, *, batch, time, write_banks, read_banks):
    """One program owns each token's reduction, so output needs no atomics."""
    width = event_outputs.shape[-1]
    # PyTorch's deterministic indexed sum selects different accumulator/reduction
    # rules at widths <= one CUDA warp. Preserve its narrow-value path exactly.
    if width <= 32:
        from .gdn2_banks import scatter_bank_outputs
        return scatter_bank_outputs(event_outputs, tokens, batch=batch, time=time)
    result = event_outputs.new_empty((batch, time, width))
    _readout[(batch * time, tr.cdiv(width, 128))](
        event_outputs, inverse, result, batch * time, width, write_banks, read_banks,
        tr.next_power_of_2(read_banks), 128, num_warps=4, enable_fp_fusion=False)
    return result
