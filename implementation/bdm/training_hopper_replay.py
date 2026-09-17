"""Hierarchical register replay between unchanged interval-64 checkpoints."""
import triton as tr
import triton.language as tl
from .training_hopper import _write, _event_vjp

@tr.jit
def _load_replay_state(pointer, mask, layout):
    if pointer.dtype.element_ty == tl.bfloat16:
        # Convert at the load boundary. Keep replay's register layout FP32;
        # a native BF16 tensor load makes Triton choose a wider per-thread
        # layout, which spills the live replay states on Hopper. The existing
        # FP32 adjoint is a layout operand only: it anchors this elementwise
        # load to the replay layout, and is not used in the PTX arithmetic.
        return tl.inline_asm_elementwise(
            "{ .reg .b16 v; .reg .pred p; setp.ne.u32 p, $2, 0; "
            "mov.b16 v, 0; @p ld.global.b16 v, [$1]; cvt.f32.bf16 $0, v; }",
            constraints="=f,l,r,f", args=[pointer.to(tl.uint64), mask.to(tl.int32), layout],
            dtype=tl.float32, is_pure=True, pack=1,
        )
    return tl.load(pointer, mask, 0).to(tl.float32)

@tr.jit
def backward_tree(Q, K, V, G, B, W, INIT, CU, COFF, CHSEQ, CHOFF, DHBOUND, SNAP, DO, ORDER, TOKENS,
                   DQ, DK, DV, DG, DB, DW, DH,
                   WRITE_EVENTS: tl.constexpr, E: tl.constexpr,
                   BK: tl.constexpr, D: tl.constexpr, RK: tl.constexpr,
                   VD: tl.constexpr, C: tl.constexpr, BC: tl.constexpr, SCALE: tl.constexpr,
                   INIT_SEQS: tl.constexpr, SCALAR: tl.constexpr, NSEQ: tl.constexpr, TOKEN_DO: tl.constexpr, SOURCE_V_KW: tl.constexpr, SOURCE_W_KW: tl.constexpr, WW, WRITE_GRADS_ONLY: tl.constexpr, HISTORY: tl.constexpr):
    tile = tl.program_id(0) % tl.cdiv(D, VD)
    cid = (tl.program_id(0) // tl.cdiv(D, VD)).to(tl.int64)
    if cid < tl.load(CHOFF + NSEQ):
        sequence = tl.load(CHSEQ + cid)
        rows, values = tl.arange(0, RK), tile * VD + tl.arange(0, VD)
        mask = (rows[:, None] < BK) & (values[None, :] < D)
        start, end = tl.load(CU + sequence), tl.load(CU + sequence + 1)
        first = tl.load(COFF + sequence)
        chunk = cid - tl.load(CHOFF + sequence)
        window_start = start + chunk * BC
        window_end = tl.minimum(window_start + BC, end)
        dh = tl.load(DHBOUND + cid * BK * D + rows[:, None] * D + values[None, :], mask, 0)
        for half in range(1, -1, -1):
            half_begin = window_start + half * 32
            if half_begin < window_end:
                if chunk == 0:
                    hhalf = _load_replay_state(INIT + (sequence % INIT_SEQS) * BK * D + rows[:, None] * D + values[None, :], mask, dh)
                else:
                    hhalf = _load_replay_state(SNAP + (first + chunk - 1) * BK * D + rows[:, None] * D + values[None, :], mask, dh)
                for e in range(window_start, half_begin):
                    if tl.load(ORDER + e) < WRITE_EVENTS:
                        hhalf = _write(K,V,G,B,W,hhalf,e,rows,values,BK,D,SCALAR,SOURCE_V_KW,ORDER,SOURCE_W_KW,WW)
                for quarter in range(0, -1, -1):
                    quarter_begin = half_begin + quarter * 16
                    if quarter_begin < window_end:
                        hquarter = hhalf
                        for e in range(half_begin, quarter_begin):
                            if tl.load(ORDER + e) < WRITE_EVENTS:
                                hquarter = _write(K,V,G,B,W,hquarter,e,rows,values,BK,D,SCALAR,SOURCE_V_KW,ORDER,SOURCE_W_KW,WW)
                        for leaf in range(3, -1, -1):
                            leaf_begin = quarter_begin + leaf * 8
                            leaf_end = tl.minimum(leaf_begin + 8, window_end)
                            if leaf_begin < leaf_end:
                                hbase = hquarter
                                for e in range(quarter_begin, leaf_begin):
                                    if tl.load(ORDER + e) < WRITE_EVENTS:
                                        hbase = _write(K,V,G,B,W,hbase,e,rows,values,BK,D,SCALAR,SOURCE_V_KW,ORDER,SOURCE_W_KW,WW)
                                h = hbase
                                for e in range(leaf_begin, leaf_end):
                                    if tl.load(ORDER + e) < WRITE_EVENTS:
                                        h = _write(K,V,G,B,W,h,e,rows,values,BK,D,SCALAR,SOURCE_V_KW,ORDER,SOURCE_W_KW,WW)
                                for e in range(leaf_end-1, leaf_begin-1, -1):
                                    if tl.load(ORDER + e) < WRITE_EVENTS:
                                        h = hbase
                                        for p in range(leaf_begin, e):
                                            if tl.load(ORDER + p) < WRITE_EVENTS:
                                                h = _write(K,V,G,B,W,h,p,rows,values,BK,D,SCALAR,SOURCE_V_KW,ORDER,SOURCE_W_KW,WW)
                                        dh = _event_vjp(e,h,dh,rows,values,tile,Q,K,V,G,B,W,DO,ORDER,TOKENS,DQ,DK,DV,DG,DB,DW,WW,WRITE_EVENTS,E,BK,D,RK,VD,SCALE,SCALAR,TOKEN_DO,SOURCE_V_KW,SOURCE_W_KW,WRITE_GRADS_ONLY,True)
                                    else:
                                        dh = _event_vjp(e,h,dh,rows,values,tile,Q,K,V,G,B,W,DO,ORDER,TOKENS,DQ,DK,DV,DG,DB,DW,WW,WRITE_EVENTS,E,BK,D,RK,VD,SCALE,SCALAR,TOKEN_DO,SOURCE_V_KW,SOURCE_W_KW,WRITE_GRADS_ONLY,False)
        if chunk == 0:
            tl.store(DH + sequence * BK * D + rows[:, None] * D + values[None, :], dh, mask)
