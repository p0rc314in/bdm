"""Retain product factors instead of dense initial memory across autograd.

As in SDM initial-bank replay, round the additive factors to parameter dtype
before widening to FP32. Preserve the existing routed-update derivatives.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _product_state(U,V,OUT,N:tl.constexpr,D:tl.constexpr,C:tl.constexpr,X:tl.constexpr):
    i=tl.program_id(0).to(tl.int64)*X+tl.arange(0,X)
    row=i//D;value=i%D
    u=tl.load(U+(row//C)*D+value,i<N,0).to(tl.float32)
    v=tl.load(V+(row%C)*D+value,i<N,0).to(tl.float32)
    tl.store(OUT+i,(u+v).to(U.dtype.element_ty).to(tl.float32),i<N)

def materialize(row_factor,column_factor,bank_count,bank_size,*,dtype=torch.float32):
    state=torch.empty((bank_count,bank_size,row_factor.shape[-1]),device=row_factor.device,dtype=dtype)
    _product_state[(tr.cdiv(state.numel(),1024),)](row_factor,column_factor,state,state.numel(),row_factor.shape[-1],column_factor.shape[0],1024)
    return state



from .routed_update import decay_transform


class ProductRoutedUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, b, w, wi, ww, ri, rw, row_factor, column_factor, bank_count, bank_size, scale, compact, retain_snapshots, state_dtype):
        from . import role_bank_dispatch as dispatch, training_hopper as kernel
        state = materialize(row_factor,column_factor,bank_count,bank_size,dtype=state_dtype)
        meta = dispatch.metadata(wi, ri, bank_count)
        packed = dispatch.pack(q, k, v, g, b, w, ww, rw, meta, state.shape[1], token_values=True, token_writes=True)
        packed, _ = decay_transform(packed, meta.order, wi, ww)
        output, (snap, offsets, c, vd) = kernel.forward(
            packed, state, meta.cu_seqlens, meta.order, wi.numel(), scale, compact, source_v_kw=wi.shape[-1], source_w_kw=wi.shape[-1], write_weights=ww, retain_snapshots=retain_snapshots, state_dtype=state_dtype)
        ctx.save_for_backward(q, k, v, g, b, w, wi, ww, ri, rw, row_factor, column_factor, meta.order,
                              meta.inverse, meta.tokens, meta.cu_seqlens, snap, offsets)
        ctx.options = (scale, c, vd, bank_count, bank_size, state_dtype)
        return dispatch.readout(output, meta.inverse, meta.tokens, batch=q.shape[0], time=q.shape[1],
                                write_banks=wi.shape[-1], read_banks=ri.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        # Unpack checkpoint hooks once, outside Dynamo's argument guards.
        saved = ctx.saved_tensors
        return _backward(saved, ctx.options, grad_output)


@torch.compile
def _backward(saved, options, grad_output):
    from . import role_bank_dispatch as dispatch, training_hopper as kernel
    q, k, v, g, b, w, wi, ww, ri, rw, row_factor, column_factor, order, inverse, tokens, cu, snap, offsets = saved
    scale, c, vd, bank_count, bank_size, state_dtype = options
    state = materialize(row_factor,column_factor,bank_count,bank_size,dtype=state_dtype)
    meta = dispatch.Dispatch(order, inverse, tokens, cu)
    packed = dispatch.pack(q, k, v, g, b, w, ww, rw, meta, state.shape[1], token_values=True, token_writes=True)
    packed, cache = decay_transform(packed, order, wi, ww)
    # Consume the existing per-token gradient by index instead of making one
    # full value-width copy per selected read/write event.
    do = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    grads = list(kernel.backward(packed, state, cu, order, wi.numel(), scale,
                                 (snap, offsets, c, vd), do, tokens=tokens, source_v_kw=wi.shape[-1],
                                 source_w_kw=wi.shape[-1], write_weights=ww, compact_grads=True))
    write, index, base, factor = cache
    dg = torch.where(write[:, None], grads[3].reshape_as(base), 0).float()
    decay_dww = (dg * base).sum(-1)[inverse[:ww.numel()]].view_as(ww)
    grads[3] = (dg * factor[:, None]).view_as(grads[3])
    dq, dk, dv, dg, db, dw, dww, drw = dispatch.vjp((q, k, v, g, b, w), ww, rw, meta, grads[:6], compact_values=True)
    dww = (dww.float() + decay_dww).to(ww.dtype)
    ds = grads[6].to(row_factor.dtype).reshape(q.shape[0], *state.shape).sum(0)
    factor_gradient = ds.reshape(row_factor.shape[0], column_factor.shape[0], -1)
    dr = factor_gradient.sum(1)
    dc = factor_gradient.sum(0)
    return dq, dk, dv, dg, db, dw, None, dww, None, drw, dr, dc, None, None, None, None, None, None

def bank_training(*args, compact_replay=None, state_dtype=torch.float32):
    # Inspect grad mode outside the custom autograd forward, where it is disabled.
    # Whole-block checkpointing already owns snapshot retention through hooks.
    return ProductRoutedUpdate.apply(*args, compact_replay, torch.is_grad_enabled(), state_dtype)
