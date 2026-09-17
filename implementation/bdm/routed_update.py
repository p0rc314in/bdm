"""Selected outer-write-strength adapter around the unchanged role kernels."""
import torch


def decay_transform(packed, order, wi, ww):
    write = order < wi.numel()
    index = order.clamp_max(wi.numel() - 1)
    base = torch.where(write[:, None], packed[3].reshape(order.numel(), -1), 0).float()
    factor = torch.where(write, ww.flatten()[index], 0).float()
    gate = (base * factor[:, None]).view_as(packed[3])
    values = list(packed)
    values[3] = gate
    return tuple(values), (write, index, base, factor)


class RoutedUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, b, w, wi, ww, ri, rw, state, bank_count, scale, compact, state_dtype):
        from . import role_bank_dispatch as dispatch, role_bank_kernel as kernel
        meta = dispatch.metadata(wi, ri, bank_count)
        packed = dispatch.pack(q, k, v, g, b, w, ww, rw, meta, state.shape[1])
        packed, _ = decay_transform(packed, meta.order, wi, ww)
        output, (snap, offsets, c, vd) = kernel.forward(
            packed, state.to(state_dtype), meta.cu_seqlens, meta.order, wi.numel(), scale, compact, state_dtype=state_dtype)
        ctx.save_for_backward(q, k, v, g, b, w, wi, ww, ri, rw, state, meta.order,
                              meta.inverse, meta.tokens, meta.cu_seqlens, snap, offsets)
        ctx.options = (scale, c, vd, state_dtype)
        return dispatch.readout(output, meta.inverse, meta.tokens, batch=q.shape[0], time=q.shape[1],
                                write_banks=wi.shape[-1], read_banks=ri.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        from . import role_bank_dispatch as dispatch, role_bank_kernel as kernel
        q, k, v, g, b, w, wi, ww, ri, rw, state, order, inverse, tokens, cu, snap, offsets = ctx.saved_tensors
        scale, c, vd, state_dtype = ctx.options
        meta = dispatch.Dispatch(order, inverse, tokens, cu)
        packed = dispatch.pack(q, k, v, g, b, w, ww, rw, meta, state.shape[1])
        packed, cache = decay_transform(packed, order, wi, ww)
        do = grad_output.reshape(-1, grad_output.shape[-1])[tokens].unsqueeze(0).unsqueeze(2).contiguous()
        grads = list(kernel.backward(packed, state.to(state_dtype), cu, order, wi.numel(), scale,
                                     (snap, offsets, c, vd), do))
        write, index, base, factor = cache
        dg = torch.where(write[:, None], grads[3].reshape_as(base), 0).float()
        decay_dww = (dg * base).sum(-1)[inverse[:ww.numel()]].view_as(ww)
        grads[3] = (dg * factor[:, None]).view_as(grads[3])
        dq, dk, dv, dg, db, dw, dww, drw = dispatch.vjp((q, k, v, g, b, w), ww, rw, meta, grads[:6])
        dww = (dww.float() + decay_dww).to(ww.dtype)
        ds = grads[6].to(state.dtype).reshape(q.shape[0], *state.shape).sum(0)
        return dq, dk, dv, dg, db, dw, None, dww, None, drw, ds, None, None, None, None


def bank_training(*args, compact_replay=None, state_dtype=torch.float32):
    return RoutedUpdate.apply(*args, compact_replay, state_dtype)
