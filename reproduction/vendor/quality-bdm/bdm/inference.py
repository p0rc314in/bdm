"""Stateful inference using the canonical controllers and role recurrence.

State is persistent [batch, banks, rows, values], BF16 by default. BF16
storage rounds at prefill handoff and after each decode update; all bank
recurrence arithmetic remains FP32. Initial factors retain parameter-dtype
addition. Cache storage precision is an inference policy, not a training change.
"""
import torch
import triton as tr
import triton.language as tl

from . import role_bank_dispatch as dispatch
from .forward_rows import _sum8, forward_rows_stateful, forward_rows_readonly_stateful
from .product_routed_update import materialize
from .routed_update import decay_transform
from .training_hopper import exp_gates
from .role_bank_kernel import _write


@tr.jit
def _decode_write(K,V,G,B,W,WI,WW,ORDER,STATE,KW:tl.constexpr,N:tl.constexpr,
                  D:tl.constexpr,VD:tl.constexpr):
    event=tl.program_id(0).to(tl.int64)
    batch=event//KW
    bank=tl.load(WI+event)
    rows=tl.arange(0,8)
    values=tl.program_id(1)*VD+tl.arange(0,VD)
    offsets=((batch*N+bank)*8+rows[:,None])*D+values[None,:]
    state=tl.load(STATE+offsets,values[None,:]<D,0).to(tl.float32)
    # Packing has already applied the same route-weighted gates as training.
    state=_write(K,V,G,B,W,state,event,rows,values,8,D,False,
                 KW,ORDER,KW,WW)
    tl.store(STATE+offsets,state,values[None,:]<D)


@tr.jit
def _decode_read(Q,RI,STATE,OUT,KR:tl.constexpr,N:tl.constexpr,
                 D:tl.constexpr,VD:tl.constexpr,SCALE:tl.constexpr,WRITE_EVENTS:tl.constexpr):
    event=tl.program_id(0).to(tl.int64)
    batch=event//KR
    bank=tl.load(RI+event)
    values=tl.program_id(1)*VD+tl.arange(0,VD)
    products=()
    for row in tl.static_range(8):
        h=tl.load(STATE+((batch*N+bank)*8+row)*D+values,values<D,0)
        q=tl.load(Q+(WRITE_EVENTS+event)*8+row).to(tl.float32)
        products+=(h*q,)
    value=_sum8(*products)*SCALE
    tl.store(OUT+event*D+values,value,values<D)


@tr.jit
def _decode_sum(VALUES,RI,OUT,KR:tl.constexpr,RK:tl.constexpr,D:tl.constexpr,C:tl.constexpr):
    batch=tl.program_id(0).to(tl.int64)
    cols=tl.program_id(1)*C+tl.arange(0,C)
    routes=tl.arange(0,RK)
    banks=tl.load(RI+batch*KR+routes,routes<KR,0x7fffffffffffffff)
    previous=tl.full((),-1,tl.int64)
    total=tl.zeros((C,),tl.float32)
    for _ in range(KR):
        bank=tl.min(tl.where(banks>previous,banks,0x7fffffffffffffff),0)
        route=tl.min(tl.where(banks==bank,routes,RK),0)
        value=tl.load(VALUES+(batch*KR+route)*D+cols,cols<D,0)
        total=(total+value.to(tl.float32)).to(value.dtype).to(tl.float32)
        previous=bank
    tl.store(OUT+batch*D+cols,total,cols<D)


@tr.jit
def _compact_readout(VALUES,INVERSE,OUT,NT:tl.constexpr,KW:tl.constexpr,KR:tl.constexpr,
                     RK:tl.constexpr,D:tl.constexpr,WIDTH:tl.constexpr,OFFSET:tl.constexpr,C:tl.constexpr):
    token=tl.program_id(0).to(tl.int64)
    cols=tl.program_id(1)*C+tl.arange(0,C)
    routes=tl.arange(0,RK)
    positions=tl.load(INVERSE+NT*KW+token*KR+routes,routes<KR,0x7fffffffffffffff)
    previous=tl.full((),-1,tl.int64)
    total=tl.zeros((C,),tl.float32)
    for _ in range(KR):
        event=tl.min(tl.where(positions>previous,positions,0x7fffffffffffffff),0)
        route=tl.min(tl.where(positions==event,routes,RK),0)
        value=tl.load(VALUES+(token*KR+route)*WIDTH+cols,cols<WIDTH,0)
        total=(total+value.to(tl.float32)).to(value.dtype).to(tl.float32)
        previous=event
    tl.store(OUT+token*D+OFFSET+cols,total,(cols<WIDTH)&(OFFSET+cols<D))


def _validate(layer, hidden, state=None):
    if layer.backend != 'role' or not hidden.is_cuda:
        raise ValueError('Stateful inference requires the role CUDA backend')
    if layer.num_heads != 1 or layer.bank_size != 8:
        raise NotImplementedError('Stateful CUDA inference currently supports H=1 and bank size8')
    if hidden.ndim != 3 or hidden.shape[1] == 0 or hidden.shape[-1] != layer.hidden_size:
        raise ValueError('hidden must be [batch, positive time, dim]')
    if state is not None and (state.shape != (hidden.shape[0],layer.bank_count,8,layer.value_width)
            or state.dtype not in (torch.float32,torch.bfloat16) or state.device != hidden.device or not state.is_contiguous()):
        raise ValueError('state must be contiguous FP32 or BF16 [batch, banks, 8, values] on the input device')


def create_state(layer, batch, *, dtype=torch.bfloat16):
    if layer.num_heads != 1 or layer.bank_size != 8:
        raise NotImplementedError('Stateful CUDA inference currently supports H=1 and bank size8')
    if dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('cache dtype must be FP32 or BF16')
    initial=materialize(layer.initial_row_factor,layer.initial_column_factor,
                        layer.bank_count,layer.bank_size,dtype=dtype)
    return initial.unsqueeze(0) if batch==1 else initial.unsqueeze(0).repeat(batch,1,1,1)


def prefill(layer, hidden, state=None, *, state_dtype=None):
    _validate(layer,hidden,state)
    if torch.is_grad_enabled():
        raise RuntimeError('Stateful inference requires no_grad or inference_mode')
    cache_dtype=state_dtype or (state.dtype if state is not None else torch.bfloat16)
    if cache_dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('cache dtype must be FP32 or BF16')
    if state is None:
        # Wider model weights must not be rounded early merely because the
        # requested final cache is BF16. BF16 factors are already exact in it.
        work_dtype=torch.float32 if layer.initial_row_factor.dtype != cache_dtype else cache_dtype
        state=create_state(layer,hidden.shape[0],dtype=work_dtype)
    elif state_dtype is not None and state_dtype != state.dtype:
        raise ValueError('state_dtype must match the supplied cache')
    q,k,v,g,b,w=layer._controllers(hidden)
    read,write=layer._route_projections(hidden)
    ww,wi=layer._route(write,layer.num_writes)
    rw,ri=layer._route(read,layer.num_reads)
    raw=_bank_prefill(layer,q,k,v,g,b,w,ww,wi,rw,ri,state)
    if state.dtype != cache_dtype:
        state=state.to(cache_dtype)
    return layer._project_bank_output(hidden,raw),state


def _bank_prefill(layer,q,k,v,g,b,w,ww,wi,rw,ri,state):
    meta=dispatch.metadata(wi,ri,layer.bank_count)
    packed=dispatch.pack(q,k,v,g,b,w,ww,rw,meta,8,token_values=True,token_writes=True)
    if layer.config.routed_decay:
        packed,_=decay_transform(packed,meta.order,wi,ww)
    packed=(*packed[:3],exp_gates(packed[3]),*packed[4:])
    d=layer.value_width
    compact=d>32
    out=v.new_empty((ri.numel() if compact else meta.order.numel(),d))
    sequences=state.shape[0]*state.shape[1]
    flat=sequences>65535
    grid=(tr.cdiv(d,1024)*sequences,) if flat else (tr.cdiv(d,1024),sequences)
    kernel=forward_rows_readonly_stateful if compact else forward_rows_stateful
    kernel[grid](*packed,state,meta.cu_seqlens,out,meta.order,ww,
        wi.numel(),d,1024,8**-.5,sequences,wi.shape[-1],flat,
        num_warps=4,enable_fp_fusion=False)
    if compact:
        raw=v.new_empty((q.shape[0],q.shape[1],d))
        _compact_readout[(q.shape[0]*q.shape[1],tr.cdiv(d,128))](
            out,meta.inverse,raw,q.shape[0]*q.shape[1],wi.shape[-1],ri.shape[-1],
            tr.next_power_of_2(ri.shape[-1]),d,d,0,128,num_warps=4,enable_fp_fusion=False)
    else:
        raw=dispatch.readout(out.view(1,-1,1,d),meta.inverse,meta.tokens,batch=q.shape[0],time=q.shape[1],
                            write_banks=wi.shape[-1],read_banks=ri.shape[-1])
    return raw


def step(layer, hidden, state):
    _validate(layer,hidden,state)
    if torch.is_grad_enabled() or hidden.shape[1] != 1:
        raise ValueError('step requires one token under no_grad/inference_mode')
    q,k,v,g,b,w=layer._controllers(hidden)
    read,write=layer._route_projections(hidden)
    ww,wi=layer._route(write,layer.num_writes)
    rw,ri=layer._route(read,layer.num_reads)
    events=wi.numel()+ri.numel()
    # No capacity-sized sorting or metadata construction during decode.
    order=torch.arange(events,device=q.device,dtype=torch.int64)
    tokens=torch.empty_like(order)
    meta=dispatch.Dispatch(order,torch.empty_like(order),tokens,order)
    packed=dispatch.pack(q,k,v,g,b,w,ww,rw,meta,8,token_values=True,token_writes=True)
    if layer.config.routed_decay:
        packed,_=decay_transform(packed,order,wi,ww)
    # _write's source indirection expects event ordinals, not bank IDs.
    eq,ek,ev,eg,eb,ew=packed
    d=layer.value_width
    _decode_write[(wi.numel(),tr.cdiv(d,256))](ek,ev,eg,eb,ew,wi,ww,order,state,
        wi.shape[-1],layer.bank_count,d,256,num_warps=4,enable_fp_fusion=False)
    reads=v.new_empty((ri.numel(),d))
    _decode_read[(ri.numel(),tr.cdiv(d,256))](eq,ri,state,reads,ri.shape[-1],layer.bank_count,
        d,256,8**-.5,wi.numel(),num_warps=4,enable_fp_fusion=False)
    raw=v.new_empty((hidden.shape[0],1,d))
    _decode_sum[(hidden.shape[0],tr.cdiv(d,128))](reads,ri,raw,ri.shape[-1],tr.next_power_of_2(ri.shape[-1]),
        d,128,num_warps=4,enable_fp_fusion=False)
    return layer._project_bank_output(hidden,raw),state
