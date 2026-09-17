"""Full-model stateful prefill and continuation; one isolated GPU process per point."""
import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from perf_model import ProfileModel, BSDM_SOURCE


class InferenceModel(ProfileModel):
    def __init__(self, arm, context):
        super().__init__(arm,context=context,width=2048,layers=16,gdn_head_dim=64,gdn_expand_v=2.,bsdm_heads=1)
        if arm=='gdn1':
            # Inductor's extracted Triton module loses the imported softplus
            # symbol for this upstream recurrent kernel. Keep its native
            # wrapper opaque; the kernel and equations are unchanged.
            import fla.layers.gated_deltanet as native_gdn1
            native_gdn1.fused_recurrent_gated_delta_rule=torch.compiler.disable(native_gdn1.fused_recurrent_gated_delta_rule)
        self.common_hash=lambda:None
        self.full_parameter_hash=lambda:None
        self.prefill_blocks=[]
        self.decode_blocks=[]
        for block in self.layers:
            block.feed_forward.recompute_activations=False
            def call(hidden,cache,decode=False,block=block):
                x=block.attention_norm(hidden)
                if self.arm.startswith('bsdm'):
                    out,cache=(block.attention.step(x,cache) if decode else block.attention.prefill(x,cache))
                elif self.arm.startswith('sdm'):
                    out,cache=block.attention.mixer(x,cache=cache)
                else:
                    out,_,_=block.attention.mixer(x,past_key_values=cache,use_cache=True)
                hidden=hidden+out
                return hidden+block.feed_forward(block.ffn_norm(hidden)),cache
            self.prefill_blocks.append(torch.compile(call))
            self.decode_blocks.append(torch.compile(call))
        self.head=torch.compile(lambda hidden:self.output(self.norm(hidden)))

    def empty_cache(self):
        if self.arm.startswith('gdn'):
            from fla.models.utils import Cache
            shared=Cache()
            return [shared]*len(self.layers)
        return [None]*len(self.layers)

    def run(self,tokens,caches,decode=False):
        hidden=self.tok_embeddings(tokens)
        functions=self.decode_blocks if decode else self.prefill_blocks
        for i,fn in enumerate(functions):
            hidden,caches[i]=fn(hidden,caches[i],decode=decode)
        return self.head(hidden[:,-1:]),caches

    def handoff(self,caches):
        return caches


def cache_bytes(caches,model):
    excluded={p.untyped_storage().data_ptr() for p in model.parameters()}
    seen=set();storage={};dtypes=set()
    def visit(value):
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,torch.Tensor):
            ptr=value.untyped_storage().data_ptr()
            if ptr not in excluded:
                storage[ptr]=value.untyped_storage().nbytes();dtypes.add(str(value.dtype))
        elif isinstance(value,dict):
            for v in value.values():visit(v)
        elif isinstance(value,(list,tuple)):
            for v in value:visit(v)
        elif hasattr(value,'__dict__') and not isinstance(value,torch.nn.Module):
            visit(vars(value))
    visit(caches)
    return sum(storage.values()),sorted(dtypes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--context',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--timed',type=int,default=5)
    a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch._dynamo.config.cache_size_limit=128;torch._dynamo.config.accumulated_cache_size_limit=1024
    torch.use_deterministic_algorithms(True,warn_only=True)
    if a.arm=='gdn2_nvidia':
        # Torch 2.11's extracted Triton signature path passes a Python int
        # into a SymPy-only helper for large indices. Normalize its type;
        # preserve the existing 32/64-bit decision and generated arithmetic.
        import sympy
        from torch._inductor.codegen import triton_utils
        original_fits=triton_utils.expr_fits_within_32bit
        triton_utils.expr_fits_within_32bit=lambda expr:original_fits(sympy.sympify(expr))
    if a.arm.startswith('bsdm'):
        from numerics import install
        install(128)
    os.environ.setdefault('PROFILE_NATIVE_SDM_COMMIT','bfe51564d1349200d21c8ad6507c248ef7f8ef5e')
    os.environ.setdefault('PROFILE_NATIVE_SDM_CHANGES','BF16 correctness; rectangular selector; wide inference scratch/sort keys')
    model=InferenceModel(a.arm,a.context).to(device='cuda',dtype=torch.bfloat16).eval()
    sequence=torch.from_numpy(np.load(a.input)['tokens'].copy()).cuda()
    prompt=sequence[:,:a.context];continuation=sequence[:,a.context:]
    record=dict(arm=a.arm,context=a.context,capacity_rows=a.context if a.arm.startswith(('sdm','bsdm')) else None,
        batch=1,width=2048,layers=16,ffn_width=model.ffn_width,heads_bsdm=1 if a.arm.startswith('bsdm') else None,selected_rows_per_role=64 if a.arm.startswith(('sdm','bsdm')) else None,
        device=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch_version=torch.__version__,cuda=torch.version.cuda,accounting=model.accounting(),
        mixer=model.mixer_record,source=str(BSDM_SOURCE),status='running',phase='prefill',
        policy='inference_mode; BF16 weights/compute; native cache dtypes; compiled blocks; no CUDA graph; no context chunking',
        input_tokens='fixed shared Wiki stream; teacher-forced continuation, full vocabulary projection each token')
    def save():a.output.write_text(json.dumps(record,indent=2)+'\n')
    save()
    from torch._dynamo.utils import counters
    samples=[];handoffs=[];peaks=[];caches=None
    with torch.inference_mode():
        stable=0;started=time.perf_counter()
        for i in range(12):
            caches=None;gc.collect();torch.cuda.synchronize()
            before=counters['stats']['unique_graphs']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,caches=model.run(prompt,model.empty_cache())
                caches=model.handoff(caches)
            torch.cuda.synchronize()
            stable=stable+1 if counters['stats']['unique_graphs']==before else 0
            print('PREFILL_WARMUP',a.arm,a.context,i+1,'stable',stable,flush=True)
            if stable>=2:break
        if stable<2:raise RuntimeError('Prefill compilation did not settle')
        record['prefill_warmup_seconds']=time.perf_counter()-started
        for i in range(a.timed):
            caches=None;gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            before=counters['stats']['unique_graphs'];start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,caches=model.run(prompt,model.empty_cache())
                torch.cuda.synchronize();handoff=time.perf_counter()
                caches=model.handoff(caches)
            torch.cuda.synchronize();end=time.perf_counter()
            assert counters['stats']['unique_graphs']==before,'Compilation during measured prefill'
            samples.append(end-start);handoffs.append(end-handoff);peaks.append(torch.cuda.max_memory_allocated())
        assert torch.isfinite(logits).all()
        state_bytes,state_dtypes=cache_bytes(caches,model)
        record.update(prefill_seconds=samples,prefill_mean_seconds=statistics.mean(samples),
            prefill_handoff_seconds=handoffs,prefill_peak_bytes=max(peaks),cache_bytes_after_prefill=state_bytes,
            cache_dtypes=state_dtypes,prefill_checksum=float(logits.float().mean()),phase='decode')
        save();print('PREFILL_DONE',record['prefill_mean_seconds'],max(peaks)/2**30,flush=True)
        stable=0
        for i in range(24):
            before=counters['stats']['unique_graphs']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,caches=model.run(continuation[:,i:i+1],caches,decode=True)
            torch.cuda.synchronize()
            stable=stable+1 if counters['stats']['unique_graphs']==before else 0
            if stable>=3:break
        if stable<3:raise RuntimeError('Decode compilation did not settle')
        record['decode_warmup_tokens']=i+1
        samples=[];peaks=[];position=i+1
        for repeat in range(5):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            before=counters['stats']['unique_graphs'];start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                for j in range(16):
                    logits,caches=model.run(continuation[:,position:position+1],caches,decode=True);position+=1
            torch.cuda.synchronize();samples.append((time.perf_counter()-start)/16)
            assert counters['stats']['unique_graphs']==before,'Compilation during measured decode'
            peaks.append(torch.cuda.max_memory_allocated())
        assert torch.isfinite(logits).all()
        record.update(decode_seconds_per_token=samples,decode_mean_seconds=statistics.mean(samples),
            decode_peak_bytes=max(peaks),cache_bytes_after_decode=cache_bytes(caches,model)[0],
            decode_checksum=float(logits.float().mean()),status='ok',phase='complete',decode_timed_tokens=80)
        save();print('DECODE_DONE',record['decode_mean_seconds'],max(peaks)/2**30,flush=True)
