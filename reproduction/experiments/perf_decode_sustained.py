"""Matched steady-state decode sweep; setup and profiler are outside measurements.

Warm up until runtime as well as compiler activity settles. Measure independent
1024-token continuations from the same prompt, rebuilding the cache untimed.
No CUDA graphs or architecture/kernel changes are introduced by this timing fix.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

OUT = Path('outputs/decode-sustained')
CONTEXTS = [262144,8192,16384,32768,65536,131072,524288,1048576]
FAMILIES = ['attention','bsdm','gdn2_nvidia','gdn1','sdm_native']
TOKENS = 1024
WARM_BLOCK = 256
MAX_WARM_BLOCKS = 32
RESERVE = MAX_WARM_BLOCKS * WARM_BLOCK + 128


def stable(values, tolerance):
    return (max(values)-min(values))/statistics.mean(values) <= tolerance


def case(args):
    import numpy as np
    import torch
    # Resolve the correct native FLA package before compiler capture.
    if args.family=='gdn1': sys.path.insert(0,'gdn1-runtime')
    from fla.modules.l2norm import l2norm  # noqa: F401
    from perf_attention_model import AttentionInferenceModel
    from perf_inference_case import InferenceModel, cache_bytes
    from torch._dynamo.utils import counters
    torch.set_num_threads(4)
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch._dynamo.config.cache_size_limit=128
    torch._dynamo.config.accumulated_cache_size_limit=1024
    torch.use_deterministic_algorithms(True,warn_only=True)
    if args.family=='gdn2_nvidia':
        import sympy
        from torch._inductor.codegen import triton_utils
        original=triton_utils.expr_fits_within_32bit
        triton_utils.expr_fits_within_32bit=lambda expr:original(sympy.sympify(expr))
    if args.family=='bsdm':
        from numerics import install
        install(128)
    n=args.context
    arm=(f'bsdm_nvidia_n{n}' if args.family=='bsdm' else
         f'{args.family}_k64_n{n}' if args.family.startswith('sdm') else args.family)
    model=(AttentionInferenceModel(arm,n,reserve_tokens=RESERVE) if arm=='attention'
           else InferenceModel(arm,n)).cuda().bfloat16().eval()
    tokens=torch.from_numpy(np.load(args.input)['tokens'].copy()).cuda()
    record=dict(family=args.family,arm=arm,context=n,status='warming',
        device=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch_version=torch.__version__,cuda=torch.version.cuda,
        batch=1,width=2048,layers=16,accounting=model.accounting(),mixer=model.mixer_record,
        timed_tokens_per_window=TOKENS,warmup_block_tokens=WARM_BLOCK,
        minimum_warmup_tokens=1024,maximum_warmup_tokens=8192,
        warmup_spread_limit=.03,measurement_spread_limit=.05,
        reserve_tokens=RESERVE,deterministic=True,profiler=False,cuda_graph=False,
        policy='same compiled blocks; stabilized warmup; fresh untimed full prefill per sample; full vocabulary projection; teacher-forced continuation',
        warmup=[],samples=[])

    def save(): args.output.write_text(json.dumps(record,indent=2)+'\n')

    def fresh():
        logits,caches=model.run(tokens[:,:n],model.empty_cache())
        caches=model.handoff(caches)
        torch.cuda.synchronize()
        return logits,caches

    def window(caches,pos,count):
        before=counters['stats']['unique_graphs']
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(count):
            logits,caches=model.run(tokens[:,pos:pos+1],caches,decode=True)
            pos+=1
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        return logits,caches,pos,dict(tokens=count,seconds=seconds,
            tokens_per_second=count/seconds,ms_per_token=1000*seconds/count,
            compiled_graphs=counters['stats']['unique_graphs']-before)

    save()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits,caches=fresh();pos=n
        for i in range(MAX_WARM_BLOCKS):
            logits,caches,pos,row=window(caches,pos,WARM_BLOCK)
            record['warmup'].append(row);save()
            print('WARMUP',args.family,n,i+1,json.dumps(row),flush=True)
            tail=record['warmup'][-4:]
            if len(tail)==4 and not any(r['compiled_graphs'] for r in tail) and stable([r['ms_per_token'] for r in tail],.03):
                break
        else:
            record.update(status='unstable',reason='Warmup did not stabilize within8192tokens')
            save();return
        record['warmup_tokens']=pos-n
        for repeat in range(6):
            # Rebuild outside timing, so every measured continuation starts at
            # the same prompt length and no large state clone doubles memory.
            del logits,caches
            gc.collect();torch.cuda.synchronize()
            logits,caches=fresh()
            # Exercise the newly constructed cache outside measurement too.
            for j in range(8):
                logits,caches=model.run(tokens[:,n+j:n+j+1],caches,decode=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            logits,caches,pos,row=window(caches,n+8,TOKENS)
            row.update(start_context=n+8,end_context=pos,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cache_bytes=cache_bytes(caches,model)[0],finite=bool(torch.isfinite(logits).all()))
            if args.family=='attention': assert all(c[2]==pos for c in caches)
            assert row['finite']
            record['samples'].append(row);save()
            print('SAMPLE',args.family,n,repeat+1,json.dumps(row),flush=True)
            tail=record['samples'][-3:]
            if len(tail)==3 and not any(r['compiled_graphs'] for r in tail) and stable([r['ms_per_token'] for r in tail],.05):
                record.update(status='ok',accepted_samples=tail,
                    tokens_per_second=sum(r['tokens'] for r in tail)/sum(r['seconds'] for r in tail),
                    decode_mean_seconds=sum(r['seconds'] for r in tail)/sum(r['tokens'] for r in tail),
                    decode_seconds_per_token=[r['seconds']/r['tokens'] for r in tail],
                    decode_peak_bytes=max(r['peak_allocated_bytes'] for r in tail),
                    cache_bytes_after_decode=tail[-1]['cache_bytes'])
                break
        else: record.update(status='unstable',reason='Post-warmup samples did not stabilize')
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    case(parser.parse_args())
