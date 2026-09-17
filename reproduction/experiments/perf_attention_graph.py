"""Attention-only dispatch isolation with real growing KV caches.

Paired on one worker: native SDPA/Python dispatch, cache-aware FA2 or FA3/Python dispatch, then CUDA graph replay.
The paired comparison separates backend and replay effects.
Every path retains all16 layers, real Wiki continuation, full vocabulary,
BF16 and all32 KV heads. CUDA graphs update the token position on device.
"""
import argparse, importlib.util, json, os
from pathlib import Path
import statistics, subprocess, sys, time

OUT=Path('outputs/attention-graph')
IMPL=os.environ.get('ATTENTION_GRAPH_IMPL','fa3')
REV={'fa3':'e29f138fc363b396e5d2706c8a5f6fa7d36f41e0','fa2':'c3f3c93a4a39e3d2cf49c13eac49229bc726ea13'}[IMPL]
BUILD='build/torch211-cxx11-cu128-x86_64-linux'


def load_attention():
    from huggingface_hub import snapshot_download
    root=Path(snapshot_download('kernels-community/flash-attn'+IMPL[-1],revision=REV,allow_patterns=[BUILD+'/*']))/BUILD
    spec=importlib.util.spec_from_file_location('bsdm_probe_'+IMPL,root/'__init__.py',submodule_search_locations=[str(root)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def case(n,inputs,save_result=lambda row:None):
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    from perf_attention_model import AttentionInferenceModel
    backend=load_attention()

    @torch.library.custom_op("bsdm_attention_probe::read_cache",mutates_args=())
    def read_cache(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,length:torch.Tensor)->torch.Tensor:
        return backend.flash_attn_with_kvcache(q,k,v,cache_seqlens=length,causal=False,num_splits=0 if IMPL=="fa2" else 16)

    @read_cache.register_fake
    def read_cache_fake(q,k,v,length):
        return torch.empty_like(q)

    @triton.jit
    def append_kv(K,V,NK,NV,P,D:tl.constexpr,BLOCK:tl.constexpr):
        j=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        p=tl.load(P)
        tl.store(K+p*D+j,tl.load(NK+j,j<D,0),j<D)
        tl.store(V+p*D+j,tl.load(NV+j,j<D,0),j<D)

    model=AttentionInferenceModel('attention',n,reserve_tokens=4096).to(device='cuda',dtype=torch.bfloat16).eval()
    tokens=torch.from_numpy(np.load(inputs)['tokens'].copy()).cuda()
    result=dict(context=n,accounting=model.accounting(),layers=16,width=2048,ffn_width=model.ffn_width,
        heads=32,kv_heads=32,batch=1,dtype='BF16',backend=IMPL,backend_revision=REV,
        policy='real Wiki prefill and growing continuation; full vocabulary; paired kernels and graph replay',
        cache_num_splits=0 if IMPL=="fa2" else 16,
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),phase='prefill')
    save_result(result)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        _,caches=model.run(tokens[:,:n],model.empty_cache())
        if os.environ.get('MEASURE_PREFILL')=='1':
            import gc
            from torch._dynamo.utils import counters
            del caches;gc.collect()
            _,caches=model.run(tokens[:,:n],model.empty_cache())
            samples=[];peaks=[]
            for _ in range(3):
                del caches;gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=counters['stats']['unique_graphs'];start=time.perf_counter()
                _,caches=model.run(tokens[:,:n],model.empty_cache());torch.cuda.synchronize()
                samples.append(time.perf_counter()-start);peaks.append(torch.cuda.max_memory_allocated())
                assert counters['stats']['unique_graphs']==before,'Compilation during measured prefill'
            result.update(prefill_seconds=samples,prefill_mean_seconds=statistics.mean(samples),prefill_peak_bytes=max(peaks))
            save_result(result)
            print('PREFILL_TIMED','attention',n,json.dumps(samples),flush=True)
        if os.environ.get('PERF_PREFILL_ONLY')=='1':
            result['status']='ok';save_result(result);return result
        result['phase']='decode_check';save_result(result)
        reference=[]
        for p in range(n,n+8):
            out,caches=model.run(tokens[:,p:p+1],caches,decode=True);reference.append(out.clone())
        pos=torch.tensor([n],device='cuda',dtype=torch.int64)
        blocks=[]
        for block,(keys,values,_) in zip(model.layers,caches):
            def step(hidden,position,block=block,keys=keys,values=values):
                q,k,v=block.attention.project(block.attention_norm(hidden),position)
                append_kv[(triton.cdiv(2048,256),)](keys,values,k,v,position,2048,256)
                length=(position+1).to(torch.int32)
                # FA3 requires an explicit split count for torch.compile shape
                # inference. Hold it fixed across both execution paths/contexts.
                read=read_cache(q,keys,values,length)
                hidden=hidden+block.attention.mixer.wo(read.flatten(2))
                return hidden+block.feed_forward(block.ffn_norm(hidden))
            blocks.append(torch.compile(step,fullgraph=True))
        embed=torch.compile(lambda position:model.tok_embeddings(torch.index_select(tokens,1,position)),fullgraph=True)
        def graph_step():
            hidden=embed(pos)
            for block in blocks:hidden=block(hidden,pos)
            out=model.head(hidden)
            pos.add_(1)
            return out
        errors=[];dispatch_outputs=[]
        for expected in reference:
            out=graph_step();dispatch_outputs.append(out.clone())
            errors.append(dict(max_abs=(out.float()-expected.float()).abs().max().item(),
                relative_l2=((out.float()-expected.float()).norm()/expected.float().norm()).item()))
            torch.testing.assert_close(out,expected,atol=.04,rtol=.04)
            assert errors[-1]['relative_l2']<.02
        result['cache_vs_native_logits']=errors
        # Compile before capture and use a nondefault stream for graph warmup.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(8):graph_step()
        torch.cuda.current_stream().wait_stream(stream)
        pos.fill_(n)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):graph_output=graph_step()
        pos.fill_(n);errors=[]
        for expected,dispatched in zip(reference,dispatch_outputs):
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(graph_output,dispatched,atol=0,rtol=0)
            errors.append(dict(max_abs=(graph_output.float()-expected.float()).abs().max().item(),
                relative_l2=((graph_output.float()-expected.float()).norm()/expected.float().norm()).item()))
            torch.testing.assert_close(graph_output,expected,atol=.04,rtol=.04)
            assert errors[-1]['relative_l2']<.02
        assert pos.item()==n+8
        result['graph_vs_native_logits']=errors
        result['graph_vs_dispatch_bit_exact']=True
        result['timings']={}
        result['phase']='decode_timing';save_result(result)
        for mode in (('cache_graph',) if os.environ.get('PERF_GRAPH_ONLY')=='1' else ('native','cache_dispatch','cache_graph')):
            pos.fill_(n);host_pos=n
            caches=[(k,v,n) for k,v,_ in caches]
            def invoke():
                nonlocal host_pos,caches
                if mode=='native':
                    _,caches=model.run(tokens[:,host_pos:host_pos+1],caches,decode=True);host_pos+=1
                elif mode=='cache_dispatch':graph_step()
                else:graph.replay()
            for _ in range(256):invoke()
            torch.cuda.synchronize();rows=[]
            for sample in range(3):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.cuda.reset_peak_memory_stats();a.record();start=time.perf_counter()
                for _ in range(256):invoke()
                b.record();b.synchronize()
                row=dict(wall_ms=1000*(time.perf_counter()-start)/256,cuda_event_ms=a.elapsed_time(b)/256,
                    start_context=n+256+sample*256,end_context=n+512+sample*256,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                rows.append(row);print('GRAPH_SAMPLE',n,mode,json.dumps(row),flush=True)
            result['timings'][mode]=rows
            actual=host_pos if mode=='native' else pos.item()
            assert actual==n+1024
            result[mode+'_final_length']=actual
        result['status']='ok'
    return result
