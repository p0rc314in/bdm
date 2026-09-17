"""Paired full-model recurrent decode, ordinary dispatch versus CUDA replay.

Same native recurrence/controller kernels, real Wiki prefill and continuation.
GDN's returned states are copied back into stable buffers for capture; this
copy is included in both candidate timings and checked against native outputs.
No model equations, cache precision, or read/write budgets are changed.
"""
import argparse, json, os
from pathlib import Path
import statistics, subprocess, sys, time, traceback

OUT=Path(os.environ.get('RECURRENT_OUT','outputs/recurrent-graph'))
CONTEXTS=(8192,32768,131072,524288)
FAMILIES=('attention','gdn2_nvidia','bsdm','gdn1','sdm_native')

class StableGDNCache:
    def __init__(self, old, layers):
        self.states=[old[i] for i in range(layers)]
    def __len__(self):return len(self.states)
    def __getitem__(self,i):return self.states[i]
    def update(self, layer_idx, offset=1, **kw):
        def copy(dst,src):
            if src is None:return
            if isinstance(src,(tuple,list)):
                for d,s in zip(dst,src):copy(d,s)
            else:dst.copy_(src)
        state=self.states[layer_idx]
        for k,v in kw.items():copy(state[k],v)
        return state


def tensors_in(caches, model):
    import torch
    excluded={p.untyped_storage().data_ptr() for p in model.parameters()}
    seen=set();ptrs=set();out=[]
    def visit(x):
        if id(x) in seen:return
        seen.add(id(x))
        if isinstance(x,torch.Tensor):
            p=x.untyped_storage().data_ptr()
            if p not in excluded and p not in ptrs:ptrs.add(p);out.append(x)
        elif isinstance(x,dict):
            for v in x.values():visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        elif hasattr(x,'__dict__') and not isinstance(x,torch.nn.Module):visit(vars(x))
    visit(caches);return out


def case(family,n,inputs):
    import numpy as np
    import torch
    if family=='attention':
        torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch._dynamo.config.cache_size_limit=128;torch._dynamo.config.accumulated_cache_size_limit=1024
        from perf_attention_graph import case as attention_case
        def save_attention(row):
            row.update(family='attention',gpu=torch.cuda.get_device_name())
            (OUT/f'{family}-{n}.json').write_text(json.dumps(row,indent=2)+'\n')
        result=attention_case(n,inputs,save_result=save_attention);result['family']='attention'
        if os.environ.get('PERF_PREFILL_ONLY')=='1':return
        if 'cache_dispatch' in result['timings']:result['timings']['dispatch']=result['timings']['cache_dispatch']
        result['timings']['graph']=result['timings']['cache_graph']
        result['gpu']=torch.cuda.get_device_name()
        (OUT/f'{family}-{n}.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    if family=='gdn1':sys.path.insert(0,'gdn1-runtime')
    from fla.modules.l2norm import l2norm
    from perf_inference_case import InferenceModel,cache_bytes
    torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch._dynamo.config.cache_size_limit=128;torch._dynamo.config.accumulated_cache_size_limit=1024
    if family=='gdn2_nvidia':
        import sympy
        from torch._inductor.codegen import triton_utils
        original=triton_utils.expr_fits_within_32bit
        triton_utils.expr_fits_within_32bit=lambda expr:original(sympy.sympify(expr))
    if family=='bsdm':
        from numerics import install
        install(128)
    arm=(f'bsdm_nvidia_n{n}' if family=='bsdm' else f'{family}_k64_n{n}' if family.startswith('sdm') else family)
    model=InferenceModel(arm,n).to(device='cuda',dtype=torch.bfloat16).eval()
    tokens=torch.from_numpy(np.load(inputs)['tokens'].copy()).cuda()
    result=dict(family=family,arm=arm,context=n,width=2048,layers=16,ffn_width=model.ffn_width,
        batch=1,dtype='BF16',accounting=model.accounting(),mixer=model.mixer_record,
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),gpu=torch.cuda.get_device_name(),
        policy='native recurrent kernels; real Wiki prefill and continuation; full vocabulary; paired dispatch/replay',
        runtime_source=os.environ.get('PROFILE_BSDM_SOURCE') if family=='bsdm' else None,
        timings={},phase='prefill')
    def save(): (OUT/f'{family}-{n}.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        print('PREFILL',family,n,flush=True)
        def fresh():
            _,c=model.run(tokens[:,:n],model.empty_cache())
            return model.handoff(c)
        caches=fresh()
        if os.environ.get('MEASURE_PREFILL')=='1':
            import gc
            from torch._dynamo.utils import counters
            del caches;gc.collect();caches=fresh()
            samples=[];peaks=[]
            for _ in range(3):
                del caches;gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=counters['stats']['unique_graphs'];start=time.perf_counter()
                caches=fresh();torch.cuda.synchronize()
                samples.append(time.perf_counter()-start);peaks.append(torch.cuda.max_memory_allocated())
                assert counters['stats']['unique_graphs']==before,'Compilation during measured prefill'
            result.update(prefill_seconds=samples,prefill_mean_seconds=statistics.mean(samples),
                prefill_peak_bytes=max(peaks),cache_bytes_after_prefill=cache_bytes(caches,model)[0]);save()
            print('PREFILL_TIMED',family,n,json.dumps(samples),flush=True)
        if os.environ.get('PERF_PREFILL_ONLY')=='1':
            result['status']='ok';save();return
        result['phase']='decode_check';save()
        original_tensors=tensors_in(caches,model)
        # CPU snapshots avoid adding a second persistent GPU state to the timing.
        initial=[t.cpu().clone() for t in original_tensors]
        print('NATIVE_CHECK',family,n,flush=True)
        refs=[]
        for i in range(8):
            out,caches=model.run(tokens[:,n+i:n+i+1],caches,decode=True);refs.append(out.clone())
        # Restore original tensor identities, including returned GDN state objects.
        if family.startswith('gdn'):
            # Native cache update replaces tensors; reconnect to the prefill tensors
            # through a fresh cache snapshot created below, before capture.
            _,caches=model.run(tokens[:,:n],model.empty_cache())
            stable=StableGDNCache(caches[0],len(model.layers));caches=[stable]*len(model.layers)
        else:
            for t,s in zip(original_tensors,initial):t.copy_(s)
        # SDM's native cache has host-side control flow on cache_len. Keep
        # that metadata on the host, as its production graph wrapper does.
        host_caches=[c for c in caches if hasattr(c,'_cache_len')]
        budgets=[getattr(c,'_guaranteed_free_rows',None) for c in host_caches]
        active=tensors_in(caches,model)
        initial=[t.cpu().clone() for t in active]
        pos=torch.tensor([n],device='cuda',dtype=torch.int64)
        # Token selection and position advancement are GPU work inside the graph.
        get_token=torch.compile(lambda p:torch.index_select(tokens,1,p),fullgraph=True)
        def step():
            out,_=model.run(get_token(pos),caches,decode=True)
            pos.add_(1)
            return out
        def reset():
            for t,s in zip(active,initial):t.copy_(s)
            pos.fill_(n)
            for c,budget in zip(host_caches,budgets):
                c._cache_len=n
                if budget is not None:c._guaranteed_free_rows=budget
        def replay():
            graph.replay()
            for c in host_caches:
                c._cache_len+=1
                if hasattr(c,'fully_allocated') and not c.fully_allocated:
                    c._guaranteed_free_rows-=c.banks*64
        reset();host_outputs=[];errors=[]
        for expected in refs:
            out=step();host_outputs.append(out.clone())
            err=dict(max_abs=(out.float()-expected.float()).abs().max().item(),
                     relative_l2=((out.float()-expected.float()).norm()/expected.float().norm()).item())
            errors.append(err)
            torch.testing.assert_close(out,expected,atol=.01,rtol=.01)
            assert err['relative_l2']<.01
        result['stable_vs_native_logits']=errors;save()
        print('CAPTURE',family,n,flush=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(8):step()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):graph_out=step()
        reset();torch.cuda.synchronize()
        graph_errors=[]
        for expected in host_outputs:
            replay();torch.cuda.synchronize()
            torch.testing.assert_close(graph_out,expected,rtol=0,atol=0)
            graph_errors.append(float((graph_out.float()-expected.float()).abs().max()))
        assert pos.item()==n+8
        result['graph_vs_dispatch_max_abs']=graph_errors
        result['cache_bytes']=cache_bytes(caches,model)[0]
        save()
        result['phase']='decode_timing';save()
        for mode in (('graph',) if os.environ.get('PERF_GRAPH_ONLY')=='1' else ('dispatch','graph')):
            reset();torch.cuda.synchronize()
            call=step if mode=='dispatch' else replay
            for _ in range(256):call()
            torch.cuda.synchronize();rows=[]
            for sample in range(3):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.cuda.reset_peak_memory_stats();a.record();start=time.perf_counter()
                for _ in range(256):call()
                b.record();b.synchronize()
                row=dict(wall_ms=1000*(time.perf_counter()-start)/256,cuda_event_ms=a.elapsed_time(b)/256,
                    start_context=n+256+sample*256,end_context=n+512+sample*256,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
                rows.append(row);print('RECURRENT_SAMPLE',family,n,mode,json.dumps(row),flush=True)
            result['timings'][mode]=rows;result[mode+'_final_length']=pos.item()
            assert pos.item()==n+1024
            assert all(c._cache_len==n+1024 for c in host_caches)
            save()
        if os.environ.get('PERF_TRACE')=='1':
            reset()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(2):replay()
                torch.cuda.synchronize()
            trace=OUT/f'{family}-{n}.trace.json';prof.export_chrome_trace(str(trace))
            events=json.loads(trace.read_text())['traceEvents']
            kernels=[e for e in events if e.get('cat')=='kernel']
            grouped={}
            for e in kernels:
                value=grouped.setdefault(e['name'],dict(count=0,total_us=0))
                value['count']+=1;value['total_us']+=e['dur']
            result['trace']=dict(tokens=2,kernels_per_token=len(kernels)/2,
                kernel_ms_per_token=sum(e['dur'] for e in kernels)/2000,
                groups=sorted([dict(name=k,**v) for k,v in grouped.items()],key=lambda r:-r['total_us']))
        result['status']='ok';save()
        print('RECURRENT_RESULT',json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    case(args.family,args.context,args.input)
