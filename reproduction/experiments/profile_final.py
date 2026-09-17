"""Same-worker final architecture timings. Profiling only: every update has lr0."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import numpy as np
import torch
from reproduction import local_store as wandb
from data import DATASETS,Dataset
from perf_model import ARMS,ARM_LABELS,ProfileModel,BSDM_SOURCE
from optim import AdamW

ROOT=Path(__file__).resolve().parent
RUNTIME_VARIANTS={'compute': {'path':str(BSDM_SOURCE), 'commit':'ed51fd03ef83ff8ce50555a66481d878c2fc4a4f'}}


def is_cuda_oom(log_text):
    """Only a typed CUDA OOM is a measured capacity limit, not a generic error."""
    return re.search(r'^torch\.(?:cuda\.)?OutOfMemoryError: CUDA out of memory',
                     log_text,re.MULTILINE) is not None


def one_case(a):
    assert torch.__version__=='2.11.0+cu128'
    torch.set_num_threads(4)
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch._dynamo.config.cache_size_limit=128
    torch._dynamo.config.accumulated_cache_size_limit=1024
    # Match all arms. The native SDM scatter product has no deterministic CUDA
    # implementation; warn-only preserves that canonical operation explicitly.
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    if a.arm.startswith('bsdm_'):
        from numerics import install
        numerical=install(128)
        torch.use_deterministic_algorithms(True,warn_only=True)
    else:numerical={'outer_softmax':'native control'}
    training=a.mode=='train'
    m=ProfileModel(a.arm,context=a.context,width=a.width,layers=a.layers,
                   gdn_head_dim=a.gdn_head_dim,gdn_expand_v=a.gdn_expand_v,
                   bsdm_heads=a.bsdm_heads).cuda().bfloat16().train(training)
    common=m.common_hash();accounting=m.accounting();parameter_hash=m.full_parameter_hash()
    optimizer=AdamW(m,'wiki') if training else None
    arrays=np.load(a.input)
    x=torch.from_numpy(arrays['tokens'].copy()).cuda()
    y=torch.from_numpy(arrays['targets'].copy()).cuda()
    assert x.shape==y.shape==(a.batch,a.context)
    dynamic=False if a.compile_policy=='static' else None
    if training:
        forward=torch.compile(lambda x,y:m(x,target=y),dynamic=dynamic)
    else:
        # All tokens traverse every canonical block. Only the final-position
        # vocabulary projection is needed for time to first next-token logits.
        # This is one uninterrupted full-sequence call, not reset-state chunks.
        forward=torch.compile(lambda x,y:m.output(m.norm(m._hidden(x)[:,-1:])),dynamic=dynamic)
    def step(trace=False):
        if training:optimizer.zero_grad()
        with torch.profiler.record_function('training/forward_loss') if trace else nullcontext():
            with nullcontext() if training else torch.no_grad():
                with torch.autocast('cuda',dtype=torch.bfloat16):value=forward(x,y)
        if training:
            with torch.profiler.record_function('training/backward') if trace else nullcontext():value.backward()
            with torch.profiler.record_function('training/clip_adamw') if trace else nullcontext():norm=optimizer.step(0.)
        else:norm=None
        torch.cuda.synchronize()
        return float(value.detach().float().mean()),norm
    start=time.perf_counter()
    from torch._dynamo.utils import counters
    stable=0
    for i in range(max(a.warmup,a.max_warmup)):
        before=counters['stats']['unique_graphs']
        loss,norm=step()
        stable=stable+1 if counters['stats']['unique_graphs']==before else 0
        if i in (0,9,a.warmup-1) or (i+1>=a.warmup and stable>=3):
            print(f'WARMUP arm={a.arm} rep={a.repeat} step={i+1} value={loss} seconds={time.perf_counter()-start:.1f}',flush=True)
        if i+1>=a.warmup and stable>=3:break
    warmup_steps=i+1
    assert stable>=3,'Warmup did not reach three consecutive compilation-free steps'
    warmup=time.perf_counter()-start
    if training:assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert np.isfinite(loss)
    graphs=counters['stats']['unique_graphs']
    torch.cuda.reset_peak_memory_stats();samples=[]
    for _ in range(a.timed):
        start=time.perf_counter();loss,norm=step();samples.append(time.perf_counter()-start)
    while sum(samples)<a.min_timed_seconds and len(samples)<a.max_timed:
        start=time.perf_counter();loss,norm=step();samples.append(time.perf_counter()-start)
    new_graphs=counters['stats']['unique_graphs']-graphs
    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    assert new_graphs==0,'New compilation invalidates timed window'
    # Trace after timing; retain enough leaf-kernel attribution to diagnose costs.
    top=[];phases=[]
    if a.trace_steps:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(a.trace_steps):step(trace=True)
        prof.export_chrome_trace(str(a.output.with_suffix('.trace.json')))
        top=sorted(prof.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)
        phases=[dict(name=e.key,device_us=e.device_time_total/a.trace_steps,cpu_us=e.cpu_time_total/a.trace_steps)
                for e in prof.key_averages() if e.key.startswith('training/')]
    replay=None
    if a.arm.startswith('bsdm_'):
        from bdm.role_bank_kernel import _replay_settings
        interval,dtype=_replay_settings(8,m.config.memory.head_value_width,torch.bfloat16,torch.cuda.get_device_capability(),False)
        replay=dict(interval=interval,snapshot_dtype=str(dtype),compact_replay=False)
    row=dict(arm=a.arm,label=ARM_LABELS[a.arm],variant=a.variant,repeat=a.repeat,batch=a.batch,context=a.context,layers=a.layers,width=a.width,ffn_width=m.ffn_width,
        mode=a.mode,measurement='full optimizer update' if training else 'full-sequence prefill to final-position logits; no persistent cache API',
        compile_policy=a.compile_policy,gdn_head_dim=a.gdn_head_dim,gdn_expand_v=a.gdn_expand_v,
        compiler_counters={k:dict(counters[k]) for k in ('stats','frames','unimplemented')},
        vocabulary=50257,mean_ms=1000*statistics.mean(samples),median_ms=1000*statistics.median(samples),
        steps_per_second=len(samples)/sum(samples),tokens_per_second=a.batch*a.context*len(samples)/sum(samples),
        samples_seconds=samples,peak_allocated_bytes=peak,peak_reserved_bytes=reserved,
        warmup_steps=warmup_steps,timed_steps=len(samples),warmup_seconds=warmup,new_timed_graphs=new_graphs,
        sample_cv=statistics.stdev(samples)/statistics.mean(samples) if len(samples)>1 else None,
        common_parameter_sha256=common,model=m.mixer_record,accounting=accounting,numerical=numerical,
        initialized_parameter_sha256=parameter_hash,runtime_source=RUNTIME_VARIANTS[a.variant],
        kernel_sha256=hashlib.sha256((BSDM_SOURCE/'bdm/role_bank_kernel.py').read_bytes()).hexdigest(),
        replay=replay,parameter_bytes=sum(p.numel()*p.element_size() for p in m.parameters()),
        master_bytes=sum(p.numel()*p.element_size() for p in optimizer.master_parameters) if training else 0,
        moment_bytes=sum(v.numel()*v.element_size() for s in optimizer.optimizer.state.values()
                         for k,v in s.items() if k in ('exp_avg','exp_avg_sq')) if training else 0,
        device=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        capability=list(torch.cuda.get_device_capability()),torch_version=torch.__version__,cuda=torch.version.cuda,
        finite_gradients=True if training else None,initialized_loss=loss if training else None,
        output_checksum=loss if not training else None,gradient_norm=norm,trained=False,learning_rate=0. if training else None,
        input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        phase_profile=phases,
        leaf_profile=[dict(name=e.key,device_us=e.self_device_time_total/a.trace_steps,cpu_us=e.self_cpu_time_total/a.trace_steps,calls=e.count/a.trace_steps) for e in top[:200]])
    a.output.write_text(json.dumps(row,indent=2)+'\n')
    print('PROFILE '+json.dumps({k:row[k] for k in ('arm','repeat','mean_ms','steps_per_second','peak_allocated_bytes','device')}),flush=True)
