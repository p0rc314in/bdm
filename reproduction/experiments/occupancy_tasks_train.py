"""Three full diagnostic runs, using the maintained optimizer and recovery path."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from reproduction import local_store as wandb
from occupancy_tasks_model import TaskModel
from occupancy_tasks_data import TaskDataset,CONDITIONS,PROTOCOL,STEPS,TOKENS_PER_STEP,sha256
from occupancy_tasks_eval import evaluate
from train import save_checkpoint,restore_checkpoint,publish_checkpoint,update
from optim import AdamW

REPORT_STEPS=(1000,3000,6000,10000,15000,20000,25000,30000)
SOURCES=('sources.json',)


def publish_eval(run,report,paths):
    artifact=wandb.Artifact(run.id+'-evaluation',type='evaluation',metadata=dict(step=report['step'],split=report['split']))
    artifact.ttl=None
    for p in paths:artifact.add_file(str(p),name=p.name)
    uploaded=run.log_artifact(artifact,aliases=[f"{report['split']}-step-{report['step']}"])
    uploaded.wait(timeout=600)
    metrics={'probe/step':report['step']}
    for c in report['conditions']:
        for field in ('bank_fraction','full_request_bank_fraction','query_accuracy','exact_accuracy','nll'):
            if field in c:metrics[f"probe/{c['id']}/{field}"]=c[field]
    run.log(metrics)
    print('MATCHED_OCCUPANCY',json.dumps(dict(step=report['step'],split=report['split'],conditions=report['conditions'])),flush=True)
    return uploaded.qualified_name


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('bsdm','bsdm_elastic'),required=True)
    p.add_argument('--elastic',type=float,default=None)
    args=p.parse_args();seed=0;started=time.monotonic()
    if args.elastic is not None and (args.arm!='bsdm_elastic' or not 0 < args.elastic < float('inf')):
        p.error('--elastic requires bsdm_elastic and a positive finite coefficient')
    elastic=(.12 if args.elastic is None else args.elastic) if args.arm=='bsdm_elastic' else 0.
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    assert torch.__version__=='2.11.0+cu128' and torch.cuda.is_bf16_supported()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    torch._dynamo.config.cache_size_limit=128;torch._dynamo.config.accumulated_cache_size_limit=1024
    from numerics import install
    numerical=install(128)
    dataset_ref=os.environ['DATASET']
    config=dict(protocol=PROTOCOL,arm=args.arm,seed=seed,dataset=dataset_ref,
        steps=STEPS,tokens_per_update=TOKENS_PER_STEP,train_tokens=STEPS*TOKENS_PER_STEP,
        train_queries=2976000,conditions=CONDITIONS,width=128,layers=8,ffn_width=512,
        memory_rows=16384,selected_rows=64,state_dtype='bfloat16',heads=1,
        elastic=elastic,optimizer='adopted Recall AdamW',
        parameter_dtype='bfloat16',moments='bfloat16',master_weights=False,gradient_norm='FP32 accumulation',
        learning_rate=3e-4,warmup=100,schedule='constant after warmup',betas=[.9,.95],clip=1.,weight_decay=.01,
        numerical=numerical,source=json.loads(Path('sources.json').read_text()),
        reporting_steps=REPORT_STEPS,compiled=True,batch='16384 / sequence_length; no microbatch splitting')
    run_name=os.environ['DSTACK_RUN_NAME'];out=Path('outputs');out.mkdir(exist_ok=True)
    with wandb.init(entity='local',project='reproduction',id=run_name,resume='allow',config=config) as run:
        run.define_metric('train/*',step_metric='train/step');run.define_metric('probe/*',step_metric='probe/step')
        prior=list(wandb.Api().run(run.path).logged_artifacts()) if run.resumed else []
        if any(a.type=='result' and 'final' in a.aliases for a in prior):
            print('ALREADY_COMPLETE',flush=True);return
        previous=next((a for a in prior if a.type=='checkpoint' and 'latest' in a.aliases),None)
        artifact=run.use_artifact(dataset_ref,type='dataset');dataset=TaskDataset(artifact.download())
        assert dataset.manifest['train_queries']==config['train_queries']
        run.config.update(dict(manifest_sha256=sha256(dataset.root/'manifest.json')))
        model=TaskModel(args.arm,seed,elastic_coefficient=args.elastic)
        # Check common tensors, not a historical hash for a different codec.
        reference=TaskModel('bsdm',seed)
        assert model.common_hash()==reference.common_hash()
        if args.arm=='bsdm_elastic':
            assert model.state_dict().keys()==reference.state_dict().keys()
            assert all(torch.equal(v,reference.state_dict()[k]) for k,v in model.state_dict().items())
        del reference
        run.config.update(dict(resolved_model=model.resolved_model(),accounting=model.accounting(),common_hash=model.common_hash()))
        model=model.cuda().bfloat16();optimizer=AdamW(model,'recall')
        step,history,evaluations=0,[],{}
        if previous:
            step,history,evaluations=restore_checkpoint(Path(previous.download())/'checkpoint.pt',model,optimizer,config)
        print(('RESUMED' if previous else 'STARTED'),f'step={step} device={torch.cuda.get_device_name()} arm={args.arm}',flush=True)
        run.summary.update(dict(device=torch.cuda.get_device_name(),resumed_from_step=step))
        forward=torch.compile(lambda x,y:model(x,target=y),dynamic=False)
        last_checkpoint=time.monotonic()
        while step<STEPS:
            x,y,c=dataset.train_batch(step);xt=torch.from_numpy(x).cuda();yt=torch.from_numpy(y).cuda()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
            loss,norm,targets=update(model,forward,optimizer,xt,yt,'recall',step+1,len(x))
            torch.cuda.synchronize();seconds=time.monotonic()-begin;step+=1
            row=dict(step=step,condition=c['id'],loss=loss,grad_norm=norm,targets=targets,
                     seconds=seconds,input_tokens=x.size,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            if args.arm=='bsdm_elastic':row['auxiliary_loss']=model.last_auxiliary_loss
            history.append(row);run.log({'train/'+k:v for k,v in row.items()})
            if step==1 or step%100==0:
                recent=history[-100:];rate=len(recent)/sum(r['seconds'] for r in recent)
                print('PROGRESS',json.dumps(dict(step=step,arm=args.arm,condition=c['id'],steps_per_second=rate,loss=loss)),flush=True)
            checkpoint_due=step==1 or time.monotonic()-last_checkpoint>=600 or step in REPORT_STEPS
            if checkpoint_due:
                save_checkpoint(out/'checkpoint.pt',model,optimizer,step,config,history,evaluations)
                previous=publish_checkpoint(run,out/'checkpoint.pt',previous,step,30);last_checkpoint=time.monotonic()
            if step in REPORT_STEPS and step<STEPS:
                report,paths=evaluate(model,dataset,'validation',step,examples=32)
                evaluations[str(step)]=publish_eval(run,report,paths)
                analysis=wandb.Artifact(run.id+'-analysis',type='model',metadata=dict(step=step));analysis.ttl=None
                torch.save(dict(model=model.state_dict(),config=config,step=step),out/'analysis.pt')
                analysis.add_file(str(out/'analysis.pt'));run.log_artifact(analysis,aliases=[f'step-{step}']).wait(timeout=300)
                save_checkpoint(out/'checkpoint.pt',model,optimizer,step,config,history,evaluations)
                previous=publish_checkpoint(run,out/'checkpoint.pt',previous,step,30);last_checkpoint=time.monotonic()
        terminal={}
        for split in ('validation','test'):
            report,paths=evaluate(model,dataset,split,step,examples=2048)
            terminal[split]=dict(artifact=publish_eval(run,report,paths),conditions=report['conditions'])
        torch.save(dict(model=model.state_dict(),config=config,step=step),out/'model.pt')
        restored=TaskModel(args.arm,seed,elastic_coefficient=args.elastic).cuda().bfloat16().eval()
        restored.load_state_dict(torch.load(out/'model.pt',map_location='cuda',weights_only=False)['model'])
        x=torch.from_numpy(dataset.train_batch(0)[0][:1]).cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            torch.testing.assert_close(restored(x),model.eval()(x),rtol=0,atol=0)
        report=dict(config=config,completed_steps=step,training_seconds=sum(r['seconds'] for r in history),
                    input_tokens=sum(r['input_tokens'] for r in history),query_presentations=sum(r['targets'] for r in history),
                    terminal_reload_verified=True,terminal=terminal,accounting=model.accounting(),
                    wall_seconds=time.monotonic()-started,model_sha256=sha256(out/'model.pt'))
        assert report['input_tokens']==491520000 and report['query_presentations']==2976000
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        result=wandb.Artifact(run.id+'-results',type='result',metadata=dict(completed_steps=step));result.ttl=None
        for filename in ('metrics.json','curve.jsonl','model.pt'):result.add_file(str(out/filename),name=filename)
        for filename in SOURCES:result.add_file(filename,name='source/'+filename)
        for root in ('dependencies/bsdm/bdm','dependencies/bsdm/bsdm','dependencies/sdm-quality/lingua'):
            for path in sorted(Path(root).rglob('*.py')):result.add_file(str(path),name='source/'+str(path))
        final=run.log_artifact(result,aliases=['final']);final.wait(timeout=600)
        previous.ttl=timedelta(days=30);previous.save()
        run.summary.update(dict(completed_steps=step,terminal_reload_verified=True,result=final.qualified_name))
        print('COMPLETE',final.qualified_name,flush=True)


if __name__=='__main__':main()
