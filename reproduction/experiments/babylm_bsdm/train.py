"""Full BabyLM training and official evaluation on one disposable dstack worker."""
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from reproduction import local_store as wandb
from .spec import CONFIG, CHECKPOINT_EXPOSURES_MILLIONS
from .model import BabyLMBSDM, SOURCE
from .data import BabyLMData
from .optimizer import FP32MasterAdamW
from .io import atomic_json, atomic_torch_save, sha256_file
from .transport import WBTransport, trim_uploaded_cache


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def learning_rate(step):
    if step <= CONFIG['warmup_steps']:
        return CONFIG['peak_lr'] * step / CONFIG['warmup_steps']
    fraction = (step-CONFIG['warmup_steps'])/(CONFIG['steps']-CONFIG['warmup_steps'])
    return CONFIG['min_lr'] + (CONFIG['peak_lr']-CONFIG['min_lr'])*.5*(1+math.cos(math.pi*fraction))


def prior_artifacts(run):
    return list(wandb.Api().run(f'{run.entity}/{run.project}/{run.id}').logged_artifacts()) if run.resumed else []


def publish_recovery(run, path, previous, step):
    artifact = wandb.Artifact(run.id+'-training-recovery', type='checkpoint', metadata={'step':step})
    artifact.ttl = None; artifact.add_file(str(path), name='checkpoint.pt')
    uploaded = run.log_artifact(artifact, aliases=['latest']); uploaded.wait()
    if previous is not None and previous.id != uploaded.id:
        previous.ttl = timedelta(days=30); previous.save()
    trim_uploaded_cache()
    return uploaded


def publish_directory(run, directory, name, kind='evaluation'):
    artifact = wandb.Artifact(run.id+'-'+name, type=kind); artifact.ttl=None
    artifact.add_dir(str(directory)); uploaded=run.log_artifact(artifact); uploaded.wait()
    trim_uploaded_cache()
    return uploaded


def score_checkpoint(run, model, root, checkpoint, exposure, out):
    from .inference import score
    folder=out/'inference'/f'{exposure:04d}';folder.mkdir(parents=True,exist_ok=True)
    sections=['fast_zero_shot','reading','aoa']
    if exposure==1000:sections+=['terminal_zero_shot','full_ewok']
    # Evaluation must not alter the subsequent training RNG stream.
    rng=rng_state()
    scores={name:score(model,root/'inference',name,folder) for name in sections}
    restore_rng(rng)
    result=dict(exposure_millions=exposure,checkpoint_sha256=sha256_file(checkpoint),
                sections=scores,files={p.name:sha256_file(p) for p in folder.iterdir() if p.is_file()})
    atomic_json(folder/'COMPLETE.json',result)
    publish_directory(run,folder,f'evaluation-{exposure:04d}')
    run.log({'eval/exposure_millions':exposure,'eval/fast_zero_shot':scores['fast_zero_shot']['metrics']})


def main():
    if torch.__version__ != '2.11.0+cu128': raise RuntimeError('Pinned runtime changed')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(4);set_seed(CONFIG['seed'])
    from numerics import install
    install(128)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch._dynamo.config.cache_size_limit=128
    torch._dynamo.config.accumulated_cache_size_limit=1024
    out=Path('outputs/babylm');out.mkdir(parents=True,exist_ok=True)
    with wandb.init(entity='local',project='reproduction',id=os.environ['DSTACK_RUN_NAME'],
                    resume='allow',job_type='babylm-finalized-bsdm',config=CONFIG) as run:
        prior=prior_artifacts(run)
        if any(a.type=='result' and 'final' in a.aliases for a in prior):
            print('ALREADY_COMPLETE',flush=True);return
        run.summary.update({'phase':'setup','device':torch.cuda.get_device_name()})
        if CONFIG.get('validate_hopper_bf16', False):
            assert torch.cuda.get_device_capability() == (9, 0), 'Hopper worker required'
            run.summary['phase'] = 'hopper-bf16-checks'
            subprocess.run([sys.executable, '-m', 'pytest', '-q',
                str(SOURCE/'tests/test_training_state_storage.py'),
                '-k', 'hopper or default_layer',
                '--junitxml='+str(out/'precision-checks.xml')],
                env=dict(os.environ, PYTHONPATH=str(SOURCE)), check=True)
            run.summary['hopper_bf16_checks_passed'] = True
        root=Path(run.use_artifact('babylm').download())
        data=BabyLMData(root/'training/manifest.json',root)
        assert data.training_steps()==CONFIG['steps'] and data.training_tokens()==CONFIG['target_tokens']
        inference_manifest=json.loads((root/'inference/manifest.json').read_text())
        assert inference_manifest['sections']['full_ewok']['sequences']==15236
        # Reuse the immutable records; verify evaluation records before training.
        for section in ('inference','finetune'):
            manifest=json.loads((root/section/'manifest.json').read_text())
            records=manifest['records']
            if isinstance(records,dict): records=[dict(path=k,sha256=v) for k,v in records.items()]
            for record in records:
                assert sha256_file(root/section/record['path'])==record['sha256']
        model=BabyLMBSDM().cuda()
        fingerprints=model.common_fingerprints()
        if CONFIG.get('memory_rows'):
            assert model.config.memory.logical_rows == CONFIG['memory_rows'] == 2048
            assert model.config.memory.selected_banks*model.config.memory.bank_size == 64
            assert CONFIG['context']==2048 and CONFIG['layers']==16 and CONFIG['width']==512
            if model.architecture=='sdm_native':
                assert all(b.attention.mixer.slots_per_head==2048 for b in model.layers)
                assert all(b.attention.mixer.args.num_reads==b.attention.mixer.args.num_writes==64 for b in model.layers)
            else:
                assert all(b.attention.bank_count*b.attention.bank_size==2048 for b in model.layers)
                assert model.config.memory.training_state_dtype=='bfloat16'
        accounting=model.accounting();run.summary.update(accounting)
        assert model.layers[0].feed_forward.w1.out_features==CONFIG['ffn_width']
        optimizer=FP32MasterAdamW(model,lr=CONFIG['peak_lr'],betas=tuple(CONFIG['betas']),weight_decay=CONFIG['weight_decay'])
        # Masters capture initialized FP32 parameters before the compute cast.
        model.bfloat16();optimizer.sync_to_model()
        if CONFIG.get('validate_hopper_bf16', False):
            from bdm.role_bank_kernel import _hopper_training_supported
            assert model.config.memory.training_state_dtype == 'bfloat16'
            q = torch.empty((1, 1, CONFIG['memory']['bank_size']), device='cuda', dtype=torch.bfloat16)
            assert _hopper_training_supported(q, CONFIG['memory']['value_width'], False)
            run.summary.update(dict(training_state_dtype='bfloat16', recurrent_checkpoint_interval=64,
                optimized_hopper_dispatch=True, phase='compiling-training'))
        run.summary.update(dict(architecture=model.architecture,memory_rows=model.config.memory.logical_rows,
                                selected_rows=64,elastic_coefficient=model.auxiliary_coefficient))
        identity=dict(config=CONFIG,data_manifest_sha256=sha256_file(root/'training/manifest.json'))
        resolved_mixer = (model.layers[0].attention.record if model.architecture=='sdm_native'
                          else (dict(architecture='attention',heads=8,head_dim=64,rope_theta=10000.) if model.architecture=='attention' else model.config.memory.record()))
        atomic_json(out/'resolved.json',dict(**identity,accounting=accounting,
                    resolved_mixer=resolved_mixer,
                    initial_common_fp32_fingerprints=fingerprints,device=torch.cuda.get_device_name()))
        trim_uploaded_cache()
        previous=next((a for a in reversed(prior) if a.type=='checkpoint' and 'latest' in a.aliases and '-training-recovery:' in a.qualified_name),None)
        step0=0;history=[]
        if previous:
            saved=torch.load(Path(previous.download())/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert saved['identity']==identity
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            step0=saved['step'];history=saved['history'];restore_rng(saved['rng']);del saved
        schedule={int(r['optimizer_step']):int(r['official_exposure_millions']) for r in data.manifest['corpus']['training']['checkpoint_exposures']}
        assert list(schedule.values())==list(CHECKPOINT_EXPOSURES_MILLIONS)
        compiled=torch.compile(model)
        started=time.monotonic();last_recovery=started
        print('STARTED',step0,CONFIG['steps'],torch.cuda.get_device_name(),flush=True)
        for index in range(step0,CONFIG['steps']):
            before=time.monotonic();step=index+1
            for group in optimizer.param_groups:group['lr']=learning_rate(step)
            values=torch.from_numpy(data.train_batch(index,8).astype(np.int64,copy=False)).pin_memory().cuda(non_blocking=True)
            model.train();optimizer.zero_grad();loss_sum=0.;aux_sum=0.
            for first in range(0,len(values),CONFIG['microbatch']):
                current=values[first:first+CONFIG['microbatch']]
                with torch.autocast('cuda',dtype=torch.bfloat16):result=compiled(current[:,:-1],target=current[:,1:])
                if isinstance(result,tuple):loss,nll,aux=result
                else:loss=nll=result;aux=result.new_zeros(())
                fraction=len(current)/len(values)
                (loss*fraction).backward();optimizer.accumulate();loss_sum+=float(nll.detach())*fraction
                aux_sum+=float(aux.detach())*fraction
            norm=float(optimizer.clip_grad_norm(CONFIG['gradient_clip']));optimizer.step()
            if step==step0+1:optimizer.assert_precision()
            if not math.isfinite(loss_sum):raise FloatingPointError('Nonfinite task loss')
            if CONFIG['elastic_coefficient'] and not (math.isfinite(aux_sum) and 0 < aux_sum <= 1.00001):
                raise FloatingPointError('Invalid or absent Elastic occupancy penalty')
            row=dict(step=step,nll=loss_sum,gradient_norm=norm,learning_rate=learning_rate(step),
                     targets=min(step*8,822810)*2048,step_seconds=time.monotonic()-before,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
            if CONFIG['elastic_coefficient']:
                row.update(auxiliary_loss=aux_sum,objective=loss_sum+CONFIG['elastic_coefficient']*aux_sum)
            history.append(row);run.log({'train/'+k:v for k,v in row.items()})
            if step%100==0:
                run.summary.update({'phase':'training','step':step})
                print('TRAIN',step,'nll',loss_sum,'seconds',row['step_seconds'],flush=True)
            if step in schedule:
                exposure=schedule[step];folder=out/'checkpoints'/f'{exposure:04d}';folder.mkdir(parents=True,exist_ok=True)
                path=folder/'model.pt'
                atomic_torch_save(path,dict(model=optimizer.master_model_state(model),identity=identity,
                    step=step,official_exposure_millions=exposure,training_tokens=row['targets'],
                    accounting=accounting,tokenizer='tiktoken:gpt2'))
                reopened=torch.load(path,map_location='cpu',weights_only=False)
                assert reopened['step']==step and all(v.dtype==torch.float32 for v in reopened['model'].values())
                del reopened
                publish_directory(run,folder,f'model-{exposure:04d}',kind='model')
                score_checkpoint(run,model,root,path,exposure,out)
            if (step==1 and CONFIG.get('startup_recovery',False)) or step==CONFIG['steps'] or time.monotonic()-last_recovery>=CONFIG['recovery_seconds']:
                path=out/'checkpoint.pt'
                atomic_torch_save(path,dict(identity=identity,step=step,model=model.state_dict(),
                    optimizer=optimizer.state_dict(),rng=rng_state(),history=history))
                previous=publish_recovery(run,path,previous,step);last_recovery=time.monotonic()
                if step==1 and CONFIG.get('startup_recovery',False):
                    reopened=torch.load(path,map_location='cpu',weights_only=False)
                    assert reopened['identity']==identity and reopened['step']==1
                    model.load_state_dict(reopened['model']);optimizer.load_state_dict(reopened['optimizer'])
                    restore_rng(reopened['rng']);del reopened
                    run.summary['startup_recovery_reopened_step']=1
        # Resumed terminal workers obtain previously delivered trajectories.
        if run.resumed:
            for artifact in prior:
                short=artifact.name.split(':')[0]
                if artifact.type=='evaluation' and '-evaluation-' in short:
                    artifact.download(root=str(out/'inference'/short.rsplit('-',1)[-1]))
                if artifact.type=='model' and '-model-1000' in short:
                    artifact.download(root=str(out/'checkpoints/1000'))
        atomic_json(out/'training.json',dict(history=history,training_seconds=time.monotonic()-started))
        del optimizer,compiled,model;gc.collect();torch.cuda.empty_cache()
        from .finetune import run_task
        checkpoint=torch.load(out/'checkpoints/1000/model.pt',map_location='cpu',weights_only=False)
        tasks=json.loads((root/'finetune/manifest.json').read_text())['tasks']
        for task in tasks:
            phase='finetune-'+task;run.summary.update({'phase':phase})
            if any(a.type=='evaluation' and f'-{phase}-COMPLETE-json:' in a.qualified_name for a in prior):continue
            transport=WBTransport(run,phase)
            latest=next((a for a in reversed(prior) if a.type=='checkpoint' and f'-{phase}-recovery:' in a.qualified_name and 'latest' in a.aliases),None)
            resume=Path(latest.download())/'checkpoint.pt' if latest else None
            transport.previous=latest
            model=BabyLMBSDM().cuda()
            run_task(model,checkpoint,task=task,prepared_root=root/'finetune',output=out/phase,
                     identity=dict(run_id=run.id,task=task,config=CONFIG),transport=transport,resume=resume)
            del model;gc.collect();torch.cuda.empty_cache()
        from .human import score_human
        human=score_human(out/'inference',Path(os.environ['BDM_HUMAN_INPUTS']))
        atomic_json(out/'human-likeness.json',human)
        final=wandb.Artifact(run.id+'-result',type='result');final.ttl=None
        for name in ('resolved.json','training.json','human-likeness.json'):final.add_file(str(out/name))
        run.log_artifact(final,aliases=['final']).wait()
        if previous:previous.ttl=timedelta(days=30);previous.save()
        run.summary.update({'phase':'complete','step':CONFIG['steps']})


if __name__=='__main__':main()
