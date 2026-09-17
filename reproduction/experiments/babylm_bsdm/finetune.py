"""One resumable official fine-tuning task, with FP32 master optimization."""
from __future__ import annotations
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from .scoring import SequenceClassifier, classification_metrics, lr_factor
from .io import atomic_json, atomic_torch_save, sha256_file
from .optimizer import FP32MasterAdamW
from .train import set_seed, rng_state, restore_rng


def load_batches(path):
    """All padding, lengths, labels and epoch order were prepared locally."""
    return {key:np.load(path/f'{key}.npy',mmap_mode='r',allow_pickle=False)
            for key in ('tokens','lengths','labels','indices','counts','epoch_offsets')}


def device_batch(data,index):
    count=int(data['counts'][index])
    rows=data['indices'][index]
    values=torch.from_numpy(np.array(data['tokens'][rows],dtype=np.int64)).pin_memory().cuda(non_blocking=True)
    lengths=torch.from_numpy(np.array(data['lengths'][rows],dtype=np.int64)).cuda()
    labels=torch.from_numpy(np.array(data['labels'][rows],dtype=np.int64)).cuda()
    return values,lengths,labels,count


@torch.no_grad()
def evaluate(classifier,data):
    classifier.eval();predictions=[];labels=[]
    for index in range(len(data['counts'])):
        values,lengths,target,count=device_batch(data,index)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=classifier(values,lengths)
        if not torch.isfinite(logits[:count]).all():raise FloatingPointError('nonfinite validation logits')
        predictions.append(logits[:count].argmax(-1).cpu().numpy());labels.append(target[:count].cpu().numpy())
    predictions=np.concatenate(predictions).astype(np.uint8)
    return classification_metrics(predictions,np.concatenate(labels)),predictions


def run_task(model,checkpoint,*,task,prepared_root,output,identity,transport,resume=None):
    meta=json.loads((prepared_root/'manifest.json').read_text())
    config=meta['tasks'][task];hyper=meta['official_hyperparameters']
    train=load_batches(prepared_root/task/'train');valid=load_batches(prepared_root/task/'valid')
    set_seed(int(hyper['seed']))
    model.float();model.load_state_dict(checkpoint['model'])
    classifier=SequenceClassifier(model,config['num_labels']).cuda()
    optimizer=FP32MasterAdamW(classifier,lr=hyper['learning_rate'],betas=tuple(hyper['betas']),
                            weight_decay=hyper['weight_decay'],eps=hyper['epsilon'],honor_no_decay=False)
    model.bfloat16();optimizer.sync_to_model()
    current=0;best_score=None;best_state=None;best_epoch=0;epochs=[];epoch_loss_sum=0.
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('fine-tuning recovery identity changed')
        classifier.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        current=state['next_batch'];best_score=state['best_score'];best_state=state['best_state']
        best_epoch=state['best_epoch'];epochs=state['epochs'];epoch_loss_sum=state['epoch_loss_sum'];restore_rng(state['rng'])
    compiled=torch.compile(classifier)
    started=time.monotonic();last_recovery=started;steps=len(train['counts']);warmup=int(hyper['warmup_proportion']*steps)
    metric=config['selection_metric'];uploads={};output.mkdir(parents=True,exist_ok=True)
    boundaries=set(int(x) for x in train['epoch_offsets'][1:])
    print(json.dumps({'marker':'STARTED',**identity,'phase':'finetuning','task':task,'step':current,'steps':steps}),flush=True)
    for index in range(current,steps):
        classifier.train();optimizer.zero_grad()
        values,lengths,labels,count=device_batch(train,index)
        if count!=len(labels):raise ValueError('official drop-last training batches must be full')
        for group in optimizer.param_groups:group['lr']=hyper['learning_rate']*lr_factor(index,warmup,steps)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=compiled(values,lengths);loss=F.cross_entropy(logits,labels)
            if model.auxiliary_coefficient:
                loss=loss+model.auxiliary_coefficient*model.auxiliary_loss
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite fine-tuning loss')
        loss.backward();optimizer.accumulate();optimizer.step();epoch_loss_sum+=float(loss.detach())
        if index==current:optimizer.assert_precision()
        step=index+1
        if step%100==0:
            print(json.dumps({'marker':'HEARTBEAT',**identity,'phase':'finetuning','task':task,'step':step,'steps':steps,'loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-started}),flush=True)
        if step in boundaries:
            epoch=int(np.searchsorted(train['epoch_offsets'],step))
            result,predictions=evaluate(compiled,valid)
            score=float(result[metric])
            if best_score is None or score>best_score:
                best_score=score;best_epoch=epoch;best_state=optimizer.master_model_state(classifier)
                best_metrics=result
            n=step-int(train['epoch_offsets'][epoch-1])
            epochs.append({'epoch':epoch,'train_loss':epoch_loss_sum/n,'validation':result})
            epoch_loss_sum=0.
        # Epoch evaluation is independent of multi-GB recovery publication.
        if time.monotonic()-last_recovery>=900 or step==steps:
            slot=(step//500)%2
            if slot in uploads:uploads[slot].result()
            name=f'recovery-{slot}.pt';path=output/name
            state={'identity':identity,'next_batch':step,'model':{n:v.detach().cpu() for n,v in classifier.state_dict().items()},
                   'optimizer':optimizer.state_dict(),'rng':rng_state(),'best_score':best_score,'best_state':best_state,
                   'best_epoch':best_epoch,'epochs':epochs,'epoch_loss_sum':epoch_loss_sum}
            atomic_torch_save(path,state);record=path.with_suffix('.pt.json')
            atomic_json(record,{**identity,'next_batch':step,'sha256':sha256_file(path),'bytes':path.stat().st_size,'object':name})
            uploads[slot]=transport.publish_pair(name,path,record);last_recovery=time.monotonic()
    if best_state is None:raise RuntimeError('no selected fine-tuning state')
    classifier.load_state_dict(best_state);metrics,predictions=evaluate(compiled,valid)
    selection_validation=epochs[best_epoch-1]['validation']
    if selection_validation[metric]!=best_score:raise ValueError('selected epoch metadata changed')
    # Repeated BF16/hard-routing inference was also nondeterministic in this runtime.
    # Preserve epoch selection and report the reopened model's actual predictions.
    reopened_score_delta=float(metrics[metric])-best_score
    best_path=output/'best.pt';atomic_torch_save(best_path,{'identity':identity,'model':best_state,'epoch':best_epoch})
    predictions_path=output/'predictions.npy'
    with predictions_path.open('wb') as handle:np.save(handle,predictions,allow_pickle=False)
    result={**identity,'marker':'COMPLETE','phase':'finetuning_complete','task':task,'selection_metric':metric,
            'best_epoch':best_epoch,'best_validation':metrics,'selection_validation':selection_validation,
            'reopened_score_delta':reopened_score_delta,'epochs':epochs,'elapsed_seconds':time.monotonic()-started,
            'checkpoint_sha256':sha256_file(best_path),'predictions_sha256':sha256_file(predictions_path),'steps':steps}
    transport.finish();transport.put('best.pt',best_path);transport.put('predictions.npy',predictions_path)
    atomic_json(output/'COMPLETE.json',result);transport.put('COMPLETE.json',output/'COMPLETE.json')
    print(json.dumps(result),flush=True)
    return result
