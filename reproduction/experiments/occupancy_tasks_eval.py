"""Joint exact recall and observed bank-union measurements."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from occupancy_tasks_data import CONDITIONS,EXAMPLES


@torch.no_grad()
def evaluate(model,dataset,split,step,examples=32):
    model.eval()
    occupancy_samples=min(64,examples)
    conditions=dataset.conditions
    layers=len(model.layers)
    banks=model.layers[0].attention.bank_count
    first=np.full((len(conditions),occupancy_samples,layers,banks),32768,dtype=np.int32)
    counts=np.zeros_like(first)
    predictions=np.empty((len(conditions),examples,16),dtype=np.uint16)
    labels=np.empty_like(predictions);ages=np.empty((len(conditions),examples,16),dtype=np.int32)
    losses=np.empty((len(conditions),examples,16),dtype=np.float32)
    try:
        for x,y,age,c,begin in dataset.evaluation_batches(split,examples):
            model.trace_routes=begin<occupancy_samples
            xt,yt=torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(xt)
            stop=begin+len(x);i=c['index']
            predictions[i,begin:stop]=logits.argmax(-1).cpu().numpy()
            labels[i,begin:stop]=y;ages[i,begin:stop]=age
            losses[i,begin:stop]=F.cross_entropy(logits.float().flatten(0,1),yt.flatten(),reduction='none').view_as(yt).cpu().numpy()
            if model.trace_routes:
                n=min(len(x),occupancy_samples-begin);length=x.shape[1]
                positions=torch.arange(1,length+1,device='cuda',dtype=torch.int64).repeat_interleave(8).expand(n,-1)
                for layer,block in enumerate(model.layers):
                    indices=block.attention._write_trace[:n].reshape(n,-1).long()
                    assert indices.shape[1]==length*8
                    times=torch.full((n,banks),32768,device='cuda',dtype=torch.int64)
                    times.scatter_reduce_(1,indices,positions,reduce='amin',include_self=True)
                    hits=torch.zeros((n,banks),device='cuda',dtype=torch.int64)
                    hits.scatter_add_(1,indices,torch.ones_like(indices))
                    first[i,begin:begin+n,layer]=times.cpu().numpy()
                    counts[i,begin:begin+n,layer]=hits.cpu().numpy()
                    block.attention._write_trace=None
    finally:model.trace_routes=False
    rows=[]
    for c in conditions:
        i,t=c['index'],c['sequence_length'];correct=predictions[i]==labels[i]
        row=dict(c,examples=examples,query_accuracy=float(correct.mean()),exact_accuracy=float(correct.all(-1).mean()),
                 nll=float(losses[i].mean()),oldest_quartile_accuracy=float(correct[ages[i]>=t*.75].mean()) if (ages[i]>=t*.75).any() else None)
        prefix=[]
        for length in sorted(set([1,8,32,128,512,t-16,t]+[v for v in (1024,2048,4096,8192,16384) if v<=t])):
            touched=(first[i]<=length).sum(-1)
            prefix.append(dict(tokens=length,bank_fraction=float(touched.mean()/banks),
                per_layer_bank_fraction=(touched.mean(0)/banks).tolist(),
                bank_value_bytes=float(touched.sum(-1).mean()*8*128*2)))
        pre=next(p for p in prefix if p['tokens']==t-16)
        row.update(occupancy_requests=occupancy_samples,bank_fraction=pre['bank_fraction'],
            bank_value_bytes=pre['bank_value_bytes'],prefixes=prefix,
            full_request_bank_fraction=prefix[-1]['bank_fraction'])
        assert (counts[i].sum(-1)==t*8).all()
        rows.append(row)
    output=Path('outputs/task-occupancy');output.mkdir(parents=True,exist_ok=True)
    stem=f'{split}-{step}'
    report=dict(protocol=dataset.manifest['protocol'],arm=model.arm,step=step,split=split,conditions=rows,
                storage='Measured selected-write union converted to BF16 values; not allocator peaks',
                primary='bank fraction before queries; accuracy reported beside it')
    (output/(stem+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/(stem+'-predictions.npz'),predictions=predictions,labels=labels,ages=ages,nll=losses)
    np.savez_compressed(output/(stem+'-routes.npz'),first_write_position=first,bank_write_counts=counts)
    return report,sorted(output.glob(stem+'*'))
