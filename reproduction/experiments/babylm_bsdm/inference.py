"""GPU scoring of precomputed causal masks; no tokenizer or batch construction."""
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from .scoring import aggregate_fast
from .io import atomic_json,sha256_file


@torch.no_grad()
def score(model,root,section,output):
    manifest=json.loads((root/'manifest.json').read_text());meta=manifest['sections'][section]
    scores=np.full(meta['sequences'],np.nan,dtype=np.float64);model.eval();started=time.monotonic()
    for bucket in meta['buckets']:
        base=root/section/str(bucket['length'])
        data={k:np.load(base/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('tokens','masks','indices')}
        for index in range(bucket['batches']):
            values=torch.from_numpy(np.array(data['tokens'][index],dtype=np.int64)).pin_memory().cuda(non_blocking=True)
            mask=torch.from_numpy(np.array(data['masks'][index],dtype=np.bool_)).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(values[:,:-1],attn_impl='sdpa')
                losses=F.cross_entropy(logits.flatten(0,1),values[:,1:].flatten(),reduction='none').reshape_as(mask)
            totals=losses.masked_fill(~mask,0).sum(-1).double().cpu().numpy()
            ids=data['indices'][index];keep=ids>=0;scores[ids[keep]]=totals[keep]
            if index%100==0:
                print(json.dumps({'marker':'HEARTBEAT','phase':'inference','section':section,'length':bucket['length'],
                                  'batch':index,'batches':bucket['batches'],'elapsed_seconds':time.monotonic()-started}),flush=True)
    if not np.isfinite(scores).all():raise FloatingPointError('missing or nonfinite sequence score')
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/(section+'_losses.npy'),scores,allow_pickle=False)
    result={'sequences':len(scores),'elapsed_seconds':time.monotonic()-started}
    if 'aggregation' in meta:
        arrays={k:np.load(root/section/(k+'.npy'),allow_pickle=False) for k in
                ('offsets','score_starts','example_offsets','labels','task_ids','subdomain_ids','length_normalized')}
        adapter=SimpleNamespace(sections={'fast_zero_shot':arrays},manifest={'fast_zero_shot':meta['aggregation']})
        metrics,predictions=aggregate_fast(adapter,scores);result['metrics']=metrics
        np.save(output/(section+'_predictions.npy'),predictions,allow_pickle=False)
    atomic_json(output/(section+'.json'),result)
    return result
