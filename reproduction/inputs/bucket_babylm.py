"""Prepare immutable shape buckets and exact causal score masks on the controller."""
import argparse
from pathlib import Path
import numpy as np
from reproduction.inputs.babylm.data import BabyLMEvaluationData
from reproduction.inputs.babylm.evaluate_checkpoints import PackedEvaluation
from reproduction.inputs.babylm.ewok import FullEWoKData
from reproduction.inputs.babylm.io import atomic_json,sha256_file


def prepare(section,sequence,offsets,starts,output):
    lengths=np.diff(offsets);n=len(starts)
    assert len(lengths)==n and np.all(starts>=1) and np.all(starts<lengths)
    buckets=(64,128,256,512,1024,2048)
    assignment=np.searchsorted(buckets,lengths-1)
    if assignment.max()>=len(buckets):raise ValueError('evaluation exceeds declared context')
    metadata={'sequences':n,'buckets':[]}
    covered=[]
    for slot,length in enumerate(buckets):
        indices=np.flatnonzero(assignment==slot)
        if not len(indices):continue
        batch=min(32,4096//length);count=(len(indices)+batch-1)//batch
        root=output/section/str(length);root.mkdir(parents=True)
        shape=(count,batch,length)
        tokens=np.lib.format.open_memmap(root/'tokens.npy',mode='w+',dtype='<u2',shape=(count,batch,length+1));tokens[:]=50256
        masks=np.lib.format.open_memmap(root/'masks.npy',mode='w+',dtype='u1',shape=shape);masks[:]=0
        ids=np.full((count,batch),-1,dtype='<i4')
        for row,index in enumerate(indices):
            b,r=divmod(row,batch);values=sequence(int(index));tokens[b,r,:len(values)]=values
            masks[b,r,int(starts[index])-1:len(values)-1]=1;ids[b,r]=index
            if int(masks[b,r].sum())!=len(values)-int(starts[index]):raise ValueError('target mask changed')
            covered.append(int(index))
        tokens.flush();masks.flush();del tokens,masks
        np.save(root/'indices.npy',ids,allow_pickle=False)
        metadata['buckets'].append({'length':length,'batch_size':batch,'batches':count})
    assert sorted(covered)==list(range(n))
    return metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=args.source;fastroot=root/'babylm2026-strict-gpt2-checkpoint-evaluation-v1'
    fast=PackedEvaluation(fastroot,sha256_file(fastroot/'manifest.json'))
    zeroroot=root/'babylm2026-strict-gpt2-causal-t2048-10epoch-v1'
    zero=BabyLMEvaluationData(zeroroot/'manifest.json',root)
    ewok=FullEWoKData(root/'full-ewok-a/manifest.json')
    result={'format':'babylm2026-precomputed-inference-v2','sections':{},'source_manifests':{},'records':[]}
    for path in (fastroot/'manifest.json',zeroroot/'manifest.json',root/'full-ewok-a/manifest.json'):
        result['source_manifests'][str(path.relative_to(root))]=sha256_file(path)
    for section in ('fast_zero_shot','reading','aoa'):
        arrays=fast.sections[section]
        result['sections'][section]=prepare(section,lambda i:fast.sequence(section,i),arrays['offsets'],arrays['score_starts'],args.output)
        if section=='fast_zero_shot':
            result['sections'][section]['aggregation']={'tasks':fast.manifest[section]['tasks'],'subdomains':fast.manifest[section]['subdomains']}
            for name in ('offsets','score_starts','example_offsets','labels','task_ids','subdomain_ids','length_normalized'):
                np.save(args.output/section/(name+'.npy'),np.asarray(arrays[name]),allow_pickle=False)
    for section,data in [('terminal_zero_shot',zero),('full_ewok',ewok)]:
        arrays=data.zero_arrays
        result['sections'][section]=prepare(section,data.zero_candidate,arrays['candidate_offsets'],arrays['score_starts'],args.output)
        result['sections'][section]['aggregation']={'tasks':data.zero_tasks,'subdomains':data.zero_subdomains}
        for name in ('candidate_offsets','score_starts','example_offsets','labels','task_ids','subdomain_ids','length_normalized'):
            np.save(args.output/section/(('offsets' if name=='candidate_offsets' else name)+'.npy'),np.asarray(arrays[name]),allow_pickle=False)
    for f in sorted(args.output.rglob('*.npy')):
        result['records'].append({'path':str(f.relative_to(args.output)),'bytes':f.stat().st_size,'sha256':sha256_file(f)})
    atomic_json(args.output/'manifest.json',result)
    print({k:v['sequences'] for k,v in result['sections'].items()})
    print('manifest_sha256',sha256_file(args.output/'manifest.json'))


if __name__=='__main__':main()
