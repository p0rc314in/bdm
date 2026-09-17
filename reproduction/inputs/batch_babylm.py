"""Verify inherited immutable records and prepare all deterministic batch metadata locally."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from reproduction.inputs.babylm.data import BabyLMEvaluationData
from reproduction.inputs.babylm.io import atomic_json, sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    data=BabyLMEvaluationData(args.source/'manifest.json',args.source.parent)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report={'format':'babylm2026-precomputed-finetune-batches-v2',
            'source_manifest_sha256':sha256_file(args.source/'manifest.json'),
            'packed_manifest_sha256':sha256_file(args.source/'evaluation/manifest.json'),
            'official_hyperparameters':data.manifest['finetune']['official_hyperparameters'],
            'tasks':{},'records':[]}
    for task,config in data.manifest['finetune']['tasks'].items():
        report['tasks'][task]={k:v for k,v in config.items() if k!='records'}
        for split in ('train','valid'):
            values=data.finetune_split(task,split);root=args.output/task/split;root.mkdir(parents=True)
            count=len(values);batch=config['batch_size'] if split=='train' else 32
            rows=np.lib.format.open_memmap(root/'tokens.npy',mode='w+',dtype='<u2',shape=(count,512));rows[:]=50256
            lengths=np.empty(count,dtype='<u2')
            for i in range(count):
                seq=values.sequence(i)
                if not 1<=len(seq)<=512:raise ValueError('invalid prepared fine-tuning length')
                rows[i,:len(seq)]=seq;lengths[i]=len(seq)
            rows.flush();del rows
            np.save(root/'lengths.npy',lengths,allow_pickle=False)
            np.save(root/'labels.npy',np.asarray(values.labels,dtype='u1'),allow_pickle=False)
            if split=='train':
                order=np.asarray(data.finetune_order(task)).reshape(config['epochs'],count)
                for epoch in order:
                    if not np.array_equal(np.sort(epoch),np.arange(count)):raise ValueError('incomplete fine-tuning permutation')
                per_epoch=count//batch
                indices=order[:,:per_epoch*batch].reshape(-1,batch)
                counts=np.full(len(indices),batch,dtype='<u2')
                boundaries=np.arange(config['epochs']+1,dtype='<u4')*per_epoch
            else:
                batches=(count+batch-1)//batch;flat=np.zeros(batches*batch,dtype='<u4');flat[:count]=np.arange(count)
                indices=flat.reshape(-1,batch);counts=np.full(batches,batch,dtype='<u2');counts[-1]=count-(batches-1)*batch
                boundaries=np.array([0,batches],dtype='<u4')
            np.save(root/'indices.npy',indices.astype('<u4'),allow_pickle=False)
            np.save(root/'counts.npy',counts,allow_pickle=False);np.save(root/'epoch_offsets.npy',boundaries,allow_pickle=False)
            print(task,split,count,'rows',len(counts),'batches',flush=True)
    for p in sorted(args.output.rglob('*.npy')):
        report['records'].append({'path':str(p.relative_to(args.output)),'sha256':sha256_file(p),'bytes':p.stat().st_size})
    atomic_json(args.output/'manifest.json',report)
    print(json.dumps({'prepared_manifest_sha256':sha256_file(args.output/'manifest.json'),'records':len(report['records'])}))


if __name__=='__main__':main()
