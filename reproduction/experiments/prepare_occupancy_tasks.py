"""Prepare once on a managed CPU worker and preserve the immutable dataset."""
import json
import os
from pathlib import Path
import numpy as np
from reproduction import local_store as wandb
from occupancy_tasks_data import *


def main():
    check_semantics()
    root=Path('prepared');root.mkdir(exist_ok=True)
    records={}
    def array(name,dtype,shape):
        path=root/(name+'.bin')
        records[name]=dict(path=path.name,dtype=dtype,shape=list(shape))
        return np.memmap(path,mode='w+',dtype=dtype,shape=shape)
    schedule=np.resize(np.arange(20,dtype=np.uint8),STEPS)
    np.random.default_rng(102337).shuffle(schedule)
    a=array('train_condition_ids','uint8',(STEPS,));a[:]=schedule;a.flush()
    batch=np.array([TOKENS_PER_STEP//CONDITIONS[int(i)]['sequence_length'] for i in schedule])
    offsets=np.r_[0,np.cumsum(batch*QUERIES)].astype(np.int64)
    a=array('train_label_offsets','int64',offsets.shape);a[:]=offsets;a.flush()
    xs=array('train_tokens','uint32',(STEPS,TOKENS_PER_STEP))
    ys=array('train_labels','uint8',(int(offsets[-1]),))
    rng=np.random.default_rng(102338)
    for step,index in enumerate(schedule):
        x,y,_=generate(rng,CONDITIONS[int(index)],int(batch[step]));xs[step]=x.ravel()
        ys[offsets[step]:offsets[step+1]]=y.ravel()
        if (step+1)%3000==0: print('PREPARE training',step+1,flush=True)
    xs.flush();ys.flush();del xs,ys
    for split,seed in (('validation',10102337),('test',20102337)):
        offsets=np.r_[0,np.cumsum([EXAMPLES*c['sequence_length'] for c in CONDITIONS])].astype(np.int64)
        a=array(split+'_offsets','int64',offsets.shape);a[:]=offsets;a.flush()
        xs=array(split+'_tokens','uint32',(int(offsets[-1]),))
        ys=array(split+'_labels','uint8',(20,EXAMPLES,QUERIES));ages=array(split+'_ages','int32',(20,EXAMPLES,QUERIES))
        rng=np.random.default_rng(seed)
        for c in CONDITIONS:
            i,t=c['index'],c['sequence_length']
            for begin in range(0,EXAMPLES,16):
                x,y,age=generate(rng,c,16);start=int(offsets[i])+begin*t
                xs[start:start+x.size]=x.ravel();ys[i,begin:begin+16]=y;ages[i,begin:begin+16]=age
            print('PREPARE',split,c['id'],flush=True)
        xs.flush();ys.flush();ages.flush();del xs,ys,ages
    for r in records.values():
        p=root/r['path'];r.update(bytes=p.stat().st_size,sha256=sha256(p))
    manifest=dict(protocol=PROTOCOL,conditions=CONDITIONS,records=records,steps=STEPS,
        tokens_per_update=TOKENS_PER_STEP,train_tokens=STEPS*TOKENS_PER_STEP,
        train_queries=int(sum(batch)*QUERIES),eval_examples_per_condition=EXAMPLES,
        codec=dict(keys=KEYS,values=VALUES,role_stride=ROLE_STRIDE,roles=['binding','distractor','query',*TASKS]),
        source='occupancy_tasks_data.py; semantic embedding pattern from adopted Adaptive Recall',
        semantics_checked=True)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DATASET_READY', sha256(root/'manifest.json'), flush=True)

if __name__=='__main__': main()
