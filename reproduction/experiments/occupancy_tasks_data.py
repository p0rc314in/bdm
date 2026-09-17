"""Explicit memory-demand diagnostic; separate from Adaptive Recall v1."""
import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL = 'bsdm-workload-occupancy-v1'
LENGTHS = (1024, 2048, 4096, 8192, 16384)
TASKS = ('local', 'delay', 'overwrite', 'growing')
KEYS, VALUES, QUERIES, ROLE_STRIDE = 2048, 64, 16, 2048 * 64
STEPS, TOKENS_PER_STEP, EXAMPLES = 30000, 16384, 2048
CONDITIONS = [dict(index=i*5+j, id=f'{task}_t{length}', family=task,
                   sequence_length=length, bindings=length//8,
                   retained_values=length//8 if task=='growing' else 32)
              for i,task in enumerate(TASKS) for j,length in enumerate(LENGTHS)]


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def generate(rng, condition, batch):
    """Packed (role, key identity/slot, value) symbols and withheld answers."""
    length, task = condition['sequence_length'], condition['family']
    count = length//8
    # Real random distractors have a distinct role; no repeated padding.
    tokens = (ROLE_STRIDE + rng.integers(0, ROLE_STRIDE, (batch, length), dtype=np.uint32))
    positions = np.linspace(1, length-QUERIES-1, count, dtype=np.int64)
    assert np.unique(positions).size == count
    labels = np.empty((batch, QUERIES), dtype=np.uint8)
    ages = np.empty((batch, QUERIES), dtype=np.int32)
    for b in range(batch):
        keys = rng.permutation(KEYS)[:count]
        values = rng.integers(0, VALUES, count, dtype=np.uint32)
        if task == 'overwrite':
            live = keys[:32].copy()
            keys = np.concatenate([rng.permutation(live) for _ in range(count//32)])
        encoded = keys.astype(np.uint32)*VALUES + values
        if task == 'delay':
            encoded[32:] += ROLE_STRIDE
        tokens[b, positions] = encoded
        if task in ('local', 'overwrite'):
            query_indices = count-32 + rng.choice(32, QUERIES, replace=False)
        elif task == 'delay':
            query_indices = rng.choice(32, QUERIES, replace=False)
        else:
            # One uniformly chosen fact from each of 16 age bins.
            size = count//QUERIES
            query_indices = np.arange(QUERIES)*size + rng.integers(0, size, QUERIES)
            rng.shuffle(query_indices)
        tokens[b, -QUERIES:] = 2*ROLE_STRIDE + keys[query_indices].astype(np.uint32)*VALUES
        labels[b] = values[query_indices]
        ages[b] = np.arange(length-QUERIES,length) - positions[query_indices]
    tokens[:,0] = (3+TASKS.index(task))*ROLE_STRIDE
    return tokens, labels, ages


def check_semantics():
    for c in CONDITIONS:
        x,y,ages=generate(np.random.default_rng(8129+c['index']),c,3)
        assert (x[:,-16:]//ROLE_STRIDE == 2).all()
        assert (x[:,-16:]%VALUES == 0).all()  # no target value in query
        assert (ages>0).all()
        for row,labels in zip(x,y,strict=True):
            table={}; recent=[]
            for pos,token in enumerate(row[:-16]):
                role,local=divmod(int(token),ROLE_STRIDE)
                if role==0:
                    key,value=divmod(local,VALUES)
                    table[key]=(value,pos);recent.append((key,value))
            assert len(table)==c['retained_values'] if c['family'] in ('delay','overwrite') else len(table)==c['bindings']
            q=(row[-16:]%ROLE_STRIDE)//VALUES
            assert [table[int(key)][0] for key in q] == labels.tolist()
            if c['family'] in ('local','overwrite'):
                assert all(int(key) in dict(recent[-32:]) for key in q)
            if c['family']=='growing':
                assert sum(int(key) in dict(recent[-32:]) for key in q)<=4
            # Counterfactual: changing the final write changes the oracle answer.
            key=int(q[0]);old,pos=table[key];edited=row.copy()
            edited[pos]=key*VALUES+(old+1)%VALUES
            assert int(edited[pos])%VALUES != int(labels[0])
    print('SEMANTICS_OK: all20 conditions, independent labels, latest writes, age coverage',flush=True)


class TaskDataset:
    def __init__(self, root):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        assert self.manifest['protocol']==PROTOCOL and self.manifest['conditions']==CONDITIONS
        self.conditions=CONDITIONS;self.arrays={}
        for name,r in self.manifest['records'].items():
            path=self.root/r['path'];assert sha256(path)==r['sha256'],name
            self.arrays[name]=np.memmap(path,mode='r',dtype=r['dtype'],shape=tuple(r['shape']))

    def train_batch(self, step):
        c=CONDITIONS[int(self.arrays['train_condition_ids'][step])];t=c['sequence_length']
        start,end=self.arrays['train_label_offsets'][step:step+2]
        return (np.array(self.arrays['train_tokens'][step],dtype=np.int64).reshape(-1,t),
                np.array(self.arrays['train_labels'][int(start):int(end)],dtype=np.int64).reshape(-1,QUERIES),c)

    def evaluation_batches(self, split, examples=EXAMPLES):
        assert 0<examples<=EXAMPLES
        for c in CONDITIONS:
            i,t=c['index'],c['sequence_length'];start,end=self.arrays[f'{split}_offsets'][i:i+2]
            xs=self.arrays[f'{split}_tokens'][int(start):int(end)].reshape(EXAMPLES,t)
            batch=min(8,TOKENS_PER_STEP//t)
            for begin in range(0,examples,batch):
                stop=min(begin+batch,examples)
                yield (np.array(xs[begin:stop],dtype=np.int64),
                       np.array(self.arrays[f'{split}_labels'][i,begin:stop],dtype=np.int64),
                       np.array(self.arrays[f'{split}_ages'][i,begin:stop]),c,begin)


if __name__=='__main__':
    check_semantics()
