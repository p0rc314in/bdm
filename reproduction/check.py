"""Bounded checks of preparation semantics, portable commands and saved recovery."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace


def check(with_torch=False):
    root=Path(__file__).resolve().parents[1]
    for path in (root/'reproduction').rglob('*.py'):ast.parse(path.read_text(),filename=str(path))
    from .runner import plan, source_identity
    source_identity(strict=True)
    for experiment,models in [('wiki',['bdm','sdm','attention','gdn1','gdn2','mom']),('recall',['bdm','sdm']),('memory',['bdm']),('babylm',['bdm','sdm','attention']),('performance',['bdm','sdm','gdn1','gdn2','attention'])]:
        for model in models:
            for phase in ['train','prefill','decode'] if experiment=='performance' else ['train']:
                args=SimpleNamespace(experiment=experiment,model=model,seed=0,benchmark='wiki',factor_scale='reference',coefficient=0.,phase=phase,context=8192)
                _,_,cmd=plan(args)
                assert (root/'reproduction/experiments'/cmd[1]).is_file() or cmd[1:3]==['-m','babylm_bsdm.train']
    from . import local_store as store
    with tempfile.TemporaryDirectory() as temp:
        old=dict(os.environ)
        try:
            os.environ['BDM_RUN_DIR']=str(Path(temp)/'run')
            source=Path(temp)/'checkpoint'; source.write_bytes(b'first recovery')
            with store.init(config={'seed':0}) as run:
                a=store.Artifact('training-recovery','checkpoint');a.add_file(source,'checkpoint.pt')
                first=run.log_artifact(a,aliases=['latest'])
                source.write_bytes(b'second recovery')
                second=run.log_artifact(a,aliases=['latest'])
                assert Path(first.download(),'checkpoint.pt').read_bytes()==b'first recovery'
            with store.init(config={'seed':0}) as run:
                assert run.resumed
                latest=[a for a in run.logged_artifacts() if 'latest' in a.aliases]
                assert len(latest)==1 and Path(latest[0].download(),'checkpoint.pt').read_bytes()==b'second recovery'
                original=store.atomic
                def interrupted(path,value):
                    if Path(path).name=='artifacts.json':raise OSError('simulated interrupted catalog write')
                    original(path,value)
                store.atomic=interrupted
                try:run.log_artifact(a,aliases=['latest'])
                except OSError:pass
                finally:store.atomic=original
            with store.init(config={'seed':0}) as run:
                recovered=run.log_artifact(a,aliases=['latest'])
                Path(recovered.download(),'checkpoint.pt').write_bytes(b'corrupt')
                try:recovered.download()
                except ValueError:pass
                else:raise AssertionError('corruption not rejected')
        finally:
            os.environ.clear();os.environ.update(old)
    import numpy as np
    from .inputs.recall_generate import CONDITIONS, generate_batch, balanced_schedule
    for condition in CONDITIONS:
        x,y=generate_batch(np.random.default_rng(102337),2,condition)
        assert x.shape==(2,condition['sequence_length']) and y.shape==(2,16)
        assert y.max()<192
    from .experiments.occupancy_tasks_data import check_semantics
    check_semantics()
    from .inputs.bucket_babylm import prepare
    with tempfile.TemporaryDirectory() as temp:
        values=[np.array([1,2,3]),np.arange(100)]
        meta=prepare('check',lambda i:values[i],np.array([0,3,103]),np.array([1,10]),Path(temp))
        covered=[];scored=0
        for bucket in meta['buckets']:
            directory=Path(temp)/'check'/str(bucket['length'])
            ids=np.load(directory/'indices.npy');covered.extend(ids[ids>=0].tolist())
            scored+=int(np.load(directory/'masks.npy').sum())
        assert sorted(covered)==[0,1] and scored==92
    if with_torch:
        import sys,torch
        torch.set_num_threads(1)
        sys.path[:0]=[str(root/'reproduction/experiments'),str(root/'reproduction/vendor/quality-bdm')]
        from model import ComparisonModel
        model=ComparisonModel('wiki','bsdm_nvidia_n2048',seed=0,backend='reference',compact_replay=False)
        assert model.accounting()['trainable']==15330504
        assert model.accounting()['learned_initial_state']==98304
        reference=ComparisonModel('wiki','bsdm_final_n2048',seed=0,backend='reference',compact_replay=False)
        assert model.common_hash()==reference.common_hash()
    print(json.dumps({'status':'passed','checks':['Python syntax','experiment command paths','artifact roundtrip and resume','interrupted publication recovery','corruption rejection','30 Recall conditions','20 occupancy conditions','BabyLM bucket masks'],'cpu_model':with_torch,'cuda':False}))
