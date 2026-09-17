"""Public execution entrypoints around the extracted scientific programs."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

from .local_store import atomic

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'reproduction'
FLA_REVISION = '4b02d15d6a68700181b180235be62a9fb95d2a38'
MODELS = {'bdm':'bsdm','sdm':'sdm_native','attention':'attention','gdn1':'gdn1','gdn2':'gdn2_nvidia','mom':'mom_m4_k2_shared_ffn240'}


def call(command, **kwargs):
    print(shlex.join(map(str,command)), flush=True)
    subprocess.run(list(map(str,command)), check=True, **kwargs)


def setup(args):
    work = args.work.resolve()
    if args.profile == 'inputs':
        call([sys.executable,'-m','pip','install','-r',PACKAGE/'requirements-inputs.txt'])
        return
    import torch
    if torch.__version__ != '2.11.0+cu128':
        raise RuntimeError('Use the documented PyTorch 2.11.0 / CUDA 12.8 devel image.')
    requirements = 'requirements-decode-audit.txt' if args.profile=='performance' else 'requirements-quality.txt'
    call([sys.executable,'-m','pip','install','--extra-index-url','https://download.pytorch.org/whl/cu128','-r',PACKAGE/requirements])
    if args.profile == 'babylm':
        call([sys.executable,'-m','pip','install','-r',PACKAGE/'requirements-human.txt'])
    runtime = work/'runtime'
    runtime.mkdir(parents=True,exist_ok=True)
    # Public pinned source.
    source = runtime/'fla-source'
    if not source.exists():
        call(['git','clone','https://github.com/fla-org/flash-linear-attention.git',source])
        call(['git','-C',source,'checkout','--detach',FLA_REVISION])
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual != FLA_REVISION:
        raise ValueError('FLA source revision changed')
    call([sys.executable,'-m','pip','install','--no-deps','--target',runtime/'gdn2',source])
    call([sys.executable,'-m','pip','install','--no-deps','--target',runtime/'gdn1',
          'flash-linear-attention==0.5.2','fla-core==0.5.2'])


def prepare(args):
    work=args.work.resolve(); data=work/'inputs'/args.dataset
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    env['PYTHONPATH']=str(ROOT)
    if args.dataset=='wiki':
        call([sys.executable,'-m','reproduction.inputs.prepare_wiki','--output',data],env=env,cwd=ROOT)
    elif args.dataset=='recall':
        from .inputs.recall_generate import prepare_dataset
        from .experiments.data import Dataset
        if not (data/'manifest.json').exists():
            prepare_dataset(data,steps=30000,batch_size=32,eval_examples=2048,stream_seed=102337)
        Dataset(data,'recall')
    elif args.dataset=='memory':
        build=work/'memory-input-build'; build.mkdir(parents=True,exist_ok=True)
        env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(PACKAGE/'experiments')])
        if not (data/'manifest.json').exists():
            call([sys.executable,PACKAGE/'experiments/prepare_occupancy_tasks.py'],cwd=build,env=env)
            data.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(build/'prepared',data)
        from .experiments.occupancy_tasks_data import TaskDataset
        TaskDataset(data)
    else:
        from .inputs.prepare_babylm import prepare
        prepare(data)
    print(f'Prepared {args.dataset}: {data}')


def plan(args):
    experiment=args.experiment
    if args.factor_scale!='reference' and experiment!='init-scale': raise ValueError('--factor-scale applies only to init-scale')
    if args.coefficient and experiment!='memory': raise ValueError('--lambda applies only to memory')
    if args.benchmark!='wiki' and experiment!='init-scale': raise ValueError('--benchmark applies only to init-scale')
    if experiment=='init-scale' and args.model!='bdm': raise ValueError('init-scale compares BDM initializers')
    if experiment=='memory' and (args.model!='bdm' or args.seed!=0): raise ValueError('memory uses BDM seed zero')
    if experiment=='babylm' and (args.model not in ['bdm','sdm','attention'] or args.seed!=0): raise ValueError('BabyLM uses BDM/SDM/attention seed zero')
    if experiment=='performance' and (args.model=='mom' or args.seed!=0): raise ValueError('performance has no MoM or seed replicas')
    benchmark=args.benchmark if experiment=='init-scale' else experiment
    if experiment in ['wiki','recall','init-scale']:
        if args.model=='bdm':
            script='state_precision_train.py' if experiment=='init-scale' and args.factor_scale=='unit' else 'init_scale_train.py'
            cmd=[script,'--benchmark',benchmark,'--seed',str(args.seed)]
        elif args.model=='sdm': cmd=['native_sdm_train.py','--benchmark',benchmark,'--seed',str(args.seed)]
        else: cmd=['campaign_train.py','--benchmark',benchmark,'--arm',MODELS[args.model],'--seed',str(args.seed)]
        label=f'{experiment}-{benchmark}-{args.model}-{args.factor_scale}-s{args.seed}'
        dataset=benchmark
    elif experiment=='memory':
        cmd=['occupancy_tasks_train.py','--arm','bsdm_elastic' if args.coefficient else 'bsdm']
        if args.coefficient: cmd += ['--elastic',str(args.coefficient)]
        label=f'memory-lambda{args.coefficient:g}'; dataset='memory'
    elif experiment=='babylm':
        cmd=['-m','babylm_bsdm.train'];label=f'babylm-{args.model}-s0';dataset='babylm'
    else:
        family=MODELS[args.model];context=str(args.context)
        label=f'performance-{args.model}-{args.phase}-{context}';dataset='wiki'
        if args.phase=='train':
            arm=(f'bsdm_nvidia_n{context}' if args.model=='bdm' else f'sdm_native_k64_n{context}' if args.model=='sdm' else family)
            cmd=['perf_bf16_training_case.py','--name','result','--context',context,'--arm',arm,'--timed','5' if args.model=='bdm' else '3']
        elif args.phase=='prefill':cmd=['perf_recurrent_graph.py','--family',family,'--context',context,'--input','input.npz']
        else:cmd=['perf_decode_sustained.py','--family',family,'--context',context,'--input','input.npz','--output','result.json']
    return label,dataset,[sys.executable,*cmd]


def link(source,target):
    source=Path(source).resolve();target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        if target.resolve()!=source: raise ValueError(f'run source changed: {target}')
    elif target.exists():raise FileExistsError(target)
    else:target.symlink_to(source,target_is_directory=True)


def run(args):
    label,dataset,command=plan(args)
    work=args.work.resolve();out=work/label;workspace=out/'workspace'
    print(json.dumps(dict(experiment=label,dataset=str(work/'inputs'/dataset),output=str(out),command=command),indent=2))
    if args.dry_run:return
    out.mkdir(parents=True,exist_ok=True)
    with (out/'writer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(arguments={k:str(v) for k,v in vars(args).items() if k not in ['dry_run','command']},
                      sources=source_identity())
        identity_path=out/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise ValueError('Configuration or released source changed; choose a new --work directory')
        complete=out/'COMPLETE.json'
        if complete.exists():
            try:finished=json.loads(complete.read_text())
            except (json.JSONDecodeError,UnicodeDecodeError):finished={}
            if isinstance(finished,dict) and finished.get('status') in ['ok','oom']:
                print(f'Already completed: {out}');return
        atomic(identity_path,identity)
        shutil.copytree(PACKAGE/'experiments',workspace,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(PACKAGE/'control-sources.json',workspace/'sources.json')
        quality=args.experiment in ['wiki','recall','init-scale']
        bdm=PACKAGE/'vendor/quality-bdm' if quality else ROOT/'implementation'
        for src,dst in [(bdm,'dependencies/bsdm'),(PACKAGE/'vendor/sdm','dependencies/sdm-native'),
                        (PACKAGE/'vendor/shell','dependencies/sdm-quality'),(PACKAGE/'vendor/shell','dependencies/sdm'),
                        (PACKAGE/'vendor/gdn2','dependencies/nvidia-gdn2'),(work/'runtime/gdn1','gdn1-runtime')]:
            link(src,workspace/dst)
        env=dict(os.environ,PYTHONPATH=str(ROOT),BDM_RUN_DIR=str(out),
            BDM_INPUTS=json.dumps({key:str(work/'inputs'/key) for key in ['wiki','recall','memory','babylm']}),
            DATASET='memory',DSTACK_RUN_NAME=label,WANDB_RUN_ID=label,
            BSDM_SOURCE=str(bdm),PROFILE_BSDM_SOURCE=str(bdm),SDM_SOURCE=str(PACKAGE/'vendor/sdm/lingua/sparse_delta_memory'),
            BSDM_BABYLM_CONFIG=str(PACKAGE/f'babylm-{args.model}.json'),
            BDM_HUMAN_INPUTS=str(work/'inputs/babylm/human'),
            PROFILE_BSDM_COMMIT='ed51fd03ef83ff8ce50555a66481d878c2fc4a4f',
            PROFILE_NATIVE_SDM_COMMIT='bfe51564d1349200d21c8ad6507c248ef7f8ef5e',
            OMP_NUM_THREADS='4',MKL_CBWR='COMPATIBLE',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            TORCHINDUCTOR_COMPILE_THREADS='2',MAX_JOBS='2',PYTHONUNBUFFERED='1')
        if args.model=='gdn2' or args.experiment in ['performance','babylm']:
            if not (work/'runtime/gdn2/fla').is_dir():raise FileNotFoundError('Run the documented setup stage first')
            env['PYTHONPATH']=str(work/'runtime/gdn2')+os.pathsep+str(ROOT)
        if args.experiment=='performance':
            import numpy as np
            from .experiments.data import Dataset
            data=Dataset(work/'inputs/wiki','wiki');n=args.context
            seq=np.asarray(data.streams['train'][:n+8320],dtype=np.int64)
            np.savez(workspace/'input.npz',tokens=seq.reshape(1,-1))
            folder=workspace/'outputs/perf-compute';folder.mkdir(parents=True,exist_ok=True)
            np.savez(folder/f'input-{n}.npz',tokens=seq[:n].reshape(1,-1),targets=seq[1:n+1].reshape(1,-1))
            (workspace/'outputs/recurrent-graph').mkdir(parents=True,exist_ok=True)
            env.update(MEASURE_PREFILL='1',PERF_PREFILL_ONLY='1',PERF_GRAPH_ONLY='1',PERF_NO_LIVE_TAIL='1')
        if args.experiment=='performance':
            run_performance(command,workspace,env,out,args)
        else:
            call(command,cwd=workspace,env=env)


def source_identity(strict=False):
    declared=json.loads((PACKAGE/'extraction.json').read_text())
    rows=[]
    for row in declared['files']:
        path=ROOT/row['path']
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if strict and actual!=row['sha256']:raise ValueError(f"Released source changed: {row['path']}")
        rows.append([row['path'],actual])
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def run_performance(command,workspace,env,out,args):
    # Retain actual failures; only a typed CUDA OOM is a capacity measurement.
    print(shlex.join(command),flush=True)
    with (out/'console.log').open('w') as log:
        process=subprocess.Popen(command,cwd=workspace,env=env,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    if code:
        console=(out/'console.log').read_text()
        if re.search(r'^torch\.(?:cuda\.)?OutOfMemoryError: CUDA out of memory',console,re.MULTILINE):
            result={'status':'oom','phase':args.phase,'context':args.context,'model':args.model}
        else:raise subprocess.CalledProcessError(code,command)
    else:
        relative={'train':'outputs/perf-compute/result.json','decode':'result.json',
                  'prefill':f'outputs/recurrent-graph/{MODELS[args.model]}-{args.context}.json'}[args.phase]
        result=json.loads((workspace/relative).read_text())
        if result.get('status')!='ok':raise RuntimeError(f"Performance case did not finish: {result.get('status')}")
    atomic(out/'COMPLETE.json',result)
