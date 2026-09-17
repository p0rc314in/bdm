"""Build the full BabyLM input tree from pinned, public source datasets."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

from .babylm.spec import EVALUATOR_REVISION


def require_evaluator_revision(path):
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=EVALUATOR_REVISION:raise ValueError('official evaluator revision changed')


def invoke(module,*args):
    subprocess.run([sys.executable,'-m',module,*map(str,args)],check=True)


def prepare(output):
    from .babylm.prepare import prepare_training, prepare_core
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    raw=output/'raw'; raw.mkdir(exist_ok=True)
    evaluator=output/'evaluator'
    if not evaluator.exists():
        subprocess.run(['git','clone','https://github.com/babylm-org/babylm-eval.git',str(evaluator)],check=True)
        subprocess.run(['git','-C',str(evaluator),'checkout','--detach',EVALUATOR_REVISION],check=True)
    require_evaluator_revision(evaluator)
    training=raw/'babylm2026-strict-gpt2-causal-t2048-10epoch-v1'
    prepare_training(training);prepare_core(training)
    for stage in ['checkpoints','full-ewok']:
        invoke('reproduction.inputs.babylm.prepare','--output',training,'--stage',stage,'--evaluator-root',evaluator)
    fast=raw/'babylm2026-strict-gpt2-checkpoint-evaluation-v1'
    if not fast.exists():fast.symlink_to(training/'checkpoint-evaluation',target_is_directory=True)
    ewok=raw/'full-ewok-a'
    if not ewok.exists():ewok.symlink_to(training/'full-ewok',target_is_directory=True)
    if not (output/'training').exists(): (output/'training').symlink_to(training,target_is_directory=True)
    if not (output/'inference/manifest.json').exists():
        invoke('reproduction.inputs.bucket_babylm','--source',raw,'--output',output/'inference')
    if not (output/'finetune/manifest.json').exists():
        invoke('reproduction.inputs.batch_babylm','--source',training,'--output',output/'finetune')
    human=output/'human'
    if not human.exists():
        human.mkdir()
        shutil.copyfile(fast/'manifest.json',human/'manifest.json')
        for section in ['reading','aoa']:
            shutil.copytree(fast/section,human/section)
        for relative in ['reading/reading_data.csv','aoa/cdi_human.csv']:
            shutil.copyfile(training/'official-source/evaluation_data/full_eval'/relative,human/Path(relative).name)
    from .babylm.io import sha256_file
    manifests={str(p.relative_to(output)):sha256_file(p) for p in
               [output/'training/manifest.json',output/'inference/manifest.json',output/'finetune/manifest.json',human/'manifest.json']}
    (output/'manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')
