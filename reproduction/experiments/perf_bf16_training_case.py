"""BF16-state refresh of the accepted complete optimizer-update probe."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import profile_final as p
from perf_production_memory import ProductionProfileModel

parser=argparse.ArgumentParser()
parser.add_argument('--name',required=True)
parser.add_argument('--context',type=int,default=262144)
parser.add_argument('--arm')
parser.add_argument('--timed',type=int,default=3)
parser.add_argument('--trace',type=int,default=0)
a=parser.parse_args()
p.ProfileModel=ProductionProfileModel
p.ProfileModel.common_hash=lambda self:None
p.ProfileModel.full_parameter_hash=lambda self:None
if os.environ.get('PROFILE_BSDM_COMMIT'):
    p.RUNTIME_VARIANTS['compute']=dict(path=str(p.BSDM_SOURCE),commit=os.environ['PROFILE_BSDM_COMMIT'],status='retained compute candidate')
out=Path('outputs/perf-compute');out.mkdir(exist_ok=True,parents=True)
pending=out/(a.name+'.pending.json')
p.one_case(SimpleNamespace(arm=a.arm or f'bsdm_nvidia_n{a.context}',variant='compute',repeat=0,
    mode='train',width=2048,layers=16,batch=1,context=a.context,bsdm_heads=1,
    gdn_head_dim=64,gdn_expand_v=2.,compile_policy='auto',warmup=3,max_warmup=20,
    timed=a.timed,max_timed=a.timed,min_timed_seconds=0,trace_steps=a.trace,
    input=out/f'input-{a.context}.npz',output=pending))
row=json.loads(pending.read_text())
if not a.arm or a.arm.startswith('bsdm'):
    assert row['accounting']['logical_state_elements_per_example']==16*a.context*2048
    row['capacity_rows']=a.context
    row['selected_rows_per_role']=64
    assert row['model']['training_state_dtype'] == 'bfloat16'
    assert row['model']['routed_decay'] and row['model']['output_bias']
    assert row['model']['output_normalization'] == 'layer'
    row['replay']=dict(interval=64, interval_unit='events', snapshot_dtype='torch.bfloat16',
        recurrence_arithmetic='torch.float32', state_adjoint_dtype='torch.float32',
        forward_rounds_at_stored_boundaries=True, backward_value_tile=512,
        activation_retention='whole-block nonreentrant checkpoint hooks')
row['status']='ok'
row['memory_policy']=dict(block_checkpoint='nonreentrant full gradient',nested_ffn_checkpoint=False,
    loss_token_chunk=1024,context_chunking=False,truncated_gradients=False)
pending.write_text(json.dumps(row,indent=2)+'\n')
pending.rename(out/(a.name+'.json'))
