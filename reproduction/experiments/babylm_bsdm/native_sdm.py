"""Original SDM mixer in the matched BabyLM shell; no residual router."""
from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import sys
from types import MethodType
import torch
from torch import nn
from control_adapters.sdm_recall_padding import pad_parallel_time

SOURCE = Path(os.environ.get('SDM_SOURCE', Path(__file__).resolve().parents[1]/'dependencies/sdm-native/lingua/sparse_delta_memory'))
COMMIT = 'bfe51564d1349200d21c8ad6507c248ef7f8ef5e'
VENDORED_COMMIT = COMMIT


def native_types():
    if 'native_sdm' not in sys.modules:
        spec = importlib.util.spec_from_file_location('native_sdm', SOURCE/'__init__.py', submodule_search_locations=[str(SOURCE)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['native_sdm'] = module
        spec.loader.exec_module(module)
    module = sys.modules['native_sdm']
    return module.SparseDeltaMemory, module.SparseDeltaMemoryArgs


@torch.compiler.disable
def padded_native_write_read(self, memory, k_idx, k_val, v, beta, g, q_idx, q_val, grad_final_memory=None):
    # Same correctness adapter as the paper's native SDM control. Short evaluator
    # buckets use native decode; longer chunks append state-neutral events only.
    time = k_idx.shape[1]
    if not self.training and time <= 64:
        from native_sdm.memory_ops import fused_decode_step
        outputs=[]
        for position in range(time):
            outputs.append(fused_decode_step(memory,k_idx[:,position],k_val[:,position],v[:,position],
                beta[:,position],g[:,position],q_idx[:,position],q_val[:,position],
                use_delta_rule=True,normalize_memory=False,key_weighted_decay=self.args.key_weighted_decay))
        return torch.stack(outputs,dim=1),memory
    chunk_size=min(self.args.memory_block_size,1 << (time-1).bit_length())
    inputs=pad_parallel_time((k_idx,k_val,v,beta,g,q_idx,q_val),chunk_size=chunk_size,rows_per_bank=self.slots_per_head)
    readings,terminal=type(self).gated_write_read(self,memory,*inputs,grad_final_memory=grad_final_memory)
    return readings[:,:time],terminal


class NativeMixer(nn.Module):
    def __init__(self,index,config):
        super().__init__()
        cls,args_cls=native_types()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed']*100000+1000+index)
            args=args_cls(dim=config['width'],num_heads=1,slots_per_head=config['memory_rows'],
                num_reads=64,num_writes=64,memory_block_size=256,norm_eps=1e-6,
                read_act='Softmax',write_act='Softmax',normalize_readings=True,
                backprop_on_memory=True,output_gate=True)
            self.mixer=cls(args,layer_id=index)
            self.mixer.init_weights()
        self.mixer.gated_write_read=MethodType(padded_native_write_read,self.mixer)
        self.record=dict(**asdict(args),upstream_commit=COMMIT,vendored_commit=VENDORED_COMMIT,
                         shared_residual_router=False,initial_memory='full learned table')
        assert self.mixer.product_key_rows*self.mixer.product_key_columns==config['memory_rows']
        assert self.mixer.memory.numel()==config['memory_rows']*config['width']

    def forward(self,hidden):
        return self.mixer(hidden)[0]
