"""Canonical final BSDM and imported controls in one shared transformer shell."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import os
import re
from pathlib import Path
import sys
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
BSDM_SOURCE=Path(os.environ.get('PROFILE_BSDM_SOURCE',str(ROOT/'dependencies/bsdm'))).resolve()
sys.path[:0]=[str(BSDM_SOURCE),str(ROOT/'dependencies/sdm')]
from bsdm import BSDMConfig, BSDMLanguageModel, LanguageModelConfig
import bdm
assert Path(bdm.__file__).resolve().parent.parent==BSDM_SOURCE, 'Wrong BSDM source imported'
from model import nvidia_layer

ARMS=('gdn1','gdn2_nvidia','attention')
ARM_LABELS={
    'sdm_native_k64_n16384':'Native SDM K64/N16384',
    'bsdm_nvidia_n32768':'BSDM N32768',
    'bsdm_nvidia_n65536':'BSDM N65536',
    'gdn1':'GDN1','gdn2_nvidia':'NVIDIA GDN2','attention':'Attention',
}
NATIVE_SDM_COMMIT='210b2260babf3e437cc5198b063c0a43bb1bebf0'


def native_sdm_layer():
    # Import the pinned upstream implementation directly, under a separate
    # package name so the shell's lingua package cannot shadow its kernels.
    name='native_sdm'
    if name not in sys.modules:
        source=ROOT/'dependencies/sdm-native/lingua/sparse_delta_memory'
        spec=importlib.util.spec_from_file_location(name,source/'__init__.py',
            submodule_search_locations=[str(source)])
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    module=sys.modules[name]
    return module.SparseDeltaMemory,module.SparseDeltaMemoryArgs

class Control(nn.Module):
    def __init__(self,arm,index,width,context,*,gdn_head_dim=128,gdn_expand_v=1.):
        super().__init__()
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1000+index)
            if arm.startswith('sdm_'):
                SparseDeltaMemory,SparseDeltaMemoryArgs=native_sdm_layer()
                rows=int(arm.rsplit('_n',1)[1])
                cfg=SparseDeltaMemoryArgs(dim=width,num_heads=width//512,slots_per_head=rows,
                    num_reads=64,num_writes=64,memory_block_size=256,norm_eps=1e-5,
                    read_act='Softmax',write_act='Softmax',normalize_readings=True,
                    backprop_on_memory=True,output_gate=True)
                self.mixer=SparseDeltaMemory(cfg,layer_id=index)
                self.mixer.init_weights()
                self.record=asdict(cfg)
                self.record['implementation']=dict(commit=os.environ.get('PROFILE_NATIVE_SDM_COMMIT',NATIVE_SDM_COMMIT),
                    upstream='183e7df809131b80ad4393741029d0f20fc3640b',
                    changes=os.environ.get('PROFILE_NATIVE_SDM_CHANGES','BF16 recurrence correctness only'),
                    layer_sha256=hashlib.sha256((ROOT/'dependencies/sdm-native/lingua/sparse_delta_memory/layer.py').read_bytes()).hexdigest(),
                    kernel_sha256=hashlib.sha256((ROOT/'dependencies/sdm-native/lingua/sparse_delta_memory/memory_ops.py').read_bytes()).hexdigest())
            elif arm=='attention':
                from lingua.transformer import Attention,RotaryEmbedding
                self.mixer=Attention(width,64,width//64,width//64,10000.)
                self.mixer.reset_parameters()
                self.rope=RotaryEmbedding(10000.,64,context)
                self.record=dict(heads=width//64,head_dim=64,rope_theta=10000.,backend='PyTorch Flash SDPA')
            else:
                if arm=='gdn1':
                    # GDN1 uses the same FLA 0.5.2 implementation as the quality
                    # controls. NVIDIA GDN2 needs an older, incompatible helper
                    # API, so isolate this import in GDN1's fresh subprocess.
                    runtime=ROOT/'gdn1-runtime'
                    assert runtime.is_dir(), 'Install the pinned GDN1 runtime first'
                    if str(runtime) not in sys.path:sys.path.insert(0,str(runtime))
                    import fla
                    assert Path(fla.__file__).resolve().is_relative_to(runtime.resolve())
                    assert importlib.metadata.version('flash-linear-attention')=='0.5.2'
                    assert importlib.metadata.version('fla-core')=='0.5.2'
                    from fla.layers import GatedDeltaNet
                    cls=GatedDeltaNet
                else: cls=nvidia_layer()
                cfg=dict(hidden_size=width,num_heads=width//128,num_v_heads=width//128,head_dim=gdn_head_dim,
                    expand_v=gdn_expand_v,mode='chunk',use_short_conv=True,conv_size=4,
                    conv_bias=False,allow_neg_eigval=False,norm_eps=1e-5,layer_idx=index)
                if arm=='gdn1':cfg['use_gate']=True
                self.mixer=cls(**cfg)
                self.record=dict(cfg)
                if arm=='gdn1':
                    self.record['runtime']=dict(fla='0.5.2',fla_core='0.5.2',
                        path=str(runtime),layer_sha256=hashlib.sha256(
                            (runtime/'fla/layers/gated_deltanet.py').read_bytes()).hexdigest())

    def forward(self,x):
        if self.arm=='attention':
            return self.mixer(x,self.rope(seqlen=x.shape[1]),mask='causal',attn_impl='sdpa')
        return self.mixer(x)[0]

class ProfileModel(BSDMLanguageModel):
    def __init__(self,arm,context=16384,*,width=512,layers=16,gdn_head_dim=128,gdn_expand_v=1.,bsdm_heads=None):
        assert arm in ARMS or re.fullmatch(r'(?:sdm_native_k64|bsdm_nvidia)_n[0-9]+',arm)
        ARM_LABELS.setdefault(arm,arm)
        assert width>=512 and width%512==0 and layers>0
        rows=int(arm.rsplit('n',1)[1]) if arm.startswith('bsdm_') else 4096
        cfg=LanguageModelConfig(memory=BSDMConfig(dim=width,bank_count=rows//8,bank_size=8,
            selected_banks=8,value_width=width,num_heads=width//512 if bsdm_heads is None else bsdm_heads,
            qk_normalization='nvidia'),vocab_size=50257,layers=layers,
            seed=0,recompute_ffn=True,compact_replay=False)
        super().__init__(cfg,backend='role')
        self.arm=arm
        if arm.startswith('bsdm_'):
            self.mixer_record=cfg.memory.record()
        else:
            for i,block in enumerate(self.layers):
                block.attention=Control(arm,i,width,context,gdn_head_dim=gdn_head_dim,gdn_expand_v=gdn_expand_v)
            self.mixer_record=self.layers[0].attention.record
        self.profile_width=width;self.profile_layers=layers
        self.gdn_head_dim=gdn_head_dim;self.gdn_expand_v=gdn_expand_v
        self.ffn_width=self.layers[0].feed_forward.w1.out_features
        assert all(block.feed_forward.w1.out_features==self.ffn_width for block in self.layers)

    def accounting(self):
        ps=dict(self.named_parameters());total=sum(p.numel() for p in ps.values())
        lexical=self.tok_embeddings.weight.numel()+self.output.weight.numel()
        initial=sum(p.numel() for name,p in ps.items() if getattr(p,'_sdm_memory_bank',False)
                    or (self.arm.startswith('sdm_native_') and name.endswith('.mixer.memory')))
        state=(self.profile_layers*int(self.arm.rsplit('n',1)[1])*self.profile_width if self.arm.startswith(('sdm_','bsdm_'))
               else self.profile_layers*(self.profile_width//128)*self.gdn_head_dim*int(self.gdn_head_dim*self.gdn_expand_v) if self.arm.startswith('gdn') else 0)
        return dict(parameters=total,input_embedding=self.tok_embeddings.weight.numel(),
            output_embedding=self.output.weight.numel(),learned_initial_memory=initial,
            processing_parameters=total-lexical-initial,logical_state_elements_per_example=state,
            input_output_tied=False,positional_parameters=0)

    def common_hash(self):
        h=hashlib.sha256()
        for name,p in sorted(self.named_parameters()):
            if '.attention.' not in name:
                h.update(name.encode());h.update(p.detach().cpu().float().numpy().tobytes())
        return h.hexdigest()

    def full_parameter_hash(self):
        h=hashlib.sha256()
        for name,p in sorted(self.named_parameters()):
            h.update(name.encode());h.update(p.detach().cpu().float().numpy().tobytes())
        return h.hexdigest()
