"""The final comparison: native control mixers in the established shared shell."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from model import ComparisonModel, NVIDIA_FINAL_ARMS, ROOT

ARMS = (*NVIDIA_FINAL_ARMS, 'attention', 'gdn1', 'gdn2_nvidia', 'mom_m4_k2_shared_ffn240')
sys.path.insert(0, str(ROOT/'dependencies/sdm-quality'))


class NativeControl(nn.Module):
    def __init__(self, arm, index, seed):
        super().__init__()
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed*100000+1000+index)
            if arm=='attention':
                from lingua.transformer import Attention,RotaryEmbedding
                self.mixer=Attention(128,32,4,4,10000.)
                self.mixer.reset_parameters()
                self.rope=RotaryEmbedding(10000.,32,2048)
                self.record=dict(heads=4,head_dim=32,rope_theta=10000.,backend='Flash SDPA')
            elif arm=='gdn1':
                from fla.layers import GatedDeltaNet
                cfg=dict(hidden_size=128,head_dim=32,num_heads=4,num_v_heads=4,
                    expand_v=1.,mode='chunk',use_short_conv=True,conv_size=4,
                    conv_bias=False,allow_neg_eigval=False,norm_eps=1e-5,layer_idx=index,use_gate=True)
                self.mixer=GatedDeltaNet(**cfg)
                self.record=cfg
            else:
                raise ValueError(arm)

    def forward(self, hidden):
        if self.arm=='attention':
            return self.mixer(hidden,self.rope(seqlen=hidden.shape[1]),mask='causal',attn_impl='sdpa')
        return self.mixer(hidden)[0]


class MoMControl(nn.Module):
    def __init__(self,index,seed):
        super().__init__()
        from control_adapters.mom_patches import install
        native=install()
        self.record=dict(hidden_size=128,num_heads=4,head_dim=32,expand_v=1.,
            num_memories=4,topk=2,shared_mem=True,single_kv_proj=False,use_output_gate=True,
            use_short_conv=True,conv_size=4,conv_bias=False,mode='chunk',norm_eps=1e-5)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed*100000+1000+index)
            self.mixer=native(layer_idx=index,**self.record)
        self.router_logits=None

    @torch.compiler.disable
    def forward(self,hidden):
        output,_,_,self.router_logits=self.mixer(hidden,use_cache=False)
        return output


class CampaignModel(ComparisonModel):
    def __init__(self,benchmark,arm,*,seed=0,backend='role',compact_replay=None):
        assert arm in ARMS
        if arm.startswith('bsdm_') or arm=='gdn2_nvidia':
            super().__init__(benchmark,arm,seed=seed,backend=backend,compact_replay=compact_replay)
            return
        super().__init__(benchmark,'bsdm_final_n1024',seed=seed,backend=backend,compact_replay=compact_replay)
        self.arm=arm
        for i,block in enumerate(self.layers):
            if arm=='mom_m4_k2_shared_ffn240':
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed*100000+1000+i)
                    block.feed_forward=type(block.feed_forward)(128,360,16,None,recompute_activations=True)
                    block.feed_forward.reset_parameters()
                assert block.feed_forward.hidden_dim==240
                block.attention=MoMControl(i,seed)
            else:
                block.attention=NativeControl(arm,i,seed)

    def forward(self,tokens,target=None):
        if self.arm!='mom_m4_k2_shared_ffn240':
            return super().forward(tokens,target=target)
        logits=super().forward(tokens)
        if target is None:return logits
        nll=F.cross_entropy(logits.float().flatten(0,-2),target.flatten())
        from fla.models.mom.modeling_mom import load_balancing_loss_func
        routers=tuple(block.attention.router_logits for block in self.layers)
        aux=load_balancing_loss_func(routers,num_experts=4,top_k=2,
                                     attention_mask=(target!=-100) if self.benchmark=='wiki' else None)
        return nll+.01*aux,nll,aux

    def resolved_model(self):
        record=self.config.record()
        if self.arm.startswith('bsdm_'):return record
        record.pop('memory')
        record.update(arm=self.arm,ffn_width=self.layers[0].feed_forward.hidden_dim,
                      mixer=getattr(self.layers[0].attention,'record',
                          dict(source='NVIDIA GDN2',heads=4,key_width=32,value_width=32)))
        return record

    def accounting(self):
        result=super().accounting()
        if self.arm=='attention':result['logical_recurrent_elements_per_example']=0
        elif self.arm=='mom_m4_k2_shared_ffn240':result['logical_recurrent_elements_per_example']=8*5*4*32*32
        return result


def validate_common(network,benchmark,arm,seed,options):
    if arm in NVIDIA_FINAL_ARMS:
        capacity = NVIDIA_FINAL_ARMS[arm][0]
        with torch.random.fork_rng(devices=[]):
            reference = ComparisonModel(benchmark, f'bsdm_final_n{capacity}', seed=seed, **options)
        actual, expected = network.state_dict(), reference.state_dict()
        assert list(actual) == list(expected)
        assert all(torch.equal(actual[name], expected[name]) for name in actual)
        assert list(dict(network.named_parameters())) == list(dict(reference.named_parameters()))
        assert all(getattr(p, '_no_weight_decay', False) == getattr(q, '_no_weight_decay', False)
                   for p,q in zip(network.parameters(), reference.parameters(), strict=True))
        assert network.accounting() == reference.accounting()
        a, b = network.config.record(), reference.config.record()
        assert a['memory'].pop('qk_normalization') == 'nvidia'
        assert b['memory'].pop('qk_normalization') == 'original'
        assert a == b
        return reference.common_hash()
    with torch.random.fork_rng(devices=[]):
        reference=ComparisonModel(benchmark,'bsdm_n2048_bias_only',seed=seed,**options)
    a=dict(network.named_parameters()); b=dict(reference.named_parameters())
    for name,value in b.items():
        if '.attention.' in name or (arm=='mom_m4_k2_shared_ffn240' and '.feed_forward.' in name):continue
        assert torch.equal(a[name],value), name
    if seed==0 and arm=='mom_m4_k2_shared_ffn240':
        expected=('b4b49f87e75bf5e5c44e5005b01abe64e23de74b5f2ab00479a9e923e9c29b01' if benchmark=='recall'
                  else None)
        if expected is not None:assert network.common_hash()==expected
    return network.common_hash()
