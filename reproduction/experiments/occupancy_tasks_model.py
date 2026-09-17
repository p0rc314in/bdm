"""Canonical final BSDM with the shared task adapter."""
import os
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, os.environ.get('BSDM_SOURCE', str(Path(__file__).resolve().parent/'dependencies/bsdm')))
from bsdm import BSDMConfig, BSDMLanguageModel, LanguageModelConfig
from model import ComparisonModel
from recall import initialize_task_embedding
from elastic_bank_regularizer import bank_occupancy
from occupancy_tasks_data import ROLE_STRIDE


class TaskEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity=nn.Embedding(192,64)
        self.slot=nn.Embedding(16,64)
        self.role=nn.Embedding(7,128)

    def forward(self,tokens):
        role=tokens//ROLE_STRIDE
        local=tokens%ROLE_STRIDE
        key,value=local//64,local%64
        first=(self.identity(key//16)+self.slot(key%16))*(role<3).unsqueeze(-1)
        second=self.identity(value)*(role<2).unsqueeze(-1)
        return torch.cat((first,second),dim=-1)+self.role(role)


class TaskModel(ComparisonModel):
    def __init__(self,arm,seed=0,elastic_coefficient=None,*,memory_rows=16384):
        assert arm in ('bsdm','bsdm_elastic')
        assert memory_rows % 8 == 0
        config=LanguageModelConfig(memory=BSDMConfig(dim=128,bank_count=memory_rows//8,bank_size=8,
            selected_banks=8,value_width=128,num_heads=1,training_state_dtype='bfloat16',
            qk_normalization='nvidia',output_normalization='layer',output_bias=True,routed_decay=True),
            vocab_size=192,layers=8,seed=seed,recompute_ffn=True,compact_replay=False)
        BSDMLanguageModel.__init__(self,config,backend='role')
        self.arm=arm;self.benchmark='recall';self.seed=seed
        self.auxiliary_coefficient=(.12 if elastic_coefficient is None else elastic_coefficient) if arm=='bsdm_elastic' else 0.
        if elastic_coefficient is not None and (arm!='bsdm_elastic' or not 0 < elastic_coefficient < float('inf')):
            raise ValueError('A positive finite elastic coefficient requires bsdm_elastic')
        self.trace_routes=False
        self.tok_embeddings=TaskEmbedding();initialize_task_embedding(self.tok_embeddings,seed)
        for block in self.layers:self._observe(block.attention)

    def _observe(self,layer):
        original=layer._route
        def route(projected,selected):
            weights,indices=original(projected,selected)
            if not layer._seen_write:
                layer._seen_write=True
                if self.trace_routes:layer._write_trace=indices.detach()
                if layer._penalty_enabled:
                    layer._penalty=bank_occupancy(weights,indices,layer.bank_count)[0].mean()
            return weights,indices
        layer._route=route;layer._seen_write=False;layer._penalty_enabled=False;layer._write_trace=None

    def forward(self,tokens,target=None):
        hidden=self.tok_embeddings(tokens)
        penalty=hidden.new_zeros((),dtype=torch.float32)
        for block in self.layers:
            layer=block.attention
            layer._seen_write=False
            layer._penalty_enabled=target is not None and self.auxiliary_coefficient>0
            hidden=block(hidden)
            if layer._penalty_enabled:penalty=penalty+layer._penalty
        logits=self.output(self.norm(hidden[:,-16:]))
        if target is None:return logits
        loss=F.cross_entropy(logits.float().flatten(0,1),target.flatten())
        if self.auxiliary_coefficient:
            penalty=penalty/len(self.layers)
            return loss+self.auxiliary_coefficient*penalty,loss,penalty
        return loss

    def resolved_model(self):
        result=self.config.record()
        result.update(arm=self.arm,elastic_coefficient=self.auxiliary_coefficient,
                      adapter='TaskEmbedding:192x64 identity,16x64 slot,7x128 role',
                      output_classes=192,ffn_width=512)
        return result

    def accounting(self):
        result=super().accounting()
        result['logical_recurrent_elements_per_example']=8*self.config.memory.bank_count*8*128
        return result
