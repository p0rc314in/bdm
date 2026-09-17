"""The reported BabyLM attention mixer in the shared role-keyed shell."""
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dependencies/sdm-quality'))
from lingua.transformer import Attention, RotaryEmbedding


class AttentionMixer(nn.Module):
    def __init__(self,index,config):
        super().__init__()
        width=config['width']
        with torch.random.fork_rng(devices=[]):
            self.mixer=Attention(width,64,8,8,10000.)
            torch.manual_seed(config['seed']*100000+6000+index)
            self.mixer.reset_parameters(width**-.5,1.)
        self.rope=RotaryEmbedding(10000.,64,config['context'])

    def forward(self,hidden):
        return self.mixer(hidden,self.rope(seqlen=hidden.shape[1]),mask='causal',attn_impl='sdpa')
