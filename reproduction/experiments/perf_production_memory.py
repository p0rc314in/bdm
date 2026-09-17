"""Shared full-gradient memory policy for production-context spot checks.

Every model keeps its full causal context, vocabulary, loss and optimizer.
Whole-block activation recomputation and token-chunked vocabulary loss bound
temporary allocations. This is a separate, explicitly reported timing policy.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perf_model import ProfileModel

def _head_loss(hidden,weight,targets):
    return F.cross_entropy(F.linear(hidden,weight).float(),targets,reduction='sum')

class ProductionProfileModel(ProfileModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for block in self.layers:
            # Outer recomputation covers the FFN; avoid redundant nested replay.
            block.feed_forward.recompute_activations=False
            block.forward=torch.compile(block.forward)
        self._head_loss=torch.compile(_head_loss)
        self.loss_chunk=1024

    @torch.compiler.disable
    def forward(self,tokens,target=None):
        assert target is not None,'Production profiling measures full training'
        hidden=self.tok_embeddings(tokens)
        for block in self.layers:
            hidden=checkpoint(block,hidden,use_reentrant=False,preserve_rng_state=False)
        hidden=self.norm(hidden).flatten(0,1)
        targets=target.flatten()
        loss=hidden.new_zeros((),dtype=torch.float32)
        for start in range(0,hidden.shape[0],self.loss_chunk):
            loss=loss+checkpoint(self._head_loss,hidden[start:start+self.loss_chunk],
                                 self.output.weight,targets[start:start+self.loss_chunk],
                                 use_reentrant=False,preserve_rng_state=False)
        # The prepared profiling stream has no ignored/padded targets.
        return loss/targets.numel()
