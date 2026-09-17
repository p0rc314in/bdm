"""Canonical BSDM with the prior BabyLM input/output weight-tying convention."""
import os
from pathlib import Path
import sys
import hashlib
import torch
from .spec import CONFIG

SOURCE = Path(os.environ.get('BSDM_SOURCE', str(Path(__file__).resolve().parents[1]/'dependencies/bsdm'))).resolve()
sys.path.insert(0, str(SOURCE))
from bsdm import BSDMConfig, BSDMLanguageModel, LanguageModelConfig
import bdm
assert Path(bdm.__file__).resolve().parent.parent == SOURCE


class BabyLMBSDM(BSDMLanguageModel):
    def __init__(self, *, backend='role', config=CONFIG):
        cfg = LanguageModelConfig(memory=BSDMConfig(dim=config['width'], **config['memory']),
            layers=config['layers'], vocab_size=config['vocab_size'], seed=config['seed'],
            recompute_ffn=True, compact_replay=False)
        super().__init__(cfg, backend=backend)
        if config['tied_embeddings']:
            self.output.weight = self.tok_embeddings.weight
        self.architecture = config.get('architecture', 'bsdm')
        self.auxiliary_coefficient = float(config['elastic_coefficient'])
        self.elastic_valid = None
        self.auxiliary_loss = None
        if self.architecture == 'sdm_native':
            from .native_sdm import NativeMixer
            assert self.auxiliary_coefficient == 0
            for index, block in enumerate(self.layers):
                block.attention = NativeMixer(index, config)
        elif self.architecture == 'attention':
            from .attention import AttentionMixer
            for index, block in enumerate(self.layers):
                block.attention = AttentionMixer(index, config)
        elif self.architecture == 'bsdm':
            if self.auxiliary_coefficient:
                for block in self.layers:
                    self._observe_write(block.attention)
        else:
            raise ValueError('Unknown BabyLM mixer: '+self.architecture)

    def _observe_write(self, layer):
        from elastic_bank_regularizer import bank_occupancy
        original = layer._route
        def route(projected, selected):
            weights, indices = original(projected, selected)
            if layer._elastic_enabled and not layer._elastic_seen:
                layer._elastic_seen = True
                layer._elastic_penalty = bank_occupancy(weights, indices, layer.bank_count,
                                                        valid=self.elastic_valid)[0].mean()
            return weights, indices
        layer._route = route
        layer._elastic_enabled = False
        layer._elastic_seen = False

    def _hidden(self, tokens):
        if not self.auxiliary_coefficient:
            return super()._hidden(tokens)
        hidden = self.tok_embeddings(tokens)
        penalty = hidden.new_zeros((), dtype=torch.float32)
        enabled = self.training and torch.is_grad_enabled()
        for block in self.layers:
            layer = block.attention
            layer._elastic_seen = False
            layer._elastic_enabled = enabled
            hidden = block(hidden)
            if enabled:
                penalty = penalty + layer._elastic_penalty
        self.auxiliary_loss = penalty / len(self.layers)
        return hidden

    def forward(self, tokens, target=None, attn_impl=None):
        # The adapter accepts the pinned evaluator's keyword; BSDM has no SDPA.
        self.elastic_valid = None
        result = super().forward(tokens, target=target)
        if target is not None and self.auxiliary_coefficient and self.training:
            return result.float()+self.auxiliary_coefficient*self.auxiliary_loss, result.float(), self.auxiliary_loss
        return result.float()

    def accounting(self):
        total = sum(p.numel() for p in self.parameters())
        inputs = self.tok_embeddings.weight.numel()
        tied = self.output.weight is self.tok_embeddings.weight
        outputs = 0 if tied else self.output.weight.numel()
        memory = sum(p.numel() for p in self.parameters() if getattr(p, '_sdm_memory_bank', False))
        return dict(total_parameters=total, input_embedding=inputs, output_embedding=outputs,
            input_output_tied=tied, output_embedding_tensor_elements=self.output.weight.numel(),
            learned_initial_memory=memory,
            processing_parameters=total-inputs-outputs-memory, positional_parameters=0,
            logical_state_elements_per_record=(0 if self.architecture=='attention' else self.config.layers*self.config.memory.logical_rows*self.dim))

    def common_fingerprints(self):
        return {n:hashlib.sha256(p.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()
                for n,p in self.named_parameters() if '.attention.' not in n}
