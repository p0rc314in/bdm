"""Final BSDM and NVIDIA GDN2 in the established, unchanged L8/D128 shell."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from recall import AdaptiveRecallEmbedding, initialize_task_embedding

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'dependencies/bsdm'))
from bsdm import BSDMConfig, BSDMLanguageModel, LanguageModelConfig

# Construction references: the shared shell of the controls and the initialization
# each run is checked against. Historical identities remain immutable for checkpoints.
REFERENCE_ARMS = {'bsdm_n2048_bias_only': (2048, 'original', True),
                  **{f'bsdm_final_n{n}': (n, 'original', True) for n in (1024, 2048)}}
NVIDIA_FINAL_ARMS = {'bsdm_nvidia_n2048': (2048, 'nvidia', True)}
ARMS = ('gdn2_nvidia', *REFERENCE_ARMS, *NVIDIA_FINAL_ARMS)


def nvidia_layer():
    # Import the canonical layer directly without importing the upstream
    # application's Lightning trainer or changing any GDN2 code.
    if 'nvidia_gdn2' not in sys.modules:
        package = ModuleType('nvidia_gdn2')
        package.__path__ = [str(ROOT / 'dependencies/nvidia-gdn2/lit_gpt')]
        sys.modules['nvidia_gdn2'] = package
    return importlib.import_module('nvidia_gdn2.gdn2').GatedDeltaNet2


class NativeGDN2(nn.Module):
    def __init__(self, width, index, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed * 100000 + 1000 + index)
            self.mixer = nvidia_layer()(hidden_size=width, head_dim=32,
                num_heads=4, num_v_heads=4, expand_v=1., mode='chunk',
                use_short_conv=True, conv_size=4, conv_bias=False,
                allow_neg_eigval=False, norm_eps=1e-5, layer_idx=index)

    def forward(self, hidden):
        return self.mixer(hidden, use_cache=False)[0]


class ComparisonModel(BSDMLanguageModel):
    def __init__(self, benchmark, arm, *, seed=0, backend='role', compact_replay=None):
        if arm not in ARMS:
            raise ValueError(arm)
        rows, qk_normalization, output_bias = {**REFERENCE_ARMS, **NVIDIA_FINAL_ARMS}.get(arm, (2048, 'nvidia', True))
        memory = BSDMConfig(dim=128, bank_count=rows // 8, bank_size=8,
                            selected_banks=8, value_width=128, num_heads=1,
                            qk_normalization=qk_normalization, output_bias=output_bias)
        config = LanguageModelConfig(memory=memory, vocab_size=50257 if benchmark == 'wiki' else 192,
                                     layers=8, seed=seed, recompute_ffn=True, compact_replay=compact_replay)
        super().__init__(config, backend=backend)
        self.benchmark, self.arm = benchmark, arm
        if arm == 'gdn2_nvidia':
            for i, layer in enumerate(self.layers):
                layer.attention = NativeGDN2(128, i, seed)
        if benchmark == 'recall':
            self.tok_embeddings = AdaptiveRecallEmbedding(128)
            initialize_task_embedding(self.tok_embeddings, seed)

    def forward(self, tokens, target=None):
        logits = super().forward(tokens)
        if self.benchmark == 'recall':
            logits = logits[:, -16:]
        if target is not None:
            return F.cross_entropy(logits.float().flatten(0, -2), target.flatten())
        return logits

    def accounting(self):
        parameters = dict(self.named_parameters())
        inputs = sum(p.numel() for n, p in parameters.items() if n.startswith('tok_embeddings.'))
        outputs = self.output.weight.numel()
        initial = sum(p.numel() for p in parameters.values() if getattr(p, '_sdm_memory_bank', False))
        total = sum(p.numel() for p in parameters.values())
        recurrent = (8 * self.config.memory.bank_count * self.config.memory.bank_size * 128 if self.arm.startswith('bsdm_')
                     else 8 * 4 * 32 * 32)
        return dict(trainable=total, input_embedding=inputs, output_embedding=outputs,
                    learned_initial_state=initial, active_non_embedding_parameters=total-inputs-outputs-initial,
                    logical_recurrent_elements_per_example=recurrent)

    def common_hash(self):
        digest = hashlib.sha256()
        for name, p in sorted(self.named_parameters()):
            if '.attention.' not in name:
                digest.update(name.encode())
                digest.update(p.detach().cpu().float().contiguous().numpy().tobytes())
        return digest.hexdigest()
