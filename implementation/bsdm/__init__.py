"""Block-Sparse Delta Memory, using the canonical BDM implementation.

The legacy bdm imports remain available for prior checkpoints and experiments.
"""
from dataclasses import dataclass, field
from bdm import BDMConfig, BankedDeltaMemory, BDMLanguageModel
from bdm import LanguageModelConfig as _LanguageModelConfig


@dataclass(frozen=True)
class BSDMConfig(BDMConfig):
    routed_decay: bool = True
    qk_normalization: str = 'nvidia'
    output_normalization: str = 'layer'
    output_bias: bool = True


@dataclass(frozen=True)
class LanguageModelConfig(_LanguageModelConfig):
    memory: BSDMConfig = field(default_factory=BSDMConfig)


BlockSparseDeltaMemory = BankedDeltaMemory
BSDMLanguageModel = BDMLanguageModel
__all__ = ['BSDMConfig', 'BlockSparseDeltaMemory', 'BSDMLanguageModel', 'LanguageModelConfig']
