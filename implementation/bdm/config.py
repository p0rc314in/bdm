"""Explicit bank geometry and balanced access for canonical BDM."""

from dataclasses import asdict, dataclass
import math


def near_square_factors(value: int) -> tuple[int, int]:
    if value <= 0:
        raise ValueError('value must be positive')
    for first in range(math.isqrt(value), 0, -1):
        if value % first == 0:
            return first, value // first
    raise AssertionError('positive integers have a factorization')


@dataclass(frozen=True)
class BDMConfig:
    dim: int = 128
    bank_count: int = 1024
    bank_size: int = 4
    selected_banks: int = 8
    value_width: int = 128
    norm_eps: float = 1e-5
    num_heads: int = 1
    routed_decay: bool = False
    qk_normalization: str = 'original'
    output_normalization: str = 'rms'
    output_bias: bool = False
    training_state_dtype: str = 'bfloat16'

    def __post_init__(self):
        for name in ('dim', 'bank_count', 'bank_size', 'selected_banks', 'value_width', 'num_heads'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.bank_size > 256:
            raise ValueError('bank_size must be at most 256')
        if self.value_width % self.num_heads:
            raise ValueError('value_width must be divisible by num_heads')
        if type(self.routed_decay) is not bool or type(self.output_bias) is not bool:
            raise ValueError('routed_decay and output_bias must be boolean')
        if self.qk_normalization not in ('original', 'nvidia'):
            raise ValueError('unknown Q/K normalization')
        if self.output_normalization not in ('rms', 'layer'):
            raise ValueError('unknown output normalization')
        if self.training_state_dtype not in ('float32', 'bfloat16'):
            raise ValueError('training_state_dtype must be float32 or bfloat16')
        if self.selected_banks > min(near_square_factors(self.bank_count)):
            raise ValueError('current product selector requires selected_banks <= both factor widths')
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('norm_eps must be finite and positive')

    @property
    def logical_rows(self):
        return self.bank_count * self.num_heads * self.bank_size

    @property
    def reads(self):
        return self.selected_banks * self.num_heads * self.bank_size

    @property
    def head_value_width(self):
        return self.value_width // self.num_heads

    @property
    def writes(self):
        return self.reads

    def record(self):
        return dict(asdict(self), logical_rows=self.logical_rows, R=self.reads, W=self.writes,
                    read_banks=self.selected_banks, write_banks=self.selected_banks,
                    router='shared_residual_product_key', memory_rule='gdn2')

    @classmethod
    def from_record(cls, record):
        normalized = dict(record)
        normalized.setdefault('num_heads', 1)
        normalized.setdefault('training_state_dtype', 'float32')
        for name in ('routed_decay', 'qk_normalization', 'output_normalization', 'output_bias'):
            normalized.setdefault(name, cls.__dataclass_fields__[name].default)
        config = cls(**{k: normalized[k] for k in cls.__dataclass_fields__})
        if normalized != config.record():
            raise ValueError('resolved geometry/access disagrees with the BDM configuration')
        return config
