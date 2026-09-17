"""Canonical BDM layer and an all-bank causal language-model shell."""

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from ._parameters import ParameterMethods
from ._shell import FeedForward, RMSNorm
from .config import BDMConfig, near_square_factors
from .gdn2_banks import pack_bank_events, product_key_route, reference_packed_gdn2, scatter_bank_outputs
from .routing import SharedResidualRouter


class BankedDeltaMemory(ParameterMethods, nn.Module):
    """Shared-residual product routing around dense GDN2 memory banks.

    ``role`` executes the audited CUDA training kernel; ``reference`` is the
    small PyTorch equation oracle; ``fla`` uses released dense GDN2 after causal
    packing. All backends share parameters. ``forward`` consumes a complete sequence.
    ``prefill`` and ``step`` expose persistent state for H1/eight-row CUDA banks.
    """

    def __init__(self, config: BDMConfig, *, layer_id=0, seed=0, backend='role', compact_replay=None):
        super().__init__()
        if backend not in ('role', 'reference', 'fla'):
            raise ValueError('backend must be role, reference, or fla')
        if seed < 0 or layer_id < 0:
            raise ValueError('seed and layer_id must be nonnegative')
        if compact_replay is not None and type(compact_replay) is not bool:
            raise ValueError('compact_replay must be None, True, or False')
        self.compact_replay = compact_replay
        self.config, self.backend = config, backend
        self.hidden_size, self.bank_count, self.bank_size = config.dim, config.bank_count, config.bank_size
        self.value_width, self.kernel_key_width = config.value_width, config.bank_size
        self.num_heads = config.num_heads
        self.head_value_width = config.head_value_width
        key_width = config.num_heads * config.bank_size
        self.num_reads = self.num_writes = config.selected_banks
        self.layer_id, self.initialization_seed = layer_id, seed * 100_000 + 6000 + layer_id
        self.norm_eps = config.norm_eps
        self.shared_residual_router = True
        self.initial_state_mode = 'product_key'
        self.routing_mode = 'product_topk'
        route_width = sum(near_square_factors(config.bank_count))
        self.factor_route_width = self.route_projection_width = route_width
        self.q_proj = nn.Linear(config.dim, key_width, bias=False)
        self.k_proj = nn.Linear(config.dim, key_width, bias=False)
        self.v_proj = nn.Linear(config.dim, config.value_width, bias=False)
        self.shared_router = SharedResidualRouter(config.dim, route_width)
        self.read_route_proj = self.write_route_proj = None
        self.f_proj = nn.Sequential(nn.Linear(config.dim, config.head_value_width, bias=False),
                                    nn.Linear(config.head_value_width, key_width, bias=False))
        self.b_proj = nn.Linear(config.dim, key_width, bias=False)
        self.w_proj = nn.Linear(config.dim, config.value_width, bias=False)
        self.A_log = nn.Parameter(torch.empty(config.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(key_width, dtype=torch.float32))
        self.A_log._no_weight_decay = self.dt_bias._no_weight_decay = True
        rows, columns = near_square_factors(config.logical_rows)
        self.register_parameter('initial_state', None)
        self.initial_row_factor = nn.Parameter(torch.zeros(rows, config.head_value_width))
        self.initial_column_factor = nn.Parameter(torch.zeros(columns, config.head_value_width))
        self.initial_row_factor._sdm_memory_bank = self.initial_column_factor._sdm_memory_bank = True
        self.output_gate = nn.Linear(config.dim, config.value_width, bias=True)
        self.output_norm_weight = nn.Parameter(torch.ones(config.value_width))
        if config.output_normalization == 'layer':
            self.output_norm_bias = nn.Parameter(torch.zeros(config.value_width))
        self.o_proj = nn.Linear(config.value_width, config.dim, bias=config.output_bias)
        self.init_weights(initialization_seed=self.initialization_seed)

    def _route(self, projected, selected_banks):
        return product_key_route(projected, bank_count=self.bank_count, selected_banks=selected_banks)

    def prefill(self, hidden, state=None, *, state_dtype=None):
        """Process a prefix; default BF16 cache storage rounds at handoff.

        Pass state_dtype=torch.float32 to retain FP32 cache storage.
        All recurrence arithmetic remains FP32 with either storage policy.
        """
        from .inference import prefill
        return prefill(self, hidden, state, state_dtype=state_dtype)

    def step(self, hidden, state):
        """Advance an existing bank state by one token (inference only)."""
        from .inference import step
        return step(self, hidden, state)

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size or hidden.shape[1] == 0:
            raise ValueError('hidden must be [batch, positive time, dim]')
        if self.num_heads != 1:
            from .heads import forward_heads
            return forward_heads(self, hidden)
        q, k, v, g, b, w = self._controllers(hidden)
        read_routes, write_routes = self._route_projections(hidden)
        ww, wi = self._route(write_routes, self.num_writes)
        rw, ri = self._route(read_routes, self.num_reads)
        if self.backend == 'role':
            from .role_bank_kernel import _hopper_training_supported
        if (self.backend == 'role' and self.config.routed_decay and self.initial_state is None
                and _hopper_training_supported(q, v.shape[-1], self.compact_replay)):
            from .product_routed_update import bank_training
            raw = bank_training(q, k, v, g, b, w, wi, ww, ri, rw,
                                self.initial_row_factor, self.initial_column_factor,
                                self.bank_count, self.bank_size, self.bank_size ** -.5,
                                compact_replay=self.compact_replay,
                                state_dtype=getattr(torch, self.config.training_state_dtype))
            return self._project_bank_output(hidden, raw)
        state = self.materialize_initial_state()
        if self.backend == 'role':
            if not hidden.is_cuda:
                raise ValueError('role backend requires CUDA; choose reference for CPU equation checks')
            from .role_bank_kernel import role_bank_training
            if self.config.routed_decay:
                from .routed_update import bank_training as role_bank_training
            raw = role_bank_training(q, k, v, g, b, w, wi, ww, ri, rw, state,
                                     self.bank_count, self.bank_size ** -.5,
                                     compact_replay=self.compact_replay,
                                     state_dtype=getattr(torch, self.config.training_state_dtype))
        else:
            events = pack_bank_events(q, k, v, g, b, w, wi, ww, ri, rw,
                                      bank_count=self.bank_count, kernel_key_width=self.bank_size,
                                      routed_decay=self.config.routed_decay)
            initial = state.unsqueeze(0).expand(hidden.shape[0], -1, -1, -1).reshape(
                hidden.shape[0] * self.bank_count, 1, self.bank_size, self.value_width).contiguous().float()
            if self.backend == 'reference':
                values = reference_packed_gdn2(q=events.q, k=events.k, v=events.v, g=events.g, b=events.b, w=events.w,
                                               scale=self.bank_size ** -.5, initial_state=initial,
                                               cu_seqlens=events.cu_seqlens,
                                               state_dtype=getattr(torch, self.config.training_state_dtype),
                                               state_interval=min(16, max(4, 128 // (1 << (self.bank_size - 1).bit_length()))))
            else:
                if self.config.training_state_dtype != 'float32':
                    raise NotImplementedError('The FLA backend does not implement BF16 training-state storage')
                from fla.ops.gdn2 import chunk_gdn2
                values, _ = chunk_gdn2(q=events.q, k=events.k, v=events.v, g=events.g, b=events.b, w=events.w,
                                       scale=self.bank_size ** -.5, initial_state=initial, output_final_state=False,
                                       use_qk_l2norm_in_kernel=False, cu_seqlens=events.cu_seqlens)
            raw = scatter_bank_outputs(values, events.token_indices, batch=hidden.shape[0], time=hidden.shape[1])
        return self._project_bank_output(hidden, raw)


@dataclass(frozen=True)
class LanguageModelConfig:
    memory: BDMConfig = field(default_factory=BDMConfig)
    vocab_size: int = 192
    layers: int = 8
    ffn_multiple_of: int = 256
    seed: int = 0
    recompute_ffn: bool = True
    compact_replay: bool | None = None

    def __post_init__(self):
        if min(self.vocab_size, self.layers, self.ffn_multiple_of) <= 0 or self.seed < 0:
            raise ValueError('invalid language-model geometry or seed')
        if self.compact_replay is not None and type(self.compact_replay) is not bool:
            raise ValueError('compact_replay must be None, True, or False')

    def record(self):
        return dict(asdict(self), memory=self.memory.record())


class BDMBlock(nn.Module):
    def __init__(self, config, index, backend):
        super().__init__()
        dim = config.memory.dim
        self.attention = BankedDeltaMemory(config.memory, layer_id=index, seed=config.seed, backend=backend,
                                           compact_replay=config.compact_replay)
        self.feed_forward = FeedForward(dim, 4 * dim, config.ffn_multiple_of, None,
                                        recompute_activations=config.recompute_ffn)
        self.attention_norm = RMSNorm(dim, eps=config.memory.norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=config.memory.norm_eps)

    def forward(self, hidden):
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class BDMLanguageModel(nn.Module):
    """Checkpoint-compatible all-bank shell with independent width and depth."""

    def __init__(self, config: LanguageModelConfig, *, backend='role'):
        super().__init__()
        self.config = config
        self.dim = config.memory.dim
        self.layers = nn.ModuleList([BDMBlock(config, i, backend) for i in range(config.layers)])
        self.tok_embeddings = nn.Embedding(config.vocab_size, self.dim)
        self.norm = RMSNorm(self.dim, eps=config.memory.norm_eps)
        self.output = nn.Linear(self.dim, config.vocab_size, bias=False)
        base, std = config.seed * 100_000, self.dim ** -.5
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base + 100)
            for parameter in (self.tok_embeddings.weight, self.output.weight):
                nn.init.trunc_normal_(parameter, mean=0, std=std, a=-3 * std, b=3 * std)
        for i, block in enumerate(self.layers):
            block.attention.init_weights(initialization_seed=base + 6000 + i)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base + 1000 + i)
                block.feed_forward.reset_parameters()

    def _hidden(self, tokens):
        hidden = self.tok_embeddings(tokens)
        for block in self.layers:
            hidden = block(hidden)
        return hidden

    def forward(self, tokens, target=None):
        # Dynamic bank dispatch can break the graph inside the layer loop.
        # Isolate that loop so Dynamo can resume at the head and training loss.
        hidden = self._hidden(tokens)
        logits = self.output(self.norm(hidden))
        if target is not None:
            # Keep the training loss inside the compiled entry point. The
            # compiler can fuse its large token-by-vocabulary intermediates.
            return torch.nn.functional.cross_entropy(logits.float().flatten(0, -2), target.flatten())
        return logits
