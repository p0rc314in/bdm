"""Dense causal Attention in the retained profiling shell, with a real KV cache.

The Lingua projections, initialization and RoPE are unchanged. FlashAttention-4
provides Hopper forward/backward; native Flash SDPA handles decode. Appends write only
the new token; neither cache growth nor full-prefix copies enter decode timing.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from flash_attn.cute.interface import flash_attn_func
from perf_model import ProfileModel
from perf_production_memory import ProductionProfileModel
from lingua.transformer import RotaryEmbedding, apply_rotary_emb

BACKEND = 'FlashAttention-4 4.0.0b31 train/prefill; PyTorch Flash SDPA decode'


@torch.compiler.disable
def full_attention(q, k, v):
    output, _ = flash_attn_func(q, k, v, causal=True)
    return output


def cached_attention(q, k, v):
    # The CuTe SM90 implementation explicitly excludes split-KV. Use the
    # standard PyTorch FlashAttention decode path over the exact live prefix.
    # These are zero-copy views; no mask allocation or GPU-to-CPU length sync.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                v.transpose(1, 2), is_causal=False)
    return output.transpose(1, 2)


class CachedAttention(nn.Module):
    def __init__(self, control, rope, capacity):
        super().__init__()
        self.mixer = control.mixer
        self.rope = rope
        self.capacity = capacity
        self.heads = control.record['heads']
        self.head_dim = control.record['head_dim']

    def project(self, x, positions=None):
        shape = (*x.shape[:2], self.heads, self.head_dim)
        q, k, v = (projection(x).view(shape)
                   for projection in (self.mixer.wq, self.mixer.wk, self.mixer.wv))
        freq = self.rope(seqlen=x.shape[1]) if positions is None else self.rope(tok_idx=positions)
        q, k = apply_rotary_emb(q, k, 1, freq)
        return q, k, v

    def forward(self, x):
        q, k, v = self.project(x)
        return self.mixer.wo(full_attention(q, k, v).flatten(2))

    def prefill(self, x):
        q, k, v = self.project(x)
        out = self.mixer.wo(full_attention(q, k, v).flatten(2))
        shape = (x.shape[0], self.capacity, self.heads, self.head_dim)
        keys, values = (torch.empty(shape, device=x.device, dtype=x.dtype) for _ in range(2))
        keys[:, :x.shape[1]].copy_(k)
        values[:, :x.shape[1]].copy_(v)
        keys[:, x.shape[1]:].zero_()
        values[:, x.shape[1]:].zero_()
        return out, (keys, values, x.shape[1])

    def step(self, x, cache):
        keys, values, used = cache
        positions = torch.scalar_tensor(used, device=x.device, dtype=torch.int64).view(1)
        q, k, v = self.project(x, positions)
        # A single unique cache destination needs a direct copy. General
        # indexed writes lower to sorting/checking kernels in deterministic mode.
        keys.narrow(1, used, 1).copy_(k)
        values.narrow(1, used, 1).copy_(v)
        used = used + 1
        out = cached_attention(q, keys[:, :used], values[:, :used])
        return self.mixer.wo(out.flatten(2)), (keys, values, used)


def install_attention(model, context, reserve_tokens=128):
    # One shared positional table, as in a standard transformer. No learned
    # parameters change, and every layer retains its own projected K/V cache.
    rope = RotaryEmbedding(10000., 64, context + reserve_tokens)
    for block in model.layers:
        block.attention = CachedAttention(block.attention, rope, context + reserve_tokens)
        block.feed_forward.recompute_activations = False
    model.mixer_record.update(backend=BACKEND, kv_heads=model.profile_width // 64,
                              cache=f'preallocated BF16 K/V, {reserve_tokens}-token reserve',
                              decode_kernel='PyTorch native Flash SDPA', sliding_window=False,
                              decode_execution='compiled SDPA; direct unique-position KV copies')


class AttentionTrainingModel(ProductionProfileModel):
    def __init__(self, arm, context, **kwargs):
        assert arm == 'attention'
        # The parent's temporary per-layer RoPE tables are replaced by the
        # shared table below; avoid constructing 16 redundant long tables.
        super().__init__(arm, context=128, **kwargs)
        install_attention(self, context)

    def common_hash(self): return None
    def full_parameter_hash(self): return None


class AttentionInferenceModel(ProfileModel):
    def __init__(self, arm, context, *, reserve_tokens=128):
        assert arm == 'attention'
        super().__init__(arm, context=128, width=2048, layers=16,
                         gdn_head_dim=64, gdn_expand_v=2., bsdm_heads=1)
        install_attention(self, context, reserve_tokens)
        self.prefill_blocks, self.decode_blocks = [], []
        for block in self.layers:
            def call(hidden, cache, decode=False, block=block):
                x = block.attention_norm(hidden)
                out, cache = (block.attention.step(x, cache) if decode
                              else block.attention.prefill(x))
                hidden = hidden + out
                return hidden + block.feed_forward(block.ffn_norm(hidden)), cache
            self.prefill_blocks.append(torch.compile(call))
            self.decode_blocks.append(torch.compile(call))
        self.head = torch.compile(lambda hidden: self.output(self.norm(hidden)))

    def empty_cache(self): return [None] * len(self.layers)
    def handoff(self, caches): return caches

    def run(self, tokens, caches, decode=False):
        hidden = self.tok_embeddings(tokens)
        functions = self.decode_blocks if decode else self.prefill_blocks
        for i, fn in enumerate(functions):
            hidden, caches[i] = fn(hidden, caches[i], decode=decode)
        return self.head(hidden[:, -1:]), caches


def validate_cache():
    """Check cached causality against full attention and the inherited SDPA layer."""
    import triton
    assert torch.__version__ == '2.11.0+cu128', torch.__version__
    assert triton.__version__ == '3.6.0', triton.__version__
    from perf_model import Control
    from torch.nn import functional as F
    torch.set_num_threads(4)
    torch.manual_seed(0)
    layer = CachedAttention(Control('attention', 0, 512, 128),
                            RotaryEmbedding(10000., 64, 128), 128).cuda().bfloat16()
    x = torch.randn(1, 71, 512, device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = layer(x)
        q, k, v = layer.project(x)
        native = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                    v.transpose(1, 2), is_causal=True).transpose(1, 2)
        torch.testing.assert_close(expected, layer.mixer.wo(native.flatten(2)), atol=.02, rtol=.02)
        prefill = torch.compile(layer.prefill)
        step = torch.compile(layer.step)
        prefix, cache = prefill(x[:, :64])
        assert torch.isfinite(cache[0]).all() and torch.isfinite(cache[1]).all()
        pieces = [prefix]
        for i in range(64, 71):
            out, cache = step(x[:, i:i+1], cache)
            print('CACHE_STEP', i, 'used', cache[2], 'finite', bool(torch.isfinite(out).all()), flush=True)
            pieces.append(out)
        actual = torch.cat(pieces, 1)
        torch.testing.assert_close(expected, actual, atol=.02, rtol=.02)
        assert cache[2] == 71
    layer.train()
    loss = layer(x.requires_grad_()).float().square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    print('ATTENTION_CACHE_AND_BACKWARD_OK', flush=True)


if __name__ == '__main__':
    validate_cache()
