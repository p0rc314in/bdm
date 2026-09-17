"""Established fixed-X128 outer softmax execution; retained for historical comparability."""
import hashlib
import json
import os
import torch


def install(xblock):
    import torch._inductor.runtime.triton_heuristics as heuristics
    from torch._inductor.async_compile import AsyncCompile, size_hints_regex
    assert xblock in (8, 128)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    torch.use_deterministic_algorithms(True)
    choices, observed, seen = [], [], set()
    native_triton, native_run = AsyncCompile.triton, heuristics.CachingAutotuner.run

    def triton_compile(self, kernel_name, source_code, device_str='cuda'):
        if '_softmax_' in kernel_name and 'backward' not in kernel_name and 'r0_numel = 8\n' in source_code:
            sizes = size_hints_regex.search(source_code)
            assert sizes
            source = source_code.split('@triton.jit\n')[1]
            key = heuristics.generate_lookup_hash_from_source_code(sizes.group(1), source)
            configuration = dict(XBLOCK=xblock, num_warps=2, num_stages=1)
            torch._inductor.config.autotune_lookup_table[key] = configuration
            choices.append(dict(name=kernel_name, lookup_hash=key, **configuration))
        return native_triton(self, kernel_name, source_code, device_str)

    def checked_run(self, *a, **kw):
        result = native_run(self, *a, **kw)
        name = self.inductor_meta.get('kernel_name', '')
        # Size hints round a seven-part vocabulary-loss reduction up to eight.
        # Match the exact reduction used by the compile hook, not that rounded hint.
        if ('_softmax_' in name and 'backward' not in name
                and 'r0_numel = 8\n' in self.fn.src):
            c = self.launchers[0].config
            actual = dict(XBLOCK=c.kwargs['XBLOCK'], num_warps=c.num_warps, num_stages=c.num_stages)
            assert actual == dict(XBLOCK=xblock, num_warps=2, num_stages=1), (name, actual)
            key = (name, str(self.size_hints))
            if key not in seen:
                seen.add(key)
                observed.append(dict(name=name, size_hints=self.size_hints, **actual))
        return result

    AsyncCompile.triton, heuristics.CachingAutotuner.run = triton_compile, checked_run
    return dict(softmax_choices=choices, observed_launches=observed)
