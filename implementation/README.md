# Block Delta Memory layer and kernels

`bsdm` exposes the Block Delta Memory (BDM) layer and its configuration.
The package and Python identifiers retain their existing names for compatibility.
The `role` backend uses PyTorch, CUDA, and Triton. In a CUDA environment:

```bash
python -m pip install -e './implementation[cuda,fla]'
```

The small-model configuration:

```python
from bsdm import BSDMConfig, BlockSparseDeltaMemory

config = BSDMConfig(
    dim=128,
    bank_count=256,
    bank_size=8,
    selected_banks=8,
    value_width=128,
    num_heads=1,
)
layer = BlockSparseDeltaMemory(config, backend="role")
```

This uses $N=2048$ and accesses $K=64$ rows per role. The dimensions are experiment
settings; [the appendix](../APPENDIX.md) gives the BabyLM and H200 configurations.
The default construction uses GDN2 normalization, coherent route-weighted
updates, the SDM output interface, BF16 state storage, and the reference
variance-preserving initialization for the two additive memory factors.
The included layer and kernels match canonical revision
`ed51fd03ef83ff8ce50555a66481d878c2fc4a4f`.

[Source acknowledgments and licensing](THIRD_PARTY_NOTICES.md).

To check the arithmetic of the recorded results from the repository root:

```bash
python scripts/verify_results.py
```

This checks the included evidence; it does not rerun training or benchmarks.

For input preparation, full training/evaluation, and the timing and occupancy
experiments, use the [experiment reproduction entry point](../reproduction/README.md).
