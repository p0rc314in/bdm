# Source acknowledgments

BDM combines mechanisms from Sparse Delta Memory, Gated DeltaNet-2 and the
Mixture-of-Memories family. The generic routed-memory construction is prior art.

`bdm/_shell.py` extracts `RMSNorm` and `FeedForward` from
[Meta's SDM/Lingua source](https://github.com/facebookresearch/sparse-delta-memory/blob/183e7df809131b80ad4393741029d0f20fc3640b/lingua/transformer.py),
revision `183e7df809131b80ad4393741029d0f20fc3640b`. Its CC BY-NC 4.0 license
is preserved in `licenses/SDM-LICENSE`. The extraction removes an inactive
statistics hook and adds optional activation recomputation while preserving
forward arithmetic and initialization. The
shared-residual router was developed by p0rc314in from the SDM routing interface.

The optional FLA backend invokes the MIT-licensed `flash-linear-attention`
package rather than embedding it. The inner controller follows NVIDIA's
Gated DeltaNet-2 reference. BDM supplies the bank execution kernels.

The model is defined by `BSDMConfig` and the dimensions in the appendix. No new blanket license is applied to
third-party-derived material; this draft does not grant a blanket license to
all original implementation files.
