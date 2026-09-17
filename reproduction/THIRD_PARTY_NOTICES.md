# Reproduction source acknowledgments

The SDM and transformer shell extractions derive from Meta's Sparse Delta Memory / Lingua
source, revision `183e7df809131b80ad4393741029d0f20fc3640b`; the CC BY-NC 4.0 license is
included alongside both extracts. The SDM correctness adaptations are identified in
`control-sources.json`. NVIDIA's GDN2 extract is from revision
`a5552fe3c67e0ebc7ef1220df68ae8896ec62d56` of `NVlabs/GatedDeltaNet-2`; its license is
included in `vendor/gdn2/LICENSE`. FLA and its MoM implementation are installed from
pinned public packages/source and retain their upstream licenses.

BabyLM input/evaluation adapters follow `babylm-org/babylm-eval` revision
`6f825c291e2c4c78ad33b1935fd64d45f52642dc`. `official_human.py` is the corresponding
Reading/AoA scorer extraction, distributed with `BABYLM_LICENSE` (Apache 2.0).
The extraction adapts file paths and returns structured results for the local runner. Dataset licenses and access conditions belong to the
cited dataset publishers; the preparation commands download data instead of
redistributing the corpora here. The adaptation records and source hashes are in
`extraction.json` and the input identity files. No blanket license overrides these
source licenses. See also `implementation/THIRD_PARTY_NOTICES.md`.
