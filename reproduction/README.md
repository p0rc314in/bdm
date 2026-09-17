# Reproducing the experiments

`python reproduce.py run …` trains or benchmarks a model and evaluates its new
outputs. It never substitutes the paper's saved JSON values. Choose an experiment
and one arm per invocation; separate `--work` directories can run independently.
`python scripts/verify_results.py` remains a separate arithmetic check of the
published evidence.

## Runtime and preparation

Use Python 3.11 or later. Input preparation needs CPU and approximately 40 GB of
free disk space; keep that environment separate from the CUDA environment because
the original WikiText manifest records its preparation-library versions. All input
sources are public and pinned. Full BabyLM evaluation requires accepting the source
EWoK dataset's access terms with your own Hugging Face account if access is gated.
No author account or private experiment artifact is required.

```bash
python -m venv .venv-inputs
.venv-inputs/bin/python reproduce.py setup --profile inputs
.venv-inputs/bin/python reproduce.py prepare wiki --work ./runs
.venv-inputs/bin/python reproduce.py prepare recall --work ./runs
# Required only for the corresponding experiments:
.venv-inputs/bin/python reproduce.py prepare memory --work ./runs
.venv-inputs/bin/python reproduce.py prepare babylm --work ./runs
```

Run GPU experiments in the standard PyTorch 2.11.0 / CUDA 12.8 development image
`pytorch/pytorch@sha256:53ab3de62f6101d1e42f9be28623ab7a468a24c070d632f211ed576e30b6abd3`,
on Linux with one NVIDIA GPU and a persistent work directory. Do not substitute a
CPU implementation for a reported GPU timing. Select `quality`, `babylm`, or
`performance` to install that path's pinned additions into the current Python
runtime. The setup stage retrieves the pinned public FLA source; the GDN2 and GDN1
helper versions are isolated because their interfaces differ.

The commands below run inside that disposable container as root. Install Git
for the pinned source checkout and explicitly allow pip to install into the
container's externally managed Python environment.

```bash
apt-get update && apt-get install -y git
export PIP_BREAK_SYSTEM_PACKAGES=1
python reproduce.py setup --profile quality --work ./runs
python reproduce.py list
python reproduce.py run wiki --model bdm --seed 0 --work ./runs --dry-run
python reproduce.py run wiki --model bdm --seed 0 --work ./runs
```

Preparation validates fixed manifests and record hashes. WikiText regenerates its
entire corpus serialization, tokenization, masks and training permutations; Recall
regenerates the frozen 30-condition stream. BabyLM builds the complete corpus,
checkpoint inputs, full terminal EWoK, padded inference buckets and seven official
fine-tuning schedules. The occupancy study has its own generator and does not use
Adaptive Recall as a substitute.

## Experiments

| Command family | Arms / settings | Matched hardware | Full work per invocation |
|---|---|---|---|
| `wiki` | `bdm`, `sdm`, `attention`, `gdn1`, `gdn2`, `mom`; `--seed 0`, `1`, `2` | RTX 4090 | 21,603 updates, three corpus passes; full validation and test |
| `recall` | same six arms and seeds | A40 | 30,000 updates; all 30 conditions on validation and test |
| `init-scale` | `--benchmark wiki` or `recall`; `--factor-scale reference` or `unit`; three seeds | RTX 4090 / A40 | same complete schedules, changing only initial factor amplitude |
| `memory` | `--lambda 0`, `.001`, `.003`, `.01`, `.03`, `.06`, `.12`; seed 0 | RTX 4090 | 30,000 updates on the 20 task/length conditions, full validation/test and measured written-bank unions |
| `babylm` | `--model bdm`, `sdm`, or `attention`; seed 0 | H100 80GB | 102,852 updates, 28 permanent exposure pairs, full terminal inference, human trajectories and seven complete fine-tunes |
| `performance` | `bdm`, `sdm`, `attention`, `gdn1`, `gdn2`; `--phase train`, `prefill`, `decode` | H200 | one context, complete L16/D2048/FFN5632 model; compiler/warmup excluded from steady-state samples |

```bash
python reproduce.py run recall --model sdm --seed 2
python reproduce.py run init-scale --benchmark wiki --factor-scale unit --seed 1
python reproduce.py run memory --lambda .01
python reproduce.py setup --profile babylm
python reproduce.py run babylm --model bdm
python reproduce.py setup --profile performance
python reproduce.py run performance --model bdm --phase decode --context 262144
```

Performance contexts are 8,192 through 1,048,576, doubling each time. Capacity is
N=T for BDM/SDM. At the largest contexts some complete models run out of memory;
an OOM is a capacity outcome, not a zero latency. Decode uses three independently
reset 1,024-token continuations after runtime stabilizes. It preserves state
across each continuation; it is not a repeatedly reset single-token timing.

Small-model runs take tens of minutes to several hours each, depending on arm and
hardware. Full BabyLM took approximately ten GPU-hours for BDM and twenty for SDM,
including evaluation and fine-tuning. Allow up to 30 minutes for an individual
large-context performance case, including compilation. Multiply actual elapsed
GPU-hours by your provider's rate; setup, compilation and input storage also cost
resources. These are planning estimates, not an automatic budget or termination
limit. Complete recovery artifacts may require hundreds of GB across a full suite.

## Outputs, recovery and interpretation

Every invocation writes to `--work/<experiment-and-settings>/`. `identity.json`
records the arguments and source identity; `workspace/outputs/` contains curves,
metrics and predictions. `artifacts.json` indexes independent, hashed copies of
checkpoints, evaluations and final outputs. Training recovery includes model,
optimizer, scheduler position, RNG and data cursor; BabyLM additionally retains all
28 exposure checkpoints and task-specific fine-tuning recovery. Repeating a training command resumes that run; repeating a completed timing
case reports its existing result. Use another `--work` directory for a new timing. A file lock rejects concurrent writers, and a changed
configuration/source requires another work directory. Keep the work directory on
persistent storage; this local transport does not upload to a hosted service.

`python reproduce.py report --work ./runs --output ./runs/report` writes a CSV of
new terminal metrics and PNG/PDF loss curves. Install its plotting dependency with
`python -m pip install matplotlib==3.10.8`. The finished
paper figures remain in `figures/`; this command visualizes newly generated outputs.

The terminal WikiText NLL means are approximately 4.34 BDM, 4.36 SDM, 4.36 attention,
4.30 GDN1, 4.29 GDN2 and 4.34 MoM. Use the actual per-seed values and sample standard
deviations in `data/quality-summary.json` when comparing a new run. These are
reference outcomes, not an acceptance test that silently retries seeds. BabyLM's
full-task percentages and the complete timing samples are in their corresponding
`data/` files. Comparisons must retain equal schedules, hardware and precision.
CUDA reduction ordering and hard routing can amplify small numerical differences;
record new outputs and any reload difference rather than overwriting the reference.

The executable families cover the main quality, BabyLM, performance, occupancy and
paired initialization-scale experiments. Earlier development configurations in
the appendix are preserved as historical evidence; they are not silently relabeled
as executions of the final architecture.

## Source mapping and checks

`extraction.json` maps retained scientific source files and their hashes to this
release. `vendor/quality-bdm` preserves the earlier measured quality runtime;
`implementation/` contains the later canonical runtime used for BabyLM, occupancy
and the final H200 BDM refresh. `control-sources.json` identifies upstream controls
and the documented MoM corrections. The SDM extraction preserves the full-table,
independent-router model and the BF16/rectangular-capacity correctness repairs.
The public source here is the retrievable object for those historical revisions;
access to a historical private Git commit is unnecessary.

```bash
python reproduce.py check             # bounded input/transport/CLI checks
python reproduce.py check --torch     # adds CPU construction/count/common-role checks
python scripts/verify_results.py      # arithmetic of the paper's recorded results
```

These checks do not claim a new full CUDA training or timing reproduction. The
from-scratch commands above create new checkpoints and predictions for reevaluation;
this release does not present checkpoint hashes alone as downloadable models.
