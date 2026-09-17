"""Original SDM control in the established full WikiText/Recall campaign."""
import argparse
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from types import MethodType

import torch
from torch import nn

import train as runner
from model import ComparisonModel, ROOT
from control_adapters.sdm_recall_padding import pad_parallel_time

ARM = 'sdm_native_k64_n2048'
SOURCE = ROOT / 'dependencies/sdm-native/lingua/sparse_delta_memory'
NATIVE_COMMIT = 'bfe51564d1349200d21c8ad6507c248ef7f8ef5e'


def native_layer():
    if 'native_sdm' not in sys.modules:
        spec = importlib.util.spec_from_file_location('native_sdm', SOURCE/'__init__.py',
            submodule_search_locations=[str(SOURCE)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['native_sdm'] = module
        spec.loader.exec_module(module)
    module = sys.modules['native_sdm']
    return module.SparseDeltaMemory, module.SparseDeltaMemoryArgs


@torch.compiler.disable
def padded_native_write_read(self, memory, k_idx, k_val, v, beta, g, q_idx, q_val,
                             grad_final_memory=None):
    """Keep native chunks inside each independent Recall example/head."""
    time = k_idx.shape[1]
    if not self.training and time <= 64:
        # Use the adopted Recall adapter's native single-token kernel loop.
        # The old Python short-sequence fallback mixes FP32 decay with BF16
        # scatter buffers. This is the same kernel used by native T=1 decode.
        from native_sdm.memory_ops import fused_decode_step
        outputs = []
        for position in range(time):
            outputs.append(fused_decode_step(memory, k_idx[:, position],
                k_val[:, position], v[:, position], beta[:, position],
                g[:, position], q_idx[:, position], q_val[:, position],
                use_delta_rule=True, normalize_memory=False,
                key_weighted_decay=self.args.key_weighted_decay))
        return torch.stack(outputs, dim=1), memory
    # Short Recall conditions include T=48. Native Triton reduction tiles
    # require powers of two; append only state-neutral tokens before chunking.
    chunk_size = min(self.args.memory_block_size, 1 << (time - 1).bit_length())
    inputs = pad_parallel_time((k_idx, k_val, v, beta, g, q_idx, q_val),
        chunk_size=chunk_size, rows_per_bank=self.slots_per_head)
    readings, terminal = type(self).gated_write_read(self, memory, *inputs,
        grad_final_memory=grad_final_memory)
    return readings[:, :time], terminal


class NativeSDMControl(nn.Module):
    def __init__(self, index, seed, benchmark):
        super().__init__()
        cls, args_cls = native_layer()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed * 100000 + 1000 + index)
            cfg = args_cls(dim=128, num_heads=1, slots_per_head=2048,
                num_reads=64, num_writes=64, memory_block_size=256, norm_eps=1e-6,
                read_act='Softmax', write_act='Softmax', normalize_readings=True,
                backprop_on_memory=True, output_gate=True)
            self.mixer = cls(cfg, layer_id=index)
            self.mixer.init_weights()
        if benchmark == 'recall':
            self.mixer.gated_write_read = MethodType(padded_native_write_read, self.mixer)
        self.record = asdict(cfg)
        self.record['implementation'] = dict(commit=NATIVE_COMMIT,
            model='original SDM', initialization='full learned table',
            changes='BF16 correctness and exact rectangular product-key capacity; Recall state-neutral padding')
        assert self.mixer.memory.numel() == 2048 * 128
        assert not hasattr(self.mixer, 'shared_router')
        assert self.mixer.product_key_rows * self.mixer.product_key_columns == 2048

    def forward(self, hidden):
        return self.mixer(hidden)[0]


class NativeSDMModel(ComparisonModel):
    def __init__(self, benchmark, arm, *, seed=0, backend='role', compact_replay=None):
        assert arm == ARM
        super().__init__(benchmark, 'bsdm_final_n1024', seed=seed,
                         backend=backend, compact_replay=compact_replay)
        self.arm = arm
        for index, block in enumerate(self.layers):
            block.attention = NativeSDMControl(index, seed, benchmark)

    def resolved_model(self):
        record = self.config.record()
        record.pop('memory')
        record.update(arm=self.arm, ffn_width=512, mixer=self.layers[0].attention.record)
        return record

    def accounting(self):
        result = super().accounting()
        result['logical_recurrent_elements_per_example'] = 8 * 2048 * 128
        assert result['learned_initial_state'] == 8 * 2048 * 128
        return result

    def run_diagnostics(self, run, benchmark):
        if benchmark != 'recall':
            return
        # Exercise every adopted sequence shape before expensive training.
        # This is startup validation, not an extra scientific run or score.
        dataset = runner.Dataset(run.use_artifact(runner.DATASETS['recall'][0]).download(), 'recall')
        before = {n: p.detach().clone() for n, p in self.named_parameters()}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            metrics, _ = runner.evaluate(self, dataset, 'validation', 'cuda', limit_per_condition=8)
        assert metrics['overall']['examples'] == 30 * 8
        assert all(torch.equal(p, before[n]) for n, p in self.named_parameters())
        print('NATIVE_RECALL_EVAL_OK: all30 shapes; native inference; parameters unchanged', flush=True)


def validate_common(network, benchmark, arm, seed, options):
    with torch.random.fork_rng(devices=[]):
        reference = ComparisonModel(benchmark, 'bsdm_n2048_bias_only', seed=seed, **options)
    actual = dict(network.named_parameters())
    for name, value in reference.named_parameters():
        if '.attention.' not in name:
            assert torch.equal(actual[name], value), name
    print('NATIVE_SDM verified independent router, full N2048 memory, K64, matched common shell', flush=True)
    return reference.common_hash()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=('wiki', 'recall'), required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    args = p.parse_args()
    args.arm = ARM
    args.microbatch = 8 if args.benchmark == 'wiki' else 32
    args.checkpoint_seconds = 300
    args.checkpoint_days = 30
    args.checkpoint_before_probe = True
    args.interrupt_after_step = 0
    args.compact_replay = 'fp32'
    args.no_compile = False
    args.campaign = True
    args.experiment_config = dict(campaign='bsdm-paper-native-sdm-three-seeds-v1',
        mixer_backend='original SDM; native kernels with correctness repairs',
        initialization_comparison='Same-seed common shell/interface; native SDM initialization',
        native_sdm_commit=NATIVE_COMMIT, shared_residual_router=False,
        initial_memory_mode='full', capacity_rows=2048, selected_rows_per_role=64,
        matched_reference='BSDM N2048/K64; Wiki RTX4090 / Recall A40')
    args.extra_source_files = ('native_sdm_train.py', 'requirements-quality.txt',
        *[str(p.relative_to(ROOT)) for p in sorted(SOURCE.rglob('*.py'))])
    runner.ComparisonModel = NativeSDMModel
    runner.verify_fresh_normalizer_initialization = validate_common
    runner.train(args)
