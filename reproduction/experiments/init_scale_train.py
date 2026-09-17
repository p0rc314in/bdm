"""One-change initialization ablation against the completed BF16 three-seed baseline."""
import argparse
import math
import torch
import train as runner
from campaign_model import CampaignModel

MULTIPLIER = 1 / math.sqrt(2)


class InitScaleModel(CampaignModel):
    def __init__(self, benchmark, arm, **options):
        super().__init__(benchmark, arm, **options)
        with torch.no_grad():
            for layer in self.layers:
                mixer = layer.attention
                # Change the existing FP32 draws before the unchanged BF16 cast.
                mixer.initial_row_factor.mul_(MULTIPLIER)
                mixer.initial_column_factor.mul_(MULTIPLIER)

    def resolved_model(self):
        record = super().resolved_model()
        record['initial_factor_multiplier'] = MULTIPLIER
        return record


def validate_initialization(network, benchmark, arm, seed, options):
    """Every tensor except the two factor tables per layer must match exactly."""
    with torch.random.fork_rng(devices=[]):
        reference = CampaignModel(benchmark, arm, seed=seed, **options)
    actual, expected = dict(network.named_parameters()), dict(reference.named_parameters())
    assert actual.keys() == expected.keys()
    changed = []
    for name, parameter in actual.items():
        value = expected[name]
        if name.endswith(('.initial_row_factor', '.initial_column_factor')):
            value = value * MULTIPLIER
            changed.append(name)
        assert torch.equal(parameter, value), name
        assert getattr(parameter, '_no_weight_decay', False) == getattr(expected[name], '_no_weight_decay', False), name
    assert len(changed) == 2 * len(network.layers)
    assert network.accounting() == reference.accounting()
    assert network.config.record() == reference.config.record()
    print(f'INITIALIZATION_ONLY multiplier={MULTIPLIER} changed_factor_tables={len(changed)} other_parameters=exact', flush=True)
    return reference.common_hash()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=('wiki', 'recall'), required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    args = p.parse_args()
    args.arm = 'bsdm_nvidia_n2048'
    args.microbatch = 8 if args.benchmark == 'wiki' else 32
    args.checkpoint_seconds = 900
    args.checkpoint_days = 30
    args.interrupt_after_step = 0
    args.compact_replay = 'fp32'  # Historical flag preserves the 16-event interval.
    args.training_state_dtype = 'bfloat16'  # Actual stored recurrent precision.
    args.no_compile = False
    args.campaign = True
    args.experiment_config = {
        'campaign': 'bsdm-init-scale-three-seeds-v1',
        'comparison': 'current-scale-vs-variance-matched-scale',
        'initial_factor_multiplier': MULTIPLIER,
        'initialization_comparison': 'Same-seed tensors; multiply only initial row/column factors by 1/sqrt(2) before BF16 conversion.',
        'baseline_run': f'bsdm-bf16-train-{args.benchmark}-s{args.seed}-{"4090" if args.benchmark == "wiki" else "a40"}-v1',
    }
    args.extra_source_files = ('init_scale_train.py', 'requirements-quality.txt')
    runner.ComparisonModel = InitScaleModel
    runner.verify_fresh_normalizer_initialization = validate_initialization
    runner.train(args)
