"""Matched full training with BF16 persistent recurrent state and FP32 arithmetic."""
import argparse
import train as runner
from campaign_model import CampaignModel, validate_common

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=('wiki','recall'), required=True)
    p.add_argument('--seed', type=int, choices=(0,1,2), required=True)
    args = p.parse_args()
    args.arm = 'bsdm_nvidia_n2048'
    args.microbatch = 8 if args.benchmark == 'wiki' else 32
    args.checkpoint_seconds = 900
    args.checkpoint_days = 30
    args.interrupt_after_step = 0
    args.compact_replay = 'fp32'  # Preserve the existing 16-event interval.
    args.training_state_dtype = 'bfloat16'  # Storage overrides dtype, not interval.
    args.no_compile = False
    args.campaign = True
    runner.ComparisonModel = CampaignModel
    runner.verify_fresh_normalizer_initialization = validate_common
    runner.train(args)
