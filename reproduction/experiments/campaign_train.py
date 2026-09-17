"""Full three-seed campaign using the existing W&B trainer and recovery."""
import argparse
import train as runner
from campaign_model import CampaignModel, ARMS, validate_common


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',choices=('wiki','recall'),required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--microbatch',type=int)
    parser.add_argument('--checkpoint-seconds',type=int,default=900)
    parser.add_argument('--checkpoint-days',type=int,choices=(30,90),default=30)
    parser.add_argument('--interrupt-after-step',type=int,default=0)
    parser.add_argument('--compact-replay',choices=('fp32',),default='fp32')
    args=parser.parse_args()
    args.microbatch=args.microbatch or (8 if args.benchmark=='wiki' else 32)
    args.no_compile=False
    args.campaign=True
    runner.ComparisonModel=CampaignModel
    runner.verify_fresh_normalizer_initialization=validate_common
    runner.train(args)
