#!/usr/bin/env python3
"""Run the paper's experiments from public inputs; inspect commands before launch."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='show experiments and hardware')
    setup = sub.add_parser('setup', help='install the pinned runtime in the current Python environment')
    setup.add_argument('--profile', choices=['inputs', 'quality', 'performance', 'babylm'], required=True)
    setup.add_argument('--work', type=Path, default=ROOT/'runs')
    prepare = sub.add_parser('prepare', help='build and checksum inputs from pinned public sources')
    prepare.add_argument('dataset', choices=['wiki', 'recall', 'memory', 'babylm'])
    prepare.add_argument('--work', type=Path, default=ROOT/'runs')
    run = sub.add_parser('run', help='execute the full selected experiment; same output resumes it')
    run.add_argument('experiment', choices=['wiki', 'recall', 'init-scale', 'memory', 'babylm', 'performance'])
    run.add_argument('--model', choices=['bdm','sdm','attention','gdn1','gdn2','mom'], default='bdm')
    run.add_argument('--seed', type=int, choices=[0,1,2], default=0)
    run.add_argument('--benchmark', choices=['wiki','recall'], default='wiki', help='initialization comparison benchmark')
    run.add_argument('--factor-scale', choices=['reference','unit'], default='reference')
    run.add_argument('--lambda', dest='coefficient', type=float, choices=[0,.001,.003,.01,.03,.06,.12], default=0.)
    run.add_argument('--phase', choices=['train','prefill','decode'], default='decode')
    run.add_argument('--context', type=int, choices=[8192,16384,32768,65536,131072,262144,524288,1048576], default=8192)
    run.add_argument('--work', type=Path, default=ROOT/'runs')
    run.add_argument('--dry-run', action='store_true', help='print the exact command without starting computation')
    summary = sub.add_parser('report', help='tabulate and plot newly generated outputs')
    summary.add_argument('--work', type=Path, default=ROOT/'runs')
    summary.add_argument('--output', type=Path, default=ROOT/'runs/report')
    check = sub.add_parser('check', help='bounded packaging, input-code and recovery tests; no training')
    check.add_argument('--torch', action='store_true', help='also check small CPU model construction')
    args = parser.parse_args()
    from reproduction import runner
    if args.command == 'list':
        print('wiki/recall: BDM, SDM, attention, GDN1, GDN2, MoM; seeds 0/1/2; RTX4090/A40')
        print('init-scale: BDM reference or unit factor amplitude; wiki/recall, three seeds')
        print('memory: BDM occupancy coefficients 0,.001,.003,.01,.03,.06,.12; RTX4090')
        print('babylm: BDM or SDM N=2048, or attention; 28 evaluations and all seven fine-tunes; H100 80GB')
        print('performance: BDM, SDM, attention, GDN1, GDN2; train/prefill/decode; H200')
    elif args.command == 'setup': runner.setup(args)
    elif args.command == 'prepare': runner.prepare(args)
    elif args.command == 'run': runner.run(args)
    elif args.command == 'report':
        from reproduction.report import report
        report(args.work,args.output)
    else:
        from reproduction.check import check
        check(args.torch)


if __name__ == '__main__': main()
