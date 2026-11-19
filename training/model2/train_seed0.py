"""Run the published seed-0 setup for a bounded period on one GPU."""
import argparse
import json
import time
from common import HERE
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error('--hours must be positive')
    setup = json.loads((HERE / 'experiments.json').read_text())['refine'][0]
    checkpoint = HERE / 'runs' / setup['name'] / 'last.pt' if args.resume else None
    train(setup, time.time() + 3600 * args.hours, checkpoint)
