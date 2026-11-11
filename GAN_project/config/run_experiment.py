"""Active experiment: virtual 2.5 cm CNN + physical-module GNN WGAN-GP."""
import argparse

from pipeline.fhcal_training import train


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='NPZ with energies and particle conditions')
    parser.add_argument('--geometry', required=True, help='Verified physical module geometry JSON (cm)')
    parser.add_argument('--module-map', help='Verified ID -> [row,col] JSON for legacy [N,7,9,10] NPZ')
    parser.add_argument('--output', default='experiments/virtual25_gnn')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-critic', type=int, default=5)
    parser.add_argument('--val-ratio', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cpu', help='cpu or cuda[:index]')
    parser.add_argument('--resume', action='store_true')
    return train(parser.parse_args(argv))


if __name__ == '__main__':
    run()
