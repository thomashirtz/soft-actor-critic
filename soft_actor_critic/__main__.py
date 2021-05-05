import argparse
from typing import Optional

from .train import train as train_model
# from .test import test as test_model


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic')
parser.add_argument(
    '-e', '--env-name', default="LunarLander-v2", type=str,
    help='Gym environment to train on (default: %(default)s'
)
parser.add_argument(
    '-b', '--batch-size', default=256, type=int,
    help='Batch sized used by the agent during the learning phase (default: %(default)s'
)
parser.add_argument(
    '-m', '--memory-size', default=1_000_000, type=int,
    help='Size of the replay buffer (default: %(default)s'
)
parser.add_argument(
    '-l', '--learning-rate', default=3e-4, type=float,
    help='Learning rate used for the optimization of the temperature and the networks (default: %(default)s'
)
parser.add_argument(
    '-g', '--gamma', default=0.99, type=float,
    help='Discount rate used by the agent (default: %(default)s'
)
parser.add_argument(
    '-t', '--tau', default=0.005, type=float,
    help='Value used for the progressive update of the target networks (default: %(default)s'
)
parser.add_argument(
    '-n', '--num-steps', default=1_000_000, type=int,
    help='Number of steps in the training process (default: %(default)s'
)
parser.add_argument(
    '--run-name', default=None, type=Optional[str],
    help='Name used for saving the weights and the logs (default: generated using the "get_run_name" function)'
)
parser.add_argument(
    '-s', '--start-step', default=1_000, type=int,
    help='Step after which the agent starts to learn (default: %(default)s'
)
parser.add_argument(
    '--hidden-layers', nargs='+', type=int, default=[256, 256],
    help='<Required> Set flag'
)
parser.add_argument(
    '-d','--checkpoint-directory', default='../checkpoints/', type=str,
    help='Root directory in which the run folder will be created (default: %(default)s'
)
parser.add_argument(
    '--seed', default=1, type=int,
    help='Seed used for the run (default: %(default)s'
)
parser.add_argument(
    '-a', '--alpha', default=0.2, type=float,
    help='Starting temperature (default: %(default)s'
)

args = parser.parse_args()
# todo separate train and test, but by default train, also require filename if test
# python3 pytorch-soft-actor-critic
# python3 pytorch-soft-actor-critic train
# python3 pytorch-soft-actor-critic test SAC-2021-blabla

if __name__ == '__main__':
    train_model(**vars(args))
