# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer."""

import sys
sys.path.append('../')

import re
import os
import gymnasium as gym
import yaml
import torch
import random
import operator
import numpy as np
import torch.nn as nn

from typing import List, Iterable
from itertools import zip_longest
from collections import defaultdict

from src.logger import colorstr


def envs():
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)

    # reading benchmark names directly from retro requires
    # importing retro here, and for some reason that crashes tensorflow
    # in ubuntu
    _game_envs['retro'] = {
        'BubbleBobble-Nes',
        'SuperMarioBros-Nes',
        'TwinBee3PokoPokoDaimaou-Nes',
        'SpaceHarrier-Nes',
        'SonicTheHedgehog-Genesis',
        'Vectorman-Genesis',
        'FinalFight-Snes',
        'SpaceInvaders-Snes',
    }

    return _game_envs


def get_env_type(env: str, game_envs: defaultdict, env_type: str = None):
    env_id = env

    if env_type is not None:
        return env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in game_envs.keys():
        env_type = env_id
        env_id = [g for g in game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(
            env_id, game_envs.keys())

    return env_type, env_id


def get_default_network(env_type: str) -> str:
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_env_details(env: gym.Env):
    dummy_step, _ = env.reset()
    # self.env.observation_space.shape
    obs_dim = dummy_step['observation'].shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    return obs_dim, act_dim, max_ep_len, act_limit


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def _seed(env: gym.Env, device: torch.device, seed: int= 0):
    # set random seeds for reproduce
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in device.type:
        torch.cuda.manual_seed(seed)


class SegmentTree:
    """ Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    Attributes:
        capacity (int)
        tree (list)
        operation (function)
    """

    def __init__(self, capacity, operation, init_value):
        """Initialization.
        Args:
            capacity (int)
            operation (function)
            init_value (float)
        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(self, start, end, node, node_start, node_end):
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start=0, end=0):
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx, val):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity):
        """Initialization.
        Args:
            capacity (int)
        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start=0, end=0):
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound):
        """Find the highest index `i` about upper bound in the tree"""
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity):
        """Initialization.
        Args:
            capacity (int)
        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start=0, end=0):
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)



def parse_configs(filepath: str):
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def update_args(args, settings):

    for key, value in recursive_items(settings):
        if key == 'alias':
            args.name = value
        if key == 'name':
            args.env = value
        if key == 'extractor':
            res = value.strip('][').split(', ')
            hidden_sizes = list(map(int, res))
            args.hidden_sizes = hidden_sizes
        
        if key == 'arch':
            _game_envs = envs()
            env_type, env_id = get_env_type(env=args.env, game_envs=_game_envs)
            net_arch = get_default_network(env_type=env_type)

            if net_arch != value:
                print(f"{colorstr(['red', 'bold'], list(['Warning']))}: Model "
                      f"configuration was {value} whereas the default value "
                      f"for {args.env} is {net_arch}.")

            args.arch = value

        if key == 'activation':
            args.activation = value
        if key == 'max_ep_len':
            args.max_ep_len = value
        if key == 'pi_lr':
            args.lr_actor = value
        if key == 'q_lr':
            args.lr_critic = value
        if key == 'replay_size':
            args.replay_size = value
        if key == 'buffer_arch':
            args.buffer_arch = value
        if key == 'polyak':
            args.polyak = value
        if key == 'gamma':
            args.gamma = value
        if key == 'mu':
            args.mu = value
        if key == 'sigma':
            args.sigma = value
        if key == 'noise_dist':
            args.noise_dist = value
        if key == 'theta':
            args.theta = value
        if key == 'prior_eps':
            args.prior_eps = value
        if key == 'start_steps':
            args.start_steps = value
        if key == 'update_every':
            args.update_every = value
        if key == 'update_after':
            args.update_after = value
        if key == 'batch_size':
            args.batch_size = value
        if key == 'demo_episodes':
            args.demo_episodes = value
        if key == 'act_noise':
            args.act_noise = value
        if key == 'beta':
            args.beta = value
        if key == 'alpha':
            args.alpha = value
        if key == 'checkpoint_freq':
            args.checkpoint_freq = value
        if key == 'seed':
            args.seed = value
        if key == 'checkpoint_dir':
            args.checkpoint_dir = value
        if key == 'device':
            args.device = value
        if key == 'auto_save':
            args.auto_save = value
        if key == 'logger':
            head, tail = os.path.split(value)
            logger_name = os.path.splitext(tail)[0]
            args.logger_name = logger_name
        if key == 'elite_metric':
            args.elite_metric = value
    return args


def _mlp_configurations(hidden_sizes: List[int], activation: str, extractor: str):
    if activation.lower() == 'relu':
        activation = nn.ReLU
    elif activation.lower() == 'sigmoid':
        activation = nn.Sigmoid
    elif activation.lower() == 'tanh':
        activation = nn.Tanh
    else:
        raise NotImplementedError(f"Activation function {activation} is currently not supported. "
                                  f"Try one of the following:\n\t1. Sigmoid\n\t2. ReLU\n\t3. Tanh")
    return {
        'extractor': extractor,
        'activation': activation,
        'hidden_sizes': hidden_sizes
    }


def _logger_configuration(output_dir: str, output_fname: str, exp_name: str):
    return {
        'output_dir': output_dir,
        'output_fname': output_fname,
        'exp_name': exp_name
    }


def info():
    print(f"\n\n"
          f"\t\t        The {colorstr(['red', 'bold'], list(['Odysseus']))} suite serves as a framework for \n"
          f"\t\t    reinforcement learning agents training on OpenAI GYM \n"
          f"\t\t     defined environments. This repository was created \n"
          f"\t\t    created to help in future projects that would require \n"
          f"\t\t     such agents to find solutions for complex problems. \n"
          f"\n")

def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def compile_experiment_configs(agent):
        if agent.activation == nn.ReLU:
            agent.activation = 'relu'
        elif agent.activation == nn.Sigmoid:
            agent.activation = 'sigmoid'
        elif agent.activation == nn.Tanh:
            agent.activation = 'tanh'
        else:
            raise NotImplementedError(f"Activation function {agent.activation} is currently not supported. "
                                      f"Try one of the following:\n\t1. Sigmoid\n\t2. ReLU\n\t3. Tanh")

        return {
            'experiment':
            {
                'alias': agent.name,
                'logger': str(agent.logger.log_f_name.resolve())
            },
            'env':
            {
                'name': agent.env_id,
                'max_ep_len': agent.max_ep_len
            },
            'ddpg':
            {
                'extractor': str(agent.hidden_sizes),
                'arch': agent.extractor,
                'activation': agent.activation,
                'pi_lr': agent.lr_actor,
                'q_lr': agent.lr_critic,
                'replay_size': agent.replay_size,
                'polyak': agent.polyak,
                'gamma': agent.gamma,
                'prior_eps': agent.prior_eps,
                'start_steps': agent.start_steps,
                'update_every': agent.update_every,
                'update_after': agent.update_after
            },
            'training':
            {
                'batch_size': agent.batch_size,
                'demo_episodes': agent.demo_episodes
            },
            'exploration':
            {
                'act_noise': agent.act_noise,
                'mu': agent.mu,
                'sigma': agent.sigma,
                'noise_dist': agent.noise_dist,
                'theta': agent.theta
            },
            'per':
            {
                'buffer_arch': agent.buffer_arch,
                'beta': agent.beta,
                'alpha': agent.alpha
            },
            'auxiliary':
            {
                'checkpoint_freq': agent.checkpoint_freq,
                'seed': agent.seed,
                'checkpoint_dir': str(agent.checkpoint_dir.resolve()),
                'device': agent.device.type,
                'elite_metric': agent.elite_criterion,
                'auto_save': agent.auto_save
            }
        }
