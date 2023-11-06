
import sys
sys.path.append('../')

import torch


def mse_loss(input, target):
    return torch.sum((input - target) ** 2)


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)
