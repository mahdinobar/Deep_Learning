import torch
import math


def generate_data(nb_samples):
    # samples uniformly distributed between [0, 1]
    inputs = torch.empty(nb_samples, 2).uniform_(0, 1)

    # Label = 0 -> outside disk of radius 1/sqrt(2*pi)
    # Label = 1 -> inside disk
    targets = (inputs - 0.5).pow(2).sum(1).sub(1 / (2 * math.pi)).sign().mul(-1).add(1).div(2).long()

    return inputs, targets
