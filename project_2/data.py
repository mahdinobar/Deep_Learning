import torch
import math


def generate_data(nb_samples):
    """ Generate dataset """
    # samples uniformly distributed between [0, 1]
    inputs = torch.empty(nb_samples, 2).uniform_(0, 1)

    # Label = 0 -> outside disk of radius 1/sqrt(2*pi)
    # Label = 1 -> inside disk
    targets = (inputs - 0.5).pow(2).sum(1).sub(1 / (2 * math.pi)).sign().mul(-1).add(1).div(2).long()

    return inputs, targets


def normalization(data):
    """ Normalization by mean and standard deviation """
    mean_value = data.mean(dim=0)
    std_value = data.std(dim=0)
    return data.sub_(mean_value).div_(std_value)


def convert_to_one_hot_labels(target):
    """ Convert target tensor to one hot vector representation """
    tmp = torch.FloatTensor(target.size(0), target.max() + 1).fill_(-1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp
