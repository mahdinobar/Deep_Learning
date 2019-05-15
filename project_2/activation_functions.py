import torch
from module import *


class Relu(Module):
    """
    ReLU activation function
    with forward and backward passes
    """
    def forward(self, inputs):
        """
        Forward computation of ReLU
        :param inputs: input
        :return: result of ReLU Forward pass
        """
        self.inputs = inputs

        temp = torch.zeros_like(inputs)
        return torch.max(inputs, temp)

    def backward(self, y):
        """
        Backward computation of ReLU
        :param y: gradient
        :return: result of ReLU Backward pass
        """
        res = y.clone()
        # zero cells where input value are smaller or equal to 0
        res[self.inputs <= 0] = 0

        return res


class Tanh(Module):
    """
    Tanh activation function
    with forward and backward passes
    """
    def forward(self, inputs):
        """
        Forward computation of tanh
        :param inputs: input
        :return: result of tanh forward pass
        """
        self.inputs = inputs

        return torch.tanh(inputs)

    def backward(self, y):
        """
        Backward computation of tanh
        :param y: gradient
        :return: result of tanh backward pass
        """
        return (1 - self.inputs.tanh()**2) * y








