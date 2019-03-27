import torch
from module import *


class Relu():
    """
    Class for ReLU activation function
    """
    def forward(self, x):
        """
        Forward computation of ReLU
        :param x: input
        :return: result of ReLU Forward pass
        """
        self.x = x

        temp = torch.zeros_like(x)
        return torch.max(x, temp)

    def backward(self, y):  #TODO check
        """
        Backward computation of ReLU
        :param y:
        :return: result of ReLU Backward pass
        """
        temp = y.clone()
        # zero cells where input value are smaller or equal to 0
        temp[self.x <= 0] = 0

        return temp


class Tanh():
    """
    Class for tanh activation function
    """
    def forward(self, x):
        """
        Forward computation of tanh
        :param x: input
        :return: result of tanh forward pass
        """
        self.x = x

        return torch.tanh(x)

    def backward(self, y): #TODO check
        """
        Backward computation of tanh
        :param y:
        :return: result of tanh backward pass
        """
        return (1 - torch.tanh(self.x)**2)*y








