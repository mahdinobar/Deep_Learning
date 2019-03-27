import torch
from module import *


class LossMSE(Module):
    """
    Class for MSE loss function
    """
    def forward(self, x, target):
        """
        Forward computation of MSE
        :param x:
        :param target:
        :return: result of MSE
        """
        self.x = x
        self.target = target

        return torch.mean((x - target)**2)

    def backward(self):
        """
        Backward computation of MSE
        :return:
        """
        error = self.x - self.target

        return 2 * error / (self.target.size(0) * self.target.size(1))
