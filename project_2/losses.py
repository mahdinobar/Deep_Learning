import torch
from module import *


class MSELoss(Module):
    """
    MSE loss
    with forward and backward passes
    """
    def forward(self, inputs, targets):
        """
        Forward computation of MSE
        :param inputs:
        :param targets:
        :return:
        """
        self.inputs = inputs
        self.targets = targets

        return torch.mean((inputs - targets)**2)

    def backward(self):
        """
        Backward computation of MSE
        :return:
        """
        errors = self.inputs - self.targets

        return 2 * errors / (self.targets.size(0) * self.targets.size(1))


class CrossEntropyLoss(Module):
    """
    Cross entropy loss function
    with forward and backward passes
    """
    def forward(self, inputs, targets):
        """
        Forward computation of the cross entropy loss
        :param inputs: main input
        :param targets: main output
        :return: cross entropy loss
        """
        self.inputs = inputs
        self.targets = targets
        self.n = inputs.size(0)

        loss = (-1. / self.n) * \
               ((inputs.gather(1, targets.view(-1, 1)).exp().squeeze() / inputs.exp().sum(1)).log().sum())

        return loss

    def backward(self):
        """
        Backward computation of the cross entropy loss
        :return:  Gradient of the cross entropy loss with respect to input
        """
        log_der = (-1 * self.inputs.exp()) / (self.inputs.exp().sum(1).view(-1, 1))

        log_der[torch.LongTensor(list(range(self.inputs.size(0)))), self.targets] = \
            log_der[torch.LongTensor(list(range(self.inputs.size(0)))), self.targets] + 1

        dl_dx = -(1./self.inputs.size(0)) * log_der

        return dl_dx
