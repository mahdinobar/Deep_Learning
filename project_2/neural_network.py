import torch
from module import *


class Linear(Module):
    def __init__(self, input_dim, output_dim, w0, b0):
        """
        Class of Linear module
        :param input_dim:
        :param output_dim:
        """
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # weights and biases
        self.w = w0
        self.b = b0
        # Weights and biases gradient
        self.dl_dw = torch.empty(self.w.size())
        self.dl_db = torch.empty(self.b.size())

    def forward(self, x):
        """
        Foward computation
        :param x:
        :return:
        """
        self.x = x
        output = x.mm(self.w.t()) + self.b
        return output

    def backward(self, y):
        """
        Backward computation
        :param y:
        :return:
        """
        self.dl_dw = y.t().mm(self.x)
        self.dl_db = y.sum(0).view(1, -1)
        dl_dx = y.mm(self.w)

        return dl_dx

    def param(self):
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]


class Sequential(Module):
    def __init__(self, *modules):
        """
        Combine other mudules into one model
        :param modules:
        """
        super(Sequential, self).__init__()
        self.modules = modules

    def forward(self, x):
        """
        Module forward pass
        :param x:
        :return:
        """
        for module in self.modules:
            # forward pass of each module
            x = module.forward(x)
        return x

    def backward(self, y):
        """
        Module backward pass
        :param y:
        :return:
        """
        for module in self.modules:
            # backward pass of each module
            y = module.backward(y)
        return y

    def param(self):
        """
        parameters of the model
        :return:
        """
        all_parameters = []
        for module in self.modules:
            all_parameters.extend(module.param())

        return all_parameters
