import torch
from module import *


class Linear(Module):
    def __init__(self, input_dim, output_dim):
        """
        Class of Linear module
        :param input_dim:
        :param output_dim:
        """
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # weights and biases
        self.w = torch.empty(output_dim, input_dim).normal_()
        self.b = torch.empty(1, output_dim).normal_()
        # Weights and biases gradient
        self.dl_dw = torch.empty(self.w.size())
        self.dl_db = torch.empty(self.b.size())

    def forward(self, x):
        """
        Foward computation
        :param x:
        :return:
        """
        output = x.mm(self.w.t()) + self.b
        self.x = x
        return output

    def backward(self, y):
        """
        Backward computation
        :param y:
        :return:
        """
        dl_dw = y.t().mm(self.x)
        dl_db = y.sum(0).view(1, -1)
        dl_dx = y.mm(self.w)

        self.dl_dw = dl_dw
        self.dl_db = dl_db
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
        x_out = x
        return x_out

    def backward(self, y):
        """
        Module backward pass
        :param y:
        :return:
        """
        for module in reversed(self.modules):
            # backward pass of each module
            y = module.backward(y)

    def param(self):
        """
        parameters of the model
        :return:
        """
        all_parameters = []
        for module in self.modules:
            all_parameters.extend(module.param())
        return all_parameters
