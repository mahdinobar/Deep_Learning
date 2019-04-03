import torch


class SGD():
    def __init__(self, model, lr, momentum=0):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.momentum_buffer = {}

    def step(self):
        for param in self.model.param():
            if p not in self.momentum_buffer.key():
                self.momentum_buffer[param] = torch.FloatTensor(param[0].size()).zero_()
            self.momentum_buffer[param] = self.momentum_buffer[param].mul_(self.momentum).add_(self.lr*param[1])
            param[0].sub_(self.momentum_buffer[param])

