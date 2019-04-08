import torch


class SGD():
    """ SGD optimization """
    def __init__(self, model, learning_rate, momentum=0):
        super(SGD, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_buffer = {}

    def step(self):
        for param in self.model.param():
            if param[0] not in self.momentum_buffer.keys():
                self.momentum_buffer[param[0]] = torch.empty(param[0].size()).zero_()
            self.momentum_buffer[param[0]] = self.momentum_buffer[param[0]].mul_(self.momentum).\
                add_(self.learning_rate * param[1])
            param[0].sub_(self.momentum_buffer[param[0]])
