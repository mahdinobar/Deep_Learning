import torch


class SGD():
    def __init__(self, model, learning_rate, momentum=0):
        """
        SGD optimization
        :param model:
        :param learning_rate:
        :param momentum:
        """
        super(SGD, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.buffer = {}

    def step(self):
        # for loop over model parameters
        for param in self.model.param():
            # if not in buffer
            if param[0] not in self.buffer.keys():
                self.buffer[param[0]] = torch.empty(param[0].size()).zero_()

            # v = v * momentum + learning_rate * gradient
            self.buffer[param[0]] = self.buffer[param[0]].mul_(self.momentum).\
                add_(self.learning_rate * param[1])
            # param (weight) - v
            param[0].sub_(self.buffer[param[0]])
