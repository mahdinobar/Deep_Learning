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


class CrossEntropy_Loss(Module):
    """
    A class for cross entropy loss
    """
    def forward(self,x,target):
        """
        Forward pass of the cross entropy loss

        Parameters:
            x:  main input
            target: main output
        Returns:
            Cross Entropy Loss
        """
        self.x=x
        self.target=target
        self.N=x.size()[0]
        exp_fyn=x.gather(1,target.view(-1,1)).exp().squeeze()
        sigma_exp_fk=x.exp().sum(1)
        Loss=(-1./self.N)*(sum((exp_fyn/sigma_exp_fk).log()))

    def backward(self):
        """
        Backward pass of the cross entropy loss
        Parameters:
                 -
        Returns:
                 Gradient of the cross entropy loss with respect to input
        """


        log_der=-1*self.x.exp()/sum(self.x.exp()).view(-1,1)
        log_der[torch.Tensor(range(self.x.size(0))),self.target]=log_der[torch.Tensor(range(self.x.size(0))),self.target]+1
        dL_dx=-(1./self.N)*log_der
        return dL_dx