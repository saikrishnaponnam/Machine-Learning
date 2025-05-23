import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Inputs:
            - pred: A tensor containing predictions, of shape (N, C)
            - target: A tensor containing target values, of shape (N, C)
        Returns:
            - loss: A scalar tensor representing the L1 loss
        """
        self.target = target
        N = pred.size(0)
        loss = (pred - target).abs().sum() / N
        return loss

    def backward(self):
        """
        Computes the gradient of the loss with respect to the predictions.
        """
        dx = (self.pred - self.target).sign()
        return dx


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Inputs:
            - pred: A tensor containing predictions, of shape (N, C)
            - target: A tensor containing target values, of shape (N, C)
        Returns:
            - loss: A scalar tensor representing the mean squared error loss
        """
        self.target = target
        N = pred.size(0)
        loss = ((pred - target) ** 2).sum() / N
        return loss

    def backward(self):
        """
        Computes the gradient of the loss with respect to the predictions.
        """
        dx = 2 * (self.target - self.pred) / self.target.size(0)
        return dx


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """
        Inputs:
            - pred: A tensor containing class probabilities, of shape (N, C)
            - target: A tensor containing class labels, of shape (N,)
        Returns:
            - loss: A scalar tensor representing the cross-entropy loss
        """
        self.target = target
        N = logits.size(0)
        logits -= logits.max(dim=1, keepdim=True).values
        exp_logits = logits.exp()
        self.probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        loss = -self.probs[torch.arange(N), target].log().sum() / N
        return loss

    def backward(self):
        """
        Computes the gradient of the loss with respect to the predictions.
        """
        dx = self.probs.clone()
        dx[range(self.target.size(0)), self.target] -= 1
        dx /= self.target.size(0)
        return dx
