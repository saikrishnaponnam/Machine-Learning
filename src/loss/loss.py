import torch
import torch.nn as nn


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
        d_in = self.probs.clone()
        d_in[range(self.target.size(0)), self.target] -= 1
        d_in /= self.target.size(0)
        return d_in
