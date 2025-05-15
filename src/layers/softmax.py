import torch

from src.layers.base_layer import BaseLayer


class Softmax(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Input:
            x: A tensor containing scores, of shape (N, C)
        output:
            A tensor of shape as x containing class probabilities
        """
        scores = x - x.max(dim=1, keepdim=True).values
        exp_scores = scores.exp()
        probs = exp_scores / exp_scores.sum(dim=1, keepdim=True)
        self.probs = probs
        return probs

    def backward(self, d_out: torch.Tensor):
        raise NotImplementedError("Softmax backward pass is not implemented.")

    def get_params(self):
        return []
