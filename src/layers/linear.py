import torch
import torch.nn as nn

from src.init import kaiming


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = kaiming(in_features, out_features)
        self.bias = torch.zeros(out_features)

        self.dW = torch.zeros_like(self.weights)
        self.db = torch.zeros_like(self.bias)

    def get_params(self):
        """
        Returns:
            - params: A list of tuples of (parameter, grads) (weights and bias)
        """
        return [(self.weights, self.dW), (self.bias, self.db)]

    def forward(self, x: torch.Tensor):
        """
        Inputs:
            - x: A tensor containing input data, of shape (N, in_features)

        Returns:
            output tensor of shape (N, out_features)
        """
        return x.matmul(self.weights) + self.bias

    def backward(self, x: torch.Tensor, d_out: torch.Tensor):
        """
        Inputs:
            - d_out: Upstream derivative tensor, of shape (N, out_features)
        Returns:
            - dw: Gradients w.r.t W, shape (in_features, out_features)
            - db: Gradients w.r.t b, shape (out_features)
            - dx: Gradients w.r.t x, shape (N, in_features)
        """
        self.dW = x.T.matmul(d_out)
        self.db = d_out.sum(dim=0)
        dx = d_out.mm(self.weights.T)

        return dx
