import torch

from src.init import kaiming
from src.layers.base_layer import BaseLayer


class Linear(BaseLayer):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.x = None
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
        self.x = x
        return x.matmul(self.weights) + self.bias

    def backward(self, d_out: torch.Tensor):
        """
        Inputs:
            - d_out: Upstream derivative tensor, of shape (N, out_features)
        Returns:
            - dx: Gradients w.r.t x, shape (N, in_features)
        """
        self.dW.copy_(self.x.T.matmul(d_out))
        self.db.copy_(d_out.sum(dim=0))
        dx = d_out.matmul(self.weights.T)

        return dx

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"

    def cuda(self, device=None):
        self.weights = self.weights.cuda()
        self.bias = self.bias.cuda()
        self.dW = self.dW.cuda()
        self.db = self.db.cuda()

    def cpu(self):
        self.weights = self.weights.cpu()
        self.bias = self.bias.cpu()
        self.dW = self.dW.cpu()
        self.db = self.db.cpu()
