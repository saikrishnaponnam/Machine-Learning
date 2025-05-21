import torch

from src.layers.base_layer import BaseLayer


class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Apply the ReLU activation function.
        """
        self.mask = x > 0
        return x * self.mask

    def backward(self, d_out: torch.Tensor):
        """
        Compute the gradient of the ReLU activation function.
        """
        return d_out * self.mask

    def get_params(self):
        return []

    def __repr__(self):
        return "ReLU()"
