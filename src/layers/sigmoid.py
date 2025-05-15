import torch

from src.layers.base_layer import BaseLayer


class Sigmoid(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        self.output = 1 / (1 + torch.exp(-x))
        return self.output

    def backward(self, d_out: torch.Tensor):
        """
        Compute the gradient of the loss with respect to the input.
        """
        sigmoid = self.output
        return d_out * sigmoid * (1 - sigmoid)

    def get_params(self):
        return []
