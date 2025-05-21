import torch

from src.layers.base_layer import BaseLayer


class Flatten(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Flatten the input tensor.
        Inputs:
            -x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            - Flattened tensor of shape (batch_size, channels * height * width)
        """
        self.input_shape = x.shape
        N = x.shape[0]
        return x.view(N, -1)

    def backward(self, d_out: torch.Tensor):
        return d_out.view(self.input_shape)

    def get_params(self):
        return []

    def __repr__(self):
        return "Flatten()"
