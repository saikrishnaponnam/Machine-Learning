import torch

from src.layers.base_layer import BaseLayer


class Tanh(BaseLayer):

    def __init__(self):
        super().__init__()

        # Backward pas cache
        self.output = None

    def forward(self, x: torch.Tensor):
        exp_x = torch.exp(-2 * x)
        self.output = (1 - exp_x) / (1 + exp_x)
        return self.output

    def backward(self, d_out: torch.Tensor):
        return d_out * (1 - self.output**2)

    def get_params(self):
        return []

    def __repr__(self):
        return "Tanh()"
