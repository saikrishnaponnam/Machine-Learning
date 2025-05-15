import torch

from src.layers.base_layer import BaseLayer


class LeakyReLU(BaseLayer):

    def __init__(self, slope=1e-2):
        super().__init__()
        self.slope = slope

    def forward(self, x: torch.Tensor):
        self.mask = x >= 0
        return torch.where(self.mask, x, self.slope * x)

    def backward(self, d_out: torch.Tensor):
        return torch.where(self.mask, d_out, self.slope * d_out)

    def get_params(self):
        return []
