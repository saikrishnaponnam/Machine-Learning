import torch

from src.layers.base_layer import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, x: torch.Tensor):
        if self.training:
            self.mask = (torch.rand_like(x) > self.dropout_rate).float()
            return x * self.mask / (1 - self.dropout_rate)
        else:
            return x

    def backward(self, d_out: torch.Tensor):
        if self.training:
            return d_out * self.mask / (1 - self.dropout_rate)
        else:
            return d_out

    def get_params(self):
        return []

    def __repr__(self):
        return f"Dropout(dropout_rate={self.dropout_rate})"
