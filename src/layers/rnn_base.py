from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class RNNBase(nn.Module, ABC):
    """
    Abstract base class for RNN layers.

    Defines the required interface for RNN layers, including forward and backward passes,
    and parameter retrieval.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None):
        """
        Forward pass through the layer.
        """
        raise NotImplementedError("Forward pass not implemented.")

    @abstractmethod
    def backward(self, d_out: torch.Tensor):
        """
        Backward pass through the layer.
        """
        raise NotImplementedError("Backward pass not implemented.")

    @abstractmethod
    def get_params(self):
        """
        Returns:
            - params: A list of tuples of (parameter, grads) (weights and bias)
        """
        raise NotImplementedError("get_params not implemented.")
