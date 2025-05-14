from abc import abstractmethod, ABC

import torch
import torch.nn as nn


class BaseLayer(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor):
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
