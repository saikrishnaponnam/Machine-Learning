from typing import List, overload, Optional

import torch
import torch.nn as nn


class Sequential(nn.Module):
    """
    A simple sequential model that allows adding layers and performing a forward pass.
    """

    @overload
    def __init__(self) -> None: ...

    def __init__(self, layers: Optional[List[nn.Module]] = None):
        super().__init__()
        self.layers = layers if layers is not None else []

    def add(self, layer: nn.Module):
        """
        Add a layer to the model.
        """
        self.layers.append(layer)

    def forward(self, x: torch.Tensor):
        """
        Perform a forward pass through the model.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out: torch.Tensor):
        """
        Perform a backward pass through the model.
        """
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

    def get_params(self):
        """
        Get parameters and gradients of all layers.
        """
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params

    def summary(self):
        print("Model Summary:")
        for i, layer in enumerate(self.layers):
            # params = layer.get_params()
            print(f"({i}): {layer}")
