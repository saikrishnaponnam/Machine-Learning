import torch
import torch.nn.functional as F

from src.layers.base_layer import BaseLayer


class MaxPool(BaseLayer):

    def __init__(self, pool_size: int, stride=1, padding=0):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

        # Backward pass cache
        self.max_ids = None
        self.input_size = None

    def forward(self, x: torch.Tensor):
        """
        Inputs:
            x: Input tensor of shape (N, C, H, W) where N is the batch size,
                C is the number of channels, H is the height, and W is the width.
        Returns:
            torch.Tensor: Output tensor after applying max pooling.
        """
        N, C, H, W = x.shape
        self.input_size = (H, W)
        x_pad = F.pad(x, (self.padding,) * 4, value=float("-inf"))
        x_unfold = F.unfold(x_pad, kernel_size=self.pool_size, stride=self.stride)
        x_unfold = x_unfold.view(N, C, self.pool_size * self.pool_size, -1)

        out, self.max_ids = x_unfold.max(dim=2)
        return out.view(N, C, (H - self.pool_size) // self.stride + 1, (W - self.pool_size) // self.stride + 1)

    def backward(self, d_out: torch.Tensor):
        """
        Backward pass for the max pooling layer.
        Inputs:
            d_out (torch.Tensor): Gradient of the loss with respect to the output of the layer. Shape (N, C_out, H_out, W_out)
        """
        N, C, H_out, W_out = d_out.shape
        dx_unfold = torch.zeros(
            (N, C, self.pool_size * self.pool_size, H_out * W_out), device=d_out.device, dtype=d_out.dtype
        )
        dx_unfold.scatter_(2, self.max_ids.unsqueeze(2), d_out.view(N, C, -1).unsqueeze(2))
        dx_unfold = dx_unfold.view(N, C * self.pool_size * self.pool_size, -1)

        dx = F.fold(dx_unfold, self.input_size, kernel_size=self.pool_size, stride=self.stride, padding=self.padding)

        if self.padding > 0:
            dx = dx[:, :, self.padding : -self.padding, self.padding : -self.padding]

        return dx

    def get_params(self):
        return []

    def __repr__(self):
        return f"MaxPool(pool_size={self.pool_size})"
