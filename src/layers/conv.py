import torch
import torch.nn.functional as F

from src.init import kaiming
from src.layers.base_layer import BaseLayer


class Conv(BaseLayer):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple, optional): Stride of the convolution. Default is 1.
            padding (int or tuple, optional): Padding added to both sides of the input. Default is 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = kaiming(in_channels, out_channels, kernel_size)
        self.weights = self.weights.view(self.out_channels, -1)
        self.bias = torch.zeros(out_channels)

        self.dW = torch.zeros_like(self.weights)
        self.db = torch.zeros_like(self.bias)

        # Cache for backward pass
        self.x_unfold = None
        self.input_size = None

    def forward(self, x: torch.Tensor):
        """
        Inputs:
            x: Input tensor of shape (N, C, H, W) where N is the batch size,
                C is the number of channels, H is the height, and W is the width.
        Returns:
            torch.Tensor: Output tensor after applying convolution.
        """
        N, _, H, W = x.shape
        self.input_size = (H, W)
        self.x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        out = self.weights.matmul(self.x_unfold) + self.bias.view(1, -1, 1)

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        return out.view(N, self.out_channels, H_out, W_out)

    def backward(self, d_out: torch.Tensor):
        """
        Backward pass for the convolution layer.
        Inputs:
            d_out (torch.Tensor): Gradient of the loss with respect to the output of the layer. Shape (N, C_out, H_out, W_out)
        """
        N = d_out.shape[0]
        d_out_flat = d_out.view(N, self.out_channels, -1)

        dw = d_out_flat.matmul(self.x_unfold.transpose(1, 2))
        dw = dw.sum(dim=0).view(self.weights.shape)
        self.dW.copy_(dw)

        self.db.copy_(d_out_flat.sum(dim=(0, 2)))

        dx = self.weights.T.matmul(d_out_flat)
        dx = F.fold(dx, self.input_size, self.kernel_size, stride=self.stride, padding=self.padding)

        return dx

    def get_params(self):
        return [(self.weights, self.dW), (self.bias, self.db)]

    def __repr__(self):
        return f"Conv(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"

    def cuda(self, device=None):
        self.weights = self.weights.cuda()
        self.bias = self.bias.cuda()
        self.dW = self.dW.cuda()
        self.db = self.db.cuda()

    def cpu(self):
        self.weights = self.weights.cpu()
        self.bias = self.bias.cpu()
        self.dW = self.dW.cpu()
        self.db = self.db.cpu()
