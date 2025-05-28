import torch

from src.layers.base_layer import BaseLayer


class BatchNorm(BaseLayer):

    def __init__(self, num_features: int, eps=1e-5, momentum=0.1, track_running_stats=True):
        """
        Initialize the BatchNorm layer.
        Args:
            num_features (int): Number of features or channels in the input tensor.
            eps (float): A small value to avoid division by zero.
            track_running_stats (bool): Whether to track running statistics for inference.
        """
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        self.training = True
        if self.track_running_stats:
            self.running_mean = torch.zeros(num_features)
            self.running_var = torch.zeros(num_features)

        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

        self.dg = torch.zeros(num_features)
        self.db = torch.zeros(num_features)

        # Cache for backward pass
        self.x_hat = None
        self.std_inv = None
        self.x_centered = None

    def forward(self, x: torch.Tensor):
        """
        Apply batch normalization to the input tensor.
        Inputs:
            x (torch.Tensor): Input tensor of shape (N, D).
                where N is the batch size, D is the number of features or channels
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, correction=0)

            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * mean + self.momentum * self.running_mean
                self.running_var = (1 - self.momentum) * var + self.momentum * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        self.x_centered = x - mean
        self.std_inv = 1.0 / torch.sqrt(var + self.eps)
        self.x_hat = self.x_centered * self.std_inv
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, d_out: torch.Tensor):
        """
        Inputs:
            d_out (torch.Tensor): Gradient of the loss with respect to the output of the batch normalization layer.
        Returns:
            torch.Tensor: Gradient of the loss with respect to the input of the batch normalization layer.
        """
        N = self.d_out.shape[0]
        self.dg.copy_((d_out * self.x_hat).sum(dim=0))
        self.db.copy_(d_out.sum(dim=0))

        dx_hat = d_out * self.gamma
        dvar = (-0.5 * dx_hat * self.x_centered * (self.std_inv**3)).sum(dim=0)
        dmean = (-dx_hat * self.std_inv).sum(dim=0) + dvar * (-2.0 * self.x_centered.mean(dim=0))
        dx = dx_hat * self.std_inv + dvar * 2.0 * self.x_centered / N + dmean / N
        return dx

    def get_params(self):
        return [(self.gamma, self.dg), (self.beta, self.db)]

    def __repr__(self):
        return f"BatchNorm(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, track_running_stats={self.track_running_stats})"

    def cuda(self, device=None):
        self.gamma = self.gamma.cuda()
        self.beta = self.beta.cuda()
        self.dg = self.dg.cuda()
        self.db = self.db.cuda()

    def cpu(self):
        self.gamma = self.gamma.cpu()
        self.beta = self.beta.cpu()
        self.dg = self.dg.cpu()
        self.db = self.db.cpu()


class SpatialBatchNorm(BatchNorm):

    def __init__(self, num_features: int, eps=1e-5, momentum=0.1, track_running_stats=True):
        super().__init__(num_features, eps, momentum, track_running_stats)

    def forward(self, x: torch.Tensor):
        """
        Apply spatial batch normalization to the input tensor.
        Inputs:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
                where N is the batch size, C is the number of channels, H is height, W is width
        """
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)  # Reshape to (N*H*W, C)
        out = super().forward(x)
        out = out.view(N, H, W, C).permute(0, 3, 1, 2)  # Reshape back to (N, C, H, W)
        return out

    def backward(self, d_out: torch.Tensor):
        N, C, H, W = d_out.shape
        d_out = d_out.permute(0, 2, 3, 1).reshape(-1, C)
        dx = super().backward(d_out)
        dx = dx.view(N, H, W, C).permute(0, 3, 1, 2)
        return dx

    def __repr__(self):
        return f"SpatialBatchNorm(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, track_running_stats={self.track_running_stats})"
