import torch


class Adam:

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        """
        Initialize the Adam optimizer.
        """
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0

        self.m = [torch.zeros_like(param) for param, _ in self.params]
        self.v = [torch.zeros_like(param) for param, _ in self.params]

    def step(self):
        """
        Perform a single optimization step.
        """
        self.t += 1
        for idx, (param, d_param) in enumerate(self.params):
            m = self.m[idx]
            v = self.v[idx]

            m.mul_(self.betas[0]).add_(d_param, alpha=1 - self.betas[0])
            v.mul_(self.betas[1]).addcmul_(d_param, d_param, value=1 - self.betas[1])

            m_bias = m / (1 - self.betas[0] ** self.t)
            v_bias = v / (1 - self.betas[1] ** self.t)

            param -= self.lr * m_bias / (v_bias.sqrt() + self.eps)
