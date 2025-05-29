import torch


class SGD:

    def __init__(self, parameters, lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        """
        Initialize the SGD optimizer.
        """
        self.dampening = dampening
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.parameters = parameters
        self.lr = lr

        if self.momentum > 0.0:
            self.vel = [torch.zeros_like(param) for param, _ in self.parameters]

    def step(self):
        """
        Perform a single optimization step.
        """
        for idx, (param, d_param) in enumerate(self.parameters):
            vel = self.vel[idx]
            vel.mul_(self.momentum).add_(d_param)
            if self.nesterov:
                param -= self.lr * (d_param + self.momentum * vel)
            else:
                param -= self.lr * vel

    # def zero_grad(self) -> None:
    #     for _, d_param in self.parameters:
    #         d_param.zero_()
