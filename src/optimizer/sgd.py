class SGD:

    def __init__(
            self, parameters, lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False
    ):
        """
        Initialize the SGD optimizer.
        """
        super().__init__(parameters)
        self.dampening = dampening
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """
        Perform a single optimization step.
        """
        for param, d_param in self.parameters():
            param.data -= d_param * self.lr

    # def zero_grad(self, set_to_none: bool = True) -> None:
