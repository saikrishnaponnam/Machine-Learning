import math

import torch


def kaiming(d_in: int, d_out: int, k=None, nonlinearity="relu"):
    # TODO: Read https://arxiv.org/abs/1502.01852 and implement custom
    gain = torch.nn.init.calculate_gain(nonlinearity)

    if k is None:
        # Linear Layer
        std = gain / math.sqrt(d_in)
        weights = torch.normal(0, std, size=(d_in, d_out))
    else:
        # Convolution layer
        std = gain / math.sqrt(d_in * k * k)
        weights = torch.normal(0, std, size=(d_out, d_in, k, k))

    return weights


def xavier(d_in: int, d_out: int):

    std = math.sqrt(2.0 / (d_in + d_out))
    return torch.normal(0, std, size=(d_in, d_out))
