import math

import torch


def kaiming(d_in: int, d_out: int, nonlinearity="relu"):
    # TODO: Read https://arxiv.org/abs/1502.01852 and implement custom
    gain = torch.nn.init.calculate_gain(nonlinearity)

    # Linear Layer
    std = gain / math.sqrt(d_in)
    with torch.no_grad():
        return torch.normal(0, std, size=(d_in, d_out))

# def xavier_(d_in, d_out):
#     """ """
#
#     with torch.no_grad():
#         return torch.normal(0, math.sqrt(1 / d), size=(n, d))
