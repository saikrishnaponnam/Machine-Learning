from typing import Optional

import torch
from torch import Tensor

from src.init import kaiming
from src.layers import ReLU, Tanh, RNNBase


class RNN(RNNBase):
    """
    A simple implementation of a vanilla RNN layer.

    Args:
        input_size (int): Number of features in the input.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int, optional): Number of recurrent layers.
        batch_first (bool, optional): If True, input/output tensors are (batch, seq, feature).
        bidirectional (bool, optional): If True, becomes a bidirectional RNN.
        nonlinearity (str, optional): Nonlinearity to use ('relu' or 'tanh').
    """

    def __init__(
        self, input_size, hidden_size, num_layers=1, batch_first=False, bidirectional=False, nonlinearity="relu"
    ):
        super().__init__()
        assert nonlinearity in ("relu", "tanh")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        # self.bidirectional = bidirectional

        self.nonlinearity = nonlinearity
        self.activation = ReLU() if nonlinearity == "relu" else Tanh()

        # Weights and biases initialization
        self.w_xh = []
        self.w_hh = []
        self.bh = []
        self.dw_xh = []
        self.dw_hh = []
        self.dbh = []
        for layer in range(num_layers):
            input_size = input_size if layer == 0 else hidden_size
            self.w_xh.append(kaiming(input_size, hidden_size))
            self.w_hh.append(kaiming(hidden_size, hidden_size))
            self.bh.append(torch.zeros(hidden_size))
            self.dw_xh.append(torch.zeros_like(self.w_xh[-1]))
            self.dw_hh.append(torch.zeros_like(self.w_hh[-1]))
            self.dbh.append(torch.zeros_like(self.bh[-1]))

        # Cache for backward pass
        self.hidden_states = None
        self.inputs = None
        self.raw_outputs = None
        self.x = None

    def forward(self, x: Tensor, hx: Optional[Tensor] = None):
        """
        Forward pass through the RNN layer.
        Inputs:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, input_size) if batch_first is False,
                              otherwise (batch, seq_len, input_size) .
            hx (torch.Tensor): Initial hidden state of shape (D * num_layers, batch, hidden_size).
                               If None, initializes to zeros. D=2 if bidirectional, otherwise D=1.
        Returns:
            output (torch.Tensor): Output tensor of shape (batch, seq_len, D * hidden_size) if batch_first is True,
                          otherwise (seq_len, batch, hidden_size).
            h_n (torch.tensor): hidden state tensor of shape (D * num_layer, batch_size, hidden_size).
        """
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, N, _ = x.shape
        if hx is None:
            hx = [torch.zeros(N, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        self.inputs = []
        self.hidden_states = []
        self.raw_outputs = []
        self.x = x

        h_prev = hx
        for t in range(seq_len):
            x_t = x[t]  # (N, input_size)
            h_t = []
            a_t = []
            # TODO: Vectorize the multi-layer computation
            for l in range(self.num_layers):
                a_t_l = x_t.matmul(self.w_xh[l]) + h_prev[l].matmul(self.w_hh[l]) + self.bh[l]
                h_t_l = self.activation(a_t_l)
                a_t += [a_t_l]
                h_t += [h_t_l]
                x_t = h_t_l
            h_prev = h_t
            self.inputs += [x[t]]
            self.hidden_states += [h_t]
            self.raw_outputs += [a_t]

        # Get final layer output across all time steps / sequence
        output = torch.stack([self.hidden_states[t][-1] for t in range(seq_len)], dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, torch.stack(h_prev, dim=0)

    def backward(self, d_out: Tensor):
        """
        Backward pass through the RNN layer.

        Args:
            d_out (torch.Tensor): Gradient of the loss with respect to the output,
                                  shape (seq_len, batch, hidden_size).

        Returns:
            dx (torch.Tensor): Gradient with respect to the input, shape (seq_len, batch, input_size).
            dh0 (torch.Tensor): Gradient with respect to the initial hidden state, shape (batch, hidden_size).
        """

        seq_len, N, _ = self.x.shape

        dw_xh = [torch.zeros_like(self.w_xh[l]) for l in range(self.num_layers)]
        dw_hh = [torch.zeros_like(self.w_hh[l]) for l in range(self.num_layers)]
        dbh = [torch.zeros_like(self.bh[l]) for l in range(self.num_layers)]

        dx = torch.zeros_like(self.x)
        dh_next = [torch.zeros(N, self.hidden_size, device=self.x.device) for _ in range(self.num_layers)]

        for t in reversed(range(seq_len)):
            d_layer = d_out[t]

            for l in reversed(range(self.num_layers)):
                x_t = self.inputs[t] if l == 0 else self.hidden_states[t][l - 1]
                h_t = self.hidden_states[t][l]
                a_t = self.raw_outputs[t][l]
                h_prev = self.hidden_states[t - 1][l] if t > 0 else torch.zeros_like(h_t)
                dh = d_layer + dh_next[l]
                if self.nonlinearity == "relu":
                    d_at = (a_t > 0).float() * dh
                else:
                    d_at = (1 - h_t**2) * dh

                if l == 0:
                    dx[t] = d_at.matmul(self.w_xh[l].T)

                dw_xh[l] += x_t.T.matmul(d_at)
                dw_hh[l] += h_prev.T.matmul(d_at)
                dbh[l] += d_at.sum(dim=0)
                dh_next[l] = d_at.matmul(self.w_hh[l].T)

        for l in range(self.num_layers):
            self.dw_xh[l].copy_(dw_xh[l])
            self.dw_hh[l].copy_(dw_hh[l])
            self.dbh[l].copy_(dbh[l])

        dh0 = dh_next
        return dx, dh0

    def get_params(self):
        return (
            [(self.w_xh[l], self.dw_xh[l]) for l in range(self.num_layers)]
            + [(self.w_hh[l], self.dw_hh[l]) for l in range(self.num_layers)]
            + [(self.bh[l], self.dbh[l]) for l in range(self.num_layers)]
        )

    def cuda(self, device=None):
        for l in range(self.num_layers):
            self.w_xh[l] = self.w_xh[l].cuda(device)
            self.w_hh[l] = self.w_hh[l].cuda(device)
            self.bh[l] = self.bh[l].cuda(device)
            self.dw_xh[l] = self.dw_xh[l].cuda(device)
            self.dw_hh[l] = self.dw_hh[l].cuda(device)
            self.dbh[l] = self.dbh[l].cuda(device)

    def cpu(self):
        for l in range(self.num_layers):
            self.w_xh[l] = self.w_xh[l].cpu()
            self.w_hh[l] = self.w_hh[l].cpu()
            self.bh[l] = self.bh[l].cpu()
            self.dw_xh[l] = self.dw_xh[l].cpu()
            self.dw_hh[l] = self.dw_hh[l].cpu()
            self.dbh[l] = self.dbh[l].cpu()
