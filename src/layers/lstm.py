from typing import Optional

import torch
from torch import Tensor

from src.init import kaiming
from src.layers import RNNBase


class LSTM(RNNBase):
    """
    A simple implementation of an LSTM layer.

    Args:
        input_size (int): Number of features in the input.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int, optional): Number of recurrent layers.
        batch_first (bool, optional): If True, input/output tensors are (batch, seq, feature).
        bidirectional (bool, optional): If True, becomes a bidirectional RNN.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        # self.bidirectional = bidirectional

        # Weights and biases initialization
        self.W_x = kaiming(input_size, 4 * hidden_size)  # Concatenated weights [W_ii, W_if, W_iC, W_io]
        self.W_h = kaiming(hidden_size, 4 * hidden_size)
        self.b = torch.zeros(4 * self.hidden_size)

        self.dW_x = torch.zeros_like(self.W_x)
        self.dW_h = torch.zeros_like(self.W_h)
        self.db = torch.zeros_like(self.b)

        # Cache for backward pass
        self.cache = []

    def forward(self, x: Tensor, hx: Optional[tuple[Tensor, Tensor]] = None):
        """
        Forward pass through the RNN layer.
        Inputs:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, input_size) if batch_first is False,
                              otherwise (batch, seq_len, input_size) .
            h_0 (torch.Tensor): Initial hidden state of shape (D * num_layers, batch, hidden_size).
                               If None, initializes to zeros. D=2 if bidirectional, otherwise D=1.
            c_0 (torch.Tensor): Initial cell state of shape (D * num_layers, batch, hidden_size).
        Returns:
            output (torch.Tensor): Output tensor of shape (batch, seq_len, D * hidden_size) if batch_first is True,
                          otherwise (seq_len, batch, hidden_size).
            h_n (torch.tensor): hidden state tensor of shape (D * num_layer, batch_size, hidden_size).
            c_n (torch.tensor): cell state tensor of shape (D * num_layer, batch_size, hidden_size).
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch, _ = x.shape

        if hx is None:
            h_t = torch.zeros(batch, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch, self.hidden_size, device=x.device)
        else:
            h_t, c_t = hx

        self.cache = []
        outputs = []

        for t in range(seq_len):
            x_t = x[t]
            gates = x_t @ self.W_x + h_t @ self.W_h + self.b
            f_t, i_t, c_tilde_t, o_t = torch.split(gates, self.hidden_size, dim=1)

            f_t = torch.sigmoid(f_t)
            i_t = torch.sigmoid(i_t)
            c_tilde_t = torch.tanh(c_tilde_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_t + i_t * c_tilde_t
            h_t = o_t * torch.tanh(c_t)

            outputs += [h_t]
            self.cache += [(x_t, f_t, i_t, o_t, c_tilde_t, c_t, h_t)]

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_t, c_t)

    def backward(self, d_out: Tensor):
        """
        Backward pass through the RNN layer.
        Inputs:
            d_out (torch.Tensor): Gradient of the loss with respect to the output,
                                  shape (seq_len, batch, hidden_size).

        Returns:
            dx (torch.Tensor): Gradient with respect to the input, shape (seq_len, batch, input_size).
            dh0 (torch.Tensor): Gradient with respect to the initial hidden state, shape (batch, hidden_size).
        """
        seq_len, N, _ = d_out.shape

        dW_x = torch.zeros_like(self.W_x)
        dW_h = torch.zeros_like(self.W_h)
        db = torch.zeros_like(self.b)

        dx = [torch.empty()] * seq_len
        # Stores gradients from the next time step (0 for the last time step)
        dh_t = torch.zeros_like(d_out[0])
        dc_t = torch.zeros_like(d_out[0])

        for t in reversed(range(seq_len)):
            x_t, f_t, i_t, o_t, c_tilde_t, c_t, h_t = self.cache[t]

            c_prev = self.cache[t - 1][5] if t > 0 else torch.zeros_like(c_t)
            h_prev = self.cache[t - 1][6] if t > 0 else torch.zeros_like(h_t)

            dh_t += d_out[t]
            dc_t += dh_t * o_t * (1 - torch.tanh(c_t) ** 2)

            df_t = dc_t * c_prev * f_t * (1 - f_t)
            di_t = dc_t * c_tilde_t * i_t * (1 - i_t)
            dc_tilde_t = dc_t * i_t * (1 - c_tilde_t**2)
            do_t = dh_t * torch.tanh(c_t) * o_t * (1 - o_t)

            d_gates = torch.cat([df_t, di_t, dc_tilde_t, do_t], dim=1)

            dW_x += x_t.T @ d_gates
            dW_h += h_prev.T @ d_gates
            db += d_gates.sum(dim=0)

            dx[t] = d_gates @ self.W_x.T
            dh_t = d_gates @ self.W_h.T
            dc_t = dc_t * f_t

        return torch.cat(dx, dim=0), (dh_t, dc_t)

    def get_params(self):
        return [(self.W_x, self.dW_x), (self.W_h, self.dW_h), (self.b, self.db)]

    def cuda(self, device=None):
        self.W_x = self.W_x.cuda(device)
        self.W_h = self.W_h.cuda(device)
        self.b = self.b.cuda(device)
        self.dW_x = self.dW_x.cuda(device)
        self.dW_h = self.dW_h.cuda(device)
        self.db = self.db.cuda(device)

    def cpu(self):
        self.W_x = self.W_x.cpu()
        self.W_h = self.W_h.cpu()
        self.b = self.b.cpu()
        self.dW_x = self.dW_x.cpu()
        self.dW_h = self.dW_h.cpu()
        self.db = self.db.cpu()
