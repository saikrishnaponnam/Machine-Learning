from typing import Optional

import torch
from torch import Tensor

from src.init import kaiming
from src.layers import RNNBase
from src.layers.dropout import Dropout


class LSTM(RNNBase):
    """
    A simple implementation of an LSTM layer.

    Args:
        input_size (int): Number of features in the input.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int, optional): Number of recurrent layers.
        bias (bool, optional): If False, then the layer does not use bias weights.
        batch_first (bool, optional): If True, input/output tensors are (batch, seq, feature).
        dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each
                           LSTM layer except the last layer, with dropout probability equal to dropout.
        bidirectional (bool, optional): If True, becomes a bidirectional RNN.
    """

    def __init__(
        self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False
    ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Multi-layer weights and biases
        self.W_x = []
        self.W_h = []
        self.b = []
        self.dW_x = []
        self.dW_h = []
        self.db = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.W_x.append(kaiming(in_size, 4 * hidden_size)) # Concatenated weights [W_ii, W_if, W_iC, W_io]
            self.W_h.append(kaiming(hidden_size, 4 * hidden_size))
            self.b.append(torch.zeros(4 * hidden_size))
            self.dW_x.append(torch.zeros_like(self.W_x[-1]))
            self.dW_h.append(torch.zeros_like(self.W_h[-1]))
            self.db.append(torch.zeros_like(self.b[-1]))

        if bidirectional:
            raise NotImplementedError("Bidirectional LSTM is not implemented yet.")

        # Dropout layers (applied to outputs of all but last layer)
        self.dropouts = [Dropout(dropout) if dropout > 0 and l < num_layers - 1 else None for l in range(num_layers)]

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
            h_n (torch.Tensor): hidden state tensor, of last token in input sequence, of shape (D * num_layer, batch_size, hidden_size).
            c_n (torch.Tensor): cell state tensor of shape (D * num_layer, batch_size, hidden_size).
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch, _ = x.shape

        if hx is None:
            h_t = [torch.zeros(batch, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
        else:
            h_t, c_t = list(hx[0]), list(hx[1])

        self.cache = [[] for _ in range(self.num_layers)]
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            for l in range(self.num_layers):
                gates = x_t @ self.W_x[l] + h_t[l] @ self.W_h[l] + self.b[l]
                i_t, f_t, c_tilde_t, o_t = torch.split(gates, self.hidden_size, dim=1)

                i_t = torch.sigmoid(i_t)
                f_t = torch.sigmoid(f_t)
                c_tilde_t = torch.tanh(c_tilde_t)
                o_t = torch.sigmoid(o_t)

                c_t[l] = f_t * c_t[l] + i_t * c_tilde_t
                h_t[l] = o_t * torch.tanh(c_t[l])
                self.cache[l].append((x_t, f_t, i_t, o_t, c_tilde_t, c_t[l], h_t[l]))
                x_t = h_t[l]
                # Apply dropout except for last layer
                if self.dropouts[l] is not None:
                    x_t = self.dropouts[l].forward(x_t)
            outputs.append(h_t[-1])
        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (torch.stack(h_t), torch.stack(c_t))

    def backward(self, d_out: Tensor):
        """
        Backward pass through the RNN layer.
        Inputs:
            d_out (torch.Tensor): Gradient of the loss with respect to the output,
                                  shape (seq_len, batch, hidden_size).
        Returns:
            dx (torch.Tensor): Gradient with respect to the input, shape (seq_len, batch, input_size).
            dh0 (torch.Tensor): Gradient with respect to the initial hidden state, shape (num_layers, batch, hidden_size).
        """
        seq_len, batch, _ = d_out.shape

        dW_x = [torch.zeros_like(self.W_x[l]) for l in range(self.num_layers)]
        dW_h = [torch.zeros_like(self.W_h[l]) for l in range(self.num_layers)]
        db = [torch.zeros_like(self.b[l]) for l in range(self.num_layers)]

        dx = [torch.zeros(batch, self.input_size, device=d_out.device, dtype=d_out.dtype) for _ in range(seq_len)]
        # Stores gradients from the next time step (0 for the last time step)
        dh_t = [
            torch.zeros(batch, self.hidden_size, device=d_out.device, dtype=d_out.dtype) for _ in range(self.num_layers)
        ]
        dc_t = [
            torch.zeros(batch, self.hidden_size, device=d_out.device, dtype=d_out.dtype) for _ in range(self.num_layers)
        ]

        for t in reversed(range(seq_len)):
            d_input = d_out[t]
            # Backward through layers
            for l in reversed(range(self.num_layers)):
                # Apply dropout backward except for last layer
                if self.dropouts[l] is not None:
                    d_input = self.dropouts[l].backward(d_input)
                x_t, f_t, i_t, o_t, c_tilde_t, c_t, h_t = self.cache[l][t]
                c_prev = self.cache[l][t - 1][5] if t > 0 else torch.zeros_like(c_t)
                h_prev = self.cache[l][t - 1][6] if t > 0 else torch.zeros_like(h_t)

                dh_t[l] += d_input
                dc_t[l] += dh_t[l] * o_t * (1 - torch.tanh(c_t) ** 2)

                df_t = dc_t[l] * c_prev * f_t * (1 - f_t)
                di_t = dc_t[l] * c_tilde_t * i_t * (1 - i_t)
                dc_tilde_t = dc_t[l] * i_t * (1 - c_tilde_t**2)
                do_t = dh_t[l] * torch.tanh(c_t) * o_t * (1 - o_t)

                d_gates = torch.cat([di_t, df_t, dc_tilde_t, do_t], dim=1)

                dW_x[l] += x_t.T @ d_gates
                dW_h[l] += h_prev.T @ d_gates
                db[l] += d_gates.sum(dim=0)

                d_input = d_gates @ self.W_x[l].T
                dh_t[l] = d_gates @ self.W_h[l].T
                dc_t[l] = dc_t[l] * f_t
            dx[t] = d_input

        for l in range(self.num_layers):
            self.dW_x[l].copy_(dW_x[l])
            self.dW_h[l].copy_(dW_h[l])
            self.db[l].copy_(db[l])
        dh0 = torch.stack(dh_t)
        dc0 = torch.stack(dc_t)
        return torch.stack(dx, dim=0), (dh0, dc0)

    def get_params(self):
        return (
            [(self.W_x[l], self.dW_x[l]) for l in range(self.num_layers)]
            + [(self.W_h[l], self.dW_h[l]) for l in range(self.num_layers)]
            + [(self.b[l], self.db[l]) for l in range(self.num_layers)]
        )

    def cuda(self, device=None):
        self.W_x = [w.cuda(device) for w in self.W_x]
        self.W_h = [w.cuda(device) for w in self.W_h]
        self.b = [b.cuda(device) for b in self.b]
        self.dW_x = [w.cuda(device) for w in self.dW_x]
        self.dW_h = [w.cuda(device) for w in self.dW_h]
        self.db = [b.cuda(device) for b in self.db]

    def cpu(self):
        self.W_x = [w.cpu() for w in self.W_x]
        self.W_h = [w.cpu() for w in self.W_h]
        self.b = [b.cpu() for b in self.b]
        self.dW_x = [w.cpu() for w in self.dW_x]
        self.dW_h = [w.cpu() for w in self.dW_h]
        self.db = [b.cpu() for b in self.db]
