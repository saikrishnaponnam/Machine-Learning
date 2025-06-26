from typing import Optional

import torch
from torch import nn, Tensor


def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    d_k = query.shape[-1]
    scores = query.matmul(key.transpose(-2, -1)) / (d_k**0.5)

    # mask is matrix of 0/1 values, where 1 means the position is valid
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float("-inf"))

    attn_weights = scores.softmax(dim=-1)
    output = attn_weights.matmul(value)

    return output, attn_weights


class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        """
        Self-Attention layer that computes attention scores and outputs a weighted sum of values.
        Inputs:
            d_model (int): Dimension of the input token encodings.
        """
        super().__init__()
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, token_encodings: Tensor, mask: Optional[Tensor] = None):
        """
        Forward pass of the self-attention layer.
        Inputs:
            token_encodings (torch.Tensor): Tensor of shape (batch_size, seq_len, d_model) containing the input token encodings.
            mask (torch.Tensor, optional): Mask tensor to apply to attention scores. Shape should be broadcastable to (batch_size, seq_len, seq_len).
        """
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        return scaled_dot_product_attention(q, k, v, mask)


if __name__ == "__main__":
    torch.manual_seed(42)

    tokens = torch.randn(3, 2)
    print(tokens)

    self_attn = SelfAttention(2)

    print(self_attn(tokens))

    mask = torch.tril(torch.zeros(3, 3))
    print(self_attn(tokens, mask))
    print(self_attn(tokens, torch.ones(3, 3)))
