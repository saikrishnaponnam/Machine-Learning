from torch import nn
from torch.nn import Linear

from src.layers.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_k = d_model // num_heads

        # Contains weights for all heads
        self.W_q = Linear(d_model, d_model, bias=False)  # W_q.shape  = d_model x (d_k x num_heads)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.W_o = Linear(d_model, d_model, bias=False)  # W_o.shape  = (d_k x num_heads) x d_model

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of the multi-head attention layer.
        Inputs:
            token_encodings (torch.Tensor): Tensor of shape (batch_size, seq_len, d_model) containing the input token encodings.
            mask (torch.Tensor, optional): Mask tensor to apply to attention scores. Shape should be broadcastable to (batch_size, seq_len, seq_len).
        """
        batch_size = q.shape[0]

        # Compute Q, K, V for all heads
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention for each head
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads and apply final linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)

        return output, attn_weights
