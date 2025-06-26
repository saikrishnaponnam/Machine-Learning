from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Linear

# from src.layers import Linear
from src.layers.mutihead_attention import MultiHeadAttention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation=F.relu):
        super().__init__()
        self.linear1 = Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def backward(self):
        pass


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

    def backward(self):
        pass


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self, d_model, nhead, dim_feedforward, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, norm_first=False
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead)
        # Layer Norm after muti-head attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        # Feedforward network
        self.ffn = PositionWiseFeedForward(d_model, dim_feedforward, dropout, activation)
        # Layer Norm after feedforward network
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

    def _sa_block(self, src: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Self-attention sub-layer of the Transformer encoder layer.
        Inputs:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor, optional): Mask tensor to apply to attention scores. Shape should be broadcastable to (batch_size, seq_len, seq_len).
        """
        attn_output, _ = self.self_attn(src, src, src, mask=attn_mask)
        return self.dropout1(attn_output)

    def _ffn_block(self, src: Tensor) -> Tensor:
        """
        Feedforward network sub-layer of the Transformer encoder layer.
        Inputs:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        """
        ffn_output = self.ffn(src)
        return self.dropout2(ffn_output)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        """
        Forward pass of the Transformer encoder layer.
        Inputs:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor, optional): Mask tensor to apply to attention scores. Shape should be broadcastable to (batch_size, seq_len, seq_len).
            is_causal (bool): If True, applies a causal mask to the attention scores.
        Reference: https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/nn/modules/transformer.py#L760
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ffn_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask))
            x = self.norm2(x + self._ffn_block(x))

        return x

    def backward(self):
        pass


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)

    def backward(self):
        pass


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        # Multi-head attention over encoder outputs
        self.cross_attention = MultiHeadAttention(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

        # Feedforward network
        self.ffn = PositionWiseFeedForward(d_model, dim_feedforward, dropout, activation)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout3 = nn.Dropout(dropout)

    def _sa_block(self, src: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Self-attention sub-layer of the Transformer encoder layer.
        Inputs:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor, optional): Mask tensor to apply to attention scores. Shape should be broadcastable to (batch_size, seq_len, seq_len).
        """
        attn_output, _ = self.self_attn(src, src, src, mask=attn_mask)
        return self.dropout1(attn_output)

    def _ca_block(self, src: Tensor, memory: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Cross-attention sub-layer of the Transformer decoder layer.
        Inputs:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            memory (torch.Tensor): Encoder output tensor of shape (batch_size, memory_seq_len, d_model).
            memory_mask (torch.Tensor, optional): Mask tensor to apply to attention scores. Shape should be broadcastable to (batch_size, memory_seq_len, seq_len).
        """
        attn_output, _ = self.cross_attention(src, memory, memory, mask=attn_mask)
        return self.dropout2(attn_output)

    def _ffn_block(self, src: Tensor) -> Tensor:
        """
        Feedforward network sub-layer of the Transformer encoder layer.
        Inputs:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        """
        ffn_output = self.ffn(src)
        return self.dropout3(ffn_output)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass of the Transformer decoder layer.
        Inputs:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            tgt (torch.Tensor): Target tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor, optional): Mask tensor to apply to attention scores for src. Shape should be broadcastable to (batch_size, seq_len, seq_len).
            tgt_mask (torch.Tensor, optional): Mask tensor to apply to attention scores for tgt. Shape should be broadcastable to (batch_size, seq_len, seq_len).
            memory_mask (torch.Tensor, optional): Mask tensor to apply to attention scores for memory. Shape should be broadcastable to (batch_size, memory_seq_len, seq_len).
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._ca_block(self.norm2(x), memory, memory_mask)
            x = x + self._ffn_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._ca_block(x, memory, memory_mask))
            x = self.norm3(x + self._ffn_block(x))

        return x

    def backward(self):
        pass


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None
    ) -> Tensor:
        for layer in self.layers:
            tgt = layer(src, tgt, src_mask, tgt_mask)
        return self.norm(tgt)

    def backward(self):
        pass


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.encoder = TransformerEncoder(
            d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, layer_norm_eps, norm_first
        )
        self.decoder = TransformerDecoder(
            d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation, layer_norm_eps, norm_first
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output

    def backward(self):
        pass
