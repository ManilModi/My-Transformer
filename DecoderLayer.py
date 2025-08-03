import torch
import torch.nn as nn
import math
from MultiHeadAttention import MultiHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # 1. Masked Self-Attention
        attn1 = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn1))

        # 2. Encoder-Decoder Attention
        attn2 = self.cross_attn(x, kv=enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout2(attn2))

        # 3. Feed Forward
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff))

        return x
