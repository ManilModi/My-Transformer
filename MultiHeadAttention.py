import torch
import torch.nn as nn
import math
from scaled_dot_product_attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, kv=None):
        kv = x if kv is None else kv

        B, T_q, _ = x.size()
        T_k = kv.size(1)

        Q = self.q_proj(x).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(kv).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(kv).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.d_k)

        return self.out_proj(attn_output)
