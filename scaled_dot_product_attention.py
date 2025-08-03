import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-1e9'))

    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights
