import torch
import torch.nn as nn
import math
from Encoder import Encoder
from Decoder import Decoder

def make_subsequent_mask(self, size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()  # upper triangular
    return mask.unsqueeze(0).unsqueeze(1)  # shape: (1, 1, T, T)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 d_ff=2048, num_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_pad_mask(self, seq, pad_idx):
        # Shape: (batch, 1, 1, seq_len)
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_subsequent_mask(self, size):
        return torch.tril(torch.ones((size, size), dtype=torch.bool)).unsqueeze(0).unsqueeze(1)

    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0, tgt_mask=None):
      src_mask = self.make_pad_mask(src, src_pad_idx)

      if tgt_mask is None:
          tgt_pad_mask = self.make_pad_mask(tgt, tgt_pad_idx)
          tgt_sub_mask = self.make_subsequent_mask(tgt.size(1)).to(tgt.device)
          tgt_mask = tgt_pad_mask & tgt_sub_mask

      enc_out = self.encoder(src, src_mask)
      dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)

      return self.fc_out(dec_out)

