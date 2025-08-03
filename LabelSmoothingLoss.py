import torch
import torch.nn as nn
import math

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, vocab_size, ignore_index=0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.ignore_index = ignore_index
        self.smoothing = label_smoothing
        self.vocab_size = vocab_size

    def forward(self, pred, target):
      pred = pred.log_softmax(dim=-1)
      target = target.contiguous().view(-1)
      pred = pred.view(-1, self.vocab_size)

      true_dist = torch.zeros_like(pred)

      true_dist.fill_(self.smoothing / (self.vocab_size - 2))
      ignore = target == self.ignore_index
      target = target.masked_fill(ignore, 0)
      true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
      true_dist.masked_fill_(ignore.unsqueeze(1), 0)

      return torch.mean(torch.sum(-true_dist * pred, dim=-1))
