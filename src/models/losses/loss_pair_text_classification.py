from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
import torch.nn.functional as F


class PairTextClassifierLoss(nn.Module):
    def __init__(self) -> None:
        super(PairTextClassifierLoss, self).__init__()
        self.cls = nn.L1Loss(reduction='none')
    
    def forward(self, pred, target):
        loss = self.cls(pred.reshape(-1), target).sum()/pred.shape[0]
        return loss