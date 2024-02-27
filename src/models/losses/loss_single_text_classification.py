from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTextClassifierLoss(nn.Module):
    def __init__(self) -> None:
        super(SingleTextClassifierLoss, self).__init__()
        self.cls = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred, target):
        loss = self.cls(pred, target).sum()/pred.shape[0]
        return loss