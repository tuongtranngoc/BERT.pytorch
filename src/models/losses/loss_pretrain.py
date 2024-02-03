from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn


class PretrainLoss(nn.Module):
    def __init__(self) -> None:
        super(PretrainLoss, self).__init__()
        self.mlm_loss = nn.CrossEntropyLoss()
        self.nsp_loss = nn.CrossEntropyLoss()
    
    def forward(self, mlm_preds, mlm_targets, mlm_weights, nsp_preds, nsp_targets, vocab_size):
        mlm_loss = self.mlm_loss(mlm_preds.reshape(-1, vocab_size), mlm_targets.reshape(-1)) * mlm_weights.reshape(-1, 1)
        mlm_loss = mlm_loss.sum() / (mlm_weights.sum() + 1e-8)
        nsp_loss = self.nsp_loss(nsp_preds, nsp_targets)

        total_loss = mlm_loss + nsp_loss
        return total_loss, mlm_loss, nsp_loss