from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPairClassification(nn.Module):
    def __init__(self, pretrain_bert) -> None:
        super(TextPairClassification, self).__init__()
        self.encoder = pretrain_bert.encoder
        self.hidden = pretrain_bert.hidden
        self.output = nn.LazyLinear(1)
        
    def forward(self, x):
        tokens_X, segments_X, valid_lens_X = x
        encoder_X = self.encoder(tokens_X, segments_X, valid_lens_X)
        return self.output(self.hidden(encoder_X[:, 0, :]))