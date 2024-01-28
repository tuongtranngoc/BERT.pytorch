from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import datasets
import transformers

import torch
import torch.nn as nn

from . import *

class BookCorpusDataset(nn.Module):
    def __init__(self) -> None:
        super(BookCorpusDataset, self).__init__()
        self.dataset = datasets.load_dataset()
        