from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import datasets
import os

from . import *

class SNLIDataset(nn.Module):
    def __init__(self, data_type='train') -> None:
        super(SNLIDataset, self).__init__()
        if not os.path.join(cfg['finetune_data_path']):
            dataset = datasets.load_from_disk('snli')
            dataset.save_to_disk(cfg['finetune_data_path'])
        dataset = datasets.load_from_disk(cfg['finetune_data_path'])[data_type]
        


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = SNLIDataset()
    data_loader = DataLoader(dataset, batch_size=28, shuffle=True)
    import ipdb; ipdb.set_trace()
    