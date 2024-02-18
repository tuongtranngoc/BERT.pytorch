from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import List, Tuple
import torch


class DataUtils:
    def __init__(self, config) -> None:
        self.config = config

    def to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.config['device'])
        elif isinstance(data, Tuple) or isinstance(data, List):
            for i, d in enumerate(data):
                if isinstance(d, torch.Tensor):
                    data[i] = d.to(self.config['device'])
                else:
                    Exception(f"{d} in {data} is not a tensor type")
            return data
        elif isinstance(data, torch.nn.Module):
            return data.to(self.config['device'])
        else:
            Exception(f"{data} is not a/tuple/list of tensor type")