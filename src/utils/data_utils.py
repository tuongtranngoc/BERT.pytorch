from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import config as cfg
import torch
from typing import List, Tuple


class DataUtils:
    @classmethod
    def to_device(cls, data):
        if isinstance(data, torch.Tensor):
            return data.to(cfg['device'])
        elif isinstance(data, Tuple) or isinstance(data, List):
            for i, d in enumerate(data):
                if isinstance(d, torch.Tensor):
                    data[i] = d.to(cfg['device'])
                else:
                    Exception(f"{d} in {data} is not a tensor type")
            return data
        elif isinstance(data, torch.nn.Module):
            return data.to(cfg['device'])
        else:
            Exception(f"{data} is not a/tuple/list of tensor type")