from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import config
from src.utils.logger import Logger
from src.data.snli import SNLIDataset
from src.utils.metrics import BatchMeter
from src.utils.data_utils import DataUtils
from src.utils.tensorboard import Tensorboard
from src.models.pretrain.bert import BERTModel

logger = Logger.get_logger('TRAINING')


class Trainer:
    def __init__() -> None:
        pass
            
    def load_pretrained_model(self):
        self.pretrained_model = BERTModel(vocab_size=len(self.dataset.vocab), 
                               num_hiddens=config['num_hiddens'],
                               ffn_num_hiddens=config['ffn_num_hiddens'],
                               num_heads=config['num_heads'],
                               num_blks=config['num_blks'],
                               dropout=config['dropout'],
                               max_len=config['max_len']).to(config['device'])
        self.pretrained_model.load_state_dict(config['pretrain_checkpoints']['best_ckpt_path'])

    def create_data_loader(self):
        self.train_dataset = SNLIDataset(data_type='train')
        self.valid_dataset = SNLIDataset(data_type='valid')
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size='')
        

