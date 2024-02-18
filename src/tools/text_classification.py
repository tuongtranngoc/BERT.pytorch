from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import Logger
from src.data.snli import SNLIDataset
from src.utils.metrics import BatchMeter
from src.utils.data_utils import DataUtils
from src.utils.tensorboard import Tensorboard
from src.models.pretrain.bert import BERTModel


class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.tsb = Tensorboard(config)
        self.datautils = DataUtils(config)
        self.logger = Logger(config).get_logger('TRAINING')
            
    def load_pretrained_model(self):
        self.pretrained_model = BERTModel(vocab_size=len(self.dataset.vocab), 
                               num_hiddens=self.config['num_hiddens'],
                               ffn_num_hiddens=self.config['ffn_num_hiddens'],
                               num_heads=self.config['num_heads'],
                               num_blks=self.config['num_blks'],
                               dropout=self.config['dropout'],
                               max_len=self.config['max_len']).to(self.config['device'])
        self.pretrained_model.load_state_dict(self.config['pretrain_checkpoints']['best_ckpt_path'])
    

    def create_data_loader(self):
        self.train_dataset = SNLIDataset(data_type='train')
        self.valid_dataset = SNLIDataset(data_type='valid')
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.config['Train']['batch_size'],
                                       shuffle=self.config['Train']['shuffle'],
                                       num_workers=self.config['Train']['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.config['Eval']['batch_size'],
                                       shuffle=self.config['Eval']['shuffle'],
                                       num_workers=self.config['Eval']['num_workers'])

    def create_model(self):
        pass
        

