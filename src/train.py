from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from src.utils.data_utils import DataUtils
from src.data.wikitext import WikiTextDataset
from src.models.pretrain.bert import BERTModel
from src.models.losses.loss_pretrain import PretrainLoss


class Trainer:
    def __init__(self) -> None:
        self.start_epoch = 0
        self.create_data_loader()
        self.create_model()
        
    def create_data_loader(self):
        self.dataset = WikiTextDataset(config['max_len'])
        self.dataset_loader = DataLoader(dataset=self.dataset,
                                         batch_size=config['batch_size'],
                                         shuffle=config['shuffle'],
                                         num_workers=config['num_worker'])

    def create_model(self):
        self.model = BERTModel(vocab_size=len(self.dataset.vocab), 
                               num_hiddens=config['num_hiddens'],
                               ffn_num_hiddens=config['ffn_num_hiddens'],
                               num_heads=config['num_heads'],
                               num_blks=config['num_blks'],
                               dropout=config['dropout'],
                               max_len=config['max_len'])
        self.loss_func = PretrainLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])


    def train(self):
        self.model.train()
        for epoch in range(self.start_epoch, config['epochs']):
            for X in self.dataset_loader:
                token_ids, segments, valid_lens, pred_positions, mlm_weights, mlm_labels, nsp_labels = DataUtils.to_device(X)
                



    def save_ckpt(self):
        pass

    def resume_training(self):
        pass


if __name__ == "__main__":
    pass