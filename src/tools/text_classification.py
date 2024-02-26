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
from src.models.finetune.single_text_classification import BertSingleClassifier
from src.models.losses.loss_single_text_classification import SingleTextClassifierLoss


class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.tsb = Tensorboard(config)
        self.datautils = DataUtils(config)
        self.logger = Logger(config).get_logger('TRAINING')
        self.create_data_loader()
        self.load_pretrained_model()
        self.create_model()
            
    def load_pretrained_model(self):
        self.pretrained_model = BERTModel(vocab_size=len(self.train_dataset.vocab), 
                               num_hiddens=self.config['num_hiddens'],
                               ffn_num_hiddens=self.config['ffn_num_hiddens'],
                               num_heads=self.config['num_heads'],
                               num_blks=self.config['num_blks'],
                               dropout=self.config['dropout'],
                               max_len=self.config['max_len']).to(self.config['device'])
        
        ckpt = torch.load(self.config['pretrain_checkpoints']['best_ckpt_path'], map_location=self.config['device'])['model']
        self.pretrained_model.load_state_dict(ckpt)
    
    def create_data_loader(self):
        self.train_dataset = SNLIDataset(self.config, data_type='train')
        self.valid_dataset = SNLIDataset(self.config, data_type='validation')
        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=self.config['Train']['shuffle'],
                                       batch_size=self.config['Train']['batch_size'],
                                       num_workers=self.config['Train']['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset,
                                       shuffle=self.config['Eval']['shuffle'],
                                       batch_size=self.config['Eval']['batch_size'],
                                       num_workers=self.config['Eval']['num_workers'])
    
    def create_model(self):
        self.model = BertSingleClassifier(self.pretrained_model).to(self.config['device'])
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'])
        self.loss_fn = SingleTextClassifierLoss()
    
    def train(self):
        metrics = {
            'loss': BatchMeter()
            }
        self.model.train()
        for epoch in range(1, self.config['Train']['epochs']):
            for i, (X, y) in enumerate(self.train_loader):
                X = self.datautils.to_device(X)
                y = self.datautils.to_device(y)
                out = self.model(X)
                loss = self.loss_fn(out, y)
                print(f"Epoch {epoch}/ Batch {i} - loss: {loss}", end='\r')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                metrics['loss'].update(loss.item())
            
            self.logger.info(f"Epoch {epoch} - loss: {metrics['loss'].get_value(): .4f}")


if __name__ == "__main__":
    from src.configs.load_config import configuration
    cfg = configuration('fintune_snli')
    trainer = Trainer(cfg)
    trainer.train()
        

