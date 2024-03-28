from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.logger import Logger
from src.data.stsb import STSBDataset
from src.utils.metrics import BatchMeter
from src.utils.data_utils import DataUtils
from src.utils.tensorboard import Tensorboard
from src.models.pretrain.bert import BERTModel
from src.models.finetune.pair_text_classification import TextPairClassification
from src.models.losses.loss_pair_text_classification import PairTextClassifierLoss


class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.tsb = Tensorboard(config)
        self.datautils = DataUtils(config)
        self.logger = Logger(config).get_logger('TRAINING')
        self.create_data_loader()
        self.load_pretrained_model()
        self.create_model()
        self.best_loss = 0.0
            
    def load_pretrained_model(self):
        self.pretrained_model = BERTModel(vocab_size=len(self.train_dataset.vocab), 
                               num_hiddens=self.config['num_hiddens'],
                               ffn_num_hiddens=self.config['ffn_num_hiddens'],
                               num_heads=self.config['num_heads'],
                               num_blks=self.config['num_blks'],
                               dropout=self.config['dropout'],
                               max_len=self.config['max_len']).to(self.config['device'])
        
        ckpt = torch.load(self.config['pretrain_checkpoints']['best_ckpt_path'], map_location=self.config['device'])
        self.pretrained_model.load_state_dict(ckpt, strict=False)
    
    def create_data_loader(self):
        self.train_dataset = STSBDataset(self.config, data_type='train')
        self.valid_dataset = STSBDataset(self.config, data_type='test')
        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=self.config['Train']['shuffle'],
                                       batch_size=self.config['Train']['batch_size'],
                                       num_workers=self.config['Train']['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset,
                                       shuffle=self.config['Eval']['shuffle'],
                                       batch_size=self.config['Eval']['batch_size'],
                                       num_workers=self.config['Eval']['num_workers'])
    
    def create_model(self):
        self.model = TextPairClassification(self.pretrained_model).to(self.config['device'])
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['lr'])
        self.loss_fn = PairTextClassifierLoss()
    
    def train(self):
        metrics = {
            'loss': BatchMeter(),
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
            self.tsb.add_scalars(tag='loss', step=epoch, loss=metrics['loss'].get_value())

            current_loss = metrics['loss'].get_value()
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_ckpt(self.config['single_classification']['best_ckpt_path'], self.best_loss, epoch)
            self.save_ckpt(self.config['single_classification']['last_ckpt_path'], current_loss, epoch)

    def save_ckpt(self, save_path, best_loss, epoch):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_loss": best_loss,
            "epoch": epoch
        }
        self.logger.info(f"Saving checkpoint to {save_path}")
        torch.save(ckpt_dict, save_path)

if __name__ == "__main__":
    from src.configs.load_config import configuration
    cfg = configuration('finetune_stsb')
    trainer = Trainer(cfg)
    trainer.train()
        

