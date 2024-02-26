from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import Logger
from src.utils.metrics import BatchMeter
from src.utils.data_utils import DataUtils
from src.utils.tensorboard import Tensorboard
from src.data.wikitext import WikiTextDataset
from src.models.pretrain.bert import BERTModel
from src.configs.load_config import configuration
from src.models.losses.loss_pretrain import PretrainLoss


class Trainer:
    def __init__(self, config) -> None:
        self.start_epoch = 0
        self.best_loss = 1e10
        self.config = config
        self.create_data_loader()
        self.create_model()
        self.datautil = DataUtils(config=config)
        self.tsb = Tensorboard(config=config)
        self.logger = Logger(config=config).get_logger('TRAINING')
        
    def create_data_loader(self):
        self.dataset = WikiTextDataset(self.config['max_len'], config=self.config)
        self.dataset_loader = DataLoader(dataset=self.dataset,
                                         shuffle=self.config['shuffle'],
                                         batch_size=self.config['batch_size'],
                                         num_workers=self.config['num_workers'])

    def create_model(self):
        self.model = BERTModel(vocab_size=len(self.dataset.vocab), 
                               dropout=self.config['dropout'],
                               num_blks=self.config['num_blks'],
                               num_heads=self.config['num_heads'],
                               num_hiddens=self.config['num_hiddens'],
                               ffn_num_hiddens=self.config['ffn_num_hiddens'],
                               max_len=self.config['max_len']).to(self.config['device'])
        self.loss_func = PretrainLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

    
    def train(self):
        metrics = {
            'total_loss': BatchMeter(),
            'mlm_loss': BatchMeter(),
            'nsp_loss': BatchMeter()
        }
        self.model.train()
        for epoch in range(self.start_epoch, self.config['epochs']):
            for i, X in enumerate(self.dataset_loader):
                token_ids, segments, valid_lens, pred_positions, mlm_weights, mlm_labels, nsp_labels = self.datautil.to_device(X)
                __, mlm_preds, nsp_preds = self.model(token_ids, segments, valid_lens.reshape(-1), pred_positions)
                total_loss, mlm_loss, nsp_loss = self.loss_func(mlm_preds, mlm_labels, mlm_weights, nsp_preds, nsp_labels, len(self.dataset.vocab))
                
                metrics['total_loss'].update(total_loss.item())
                metrics['nsp_loss'].update(nsp_loss.item())
                metrics['mlm_loss'].update(mlm_loss.item())

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch} - batch {i} - total_loss: {total_loss} - mlm_loss: {mlm_loss} - nsp_loss: {nsp_loss}", end='\r')

            self.logger.info(f"Epoch {epoch} - total_loss: {metrics['total_loss'].get_value(): .3f} - mlm_loss: {metrics['mlm_loss'].get_value(): .3f} - nsp_loss: {metrics['nsp_loss'].get_value(): .3f}")
            self.tsb.add_scalars(tag='total_loss',
                                    step=epoch,
                                    loss=metrics['total_loss'].get_value())
            self.tsb.add_scalars(tag='mlm_loss',
                                    step=epoch,
                                    loss=metrics['mlm_loss'].get_value())
            self.tsb.add_scalars(tag='nsp_loss',
                                    step=epoch,
                                    loss=metrics['nsp_loss'].get_value())
            
            current_loss = metrics['total_loss'].get_value()
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_ckpt(self.config['pretrain_checkpoints']['best_ckpt_path'], self.best_loss, epoch)
            self.save_ckpt(self.config['pretrain_checkpoints']['last_ckpt_path'], current_loss, epoch)

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
    cfg = configuration('pretrain_wikitext')
    trainer = Trainer(cfg)
    trainer.train()