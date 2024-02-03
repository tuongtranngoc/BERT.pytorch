from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from src.utils.logger import Logger
from src.utils.metrics import BatchMeter
from src.utils.data_utils import DataUtils
from src.utils.tensorboard import Tensorboard
from src.data.wikitext import WikiTextDataset
from src.models.pretrain.bert import BERTModel
from src.models.losses.loss_pretrain import PretrainLoss


logger = Logger.get_logger('TRAINING')

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
                                         num_workers=config['num_workers'])

    def create_model(self):
        self.model = BERTModel(vocab_size=len(self.dataset.vocab), 
                               num_hiddens=config['num_hiddens'],
                               ffn_num_hiddens=config['ffn_num_hiddens'],
                               num_heads=config['num_heads'],
                               num_blks=config['num_blks'],
                               dropout=config['dropout'],
                               max_len=config['max_len']).to(config['device'])
        self.loss_func = PretrainLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])


    def train(self):
        metrics = {
            'total_loss': BatchMeter(),
            'mlm_loss': BatchMeter(),
            'nsp_loss': BatchMeter()
        }
        self.model.train()
        for epoch in range(self.start_epoch, config['epochs']):
            for i, X in enumerate(self.dataset_loader):
                token_ids, segments, valid_lens, pred_positions, mlm_weights, mlm_labels, nsp_labels = DataUtils.to_device(X)
                __, mlm_preds, nsp_preds = self.model(token_ids, segments, valid_lens.reshape(-1), pred_positions)
                # import ipdb; ipdb.set_trace();
                total_loss, mlm_loss, nsp_loss = self.loss_func(mlm_preds, mlm_labels, mlm_weights, nsp_preds, nsp_labels, len(self.dataset.vocab))

                metrics['total_loss'].update(total_loss.item())
                metrics['nsp_loss'].update(nsp_loss.item())
                metrics['mlm_loss'].update(mlm_loss.item())

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch} - batch {i} - total_loss: {total_loss} - mlm_loss: {mlm_loss} - nsp_loss: {nsp_loss}", end='\r')

            logger.info(f"Epoch {epoch} - total_loss: {metrics['total_loss'].get_value(): .3f} - mlm_loss: {metrics['mlm_loss'].get_value(): .3f} - nsp_loss: {metrics['nsp_loss'].get_value(): .3f}")
            Tensorboard.add_scalars(tag='total_loss',
                                    step=epoch,
                                    loss=metrics['total_loss'].get_value())
            Tensorboard.add_scalars(tag='mlm_loss',
                                    step=epoch,
                                    loss=metrics['mlm_loss'].get_value())
            Tensorboard.add_scalars(tag='nsp_loss',
                                    step=epoch,
                                    loss=metrics['nsp_loss'].get_value())


    def save_ckpt(self):
        pass

    def resume_training(self):
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()