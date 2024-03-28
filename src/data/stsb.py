from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

import multiprocessing
import datasets
import json
import os

from . import *


class STSBDataset(nn.Module):
    def __init__(self, config, data_type='train') -> None:
        super(STSBDataset, self).__init__()
        self.cfg = config
        if not os.path.join(self.cfg['finetune_data_path']):
            dataset = datasets.load_dataset('snli')
            dataset.save_to_disk(self.cfg['finetune_data_path'])
        dataset = datasets.load_from_disk(self.cfg['finetune_data_path'])[data_type]
        
        # Seperate dataset
        lables = np.array(dataset['similarity_score'])
        pos_idxs = np.where(lables > 0)[0]
        sentence1 = np.array(dataset['sentence1'])[pos_idxs].tolist()
        sentence2 = np.array(dataset['sentence2'])[pos_idxs].tolist()
        
        # Vocab definition
        self.vocab = Vocab()
        self.vocab.token_to_idx = json.load(open(self.cfg['vocab_path']))
        self.vocab.idx_to_token = list(self.vocab.token_to_idx.keys())
        all_tokens_pairs = [[sent1_tokens, sent2_tokens] for sent1_tokens, sent2_tokens in zip \
                (self.tokenize([sent.lower() for sent in sentence1]), \
                 self.tokenize([sent.lower() for sent in sentence2]))]
        self.labels = torch.tensor(lables, dtype=torch.long)
        self.max_len = self.cfg['max_len']
        self.all_token_ids, self.all_segments, self.valid_lens = self._multi_preprocess(all_tokens_pairs)

    def tokenize(self, lines, token='word'):
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            print('Error: Unknown token type:' + token)
    
    def get_tokens_and_segments(self, tokens_a, tokens_b=None):
        tokens = ['<cls>'] + tokens_a + ['<sep>']
        segments = [0] * (len(tokens_a) + 2)
        if tokens_b is not None:
            tokens += tokens_b + ['<sep>']
            segments += [1] * (len(tokens_b) + 1)
        return tokens, segments
    
    def _multi_preprocess(self, all_tokens_pairs):
        pool = multiprocessing.Pool(4)
        out = pool.map(self._single_preprocess, all_tokens_pairs)
        all_token_ids = torch.tensor([token_ids for token_ids, __, __ in out], dtype=torch.long)
        all_segments = torch.tensor([segments for __, segments, __ in out], dtype=torch.long)
        valid_lens = torch.tensor([valid_len for __, __, valid_len in out], dtype=torch.long)

        return all_token_ids, all_segments, valid_lens

    def _single_preprocess(self, all_tokens_pairs):
        sent1_tokens, sent2_tokens = all_tokens_pairs
        self._truncate_pair_of_tokens(sent1_tokens, sent2_tokens)
        tokens, segments = self.get_tokens_and_segments(sent1_tokens, sent2_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len
    
    def _truncate_pair_of_tokens(self, sent1_tokens, sent2_tokens):
        while len(sent1_tokens) + len(sent2_tokens) > self.max_len - 3:
            if len(sent1_tokens) > len(sent2_tokens):
                sent1_tokens.pop()
            else:
                sent2_tokens.pop()
    
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx]), self.labels[idx]
    
    def __len__(self): return len(self.all_token_ids)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.configs.load_config import configuration
    cfg = configuration('finetune_stsb')
    dataset = STSBDataset(config=cfg,data_type='train')
    data_loader = DataLoader(dataset, batch_size=28, shuffle=True)
    import ipdb; ipdb.set_trace()
