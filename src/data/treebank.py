from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

import multiprocessing
from tqdm import tqdm
import datasets
import json
import glob
import os

from . import *


class TreeBankDataset(nn.Module):
    def __init__(self, config, data_type='train') -> None:
        super(TreeBankDataset, self).__init__()
        self.cfg = config
        with open(os.path.join(self.cfg["finetune_data_path"], "tagset-map.json")) as f_map:
            mapping_labels = json.load(f_map)
            mapping_labels = {
                k: i 
                for i, k in enumerate(mapping_labels.keys())
            }
        with open(os.path.join(self.cfg["finetune_data_path"], data_type + '.json')) as f:
            dataset = json.load(f)

        all_tokens = []
        labels = []
        for (sent, tags) in tqdm(dataset):
            sents, _labels = [], []
            for token, tag in zip(self.tokenize(sent.lower()),tags):
                if tag in mapping_labels:
                    sents.append(token)
                    _labels.append(mapping_labels[tag])
            all_tokens.extend(sents)
            labels.extend(_labels)

        # Vocab definition
        self.vocab = Vocab()
        self.vocab.token_to_idx = json.load(open(self.cfg['vocab_path']))
        self.vocab.idx_to_token = list(self.vocab.token_to_idx.keys())
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_len = self.cfg['max_len']
        self.all_token_ids, self.all_segments, self.valid_lens = self._multi_preprocess(all_tokens)

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

    def _single_preprocess(self, tokens):
        self._truncate_pair_of_tokens(tokens)
        tokens, segments = self.get_tokens_and_segments(tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len
    
    def _truncate_pair_of_tokens(self, tokens):
        while len(tokens) > self.max_len - 3:
            tokens.pop()
    
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx]), self.labels[idx]
    
    def __len__(self): return len(self.all_token_ids)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.configs.load_config import configuration
    cfg = configuration('finetune_treebank')
    dataset = TreeBankDataset(config=cfg,data_type='train')
    data_loader = DataLoader(dataset, batch_size=28, shuffle=True)
    import ipdb; ipdb.set_trace()
