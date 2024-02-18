from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import multiprocessing
import datasets
import json
import os

from . import *


class SNLIDataset(nn.Module):
    def __init__(self, config, data_type='train') -> None:
        self.self.cfg = config
        super(SNLIDataset, self).__init__()
        if not os.path.join(self.cfg['finetune_data_path']):
            dataset = datasets.load_from_disk('snli')
            dataset.save_to_disk(self.cfg['finetune_data_path'])
        dataset = datasets.load_from_disk(self.cfg['finetune_data_path'])[data_type]
        self.vocab = Vocab()
        self.vocab.token_to_idx = json.load(open(self.cfg['vocab_path']))
        self.vocab.idx_to_token = list(self.vocab.token_to_idx.keys())
        all_premise_hypothesis_tokens = [[p_tokens, h_tokens] for p_tokens, h_tokens in zip \
                (self.tokenize([sent.lower() for sent in dataset['premise']]), self.tokenize([sent.lower() for sent in dataset['hypothesis']]))]
        self.labels = torch.tensor(dataset['label'])
        self.max_len = self.cfg['max_len']
        self.all_token_ids, self.all_segments, self.valid_lens = self._multi_preprocess(all_premise_hypothesis_tokens)

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
    
    def _multi_preprocess(self, all_premise_hypthesis_tokens):
        pool = multiprocessing.Pool(4)
        out = pool.map(self._single_preprocess, all_premise_hypthesis_tokens)
        all_token_ids = torch.tensor([token_ids for token_ids, __, __ in out], dtype=torch.long)
        all_segments = torch.tensor([segments for __, segments, __ in out], dtype=torch.long)
        valid_lens = torch.tensor([valid_len for __, __, valid_len in out])

        return all_token_ids, all_segments, valid_lens

    def _single_preprocess(self, premise_hypthesis_tokens):
        p_tokens, h_tokens = premise_hypthesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = self.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], self.labels[idx]
    
    def __len__(self): return len(self.all_token_ids)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = SNLIDataset()
    data_loader = DataLoader(dataset, batch_size=28, shuffle=True)
    import ipdb; ipdb.set_trace()
    