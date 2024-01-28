from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import datasets
import transformers

import torch
import torch.nn as nn

import random

from . import *

class BookCorpusDataset(nn.Module):
    def __init__(self) -> None:
        super(BookCorpusDataset, self).__init__()
        self.dataset = datasets.load_dataset(cfg['Train']['bookcorpus'])
        self.batch_size = cfg['Train']['batch_size']
        self.word_dict = {
            '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            '[MASK]': 3
        }

    def preprocess_data(self):
        batch = []
        positive = negative = 0
        while positive != self.batch_size or negative != self.batch_size:
            token_a_index = random.randrange(len(self.dataset['train'][0]))
            token_b_index = random.randrange(len(self.dataset['train'][0]))

            tokens_a = self.word_dict[token_a_index]
            tokens_b = self.word_dict[token_b_index]

            input_ids = [self.word_dict['CLS']] + tokens_a + [self.word_dict['SEP']] + tokens_b + [self.word_dict['SEP']]
            segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
            
            # Create MaskLM
            # 15% of tokens in one sentence
            max_pred = len(input_ids)
            n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))
            cand_make_pos = [i for i, token in enumerate(input_ids) if token != self.word_dict['[CLS]'] and token != self.word_dict['[SEP]']]
            random.shuffle(cand_make_pos)

            masked_tokens, masked_pos = [], []
            for pos in cand_make_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos])

                if random.uniform() < 0.8:
                    input_ids[pos] = self.word_dict['[MASK]']
                elif random.uniform() < 0.5:
                    index = random.randint(0, self.vocab_size-1)
                    input_ids[pos] = self.word_dict[self.number_dict[index]]

            # Zero padding
            if self.max_pred > n_pred:
                n_pad = self.max_pred - n_pred
                masked_pos.extend([0] * n_pad)
                masked_tokens.extend([0] * n_pad)

            if token_a_index + 1 == token_a_index and positive < self.batch_size/2:
                # isNext
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
                positive += 1
            elif token_a_index + 1 != token_b_index and negative < self.batch_size/2:
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
                negative += 1

            return batch



    def create_word_dict(self, word_list):
        for i, w in enumerate(word_list):
            word_list[w] = i+4
            self.number_dict = {i: w for i, w in enumerate(word_list)}
            self.vocab_size = len(word_list)

        

