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

    def _get_next_sentence(self, sentence, next_sentence, paragraphs):
        if random.random() < 0.5:
            is_next =True
        else:
            next_sentence = random.choice(random.choices(paragraphs))
        return sentence, next_sentence, is_next
    
    def _get_nsp_data_from_paragraph(self, pragraph, pragraphs, vocab, max_len):
        """Generating the Next Sentence Prediction
            max_len: maximum length of a BERT input sequence during pretraining
            pragraph: A list of sentences, where each sentence is a list of tokens
        """
        nps_data_from_paragraph = []
        for i in range(len(pragraph)-1):
            tokens_a, tokens_b, is_next = self._get_next_sentence(pragraph[i], pragraph[i+1], pragraphs)
            # <cls> token_a <sep> token_b <sep>
            if len(tokens_a) + len(tokens_b) + 3 > max_len:
                continue
            tokens, segments = self.get_tokens_and_segments(tokens_a, tokens_b)
            nps_data_from_paragraph.append((tokens, segments, is_next))
        return nps_data_from_paragraph

    def get_tokens_and_segments(self, token_a, token_b=None):
        tokens = ['<cls>'] + token_a + ['<sep>']
        segments = [0] * (len(token_a) + 2)
        if token_b is not None:
            tokens += token_b + ['<sep>']
            segments += [1] * (len(token_b) + 1)
        return tokens, segments
    
    def _replace_maskedlm_tokens(self, tokens, cand_pred_positions, num_mlm_preds, vocab):
        """Replace mask for tokens
            tokens: A list of tokens representing a BERT input sentence
            cand_pred_positions: A list of token indices of BERT input sequence
            num_mlm_preds: the number of predictions            
        """
        mlm_input_tokens = [token for token in tokens]
        pred_position_and_labels = []
        random.shuffle(cand_pred_positions)
        
        for mlm_pred_position in cand_pred_positions:
            if len(pred_position_and_labels) >= num_mlm_preds:
                break   
            masked_token = None
            if random.random() < 0.8:
                masked_token = '<mask>'
            else:
                if random.random() < 0.5:
                    masked_token = tokens[mlm_pred_position]
                else:
                    masked_token = random.choice(vocab.idx_to_token)
            mlm_input_tokens[mlm_pred_position] = masked_token
            pred_position_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
            
        return mlm_input_tokens, pred_position_and_labels
    
    def _get_maskedlm_data_from_tokens(self, tokens, vocab):
        cand_pred_positions = []
        for i, token in enumerate(tokens):
            if token in ['<cls>', '<sep>']:
                continue
            cand_pred_positions.append(i)
        num_mlm_preds = max(1, round(len(tokens) * 0.15))
        mlm_input_tokens, pred_positions_and_labels = self._replace_maskedlm_tokens(
            tokens=tokens, cand_pred_positions=cand_pred_positions, num_mlm_preds=num_mlm_preds, vocab=vocab
        )
        pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
        pred_positions = [v[0] for v in pred_positions_and_labels]
        mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

        return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
    
    def _pad_bert_inputs(self, examples, max_len, vocab):
        """Append the specical <pad> tokens to inputs
            examples: the outputs from the helper functions nsp and mlm
        """
        max_num_mlm_preds = round(max_len * 0.15)
        all_token_ids, all_segments, valid_lens = [], [], []
        all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
        nsp_labels = []

        for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
            all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
            all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
            valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
            all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))

            all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
            all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
            nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

        return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)
    
    

