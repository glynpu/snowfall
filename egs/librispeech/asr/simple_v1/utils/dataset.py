from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional

from utils.preprocessor import PreProcessor

import k2
import numpy as np
import torch


def auxlabel_to_word(word_seqs: k2.RaggedInt,
                     symbol_table: k2.SymbolTable) -> List[str]:
    word_ids = word_seqs.values()
    words = [symbol_table.get(word_idx.item()) for word_idx in word_ids]
    ragged_shape = word_seqs.row_splits(1)
    utts = []
    for idx, start_idx in enumerate(ragged_shape[:-1]):
        utts.append(' '.join(words[start_idx:ragged_shape[idx + 1]]))
    return utts


class CollateFunc(object):
    '''Collate function for LMDataset
    '''

    def __init__(self, pad_index=None):
        # pad_index should be identical to ignore_index of torch.nn.NLLLoss
        # and padding_idx in torch.nn.Embedding
        self.pad_index = pad_index

    def __call__(self, batch: List[List[int]]):
        '''batch contains token_id.
           batch can be viewd as a ragged 2-d array, with a row represents a token_id.
           token_id reprents a tokenized text, whose format is:
           <bos_id> token_id token_id token_id *** <eos_id>
        '''
        # data_pad: [batch_size, seq_len]
        # each seq_len always different
        data_pad = pad_sequence(
            [torch.from_numpy(np.array(x)).long() for x in batch], True,
            self.pad_index)
        data_pad = data_pad.contiguous()
        xs_pad = data_pad[:, :-1].contiguous()
        ys_pad = data_pad[:, 1:].contiguous()
        # xs_pad/ys_pad: [batch_size, max_seq_len]
        # max_seq_len == max(len([<sos> token token token ... token])
        #             == max(len([token token token ... token <eos>])
        return xs_pad, ys_pad

@dataclass
class DatasetOption:
    input_type: Optional[str] = 'text_file'
    batch_size: int = 32
    pad_value: int = 0
    preprocessor: Optional[PreProcessor] = None
    words_txt: Optional[Path] = None

    @property
    def symbol_table(self):
        return None if self.words_txt is None else k2.SymbolTable.from_file(self.words_txt)


class AbsLMDataIterator(ABC):

    def __init__(self, dataset_option):
        self.input_type = dataset_option.input_type
        self.batch_size = dataset_option.batch_size
        self.pad_value = dataset_option.pad_value
        self.preprocessor = dataset_option.preprocessor
        if self.input_type == 'auxlabel':
            assert dataset_option.symbol_table is not None
        self.symbol_table = dataset_option.symbol_table

        self.collate_fn = CollateFunc(self.pad_value)

    def _reset_container(self):
        self.token_ids_list = []
        self.token_lens = []
        self.word_lens = []

    # text_source may text_file/word_seqs
    def __call__(self, text_source):
        self._reset_container()
        for text in self._text_generator(text_source):
            # text = text.strip().split(maxsplit=1)[1]
            self.word_lens.append(len(text.split()) + 1)  # +1 for <eos>

            token_ids = self.preprocessor(text)
            self.token_ids_list.append(token_ids)
            self.token_lens.append(len(token_ids) - 1)  # -1 to remove <sos>

            if len(self.token_ids_list) == self.batch_size:
                xs_pad, ys_pad = self.collate_fn(self.token_ids_list)

                yield xs_pad, ys_pad, self.word_lens, self.token_lens
                self._reset_container()

        if len(self.token_ids_list) != 0:
            xs_pad, ys_pad = self.collate_fn(self.token_ids_list)
            yield xs_pad, ys_pad, self.word_lens, self.token_lens
            self._reset_container()


class TextFileDataIterator(AbsLMDataIterator):

    def __init__(self, dataset_option):
        super().__init__(dataset_option)

    def _text_generator(self, text_file):
        with open(text_file, 'r') as f:
            for text in f:
                text = text.strip().split(maxsplit=1)[1]
                yield text


class AuxlabelDataIterator(AbsLMDataIterator):

    def __init__(self, dataset_option):
        super().__init__(dataset_option)

    def _text_generator(self, word_seqs):
        # word_seqs --> text
        texts = auxlabel_to_word(word_seqs, self.symbol_table)
        for text in texts:
            yield text
