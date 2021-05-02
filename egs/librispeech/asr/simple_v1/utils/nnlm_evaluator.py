from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import k2
import os
from snowfall.models.transformer import pad_list

from utils.sentencepiece_tokenizer import SentencepiecesTokenizer
from utils.load_espnet_model import build_model_from_file

# TODO(Liyong Guo): types may need to be supported ['text', 'token', 'token_id']
_types_supported = ['text_file', 'auxlabel']


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


def auxlabel_to_word(word_seqs: k2.RaggedInt,
                     symbol_table: k2.SymbolTable) -> List[str]:
    word_ids = word_seqs.values()
    words = [symbol_table.get(word_idx.item()) for word_idx in word_ids]
    ragged_shape = word_seqs.row_splits(1)
    utts = []
    for idx, start_idx in enumerate(ragged_shape[:-1]):
        utts.append(' '.join(words[start_idx:ragged_shape[idx + 1]]))
    return utts


def get_tokenizer(tokenizer_type, tokeinzer_file):
    assert os.path.exists(tokenizer_file), f'{tokenizer_file} is not exists'
    if tokenizer_type == 'sentencepiece':
        return SentencepiecesTokenizer(tokenizer_file)
    else:
        raise raiseNotImplementedError(f'{tokenizer_type} is not supported')


def _validate_input_type(input_type: Optional[str] = None):
    # A valid input_type must be assigned from the client
    assert input_type is not None
    assert input_type in _types_supported


class DataIterator:

    def __init__(self,
                 input_type: Optional[str] = None,
                 batch_size: int = 32,
                 pad_value: int = 0,
                 preprocessor=None):
        super().__init__()
        _validate_input_type(input_type)
        self.pad_value = pad_value
        self.input_type = input_type
        self.batch_size = batch_size
        self.process_fn = self._get_process_fn(input_type)
        self.preprocessor = preprocessor
        if input_type == 'auxlabel':
            assert symbol_table is not None
            self.symbol_table = symbol_table

        self.__call__ = self._get_func('call')

        self.collate_fn = CollateFunc(self.pad_value)

    def _process_text_file(self, text_file):
        uttier = (utt.split().maxsplit(2)[1] for utt in open(text_file))
        for utt in uttier:
            utt = self.preprocessor(utt)
            yield utt

    def _process_auxlabel(self, word_seqs):
        # auxlabel --> text --> token_id
        texts = auxlabel_to_word(word_seqs)
        utts = [self.tokennizer.text2tokens(text) for text in texts]
        return utt

    def _get_func(self, prefix):
        fn_name = f'_{prefix}_{self.input_type}'
        fn = getattr(self, fn_name)

    def _get_process_fn(self, input_type: str):
        _validate_input_type(input_type)
        fn = self._get_func('process')
        return fn

    # each _call* functions is a generater
    def _call_auxlabel(self, word_seqs):
        utts = self._process_auxlabel(word_seqs)
        self._call_token_id(utts)

    def _call_token_id(self, utts):
        self.len = len(utts)
        tensor_list = []
        for idx, utt in enumerate(utts):
            utt = self.process_fn(utt)
            batch.append(torch_from_numpy(utt))
            if len(tensor_list) == self.batch_size or idx == self.len - 1:
                # batch = pad_list(tensor_list, pad_value)

                yield batch
                batch = []

    def _call_text_file(self, text_file):
        # assert os.path.exist(text_file)
        with open(text_file, 'r') as f:
            token_ids_list = []
            token_lens = []
            word_lens = []
            for text in f:
                text = text.strip().split(maxsplit=1)[1]
                word_lens.append(len(text.split()) + 1)  # +1 for <eos>

                token_ids = self.preprocessor(text)
                token_ids_list.append(token_ids)
                token_lens.append(len(token_ids) - 1)  # -1 to remove <sos>

                if len(token_ids_list) == self.batch_size:
                    xs_pad, ys_pad = self.collate_fn(token_ids_list)
                    yield xs_pad, ys_pad, word_lens, token_lens
                    token_ids_list = []
                    word_lens = []
                    token_lens = []

            if len(token_ids_list) != 0:
                xs_pad, ys_pad = self.collate_fn(token_ids_list)
                yield xs_pad, ys_pad, word_lens, token_lens


class NNLMEvaluator:

    def __init__(self,
                 input_type: Optional[str] = None,
                 batch_size: int = 32,
                 lm_config_file: Optional[Path] = None,
                 lm_model_file: Optional[Path] = None,
                 preprocessor=None):
        super().__init__()
        _validate_input_type(input_type)
        self.input_type = input_type
        self.preprocessor = preprocessor
        self.dataset = DataIterator(input_type=input_type,
                                    preprocessor=preprocessor)
        self.lm, _ = build_model_from_file(lm_config_file, lm_model_file)

    def _get_func(self, prefix):
        fn_name = f'_{prefix}_{self.input_type}'
        fn = getattr(self, fn_name)

    def _score_auxlabel(self, word_seqs):
        retval = []
        for batch in self.dataset(word_seqs):
            print(len(batch))
            score = self.lm.nll(batch)
            retval.append(score)
        return retval

    def _score_text(self, text_file):
        nlls = []
        lengths = []
        total_nll = 0.0
        total_ntokens = 0
        total_nwords = 0
        for xs_pad, target_pad, word_lens, token_lens in self.dataset._call_text_file(
                text_file):
            nll = self.lm.nll(xs_pad, target_pad, token_lens)
            nll = nll.detach().cpu().numpy().sum(1)
            total_nll += nll.sum()
            total_ntokens += sum(token_lens)
            total_nwords += sum(word_lens)
        token_ppl = np.exp(total_nll / total_ntokens)
        word_ppl = np.exp(total_nll / total_nwords)
        print(token_ppl, total_ntokens, word_ppl, total_nwords)
        return token_ppl

    def score(self, utts, input_type: Optional[str] = None):
        _validate_input_type(input_type)

        assert input_type == self.input_type, \
        'Unmatch input_type: utts is {input_type} while evaluator is for {self.input_type}'

        score_fn = self._get_func('score')

        return score_fn(utts)
