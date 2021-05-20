from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union
from pathlib import Path

import numpy as np

from torchtext.data.functional import load_sp_model

class NumericalizerMixin(ABC):

    def _assign_special_symbols(self):
        # <sos> and <eos> share same index for model download from espnet model zoo
        assert '<sos/eos>' in self.token2idx \
                or ('<sos>' in self.token2idx and '<eos>' in self.tokenid)
        assert '<unk>' in self.token2idx
        self.sos_idx = self.token2idx['<sos/eos>'] if '<sos/eos>' in self.token2idx else self.token2idx['<sos>']
        self.eos_idx = self.token2idx['<sos/eos>'] if '<sos/eos>' in self.token2idx else self.token2idx['<eos>']
        self.unk_idx = self.token2idx['<unk>']


@dataclass
class SpmNumericalizer(NumericalizerMixin):
    def __init__(self,
                 tokenizer_type,
                 tokenizer_file,
                 token_list,
                 unk_symbol='<unk>'):
        assert tokenizer_type == 'spm'
        self.tokenizer_file = tokenizer_file
        self.token_list = token_list
        self._token2idx = None
        self._tokenizer = None
        self._assign_special_symbols()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_sp_model(self.tokenizer_file)
        return self._tokenizer

    def text2tokens(self, line: str) -> List[str]:
        return self.tokenizer.EncodeAsPieces(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.tokenizer.DecodePieces(list(tokens))

    @property
    def token2idx(self):
        if self._token2idx is None:
            self._token2idx = {}
            for idx, token in enumerate(self.token_list):
                if token in self._token2idx:
                    raise RuntimeError(f'Symbol "{token}" is duplicated')
                self._token2idx[token] = idx

        return self._token2idx

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]


    def __call__(self, text: str) -> List[int]:
        tokens = self.text2tokens(text)
        token_idxs = [self.sos_idx] + [self.token2idx.get(token, self.unk_idx) for token in tokens] + [self.eos_idx]
        return token_idxs


def get_numericalizer(
    tokenizer_type,
    tokenizer_file,
    token_list,
):
    if tokenizer_type == 'spm':
        numericalizer = SpmNumericalizer(tokenizer_type=tokenizer_type,
                                         tokenizer_file=tokenizer_file,
                                         token_list=token_list)
    elif tokenizer_type == 'huggingface':
        raise NotImplementedError(f'{token_type} is to be supported')
    else:
        raise ValueError(f'Unsupported tokenizer type {token_type}')

    return numericalizer