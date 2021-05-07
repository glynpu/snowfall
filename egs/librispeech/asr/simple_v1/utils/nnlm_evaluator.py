import os

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
from utils.dataset import DatasetOption, TextFileDataIterator, AuxlabelDataIterator, AbsLMDataIterator
from utils.load_espnet_model import build_model_from_file
from snowfall.models.lm_transformer import TransformerLM
from utils.numericalizer import get_numericalizer

import numpy as np
import torch

# TODO(Liyong Guo): types may need to be supported ['text', 'token', 'token_id']
_TYPES_SUPPORTED= ['text_file', 'auxlabel']


def _validate_input_type(input_type: Optional[str] = None):
    # A valid input_type must be assigned from the client
    assert input_type is not None
    assert input_type in _TYPES_SUPPORTED


def build_nnlmevaluator(args,
                        device='cpu',
                        input_type='text_file',
                        batch_size=32):
    _validate_input_type(input_type)

    model, train_args = build_model_from_file(config=args.lm_train_config,
                                              model_file=args.lm_model_file,
                                              model_type='espnet')
    model.to(device)

    numericalizer = get_numericalizer(tokenizer_type='spm',
                                      tokenizer_file=train_args.bpemodel,
                                      token_list=train_args.token_list)
    if input_type == 'text_file':
        dataset_option = DatasetOption(input_type=input_type,
                                       preprocessor=numericalizer)

        dataset = TextFileDataIterator(dataset_option)
    elif input_type == 'auxlabel':
        dataset_option = DatasetOption(input_type=input_type,
                                       preprocessor=numericalizer,
                                       words_txt='./data/lang_nosp/words.txt')
        dataset = AuxlabelDataIterator(dataset_option)

    evaluator = NNLMEvaluator(lm=model, dataset=dataset, device=device)
    return evaluator

@dataclass(frozen=True)
class PPLResult:
    nlls: List[float]
    ntokens: int
    nwords: int

    @property
    def total_nll(self):
        return sum(self.nlls)

    @property
    def token_ppl(self):
        return np.exp(self.total_nll / self.ntokens)

    @property
    def word_ppl(self):
        return np.exp(self.total_nll / self.nwords)

@dataclass
class NNLMEvaluator:
    lm: TransformerLM
    dataset: AbsLMDataIterator
    device: Union[str, torch.device]

    @torch.no_grad()
    def nll(self, text_source):
        nlls = []
        total_nll = 0.0
        total_ntokens = 0
        total_nwords = 0
        for xs_pad, target_pad, word_lens, token_lens in self.dataset(
                text_source):
            xs_pad = xs_pad.to(self.device)
            target_pad = target_pad.to(self.device)
            nll = self.lm.nll(xs_pad, target_pad, token_lens)
            nll = nll.detach().cpu().numpy().sum(1)
            nlls.extend(nll)
            total_ntokens += sum(token_lens)
            total_nwords += sum(word_lens)
        ppl_result = PPLResult(nlls=nlls,
                               ntokens=total_ntokens,
                               nwords=total_nwords)
        return ppl_result
