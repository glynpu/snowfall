import os

from pathlib import Path
from typing import List, Optional
from utils.dataset import DatasetOption, TextFileDataIterator, AuxlabelDataIterator
from utils.load_espnet_model import build_model_from_file
from utils.preprocessor import build_preprocessor

import numpy as np
import torch

# TODO(Liyong Guo): types may need to be supported ['text', 'token', 'token_id']
_types_supported = ['text_file', 'auxlabel']


def _validate_input_type(input_type: Optional[str] = None):
    # A valid input_type must be assigned from the client
    assert input_type is not None
    assert input_type in _types_supported


def build_nnlmevaluator(args,
                        device='cpu',
                        input_type='text_file',
                        batch_size=32):
    _validate_input_type(input_type)

    model, train_args = build_model_from_file(config=args.lm_train_config,
                                              model_file=args.lm_model_file,
                                              model_type='espnet')
    model.to(device)

    preprocessor = build_preprocessor(tokenizer_type='spm',
                                      tokenizer_file=train_args.bpemodel,
                                      token_list=train_args.token_list)
    if input_type == 'text_file':
        dataset_option = DatasetOption(input_type=input_type,
                                       preprocessor=preprocessor)

        dataset = TextFileDataIterator(dataset_option)
    elif input_type == 'auxlabel':
        dataset_option = DatasetOption(input_type=input_type,
                                       preprocessor=preprocessor,
                                       symbol_table='./data/lang_nosp/words.txt')
        dataset = AuxlabelDataIterator(dataset_option)

    evaluator = NNLMEvaluator(lm=model, dataset=dataset, device=device)
    return evaluator


class PPLResult:

    def __init__(self, nlls=List[float], ntokens=None, nwords=None):

        self.nlls = nlls
        self.ntokens = ntokens
        self.nwords = nwords
        self.total_nll = sum(nlls)
        self.token_ppl = np.exp(self.total_nll / self.ntokens)
        self.word_ppl = np.exp(self.total_nll / self.nwords)


class NNLMEvaluator:

    def __init__(self, lm=None, dataset=None, device='cpu'):
        super().__init__()
        self.lm = lm
        self.dataset = dataset
        self.device = device

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
            # token_lens = token_lens
            nll = self.lm.nll(xs_pad, target_pad, token_lens)
            nll = nll.detach().cpu().numpy().sum(1)
            nlls.extend(nll)
            total_ntokens += sum(token_lens)
            total_nwords += sum(word_lens)
        ppl_result = PPLResult(nlls=nlls,
                               ntokens=total_ntokens,
                               nwords=total_nwords)
        return ppl_result
