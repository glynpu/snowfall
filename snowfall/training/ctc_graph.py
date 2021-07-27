# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from typing import Iterable
from typing import List

import torch
import k2

from snowfall.lexicon import Lexicon


class CtcTrainingGraphCompiler(object):

    def __init__(self,
                 lexicon: Lexicon,
                 device: torch.device,
                 oov: str = '<UNK>'):
        '''
        Args:
          lexicon:
            It is built from `data/lang/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary word. When a word in the transcript
            does not exist in the lexicon, it is replaced with `oov`.
        '''
        L_inv = lexicon.L_inv.to(device)
        assert L_inv.requires_grad is False

        assert oov in lexicon.words

        self.L_inv = k2.arc_sort(L_inv)
        self.oov = oov
        self.words = lexicon.words
        self.bpe = lexicon.bpe
        self.L_bpe = lexicon.L_bpe

        max_token_id = max(lexicon.phone_symbols())
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)

        self.ctc_topo = ctc_topo.to(device)
        self.device = device

    def compile(self, texts: Iterable[str]) -> k2.Fsa:
        if self.bpe:
          return self.compile_bpe(texts)
        word_ids = []
        for text in texts:
          tokens = (token if token in self.words else self.oov
                    for token in text.split(' '))
          word_id = [self.words[token] for token in tokens]
          word_ids.append(word_id)
        word_fsa = k2.linear_fsa(word_ids, self.device)

        word_fsa_with_self_loops = k2.add_epsilon_self_loops(word_fsa)

        fsa = k2.intersect(
            self.L_inv, word_fsa_with_self_loops,
            treat_epsilons_specially=False
        )
        # fsa has word ID as labels and token ID as aux_labels, so
        # we need to invert it
        transcript_fsa = fsa.invert_()

        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            transcript_fsa
        )
        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)
        decoding_graph = k2.compose(
            self.ctc_topo, fsa_with_self_loops, treat_epsilons_specially=False
        )
        assert decoding_graph.requires_grad is False
        return decoding_graph

    def compile_bpe(self, texts: Iterable[str]) -> k2.Fsa:
        token_ids = []
        for text in texts:
          words = (word if word in self.words else self.oov
                   for word in text.split(' '))
          token_ids.append(
            [token_id for word in words for token_id in self.L_bpe[word]])
        return k2.ctc_graph(token_ids, device=self.device)
