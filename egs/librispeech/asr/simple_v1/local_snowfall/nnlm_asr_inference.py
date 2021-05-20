#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple, Union, List, Dict

import numpy as np
import random
import re
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from local_snowfall.asr import build_model
from utils.numericalizer import SpmNumericalizer

import k2
from snowfall.training.ctc_graph import build_ctc_topo
from local_snowfall.data import wav_loader
from local_snowfall.common import _load_espnet_model_config

from utils.nnlm_evaluator import build_nnlmevaluator

from snowfall.decoding.lm_rescore import decode_with_lm_rescoring

from snowfall.common import get_texts



class Speech2Text:
    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        device: str = "cpu",
    ):
        assert check_argument_types()
        lang_dir = './'
        # 1. Build ASR model
        asr_train_args = _load_espnet_model_config(asr_train_config)

        model = build_model(asr_train_args, asr_model_file, device)
        asr_model = model.to(device)

        dtype: str = "float32"
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        # token_list = asr_model.token_list
        token_list = asr_train_args.token_list

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        token_type = asr_train_args.token_type
        bpemodel = asr_train_args.bpemodel


        self.numericalizer = SpmNumericalizer(tokenizer_type='spm',
                tokenizer_file=asr_train_args.bpemodel,
                token_list=token_list,
                unk_symbol='<unk>')

        # logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.device = device
        self.dtype = dtype
        phone_ids_with_blank = [i for i in range(5000)]
        ctc_path = Path(lang_dir) / 'ctc_topo.pt'
        if not os.path.exists(ctc_path):
            ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))
            torch.save(ctc_topo.as_dict(), ctc_path)
        else:
            print("Loading pre-compiled ctc topo fst")
            d_ctc_topo = torch.load(ctc_path)
            ctc_topo = k2.Fsa.from_dict(d_ctc_topo)
        self.ctc_topo = ctc_topo.to(device)

        import argparse
        d_args ={'lm_train_config':'exp/lm_train_lm_transformer2_en_bpe5000/config.yaml', 'lm_model_file': 'exp/lm_train_lm_transformer2_en_bpe5000/valid.loss.ave.pth'}
        args = argparse.Namespace(**d_args)
        self.evaluator = build_nnlmevaluator(args, device=device, input_type='auxlabel', numericalizer=self.numericalizer)

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
        ): # -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lenghts: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

        speech = speech.to(torch.device(self.device))
        lengths = lengths.to(torch.device(self.device))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        # batch = to_device(batch, device=self.device)

        # import pdb; pdb.set_trace()
        nnet_output, _ = self.asr_model.encode(**batch)

        blank_bias = -1.0
        nnet_output[:, :, 0] += blank_bias

        # assert nnet_output.shape == torch.Size([1, 162, 5000])
        # sequence_index, start_frame, duration
        # import pdb; pdb.set_trace()
        old_supervision_segments = torch.tensor([[0, 0, nnet_output.shape[1]]], dtype=torch.int32)

        supervision_segments = torch.stack(
            (torch.tensor([0]),
             torch.tensor([0]),
             torch.tensor([nnet_output.shape[1]])), 1).to(torch.int32)
        supervision_segments = torch.clamp(supervision_segments, min=0)
        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]
        # indices = torch.argsort(supervision_segments[:, 2], descending=True)
        # assert (supervision_segments.numpy() == [[0, 0, 162]]).all()

        dense_fsa_vec = k2.DenseFsaVec(nnet_output, old_supervision_segments)

        # 0 is for blank

        output_beam_size = 8
        lattices = k2.intersect_dense_pruned(self.ctc_topo, dense_fsa_vec, 20.0, output_beam_size, 30, 10000)

        G = None
        num_paths=10
        use_whole_lattice=False
        # self.evaluator.converter = self.converter
        if True:
            best_paths = decode_with_lm_rescoring(
                lattices,
                G,
                self.evaluator,
                num_paths=num_paths,
                use_whole_lattice=use_whole_lattice)

        token_int = best_paths.aux_labels[best_paths.aux_labels != 0]

        token_int = list(filter(lambda x: x != 0, token_int))[:-1]
        token = self.numericalizer.ids2tokens(token_int)
        if token[-1] == '<sos/eos>':
            token = token[:-1]

        text = self.numericalizer.tokens2text(token)

        results = []
        hyp = None
        results.append((text, token, token_int, hyp))
        return results

def get_parser():
    parser = argparse.ArgumentParser(
        description="ASR Decoding with model from espnet model zoo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--seed", type=int, default=2021, help="Random seed")

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)

    return parser

def main(cmd=None):
    parser = get_parser()
    print(f'cmd {cmd}')
    args = parser.parse_args(cmd)
    asr_train_config = args.asr_train_config
    asr_model_file = args.asr_model_file
    seed = args.seed

    device = "cuda"

    # 1. Set random-seed
    # set_all_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # 2. Build speech2text
    import pdb; pdb.set_trace()
    speech2text = Speech2Text(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        # token_type=token_type,
        device=device,
    )



    new_loader = wav_loader('dump/raw/test_clean/wav.scp')

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with open('exp/hyp_2021-05-11.text', 'w') as writer:
        # for keys, batch in loader:
        for keys, batch in new_loader:
            # import pdb; pdb.set_trace()
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            # try:
            results = speech2text(**batch)
            for n, (text, token, token_int, hyp) in enumerate(results):
                print(text)
                key = keys[n]
                writer.write(f'{key} {text}\n')





if __name__ == "__main__":
    main()
