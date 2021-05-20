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

from lhotse import load_manifest
from lhotse.dataset import K2SpeechRecognitionDataset, SingleCutSampler
from lhotse.dataset.input_strategies import AudioSamples
from pathlib import Path

from local_snowfall.asr import build_model
from utils.numericalizer import SpmNumericalizer

import k2
from snowfall.training.ctc_graph import build_ctc_topo
from lhotse import load_manifest
from local_snowfall.common import _load_espnet_model_config

from utils.nnlm_evaluator import build_nnlmevaluator

from snowfall.decoding.lm_rescore import decode_with_lm_rescoring
from kaldialign import edit_distance

def decode(dataloader: torch.utils.data.DataLoader, model: None,
                   device: Union[str, torch.device], ctc_topo: None, evaluator=None, numericalizer=None):
    tot_num_cuts = len(dataloader.dataset.cuts)
    num_cuts = 0
    dtype: str = "float32"
    results = []
    for batch_idx, batch in enumerate(dataloader):
        assert isinstance(batch, dict), type(batch)
        speech = batch['inputs'].squeeze()
        ref = batch['supervisions']['text']
        lengths = torch.tensor([speech.shape[0]])
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, dtype))
        # lenghts: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        speech = speech.to(torch.device(device))
        lengths = lengths.to(torch.device(device))
        batch = {"speech": speech, "speech_lengths": lengths}

        nnet_output, _ = model.encode(**batch)
        nnet_output = nnet_output.detach()

        blank_bias = -1.0
        nnet_output[:, :, 0] += blank_bias

        old_supervision_segments = torch.tensor([[0, 0, nnet_output.shape[1]]], dtype=torch.int32)

        supervision_segments = torch.stack(
            (torch.tensor([0]),
             torch.tensor([0]),
             torch.tensor([nnet_output.shape[1]])), 1).to(torch.int32)
        supervision_segments = torch.clamp(supervision_segments, min=0)
        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]
        
        with torch.no_grad():
            dense_fsa_vec = k2.DenseFsaVec(nnet_output, old_supervision_segments)

            # 0 is for blank

            output_beam_size = 8
            lattices = k2.intersect_dense_pruned(ctc_topo, dense_fsa_vec, 20.0, output_beam_size, 30, 10000)

        G = None
        num_paths=10
        use_whole_lattice=False
        best_paths = decode_with_lm_rescoring(
            lattices,
            G,
            evaluator,
            num_paths=num_paths,
            use_whole_lattice=use_whole_lattice)

        token_int = best_paths.aux_labels[best_paths.aux_labels != 0]

        token_int = list(filter(lambda x: x != 0, token_int))[:-1]
        token = numericalizer.ids2tokens(token_int)
        if token[-1] == '<sos/eos>':
            token = token[:-1]

        text = numericalizer.tokens2text(token)
        for i in range(len(ref)):
            hyp_words = text.split(' ')
            ref_words = ref[i].split(' ')
            results.append((ref_words, hyp_words))
        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, cuts processed until now is {}/{} ({:.6f}%)'.format(
                    batch_idx, num_cuts, tot_num_cuts,
                    float(num_cuts) / tot_num_cuts * 100))
        num_cuts += 1
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
    logging.basicConfig(level=logging.DEBUG)
    print(f'cmd {cmd}')
    args = parser.parse_args(cmd)
    asr_train_config = args.asr_train_config
    asr_model_file = args.asr_model_file
    seed = args.seed

    device = "cuda"

    # 1. Set random-seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    asr_train_args = _load_espnet_model_config(asr_train_config)
    asr_model = build_model(asr_train_args, asr_model_file, device)
    dtype: str = "float32"
    asr_model.to(dtype=getattr(torch, dtype)).eval()

    token_list = asr_train_args.token_list
    token_type = asr_train_args.token_type
    bpemodel = asr_train_args.bpemodel
    numericalizer = SpmNumericalizer(tokenizer_type='spm',
            tokenizer_file=asr_train_args.bpemodel,
            token_list=token_list,
            unk_symbol='<unk>')

    phone_ids_with_blank = [i for i in range(5000)]

    lang_dir = './'
    ctc_path = Path(lang_dir) / 'ctc_topo.pt'

    if not os.path.exists(ctc_path):
        ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))
        torch.save(ctc_topo.as_dict(), ctc_path)
    else:
        print("Loading pre-compiled ctc topo fst")
        d_ctc_topo = torch.load(ctc_path)
        ctc_topo = k2.Fsa.from_dict(d_ctc_topo)
        ctc_topo = ctc_topo.to(device)



    d_args ={'lm_train_config':'exp/lm_train_lm_transformer2_en_bpe5000/config.yaml', 'lm_model_file': 'exp/lm_train_lm_transformer2_en_bpe5000/valid.loss.ave.pth'}
    args = argparse.Namespace(**d_args)
    evaluator = build_nnlmevaluator(args, device=device, input_type='auxlabel', numericalizer=numericalizer)
    # new_loader = wav_loader('dump/raw/test_clean/wav.scp')
    feature_dir = Path('exp/data')

    test_sets = ['test-clean', 'test-other']
    for test_set in test_sets:
        cuts_test = load_manifest(feature_dir / f'cuts_{test_set}.json.gz')
        sampler = SingleCutSampler(cuts_test, max_cuts=1)

        test = K2SpeechRecognitionDataset(cuts_test, input_strategy=AudioSamples())
        test_dl = torch.utils.data.DataLoader(test, batch_size=None, sampler=sampler)
        results = decode(dataloader=test_dl,
                          model=asr_model,
                          device=device,
                          ctc_topo=ctc_topo,
                          evaluator=evaluator,
                          numericalizer=numericalizer)

        # import pdb; pdb.set_trace()
        dists = [edit_distance(r, h) for r, h in results]
        errors = {
            key: sum(dist[key] for dist in dists)
            for key in ['sub', 'ins', 'del', 'total']
        }
        total_words = sum(len(ref) for ref, _ in results)
        # Print Kaldi-like message:
        # %WER 2.62 [ 1380 / 52576, 176 ins, 106 del, 1098 sub ]
        logging.info(
            f'[{test_set}] %WER {errors["total"] / total_words:.2%} '
            f'[{errors["total"]} / {total_words}, {errors["ins"]} ins, {errors["del"]} del, {errors["sub"]} sub ]'
        )




if __name__ == "__main__":
    main()
