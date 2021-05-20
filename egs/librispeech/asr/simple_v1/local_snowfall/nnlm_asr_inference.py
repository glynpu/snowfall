#!/usr/bin/env python3
import argparse
import logging
import os
import random
import re
import sys

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List, Dict

import k2
import numpy as np
import torch

from kaldialign import edit_distance
from lhotse import load_manifest
from lhotse.dataset import K2SpeechRecognitionDataset, SingleCutSampler
from lhotse.dataset.input_strategies import AudioSamples

from local_snowfall.asr import build_model

from lhotse import load_manifest

from snowfall.decoding.lm_rescore import decode_with_lm_rescoring
from snowfall.training.ctc_graph import build_ctc_topo
from utils.nnlm_evaluator import build_nnlmevaluator


def decode(dataloader: torch.utils.data.DataLoader, model: None,
                   device: Union[str, torch.device], ctc_topo: None, evaluator=None, numericalizer=None):
    tot_num_cuts = len(dataloader.dataset.cuts)
    num_cuts = 0
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
        speech = speech.unsqueeze(0)
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
        if batch_idx % 20 == 0:
            # 4.35% [2 / 46, 0 ins, 1 del, 1 sub ]
            return results
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
    group.add_argument('--lm_train_config', type=str, required=True)
    group.add_argument('--lm_model_file', type=str, required=True)

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

    asr_model, numericalizer = build_model(asr_train_config, asr_model_file, device)

    asr_model.eval()

    # phone_ids_with_blank = [i for i in range(len(token_list))]
    # import pdb; pdb.set_trace()
    phone_ids_with_blank = [i for i in range(len(numericalizer.token_list))]

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



    # args = argparse.Namespace(**d_args)
    evaluator = build_nnlmevaluator(args.lm_train_config, args.lm_model_file, device=device, input_type='auxlabel', numericalizer=numericalizer)
    feature_dir = Path('exp/data')

    # test_sets = ['test-clean', 'test-other']
    test_sets = ['test-clean']
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
