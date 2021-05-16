#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import re
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List, Dict

from local_snowfall.asr import build_model
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed




import k2
from snowfall.training.ctc_graph import build_ctc_topo
from local_snowfall.data import wav_loader
from local_snowfall.common import _load_espnet_model_config
from utils.nnlm_evaluator import build_nnlmevaluator

from snowfall.decoding.lm_rescore import decode_with_lm_rescoring
from snowfall.common import get_texts

def rename_state_dict(rename_patterns: List[Tuple[str, str]],
                      state_dict: Dict[str, torch.Tensor]):
    # Rename state dict to load espent model
    if rename_patterns is not None:
        for old_pattern, new_pattern in rename_patterns:
            # old_keys = [k for k in state_dict if k.find(old_pattern) != -1]
            old_keys = [k for k in state_dict if re.match(old_pattern, k) is not None]
            for k in old_keys:
                v = state_dict.pop(k)
                new_k = re.sub(old_pattern, new_pattern, k)
                state_dict[new_k] = v

def combine_qkv(state_dict: Dict[str, torch.Tensor], num_encoder_layers=11):
    for layer in range(num_encoder_layers + 1):
        q_w = state_dict[f'encoder.encoders.{layer}.self_attn.linear_q.weight']
        k_w = state_dict[f'encoder.encoders.{layer}.self_attn.linear_k.weight']
        v_w = state_dict[f'encoder.encoders.{layer}.self_attn.linear_v.weight']
        q_b = state_dict[f'encoder.encoders.{layer}.self_attn.linear_q.bias']
        k_b = state_dict[f'encoder.encoders.{layer}.self_attn.linear_k.bias']
        v_b = state_dict[f'encoder.encoders.{layer}.self_attn.linear_v.bias']

        for param_type in ['weight', 'bias']:
            for layer_type in ['q', 'k', 'v']:
                key_to_remove = f'encoder.encoders.{layer}.self_attn.linear_{layer_type}.{param_type}'
                state_dict.pop(key_to_remove)

        in_proj_weight = torch.cat([q_w, k_w, v_w], dim=0)
        in_proj_bias= torch.cat([q_b, k_b, v_b], dim=0)
        key_weight = f'encoder.encoders.{layer}.self_attn.in_proj.weight'
        state_dict[key_weight] = in_proj_weight
        key_bias = f'encoder.encoders.{layer}.self_attn.in_proj.bias'
        state_dict[key_bias] = in_proj_bias


class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        batch_size: int = 1,
        lang_dir: str = './',
    ):
        assert check_argument_types()

        # 1. Build ASR model
        asr_train_args = _load_espnet_model_config(asr_train_config)



        model = build_model(asr_train_args)
        asr_model = model.to(device)
        state_dict = torch.load(asr_model_file, map_location=device)
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('decoder')}
        rename_patterns = [
            ('encoder.embed.out.0.weight', 'encoder.embed.out.weight'),
            ('encoder.embed.out.0.bias', 'encoder.embed.out.bias'),
            (r'(encoder.encoders.)(\d+)(.self_attn.)linear_out([\s\S*])', r'\1\2\3out_proj\4'),
            (r'(encoder.encoders.)(\d+)', r'\1layers.\2'),

            # encoder.encoders.layers.0.feed_forward.w_1.weight 
            # encoder.encoders.layers.10.feed_forward.0.bias
            (r'(encoder.encoders.layers.)(\d+)(.feed_forward.)(w_1)', r'\1\2.feed_forward.0'),
            (r'(encoder.encoders.layers.)(\d+)(.feed_forward.)(w_2)', r'\1\2.feed_forward.3'),
            (r'(encoder.encoders.layers.)(\d+)(.feed_forward_macaron.)(w_1)', r'\1\2.feed_forward_macaron.0'),
            (r'(encoder.encoders.layers.)(\d+)(.feed_forward_macaron.)(w_2)', r'\1\2.feed_forward_macaron.3'),
            (r'(encoder.embed.)([\s\S*])', r'encoder.encoder_embed.\2'),
            (r'(encoder.encoders.)([\s\S*])', r'encoder.encoder.\2'),
            (r'(ctc.ctc_lo.)([\s\S*])', r'encoder.encoder_output_layer.1.\2'),
        ]
        combine_qkv(state_dict, num_encoder_layers=11)



        rename_state_dict(rename_patterns=rename_patterns, state_dict=state_dict)
        asr_model.load_state_dict(state_dict, strict=False)

        dtype: str = "float32"
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        # token_list = asr_model.token_list
        token_list = asr_train_args.token_list

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        token_type = asr_train_args.token_type
        bpemodel = asr_train_args.bpemodel

        tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        converter = TokenIDConverter(token_list=token_list)

        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        # self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
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
        self.evaluator = build_nnlmevaluator(args, device=device, input_type='auxlabel', converter=converter, tokenizer=tokenizer)

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
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

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
        self.evaluator.converter = self.converter
        if True:
            best_paths = decode_with_lm_rescoring(
                lattices,
                G,
                self.evaluator,
                num_paths=num_paths,
                use_whole_lattice=use_whole_lattice)

        # hyps = get_texts(best_paths, indices)
        token_int = best_paths.aux_labels[best_paths.aux_labels != 0]
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        # best_paths = k2.shortest_path(lattices, use_double_scores=True)

        # token_int = best_paths[0].aux_labels.cpu().numpy()
        token_int = list(filter(lambda x: x != 0, token_int))[:-1]
        token = self.converter.ids2tokens(token_int)
        # token = list(filter(lambda x: x != '<sox/eos>', token))
        if token[-1] == '<sos/eos>':
            token = token[:-1]

        text = self.tokenizer.tokens2text(token)

        # import pdb; pdb.set_trace()
        results = []
        hyp = None
        results.append((text, token, token_int, hyp))
        return results
        # return None

def inference(
    batch_size: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    asr_train_config: str,
    asr_model_file: str,
    token_type: Optional[str],
    bpemodel: Optional[str],
):
    assert check_argument_types()
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")


    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text = Speech2Text(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        token_type=token_type,
        bpemodel=bpemodel,
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

def get_parser():
    # parser = config_argparse.ArgumentParser(
    parser = argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )

    return parser


def main(cmd=None):
    parser = get_parser()
    print(f'cmd {cmd}')
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    inference(**kwargs)


if __name__ == "__main__":
    main()
