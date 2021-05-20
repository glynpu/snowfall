import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

from local_snowfall.default import DefaultFrontend
from local_snowfall.default import GlobalMVN
from snowfall.models.conformer import Conformer
from local_snowfall.common import rename_state_dict, combine_qkv

def build_model(args: argparse.Namespace, asr_model_file, device):
    token_list = list(args.token_list)
    vocab_size = len(token_list)

    # import pdb; pdb.set_trace()
    # {'fs': '16k', 'hop_length': 256, 'n_fft': 512}
    frontend = DefaultFrontend(**args.frontend_conf)
    input_size = frontend.output_size()
    normalize = GlobalMVN(**args.normalize_conf)

    encoder = Conformer(num_features=80,
            num_classes=5000,
            subsampling_factor=4,
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            num_encoder_layers=12,
            cnn_module_kernel=31,
            num_decoder_layers=0)

    model = ESPnetASRModel(
        frontend=frontend,
        normalize=normalize,
        encoder=encoder,
    )

    state_dict = torch.load(asr_model_file, map_location=device)

    state_dict = {k:v for k,v in state_dict.items() if not k.startswith('decoder')}
    rename_patterns = [
        ('encoder.embed.out.0.weight', 'encoder.embed.out.weight'),
        ('encoder.embed.out.0.bias', 'encoder.embed.out.bias'),
        (r'(encoder.encoders.)(\d+)(.self_attn.)linear_out([\s\S*])', r'\1\2\3out_proj\4'),
        (r'(encoder.encoders.)(\d+)', r'\1layers.\2'),
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

    model.load_state_dict(state_dict, strict=False)
    return model

class ESPnetASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        frontend: None,
        normalize: None,
        encoder: None,
    ):

        super().__init__()
        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        feats, feats_lengths = self.frontend(speech, speech_lengths)

        feats, feats_lengths = self.normalize(feats, feats_lengths)
        # feats = speech
        # feats_lengths = speech_lengths
        supervision = {}

        supervision['sequence_idx'] = torch.tensor([[0]])
        supervision['start_frame'] = torch.tensor([[0]])
        supervision['num_frames'] = torch.tensor([[feats_lengths]])
        feats = feats.permute(0, 2, 1)

        nnet_output, _, _ = self.encoder(feats, supervision)
        nnet_output = nnet_output.permute(2, 0, 1)
        return nnet_output, supervision
