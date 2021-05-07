import argparse
import re
from typing import Dict, List, Tuple, Union
from pathlib import Path

import torch
import yaml

from snowfall.models.lm_transformer import TransformerLM


def rename_state_dict(rename_patterns: List[Tuple[str, str]],
                      state_dict: Dict[str, torch.Tensor]):
    # Rename state dict to load espent model
    if rename_patterns is not None:
        for old_pattern, new_pattern in rename_patterns:
            old_keys = [k for k in state_dict if k.find(old_pattern)]
            for k in old_keys:
                v = state_dict.pop(k)
                new_k = re.sub(old_pattern, new_pattern, k)
                state_dict[new_k] = v


def load_espnet_model(
    config: Dict,
    model_file: Union[Path, str],
):
    """This method is used to load LM model downloaded from espnet model zoo.

    Args:
        config_file: The yaml file saved when training.
        model_file: The model file saved when training.

    """
    model = TransformerLM(**config)

    assert model_file is not None, f"model file doesn't exist"
    state_dict = torch.load(model_file)

    rename_patterns = [
        ('.feed_forward.w_1', '.linear1'),
        ('.feed_forward.w_2', '.linear2'),
        ('.encoder.embed', '.input_embed'),
        (r'(lm.encoder.encoders.)(\d+)', r'\1layers.\2'),
        (r'(lm.)([\s\S]*)', r'\2'),
    ]

    rename_state_dict(rename_patterns=rename_patterns, state_dict=state_dict)
    model.load_state_dict(state_dict)

    return model


def build_model_from_file(config=None, model_file=None, model_type='espnet'):
    if model_type == 'espnet':
        return load_espnet_model(config, model_file)
    elif model_type == 'snowfall':
        raise NotImplementedError(f'Snowfall model to be suppported')
    else:
        raise ValueError(f'Unsupported model type {model_type}')
