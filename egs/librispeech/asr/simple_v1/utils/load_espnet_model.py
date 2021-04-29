import argparse
import re
from typing import Dict
from typing import Tuple
from typing import Union
from typing import List
import yaml
from pathlib import Path

import torch
from typeguard import check_argument_types
from typeguard import check_return_type

# from utils.transformer_lm import TransformerLM
from snowfall.models.lm_transformer import TransformerLM

def build_model(args: argparse.Namespace):
    assert check_argument_types()
    if isinstance(args.token_list, str):
        with open(args.token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]

        # "args" is saved as it is in a yaml file by BaseTask.main().
        # Overwriting token_list to keep it as "portable".
        args.token_list = token_list.copy()
    elif isinstance(args.token_list, (tuple, list)):
        token_list = args.token_list.copy()
    else:
        raise RuntimeError("token_list must be str or dict")

    vocab_size = len(token_list)
    # logging.info(f"Vocabulary size: {vocab_size }")

    # 1. Build LM model
    # lm_class = lm_choices.get_class(args.lm)
    model = TransformerLM(vocab_size=vocab_size, **args.lm_conf)

    return model


def rename_state_dict(rename_patterns: List[Tuple[str, str]],
                      state_dict: Dict[str, torch.Tensor]):
    """Replace keys of old prefix with new prefix in state dict."""
    # need this list not to break the dict iterator
    if rename_patterns is not None:
        for old_pattern, new_pattern in rename_patterns:
            old_keys = [k for k in state_dict if k.find(old_pattern)]
            for k in old_keys:
                v = state_dict.pop(k)
                new_k = re.sub(old_pattern, new_pattern, k)
                state_dict[new_k] = v


def build_model_from_file(
    config_file: Union[Path, str],
    model_file: Union[Path, str] = None,
    device: str = "cpu",
):
    """This method is used for inference or fine-tuning.

    Args:
        config_file: The yaml file saved when training.
        model_file: The model file saved when training.
        device:

    """
    assert check_argument_types()
    config_file = Path(config_file)

    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args = argparse.Namespace(**args)
    model = build_model(args)
    model.to(device)
    if model_file is not None:
        state_dict = torch.load(model_file, map_location=device)

        rename_patterns = [('.feed_forward.w_1', '.linear1'),
                           ('.feed_forward.w_2', '.linear2'),
                           ('.encoder.embed', '.input_embed'),
                           (r'(lm.encoder.encoders.)(\d+)', r'\1layers.\2' ),
                           (r'(lm.)([\s\S]*)', r'\2' ),
                           ]
        rename_state_dict(rename_patterns=rename_patterns,
                          state_dict=state_dict)
        mk = [key for key in model.state_dict().keys()]
        model.load_state_dict(state_dict)

    return model, args
