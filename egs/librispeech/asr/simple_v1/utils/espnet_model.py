import argparse
import re
from typing import Dict
from typing import Tuple
from typing import Union
from typing import List
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
from typeguard import check_argument_types
from typeguard import check_return_type

from utils.nets_utils import make_pad_mask
from utils.transformer_lm import TransformerLM


class ESPnetLanguageModel(torch.nn.Module):

    def __init__(self, lm, vocab_size: int, ignore_id: int = 0):
        assert check_argument_types()
        super().__init__()
        self.lm = lm
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        self.ignore_id = ignore_id

    def nll(self, text: torch.Tensor,
            text_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = text.size(0)
        # For data parallel
        text = text[:, :text_lengths.max()]

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x = F.pad(text, [1, 0], "constant", self.eos)
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.sos
        x_lengths = text_lengths + 1

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        # 3. Calc negative log likelihood
        # nll: (BxL,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]),
                              t.view(-1),
                              reduction="none")
        # nll: (BxL,) -> (BxL,)
        nll.masked_fill_(make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0)
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)
        return nll, x_lengths


def build_model(args: argparse.Namespace) -> ESPnetLanguageModel:
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
    lm = TransformerLM(vocab_size=vocab_size, **args.lm_conf)

    # 2. Build ESPnetModel
    # Assume the last-id is sos_and_eos
    model = ESPnetLanguageModel(lm=lm,
                                vocab_size=vocab_size,
                                **args.model_conf)

    # FIXME(kamo): Should be done in model?
    # 3. Initialize
    # if args.init is not None:
    #     initialize(model, args.init)

    assert check_return_type(model)
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
                # new_k = k.replace(old_pattern, new_pattern)
                state_dict[new_k] = v


    # old_keys = [k for k in state_dict if k.startswith(old_prefix)]
    # if len(old_keys) > 0:
    #     logging.warning(f"Rename: {old_prefix} -> {new_prefix}")
    # for k in old_keys:
    #     v = state_dict.pop(k)
    #     new_k = k.replace(old_prefix, new_prefix)
    #     state_dict[new_k] = v


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
        if device == "cuda":
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            device = f"cuda:{torch.cuda.current_device()}"
        state_dict = torch.load(model_file, map_location=device)
        rename_patterns = [('.feed_forward.w_1', '.linear1'),
                           ('.feed_forward.w_2', '.linear2'),
                           ('.encoder.embed', '.input_embed'),
                           (r'(lm.encoder.encoders.)(\d+)', r'\1layers.\2' ),
                           ]
        rename_state_dict(rename_patterns=rename_patterns,
                          state_dict=state_dict)
        # import pdb
        # pdb.set_trace()
        mk = [key for key in model.state_dict().keys()]
        model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load(model_file, map_location=device))

    return model, args
