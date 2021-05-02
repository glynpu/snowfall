import argparse
import torch
import logging
import numpy as np
from utils.load_espnet_model import build_model_from_file
from utils.preprocessor import build_preprocessor
from utils.nnlm_evaluator import NNLMEvaluator


def get_parser():
    parser = argparse.ArgumentParser(
        description="Calc perplexity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train_config",
        type=str,
        default='exp/lm_train_lm_transformer2_en_bpe5000/config.yaml')
    parser.add_argument(
        "--model_file",
        type=str,
        default='exp/lm_train_lm_transformer2_en_bpe5000/valid.loss.ave.pth')
    return parser


def score_text():
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level='INFO')

    device = "cpu"
    model, train_args = build_model_from_file(args.train_config,
                                              args.model_file, device)
    numel_params = [param.numel() for param in model.parameters()]
    total_params = sum(numel_params)
    assert total_params == 53711240

    preprocessor = build_preprocessor(tokenizer_type='spm',
                                      tokenizer_file=train_args.bpemodel,
                                      token_list=train_args.token_list)
    evaluator = NNLMEvaluator(input_type='text_file',
                              lm_config_file=args.train_config,
                              lm_model_file=args.model_file,
                              preprocessor=preprocessor)

    file_name = 'dump/raw/test_clean/text'
    with torch.no_grad():
        token_ppl = evaluator._score_text(file_name)
    print(token_ppl)


if __name__ == '__main__':
    score_text()
