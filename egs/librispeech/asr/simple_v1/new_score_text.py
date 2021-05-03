import argparse
import torch
import logging
import numpy as np
from utils.nnlm_evaluator import build_nnlmevaluator


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

    evaluator = build_nnlmevaluator(args, device='cpu', input_type='text_file')

    file_name = 'dump/raw/test_clean/text'
    with torch.no_grad():
        ppl_result = evaluator.nll(file_name)
    print(ppl_result.token_ppl)


if __name__ == '__main__':
    score_text()
