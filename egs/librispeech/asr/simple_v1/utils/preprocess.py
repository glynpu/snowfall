import argparse
import numpy as np
from typing import Callable
from typing import Dict
from typing import Optional
from utils.data import CommonPreprocessor
from typeguard import check_argument_types
from typeguard import check_return_type

def build_preprocess_fn(
    args: argparse.Namespace, train: bool
) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
    assert check_argument_types()
    if args.use_preprocessor:
        retval = CommonPreprocessor(
            train=train,
            token_type=args.token_type,
            token_list=args.token_list,
            bpemodel=args.bpemodel,
            text_cleaner=args.cleaner,
            g2p_type=args.g2p,
            non_linguistic_symbols=args.non_linguistic_symbols,
        )
    else:
        retval = None

    assert check_return_type(retval)
    return retval
