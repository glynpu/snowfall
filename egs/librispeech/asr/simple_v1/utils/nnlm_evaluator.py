from typing import List
from typing import Optional

import k2
from utils.sentencepiece_tokenizer import SentencepiecesTokenizer
from utils.load_espnet_model import build_model_from_file

# TODO(Liyong Guo): types may need to be supported ['text', 'token', 'token_id']
_types_supported = ['auxlabel']


def auxlabel_to_word(word_seqs: k2.RaggedInt,
                     symbol_table: k2.SymbolTable) -> List[str]:
    word_ids = word_seqs.values()
    words = [symbol_table.get(word_idx.item()) for word_idx in word_ids]
    ragged_shape = word_seqs.row_splits(1)
    utts = []
    for idx, start_idx in enumerate(ragged_shape[:-1]):
        utts.append(' '.join(words[start_idx:ragged_shape[idx + 1]]))
    return utts


def get_tokenizer(tokenizer_type, tokeinzer_file):
    assert os.path.exists(tokenizer_file), f'{tokenizer_file} is not exists'
    if tokenizer_type == 'sentencepiece':
        return SentencepiecesTokenizer(tokenizer_file)
    else:
        raise raiseNotImplementedError(f'{tokenizer_type} is not supported')


def _validate_input_type(input_type: Optional[str] = None):
    # An valid input_type must be assigned from the client
    assert input_type is not None
    assert input_type in _types_supported


class DataIterator:

    def __init__(self,
                 input_type: Optional[str] = None,
                 batch_size: int = 32,
                 symbol_table=None,
                 tokenizer_type=None,
                 tokenizer_file=None):
        _validate_input_type(input_type)
        self.batch_size = batch_size
        self.process_fn = self._get_process_fn(input_type)
        if input_type == 'auxlabel':
            assert symbol_table is not None
            self.symbol_table = symbol_table
        call_fn_name = f'_call_{input_type}'
        self.__call__ = getattr(self, call_fn)
        self.tokenizer = get_tokenizer(tokenizer_type, tokenizer_file)

    def _process_auxlabel(self, word_seqs):
        # auxlabel --> text --> token_id
        texts = auxlabel_to_word(word_seqs)
        utts = [self.tokennizer.text2tokens(text) for text in texts]
        return utt

    def _get_process_fn(self, input_type: str):
        _validate_input_type(input_type)

        fn_name = f'_process_{input_type}'
        fn = getattr(self, fn_name)
        return fn

    def _pad_batch(self, batch):
        return path

    # each _call* functions is a generater
    def _call_auxlabel(self, word_seqs):
        utts = self._process_auxlabel(word_seqs)
        self._call_token_id(utts)

    def _call_token_id(self, utts):
        self.len = len(utts)
        batch = []
        for idx, utt in enumerate(utts):
            utt = self.process_fn(utt)
            batch.append(utt)
            if len(batch) == self.batch_size or idx == self.len - 1:
                self._pad_batch(batch)
                yield batch
                batch = []


class NNLMEvaluator:

    def __init__(self,
                 input_type: Optional[str] = None,
                 batch_size: int = 32,
                 lm_config_file: Optional[Path] = None,
                 lm_model_file: Optional[Paht] = None):
        super().__init__()
        _validate_input_type(input_type)
        self.input_type = input_type
        self.dataset = DataIterator(input_type)
        self.lm = build_model_from_file(lm_config_file, lm_model_file)

    def _score_auxlabel(self, word_seqs):
        retval = []
        print(utts)
        for batch in self.dataset(word_seqs):
            print(len(batch))
            score = self.lm(batch)
            # retval.append(score)
        return retval

    def score(self, utts, input_type: Optional[str] = None):
        _validate_input_type(input_type)

        # An valid input_type must be assigned from the client
        assert input_type == self.input_type, \
        'Unmatch input_type: utts is {input_type} while evaluator is for {self.input_type}'

        fn_name = f'_score_{input_type}'
        fn = getattr(self, fn_name)
        return fn(utts)
