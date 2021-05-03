from typing import List, Iterable
from utils.sentencepiece_tokenizer import SentencepiecesTokenizer


class PreProcessor(object):

    def _assign_special_symbols(self):
        if '<sos/eos>' in self.token2id:
            # <sos> and <eos> share same index for model download from espnet model zoo
            self.sos = self.token2id['<sos/eos>']
            self.eos = self.token2id['<sos/eos>']
        elif '<sos>' in self.token2id and '<eos>' in self.token2id:
            self.sos = self.token2id['<sos>']
            self.eos = self.token2id['<eos>']
        else:
            raise RuntimeError(f'No index for <sos> or <eos>')
        if '<unk>' in self.token2id:
            self.unk_id = self.token2id['<unk>']
        else:
            raise RuntimeError(
                f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
            )


class SpmPreProcessor(PreProcessor):

    def __init__(self,
                 tokenizer_type,
                 tokenizer_file,
                 token_list,
                 unk_symbol='<unk>'):
        assert tokenizer_type == 'spm'
        self.tokenizer = SentencepiecesTokenizer(tokenizer_file)

        self.token2id = {}
        for idx, token in enumerate(token_list):
            if token in self.token2id:
                raise RuntimeError(f'Symbol "{token}" is duplicated')
            self.token2id[token] = idx

        self._assign_special_symbols()

    def _tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(token, self.unk_id) for token in tokens]

    def __call__(self, text: str):
        tokens = self.tokenizer.text2tokens(text)
        token_ids = self._tokens2ids(tokens)
        token_ids.insert(0, self.sos)
        token_ids.append(self.eos)
        return token_ids


def build_preprocessor(
    tokenizer_type,
    tokenizer_file,
    token_list,
):
    if tokenizer_type == 'spm':
        preprocessor = SpmPreProcessor(tokenizer_type=tokenizer_type,
                                       tokenizer_file=tokenizer_file,
                                       token_list=token_list)
    elif tokenizer_type == 'huggingface':
        raise NotImplementedError(f'{token_type} is to be supported')
    else:
        raise NotImplementedError(f'Unsupported tokenizer type {token_type}')

    return preprocessor
