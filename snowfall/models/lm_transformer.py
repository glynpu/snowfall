from typing import Any
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from snowfall.models.attention import MultiHeadedAttention
from snowfall.models.transformer import generate_square_subsequent_mask
from snowfall.models.transformer import make_pad_mask
from snowfall.models.transformer import TransformerEncoderLayer


class LMEncoder(nn.Module):

    def __init__(
        self,
        attention_dim=512,
        attention_heads=8,
        attention_dropout_rate=0.0,
        num_blocks=16,
        dim_feedforward=2048,
        normalize_before=True,
    ):
        super().__init__()

        self.normalize_before = normalize_before

        encoder_layer = TransformerEncoderLayer(
            d_model=attention_dim,
            custom_attn=MultiHeadedAttention(attention_heads, attention_dim,
                                             attention_dropout_rate),
            nhead=attention_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=attention_dropout_rate,
        )

        self.encoders = nn.TransformerEncoder(encoder_layer, num_blocks, None)

        if self.normalize_before:
            self.after_norm = nn.LayerNorm(attention_dim)

    def forward(self, xs, masks):
        # xs: [batch_size, max_seq_len]
        # masks: [1, max_seq_len, max_seq_len], looks like
        #   tensor([[[ True, False, False,  ..., False, False, False],
        #            [ True,  True, False,  ..., False, False, False],
        #            [ True,  True,  True,  ..., False, False, False],
        #            ...,
        #            [ True,  True,  True,  ...,  True, False, False],
        #            [ True,  True,  True,  ...,  True,  True, False],
        #            [ True,  True,  True,  ...,  True,  True,  True]]])

        xs = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        pos_enc: str = None,
        embed_unit: int = 128,
        att_unit: int = 512,
        head: int = 8,
        unit: int = 2048,
        layer: int = 16,
        dropout_rate: float = 0.0,
        ignore_id: int = 0,
    ):
        super().__init__()

        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.ignore_id = ignore_id

        self.embed = nn.Embedding(vocab_size, embed_unit)
        self.input_embed = nn.Sequential(
            nn.Linear(embed_unit, att_unit),
            nn.LayerNorm(att_unit),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.encoder = LMEncoder(attention_dim=att_unit,
                                 attention_heads=head,
                                 num_blocks=layer,
                                 dim_feedforward=unit)
        self.decoder = nn.Linear(att_unit, vocab_size)

    def forward(
        self,
        input: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        # input: [batch_size, max_seq_len]
        x = self.embed(input)
        x = self.input_embed(x)
        mask = (generate_square_subsequent_mask(
            input.shape[-1]) == 0).unsqueeze(0).to(x.device)
        h = self.encoder(x, mask)
        y = self.decoder(h)
        # y: [batch_size, max_seq_len, vocab_size]
        return y

    def nll(self, xs_pad, target_pad, token_lens):
        # xs_pad/ys_pad: [batch_size, max_seq_len]
        # max_seq_len == max(len([<sos> token token token ... token])
        #             == max(len([token token token ... token <eos>])
        y = self.forward(xs_pad)
        # nll: (batch_size * max_seq_len,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]),
                              target_pad.view(-1),
                              reduction="none")
        # assign padded postion with 0.0
        nll.masked_fill_(
            make_pad_mask(token_lens).to(nll.device).view(-1), 0.0)

        # nll: (batch_size * max_seq_len,) -> (batch_size, max_seq_len)
        nll = nll.view(xs_pad.size(0), -1)
        return nll
