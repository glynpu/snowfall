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
                custom_attn=MultiHeadedAttention(attention_heads,attention_dim,attention_dropout_rate),
                nhead=attention_heads,
                dim_feedforward=dim_feedforward,
                normalize_before=True,
                dropout=attention_dropout_rate,
            )

        self.encoders = nn.TransformerEncoder(encoder_layer, num_blocks, None)

        if self.normalize_before:
            self.after_norm = nn.LayerNorm(attention_dim)

    def forward(self, xs, masks):
        xs = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks


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
        self.encoder = LMEncoder(
            attention_dim=att_unit,
            attention_heads=head,
            num_blocks=layer,
            dim_feedforward=unit
        )
        self.decoder = nn.Linear(att_unit, vocab_size)

    def forward(self, input: torch.Tensor,
                ) -> Tuple[torch.Tensor, None]:
        x = self.embed(input)
        x = self.input_embed(x)
        mask = (generate_square_subsequent_mask(
            input.shape[-1]) == 0).unsqueeze(0).to(x.device)
        h, _ = self.encoder(x, mask)
        y = self.decoder(h)
        return y

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
        y = self.forward(x)

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

