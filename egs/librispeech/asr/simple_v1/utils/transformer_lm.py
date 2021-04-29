from typing import Any
from typing import List
from typing import Tuple

import torch
import torch.nn as nn

from utils.encoder import Encoder
from snowfall.models.transformer import generate_square_subsequent_mask


class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        pos_enc: str = None,
        embed_unit: int = 128,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        layer: int = 4,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_unit)
        self.encoder = Encoder(
            idim=embed_unit,
            attention_dim=att_unit,
            attention_heads=head,
            linear_units=unit,
            num_blocks=layer,
            dropout_rate=dropout_rate,
            input_layer="linear",
        )
        self.decoder = nn.Linear(att_unit, vocab_size)

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1),
                            device=ys_mask.device).unsqueeze(0)
        m = (generate_square_subsequent_mask(
            ys_in_pad.shape[-1]) == 0).unsqueeze(0).to(ys_mask.device)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, input: torch.Tensor,
                hidden: None) -> Tuple[torch.Tensor, None]:
        """Compute LM loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(input)
        mask = (generate_square_subsequent_mask(
            input.shape[-1]) == 0).unsqueeze(0).to(x.device)
        # import pdb; pdb.set_trace()
        h, _ = self.encoder(x, mask)
        y = self.decoder(h)
        return y, None
