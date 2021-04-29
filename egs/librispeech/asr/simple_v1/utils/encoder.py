# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
import torch
import torch.nn as nn

from utils.attention import MultiHeadedAttention
from snowfall.models.transformer import TransformerEncoderLayer
# from snowfall.models.transformer import TransformerEncoder
from utils.repeat import repeat

class Encoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        conv_wshare (int): The number of kernel of convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        conv_kernel_length (Union[int, str]): Kernel size str of convolution
            (e.g. 71_71_71_71_71_71). Only used in self_attention_layer_type
            == "lightconv*" or "dynamiconv*".
        conv_usebias (bool): Whether to use bias in convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        selfattention_layer_type (str): Encoder attention layer type.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length="11",
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        # pos_enc_class=None,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        selfattention_layer_type="selfattn",
        padding_idx=-1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        self.normalize_before = normalize_before
        encoder_selfattn_layer = MultiHeadedAttention
        encoder_selfattn_layer_args = [
            (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        ] * num_blocks

        encoder_layer = TransformerEncoderLayer(
                d_model=attention_dim,
                custom_attn=encoder_selfattn_layer(attention_heads,attention_dim,attention_dropout_rate),
                nhead=8,
                dim_feedforward=2048,
                normalize_before=True,
                dropout=0.0,
            )

        self.encoders = nn.TransformerEncoder(encoder_layer, num_blocks, None)
        # self.encoders = repeat(
        #     num_blocks,
        #     (encoder_layer for i in range lnum),
        #     )
        # self.encoders = repeat(
        #     num_blocks,
        #     lambda lnum: TransformerEncoderLayer(
        #         d_model=attention_dim,
        #         custom_attn=encoder_selfattn_layer(*encoder_selfattn_layer_args[lnum]),
        #         nhead=8,
        #         dim_feedforward=2048,
        #         normalize_before=True,
        #         dropout=0.0,
        #     ),
        # )

        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        xs = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
