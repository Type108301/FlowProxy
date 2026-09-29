"""UNetEx: DeepCFD 多解码器 U-Net。

改编自 Ribeiro et al., arXiv:2004.08826，GitHub mdribeiro/DeepCFD (MIT)。
每个输出场（Ux / Uy / p）使用独立 decoder，共享 encoder。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn_wrap


def _conv(in_ch, out_ch, kernel_size, weight_norm, batch_norm, activation, conv_cls):
    assert kernel_size % 2 == 1
    conv = conv_cls(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
    if weight_norm:
        conv = wn_wrap(conv)
    layers: list[nn.Module] = [conv]
    if activation is not None:
        layers.append(activation())
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    return nn.Sequential(*layers)


def _encoder_block(in_ch, out_ch, kernel_size, wn, bn, activation, n_layers):
    blocks = []
    for i in range(n_layers):
        src, dst = (in_ch, out_ch) if i == 0 else (out_ch, out_ch)
        blocks.append(_conv(src, dst, kernel_size, wn, bn, activation, nn.Conv2d))
    return nn.Sequential(*blocks)


def _decoder_block(in_ch, out_ch, kernel_size, wn, bn, activation, n_layers, final_layer):
    blocks = []
    for i in range(n_layers):
        src = in_ch * 2 if i == 0 else in_ch
        dst = out_ch if i == n_layers - 1 else in_ch
        use_bn = False if (final_layer and i == n_layers - 1) else bn
        act = None if (final_layer and i == n_layers - 1) else activation
        blocks.append(_conv(src, dst, kernel_size, wn, use_bn, act, nn.ConvTranspose2d))
    return nn.Sequential(*blocks)


class UNetEx(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        filters: list[int] | None = None,
        layers: int = 2,
        weight_norm: bool = False,
        batch_norm: bool = False,
        activation=nn.ReLU,
    ):
        super().__init__()
        filters = filters or [8, 16, 32, 32]
        enc = []
        for i, f in enumerate(filters):
            src = in_channels if i == 0 else filters[i - 1]
            enc.append(_encoder_block(src, f, kernel_size, weight_norm, batch_norm, activation, layers))
        self.encoder = nn.Sequential(*enc)
        decoders = []
        for _ in range(out_channels):
            dec = []
            for i, f in enumerate(filters):
                if i == 0:
                    dec.append(
                        _decoder_block(f, 1, kernel_size, weight_norm, batch_norm, activation, layers, True)
                    )
                else:
                    dec.append(
                        _decoder_block(
                            f, filters[i - 1], kernel_size, weight_norm, batch_norm, activation, layers, False
                        )
                    )
            decoders.append(nn.Sequential(*reversed(dec)))
        self.decoders = nn.ModuleList(decoders)

    def encode(self, x):
        tensors, indices, sizes = [], [], []
        for block in self.encoder:
            x = block(x)
            sizes.append(x.size())
            tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)
        return x, tensors, indices, sizes

    def decode_one(self, decoder, x, tensors, indices, sizes):
        tensors, indices, sizes = tensors[:], indices[:], sizes[:]
        for block in decoder:
            tensor = tensors.pop()
            size = sizes.pop()
            ind = indices.pop()
            x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
            x = torch.cat([tensor, x], dim=1)
            x = block(x)
        return x

    def forward(self, x):
        bottleneck, tensors, indices, sizes = self.encode(x)
        outs = [self.decode_one(d, bottleneck, tensors, indices, sizes) for d in self.decoders]
        return torch.cat(outs, dim=1)
