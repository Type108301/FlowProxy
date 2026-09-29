"""2D Fourier Neural Operator。

实现参考 Li et al., ICLR 2021 (arXiv:2010.08895) 与 neuraloperator 库的核心思想：
提升层 → 若干谱卷积块 → 投影层。本仓库为独立精简实现，不依赖 neuraloperator。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        # 实部/虚部分开存。GradScaler 不能处理 complex 参数，AMP 下 complex32 的 einsum 也没有 CUDA 实现。
        shape = (in_channels, out_channels, modes1, modes2, 2)
        self.weight1 = nn.Parameter(scale * torch.rand(*shape))
        self.weight2 = nn.Parameter(scale * torch.rand(*shape))

    def _mul(self, input_ft, weights):
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def _complex_weight(self, weight: torch.Tensor, m1: int, m2: int) -> torch.Tensor:
        w = weight[:, :, :m1, :m2].float().contiguous()
        return torch.view_as_complex(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 谱卷积固定在 float32 / complex64 上算，避免 AMP 把 rfft2 降成 ComplexHalf。
        in_dtype = x.dtype
        device_type = x.device.type if x.device.type in {"cpu", "cuda"} else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            b, _, h, w = x32.shape
            x_ft = torch.fft.rfft2(x32)
            out_ft = torch.zeros(
                b, self.out_channels, h, w // 2 + 1, device=x32.device, dtype=torch.cfloat
            )
            m1, m2 = min(self.modes1, h), min(self.modes2, w // 2 + 1)
            out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self._complex_weight(self.weight1, m1, m2))
            out_ft[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self._complex_weight(self.weight2, m1, m2))
            out = torch.fft.irfft2(out_ft, s=(h, w))
        if out.dtype != in_dtype:
            out = out.to(dtype=in_dtype)
        return out


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.w(x))


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, int] = (16, 16),
        hidden_channels: int = 32,
        n_layers: int = 4,
        lifting_channels: int = 64,
        projection_channels: int = 64,
        grid_embedding: bool = True,
    ):
        super().__init__()
        self.grid_embedding = grid_embedding
        lift_in = in_channels + (2 if grid_embedding else 0)
        self.lift = nn.Sequential(
            nn.Conv2d(lift_in, lifting_channels, 1),
            nn.GELU(),
            nn.Conv2d(lifting_channels, hidden_channels, 1),
        )
        self.blocks = nn.ModuleList(
            [FNOBlock(hidden_channels, n_modes[0], n_modes[1]) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, projection_channels, 1),
            nn.GELU(),
            nn.Conv2d(projection_channels, out_channels, 1),
        )

    def _grid(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype)
        xx = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grid_embedding:
            x = torch.cat([x, self._grid(x)], dim=1)
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)
