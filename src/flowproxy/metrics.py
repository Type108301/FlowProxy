from __future__ import annotations

import torch


def relative_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """相对 L2。pred/target: [B, C, H, W]。可选流体掩膜。"""
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        pred = pred * mask
        target = target * mask
    num = torch.linalg.vector_norm(pred - target, dim=(2, 3))
    den = torch.linalg.vector_norm(target, dim=(2, 3)).clamp_min(eps)
    return (num / den).mean(dim=0)


def mse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=(0, 2, 3))


def divergence(u: torch.Tensor, v: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    dudx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / (2 * dx)
    dvdy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / (2 * dy)
    dudx = dudx[:, :, 1:-1, :]
    dvdy = dvdy[:, :, :, 1:-1]
    return dudx + dvdy
