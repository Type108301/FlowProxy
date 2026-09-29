from __future__ import annotations

import torch
import torch.nn.functional as F

from .metrics import divergence, relative_l2


def channel_weights(y: torch.Tensor) -> torch.Tensor:
    """DeepCFD 风格：按通道 RMS 平衡 Ux/Uy/p。"""
    b, c, h, w = y.shape
    rms = torch.sqrt(y.permute(0, 2, 3, 1).reshape(b * h * w, c).pow(2).mean(dim=0) + 1e-8)
    return rms.view(1, c, 1, 1)


def supervised_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - target) ** 2
    if weights is not None:
        err = err / weights
    return err.mean()


def physics_bundle(
    pred: torch.Tensor,
    mask: torch.Tensor,
    dx: float,
    dy: float,
    lambda_div: float,
    lambda_wall: float,
) -> dict[str, torch.Tensor]:
    """连续性残差 + 物体内速度惩罚。mask=1 表示流体。"""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    fluid = mask
    solid = 1.0 - mask
    u, v = pred[:, 0:1], pred[:, 1:2]
    div = divergence(u, v, dx, dy)
    fluid_c = fluid[:, :, 1:-1, 1:-1]
    loss_div = (div.pow(2) * fluid_c).mean()
    loss_wall = (pred[:, 0:2].pow(2) * solid).mean()
    total = lambda_div * loss_div + lambda_wall * loss_wall
    return {"phys": total, "div": loss_div, "wall": loss_wall}


def report_errors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> dict[str, float]:
    rel = relative_l2(pred, target, mask)
    names = ["rel_l2_ux", "rel_l2_uy", "rel_l2_p"]
    out = {k: float(rel[i]) for i, k in enumerate(names[: pred.shape[1]])}
    out["mse"] = float(F.mse_loss(pred, target))
    return out
