"""解析几何上的卡门涡街合成场，仅用于本机打通训练/可视化闭环。

正式训练标签必须来自 OpenFOAM，见 cfd/ 与 autodl/。
"""

from __future__ import annotations

import numpy as np

from .geometry import geometry_channels, make_grid


def _vortex(xx, yy, x0, y0, gamma, eps):
    dx = xx - x0
    dy = yy - y0
    r2 = dx * dx + dy * dy + eps * eps
    decay = 1.0 - np.exp(-(dx * dx + dy * dy) / (eps * eps + 1e-8))
    u = -gamma * dy / r2 * decay
    v = gamma * dx / r2 * decay
    return u, v


def karman_fields(cfg: dict, Re: float, t: float, cx=0.0, cy=0.0, D=1.0, shape="cylinder") -> dict[str, np.ndarray]:
    phys = cfg["physics"]
    U = float(phys["U_inf"])
    R = 0.5 * D
    geo = geometry_channels(cfg, shape=shape, cx=cx, cy=cy, D=D, Re=Re)
    xx, yy = geo["xx"], geo["yy"]
    mask = geo["mask"]

    dx = xx - cx
    dy = yy - cy
    r2 = np.maximum(dx * dx + dy * dy, (0.25 * R) ** 2)
    u_pot = U * (1.0 + R**2 * (dy * dy - dx * dx) / (r2 * r2))
    v_pot = U * (-2.0 * R**2 * dx * dy / (r2 * r2))

    st = 0.16 + 0.02 * (Re - 100.0) / 100.0
    period = D / max(st * U, 1e-6)
    spacing = 4.5 * D
    strength = 0.8 * U * D * (0.6 + 0.4 * (Re / 200.0))
    eps = 0.35 * D
    u, v = u_pot.copy(), v_pot.copy()
    n_vort = 8
    x_front = cx + 1.2 * D
    for n in range(n_vort):
        x0 = x_front + n * spacing - (t * 0.85 * U) % spacing
        sign = 1.0 if n % 2 == 0 else -1.0
        y0 = cy + sign * 0.6 * D
        phase = 2.0 * np.pi * t / period
        y0 = y0 + 0.08 * D * np.sin(phase + n)
        du, dv = _vortex(xx, yy, x0, y0, sign * strength, eps)
        u += du
        v += dv

    u = u * mask
    v = v * mask
    p = 0.5 * (U * U - u * u - v * v) * mask
    fields = np.stack([u, v, p], axis=0).astype(np.float32)
    return {**geo, "fields": fields, "t": np.float32(t), "Re": np.float32(Re)}


def make_synthetic_case(
    cfg: dict,
    case_id: str,
    Re: float,
    times: np.ndarray,
    cx=0.0,
    cy=0.0,
    D=1.0,
    shape="cylinder",
) -> dict:
    frames = [karman_fields(cfg, Re, float(t), cx=cx, cy=cy, D=D, shape=shape) for t in times]
    geom = frames[0]["geom"]
    fields = np.stack([f["fields"] for f in frames], axis=0)
    return {
        "case_id": case_id,
        "shape": shape,
        "Re": float(Re),
        "cx": float(cx),
        "cy": float(cy),
        "D": float(D),
        "times": np.asarray(times, dtype=np.float32),
        "geom": geom.astype(np.float32),
        "fields": fields.astype(np.float32),
        "mask": frames[0]["mask"].astype(np.float32),
    }
