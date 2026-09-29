from __future__ import annotations

import numpy as np


def make_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = cfg["domain"]
    g = cfg["grid"]
    x = np.linspace(d["x_min"], d["x_max"], g["nx"], dtype=np.float32)
    y = np.linspace(d["y_min"], d["y_max"], g["ny"], dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return x, y, xx, yy


def triangle_vertices(cx: float, cy: float, D: float) -> np.ndarray:
    """等边三角形，边长 D，重心在 (cx, cy)，底边迎风、顶点指向下游。"""
    side = float(D)
    height = np.sqrt(3.0) * 0.5 * side
    x_base = float(cx) - height / 3.0
    x_apex = float(cx) + 2.0 * height / 3.0
    half = 0.5 * side
    return np.asarray(
        [
            [x_base, float(cy) - half],
            [x_base, float(cy) + half],
            [x_apex, float(cy)],
        ],
        dtype=np.float64,
    )


def _triangle_sdf(xx: np.ndarray, yy: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """三角形符号距离：外部为正，内部为负。"""
    p = np.stack([xx, yy], axis=-1).astype(np.float64)
    v0, v1, v2 = verts
    e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
    sign = np.sign(e0[0] * e2[1] - e0[1] * e2[0])

    def edge(pa: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = p - pa
        ee = float(np.dot(e, e))
        t = np.clip((v[..., 0] * e[0] + v[..., 1] * e[1]) / ee, 0.0, 1.0)
        pq = v - t[..., None] * e
        dist2 = pq[..., 0] ** 2 + pq[..., 1] ** 2
        cross = v[..., 0] * e[1] - v[..., 1] * e[0]
        return dist2, cross

    pairs = (edge(v0, e0), edge(v1, e1), edge(v2, e2))
    dist2 = np.minimum(np.minimum(pairs[0][0], pairs[1][0]), pairs[2][0])
    inside = (sign * pairs[0][1] >= 0.0) & (sign * pairs[1][1] >= 0.0) & (sign * pairs[2][1] >= 0.0)
    sdf = np.sqrt(dist2)
    sdf = np.where(inside, -sdf, sdf)
    return sdf.astype(np.float32)


def obstacle_sdf(xx: np.ndarray, yy: np.ndarray, shape: str, cx: float, cy: float, D: float) -> np.ndarray:
    """障碍物符号距离：流体区为正，内部为负。与 DeepCFD 的 SDF 通道同义。"""
    dx = xx - cx
    dy = yy - cy
    if shape == "cylinder":
        return np.sqrt(dx * dx + dy * dy) - 0.5 * D
    if shape == "square":
        return np.maximum(np.abs(dx), np.abs(dy)) - 0.5 * D
    if shape == "triangle":
        return _triangle_sdf(xx, yy, triangle_vertices(cx, cy, D))
    raise ValueError(f"unknown shape: {shape}")


def wall_sdf(yy: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """上下壁面距离（通道侧壁）。"""
    return np.minimum(yy - y_min, y_max - yy).astype(np.float32)


def geometry_channels(
    cfg: dict,
    shape: str | None = None,
    cx: float | None = None,
    cy: float | None = None,
    D: float | None = None,
    Re: float = 100.0,
) -> dict[str, np.ndarray]:
    phys = cfg["physics"]
    domain = cfg["domain"]
    shape = shape or phys["shape"]
    cx = phys["cx"] if cx is None else cx
    cy = phys["cy"] if cy is None else cy
    D = phys["D"] if D is None else D
    _, _, xx, yy = make_grid(cfg)
    sdf_obs = obstacle_sdf(xx, yy, shape, cx, cy, D).astype(np.float32)
    mask = (sdf_obs > 0).astype(np.float32)
    sdf_wall = wall_sdf(yy, domain["y_min"], domain["y_max"])
    re_ch = np.full_like(sdf_obs, np.float32(Re / 200.0))
    geom = np.stack([sdf_obs, mask, sdf_wall, re_ch], axis=0)
    return {
        "xx": xx.astype(np.float32),
        "yy": yy.astype(np.float32),
        "sdf_obs": sdf_obs,
        "mask": mask,
        "sdf_wall": sdf_wall,
        "geom": geom,
    }
