from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import make_grid


def interpolate_points_to_grid(
    points_xy: np.ndarray,
    values: np.ndarray,
    cfg: dict,
    fill: float = 0.0,
) -> np.ndarray:
    """将非结构点云插值到规则网格。values: [N, C] -> [C, ny, nx] 注意 imshow 用 (H=ny,W=nx)。"""
    from scipy.interpolate import LinearNDInterpolator

    _, _, xx, yy = make_grid(cfg)
    interp = LinearNDInterpolator(points_xy, values, fill_value=fill)
    packed = np.stack([xx.ravel(), yy.ravel()], axis=1)
    out = interp(packed)
    out = np.nan_to_num(out, nan=fill)
    c = values.shape[1]
    ny, nx = xx.shape
    return out.reshape(ny, nx, c).transpose(2, 0, 1).astype(np.float32)


def read_vtk_internal(vtk_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 foamToVTK 输出的点坐标与 U,p。优先 pyvista，其次 meshio。"""
    vtk_path = Path(vtk_path)
    try:
        import pyvista as pv

        mesh = pv.read(vtk_path)
        pts = np.asarray(mesh.points)[:, :2]
        if "U" in mesh.point_data:
            U = np.asarray(mesh.point_data["U"])[:, :2]
        elif "U" in mesh.cell_data:
            mesh = mesh.cell_data_to_point_data()
            U = np.asarray(mesh.point_data["U"])[:, :2]
        else:
            raise KeyError("U not in VTK")
        p = np.asarray(mesh.point_data["p"]).reshape(-1, 1)
        values = np.concatenate([U, p], axis=1)
        return pts, values
    except Exception:
        import meshio

        mesh = meshio.read(vtk_path)
        pts = mesh.points[:, :2]
        if "U" in mesh.point_data:
            U = np.asarray(mesh.point_data["U"])[:, :2]
        else:
            U = np.asarray(mesh.cell_data["U"][0])[:, :2]
            pts = _cell_centers(mesh)
        p_key = "p" if "p" in mesh.point_data else None
        if p_key:
            p = np.asarray(mesh.point_data["p"]).reshape(-1, 1)
        else:
            p = np.asarray(mesh.cell_data["p"][0]).reshape(-1, 1)
        return pts, np.concatenate([U, p], axis=1)


def _cell_centers(mesh) -> np.ndarray:
    pts = mesh.points
    block = mesh.cells_dict
    if "hexahedron" in block:
        conn = block["hexahedron"]
    elif "quad" in block:
        conn = block["quad"]
    else:
        conn = next(iter(block.values()))
    return pts[conn].mean(axis=1)[:, :2]


def find_vtk_files(case_dir: str | Path) -> list[Path]:
    case_dir = Path(case_dir)
    vtu = sorted(case_dir.glob("VTK/**/*.vtu"))
    vtk = sorted(case_dir.glob("VTK/**/*.vtk"))
    files = vtu or vtk
    return [p for p in files if "internal" in p.name.lower() or p.suffix in {".vtu", ".vtk"}]
