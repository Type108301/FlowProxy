from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import save_cases_h5
from .geometry import geometry_channels
from .interpolate import find_vtk_files, interpolate_points_to_grid, read_vtk_internal


def parse_case_meta(case_dir: Path) -> dict:
    meta_path = case_dir / "meta.json"
    if meta_path.exists():
        import json

        return json.loads(meta_path.read_text(encoding="utf-8"))
    name = case_dir.name
    # cylinder_Re100_cy0.0_D1.0
    parts = name.split("_")
    meta = {"shape": "cylinder", "Re": 100.0, "cx": 0.0, "cy": 0.0, "D": 1.0, "case_id": name}
    for p in parts:
        if p.startswith("Re"):
            meta["Re"] = float(p[2:])
        elif p.startswith("cy"):
            meta["cy"] = float(p[2:])
        elif p.startswith("D") and p[1:2].isdigit():
            meta["D"] = float(p[1:])
        elif p in {"cylinder", "square", "triangle"}:
            meta["shape"] = p
    return meta


def case_to_arrays(cfg: dict, case_dir: str | Path) -> dict:
    case_dir = Path(case_dir)
    meta = parse_case_meta(case_dir)
    files = find_vtk_files(case_dir)
    if not files:
        raise FileNotFoundError(f"no VTK in {case_dir}, run foamToVTK first")
    geo = geometry_channels(
        cfg,
        shape=meta["shape"],
        cx=meta.get("cx", 0.0),
        cy=meta["cy"],
        D=meta["D"],
        Re=meta["Re"],
    )
    frames = []
    times = []
    for vtk in files:
        pts, values = read_vtk_internal(vtk)
        field = interpolate_points_to_grid(pts, values, cfg, fill=0.0)
        field = field * geo["mask"][None, ...] if field.ndim == 3 else field
        # field: [3, ny, nx] but interpolate returns [C, ny, nx]
        frames.append(field)
        try:
            times.append(float("".join(ch if (ch.isdigit() or ch == ".") else " " for ch in vtk.stem).split()[-1]))
        except Exception:
            times.append(float(len(times)))
    order = np.argsort(times)
    fields = np.stack([frames[i] for i in order], axis=0).astype(np.float32)
    times_arr = np.asarray([times[i] for i in order], dtype=np.float32)
    skip = float(cfg["cfd"].get("skip_time", 0.0))
    keep = times_arr >= skip
    if keep.any():
        fields = fields[keep]
        times_arr = times_arr[keep]
    forces = _read_force_coeffs(case_dir)
    return {
        "case_id": meta.get("case_id", case_dir.name),
        "shape": meta["shape"],
        "Re": float(meta["Re"]),
        "cx": float(meta.get("cx", 0.0)),
        "cy": float(meta["cy"]),
        "D": float(meta["D"]),
        "times": times_arr,
        "geom": geo["geom"],
        "fields": fields,
        "mask": geo["mask"],
        **({"forces": forces} if forces is not None else {}),
    }


def _read_force_coeffs(case_dir: Path) -> np.ndarray | None:
    hits = list(case_dir.glob("postProcessing/forceCoeffs*/**/coefficient.dat"))
    hits += list(case_dir.glob("postProcessing/forceCoeffs*/**/forceCoeffs.dat"))
    if not hits:
        return None
    raw = []
    for line in hits[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = [float(x) for x in line.split()]
        if len(cols) >= 4:
            raw.append([cols[0], cols[1], cols[3]])  # time, Cd, Cl
    if not raw:
        return None
    return np.asarray(raw, dtype=np.float32)


def build_from_runs(cfg: dict, runs_dir: str | Path, out_h5: str | Path, split: dict[str, list[str]] | None = None) -> None:
    runs_dir = Path(runs_dir)
    cases = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "VTK").exists() and not list(child.glob("VTK/**/*")):
            continue
        try:
            cases.append(case_to_arrays(cfg, child))
            print(f"ingested {child.name}", flush=True)
        except Exception as exc:
            print(f"skip {child.name}: {exc}", flush=True)
    if not cases:
        raise RuntimeError(f"no valid VTK cases in {runs_dir}")
    save_cases_h5(out_h5, cases, split)
    print(f"wrote {len(cases)} cases -> {out_h5}")
