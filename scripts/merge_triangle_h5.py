#!/usr/bin/env python3
"""把已有圆柱场和三角形 npz 写成续训用的 dataset_triangle.h5，并写入划分。

不改 data/processed/dataset.h5。圆柱场从那份文件拷出，不重跑圆柱 CFD。
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.dataset import save_cases_h5  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402


def _meta_from_id(cid: str) -> dict:
    meta = {"Re": 0.0, "cy": 0.0, "D": 0.0}
    for part in cid.split("_"):
        if part.startswith("Re") and part[2:3].isdigit():
            meta["Re"] = float(part[2:])
        elif part.startswith("cy") and len(part) > 2 and part[2] in "-+0123456789.":
            meta["cy"] = float(part[2:])
        elif part.startswith("D") and len(part) > 1 and part[1].isdigit():
            meta["D"] = float(part[1:])
    return meta


def _is_holdout(cid: str, kind: str) -> bool:
    meta = _meta_from_id(cid)
    Re, cy, D = meta["Re"], meta["cy"], meta["D"]
    test = (
        (Re == 200 and cy == 0 and D == 1)
        or (Re == 100 and cy == 0.4 and D == 1)
        or (Re == 150 and cy == -0.4 and D == 0.9)
        or (Re == 80 and cy == 0.4 and D == 1.1)
    )
    val = (
        (Re == 150 and cy == 0 and D == 1)
        or (Re == 80 and cy == -0.4 and D == 1.1)
        or (Re == 100 and cy == 0 and D == 0.9)
        or (Re == 200 and cy == 0.4 and D == 0.9)
    )
    if kind == "test":
        return test
    if kind == "val":
        return val
    return False


def load_cylinders(path: Path) -> list[dict]:
    cases = []
    with h5py.File(path, "r") as handle:
        for cid in handle["cases"]:
            group = handle["cases"][cid]
            item = {
                "case_id": cid,
                "shape": str(group.attrs["shape"]),
                "Re": float(group.attrs["Re"]),
                "cx": float(group.attrs["cx"]),
                "cy": float(group.attrs["cy"]),
                "D": float(group.attrs["D"]),
                "times": np.asarray(group["times"][()], dtype=np.float32),
                "geom": np.asarray(group["geom"][()], dtype=np.float32),
                "fields": np.asarray(group["fields"][()], dtype=np.float32),
                "mask": np.asarray(group["mask"][()], dtype=np.float32),
            }
            if "forces" in group:
                item["forces"] = np.asarray(group["forces"][()], dtype=np.float32)
            cases.append(item)
    return cases


def load_triangles(npz_dir: Path) -> list[dict]:
    cases = []
    for path in sorted(npz_dir.glob("*.npz")):
        blob = np.load(path, allow_pickle=False)
        item = {
            "case_id": str(blob["case_id"]),
            "shape": str(blob["shape"]),
            "Re": float(blob["Re"]),
            "cx": float(blob["cx"]),
            "cy": float(blob["cy"]),
            "D": float(blob["D"]),
            "times": np.asarray(blob["times"], dtype=np.float32),
            "geom": np.asarray(blob["geom"], dtype=np.float32),
            "fields": np.asarray(blob["fields"], dtype=np.float32),
            "mask": np.asarray(blob["mask"], dtype=np.float32),
        }
        forces = np.asarray(blob["forces"], dtype=np.float32)
        if forces.size:
            item["forces"] = forces
        if item["shape"] != "triangle":
            raise RuntimeError(f"{path.name} shape is {item['shape']}")
        if item["fields"].shape[1:] != (3, 64, 64) or item["geom"].shape != (4, 64, 64):
            raise RuntimeError(f"{path.name} has unexpected shapes {item['fields'].shape} {item['geom'].shape}")
        cases.append(item)
    return cases


def build_split(cases: list[dict]) -> dict[str, list[str]]:
    split = {"train": [], "val": [], "test": []}
    for case in cases:
        cid = case["case_id"]
        if _is_holdout(cid, "test"):
            split["test"].append(cid)
        elif _is_holdout(cid, "val"):
            split["val"].append(cid)
        else:
            split["train"].append(cid)
    return split


def main():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    src = resolve_under_root(cfg, "data_processed") / "dataset.h5"
    npz_dir = resolve_under_root(cfg, "data_processed") / "triangle_cases"
    dest = resolve_under_root(cfg, "data_processed") / "dataset_triangle.h5"
    cylinders = load_cylinders(src)
    triangles = load_triangles(npz_dir)
    if len(cylinders) != 36:
        raise SystemExit(f"expected 36 cylinders, found {len(cylinders)}")
    if len(triangles) != 36:
        raise SystemExit(f"expected 36 triangles, found {len(triangles)} in {npz_dir}")
    cases = cylinders + triangles
    split = build_split(cases)
    for name, ids in split.items():
        shapes = {}
        for cid in ids:
            shape = cid.split("_", 1)[0]
            shapes[shape] = shapes.get(shape, 0) + 1
        print(f"{name}: {len(ids)} {shapes}")
    save_cases_h5(dest, cases, split)
    print(f"wrote {len(cases)} cases -> {dest}")


if __name__ == "__main__":
    main()
