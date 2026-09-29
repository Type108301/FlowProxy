#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.build_dataset import build_from_runs  # noqa: E402
from flowproxy.dataset import save_cases_h5  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from flowproxy.splits import default_split  # noqa: E402
from flowproxy.synthetic import make_synthetic_case  # noqa: E402
import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--from-vtk", type=Path, default=None, help="OpenFOAM 算例根目录（含 VTK）")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = args.out or (resolve_under_root(cfg, "data_processed") / "dataset.h5")
    if args.synthetic:
        phys = cfg["physics"]
        times = np.linspace(float(cfg["cfd"]["skip_time"]), float(cfg["cfd"]["end_time"]), 32)
        cases = []
        for Re in phys["Re_list"]:
            for cy in phys["cy_offsets"]:
                for scale in phys["D_scales"]:
                    D = phys["D"] * scale
                    cid = f"{phys['shape']}_Re{Re:g}_cy{cy:g}_D{D:g}"
                    cases.append(
                        make_synthetic_case(cfg, cid, Re=float(Re), times=times, cx=phys["cx"], cy=float(cy), D=float(D), shape=phys["shape"])
                    )
        split = default_split([c["case_id"] for c in cases], cfg)
        save_cases_h5(out, cases, split)
        print(f"synthetic {len(cases)} -> {out}")
        return
    vtk_root = args.from_vtk or (resolve_under_root(cfg, "data_raw") / "cases")
    build_from_runs(cfg, vtk_root, out)


if __name__ == "__main__":
    main()
