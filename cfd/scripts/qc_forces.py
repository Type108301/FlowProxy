#!/usr/bin/env python3
"""读取 forceCoeffs，粗检 Cd 均值与 Cl 周期，不合格算例打印警告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_coeffs(case: Path) -> np.ndarray | None:
    hits = list(case.glob("postProcessing/**/coefficient.dat"))
    hits += list(case.glob("postProcessing/**/forceCoeffs.dat"))
    if not hits:
        return None
    rows = []
    for line in hits[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = [float(x) for x in line.split()]
        if len(cols) >= 4:
            rows.append(cols)
    return np.asarray(rows) if rows else None


def qc_one(case: Path, skip_time: float = 40.0) -> dict:
    meta = {}
    if (case / "meta.json").exists():
        meta = json.loads((case / "meta.json").read_text(encoding="utf-8"))
    arr = load_coeffs(case)
    report = {"case": case.name, "ok": False, "reason": "", **meta}
    if arr is None:
        report["reason"] = "no forceCoeffs"
        return report
    t = arr[:, 0]
    # OpenFOAM forceCoeffs: Time Cd Cs Cl ...
    cd = arr[:, 1]
    cl = arr[:, 3] if arr.shape[1] > 3 else arr[:, 2]
    keep = t >= skip_time
    if keep.sum() < 20:
        report["reason"] = "too few samples after skip_time"
        return report
    cd_m = float(np.mean(cd[keep]))
    cl_rms = float(np.sqrt(np.mean(cl[keep] ** 2)))
    report["Cd_mean"] = cd_m
    report["Cl_rms"] = cl_rms
    # Re=100 圆柱文献 Cd ~ 1.3–1.4；这里只做粗门禁
    if cd_m < 0.4 or cd_m > 3.5:
        report["reason"] = f"Cd_mean={cd_m:.3f} out of [0.4, 3.5]"
        return report
    if cl_rms < 1e-4:
        report["reason"] = "Cl almost zero, vortex street may not have started"
        return report
    report["ok"] = True
    report["reason"] = "pass"
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", type=Path)
    p.add_argument("--skip-time", type=float, default=40.0)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()
    reports = []
    for child in sorted(args.runs.iterdir()):
        if (child / "meta.json").exists():
            reports.append(qc_one(child, args.skip_time))
    n_ok = sum(1 for r in reports if r["ok"])
    print(f"{n_ok}/{len(reports)} passed")
    for r in reports:
        flag = "OK " if r["ok"] else "BAD"
        extra = f"Cd={r.get('Cd_mean', float('nan')):.3f}" if "Cd_mean" in r else r["reason"]
        print(f"  {flag} {r['case']:40s} {extra}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
