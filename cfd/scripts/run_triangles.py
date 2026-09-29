#!/usr/bin/env python3
"""生成等边三角形 icoFoam 标签，并写成随后可并入续训 h5 的 npz。

不覆盖 data/raw/cases 里的圆柱。每个算例成功入库后删掉 VTK 和时间目录，避免把数据盘写满。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cfd" / "scripts"))

from flowproxy.build_dataset import case_to_arrays  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from generate_case import write_case  # noqa: E402
from run_batch import run_one  # noqa: E402


def triangle_jobs() -> list[dict]:
    """与 36 个圆柱同一组 Re、cy、D，使两类样本数量接近。"""
    jobs = []
    for Re in (80.0, 100.0, 150.0, 200.0):
        for cy in (-0.4, 0.0, 0.4):
            for diameter in (0.9, 1.0, 1.1):
                jobs.append(
                    {
                        "Re": Re,
                        "cx": 0.0,
                        "cy": cy,
                        "D": diameter,
                        "shape": "triangle",
                    }
                )
    return jobs


def slim_case(case: Path) -> None:
    vtk = case / "VTK"
    if vtk.exists():
        shutil.rmtree(vtk)
    for child in list(case.iterdir()):
        if child.is_dir() and child.name[:1].isdigit():
            shutil.rmtree(child)


def store_case(cfg: dict, case: Path, out_dir: Path) -> Path:
    payload = case_to_arrays(cfg, case)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{payload['case_id']}.npz"
    forces = payload.get("forces")
    if forces is None:
        forces = np.zeros((0, 3), dtype=np.float32)
    np.savez_compressed(
        dest,
        fields=payload["fields"],
        times=payload["times"],
        geom=payload["geom"],
        mask=payload["mask"],
        forces=np.asarray(forces, dtype=np.float32),
        Re=np.float64(payload["Re"]),
        cx=np.float64(payload["cx"]),
        cy=np.float64(payload["cy"]),
        D=np.float64(payload["D"]),
        shape=np.array(payload["shape"]),
        case_id=np.array(payload["case_id"]),
    )
    return dest


def run_job(cfg: dict, out: Path, npz_dir: Path, job: dict, timeout: int | None, keep_heavy: bool) -> str:
    case_id = f"{job['shape']}_Re{job['Re']:g}_cy{job['cy']:g}_D{job['D']:g}"
    existing = npz_dir / f"{case_id}.npz"
    if existing.is_file():
        frames = int(np.load(existing)["fields"].shape[0])
        return f"{case_id} already stored ({frames} frames)"
    case = write_case(cfg, out, **job)
    rc = run_one(case, timeout)
    if rc != 0:
        raise RuntimeError(f"{case.name} Allrun rc={rc}")
    dest = store_case(cfg, case, npz_dir)
    frames = int(np.load(dest)["fields"].shape[0])
    if not keep_heavy:
        slim_case(case)
    return f"{case.name} -> {dest.name} frames={frames}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--out", default=None)
    p.add_argument("--npz-dir", default=None)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--keep-heavy", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else resolve_under_root(cfg, "data_raw") / "cases"
    npz_dir = Path(args.npz_dir) if args.npz_dir else resolve_under_root(cfg, "data_processed") / "triangle_cases"
    jobs = triangle_jobs()
    if args.max_cases:
        jobs = jobs[: args.max_cases]
    print(f"{len(jobs)} triangle cases -> {out}", flush=True)
    failed = []
    workers = max(1, args.parallel)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_job, cfg, out, npz_dir, job, args.timeout or None, args.keep_heavy): job
            for job in jobs
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            job = futures[fut]
            name = f"triangle_Re{job['Re']:g}_cy{job['cy']:g}_D{job['D']:g}"
            try:
                print(f"[{done}/{len(jobs)}] {fut.result()}", flush=True)
            except Exception as exc:
                failed.append(name)
                print(f"[{done}/{len(jobs)}] FAILED {name}: {exc}", flush=True)
    if failed:
        print("failed:", ", ".join(failed), flush=True)
        sys.exit(1)
    print(f"all {len(jobs)} triangles stored in {npz_dir}", flush=True)


if __name__ == "__main__":
    main()
